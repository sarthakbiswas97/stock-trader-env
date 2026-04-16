"""GRPO v2 Training — Multi-dimensional rewards with dynamic scaling.

Four reward functions scored independently by TRL's GRPOTrainer:
  1. format_reward:     Valid <think>...</think>\\nACTION format
  2. reasoning_reward:  Signal grounding, specificity, reasoning-action consistency
  3. trading_reward:    Counterfactual P&L with tanh(z) scaling + opportunity cost
  4. prediction_reward: Directional prediction accuracy from reasoning

All trading rewards scale dynamically with volatility-normalized price changes
(z-score). No hardcoded reward constants — everything is a function of the
actual market move magnitude.

Temperature decays via cosine schedule (0.9 -> 0.6) using a GRPOTrainer subclass.

Usage:
    cd /workspace/stock-trader-env
    pip install -r requirements-training.txt
    pip install -e .

    python scripts/train_grpo.py --prompts-dataset data/grpo/grpo_v2_prompts_train_20000.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

# Monkey-patch: llm_blender (trl dependency) imports TRANSFORMERS_CACHE
# which was removed in transformers 5.x. Patch before any trl import.
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    from huggingface_hub.constants import HF_HUB_CACHE
    _hub.TRANSFORMERS_CACHE = HF_HUB_CACHE

import mlflow
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sft_checkpoint": "sarthakbiswas/stock-trader-sft-v2-model",
    "prompts_dataset": "",
    "max_seq_length": 1024,
    "max_completion_length": 200,
    "num_generations": 4,
    "max_steps": 500,
    "batch_size": 1,
    "grad_accum": 8,
    "lr": 5e-7,
    "beta": 0.01,
    "t_start": 0.9,
    "t_end": 0.6,
    "output_dir": "/workspace/grpo-checkpoint",
}

# Steepness for tanh scaling — controls how quickly rewards saturate
TANH_STEEPNESS = 1.5
VOL_FLOOR = 0.005


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def extract_completion_text(completion: str | list) -> str:
    """Extract plain text from completion (handles str or list-of-dicts)."""
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
            else:
                parts.append(str(msg))
        return "".join(parts)
    return str(completion)


def is_valid_format(text: str) -> bool:
    """Check if completion follows <think>...</think>\\nACTION format."""
    has_think = "<think>" in text and "</think>" in text
    lines = text.strip().split("\n")
    last_line = lines[-1].strip().upper() if lines else ""
    has_action = last_line.startswith(("BUY", "SELL", "HOLD"))
    return has_think and has_action


def parse_action_word(text: str) -> str:
    """Extract the action word (BUY/SELL/HOLD) from completion text."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip().upper()
        if line.startswith(("BUY", "SELL", "HOLD")):
            match = re.match(r"(BUY|SELL|HOLD)", line)
            if match:
                return match.group(1)
    return "HOLD"


def extract_think_block(text: str) -> str:
    """Extract content between <think> and </think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Reasoning analysis helpers
# ---------------------------------------------------------------------------

# Signal patterns that should appear in reasoning when grounded in observation
_SIGNAL_KEYWORDS = {
    "rsi": re.compile(r"\bRSI\b", re.IGNORECASE),
    "macd": re.compile(r"\bMACD\b", re.IGNORECASE),
    "volume": re.compile(r"\bvolume\b", re.IGNORECASE),
    "bollinger": re.compile(r"\bbollinger\b", re.IGNORECASE),
    "trend": re.compile(r"\btrend\b", re.IGNORECASE),
    "momentum": re.compile(r"\bmomentum\b", re.IGNORECASE),
    "vix": re.compile(r"\bVIX\b", re.IGNORECASE),
    "usd_inr": re.compile(r"\bUSD[/\s]*INR\b", re.IGNORECASE),
    "crude": re.compile(r"\b(?:crude|brent)\b", re.IGNORECASE),
    "candlestick": re.compile(r"\b(?:doji|hammer|engulfing|shooting.?star|candle)\b", re.IGNORECASE),
    "gap": re.compile(r"\bgap\b", re.IGNORECASE),
    "support": re.compile(r"\bsupport\b", re.IGNORECASE),
    "resistance": re.compile(r"\bresistance\b", re.IGNORECASE),
}

_BULLISH_WORDS = frozenset({
    "bullish", "oversold", "upward", "recovery", "buy signal",
    "entry", "support", "accumulate", "upside", "bounce",
    "positive", "strength", "breakout",
})

_BEARISH_WORDS = frozenset({
    "bearish", "overbought", "downward", "decline", "sell signal",
    "resistance", "distribute", "downside", "breakdown",
    "negative", "weakness", "drop",
})

_LAZY_PHRASES = [
    "no clear signal", "waiting for confirmation", "mixed signals",
    "nothing to do", "staying flat", "no setup", "uncertain",
    "based on the data", "looking at the indicators",
    "the technical analysis shows", "considering the current situation",
]


def extract_referenced_signals(think_text: str) -> set[str]:
    """Find which technical signals are referenced in the reasoning."""
    return {
        name for name, pattern in _SIGNAL_KEYWORDS.items()
        if pattern.search(think_text)
    }


def extract_sentiment_from_reasoning(think_text: str) -> str:
    """Infer directional sentiment from reasoning text."""
    text_lower = think_text.lower()
    bull_count = sum(1 for w in _BULLISH_WORDS if w in text_lower)
    bear_count = sum(1 for w in _BEARISH_WORDS if w in text_lower)

    if bull_count > bear_count + 1:
        return "bullish"
    if bear_count > bull_count + 1:
        return "bearish"
    return "neutral"


def detect_lazy_reasoning(think_text: str) -> bool:
    """Return True if reasoning is generic without specific data references."""
    signals = extract_referenced_signals(think_text)
    has_lazy = any(phrase in think_text.lower() for phrase in _LAZY_PHRASES)
    return has_lazy and len(signals) < 2


# ---------------------------------------------------------------------------
# Anti-HOLD collapse tracking
# ---------------------------------------------------------------------------

_action_window: list[str] = []
_ACTION_WINDOW_SIZE = 200
_action_counter: Counter = Counter()


def _reset_tracking() -> None:
    """Reset action tracking state before training."""
    _action_window.clear()
    _action_counter.clear()


def _track_action(action: str) -> None:
    """Track action in rolling window and global counter."""
    _action_window.append(action)
    if len(_action_window) > _ACTION_WINDOW_SIZE:
        _action_window.pop(0)
    _action_counter[action] += 1


def _hold_collapse_penalty(action: str) -> float:
    """Dynamic penalty if HOLD dominates the rolling window."""
    if len(_action_window) < 50:
        return 0.0

    hold_pct = _action_window.count("HOLD") / len(_action_window)

    if action == "HOLD":
        if hold_pct > 0.75:
            return -0.4
        if hold_pct > 0.65:
            return -0.2
        return 0.0
    else:
        # Bonus for BUY/SELL when HOLD is dominating
        if hold_pct > 0.75:
            return 0.2
        if hold_pct > 0.65:
            return 0.1
        return 0.0


# ---------------------------------------------------------------------------
# Reward Function 1: Format compliance
# ---------------------------------------------------------------------------


def format_reward(completions: list, **kwargs) -> list[float]:
    """Reward for valid <think>...</think>\\nACTION format.

    +1.0 valid format, -1.0 invalid.
    """
    rewards = []
    for completion in completions:
        text = extract_completion_text(completion)
        rewards.append(1.0 if is_valid_format(text) else -1.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward Function 2: Reasoning quality
# ---------------------------------------------------------------------------


def reasoning_reward(completions: list, **kwargs) -> list[float]:
    """Reward for reasoning quality — grounding, specificity, consistency.

    Components (summed, clamped to [-1, +1]):
      a) Signal grounding: +0.08 per signal referenced (up to 5), -0.3 if zero
      b) Specificity: +0.05 per number cited (up to 4)
      c) Lazy reasoning penalty: -0.3
      d) Reasoning-action consistency: bullish+BUY=+0.3, bullish+SELL=-0.5
    """
    rewards = []
    for completion in completions:
        text = extract_completion_text(completion)
        think_text = extract_think_block(text)
        action = parse_action_word(text)

        if not think_text:
            rewards.append(-0.5)
            continue

        score = 0.0

        # (a) Signal grounding
        signals = extract_referenced_signals(think_text)
        signal_score = min(len(signals), 5) * 0.08
        if len(signals) == 0:
            signal_score = -0.3
        score += signal_score

        # (b) Specificity — numbers in reasoning indicate data-driven thinking
        numbers = re.findall(r"\d+\.?\d*", think_text)
        score += min(len(numbers), 4) * 0.05

        # (c) Lazy reasoning penalty
        if detect_lazy_reasoning(think_text):
            score -= 0.3

        # (d) Reasoning-action consistency
        sentiment = extract_sentiment_from_reasoning(think_text)
        if sentiment == "bullish" and action == "BUY":
            score += 0.3
        elif sentiment == "bearish" and action == "SELL":
            score += 0.3
        elif sentiment == "bullish" and action == "SELL":
            score -= 0.5
        elif sentiment == "bearish" and action == "BUY":
            score -= 0.5

        rewards.append(max(-1.0, min(1.0, score)))

    return rewards


# ---------------------------------------------------------------------------
# Reward Function 3: Trading P&L + opportunity cost (dynamic, tanh-scaled)
# ---------------------------------------------------------------------------


def trading_reward(completions: list, **kwargs) -> list[float]:
    """Counterfactual P&L with volatility-normalized tanh scaling.

    z = price_change / rolling_vol
    base = tanh(z * STEEPNESS)     # bounded [-1, +1], scales with move magnitude

    Every scenario produces a non-zero reward proportional to the actual
    price movement. No hardcoded constants — HOLD is never "safe" when
    the market moves.
    """
    current_prices = kwargs.get("current_price", [])
    next_1d = kwargs.get("next_price_1d", [])
    next_2d = kwargs.get("next_price_2d", [])
    next_5d = kwargs.get("next_price_5d", [])
    has_positions = kwargs.get("has_position", [])
    rolling_vols = kwargs.get("rolling_vol", [])

    rewards = []
    for i, completion in enumerate(completions):
        text = extract_completion_text(completion)
        action = parse_action_word(text)

        # Track for anti-collapse monitoring
        _track_action(action)

        cur = current_prices[i] if i < len(current_prices) else 0.0
        nxt = next_1d[i] if i < len(next_1d) else cur
        vol = rolling_vols[i] if i < len(rolling_vols) and rolling_vols[i] > VOL_FLOOR else 0.02
        has_pos = has_positions[i] if i < len(has_positions) else False

        # Volatility-normalized price change
        price_change = (nxt - cur) / cur if cur > 0 else 0.0
        z = price_change / vol
        base = math.tanh(z * TANH_STEEPNESS)

        # Multi-day confirmation amplifier
        nxt_2d = next_2d[i] if i < len(next_2d) else cur
        nxt_5d = next_5d[i] if i < len(next_5d) else cur
        change_2d = (nxt_2d - cur) / cur if cur > 0 else 0.0
        change_5d = (nxt_5d - cur) / cur if cur > 0 else 0.0

        same_direction = (
            (price_change > 0 and change_2d > 0 and change_5d > 0)
            or (price_change < 0 and change_2d < 0 and change_5d < 0)
        )
        confirmation = 1.2 if same_direction else 0.9

        # Compute reward based on action + market state
        if action == "BUY" and not has_pos:
            reward = base * confirmation
        elif action == "SELL" and has_pos:
            reward = -base * confirmation  # profit from selling before drop
        elif action == "HOLD" and has_pos:
            reward = base * 0.8  # held through gain/loss, slightly less than active trade
        elif action == "HOLD" and not has_pos:
            # Dynamic opportunity cost / avoided loss
            if z > 0:
                reward = -base * 0.6  # missed opportunity (scales with move)
            else:
                reward = math.tanh(abs(z) * TANH_STEEPNESS) * 0.4  # avoided loss
        else:
            # Invalid: BUY with position or SELL without
            reward = -0.3

        # Anti-HOLD collapse penalty
        reward += _hold_collapse_penalty(action)

        rewards.append(float(reward))

    # Log action distribution periodically
    total = sum(_action_counter.values())
    if total > 0 and total % 50 < len(completions):
        dist = {k: f"{v / total * 100:.0f}%" for k, v in _action_counter.most_common()}
        hold_pct = _action_counter.get("HOLD", 0) / total * 100
        logger.info("Actions (%d): %s | HOLD%%=%.1f", total, dist, hold_pct)

    return rewards


# ---------------------------------------------------------------------------
# Reward Function 4: Prediction accuracy
# ---------------------------------------------------------------------------


def prediction_reward(completions: list, **kwargs) -> list[float]:
    """Reward for directional prediction accuracy from reasoning.

    If reasoning expresses bullish/bearish sentiment, compare against
    actual next-day direction. Reward scales with z-score magnitude.
    """
    current_prices = kwargs.get("current_price", [])
    next_1d = kwargs.get("next_price_1d", [])
    rolling_vols = kwargs.get("rolling_vol", [])

    rewards = []
    for i, completion in enumerate(completions):
        text = extract_completion_text(completion)
        think_text = extract_think_block(text)
        sentiment = extract_sentiment_from_reasoning(think_text)

        if sentiment == "neutral" or not think_text:
            rewards.append(0.0)
            continue

        cur = current_prices[i] if i < len(current_prices) else 0.0
        nxt = next_1d[i] if i < len(next_1d) else cur
        vol = rolling_vols[i] if i < len(rolling_vols) and rolling_vols[i] > VOL_FLOOR else 0.02

        price_change = (nxt - cur) / cur if cur > 0 else 0.0
        z = price_change / vol
        z_magnitude = min(abs(z), 3.0) / 3.0  # normalize to [0, 1]

        actual_direction = "up" if price_change > 0 else "down"

        if sentiment == "bullish" and actual_direction == "up":
            rewards.append(0.5 * (0.3 + 0.7 * z_magnitude))  # scale with magnitude
        elif sentiment == "bearish" and actual_direction == "down":
            rewards.append(0.5 * (0.3 + 0.7 * z_magnitude))
        elif sentiment == "bullish" and actual_direction == "down":
            rewards.append(-0.3 * (0.3 + 0.7 * z_magnitude))
        elif sentiment == "bearish" and actual_direction == "up":
            rewards.append(-0.3 * (0.3 + 0.7 * z_magnitude))
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Temperature-scheduled GRPOTrainer subclass
# ---------------------------------------------------------------------------


def _create_trainer_class():
    """Create ScheduledTemperatureGRPOTrainer (deferred import)."""
    from trl import GRPOTrainer

    class ScheduledTemperatureGRPOTrainer(GRPOTrainer):
        """GRPOTrainer with cosine temperature decay."""

        def __init__(self, *args, t_start: float = 0.9, t_end: float = 0.6, **kwargs):
            super().__init__(*args, **kwargs)
            self._t_start = t_start
            self._t_end = t_end

        def _generate_and_score_completions(self, inputs):
            progress = self.state.global_step / max(self.state.max_steps, 1)
            new_temp = self._t_end + 0.5 * (self._t_start - self._t_end) * (
                1 + math.cos(math.pi * progress)
            )
            self.temperature = new_temp
            self.generation_config.temperature = new_temp

            if self.state.global_step % 10 == 0:
                logger.info(
                    "Step %d: temperature=%.3f, progress=%.1f%%",
                    self.state.global_step, new_temp, progress * 100,
                )

            return super()._generate_and_score_completions(inputs)

    return ScheduledTemperatureGRPOTrainer


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run GRPO v2 training."""

    # ------------------------------------------------------------------
    # 1. Environment check
    # ------------------------------------------------------------------
    logger.info("=== GRPO v2 Training ===")
    logger.info("CUDA: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("")

    # ------------------------------------------------------------------
    # 2. Load SFT model with Unsloth
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logger.info("Loading SFT checkpoint: %s", args.sft_checkpoint)
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.sft_checkpoint,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    logger.info("Model loaded")

    # ------------------------------------------------------------------
    # 3. Load prompts dataset (flat columns, no metadata store)
    # ------------------------------------------------------------------
    if not args.prompts_dataset:
        logger.error("--prompts-dataset required. Run collect_grpo_prompts.py first.")
        return

    logger.info("Loading prompts from: %s", args.prompts_dataset)

    prompts_data = []
    prompts_path = Path(args.prompts_dataset)
    if prompts_path.exists():
        with open(prompts_path) as f:
            for line in f:
                prompts_data.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset(args.prompts_dataset, split="train")
        prompts_data = list(ds)

    logger.info("Loaded %d prompts", len(prompts_data))

    # Create HF Dataset — TRL passes all non-"prompt" columns as kwargs
    from datasets import Dataset
    train_dataset = Dataset.from_list(prompts_data)
    logger.info("Dataset columns: %s", train_dataset.column_names)
    logger.info("Dataset ready: %d prompts", len(train_dataset))

    # Log regime distribution
    if "regime" in train_dataset.column_names:
        regime_counts = Counter(train_dataset["regime"])
        logger.info("Regime distribution: %s", dict(regime_counts.most_common()))

    # ------------------------------------------------------------------
    # 4. MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("file:///workspace/mlruns")
    mlflow.set_experiment("grpo-v2-training")

    run_name = (
        f"grpo-v2-gen{args.num_generations}-lr{args.lr}"
        f"-steps{args.max_steps}-t{args.t_start}-{args.t_end}"
    )
    mlflow_run = mlflow.start_run(run_name=run_name)

    mlflow.log_params({
        "sft_checkpoint": args.sft_checkpoint,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "learning_rate": args.lr,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "num_prompts": len(train_dataset),
        "tanh_steepness": TANH_STEEPNESS,
        "reward_functions": "format,reasoning,trading,prediction",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    })

    logger.info("MLflow run: %s (ID: %s)", run_name, mlflow_run.info.run_id)

    # ------------------------------------------------------------------
    # 5. GRPO Training with temperature scheduling
    # ------------------------------------------------------------------
    from trl import GRPOConfig

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        seed=42,
        report_to="mlflow",
        temperature=args.t_start,
        loss_type="grpo",
        scale_rewards="batch",
    )

    # Reset action tracking
    _reset_tracking()

    ScheduledTemperatureGRPOTrainer = _create_trainer_class()

    trainer = ScheduledTemperatureGRPOTrainer(
        t_start=args.t_start,
        t_end=args.t_end,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[format_reward, reasoning_reward, trading_reward, prediction_reward],
        processing_class=tokenizer,
    )

    start_time = time.time()
    logger.info("Starting GRPO v2 training...")
    logger.info("  Prompts: %d", len(train_dataset))
    logger.info("  Max steps: %d", args.max_steps)
    logger.info("  Generations per prompt: %d", args.num_generations)
    logger.info("  Temperature: %.2f -> %.2f (cosine)", args.t_start, args.t_end)
    logger.info("  Beta (KL): %.4f", args.beta)
    logger.info("  Learning rate: %.1e", args.lr)
    logger.info("  Reward functions: format + reasoning + trading + prediction")
    logger.info("  Scale rewards: batch")
    logger.info("  Anti-HOLD collapse: rolling window (%d)", _ACTION_WINDOW_SIZE)
    logger.info("")

    trainer.train()
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("Training complete in %.1f minutes", elapsed / 60)
    logger.info("Final action distribution: %s", dict(_action_counter.most_common()))

    total_actions = sum(_action_counter.values())
    if total_actions > 0:
        hold_pct = _action_counter.get("HOLD", 0) / total_actions * 100
        buy_pct = _action_counter.get("BUY", 0) / total_actions * 100
        sell_pct = _action_counter.get("SELL", 0) / total_actions * 100
        mlflow.log_metrics({
            "final_hold_pct": round(hold_pct, 1),
            "final_buy_pct": round(buy_pct, 1),
            "final_sell_pct": round(sell_pct, 1),
        })

    mlflow.log_metrics({"training_time_minutes": round(elapsed / 60, 1)})

    # ------------------------------------------------------------------
    # 6. Save checkpoint
    # ------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Checkpoint saved to %s", args.output_dir)

    # ------------------------------------------------------------------
    # 7. Push to HF Hub (optional)
    # ------------------------------------------------------------------
    if args.push_to_hub and args.hf_repo:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping push to hub.")
        else:
            logger.info("Pushing model to HF Hub: %s", args.hf_repo)
            model.push_to_hub(args.hf_repo, token=hf_token)
            tokenizer.push_to_hub(args.hf_repo, token=hf_token)

            # Push MLflow data
            mlflow.end_run()
            mlruns_dir = "/workspace/mlruns"
            if os.path.exists(mlruns_dir):
                import tarfile
                mlruns_tar = os.path.join(args.output_dir, "mlruns.tar.gz")
                with tarfile.open(mlruns_tar, "w:gz") as tar:
                    tar.add(mlruns_dir, arcname="mlruns")

                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=mlruns_tar,
                    path_in_repo="mlruns.tar.gz",
                    repo_id=args.hf_repo,
                    repo_type="model",
                    token=hf_token,
                )
                logger.info("MLflow data pushed to HF Hub")

            logger.info("Done: https://huggingface.co/%s", args.hf_repo)
    else:
        mlflow.end_run()

    logger.info("")
    logger.info("=== Done ===")
    logger.info("Checkpoint: %s", args.output_dir)
    logger.info("Evaluate: PYTHONPATH=. python scripts/eval_sft.py --checkpoint %s", args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO v2 Training")
    parser.add_argument("--sft-checkpoint", default=DEFAULTS["sft_checkpoint"])
    parser.add_argument("--prompts-dataset", default=DEFAULTS["prompts_dataset"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--max-completion-length", type=int, default=DEFAULTS["max_completion_length"])
    parser.add_argument("--num-generations", type=int, default=DEFAULTS["num_generations"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    parser.add_argument("--t-start", type=float, default=DEFAULTS["t_start"])
    parser.add_argument("--t-end", type=float, default=DEFAULTS["t_end"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo", default="sarthakbiswas/stock-trader-grpo-v2-model")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
