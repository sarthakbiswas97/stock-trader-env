"""GRPO v3 Training — Hard-example selection + model's own trajectories.

v2.1 scored 0.395 (matched SFT v2) but with high variance (0.1-0.6 bimodal).
v3 applies findings from Pikus et al. "Hard Examples Are All You Need":

  - Train on model's OWN trajectories (not heuristic agent's)
  - Filter to hardest prompts via difficulty estimation (learnability scoring)
  - G=8 (up from 4) for better within-group variance
  - 1000 steps (up from 500) — paper showed hard examples sustain learning longer
  - beta=0.04 (up from 0.01) — v2.1 had KL diverge to 3.9
  - scale_rewards="batch" (Dr. GRPO fix for per-group std explosion)

Pipeline:
  1. collect_model_episodes.py — run model against env, collect trajectories
  2. estimate_difficulty.py — probe model, rank by learnability, keep top 35%
  3. train_grpo.py — GRPO on hard-filtered subset

Same two reward functions as v2.1:
  1. format_gate:    Penalty-only (-1.0 invalid, 0.0 valid). Not a reward source.
  2. trading_reward: 30% step-level (5D composite, asymmetric) + 70% episode-level return.

Usage:
    cd /workspace/stock-trader-env
    pip install -r requirements-training.txt
    pip install -e .

    # v3: Train on hard-filtered model trajectories
    python scripts/train_grpo.py \\
        --prompts-dataset data/grpo/hard_prompts_1400.jsonl \\
        --sft-checkpoint sarthakbiswas/stock-trader-grpo-v2.1-model
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path

# Suppress noisy logs before any imports trigger them
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Unsloth should be imported.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*and.*max_length.*")
os.environ["HTTPX_LOG_LEVEL"] = "WARNING"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

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
    "sft_checkpoint": "sarthakbiswas/stock-trader-grpo-v2.1-model",
    "prompts_dataset": "",
    "max_seq_length": 1024,
    "max_completion_length": 200,
    "num_generations": 8,       # v3: up from 4 — more within-group variance (Pikus et al.)
    "max_steps": 1000,          # v3: up from 500 — hard examples sustain learning longer
    "batch_size": 1,
    "grad_accum": 8,
    "lr": 5e-7,
    "beta": 0.04,               # v3: up from 0.01 — v2.1 had KL diverge to 3.9
    "t_start": 0.9,
    "t_end": 0.6,
    "output_dir": "/workspace/grpo-v3-checkpoint",
}

# Steepness for tanh scaling — controls how quickly rewards saturate
TANH_STEEPNESS = 1.5
VOL_FLOOR = 0.005

# v2.1: Reward blending weights (step vs episode)
STEP_WEIGHT = 0.30
EPISODE_WEIGHT = 0.70

# Asymmetric penalty multiplier for bad trades (BUY before drop, SELL before rise)
BAD_TRADE_PENALTY = 1.5

# 5D composite weights: how much each horizon contributes to step-level reward
HORIZON_WEIGHTS = {"1d": 0.2, "2d": 0.3, "5d": 0.5}


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
# Reward Function 1: Format gate (penalty only)
# ---------------------------------------------------------------------------


def format_gate(completions: list, **kwargs) -> list[float]:
    """Penalty-only gate for valid <think>...</think>\\nACTION format.

    0.0 for valid format (no reward — format is expected, not rewarded).
    -1.0 for invalid format (penalty — must learn the format).

    v2 gave +1.0 for valid format, which meant 84% of reward came from
    format/reasoning and 0% from actual trading. Now format is just a gate.
    """
    rewards = []
    for completion in completions:
        text = extract_completion_text(completion)
        rewards.append(0.0 if is_valid_format(text) else -1.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward Function 2: Trading reward (step-level + episode-level blend)
# ---------------------------------------------------------------------------


def _compute_step_reward(
    action: str,
    has_pos: bool,
    cur: float,
    fwd_prices: dict[str, float],
    vol: float,
) -> float:
    """5D composite step-level reward with asymmetric penalties.

    Uses weighted combination of 1d/2d/5d horizons instead of 1D only.
    BUY before drop or SELL before rise penalized at BAD_TRADE_PENALTY rate.
    """
    if cur <= 0:
        return 0.0

    # Compute z-scores at each horizon
    z_scores = {}
    for horizon, price in fwd_prices.items():
        change = (price - cur) / cur
        z_scores[horizon] = change / vol

    # Weighted composite z-score across horizons
    composite_z = sum(
        z_scores.get(h, 0.0) * w for h, w in HORIZON_WEIGHTS.items()
    )
    base = math.tanh(composite_z * TANH_STEEPNESS)

    if action == "BUY" and not has_pos:
        if composite_z >= 0:
            # Good BUY: bought before rise
            return base
        else:
            # Bad BUY: bought before drop — asymmetric penalty
            return base * BAD_TRADE_PENALTY

    elif action == "SELL" and has_pos:
        # SELL profit = negative of price movement (sold before drop = good)
        if composite_z <= 0:
            # Good SELL: sold before drop
            return -base  # -base is positive when composite_z < 0
        else:
            # Bad SELL: sold before rise — asymmetric penalty
            return -base * BAD_TRADE_PENALTY

    elif action == "HOLD" and has_pos:
        # Held through movement — slightly discounted vs active trade
        return base * 0.7

    elif action == "HOLD" and not has_pos:
        # Opportunity cost (missed rise) or avoided loss (dodged drop)
        if composite_z > 0:
            return -abs(base) * 0.5  # missed opportunity
        else:
            return abs(base) * 0.3  # avoided loss (smaller reward than active SELL)

    else:
        # Invalid: BUY with position or SELL without
        return -0.3


def _compute_episode_reward(episode_return: float) -> float:
    """Episode-level reward aligned with eval's grade_single_stock().

    Maps episode return to [-1, +1] using the same breakpoints as the grader:
      - return <= -5%: strongly negative
      - return ~0%: near zero
      - return > 0%: positive, scaling with magnitude
    """
    # tanh scaling centered at 0 with steepness tuned to trading returns
    # A 5% episode return maps to ~tanh(5 * 0.4) = tanh(2.0) = 0.96
    # A -2% episode return maps to ~tanh(-2 * 0.4) = tanh(-0.8) = -0.66
    return math.tanh(episode_return * 100 * 0.4)


def trading_reward(completions: list, **kwargs) -> list[float]:
    """Blended trading reward: step-level (30%) + episode-level (70%).

    Step-level: 5D composite counterfactual P&L with asymmetric penalties.
    Episode-level: Pre-computed episode return from prompt metadata, scaled
    to align with eval's grade_single_stock() grading function.

    The 70/30 blend ensures the model optimizes for what eval actually
    measures (episode return) while still getting per-step signal for
    credit assignment.
    """
    current_prices = kwargs.get("current_price", [])
    next_1d = kwargs.get("next_price_1d", [])
    next_2d = kwargs.get("next_price_2d", [])
    next_5d = kwargs.get("next_price_5d", [])
    has_positions = kwargs.get("has_position", [])
    rolling_vols = kwargs.get("rolling_vol", [])
    episode_returns = kwargs.get("episode_return", [])

    rewards = []
    for i, completion in enumerate(completions):
        text = extract_completion_text(completion)
        action = parse_action_word(text)

        # Track for anti-collapse monitoring
        _track_action(action)

        cur = current_prices[i] if i < len(current_prices) else 0.0
        vol = rolling_vols[i] if i < len(rolling_vols) and rolling_vols[i] > VOL_FLOOR else 0.02
        has_pos = has_positions[i] if i < len(has_positions) else False

        # Forward prices for step-level reward
        fwd_prices = {
            "1d": next_1d[i] if i < len(next_1d) else cur,
            "2d": next_2d[i] if i < len(next_2d) else cur,
            "5d": next_5d[i] if i < len(next_5d) else cur,
        }

        # Step-level reward (5D composite, asymmetric)
        step_reward = _compute_step_reward(action, has_pos, cur, fwd_prices, vol)

        # Episode-level reward (from pre-computed metadata)
        ep_return = episode_returns[i] if i < len(episode_returns) else 0.0
        episode_reward = _compute_episode_reward(ep_return)

        # Blend: 30% step + 70% episode
        reward = STEP_WEIGHT * step_reward + EPISODE_WEIGHT * episode_reward

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
    """Run GRPO v3 training."""

    # ------------------------------------------------------------------
    # 1. Environment check
    # ------------------------------------------------------------------
    logger.info("=== GRPO v3 Training ===")
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
    mlflow.set_experiment("grpo-v3-training")

    run_name = (
        f"grpo-v3-gen{args.num_generations}-lr{args.lr}"
        f"-steps{args.max_steps}-beta{args.beta}"
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
        "reward_functions": "format_gate,trading",
        "step_weight": STEP_WEIGHT,
        "episode_weight": EPISODE_WEIGHT,
        "bad_trade_penalty": BAD_TRADE_PENALTY,
        "horizon_weights": str(HORIZON_WEIGHTS),
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
        save_steps=200,
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
        reward_funcs=[format_gate, trading_reward],
        processing_class=tokenizer,
    )

    start_time = time.time()
    logger.info("Starting GRPO v3 training...")
    logger.info("  Prompts: %d", len(train_dataset))
    logger.info("  Max steps: %d", args.max_steps)
    logger.info("  Generations per prompt: %d", args.num_generations)
    logger.info("  Temperature: %.2f -> %.2f (cosine)", args.t_start, args.t_end)
    logger.info("  Learning rate: %.1e", args.lr)
    logger.info("  Beta (KL penalty): %.4f", args.beta)
    logger.info("  Reward functions: format_gate + trading (30%% step / 70%% episode)")
    logger.info("  Scale rewards: batch (Dr. GRPO)")
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
    parser.add_argument("--hf-repo", default="sarthakbiswas/stock-trader-grpo-v3-model")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
