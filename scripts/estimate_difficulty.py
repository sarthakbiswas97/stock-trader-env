"""Estimate prompt difficulty and filter to hardest examples for GRPO.

Based on Pikus et al. "Hard Examples Are All You Need":
  - Hard examples maintain non-zero reward variance throughout GRPO training
  - Training on hardest 10% yields up to 47% more improvement than easy examples
  - Easy examples quickly "solve" (all G outputs converge), killing learning signal

This script:
  1. Loads collected prompts (from collect_model_episodes.py or collect_grpo_prompts.py)
  2. Runs G completions per prompt through the model
  3. Computes reward for each completion using trading_reward()
  4. Calculates "learnability" = reward_std * (1 - abs(mean_reward))
  5. Ranks prompts by learnability and keeps top N%

High learnability = model produces diverse actions with mixed rewards on this prompt.
These are the prompts where GRPO has the strongest learning signal.

Usage:
    PYTHONPATH=. python scripts/estimate_difficulty.py \\
        --checkpoint sarthakbiswas/stock-trader-grpo-v2.1-model \\
        --prompts data/grpo/model_episodes_train_4000.jsonl \\
        --num-samples 8 --keep-pct 35

    # Output: data/grpo/hard_prompts_train_1400.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from baselines.llm_agent import parse_action

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/grpo")

# Import reward constants from train_grpo to ensure consistency
TANH_STEEPNESS = 1.5
VOL_FLOOR = 0.005
STEP_WEIGHT = 0.30
EPISODE_WEIGHT = 0.70
BAD_TRADE_PENALTY = 1.5
HORIZON_WEIGHTS = {"1d": 0.2, "2d": 0.3, "5d": 0.5}


# ---------------------------------------------------------------------------
# Reward computation (duplicated from train_grpo.py for standalone use)
# ---------------------------------------------------------------------------


def _compute_step_reward(
    action: str,
    has_pos: bool,
    cur: float,
    fwd_prices: dict[str, float],
    vol: float,
) -> float:
    """5D composite step-level reward with asymmetric penalties."""
    if cur <= 0:
        return 0.0

    z_scores = {}
    for horizon, price in fwd_prices.items():
        change = (price - cur) / cur
        z_scores[horizon] = change / vol

    composite_z = sum(z_scores.get(h, 0.0) * w for h, w in HORIZON_WEIGHTS.items())
    base = math.tanh(composite_z * TANH_STEEPNESS)

    if action == "BUY" and not has_pos:
        return base * BAD_TRADE_PENALTY if composite_z < 0 else base
    elif action == "SELL" and has_pos:
        return -base * BAD_TRADE_PENALTY if composite_z > 0 else -base
    elif action == "HOLD" and has_pos:
        return base * 0.7
    elif action == "HOLD" and not has_pos:
        return -abs(base) * 0.5 if composite_z > 0 else abs(base) * 0.3
    else:
        return -0.3


def _compute_episode_reward(episode_return: float) -> float:
    """Episode-level reward aligned with eval grading."""
    return math.tanh(episode_return * 100 * 0.4)


def compute_reward(action: str, prompt: dict) -> float:
    """Compute blended trading reward for one action on one prompt."""
    cur = prompt.get("current_price", 0.0)
    vol = prompt.get("rolling_vol", 0.02)
    if vol < VOL_FLOOR:
        vol = 0.02
    has_pos = prompt.get("has_position", False)

    fwd_prices = {
        "1d": prompt.get("next_price_1d", cur),
        "2d": prompt.get("next_price_2d", cur),
        "5d": prompt.get("next_price_5d", cur),
    }

    step_reward = _compute_step_reward(action, has_pos, cur, fwd_prices, vol)
    ep_return = prompt.get("episode_return", 0.0)
    episode_reward = _compute_episode_reward(ep_return)

    return STEP_WEIGHT * step_reward + EPISODE_WEIGHT * episode_reward


# ---------------------------------------------------------------------------
# Difficulty estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifficultyScore:
    """Difficulty metrics for one prompt."""

    prompt_idx: int
    reward_mean: float
    reward_std: float
    learnability: float
    action_entropy: float
    actions: tuple[str, ...]
    rewards: tuple[float, ...]


def estimate_prompt_difficulty(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: dict,
    num_samples: int = 8,
    temperature: float = 0.8,
    prompt_idx: int = 0,
) -> DifficultyScore:
    """Generate num_samples completions for one prompt and compute difficulty.

    Learnability = reward_std * (1 - abs(reward_mean))
      - High std: model produces diverse outcomes (GRPO has signal)
      - Low abs(mean): prompt isn't already "solved" (not all positive or negative)
    """
    messages = prompt["prompt"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    actions = []
    rewards = []

    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True,
        )
        action = parse_action(response)
        reward = compute_reward(action, prompt)

        actions.append(action)
        rewards.append(reward)

    # Compute statistics
    reward_mean = sum(rewards) / len(rewards)
    reward_var = sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)
    reward_std = reward_var ** 0.5

    # Learnability: high variance + not already solved
    learnability = reward_std * (1.0 - abs(reward_mean))

    # Action entropy (diversity of actions)
    action_counts = Counter(actions)
    total = len(actions)
    entropy = 0.0
    for count in action_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return DifficultyScore(
        prompt_idx=prompt_idx,
        reward_mean=reward_mean,
        reward_std=reward_std,
        learnability=learnability,
        action_entropy=entropy,
        actions=tuple(actions),
        rewards=tuple(rewards),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate prompt difficulty for GRPO")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--prompts", required=True, help="Input JSONL file with prompts")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of completions per prompt (G for probing)")
    parser.add_argument("--keep-pct", type=float, default=35.0,
                        help="Percentage of hardest prompts to keep")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for difficulty probing")
    parser.add_argument("--min-regime-pct", type=float, default=10.0,
                        help="Minimum percentage of each regime in output")
    parser.add_argument("--output", default="", help="Output path (auto-generated if empty)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Not used for generation, but controls progress reporting")
    args = parser.parse_args()

    # Load model
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logger.info("Loading model: %s", args.checkpoint)
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.checkpoint,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded")

    # Load prompts
    prompts_path = Path(args.prompts)
    prompts: list[dict] = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    # Estimate difficulty for each prompt
    logger.info("Estimating difficulty: %d samples per prompt, temp=%.2f",
                args.num_samples, args.temperature)

    scores: list[DifficultyScore] = []
    for i in tqdm(range(len(prompts)), desc="Probing prompts"):
        score = estimate_prompt_difficulty(
            model=model,
            tokenizer=tokenizer,
            prompt=prompts[i],
            num_samples=args.num_samples,
            temperature=args.temperature,
            prompt_idx=i,
        )
        scores.append(score)

        if (i + 1) % 100 == 0:
            recent = scores[-100:]
            mean_learnability = sum(s.learnability for s in recent) / len(recent)
            mean_entropy = sum(s.action_entropy for s in recent) / len(recent)
            logger.info(
                "  Prompt %d/%d — mean_learnability=%.4f, mean_entropy=%.2f",
                i + 1, len(prompts), mean_learnability, mean_entropy,
            )

    # Summary before filtering
    all_learnability = [s.learnability for s in scores]
    all_entropy = [s.action_entropy for s in scores]
    all_std = [s.reward_std for s in scores]
    logger.info("")
    logger.info("=== Difficulty Summary ===")
    logger.info("Learnability: mean=%.4f, max=%.4f, min=%.4f",
                sum(all_learnability) / len(all_learnability),
                max(all_learnability), min(all_learnability))
    logger.info("Reward std: mean=%.4f, max=%.4f",
                sum(all_std) / len(all_std), max(all_std))
    logger.info("Action entropy: mean=%.2f, max=%.2f",
                sum(all_entropy) / len(all_entropy), max(all_entropy))

    # Zero learnability = all samples got same reward = no GRPO signal
    zero_learn = sum(1 for s in scores if s.learnability < 1e-6)
    logger.info("Zero-learnability prompts: %d (%.1f%%) — would waste training steps",
                zero_learn, zero_learn / len(scores) * 100)

    # Sort by learnability (highest first) and keep top N%
    ranked = sorted(enumerate(scores), key=lambda x: x[1].learnability, reverse=True)
    n_keep = int(len(ranked) * args.keep_pct / 100.0)
    kept_indices = {idx for idx, _ in ranked[:n_keep]}

    # Check regime coverage and supplement if needed
    kept_prompts = [prompts[idx] for idx in sorted(kept_indices)]
    regime_counts = Counter(p["regime"] for p in kept_prompts)
    total_kept = len(kept_prompts)
    min_regime_count = int(total_kept * args.min_regime_pct / 100.0)

    supplemented = 0
    for regime in ["strong_bear", "mild_bear", "sideways", "mild_bull", "strong_bull"]:
        current = regime_counts.get(regime, 0)
        if current < min_regime_count:
            # Find prompts of this regime not yet selected, ranked by learnability
            candidates = [
                (idx, scores[idx])
                for idx in range(len(prompts))
                if idx not in kept_indices and prompts[idx]["regime"] == regime
            ]
            candidates.sort(key=lambda x: x[1].learnability, reverse=True)
            needed = min_regime_count - current
            for idx, _ in candidates[:needed]:
                kept_indices.add(idx)
                supplemented += 1

    if supplemented > 0:
        logger.info("Supplemented %d prompts for regime coverage", supplemented)

    # Build final output
    final_prompts = [prompts[idx] for idx in sorted(kept_indices)]

    # Remove model_action and model_response if present (not needed for training)
    cleaned = []
    for p in final_prompts:
        cleaned_p = {k: v for k, v in p.items() if k not in ("model_action", "model_response")}
        cleaned.append(cleaned_p)

    logger.info("")
    logger.info("=== Filtering Results ===")
    logger.info("Input: %d prompts", len(prompts))
    logger.info("Kept: %d prompts (%.1f%% + %d regime supplements)",
                len(cleaned), args.keep_pct, supplemented)

    regime_final = Counter(p["regime"] for p in cleaned)
    logger.info("Regime distribution: %s", dict(regime_final.most_common()))

    # Learnability stats of kept vs dropped
    kept_scores = [scores[idx] for idx in sorted(kept_indices)]
    dropped_scores = [scores[idx] for idx in range(len(scores)) if idx not in kept_indices]
    kept_learn = sum(s.learnability for s in kept_scores) / len(kept_scores) if kept_scores else 0
    dropped_learn = sum(s.learnability for s in dropped_scores) / len(dropped_scores) if dropped_scores else 0
    logger.info("Mean learnability — kept: %.4f, dropped: %.4f (%.1fx harder)",
                kept_learn, dropped_learn,
                kept_learn / dropped_learn if dropped_learn > 0 else float("inf"))

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"hard_prompts_{len(cleaned)}.jsonl"

    with open(output_path, "w") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Saved %d hard prompts to %s", len(cleaned), output_path)

    # Save difficulty scores for analysis
    scores_path = output_path.with_suffix(".scores.json")
    score_records = [
        {
            "idx": s.prompt_idx,
            "learnability": round(s.learnability, 6),
            "reward_std": round(s.reward_std, 6),
            "reward_mean": round(s.reward_mean, 6),
            "action_entropy": round(s.action_entropy, 4),
            "actions": list(s.actions),
            "rewards": [round(r, 4) for r in s.rewards],
            "kept": s.prompt_idx in kept_indices,
        }
        for s in scores
    ]
    with open(scores_path, "w") as f:
        json.dump(score_records, f, indent=2)

    logger.info("Saved difficulty scores to %s", scores_path)


if __name__ == "__main__":
    main()
