"""Collect GRPO training prompts from the MODEL's own episodes.

Instead of using a heuristic agent to generate portfolio states (v2/v2.1),
this script runs the actual trained model against the live environment and
collects the trajectories. This is the key insight from the research:

  "Model trains on its OWN successful trajectories, not a heuristic agent's."

For each episode:
  1. Model plays a full 20-day episode against the gym env
  2. Episode is scored by the env grader (grade_single_stock)
  3. All (observation, model_action, metadata) pairs are collected
  4. Forward prices and volatility metadata are computed from raw market data

The output format is identical to collect_grpo_prompts.py so it can be
fed directly to estimate_difficulty.py and then train_grpo.py.

Usage:
    PYTHONPATH=. python scripts/collect_model_episodes.py \\
        --checkpoint sarthakbiswas/stock-trader-grpo-v2.1-model \\
        --episodes 200 --split train

    # Or with a local checkpoint:
    PYTHONPATH=. python scripts/collect_model_episodes.py \\
        --checkpoint /workspace/grpo-checkpoint --episodes 200
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from baselines.llm_agent import SYSTEM_PROMPT, parse_action
from server.market_simulator import _load_stock_data
from training.gym_wrapper import StockTradingGymEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OHLCV_DIR = Path("data/ohlcv")
MACRO_DIR = Path("data/macro")
OUTPUT_DIR = Path("data/grpo")

LOOKBACK = 50
EPISODE_DAYS = 20
MAX_FORWARD_HORIZON = 5
VOL_FLOOR = 0.005

HF_MARKET_DATA = "sarthakbiswas/stock-trader-market-data"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    """Immutable record of one completed episode."""

    episode_id: str
    seed: int
    score: float
    episode_return: float
    prompts: tuple[dict, ...]
    actions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------


def ensure_market_data() -> None:
    """Download market data from HF Hub if not locally available."""
    ohlcv_exists = OHLCV_DIR.exists() and any(OHLCV_DIR.glob("*.csv"))
    macro_exists = MACRO_DIR.exists() and any(MACRO_DIR.glob("*.csv"))

    if ohlcv_exists and macro_exists:
        return

    logger.info("Downloading market data from HF Hub: %s", HF_MARKET_DATA)
    from datasets import load_dataset

    ds = load_dataset(HF_MARKET_DATA)

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    ohlcv_df = ds["ohlcv"].to_pandas()
    for symbol, group in ohlcv_df.groupby("symbol"):
        group = group.drop(columns=["symbol", "data_type"])
        group.to_csv(OHLCV_DIR / f"{symbol}_daily.csv", index=False)

    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    macro_df = ds["macro"].to_pandas()
    for name, group in macro_df.groupby("symbol"):
        group = group.drop(columns=["symbol", "data_type"])
        group.to_csv(MACRO_DIR / f"{name}_daily.csv", index=False)

    logger.info("Market data downloaded")


def compute_rolling_volatility(closes: pd.Series, idx: int, period: int = 20) -> float:
    """20-day rolling std of daily returns. Floors at VOL_FLOOR."""
    if idx < period:
        return 0.02
    window = closes.iloc[idx - period : idx]
    returns = window.pct_change().dropna()
    if len(returns) < 5:
        return 0.02
    return max(float(returns.std()), VOL_FLOOR)


def classify_regime(z_score: float) -> str:
    """Classify market regime from volatility-normalized daily return."""
    if z_score <= -2.0:
        return "strong_bear"
    if z_score <= -0.8:
        return "mild_bear"
    if z_score >= 2.0:
        return "strong_bull"
    if z_score >= 0.8:
        return "mild_bull"
    return "sideways"


def get_forward_prices(
    closes: pd.Series, idx: int, horizons: tuple[int, ...] = (1, 2, 5),
) -> dict[str, float]:
    """Get forward prices at multiple horizons. Falls back to current price."""
    current = float(closes.iloc[idx])
    result = {}
    for h in horizons:
        fwd_idx = idx + h
        if fwd_idx < len(closes):
            result[f"{h}d"] = float(closes.iloc[fwd_idx])
        else:
            result[f"{h}d"] = current
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint: str) -> tuple:
    """Load model and tokenizer via Unsloth."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logger.info("Loading model: %s", checkpoint)
    model, tokenizer = FastLanguageModel.from_pretrained(
        checkpoint,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded")
    return model, tokenizer


def model_inference(
    model: torch.nn.Module,
    tokenizer: object,
    observation: str,
    temperature: float = 0.7,
) -> tuple[str, str]:
    """Run model inference on one observation.

    Returns:
        (action, raw_response) tuple.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is today's market data:\n\n{observation}\n\nWhat is your trading action?",
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

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
    return action, response


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------


def run_episode(
    model: torch.nn.Module,
    tokenizer: object,
    episode_seed: int,
    task_id: str,
    split: str,
    temperature: float = 0.7,
    simulator_mode: str = "replay",
) -> EpisodeResult:
    """Run one full episode with the model and collect all data.

    Returns an EpisodeResult with prompts in the same format as
    collect_grpo_prompts.py output.

    Supports both replay (CSV) and neural (world model) simulator modes.
    """
    env = StockTradingGymEnv(
        task_id=task_id, seed=episode_seed, obs_mode="text",
        split=split, simulator_mode=simulator_mode,
    )
    obs, info = env.reset()
    initial_value = info["portfolio_value"]

    # We need access to raw market data for metadata computation.
    # The gym env wraps StockTradingEnvironment which has the simulator.
    sim = env._env._sim
    symbol = sim.symbols[0]  # single_stock task has one symbol

    if simulator_mode == "neural":
        # Neural env: prices are in sim._generated[symbol]
        # The DataFrame has LOOKBACK seed rows + episode_days generated rows
        from server.neural_simulator import LOOKBACK as NEURAL_LOOKBACK
        gen_df = sim._generated[symbol]
        closes = gen_df["close"].reset_index(drop=True)
        start_idx = 0  # closes already starts from seed beginning
        lookback_offset = NEURAL_LOOKBACK  # episode starts at this index
    else:
        # Replay: prices are in CSV files
        start_idx = sim._start_idx[symbol]
        df = _load_stock_data(symbol)
        closes = df["close"]
        lookback_offset = LOOKBACK  # use the script's constant

    observations: list[str] = []
    actions: list[str] = []
    prompts: list[dict] = []

    day = 0
    while True:
        # Collect observation before acting
        observations.append(obs)

        # Get model's action
        action, response = model_inference(model, tokenizer, obs, temperature)
        actions.append(action)

        # Compute metadata for this step
        data_idx = start_idx + lookback_offset + day
        if data_idx < len(closes):
            price = float(closes.iloc[data_idx])
            prev_price = float(closes.iloc[data_idx - 1]) if data_idx > 0 else price
            rolling_vol = compute_rolling_volatility(closes, data_idx)
            daily_return = (price - prev_price) / prev_price if prev_price > 0 else 0.0
            z_score = daily_return / rolling_vol if rolling_vol > 0 else 0.0
            regime = classify_regime(z_score)
            fwd_prices = get_forward_prices(closes, data_idx)
        else:
            price = 0.0
            rolling_vol = 0.02
            z_score = 0.0
            regime = "sideways"
            fwd_prices = {"1d": 0.0, "2d": 0.0, "5d": 0.0}

        # Build prompt messages (same format as collect_grpo_prompts.py)
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Here is today's market data:\n\n{obs}\n\nWhat is your trading action?",
            },
        ]

        # Portfolio state from env info
        has_position = info["num_positions"] > 0
        portfolio_value = info["portfolio_value"]
        cash = info["cash"]
        cash_fraction = cash / portfolio_value if portfolio_value > 0 else 1.0

        prompt_record = {
            "prompt": prompt_messages,
            "symbol": symbol,
            "current_price": round(price, 2),
            "next_price_1d": round(fwd_prices.get("1d", price), 2),
            "next_price_2d": round(fwd_prices.get("2d", price), 2),
            "next_price_5d": round(fwd_prices.get("5d", price), 2),
            "has_position": has_position,
            "position_qty": 0,  # Simplified — exact qty not critical for reward
            "position_avg_price": 0.0,
            "position_pnl_pct": 0.0,
            "cash_fraction": round(cash_fraction, 4),
            "portfolio_value": round(portfolio_value, 2),
            "rolling_vol": round(rolling_vol, 6),
            "z_score": round(z_score, 4),
            "regime": regime,
            "day": day + 1,
            "total_days": EPISODE_DAYS,
            "episode_id": f"{symbol}_model_ep{episode_seed}",
            "episode_return": 0.0,  # Placeholder — filled after episode completes
            "model_action": action,  # Extra: what the model actually chose
            "model_response": response[:500],  # Extra: raw response (truncated)
        }
        prompts.append(prompt_record)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        day += 1

        if terminated:
            break

    env.close()

    # Compute episode return and score
    final_value = info["portfolio_value"]
    episode_return = (final_value - initial_value) / initial_value
    score = info["score"]

    # Backfill episode_return into all prompts
    filled_prompts = []
    for p in prompts:
        updated = dict(p)
        updated["episode_return"] = round(episode_return, 6)
        filled_prompts.append(updated)

    return EpisodeResult(
        episode_id=f"{symbol}_model_ep{episode_seed}",
        seed=episode_seed,
        score=score,
        episode_return=episode_return,
        prompts=tuple(filled_prompts),
        actions=tuple(actions),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect prompts from model's own episodes")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (HF repo or local path)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run")
    parser.add_argument("--task", default="single_stock")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--simulator-mode", default="replay", choices=["replay", "neural"],
                        help="Environment mode: replay (CSV) or neural (world model)")
    parser.add_argument("--min-z", type=float, default=0.0,
                        help="Min |z_score| to keep a prompt (0.0 = keep all)")
    parser.add_argument("--output", default="", help="Output JSONL path (auto-generated if empty)")
    args = parser.parse_args()

    ensure_market_data()
    model, tokenizer = load_model(args.checkpoint)

    logger.info("Running %d episodes: task=%s, split=%s, seed=%d, temp=%.2f",
                args.episodes, args.task, args.split, args.seed, args.temperature)

    all_results: list[EpisodeResult] = []
    all_prompts: list[dict] = []

    for i in range(args.episodes):
        episode_seed = args.seed + i

        result = run_episode(
            model=model,
            tokenizer=tokenizer,
            episode_seed=episode_seed,
            task_id=args.task,
            split=args.split,
            temperature=args.temperature,
            simulator_mode=args.simulator_mode,
        )
        all_results.append(result)

        # Filter by z_score if requested
        for p in result.prompts:
            if abs(p["z_score"]) >= args.min_z:
                all_prompts.append(p)

        if (i + 1) % 10 == 0:
            scores = [r.score for r in all_results]
            mean_score = sum(scores) / len(scores)
            good_eps = sum(1 for s in scores if s >= 0.5)
            logger.info(
                "Episode %d/%d — mean_score=%.3f, good_episodes=%d/%d, prompts=%d",
                i + 1, args.episodes, mean_score, good_eps, len(scores), len(all_prompts),
            )

    # Summary statistics
    scores = [r.score for r in all_results]
    returns = [r.episode_return for r in all_results]
    mean_score = sum(scores) / len(scores)
    mean_return = sum(returns) / len(returns)

    logger.info("")
    logger.info("=== Collection Summary ===")
    logger.info("Episodes: %d", len(all_results))
    logger.info("Mean score: %.3f", mean_score)
    logger.info("Mean return: %+.2f%%", mean_return * 100)
    logger.info("Score distribution:")

    score_buckets = {">=0.5 (good)": 0, "0.3-0.5 (mediocre)": 0, "<0.3 (bad)": 0}
    for s in scores:
        if s >= 0.5:
            score_buckets[">=0.5 (good)"] += 1
        elif s >= 0.3:
            score_buckets["0.3-0.5 (mediocre)"] += 1
        else:
            score_buckets["<0.3 (bad)"] += 1
    for bucket, count in score_buckets.items():
        logger.info("  %s: %d (%.0f%%)", bucket, count, count / len(scores) * 100)

    logger.info("Total prompts collected: %d", len(all_prompts))

    # Regime distribution
    regime_counts = Counter(p["regime"] for p in all_prompts)
    logger.info("Regime distribution: %s", dict(regime_counts.most_common()))

    # Action distribution (what the model actually chose)
    action_counts = Counter(p["model_action"] for p in all_prompts)
    logger.info("Model action distribution: %s", dict(action_counts.most_common()))

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"model_episodes_{args.split}_{len(all_prompts)}.jsonl"

    # Remove model_response before saving (large, not needed for training)
    save_prompts = []
    for p in all_prompts:
        cleaned = {k: v for k, v in p.items() if k != "model_response"}
        save_prompts.append(cleaned)

    with open(output_path, "w") as f:
        for p in save_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Saved %d prompts to %s", len(save_prompts), output_path)

    # Also save episode-level results for analysis
    episodes_path = output_path.with_suffix(".episodes.json")
    episode_records = [
        {
            "episode_id": r.episode_id,
            "seed": r.seed,
            "score": round(r.score, 4),
            "episode_return": round(r.episode_return, 6),
            "n_prompts": len(r.prompts),
            "actions": list(r.actions),
        }
        for r in all_results
    ]
    with open(episodes_path, "w") as f:
        json.dump(episode_records, f, indent=2)

    logger.info("Saved episode results to %s", episodes_path)


if __name__ == "__main__":
    main()
