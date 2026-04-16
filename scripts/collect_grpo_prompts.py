"""Collect GRPO training prompts by running episodes and saving observations + market metadata.

Each prompt includes the observation text (what the model sees) plus metadata
needed to compute counterfactual rewards for any candidate action (current price,
next-day price, position state).

Usage:
    PYTHONPATH=. python scripts/collect_grpo_prompts.py
    PYTHONPATH=. python scripts/collect_grpo_prompts.py --episodes 200 --split train
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from baselines.llm_agent import SYSTEM_PROMPT
from training.gym_wrapper import StockTradingGymEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "grpo"


def ensure_market_data() -> None:
    """Download market data from HF Hub if not locally available."""
    ohlcv_dir = Path("data/ohlcv")
    macro_dir = Path("data/macro")

    if ohlcv_dir.exists() and any(ohlcv_dir.glob("*.csv")) and macro_dir.exists() and any(macro_dir.glob("*.csv")):
        return

    logger.info("Market data missing. Downloading from HF Hub...")
    from datasets import load_dataset

    ds = load_dataset("sarthakbiswas/stock-trader-market-data")

    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_df = ds["ohlcv"].to_pandas()
    for symbol, group in ohlcv_df.groupby("symbol"):
        group = group.drop(columns=["symbol", "data_type"])
        group.to_csv(ohlcv_dir / f"{symbol}_daily.csv", index=False)

    macro_dir.mkdir(parents=True, exist_ok=True)
    macro_df = ds["macro"].to_pandas()
    for name, group in macro_df.groupby("symbol"):
        group = group.drop(columns=["symbol", "data_type"])
        group.to_csv(macro_dir / f"{name}_daily.csv", index=False)

    logger.info("Downloaded %d stocks + %d macro", len(ohlcv_df["symbol"].unique()), len(macro_df["symbol"].unique()))


def collect_prompts(
    task_id: str,
    n_episodes: int,
    seed: int,
    split: str,
) -> list[dict]:
    """Run episodes with random actions, collect observations + metadata."""
    rng = random.Random(seed)
    prompts = []

    for ep in range(n_episodes):
        episode_seed = seed + ep
        env = StockTradingGymEnv(
            task_id=task_id, seed=episode_seed, obs_mode="text", split=split,
        )
        obs, info = env.reset()

        # Access internal environment for market metadata
        internal_env = env._env
        sim = internal_env._sim
        portfolio = internal_env._portfolio
        symbols = internal_env._task_config["symbols"]

        step = 0
        while True:
            # Get current market state for metadata
            current_prices = {sym: sim.get_price(sym) for sym in symbols}
            portfolio_value = portfolio.get_value(current_prices)
            has_position = len(portfolio.positions) > 0

            position_info = {}
            if has_position:
                for sym, pos in portfolio.positions.items():
                    price = current_prices.get(sym, pos["avg_price"])
                    pnl_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100
                    position_info[sym] = {"qty": pos["qty"], "avg_price": pos["avg_price"], "pnl_pct": round(pnl_pct, 2)}

            cash_fraction = portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0

            # Peek at next-day prices (for counterfactual reward computation)
            # Advance day temporarily to get next prices, then revert
            next_prices = {}
            for sym in symbols:
                idx = sim._start_idx[sym] + 50 + sim._current_day + 1
                df = sim._all_data[sym]
                if idx < len(df):
                    next_prices[sym] = float(df.iloc[idx]["close"])
                else:
                    next_prices[sym] = current_prices[sym]

            # Build the prompt (same format as eval/inference)
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Here is today's market data:\n\n" + obs + "\n\nWhat is your trading action?"},
            ]

            # Store prompt + metadata
            prompts.append({
                "prompt": prompt_messages,
                "metadata": {
                    "episode": ep,
                    "step": step,
                    "current_prices": current_prices,
                    "next_prices": next_prices,
                    "has_position": has_position,
                    "position_info": position_info,
                    "cash_fraction": round(cash_fraction, 4),
                    "portfolio_value": round(portfolio_value, 2),
                    "day": sim.current_day + 1,
                    "total_days": sim.episode_days,
                },
            })

            # Take a random action to advance the episode
            action = rng.choice(["HOLD", "HOLD", "BUY", "SELL"])
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated:
                break

        env.close()

        if (ep + 1) % 20 == 0:
            logger.info("Collected %d/%d episodes (%d prompts)", ep + 1, n_episodes, len(prompts))

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GRPO training prompts")
    parser.add_argument("--task", default="single_stock")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ensure_market_data()

    logger.info("Collecting GRPO prompts: task=%s, episodes=%d, split=%s", args.task, args.episodes, args.split)
    prompts = collect_prompts(args.task, args.episodes, args.seed, args.split)

    # Save as JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"grpo_prompts_{args.task}_{args.split}_{args.episodes}ep.jsonl"

    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Saved %d prompts to %s", len(prompts), output_path)
    logger.info("Average %.1f steps per episode", len(prompts) / args.episodes)

    # Print action distribution from metadata
    has_pos_count = sum(1 for p in prompts if p["metadata"]["has_position"])
    logger.info("Prompts with position: %d (%.1f%%)", has_pos_count, has_pos_count / len(prompts) * 100)


if __name__ == "__main__":
    main()
