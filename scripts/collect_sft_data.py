"""Collect SFT training data by running agents through episodes with trajectory logging.

Generates (observation, reasoning, action) triples formatted for chat fine-tuning.

Usage:
    PYTHONPATH=. python scripts/collect_sft_data.py
    PYTHONPATH=. python scripts/collect_sft_data.py --episodes 500 --task portfolio
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from baselines.llm_agent import SYSTEM_PROMPT
from baselines.rule_based_agent import rule_based_agent, _parse_stocks, _parse_positions
from training.gym_wrapper import StockTradingGymEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sft"


def generate_reasoning(observation: str, action: str) -> str:
    """Generate synthetic chain-of-thought reasoning for a rule-based action.

    Since the rule-based agent doesn't explain itself, we reconstruct
    the reasoning from the observation data. This teaches the LLM
    to think before acting.
    """
    stocks = _parse_stocks(observation)
    positions = _parse_positions(observation)

    reasons = []

    if action == "HOLD":
        reasons.append("No clear entry or exit signals.")
        for sym, data in stocks.items():
            if 35 <= data["rsi"] <= 65:
                reasons.append(f"{sym} RSI is {data['rsi']} (neutral range).")
                break
        if positions:
            for sym, pnl in positions.items():
                if -3 < pnl < 3:
                    reasons.append(f"Holding {sym} at {pnl:+.1f}% P&L, within tolerance.")
                    break

    elif action.startswith("BUY"):
        symbol = action.split()[1] if len(action.split()) > 1 else list(stocks.keys())[0] if stocks else "unknown"
        data = stocks.get(symbol, {"rsi": 0, "trend": "unknown"})
        reasons.append(f"{symbol} RSI is {data['rsi']} (oversold territory).")
        if data["trend"] != "bearish":
            reasons.append(f"Trend is {data['trend']}, supporting entry.")
        reasons.append("Mean-reversion setup: buy the dip.")

    elif action.startswith("SELL"):
        symbol = action.split()[1] if len(action.split()) > 1 else list(positions.keys())[0] if positions else "unknown"
        pnl = positions.get(symbol, 0.0)
        data = stocks.get(symbol, {"rsi": 50, "trend": "unknown"})
        if pnl > 3:
            reasons.append(f"{symbol} P&L is {pnl:+.1f}%, hitting profit target.")
        elif pnl < -3:
            reasons.append(f"{symbol} P&L is {pnl:+.1f}%, triggering stop-loss.")
        elif data["rsi"] > 65:
            reasons.append(f"{symbol} RSI is {data['rsi']} (overbought), taking profit.")
        reasons.append("Exiting position to lock in gains or cut losses.")

    return " ".join(reasons) if reasons else "Market conditions are neutral."


def collect_episodes(
    task_id: str,
    n_episodes: int,
    seed: int,
    split: str,
) -> list[dict]:
    """Run rule-based agent through episodes and format as SFT examples."""
    examples = []

    for i in range(n_episodes):
        episode_seed = seed + i
        env = StockTradingGymEnv(
            task_id=task_id, seed=episode_seed, obs_mode="text", split=split,
        )
        obs, info = env.reset()

        while True:
            action = rule_based_agent(obs)
            reasoning = generate_reasoning(obs, action)

            # Format as chat message
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs},
                    {"role": "assistant", "content": f"<think>{reasoning}</think>\n{action}"},
                ]
            })

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        env.close()

        if (i + 1) % 50 == 0:
            logger.info("Collected %d/%d episodes (%d examples)", i + 1, n_episodes, len(examples))

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SFT training data")
    parser.add_argument("--task", default="single_stock", help="Task ID")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--split", default="train", help="Data split")
    args = parser.parse_args()

    logger.info("Collecting SFT data: task=%s, episodes=%d, split=%s", args.task, args.episodes, args.split)

    examples = collect_episodes(args.task, args.episodes, args.seed, args.split)

    # Save as JSONL (one example per line — standard format for SFT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"sft_{args.task}_{args.split}_{args.episodes}ep.jsonl"

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Saved %d examples to %s", len(examples), output_path)
    logger.info("Average %.1f steps per episode", len(examples) / args.episodes)

    # Print a sample
    logger.info("")
    logger.info("=== Sample Example ===")
    sample = examples[5] if len(examples) > 5 else examples[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:200]
        logger.info("[%s] %s...", role, content)


if __name__ == "__main__":
    main()
