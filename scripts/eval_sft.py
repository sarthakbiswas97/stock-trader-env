"""Evaluate SFT model against the trading environment.

Downloads market data from HF Hub if not locally available,
ensuring observations match the training format (including macro context).

Usage:
    PYTHONPATH=. python scripts/eval_sft.py --checkpoint /workspace/sft-v2-checkpoint
    PYTHONPATH=. python scripts/eval_sft.py --checkpoint sarthakbiswas/stock-trader-sft-v2-model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from baselines.llm_agent import SYSTEM_PROMPT, parse_action
from training.evaluate import evaluate_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OHLCV_DIR = Path("data/ohlcv")
MACRO_DIR = Path("data/macro")
HF_MARKET_DATA = "sarthakbiswas/stock-trader-market-data"


def ensure_market_data() -> None:
    """Download market data from HF Hub if not locally available."""
    ohlcv_exists = OHLCV_DIR.exists() and any(OHLCV_DIR.glob("*.csv"))
    macro_exists = MACRO_DIR.exists() and any(MACRO_DIR.glob("*.csv"))

    if ohlcv_exists and macro_exists:
        ohlcv_count = len(list(OHLCV_DIR.glob("*.csv")))
        macro_count = len(list(MACRO_DIR.glob("*.csv")))
        logger.info("Market data found: %d stocks, %d macro instruments", ohlcv_count, macro_count)
        return

    logger.info("Market data missing. Downloading from HF Hub: %s", HF_MARKET_DATA)
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

    logger.info(
        "Downloaded %d stocks + %d macro instruments",
        len(ohlcv_df["symbol"].unique()),
        len(macro_df["symbol"].unique()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument("--checkpoint", default="/workspace/sft-v2-checkpoint")
    parser.add_argument("--task", default="single_stock")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    # Ensure market data is available (downloads from HF Hub if missing)
    ensure_market_data()

    logger.info("Loading model from %s", args.checkpoint)
    model, tokenizer = FastLanguageModel.from_pretrained(args.checkpoint)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded")

    def sft_agent(observation: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Here is today's market data:\n\n" + observation + "\n\nWhat is your trading action?"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_action(response)
        logger.info("  Action: %s | Response: %s", action, response[:100])
        return action

    logger.info("Evaluating: task=%s, episodes=%d, seed=%d, split=%s", args.task, args.episodes, args.seed, args.split)
    results = evaluate_agent(sft_agent, task_id=args.task, n_episodes=args.episodes, seed=args.seed, split=args.split)

    logger.info("")
    logger.info("=== Results ===")
    logger.info("Score:  %.3f (+/- %.3f)", results.mean_score, results.std_score)
    logger.info("Return: %+.2f%% (+/- %.2f%%)", results.mean_return, results.std_return)
    logger.info("Sharpe: %.2f", results.sharpe)
    logger.info("Episodes: %d", results.episodes)
    logger.info("Per-episode scores: %s", [round(s, 3) for s in results.scores])


if __name__ == "__main__":
    main()
