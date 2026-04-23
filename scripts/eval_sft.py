"""Evaluate SFT model against the trading environment.

Downloads market data from HF Hub if not locally available,
ensuring observations match the training format (including macro context).

Results are printed to stdout AND saved to a JSON file, so they
survive noisy warning output from transformers/unsloth.

Usage:
    PYTHONPATH=. python scripts/eval_sft.py --checkpoint /workspace/sft-v2-checkpoint
    PYTHONPATH=. python scripts/eval_sft.py --checkpoint sarthakbiswas/stock-trader-sft-v2-model
    PYTHONPATH=. python scripts/eval_sft.py --checkpoint /workspace/raft-checkpoint --simulator-mode neural
"""

from __future__ import annotations

# Suppress warnings BEFORE any other imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*Unsloth should be imported.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# Silence noisy loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("unsloth").setLevel(logging.ERROR)

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
        print(f"Market data found: {ohlcv_count} stocks, {macro_count} macro instruments")
        return

    print(f"Market data missing. Downloading from HF Hub: {HF_MARKET_DATA}")
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

    print(
        f"Downloaded {len(ohlcv_df['symbol'].unique())} stocks "
        f"+ {len(macro_df['symbol'].unique())} macro instruments"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument("--checkpoint", default="/workspace/sft-v2-checkpoint")
    parser.add_argument("--task", default="single_stock")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--simulator-mode", default="replay", choices=["replay", "neural"])
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    ensure_market_data()

    print(f"Loading model from {args.checkpoint}")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.checkpoint, max_seq_length=1024, load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    print("Model loaded")

    from baselines.llm_agent import SYSTEM_PROMPT, parse_action
    from training.evaluate import evaluate_agent

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
        return parse_action(response)

    print(f"Evaluating: task={args.task}, episodes={args.episodes}, seed={args.seed}, mode={args.simulator_mode}")
    results = evaluate_agent(
        sft_agent, task_id=args.task, n_episodes=args.episodes,
        seed=args.seed, split=args.split, simulator_mode=args.simulator_mode,
        log_to_mlflow=False,
    )

    # Print results to stdout (survives any stderr noise)
    print("")
    print("=" * 50)
    print(f"RESULTS ({args.simulator_mode} env, {args.episodes} episodes)")
    print("=" * 50)
    print(f"Score:  {results.mean_score:.3f} (+/- {results.std_score:.3f})")
    print(f"Return: {results.mean_return:+.2f}% (+/- {results.std_return:.2f}%)")
    print(f"Sharpe: {results.sharpe:.2f}")
    print(f"Per-episode scores: {[round(s, 3) for s in results.scores]}")
    print("=" * 50)

    # Save to JSON file
    output_path = args.output or f"/root/eval_{args.simulator_mode}_{args.episodes}ep.json"
    results_dict = {
        "checkpoint": args.checkpoint,
        "simulator_mode": args.simulator_mode,
        "task": args.task,
        "episodes": args.episodes,
        "seed": args.seed,
        "mean_score": results.mean_score,
        "std_score": results.std_score,
        "mean_return": results.mean_return,
        "std_return": results.std_return,
        "sharpe": results.sharpe,
        "scores": list(results.scores),
        "returns": list(results.returns),
    }
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
