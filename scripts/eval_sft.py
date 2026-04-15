"""Evaluate SFT model against the trading environment.

Usage:
    PYTHONPATH=. python scripts/eval_sft.py --checkpoint /workspace/sft-v2-checkpoint
"""

from __future__ import annotations

import argparse
import logging

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from baselines.llm_agent import SYSTEM_PROMPT, parse_action
from training.evaluate import evaluate_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument("--checkpoint", default="/workspace/sft-v2-checkpoint")
    parser.add_argument("--task", default="single_stock")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

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
    logger.info("Score:  %.3f", results.mean_score)
    logger.info("Return: %+.2f%%", results.mean_return)
    logger.info("Sharpe: %.2f", results.mean_sharpe)


if __name__ == "__main__":
    main()
