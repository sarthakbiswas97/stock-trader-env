"""Offline LLM-as-Judge scoring for GRPO training data.

Pre-scores BUY/SELL trading decisions using GPT-4o-mini rubric.
Scores are cached in SQLite to avoid re-scoring on re-runs.
HOLD actions get a default score of 0.0 (neutral).

Usage:
    # Score GRPO prompts
    OPENAI_API_KEY=... PYTHONPATH=. python scripts/score_with_llm.py \
        --input data/grpo/prompts_train_4000.jsonl \
        --output data/grpo/prompts_train_4000_scored.jsonl

    # Score with concurrency control
    OPENAI_API_KEY=... PYTHONPATH=. python scripts/score_with_llm.py \
        --input data/grpo/prompts.jsonl \
        --output data/grpo/prompts_scored.jsonl \
        --max-concurrent 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

from training.judge_prompt import (
    JudgeContext,
    build_judge_prompt,
    parse_judge_response,
)
from training.llm_client import LLMJudge, openai_4o_mini, deepseek_v3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def extract_action_from_prompt(row: dict) -> str:
    """Extract the action word (BUY/SELL/HOLD) from a GRPO prompt row.

    The prompt column contains chat messages. The action is what the model
    would generate, but in collected data it's stored in metadata or
    can be inferred from the prompt context.
    """
    # Check if action is directly in the row
    if "action" in row:
        action = str(row["action"]).strip().upper()
        match = re.match(r"(BUY|SELL|HOLD)", action)
        return match.group(1) if match else "HOLD"

    # Check response field
    if "response" in row:
        text = str(row["response"]).strip()
        for line in reversed(text.split("\n")):
            match = re.match(r"(BUY|SELL|HOLD)", line.strip().upper())
            if match:
                return match.group(1)

    return "HOLD"


def extract_reasoning(row: dict) -> str:
    """Extract <think>...</think> reasoning from row."""
    response = row.get("response", "")
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(0)
    return response[:300] if response else "<think>No reasoning provided.</think>"


def extract_observation(row: dict) -> str:
    """Extract observation text from prompt messages or direct field."""
    # Direct observation field
    if "observation" in row:
        return str(row["observation"])[:800]

    # From prompt messages (GRPO format)
    prompt = row.get("prompt", [])
    if isinstance(prompt, list):
        for msg in prompt:
            if msg.get("role") == "user":
                return msg.get("content", "")[:800]

    return ""


def build_context_from_row(row: dict) -> JudgeContext | None:
    """Build JudgeContext from a GRPO dataset row."""
    action = extract_action_from_prompt(row)
    if action == "HOLD":
        return None

    observation = extract_observation(row)
    if not observation:
        return None

    reasoning = extract_reasoning(row)
    has_position = bool(row.get("has_position", False))
    position_pnl = row.get("position_pnl")

    return JudgeContext(
        observation=observation,
        action=action,
        reasoning=reasoning,
        has_position=has_position,
        position_pnl=float(position_pnl) if position_pnl is not None else None,
    )


async def score_dataset(
    input_path: Path,
    output_path: Path,
    provider: str,
    max_concurrent: int,
    only_trades: bool,
) -> None:
    """Score all rows in a JSONL dataset."""
    # Load data
    rows: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    logger.info("Loaded %d rows from %s", len(rows), input_path)

    # Initialize judge
    config = openai_4o_mini() if provider == "openai" else deepseek_v3()
    judge = LLMJudge(config, use_cache=True)

    scored = 0
    cached = 0
    failed = 0
    total_score = 0.0

    # Score each row
    for i, row in enumerate(rows):
        ctx = build_context_from_row(row)

        if ctx is None:
            # HOLD or unparseable — neutral score
            row["llm_score"] = 0.0
            row["llm_criteria"] = {}
            continue

        messages = build_judge_prompt(ctx)
        raw_response = await judge.score(messages)

        if raw_response:
            result = parse_judge_response(raw_response)
            row["llm_score"] = result.total
            row["llm_criteria"] = result.criteria_dict
            total_score += result.total
            scored += 1
        else:
            row["llm_score"] = 0.0
            row["llm_criteria"] = {}
            failed += 1

        if (i + 1) % 50 == 0:
            mean = total_score / scored if scored > 0 else 0
            logger.info(
                "Progress: %d/%d | Scored: %d | Failed: %d | Mean score: %.3f",
                i + 1, len(rows), scored, failed, mean,
            )

    judge.close()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    mean = total_score / scored if scored > 0 else 0
    holds = len(rows) - scored - failed
    logger.info("")
    logger.info("=== Scoring Complete ===")
    logger.info("Total rows: %d", len(rows))
    logger.info("Scored (BUY/SELL): %d | Mean score: %.3f", scored, mean)
    logger.info("HOLDs (skipped): %d", holds)
    logger.info("Failed: %d", failed)
    logger.info("Output: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge batch scoring")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output scored JSONL path")
    parser.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument(
        "--only-trades", action="store_true", default=True,
        help="Only score BUY/SELL, skip HOLD (saves API cost)",
    )
    args = parser.parse_args()

    start = time.time()
    asyncio.run(score_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        provider=args.provider,
        max_concurrent=args.max_concurrent,
        only_trades=args.only_trades,
    ))
    logger.info("Total time: %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
