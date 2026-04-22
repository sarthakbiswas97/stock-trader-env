"""Reverse distillation: generate expert reasoning from oracle labels via LLM.

Replaces template-based reasoning in SFT data with causal explanations
from GPT-4o-mini. The model learns WHY indicators support an action,
not just WHAT the indicators are.

Usage:
    # Pilot (5K examples, ~$1.20)
    PYTHONPATH=. python scripts/distill_reasoning.py --limit 5000

    # Full run (20K examples, ~$5)
    PYTHONPATH=. python scripts/distill_reasoning.py --limit 20000

    # Inspect pilot output
    head -5 data/sft/sft_distilled_train.jsonl | python -m json.tool
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SFT_DIR = Path(__file__).parent.parent / "data" / "sft"
INPUT_FILE = SFT_DIR / "sft_oracle_v2_train.jsonl"
OUTPUT_FILE = SFT_DIR / "sft_distilled_train.jsonl"

DISTILLATION_PROMPT = """You are a professional Indian equity trader analyzing a trading decision.

Market observation:
{observation}

The correct action is: {action}

Write 2-4 sentences of expert reasoning explaining WHY this action is correct.
Requirements:
- Reference specific indicator values from the observation and explain what they mean TOGETHER
- If indicators conflict (e.g. RSI oversold but trend bearish), explain which takes priority and why
- Include one sentence about risk (what could go wrong with this trade)
- Be concise and actionable — no generic advice

Respond with ONLY the reasoning text, no action."""

MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.1


def load_sft_records(path: Path, limit: int | None = None) -> list[dict]:
    """Load existing SFT JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    logger.info(f"Loaded {len(records)} records from {path.name}")
    return records


def extract_observation_and_action(record: dict) -> tuple[str, str]:
    """Extract the observation text and action from an SFT record."""
    messages = record["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")

    observation = user_msg.replace("Here is today's market data:\n\n", "").replace(
        "\n\nWhat is your trading action?", ""
    )

    lines = assistant_msg.strip().split("\n")
    action = lines[-1].strip()

    return observation, action


def distill_reasoning(
    client: OpenAI,
    observation: str,
    action: str,
) -> str | None:
    """Generate expert reasoning via GPT-4o-mini."""
    prompt = DISTILLATION_PROMPT.format(observation=observation, action=action)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    return None


def build_distilled_record(original: dict, reasoning: str) -> dict:
    """Replace template reasoning with distilled reasoning."""
    messages = original["messages"]
    action_msg = next(m["content"] for m in messages if m["role"] == "assistant")

    lines = action_msg.strip().split("\n")
    action = lines[-1].strip()

    new_assistant = f"<think>{reasoning}</think>\n{action}"

    return {
        "messages": [
            messages[0],  # system
            messages[1],  # user (observation)
            {"role": "assistant", "content": new_assistant},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse distillation for SFT reasoning")
    parser.add_argument("--limit", type=int, default=5000, help="Number of examples to distill")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N records")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY or pass --api-key")
        return

    client = OpenAI(api_key=api_key)

    all_records = load_sft_records(INPUT_FILE, limit=args.offset + args.limit)
    records = all_records[args.offset:]
    distilled = []
    failed = 0
    start_time = time.time()

    for i, record in enumerate(records):
        observation, action = extract_observation_and_action(record)
        reasoning = distill_reasoning(client, observation, action)

        if reasoning:
            distilled.append(build_distilled_record(record, reasoning))
        else:
            failed += 1
            distilled.append(record)  # keep original if distillation fails

        if (i + 1) % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(records) - i - 1) / rate
            logger.info(
                f"Progress: {i + 1}/{len(records)} "
                f"({failed} failed, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

        time.sleep(RATE_LIMIT_DELAY)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for record in distilled:
            f.write(json.dumps(record) + "\n")

    elapsed = time.time() - start_time
    logger.info(
        f"Done: {len(distilled)} records ({failed} failed) "
        f"in {elapsed:.0f}s → {args.output}"
    )


if __name__ == "__main__":
    main()
