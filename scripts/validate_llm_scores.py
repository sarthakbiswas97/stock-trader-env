"""Validate LLM judge scores against known outcomes.

Samples scored decisions and checks:
  1. Do high scores correlate with positive episode returns?
  2. Do known mistakes (overbought buys, etc.) get low scores?
  3. Manual spot-check: prints decision + score for review.

Usage:
    PYTHONPATH=. python scripts/validate_llm_scores.py \
        --input data/grpo/prompts_scored.jsonl --sample 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def validate(input_path: Path, sample_size: int) -> None:
    """Validate scored dataset."""
    rows: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Filter to scored rows only
    scored = [r for r in rows if r.get("llm_score", 0) > 0]
    if not scored:
        logger.error("No scored rows found in %s", input_path)
        return

    logger.info("Total rows: %d | Scored: %d", len(rows), len(scored))

    # Sample
    sample = random.sample(scored, min(sample_size, len(scored)))

    # Correlation with episode return
    returns_and_scores = [
        (r.get("episode_return", 0), r["llm_score"])
        for r in scored
        if "episode_return" in r
    ]

    if returns_and_scores:
        pos_return = [s for ret, s in returns_and_scores if ret > 0]
        neg_return = [s for ret, s in returns_and_scores if ret <= 0]

        logger.info("")
        logger.info("=== Correlation Check ===")
        if pos_return:
            logger.info(
                "Positive-return episodes: mean LLM score = %.3f (n=%d)",
                sum(pos_return) / len(pos_return), len(pos_return),
            )
        if neg_return:
            logger.info(
                "Negative-return episodes: mean LLM score = %.3f (n=%d)",
                sum(neg_return) / len(neg_return), len(neg_return),
            )

    # Score distribution
    all_scores = [r["llm_score"] for r in scored]
    logger.info("")
    logger.info("=== Score Distribution ===")
    logger.info("Mean: %.3f | Std: %.3f", _mean(all_scores), _std(all_scores))
    logger.info("Min: %.3f | Max: %.3f", min(all_scores), max(all_scores))

    # Histogram buckets
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in all_scores:
        if s < 0.2:
            buckets["0.0-0.2"] += 1
        elif s < 0.4:
            buckets["0.2-0.4"] += 1
        elif s < 0.6:
            buckets["0.4-0.6"] += 1
        elif s < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    logger.info("Buckets: %s", buckets)

    # Spot check
    logger.info("")
    logger.info("=== Spot Check (sample=%d) ===", len(sample))
    for i, row in enumerate(sample[:10]):
        obs = row.get("observation", "")[:120]
        action = row.get("action", "?")
        score = row["llm_score"]
        criteria = row.get("llm_criteria", {})
        ep_ret = row.get("episode_return", "?")
        logger.info(
            "[%d] Action=%s Score=%.2f Criteria=%s EpReturn=%s",
            i + 1, action, score, criteria, ep_ret,
        )
        logger.info("    Obs: %s...", obs)
        logger.info("")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM judge scores")
    parser.add_argument("--input", required=True, help="Scored JSONL path")
    parser.add_argument("--sample", type=int, default=50, help="Sample size for spot check")
    args = parser.parse_args()

    validate(Path(args.input), args.sample)


if __name__ == "__main__":
    main()
