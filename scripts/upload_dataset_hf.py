"""Upload SFT dataset to HuggingFace Hub.

Usage:
    HF_TOKEN=hf_... python scripts/upload_dataset_hf.py
    HF_TOKEN=hf_... python scripts/upload_dataset_hf.py --repo-id sarthakbiswas/stock-trader-sft-v2
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SFT_DIR = Path(__file__).parent.parent / "data" / "sft"
DEFAULT_REPO = "sarthakbiswas/stock-trader-sft-v2"


def load_jsonl_dataset(path: Path) -> Dataset:
    """Load a JSONL file as a HuggingFace Dataset."""
    import json
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload SFT dataset to HF Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="HF repo ID")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("Set HF_TOKEN environment variable with a write-access token")
        return

    train_path = SFT_DIR / "sft_oracle_v2_train.jsonl"
    val_path = SFT_DIR / "sft_oracle_v2_val.jsonl"

    if not train_path.exists() or not val_path.exists():
        logger.error("SFT data not found. Run collect_sft_data.py first.")
        return

    logger.info("Loading datasets...")
    train_ds = load_jsonl_dataset(train_path)
    val_ds = load_jsonl_dataset(val_path)

    logger.info("Train: %d examples", len(train_ds))
    logger.info("Val: %d examples", len(val_ds))

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })

    logger.info("Pushing to %s...", args.repo_id)
    dataset_dict.push_to_hub(
        args.repo_id,
        token=token,
        private=False,
    )

    logger.info("Done! Dataset available at: https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
