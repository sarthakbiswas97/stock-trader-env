"""Upload OHLCV + macro market data to HuggingFace Hub.

Usage:
    HF_TOKEN=hf_... python scripts/upload_market_data_hf.py
    HF_TOKEN=hf_... python scripts/upload_market_data_hf.py --repo-id sarthakbiswas/stock-trader-market-data
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OHLCV_DIR = Path(__file__).parent.parent / "data" / "ohlcv"
MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"
DEFAULT_REPO = "sarthakbiswas/stock-trader-market-data"


def load_all_csvs(directory: Path, data_type: str) -> Dataset:
    """Load all CSVs from a directory into a single Dataset with a 'symbol' column."""
    all_rows = []
    for csv_path in sorted(directory.glob("*_daily.csv")):
        name = csv_path.stem.replace("_daily", "")
        df = pd.read_csv(csv_path)
        df["symbol"] = name
        df["data_type"] = data_type
        all_rows.append(df)

    if not all_rows:
        logger.warning("No CSVs found in %s", directory)
        return Dataset.from_dict({})

    combined = pd.concat(all_rows, ignore_index=True)
    return Dataset.from_pandas(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload market data to HF Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("Set HF_TOKEN environment variable")
        return

    logger.info("Loading OHLCV data from %s", OHLCV_DIR)
    ohlcv_ds = load_all_csvs(OHLCV_DIR, "ohlcv")
    logger.info("OHLCV: %d rows", len(ohlcv_ds))

    logger.info("Loading macro data from %s", MACRO_DIR)
    macro_ds = load_all_csvs(MACRO_DIR, "macro")
    logger.info("Macro: %d rows", len(macro_ds))

    dataset_dict = DatasetDict({
        "ohlcv": ohlcv_ds,
        "macro": macro_ds,
    })

    logger.info("Pushing to %s...", args.repo_id)
    dataset_dict.push_to_hub(args.repo_id, token=token)
    logger.info("Done! https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
