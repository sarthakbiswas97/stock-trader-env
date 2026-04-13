"""Train/val/test data splits based on date ranges.

Prevents temporal leakage by ensuring training, validation, and test data
come from non-overlapping time periods. Financial data is sequential — random
splits would leak future information into training.

Split boundaries:
    Train: Oct 2020 - Dec 2023 (~3.3 years)
    Val:   Jan 2024 - Jun 2024 (6 months)
    Test:  Jul 2024 - Mar 2026 (~1.8 years)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    """Immutable date range for a data split."""

    name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp


# IST timezone matches the OHLCV CSV timestamps
_tz = "Asia/Kolkata"

SPLITS: dict[str, DataSplit] = {
    "train": DataSplit(
        name="train",
        start_date=pd.Timestamp("2020-10-01", tz=_tz),
        end_date=pd.Timestamp("2023-12-31", tz=_tz),
    ),
    "val": DataSplit(
        name="val",
        start_date=pd.Timestamp("2024-01-01", tz=_tz),
        end_date=pd.Timestamp("2024-06-30", tz=_tz),
    ),
    "test": DataSplit(
        name="test",
        start_date=pd.Timestamp("2024-07-01", tz=_tz),
        end_date=pd.Timestamp("2026-03-31", tz=_tz),
    ),
}


def get_valid_index_range(
    timestamps: pd.Series,
    split: DataSplit,
    lookback: int,
    episode_days: int,
) -> tuple[int, int]:
    """Find the valid start index range for a given split.

    The episode window needs:
        [start_idx ... start_idx + lookback ... start_idx + lookback + episode_days]
    The actual trading dates (after lookback) must fall within the split's date range.

    Returns:
        (min_start, max_start) — inclusive range for the random start index.
        Raises ValueError if no valid range exists.
    """
    # Find first index where timestamp >= split.start_date
    # The trading window starts at start_idx + lookback, so we need
    # timestamps[start_idx + lookback] >= split.start_date
    min_start = None
    for i in range(len(timestamps) - lookback - episode_days):
        trading_start = timestamps.iloc[i + lookback]
        if trading_start >= split.start_date:
            min_start = i
            break

    if min_start is None:
        raise ValueError(
            f"No valid start index for split '{split.name}': "
            f"data doesn't reach {split.start_date}"
        )

    # Find last index where the episode end is still within split.end_date
    max_start = min_start
    for i in range(min_start, len(timestamps) - lookback - episode_days):
        trading_end = timestamps.iloc[i + lookback + episode_days - 1]
        if trading_end <= split.end_date:
            max_start = i
        else:
            break

    if max_start < min_start:
        raise ValueError(
            f"No valid range for split '{split.name}': "
            f"episode of {episode_days} days doesn't fit"
        )

    return (min_start, max_start)
