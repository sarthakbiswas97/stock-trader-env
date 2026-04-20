"""Data pipeline: OHLCV CSVs → returns-space training sequences for the world model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

OHLCV_DIR = Path(__file__).parent.parent / "data" / "ohlcv"

N_PRICE_FEATURES = 5  # open_ret, high_ret, low_ret, close_ret, log_vol_change
N_DERIVED_FEATURES = 2  # intraday_range, body_ratio
N_FEATURES = N_PRICE_FEATURES + N_DERIVED_FEATURES
INPUT_DIM = N_FEATURES


@dataclass(frozen=True)
class SequenceStats:
    """Feature statistics for normalization."""

    feature_means: np.ndarray
    feature_stds: np.ndarray
    n_sequences: int


def load_all_ohlcv(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all OHLCV CSVs from the data directory."""
    data_dir = data_dir or OHLCV_DIR
    result = {}
    for path in sorted(data_dir.glob("*_daily.csv")):
        symbol = path.stem.replace("_daily", "")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) < 100:
            logger.warning(f"Skipping {symbol}: only {len(df)} rows")
            continue
        result[symbol] = df
    logger.info(f"Loaded {len(result)} symbols from {data_dir}")
    return result


def ohlcv_to_features(df: pd.DataFrame) -> np.ndarray:
    """OHLCV DataFrame → (n_days-1, N_FEATURES) array in returns space."""
    close = df["close"].values
    opn = df["open"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values.astype(float)

    # Returns relative to previous close
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    open_ret = (opn[1:] - prev_close[:-1]) / prev_close[:-1]
    high_ret = (high[1:] - prev_close[:-1]) / prev_close[:-1]
    low_ret = (low[1:] - prev_close[:-1]) / prev_close[:-1]
    close_ret = (close[1:] - close[:-1]) / close[:-1]

    # Log volume change (avoids scale issues)
    vol_safe = np.maximum(volume, 1.0)
    log_vol = np.log(vol_safe)
    log_vol_change = log_vol[1:] - log_vol[:-1]

    # Derived features
    intraday_range = (high[1:] - low[1:]) / prev_close[:-1]
    body = np.abs(close[1:] - opn[1:])
    candle_range = high[1:] - low[1:]
    safe_range = np.where(candle_range > 0, candle_range, 1.0)
    body_ratio = np.where(candle_range > 0, body / safe_range, 0.5)

    features = np.column_stack([
        open_ret, high_ret, low_ret, close_ret, log_vol_change,
        intraday_range, body_ratio,
    ])

    # Clip extreme values
    features = np.clip(features, -0.5, 0.5)

    return features.astype(np.float32)


def features_to_ohlcv(
    features: np.ndarray,
    prev_close: float,
    prev_volume: float,
) -> dict[str, float]:
    """Reconstruct OHLCV dict from returns-space features + previous day's values."""
    open_ret, high_ret, low_ret, close_ret, log_vol_change = features[:5]

    opn = prev_close * (1 + open_ret)
    high = prev_close * (1 + high_ret)
    low = prev_close * (1 + low_ret)
    close = prev_close * (1 + close_ret)

    # Ensure OHLC consistency
    high = max(high, opn, close)
    low = min(low, opn, close)
    if low <= 0:
        low = prev_close * 0.9

    volume = prev_volume * np.exp(log_vol_change)
    volume = max(volume, 100.0)

    return {
        "open": round(float(opn), 2),
        "high": round(float(high), 2),
        "low": round(float(low), 2),
        "close": round(float(close), 2),
        "volume": int(volume),
    }


class MarketSequenceDataset(Dataset):
    """(sequence, target) pairs for world model training."""

    def __init__(
        self,
        symbols_data: dict[str, pd.DataFrame],
        seq_len: int = 50,
        stride: int = 1,
    ):
        self.seq_len = seq_len
        self.sequences: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

        for symbol, df in symbols_data.items():
            features = ohlcv_to_features(df)
            if len(features) < seq_len + 1:
                continue

            for i in range(0, len(features) - seq_len, stride):
                self.sequences.append(features[i : i + seq_len])
                self.targets.append(features[i + seq_len, :N_PRICE_FEATURES])

        logger.info(
            f"Created {len(self.sequences)} sequences "
            f"from {len(symbols_data)} symbols (seq_len={seq_len})"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.from_numpy(self.targets[idx]),
        )

    def compute_stats(self) -> SequenceStats:
        """Compute feature statistics for normalization."""
        all_features = np.concatenate(self.sequences, axis=0)
        return SequenceStats(
            feature_means=all_features.mean(axis=0),
            feature_stds=np.maximum(all_features.std(axis=0), 1e-8),
            n_sequences=len(self.sequences),
        )
