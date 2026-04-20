"""Neural market simulator — generates market sequences using a trained world model.

Drop-in replacement for MarketSimulator. Uses real data as seed,
generates forward using the V-M-C world model. Agent actions do NOT
affect market prices (a single retail agent has zero market impact).

The generated sequence is compared against ground truth (real CSV data)
for validation and co-evolution (backpropagation to improve the model).
"""

from __future__ import annotations

import logging
import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from server.macro_data import get_macro_snapshot, load_macro_data
from server.market_simulator import (
    TASK_EPISODE_DAYS,
    TASK_INITIAL_CAPITAL,
    TASK_STOCKS,
    _load_stock_data,
)
from world_model.data import N_PRICE_FEATURES, features_to_ohlcv, ohlcv_to_features
from world_model.trainer import get_device, load_world_model

logger = logging.getLogger(__name__)

MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"
LOOKBACK = 50


class NeuralSimulator:
    """Generates market sequences using a trained world model.

    Same interface as MarketSimulator so environment.py can use either.
    """

    def __init__(
        self,
        task_id: str,
        seed: int | None = None,
        temperature: float = 0.3,
        checkpoint_path: Path | None = None,
    ):
        self.task_id = task_id
        self.symbols = TASK_STOCKS[task_id]
        self.episode_days = TASK_EPISODE_DAYS[task_id]
        self.initial_capital = TASK_INITIAL_CAPITAL[task_id]
        self.temperature = temperature
        self.rng = random.Random(seed)

        self._all_data: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            self._all_data[sym] = _load_stock_data(sym)

        self._macro_data = load_macro_data(MACRO_DIR)

        self._device = get_device()
        self._model, self._checkpoint = load_world_model(checkpoint_path, self._device)

        # Episode state (populated on reset)
        self._generated: dict[str, pd.DataFrame] = {}
        self._ground_truth: dict[str, pd.DataFrame] = {}
        self._start_idx: dict[str, int] = {}
        self._current_day: int = 0

    def reset(self) -> None:
        """Start a new episode: seed with real data, generate forward."""
        self._current_day = 0

        min_rows = min(len(df) for df in self._all_data.values())
        max_start = min_rows - LOOKBACK - self.episode_days - 1
        if max_start < 1:
            max_start = 1

        start = self.rng.randint(0, max_start)

        for sym in self.symbols:
            self._start_idx[sym] = start
            self._generate_episode(sym, start)

    def _generate_episode(self, symbol: str, start: int) -> None:
        """Generate episode days for one symbol using the world model."""
        df = self._all_data[symbol]
        seed_end = start + LOOKBACK
        seed_df = df.iloc[start:seed_end].copy()

        # Store ground truth for comparison
        truth_end = min(seed_end + self.episode_days, len(df))
        self._ground_truth[symbol] = df.iloc[seed_end:truth_end].copy()

        # Convert seed to features and generate forward
        seed_features = ohlcv_to_features(seed_df)
        seq_len = self._model.config.seq_len
        if len(seed_features) < seq_len:
            # Pad with zeros if seed is shorter than model expects
            pad = np.zeros((seq_len - len(seed_features), seed_features.shape[1]), dtype=np.float32)
            seed_features = np.concatenate([pad, seed_features], axis=0)

        input_seq = seed_features[-seq_len:]

        # Track prices for reconstruction
        last_close = float(seed_df.iloc[-1]["close"])
        last_volume = float(seed_df.iloc[-1]["volume"])
        last_timestamp = seed_df.iloc[-1]["timestamp"]

        generated_rows = []
        with torch.no_grad():
            for day in range(self.episode_days):
                tensor = torch.from_numpy(input_seq).unsqueeze(0).to(self._device)
                predicted, _ = self._model.predict_next(tensor, temperature=self.temperature)
                day_features = predicted.squeeze(0).cpu().numpy()

                # Clip to realistic daily move bounds (±10% max)
                day_features = np.clip(day_features, -0.10, 0.10)

                ohlcv = features_to_ohlcv(day_features, last_close, last_volume)
                ohlcv["timestamp"] = last_timestamp + pd.Timedelta(days=day + 1)
                generated_rows.append(ohlcv)

                last_close = ohlcv["close"]
                last_volume = ohlcv["volume"]

                # Slide the window: drop first row, append new features
                new_features = np.zeros((1, seed_features.shape[1]), dtype=np.float32)
                new_features[0, :N_PRICE_FEATURES] = day_features[:N_PRICE_FEATURES]
                input_seq = np.concatenate([input_seq[1:], new_features], axis=0)

        gen_df = pd.DataFrame(generated_rows)
        # Prepend the real lookback data for indicator computation
        self._generated[symbol] = pd.concat(
            [seed_df.reset_index(drop=True), gen_df.reset_index(drop=True)],
            ignore_index=True,
        )

    @property
    def current_day(self) -> int:
        return self._current_day

    @property
    def is_done(self) -> bool:
        return self._current_day >= self.episode_days

    def advance_day(self) -> None:
        self._current_day += 1

    def get_price(self, symbol: str) -> float:
        idx = LOOKBACK + self._current_day
        df = self._generated[symbol]
        if idx >= len(df):
            idx = len(df) - 1
        return float(df.iloc[idx]["close"])

    def get_daily_change(self, symbol: str) -> float:
        idx = LOOKBACK + self._current_day
        df = self._generated[symbol]
        if idx < 1 or idx >= len(df):
            return 0.0
        today = df.iloc[idx]["close"]
        yesterday = df.iloc[idx - 1]["close"]
        return (today - yesterday) / yesterday * 100

    def get_lookback_data(self, symbol: str, lookback: int = 50) -> pd.DataFrame:
        end_idx = LOOKBACK + self._current_day
        start_idx = max(0, end_idx - lookback)
        df = self._generated[symbol]
        if end_idx >= len(df):
            end_idx = len(df) - 1
        return df.iloc[start_idx:end_idx + 1].copy()

    def get_market_breadth(self) -> dict:
        up = 0
        down = 0
        total_change = 0.0
        for sym in self.symbols:
            change = self.get_daily_change(sym)
            total_change += change
            if change > 0:
                up += 1
            elif change < 0:
                down += 1

        n = len(self.symbols)
        avg_change = total_change / n if n > 0 else 0.0
        breadth_pct = down / n * 100 if n > 0 else 0.0

        return {
            "avg_change": round(avg_change, 2),
            "advancing": up,
            "declining": down,
            "breadth_weak": breadth_pct > 70,
            "market_down": avg_change < -0.5,
        }

    def get_5day_trend(self, symbol: str) -> float:
        idx = LOOKBACK + self._current_day
        df = self._generated[symbol]
        if idx < 5 or idx >= len(df):
            return 0.0
        today = df.iloc[idx]["close"]
        five_ago = df.iloc[idx - 5]["close"]
        return (today - five_ago) / five_ago * 100

    def get_current_date(self) -> date | None:
        if not self._start_idx:
            return None
        sym = self.symbols[0]
        idx = LOOKBACK + self._current_day
        df = self._generated[sym]
        if idx >= len(df):
            idx = len(df) - 1
        ts = df.iloc[idx]["timestamp"]
        if isinstance(ts, pd.Timestamp):
            return ts.date()
        return ts

    def get_macro_snapshot_data(self) -> dict:
        if self._macro_data is None:
            return {}
        current_date = self.get_current_date()
        if current_date is None:
            return {}
        return get_macro_snapshot(self._macro_data, current_date)

    def get_ground_truth(self, symbol: str) -> pd.DataFrame:
        """Return the real data for the episode window (for validation)."""
        return self._ground_truth.get(symbol, pd.DataFrame())

    def compute_prediction_error(self, symbol: str) -> dict[str, float]:
        """Compare generated sequence with ground truth for one symbol."""
        truth = self._ground_truth.get(symbol)
        if truth is None or truth.empty:
            return {}

        gen_df = self._generated[symbol]
        gen_episode = gen_df.iloc[LOOKBACK:LOOKBACK + len(truth)]

        if len(gen_episode) == 0:
            return {}

        n = min(len(gen_episode), len(truth))
        gen_close = gen_episode["close"].values[:n]
        true_close = truth["close"].values[:n]

        gen_returns = np.diff(gen_close) / gen_close[:-1]
        true_returns = np.diff(true_close) / true_close[:-1]

        if len(gen_returns) == 0:
            return {}

        return {
            "mae_returns": float(np.mean(np.abs(gen_returns - true_returns))),
            "direction_accuracy": float(np.mean(np.sign(gen_returns) == np.sign(true_returns))),
            "volatility_ratio": float(np.std(gen_returns) / max(np.std(true_returns), 1e-8)),
            "gen_total_return": float((gen_close[-1] - gen_close[0]) / gen_close[0]),
            "true_total_return": float((true_close[-1] - true_close[0]) / true_close[0]),
        }
