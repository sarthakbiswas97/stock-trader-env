"""Observation converters — transform MarketObservation into training-ready formats.

Two modes:
    - "text": Returns the market_summary string for LLM agents
    - "numeric": Returns a fixed-size numpy array for PPO/DQN agents
"""

from __future__ import annotations

import numpy as np

from models import MarketObservation


# Maximum stocks across all tasks (full_autonomous has 25)
MAX_STOCKS = 25

# Per-stock features in numeric mode
STOCK_FEATURES = 4  # price_change_norm, rsi_norm, volume_spike_norm, is_held

# Global features: day_progress, cash_ratio, value_ratio, num_positions_ratio
GLOBAL_FEATURES = 4

# Total numeric observation size
NUMERIC_OBS_SIZE = GLOBAL_FEATURES + MAX_STOCKS * STOCK_FEATURES


def obs_to_text(obs: MarketObservation) -> str:
    """Extract the text observation for LLM agents."""
    return obs.market_summary


def obs_to_numeric(
    obs: MarketObservation,
    initial_capital: float,
) -> np.ndarray:
    """Convert observation to a fixed-size numeric vector.

    Layout:
        [0]     day / total_days (progress through episode)
        [1]     cash / initial_capital
        [2]     portfolio_value / initial_capital
        [3]     num_positions / MAX_STOCKS
        [4..N]  per-stock features (padded to MAX_STOCKS)

    Per-stock features (4 per stock):
        [0] price P&L percent / 100 (normalized, clipped to [-1, 1])
        [1] 0.0 (placeholder — RSI not directly in observation)
        [2] 0.0 (placeholder — volume spike not directly in observation)
        [3] 1.0 if position held, 0.0 otherwise

    Note: RSI and volume spike are embedded in market_summary text but not
    as structured fields. For numeric mode, we use what's available in the
    structured observation. The text mode is strictly richer.
    """
    vec = np.zeros(NUMERIC_OBS_SIZE, dtype=np.float32)

    # Global features
    total = obs.total_days if obs.total_days > 0 else 1
    capital = initial_capital if initial_capital > 0 else 1
    vec[0] = obs.day / total
    vec[1] = obs.cash / capital
    vec[2] = obs.portfolio_value / capital
    vec[3] = len(obs.positions) / MAX_STOCKS

    # Per-stock features from positions
    for i, pos in enumerate(obs.positions):
        if i >= MAX_STOCKS:
            break
        base = GLOBAL_FEATURES + i * STOCK_FEATURES
        vec[base] = np.clip(pos.pnl_percent / 100, -1, 1)
        # Slots [1] and [2] stay 0 — see docstring
        vec[base + 3] = 1.0

    return vec
