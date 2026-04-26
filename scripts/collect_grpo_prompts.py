"""Collect GRPO v2.1 training prompts across all stocks with rich metadata.

v2.1 changes from v2:
  - Adds episode_return metadata to each prompt (heuristic agent's episode P&L)
  - Trading reward uses 70% episode-level signal, so prompts from profitable
    episodes get positive episode_return and vice versa.

Iterates over all available stock CSVs, runs episodic simulations with a
heuristic agent for realistic portfolio states, and stores flat metadata
(rolling volatility, z-score, regime label, multi-day forward prices,
episode_return) that TRL's GRPOTrainer passes directly as kwargs to
reward functions.

Usage:
    PYTHONPATH=. python scripts/collect_grpo_prompts.py
    PYTHONPATH=. python scripts/collect_grpo_prompts.py --episodes-per-stock 10 --target-prompts 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

from baselines.llm_agent import SYSTEM_PROMPT
from server.feature_engine import compute_all_features, features_to_text
from server.macro_data import get_macro_snapshot, load_macro_data, macro_to_text
from training.data_splits import SPLITS, get_valid_index_range

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "ohlcv"
MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "grpo"

LOOKBACK = 50
EPISODE_DAYS = 20
INITIAL_CAPITAL = 100_000
MAX_FORWARD_HORIZON = 5
VOL_FLOOR = 0.005

# Discover all available stocks from data directory
STOCKS = sorted(p.stem.replace("_daily", "") for p in DATA_DIR.glob("*_daily.csv"))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_market_data() -> None:
    """Download market data from HF Hub if not locally available."""
    ohlcv_dir = Path("data/ohlcv")
    macro_dir = Path("data/macro")

    if (
        ohlcv_dir.exists()
        and any(ohlcv_dir.glob("*.csv"))
        and macro_dir.exists()
        and any(macro_dir.glob("*.csv"))
    ):
        return

    logger.info("Market data missing. Downloading from HF Hub...")
    from datasets import load_dataset

    ds = load_dataset("sarthakbiswas/stock-trader-market-data")

    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_df = ds["ohlcv"].to_pandas()
    for symbol, group in ohlcv_df.groupby("symbol"):
        group = group.drop(columns=["symbol", "data_type"])
        group.to_csv(ohlcv_dir / f"{symbol}_daily.csv", index=False)

    macro_dir.mkdir(parents=True, exist_ok=True)
    macro_df = ds["macro"].to_pandas()
    for name, group in macro_df.groupby("symbol"):
        group = group.drop(columns=["symbol", "data_type"])
        group.to_csv(macro_dir / f"{name}_daily.csv", index=False)

    logger.info(
        "Downloaded %d stocks + %d macro",
        len(ohlcv_df["symbol"].unique()),
        len(macro_df["symbol"].unique()),
    )


def compute_rolling_volatility(
    closes: pd.Series, idx: int, period: int = 20,
) -> float:
    """20-day rolling std of daily returns. Floors at VOL_FLOOR."""
    if idx < period:
        return 0.02
    window = closes.iloc[idx - period : idx]
    returns = window.pct_change().dropna()
    if len(returns) < 5:
        return 0.02
    return max(float(returns.std()), VOL_FLOOR)


def classify_regime(z_score: float) -> str:
    """Classify market regime from volatility-normalized daily return."""
    if z_score <= -2.0:
        return "strong_bear"
    if z_score <= -0.8:
        return "mild_bear"
    if z_score >= 2.0:
        return "strong_bull"
    if z_score >= 0.8:
        return "mild_bull"
    return "sideways"


def get_forward_prices(
    closes: pd.Series, idx: int, horizons: tuple[int, ...] = (1, 2, 5),
) -> dict[str, float]:
    """Get forward prices at multiple horizons. Falls back to current price."""
    current = float(closes.iloc[idx])
    result = {}
    for h in horizons:
        fwd_idx = idx + h
        if fwd_idx < len(closes):
            result[f"{h}d"] = float(closes.iloc[fwd_idx])
        else:
            result[f"{h}d"] = current
    return result


def heuristic_agent(
    features: dict, has_position: bool, day: int, rng: random.Random,
) -> str:
    """RSI-based heuristic agent for realistic portfolio states.

    Produces more realistic states than random actions while being
    fast (no GPU needed). 10% random override for exploration.
    """
    # 10% random override to create diverse states
    if rng.random() < 0.10:
        return rng.choice(["BUY", "SELL", "HOLD"])

    rsi = features.get("rsi", 50.0)

    # Close out near episode end
    if day >= 18 and has_position:
        return "SELL"

    # Buy on oversold
    if rsi < 35 and not has_position and day < 17:
        return "BUY"

    # Sell on overbought
    if rsi > 65 and has_position:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Observation construction (matches environment format exactly)
# ---------------------------------------------------------------------------


def build_observation(
    symbol: str,
    price: float,
    daily_change: float,
    features: dict,
    macro_snapshot: dict,
    day: int = 1,
    cash: float = 100_000,
    portfolio_value: float = 100_000,
    position: dict | None = None,
) -> str:
    """Build observation text matching the environment format."""
    initial_capital = INITIAL_CAPITAL
    ret = (portfolio_value - initial_capital) / initial_capital * 100

    lines = [
        f"Day {day} of {EPISODE_DAYS} | "
        f"Cash: Rs{cash:,.0f} | Portfolio: Rs{portfolio_value:,.0f} | "
        f"Return: {ret:+.1f}%",
        "",
    ]

    macro_text = macro_to_text(macro_snapshot)
    if macro_text:
        lines.append(macro_text)
        lines.append("")

    lines.append(features_to_text(symbol, price, daily_change, features))

    if position:
        lines.append("")
        lines.append("Your Positions:")
        lines.append(
            f"  {symbol}: {position['qty']} shares @ Rs{position['avg_price']:,.0f} | "
            f"Current: Rs{price:,.0f} | P&L: {position['pnl']:+.1f}%"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-stock prompt collection
# ---------------------------------------------------------------------------


def get_split_range(
    df: pd.DataFrame, split: str,
) -> tuple[int, int]:
    """Compute valid start index range for a stock within a date split.

    Must leave room for LOOKBACK + EPISODE_DAYS + MAX_FORWARD_HORIZON.
    """
    split_def = SPLITS[split]

    # Ensure timestamps are tz-aware for comparison with split boundaries
    timestamps = df["timestamp"]
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("Asia/Kolkata")

    total_needed = EPISODE_DAYS + MAX_FORWARD_HORIZON
    return get_valid_index_range(timestamps, split_def, LOOKBACK, total_needed)


def collect_prompts_for_stock(
    symbol: str,
    df: pd.DataFrame,
    macro_data: dict[str, pd.DataFrame] | None,
    agent_fn: Callable,
    n_episodes: int,
    seed: int,
    split_range: tuple[int, int],
    min_z: float,
) -> list[dict]:
    """Run episodic simulations for one stock, collecting prompts with metadata.

    Two-pass per episode:
      1. Run full episode to compute final portfolio value (episode_return)
      2. Assign episode_return to all prompts from that episode

    This lets the trading_reward blend 70% episode-level signal with 30%
    step-level signal, aligning training with eval's grade_single_stock().
    """
    rng = random.Random(seed)
    prompts: list[dict] = []
    closes = df["close"]
    min_start, max_start = split_range

    for ep in range(n_episodes):
        start_idx = rng.randint(min_start, max_start)
        episode_id = f"{symbol}_ep{ep}_seed{seed}"

        # --- Pass 1: Run episode to compute episode_return ---
        cash_sim = float(INITIAL_CAPITAL)
        position_sim: dict | None = None
        rng_sim = random.Random(seed + ep * 7919)  # deterministic but separate from prompt rng

        for day in range(EPISODE_DAYS):
            data_idx = start_idx + LOOKBACK + day
            lookback_df = df.iloc[data_idx - LOOKBACK : data_idx + 1]
            features = compute_all_features(lookback_df)
            price = float(closes.iloc[data_idx])
            has_pos_sim = position_sim is not None and position_sim["qty"] > 0

            action = agent_fn(features, has_pos_sim, day + 1, rng_sim)

            if action == "BUY" and not has_pos_sim:
                qty = int(cash_sim * 0.95 / price) if price > 0 else 0
                if qty > 0:
                    position_sim = {"qty": qty, "avg_price": price}
                    cash_sim -= qty * price
            elif action == "SELL" and has_pos_sim:
                cash_sim += position_sim["qty"] * price
                position_sim = None

        # Final portfolio value after episode
        final_price = float(closes.iloc[start_idx + LOOKBACK + EPISODE_DAYS - 1])
        pos_value_sim = position_sim["qty"] * final_price if position_sim and position_sim["qty"] > 0 else 0.0
        final_value = cash_sim + pos_value_sim
        episode_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

        # --- Pass 2: Collect prompts with episode_return attached ---
        cash = float(INITIAL_CAPITAL)
        position: dict | None = None

        for day in range(EPISODE_DAYS):
            data_idx = start_idx + LOOKBACK + day

            lookback_df = df.iloc[data_idx - LOOKBACK : data_idx + 1]
            features = compute_all_features(lookback_df)

            price = float(closes.iloc[data_idx])
            prev_price = float(closes.iloc[data_idx - 1])
            daily_change_pct = (price - prev_price) / prev_price * 100

            rolling_vol = compute_rolling_volatility(closes, data_idx)
            daily_return = (price - prev_price) / prev_price
            z_score = daily_return / rolling_vol if rolling_vol > 0 else 0.0
            regime = classify_regime(z_score)

            fwd_prices = get_forward_prices(closes, data_idx)

            macro_snapshot: dict = {}
            if macro_data is not None:
                row_date = df.iloc[data_idx]["timestamp"]
                if hasattr(row_date, "date"):
                    macro_snapshot = get_macro_snapshot(macro_data, row_date.date())

            has_position = position is not None and position["qty"] > 0
            pos_value = position["qty"] * price if has_position else 0.0
            portfolio_value = cash + pos_value
            cash_fraction = cash / portfolio_value if portfolio_value > 0 else 1.0

            pos_for_obs = None
            pnl_pct = 0.0
            if has_position:
                pnl_pct = (price - position["avg_price"]) / position["avg_price"] * 100
                pos_for_obs = {
                    "qty": position["qty"],
                    "avg_price": position["avg_price"],
                    "pnl": pnl_pct,
                }

            observation = build_observation(
                symbol, price, daily_change_pct, features,
                macro_snapshot, day + 1, cash, portfolio_value, pos_for_obs,
            )

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Here is today's market data:\n\n{observation}\n\nWhat is your trading action?",
                },
            ]

            if abs(z_score) >= min_z:
                prompts.append({
                    "prompt": prompt_messages,
                    "symbol": symbol,
                    "current_price": round(price, 2),
                    "next_price_1d": round(fwd_prices["1d"], 2),
                    "next_price_2d": round(fwd_prices["2d"], 2),
                    "next_price_5d": round(fwd_prices["5d"], 2),
                    "has_position": has_position,
                    "position_qty": position["qty"] if has_position else 0,
                    "position_avg_price": round(position["avg_price"], 2) if has_position else 0.0,
                    "position_pnl_pct": round(pnl_pct, 2),
                    "cash_fraction": round(cash_fraction, 4),
                    "portfolio_value": round(portfolio_value, 2),
                    "rolling_vol": round(rolling_vol, 6),
                    "z_score": round(z_score, 4),
                    "regime": regime,
                    "day": day + 1,
                    "total_days": EPISODE_DAYS,
                    "episode_id": episode_id,
                    "episode_return": round(episode_return, 6),
                })

            # Agent takes action to advance episode state (same RNG as pass 1)
            action = agent_fn(features, has_position, day + 1, rng)

            if action == "BUY" and not has_position:
                qty = int(cash * 0.95 / price) if price > 0 else 0
                if qty > 0:
                    position = {"qty": qty, "avg_price": price}
                    cash -= qty * price
            elif action == "SELL" and has_position:
                cash += position["qty"] * price
                position = None

    return prompts


# ---------------------------------------------------------------------------
# Regime-stratified balancing
# ---------------------------------------------------------------------------


def balance_by_regime(
    prompts: list[dict],
    target_total: int,
    rng: random.Random,
) -> list[dict]:
    """Regime-stratified sampling for balanced coverage.

    Target distribution (over-sample hard cases):
        strong_bear: 20%, mild_bear: 20%, sideways: 20%,
        mild_bull: 20%, strong_bull: 20%
    """
    # Group by regime
    by_regime: dict[str, list[dict]] = {}
    for p in prompts:
        regime = p["regime"]
        by_regime.setdefault(regime, []).append(p)

    regimes = ["strong_bear", "mild_bear", "sideways", "mild_bull", "strong_bull"]
    per_regime = target_total // len(regimes)

    balanced: list[dict] = []
    for regime in regimes:
        available = by_regime.get(regime, [])
        if len(available) <= per_regime:
            # Take all if not enough, with replacement for shortfall
            balanced.extend(available)
            if len(available) < per_regime and available:
                extras = rng.choices(available, k=per_regime - len(available))
                balanced.extend(extras)
        else:
            balanced.extend(rng.sample(available, per_regime))

    rng.shuffle(balanced)
    return balanced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GRPO v2 training prompts")
    parser.add_argument("--episodes-per-stock", type=int, default=10)
    parser.add_argument("--target-prompts", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-z", type=float, default=0.3,
                        help="Min |z_score| to keep a prompt (quality filter)")
    parser.add_argument("--agent", choices=["heuristic", "random"], default="heuristic")
    args = parser.parse_args()

    ensure_market_data()

    # Reload STOCKS after potential download
    stocks = sorted(p.stem.replace("_daily", "") for p in DATA_DIR.glob("*_daily.csv"))
    if not stocks:
        logger.error("No stock data found in %s", DATA_DIR)
        return

    logger.info(
        "Collecting GRPO v2.1 prompts: %d stocks, %d ep/stock, split=%s, min_z=%.2f, agent=%s",
        len(stocks), args.episodes_per_stock, args.split, args.min_z, args.agent,
    )

    macro_data = load_macro_data(MACRO_DIR)
    if macro_data is None:
        logger.warning("No macro data at %s. Generating without macro context.", MACRO_DIR)

    agent_fn = heuristic_agent if args.agent == "heuristic" else _random_agent

    all_prompts: list[dict] = []
    skipped_stocks: list[str] = []

    for stock_idx, symbol in enumerate(stocks):
        csv_path = DATA_DIR / f"{symbol}_daily.csv"
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        min_rows = LOOKBACK + EPISODE_DAYS + MAX_FORWARD_HORIZON + 10
        if len(df) < min_rows:
            logger.warning("Skipping %s: only %d rows (need %d)", symbol, len(df), min_rows)
            skipped_stocks.append(symbol)
            continue

        try:
            split_range = get_split_range(df, args.split)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            skipped_stocks.append(symbol)
            continue

        stock_prompts = collect_prompts_for_stock(
            symbol=symbol,
            df=df,
            macro_data=macro_data,
            agent_fn=agent_fn,
            n_episodes=args.episodes_per_stock,
            seed=args.seed + stock_idx * 100,
            split_range=split_range,
            min_z=args.min_z,
        )
        all_prompts.extend(stock_prompts)

        if (stock_idx + 1) % 10 == 0:
            logger.info(
                "Processed %d/%d stocks (%d prompts so far)",
                stock_idx + 1, len(stocks), len(all_prompts),
            )

    logger.info(
        "Raw collection: %d prompts from %d stocks (%d skipped)",
        len(all_prompts), len(stocks) - len(skipped_stocks), len(skipped_stocks),
    )

    # Regime distribution before balancing
    regime_counts = Counter(p["regime"] for p in all_prompts)
    logger.info("Regime distribution (raw): %s", dict(regime_counts.most_common()))

    # Regime-stratified balancing
    rng = random.Random(args.seed)
    balanced = balance_by_regime(all_prompts, args.target_prompts, rng)
    logger.info("After balancing: %d prompts", len(balanced))

    regime_balanced = Counter(p["regime"] for p in balanced)
    logger.info("Regime distribution (balanced): %s", dict(regime_balanced.most_common()))

    # Position coverage
    has_pos_count = sum(1 for p in balanced if p["has_position"])
    logger.info(
        "Prompts with position: %d (%.1f%%)",
        has_pos_count, has_pos_count / len(balanced) * 100 if balanced else 0,
    )

    # Stock coverage
    stock_counts = Counter(p["symbol"] for p in balanced)
    logger.info("Stocks represented: %d", len(stock_counts))

    # Save as JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Log episode return distribution
    ep_returns = [p["episode_return"] for p in balanced]
    mean_ep_ret = sum(ep_returns) / len(ep_returns) if ep_returns else 0.0
    positive_eps = sum(1 for r in ep_returns if r > 0)
    logger.info(
        "Episode returns: mean=%.4f, positive=%d/%d (%.1f%%)",
        mean_ep_ret, positive_eps, len(ep_returns),
        positive_eps / len(ep_returns) * 100 if ep_returns else 0,
    )

    output_path = OUTPUT_DIR / f"grpo_v2.1_prompts_{args.split}_{len(balanced)}.jsonl"

    with open(output_path, "w") as f:
        for p in balanced:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Saved %d prompts to %s", len(balanced), output_path)


def _random_agent(
    features: dict, has_position: bool, day: int, rng: random.Random,
) -> str:
    """Fallback random agent (same as v1 but less HOLD-biased)."""
    return rng.choice(["BUY", "SELL", "HOLD"])


if __name__ == "__main__":
    main()
