"""Collect oracle-labeled SFT training data for the stock trading LLM agent.

Uses hindsight (future price data) to label optimal actions, combined with
current technical indicators + macro context to generate multi-strategy reasoning.

Features:
- Multi-timeframe oracle: 2/5/10-day lookahead with volatility-adjusted thresholds
- Candlestick patterns, gap analysis, range expansion from OHLCV
- Macro context: VIX, USD/INR, Brent Crude, sector rotation, RBI rate
- Time-aware reasoning: references episode day and position duration
- Balanced sampling: target ~40% HOLD / 35% BUY / 25% SELL

Usage:
    PYTHONPATH=. python scripts/collect_sft_data.py
    PYTHONPATH=. python scripts/collect_sft_data.py --target-total 40000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.llm_agent import SYSTEM_PROMPT
from server.feature_engine import compute_all_features, features_to_text
from server.macro_data import get_macro_snapshot, load_macro_data, macro_to_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "ohlcv"
MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sft"

STOCKS = sorted(
    p.stem.replace("_daily", "")
    for p in DATA_DIR.glob("*_daily.csv")
)

LOOKBACK = 50
TASK_EPISODE_DAYS = 20

# Multi-timeframe configurations: (lookahead_days, vol_multiplier)
# threshold = vol_multiplier * stock's 20-day daily stdev * sqrt(lookahead)
TIMEFRAMES = [
    (2, 0.5),   # Quick swing: small moves over 2 days
    (5, 0.8),   # Standard swing: medium moves over 5 days
    (10, 1.2),  # Position trade: larger moves over 10 days
]


# ---------------------------------------------------------------------------
# Volatility-adjusted oracle
# ---------------------------------------------------------------------------

def compute_daily_volatility(closes: pd.Series, idx: int, period: int = 20) -> float:
    """Compute annualized daily stdev of returns at a given index."""
    if idx < period:
        return 0.02  # Fallback: 2% daily vol
    window = closes.iloc[idx - period : idx]
    returns = window.pct_change().dropna()
    if len(returns) < 5:
        return 0.02
    return float(returns.std())


def oracle_label(
    closes: pd.Series,
    idx: int,
    lookahead: int,
    threshold: float,
) -> tuple[str, dict]:
    """Determine the optimal action using future price data.

    Returns (action, stats) where stats has lookahead metrics.
    """
    current_price = float(closes.iloc[idx])
    future = closes.iloc[idx + 1 : idx + 1 + lookahead]

    if len(future) == 0:
        return "HOLD", {}

    max_price = float(future.max())
    min_price = float(future.min())
    max_up = (max_price - current_price) / current_price
    max_down = (min_price - current_price) / current_price
    end_return = (float(future.iloc[-1]) - current_price) / current_price

    stats = {
        "max_up_pct": round(max_up * 100, 2),
        "max_down_pct": round(max_down * 100, 2),
        "end_return_pct": round(end_return * 100, 2),
        "lookahead": lookahead,
        "threshold_pct": round(threshold * 100, 2),
    }

    buy_signal = max_up > threshold
    sell_signal = max_down < -threshold

    if buy_signal and sell_signal:
        # Both triggered -- use whichever peak/trough comes first
        for i in range(len(future)):
            price_i = float(future.iloc[i])
            up_triggered = (price_i - current_price) / current_price > threshold
            down_triggered = (price_i - current_price) / current_price < -threshold
            if up_triggered:
                return "BUY", stats
            if down_triggered:
                return "SELL", stats
        return "HOLD", stats
    elif buy_signal:
        return "BUY", stats
    elif sell_signal:
        return "SELL", stats

    return "HOLD", stats


def passes_indicator_gate(features: dict, action: str) -> bool:
    """Check if current indicators support the oracle's action."""
    if action == "HOLD":
        return True

    if action == "BUY":
        return any([
            features["rsi"] < 50,
            features["macd"]["signal"] == "bullish",
            features["macd"]["crossover"],
            "lower" in features["bollinger"] or "below" in features["bollinger"],
            features["volume_spike"] > 1.3,
            features.get("candlestick", "") in ("hammer (bullish reversal)", "bullish_engulfing (reversal)"),
            features.get("gap", {}).get("type") == "up",
        ])

    if action == "SELL":
        return any([
            features["rsi"] > 50,
            features["macd"]["signal"] == "bearish",
            "upper" in features["bollinger"] or "above" in features["bollinger"],
            "down" in features["momentum"],
            features.get("candlestick", "") in ("shooting_star (bearish reversal)", "bearish_engulfing (reversal)"),
            features.get("gap", {}).get("type") == "down",
        ])

    return False


def has_partial_signal(features: dict) -> tuple[bool, str]:
    """Check if there's a partial trading signal (for rejected-signal HOLDs)."""
    buy_signals = [
        features["rsi"] < 40,
        features["macd"]["crossover"] and features["macd"]["signal"] == "bullish",
        "lower" in features.get("bollinger", ""),
        features.get("candlestick", "") in ("hammer (bullish reversal)", "bullish_engulfing (reversal)"),
    ]
    sell_signals = [
        features["rsi"] > 60,
        features["macd"]["crossover"] and features["macd"]["signal"] == "bearish",
        "upper" in features.get("bollinger", ""),
        features.get("candlestick", "") in ("shooting_star (bearish reversal)", "bearish_engulfing (reversal)"),
    ]

    if any(buy_signals):
        return True, "buy"
    if any(sell_signals):
        return True, "sell"
    return False, ""


# ---------------------------------------------------------------------------
# Time-aware reasoning generation
# ---------------------------------------------------------------------------

def generate_reasoning(
    features: dict,
    macro_snapshot: dict,
    action: str,
    symbol: str,
    day: int,
    hold_type: str = "no_signal",
    stats: dict | None = None,
) -> str:
    """Generate chain-of-thought reasoning citing indicators, macro, and time context."""
    parts = []

    rsi = features["rsi"]
    rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    macd = features["macd"]
    trend = features["trend"]
    bollinger = features["bollinger"]
    volume = features["volume_spike"]
    momentum = features["momentum"]
    candlestick = features.get("candlestick", "none")
    gap = features.get("gap", {"type": "none"})
    range_info = features.get("range", {"label": "normal"})

    # Time context
    if day >= 17:
        time_ctx = "Late in the episode, limited trading days remaining."
    elif day <= 3:
        time_ctx = "Early in the episode, time to build positions."
    else:
        time_ctx = ""

    if action == "BUY":
        if rsi < 40:
            parts.append(f"{symbol} RSI at {rsi:.0f} ({rsi_label}), approaching oversold.")
        if macd["crossover"] and macd["signal"] == "bullish":
            parts.append("MACD bullish crossover just triggered.")
        elif macd["signal"] == "bullish":
            parts.append("MACD positive, bullish momentum building.")
        if "lower" in bollinger:
            parts.append("Price at lower Bollinger band, potential bounce.")
        if volume > 1.5:
            parts.append(f"Volume {volume:.1f}x average, accumulation detected.")
        if candlestick in ("hammer (bullish reversal)", "bullish_engulfing (reversal)"):
            parts.append(f"Candlestick: {candlestick}.")
        if gap["type"] == "up":
            parts.append(f"Gapped up {gap['pct']:+.1f}%, bullish sentiment.")
        if range_info["label"] in ("expanded", "very_expanded"):
            parts.append(f"Range expanding ({range_info['ratio']:.1f}x avg), breakout potential.")
        if trend == "bullish":
            parts.append("Trend is bullish, supporting entry.")

        # Macro
        if macro_snapshot.get("vix_label") in ("low", "normal"):
            parts.append(f"VIX at {macro_snapshot.get('vix', 0):.0f} ({macro_snapshot['vix_label']}), favorable risk environment.")
        if macro_snapshot.get("usdinr_change", 0) < -0.2:
            parts.append("INR strengthening, positive for equities.")

        if time_ctx and day >= 17:
            parts.append("Late in episode but signal is strong, taking the trade.")
        elif time_ctx:
            parts.append(time_ctx)

        if not parts:
            parts.append(f"Technical setup supports entry. RSI {rsi:.0f}, trend {trend}.")
        parts.append("Entry setup identified.")

    elif action == "SELL":
        if rsi > 60:
            parts.append(f"{symbol} RSI at {rsi:.0f} ({rsi_label}), overbought territory.")
        if macd["signal"] == "bearish":
            parts.append("MACD bearish, momentum fading.")
        if macd["crossover"] and macd["signal"] == "bearish":
            parts.append("MACD bearish crossover signals reversal.")
        if "upper" in bollinger:
            parts.append("Price at upper Bollinger band, likely resistance.")
        if "down" in momentum:
            parts.append(f"Momentum negative ({momentum}).")
        if candlestick in ("shooting_star (bearish reversal)", "bearish_engulfing (reversal)"):
            parts.append(f"Candlestick: {candlestick}.")
        if gap["type"] == "down":
            parts.append(f"Gapped down {gap['pct']:+.1f}%, bearish sentiment.")
        if trend == "bearish":
            parts.append("Trend has turned bearish.")

        # Macro
        if macro_snapshot.get("vix_label") in ("elevated", "high"):
            parts.append(f"VIX elevated at {macro_snapshot.get('vix', 0):.0f}, risk increasing.")
        if macro_snapshot.get("usdinr_change", 0) > 0.3:
            parts.append("INR weakening, possible FII outflows.")
        if macro_snapshot.get("brent_change", 0) > 1.5:
            parts.append("Crude spiking, negative for Indian markets.")

        if day >= 17:
            parts.append("Late in episode, locking in gains before time runs out.")
        if not parts:
            parts.append(f"Exit signals present. RSI {rsi:.0f}, trend {trend}.")
        parts.append("Taking profit / cutting loss.")

    elif action == "HOLD":
        if hold_type == "rejected_signal":
            has_signal, signal_type = has_partial_signal(features)
            if signal_type == "buy":
                parts.append(f"RSI at {rsi:.0f} suggests potential entry,")
                if trend == "bearish":
                    parts.append("but trend is bearish — likely a value trap, not a bounce.")
                elif macro_snapshot.get("vix_label") in ("elevated", "high"):
                    parts.append(f"but VIX at {macro_snapshot.get('vix', 0):.0f} (elevated) signals high risk environment.")
                elif macro_snapshot.get("usdinr_change", 0) > 0.3:
                    parts.append("but INR weakening suggests foreign selling pressure.")
                elif candlestick in ("shooting_star (bearish reversal)", "bearish_engulfing (reversal)"):
                    parts.append(f"but bearish candlestick ({candlestick}) contradicts the signal.")
                else:
                    parts.append("but conflicting signals present. Waiting for confirmation.")
            elif signal_type == "sell":
                parts.append(f"RSI at {rsi:.0f} suggests overbought,")
                if trend == "bullish":
                    parts.append("but strong bullish trend supports holding. Not exiting yet.")
                elif "up" in momentum:
                    parts.append("but momentum still positive. Premature to exit.")
                else:
                    parts.append("but no clear catalyst for decline. Holding position.")
            else:
                parts.append(f"Mixed signals — RSI {rsi:.0f}, {macd['signal']} MACD. No clear edge.")
            if day <= 3:
                parts.append("Early in episode, patience is warranted.")
        else:
            parts.append(f"Indicators neutral — RSI {rsi:.0f} ({rsi_label}), MACD {macd['signal']}.")
            if "middle" in bollinger:
                parts.append("Price in middle of Bollinger bands.")
            if 0.7 <= volume <= 1.3:
                parts.append("Volume normal, no unusual activity.")
            if candlestick == "doji (indecision)":
                parts.append("Doji candle confirms market indecision.")
            if range_info["label"] == "compressed":
                parts.append("Range compressed, waiting for breakout direction.")
            parts.append("No actionable setup. Staying flat.")
            if day >= 17:
                parts.append("Late in episode, no need to force a trade.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Observation construction
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
    initial_capital = 100_000
    ret = (portfolio_value - initial_capital) / initial_capital * 100

    lines = [
        f"Day {day} of {TASK_EPISODE_DAYS} | "
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
# Main data generation
# ---------------------------------------------------------------------------

def generate_examples(seed: int = 42) -> dict[str, list[dict]]:
    """Generate oracle-labeled SFT examples from all stocks with multi-timeframe oracle."""
    rng = random.Random(seed)
    np.random.seed(seed)

    macro_data = load_macro_data(MACRO_DIR)
    if macro_data is None:
        logger.warning("No macro data at %s. Generating without macro context.", MACRO_DIR)

    all_examples: dict[str, list[dict]] = {
        "BUY": [], "SELL": [], "HOLD_no_signal": [], "HOLD_rejected": [],
    }

    max_lookahead = max(la for la, _ in TIMEFRAMES)

    for stock_idx, symbol in enumerate(STOCKS):
        csv_path = DATA_DIR / f"{symbol}_daily.csv"
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < LOOKBACK + max_lookahead + 10:
            logger.warning("Skipping %s: only %d rows", symbol, len(df))
            continue

        closes = df["close"]

        for idx in range(LOOKBACK, len(df) - max_lookahead):
            # Pick a random timeframe for this stock-day
            lookahead, vol_mult = rng.choice(TIMEFRAMES)

            # Volatility-adjusted threshold
            daily_vol = compute_daily_volatility(closes, idx)
            threshold = vol_mult * daily_vol * np.sqrt(lookahead)
            # Clamp threshold to reasonable range
            threshold = max(0.008, min(threshold, 0.08))  # 0.8% to 8%

            # Oracle label
            action, stats = oracle_label(closes, idx, lookahead, threshold)

            # Compute features
            lookback_df = df.iloc[idx - LOOKBACK : idx + 1].copy()
            features = compute_all_features(lookback_df)

            # Indicator gate
            if action in ("BUY", "SELL") and not passes_indicator_gate(features, action):
                action = "HOLD"

            # Macro snapshot
            macro_snapshot = {}
            if macro_data is not None:
                row_date = df.iloc[idx]["timestamp"]
                if isinstance(row_date, pd.Timestamp):
                    macro_snapshot = get_macro_snapshot(macro_data, row_date.date())

            # HOLD subcategory
            hold_type = "no_signal"
            if action == "HOLD":
                has_signal, _ = has_partial_signal(features)
                if has_signal:
                    hold_type = "rejected_signal"

            # Price and change
            price = float(closes.iloc[idx])
            daily_change = 0.0
            if idx > 0:
                daily_change = (price - float(closes.iloc[idx - 1])) / float(closes.iloc[idx - 1]) * 100

            # Simulate varied portfolio state
            day = rng.randint(1, TASK_EPISODE_DAYS)
            if action == "SELL" or (action == "HOLD" and rng.random() < 0.3):
                qty = rng.randint(10, 100)
                avg_price = price * (1 + rng.uniform(-0.05, 0.05))
                pnl = (price - avg_price) / avg_price * 100
                position_value = qty * price
                cash = 100_000 - qty * avg_price
                portfolio_value = cash + position_value
                position = {"qty": qty, "avg_price": avg_price, "pnl": pnl}
            else:
                cash = 100_000 * rng.uniform(0.6, 1.0)
                portfolio_value = cash
                position = None

            observation = build_observation(
                symbol=symbol,
                price=price,
                daily_change=daily_change,
                features=features,
                macro_snapshot=macro_snapshot,
                day=day,
                cash=cash,
                portfolio_value=portfolio_value,
                position=position,
            )

            reasoning = generate_reasoning(
                features=features,
                macro_snapshot=macro_snapshot,
                action=action,
                symbol=symbol,
                day=day,
                hold_type=hold_type,
                stats=stats,
            )

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here is today's market data:\n\n{observation}\n\nWhat is your trading action?"},
                    {"role": "assistant", "content": f"<think>{reasoning}</think>\n{action}"},
                ],
            }

            if action == "BUY":
                all_examples["BUY"].append(example)
            elif action == "SELL":
                all_examples["SELL"].append(example)
            elif hold_type == "rejected_signal":
                all_examples["HOLD_rejected"].append(example)
            else:
                all_examples["HOLD_no_signal"].append(example)

        if (stock_idx + 1) % 10 == 0:
            total = sum(len(v) for v in all_examples.values())
            logger.info(
                "Processed %d/%d stocks (%d examples so far)",
                stock_idx + 1, len(STOCKS), total,
            )

    return all_examples


def balance_and_save(
    all_examples: dict[str, list[dict]],
    target_total: int = 40_000,
    seed: int = 42,
    val_ratio: float = 0.1,
) -> None:
    """Balance classes to target distribution, split train/val, save."""
    rng = random.Random(seed)

    n_buy = len(all_examples["BUY"])
    n_sell = len(all_examples["SELL"])
    n_hold_no = len(all_examples["HOLD_no_signal"])
    n_hold_rej = len(all_examples["HOLD_rejected"])
    total_raw = n_buy + n_sell + n_hold_no + n_hold_rej

    logger.info("")
    logger.info("=== Raw Distribution ===")
    logger.info("BUY:                 %6d (%5.1f%%)", n_buy, n_buy / total_raw * 100)
    logger.info("SELL:                %6d (%5.1f%%)", n_sell, n_sell / total_raw * 100)
    logger.info("HOLD (no signal):    %6d (%5.1f%%)", n_hold_no, n_hold_no / total_raw * 100)
    logger.info("HOLD (rejected):     %6d (%5.1f%%)", n_hold_rej, n_hold_rej / total_raw * 100)
    logger.info("Total raw:           %6d", total_raw)

    # Target: 40% HOLD / 35% BUY / 25% SELL
    n_buy_target = int(target_total * 0.35)
    n_sell_target = int(target_total * 0.25)
    n_hold_target = target_total - n_buy_target - n_sell_target

    # Split HOLD: 50% no-signal + 50% rejected
    n_hold_no_target = n_hold_target // 2
    n_hold_rej_target = n_hold_target - n_hold_no_target

    # Sample each bucket
    buy_sampled = _sample(all_examples["BUY"], n_buy_target, rng)
    sell_sampled = _sample(all_examples["SELL"], n_sell_target, rng)
    hold_no_sampled = _sample(all_examples["HOLD_no_signal"], n_hold_no_target, rng)
    hold_rej_sampled = _sample(all_examples["HOLD_rejected"], n_hold_rej_target, rng)

    final = buy_sampled + sell_sampled + hold_no_sampled + hold_rej_sampled
    rng.shuffle(final)

    # Print final distribution
    action_counts = Counter()
    for ex in final:
        msg = ex["messages"][-1]["content"]
        action_word = msg.split("\n")[-1].strip().split()[0]
        action_counts[action_word] += 1

    total_final = len(final)
    logger.info("")
    logger.info("=== Final Distribution (target: %d) ===", target_total)
    for action in ["BUY", "SELL", "HOLD"]:
        count = action_counts.get(action, 0)
        pct = count / total_final * 100 if total_final > 0 else 0
        logger.info("%s:  %6d (%5.1f%%)", action, count, pct)
    logger.info("Total: %6d", total_final)

    # Stratified train/val split
    train, val = _stratified_split(final, val_ratio, rng)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "sft_oracle_v2_train.jsonl"
    val_path = OUTPUT_DIR / "sft_oracle_v2_val.jsonl"

    _save_jsonl(train, train_path)
    _save_jsonl(val, val_path)

    logger.info("")
    logger.info("Saved %d train examples to %s", len(train), train_path)
    logger.info("Saved %d val examples to %s", len(val), val_path)

    # Print sample example
    sample = final[0]
    logger.info("")
    logger.info("=== Sample Example ===")
    for msg in sample["messages"]:
        content = msg["content"][:300]
        logger.info("[%s] %s...", msg["role"].upper(), content)


def _sample(items: list, n: int, rng: random.Random) -> list:
    """Sample n items. If n > len(items), return all."""
    if n >= len(items):
        return items[:]
    return rng.sample(items, n)


def _stratified_split(
    examples: list[dict],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Split maintaining class ratios."""
    by_action: dict[str, list[dict]] = {}
    for ex in examples:
        msg = ex["messages"][-1]["content"]
        action = msg.split("\n")[-1].strip().split()[0]
        by_action.setdefault(action, []).append(ex)

    train, val = [], []
    for action_examples in by_action.values():
        rng.shuffle(action_examples)
        n_val = max(1, int(len(action_examples) * val_ratio))
        val.extend(action_examples[:n_val])
        train.extend(action_examples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _save_jsonl(examples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect oracle-labeled SFT data")
    parser.add_argument("--target-total", type=int, default=40_000, help="Target total examples after balancing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("Oracle SFT data generation (v2)")
    logger.info("  Stocks: %d", len(STOCKS))
    logger.info("  Timeframes: %s", [(la, f"{vm}x vol") for la, vm in TIMEFRAMES])
    logger.info("  Target total: %d", args.target_total)
    logger.info("  Seed: %d", args.seed)
    logger.info("")

    all_examples = generate_examples(seed=args.seed)
    balance_and_save(all_examples, target_total=args.target_total, seed=args.seed)


if __name__ == "__main__":
    main()
