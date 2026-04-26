"""Observation builder — constructs text observations for the LLM agent.

Builds the market summary text that the agent reads each trading day,
including technical indicators, portfolio state, risk metrics, and
position details. Extracted from environment.py for single-responsibility.
"""

from __future__ import annotations

from models import MarketObservation, PositionInfo
from server.feature_engine import compute_all_features, features_to_text
from server.macro_data import macro_to_text
from server.portfolio import Portfolio
from server import __version__


def build_observation(
    sim,
    portfolio: Portfolio,
    config: dict,
    task_id: str,
    score: float,
    reward: float,
    done: bool,
) -> MarketObservation:
    """Build the full observation for the current environment state."""
    if sim is None or portfolio is None:
        return MarketObservation(
            done=True, reward=0.0, day=0, total_days=0,
            portfolio_value=0, cash=0, positions=[],
            market_summary="No episode active.", available_actions=["HOLD"],
            task_id="", score=0.0,
            env_version=__version__,
            task_version="",
        )

    prices = {}
    for sym in config["symbols"]:
        try:
            prices[sym] = sim.get_price(sym)
        except (IndexError, KeyError):
            prices[sym] = 0.0

    value = portfolio.get_value(prices)
    positions = portfolio.get_position_info(prices, current_day=sim.current_day)

    summary = _build_market_summary(sim, portfolio, config, prices, positions, value, done)
    available = _build_available_actions(sim, portfolio, config, prices, done)

    return MarketObservation(
        done=done,
        reward=round(reward, 4),
        day=sim.current_day + 1,
        total_days=config["episode_days"],
        portfolio_value=round(value, 2),
        cash=round(portfolio.cash, 2),
        positions=positions,
        market_summary=summary,
        available_actions=list(set(available)),
        task_id=task_id,
        score=round(score, 4),
        env_version=__version__,
        task_version=config["version"],
    )


def _build_market_summary(
    sim,
    portfolio: Portfolio,
    config: dict,
    prices: dict[str, float],
    positions: list[PositionInfo],
    value: float,
    done: bool,
) -> str:
    """Build the text summary the agent reads."""
    lines: list[str] = []

    # Day + portfolio snapshot
    lines.append(
        f"Day {sim.current_day + 1} of {config['episode_days']} | "
        f"Cash: Rs{portfolio.cash:,.0f} | Portfolio: Rs{value:,.0f} | "
        f"Return: {(value - config['initial_capital']) / config['initial_capital'] * 100:+.1f}%"
    )

    # Regime gate warning (hard task)
    if config["regime_gate"] and not done:
        breadth = sim.get_market_breadth()
        if breadth["market_down"] or breadth["breadth_weak"]:
            lines.append(
                f"\nWARNING: REGIME GATE ACTIVE — Market avg {breadth['avg_change']:+.1f}%, "
                f"{breadth['declining']}/{len(config['symbols'])} stocks declining. "
                f"Trading restricted — HOLD recommended."
            )

    lines.append("")

    # Macro context
    macro_snap = sim.get_macro_snapshot_data()
    if macro_snap:
        lines.append(macro_to_text(macro_snap))
        lines.append("")

    # Stock data with technical indicators
    for sym in config["symbols"]:
        try:
            lookback = sim.get_lookback_data(sym)
            features = compute_all_features(lookback)
            price = prices[sym]
            change = sim.get_daily_change(sym)
            lines.append(features_to_text(sym, price, change, features))
        except (IndexError, KeyError, ValueError):
            lines.append(f"{sym}: Data unavailable")

    # Position summary with age
    if positions:
        lines.append("\nYour Positions:")
        for p in positions:
            age = portfolio.get_position_age(p.symbol, sim.current_day)
            age_str = f" | Held: {age} days" if age > 0 else ""
            lines.append(
                f"  {p.symbol}: {p.quantity} shares @ Rs{p.avg_price:,.0f} | "
                f"Current: Rs{p.current_price:,.0f} | P&L: {p.pnl_percent:+.1f}%{age_str}"
            )

    # Portfolio risk status
    lines.append(_build_risk_summary(portfolio))

    # Constraints reminder (medium/hard)
    if config["transaction_cost"] > 0:
        lines.append(
            f"\nConstraints: {config['transaction_cost']*100:.1f}% cost + "
            f"{config['slippage']*100:.1f}% slippage per trade | "
            f"Max {config['position_limit_per_stock']*100:.0f}% per stock | "
            f"Trades today: {portfolio.trades_today}/{config['max_trades_per_day']}"
        )

    return "\n".join(lines)


def _build_risk_summary(portfolio: Portfolio) -> str:
    """Build the portfolio risk section of the observation."""
    dd_pct = portfolio.current_drawdown * 100
    capacity_pct = portfolio.trading_capacity * 100

    parts = [f"\nPortfolio Risk: Drawdown: {dd_pct:-.1f}% from peak | Trading capacity: {capacity_pct:.0f}%"]

    win_rate = portfolio.recent_win_rate
    streak = portfolio.losing_streak

    if win_rate is not None:
        n = len(portfolio._recent_outcomes)
        wins = sum(portfolio._recent_outcomes)
        losses = n - wins
        parts.append(f"  Recent trades: {wins}W/{losses}L ({win_rate*100:.0f}%)")
        if streak >= 3:
            parts.append(f"  WARNING: {streak}-trade losing streak — consider reducing exposure")
    else:
        parts.append("  No completed trades yet")

    return "\n".join(parts)


def _build_available_actions(
    sim,
    portfolio: Portfolio,
    config: dict,
    prices: dict[str, float],
    done: bool,
) -> list[str]:
    """Build list of available actions."""
    available = ["HOLD"]
    if not done:
        for sym in config["symbols"]:
            if portfolio.cash > prices.get(sym, float("inf")):
                if len(config["symbols"]) == 1:
                    available.append("BUY")
                else:
                    available.append(f"BUY {sym}")
            if sym in portfolio.positions:
                if len(config["symbols"]) == 1:
                    available.append("SELL")
                else:
                    available.append(f"SELL {sym}")
    return available
