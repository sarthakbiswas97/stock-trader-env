"""Task definitions and graders for the 3 difficulty levels."""

from __future__ import annotations

import math

import numpy as np

# Validator requires scores in open interval (0, 1) — never exactly 0.0 or 1.0
SCORE_MIN = 0.001
SCORE_MAX = 0.999


def _clamp_score(score: float) -> float:
    """Clamp score to the open interval (0, 1)."""
    return round(max(SCORE_MIN, min(SCORE_MAX, score)), 4)

TASK_CONFIGS = {
    "single_stock": {
        "version": "1.0.0",
        "name": "Single Stock Trading",
        "difficulty": "easy",
        "description": "Trade a single stock (RELIANCE) over 20 days. No transaction costs, no position limits.",
        "episode_days": 20,
        "initial_capital": 100_000,
        "symbols": ["RELIANCE"],
        "transaction_cost": 0.0,
        "slippage": 0.0,
        "max_position_pct": 1.0,       # Can go all-in
        "max_trades_per_day": 100,      # No limit effectively
        "regime_gate": False,
        "position_limit_per_stock": 1.0,
    },
    "portfolio": {
        "version": "1.0.0",
        "name": "Portfolio Management",
        "difficulty": "medium",
        "description": "Manage a 10-stock portfolio over 30 days. Transaction costs apply, max 30% in any single stock.",
        "episode_days": 30,
        "initial_capital": 200_000,
        "symbols": [
            "RELIANCE", "INFY", "TCS", "HDFCBANK", "SBIN",
            "ICICIBANK", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
        ],
        "transaction_cost": 0.001,      # 0.1%
        "slippage": 0.001,              # 0.1%
        "max_position_pct": 0.4,        # Max 40% total exposure
        "max_trades_per_day": 10,
        "regime_gate": False,
        "position_limit_per_stock": 0.3, # Max 30% in one stock
    },
    "full_autonomous": {
        "version": "1.0.0",
        "name": "Full Autonomous Trading",
        "difficulty": "hard",
        "description": "Trade 25 stocks over 40 days with regime gate, position limits, realistic slippage and costs. The agent must learn WHEN NOT to trade.",
        "episode_days": 40,
        "initial_capital": 500_000,
        "symbols": [
            "RELIANCE", "INFY", "TCS", "HDFCBANK", "SBIN",
            "ICICIBANK", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
            "AXISBANK", "BAJFINANCE", "SUNPHARMA", "TITAN", "HINDUNILVR",
            "HCLTECH", "WIPRO", "NTPC", "POWERGRID", "ADANIENT",
            "TATASTEEL", "JSWSTEEL", "COALINDIA", "ONGC", "MARUTI",
        ],
        "transaction_cost": 0.001,      # 0.1%
        "slippage": 0.002,              # 0.2%
        "max_position_pct": 0.8,        # Max 80% total exposure
        "max_trades_per_day": 5,
        "regime_gate": True,
        "position_limit_per_stock": 0.15, # Max 15% in one stock
    },
}


def grade_single_stock(
    final_value: float,
    initial_capital: float,
    buy_and_hold_return: float,
    total_trades: int,
) -> float:
    """Grade single stock: agent return vs buy-and-hold benchmark."""
    agent_return = (final_value - initial_capital) / initial_capital

    if agent_return <= -0.05:
        score = 0.1
    elif agent_return <= 0:
        score = 0.2 + (agent_return + 0.05) / 0.05 * 0.1
    elif buy_and_hold_return <= 0:
        score = min(0.8, 0.5 + agent_return * 3)
    else:
        ratio = agent_return / max(buy_and_hold_return, 0.001)
        if ratio >= 1.5:
            score = 0.8 + (ratio - 1.5) * 0.2
        elif ratio >= 1.0:
            score = 0.6 + (ratio - 1.0) * 0.4
        elif ratio >= 0.5:
            score = 0.4 + (ratio - 0.5) * 0.4
        else:
            score = 0.3 + ratio * 0.2

    return _clamp_score(score)


def grade_portfolio(
    final_value: float,
    initial_capital: float,
    daily_returns: list[float],
    risk_violations: int,
    total_trades: int,
) -> float:
    """Grade portfolio: 60% risk-adjusted return, 25% discipline, 15% activity."""
    agent_return = (final_value - initial_capital) / initial_capital

    if len(daily_returns) > 1 and any(r != 0 for r in daily_returns):
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        return_score = max(0.0, min(1.0, (sharpe + 1) / 4))
    else:
        return_score = 0.3 if agent_return >= 0 else 0.1

    discipline_score = max(0.0, 1.0 - risk_violations * 0.1)

    episode_days = 30
    trades_per_day = total_trades / max(episode_days, 1)
    if trades_per_day < 0.1:
        activity_score = 0.3  # Too passive
    elif trades_per_day > 5:
        activity_score = 0.4  # Overtrading
    else:
        activity_score = 0.8 + min(0.2, trades_per_day * 0.05)

    score = return_score * 0.60 + discipline_score * 0.25 + activity_score * 0.15
    return _clamp_score(score)


def grade_full_autonomous(
    final_value: float,
    initial_capital: float,
    daily_returns: list[float],
    risk_violations: int,
    regime_gated_days: int,
    regime_respected: int,
    total_trades: int,
    max_drawdown: float,
) -> float:
    """Grade full autonomous: 35% return, 25% risk-adjusted, 25% regime discipline, 15% risk mgmt."""
    agent_return = (final_value - initial_capital) / initial_capital

    return_score = max(0.0, min(1.0, (agent_return + 0.05) / 0.15))

    if len(daily_returns) > 1 and any(r != 0 for r in daily_returns):
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        risk_adj_score = max(0.0, min(1.0, (sharpe + 1) / 4))
    else:
        risk_adj_score = 0.2
    if regime_gated_days > 0:
        regime_score = regime_respected / regime_gated_days
    else:
        regime_score = 1.0  # No gated days = no opportunity to violate

    violation_penalty = max(0.0, 1.0 - risk_violations * 0.05)
    drawdown_penalty = max(0.0, 1.0 - abs(max_drawdown) / 0.15)
    risk_score = (violation_penalty + drawdown_penalty) / 2

    score = (
        return_score * 0.35
        + risk_adj_score * 0.25
        + regime_score * 0.25
        + risk_score * 0.15
    )
    return _clamp_score(score)


GRADERS = {
    "single_stock": grade_single_stock,
    "portfolio": grade_portfolio,
    "full_autonomous": grade_full_autonomous,
}
