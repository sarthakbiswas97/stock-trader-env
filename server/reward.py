"""Reward computation for the trading environment.

Decomposes reward into independent signals:
  1. HOLD quality — is inaction justified by market signals?
  2. Position age cost — stale positions incur holding cost
  3. Streak penalty — penalize buying on a losing streak

Extracted from environment.py for single-responsibility.
"""

from __future__ import annotations

from server.portfolio import Portfolio, LOSING_STREAK_THRESHOLD


def evaluate_hold(
    portfolio: Portfolio,
    prices: dict[str, float],
    task_config: dict,
    get_rsi_fn,
) -> float:
    """Evaluate HOLD quality based on market signals and portfolio state.

    Returns a small reward/penalty:
      - Missed opportunity (strong signal ignored): -0.15
      - Holding a losing position (> -5% P&L): -0.10
      - Justified patience (no clear signal): +0.01
    """
    has_positions = bool(portfolio.positions)
    worst_pnl = get_worst_position_pnl(portfolio, prices)

    # Check RSI across all task symbols for actionable signals
    extreme_rsi = False
    for sym in task_config["symbols"]:
        rsi = get_rsi_fn(sym)
        if rsi is not None and (rsi < 25 or rsi > 75):
            extreme_rsi = True
            break

    # Holding a losing position (> 5% drawdown) — should cut losses
    if has_positions and worst_pnl is not None and worst_pnl < -5.0:
        return -0.1

    # Strong signal ignored — missed opportunity
    if extreme_rsi:
        return -0.15

    # No strong signal — justified patience
    return 0.01


def compute_holding_cost(portfolio: Portfolio, current_day: int) -> float:
    """Compute total holding cost for aged positions.

    Positions held longer than 5 days incur a daily cost,
    forcing the agent to learn exit timing.
    """
    return portfolio.compute_holding_cost(current_day)


def compute_streak_penalty(portfolio: Portfolio, action_type: str) -> float:
    """Penalize buying during a losing streak.

    When the agent's last 3+ trades were losses and it tries to BUY,
    apply a penalty. Forces the agent to learn when to STOP trading —
    like a professional trader's kill switch.
    """
    if action_type != "BUY":
        return 0.0

    if portfolio.losing_streak >= LOSING_STREAK_THRESHOLD:
        return -0.1

    return 0.0


def get_worst_position_pnl(
    portfolio: Portfolio,
    prices: dict[str, float],
) -> float | None:
    """Get the worst P&L percentage across all positions."""
    if not portfolio.positions:
        return None
    worst = 0.0
    for sym, pos in portfolio.positions.items():
        price = prices.get(sym, pos["avg_price"])
        pnl = (
            (price - pos["avg_price"]) / pos["avg_price"] * 100
            if pos["avg_price"] > 0 else 0.0
        )
        worst = min(worst, pnl)
    return worst
