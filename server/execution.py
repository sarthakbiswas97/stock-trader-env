"""Trade execution — buy and sell order processing.

Handles position sizing, slippage, transaction costs, position limits,
and drawdown-based capacity scaling. Extracted from environment.py
for single-responsibility.
"""

from __future__ import annotations

from server.portfolio import Portfolio


def execute_buy(
    portfolio: Portfolio,
    symbol: str,
    fraction: float,
    prices: dict[str, float],
    config: dict,
    current_day: int,
) -> float:
    """Execute a buy order. Returns reward adjustment.

    Applies drawdown-based capacity scaling — the agent cannot deploy
    full capital when the portfolio is in drawdown. This forces risk
    management: size down when losing.
    """
    price = prices[symbol]

    # Check trade limit
    if portfolio.trades_today >= config["max_trades_per_day"]:
        portfolio.risk_violations += 1
        return -0.1

    # Check position limit per stock
    current_value = portfolio.get_value(prices)
    existing_value = 0
    if symbol in portfolio.positions:
        existing_value = portfolio.positions[symbol]["qty"] * price
    max_for_stock = current_value * config["position_limit_per_stock"]
    available_for_stock = max(0, max_for_stock - existing_value)

    # Drawdown-based capacity scaling
    capacity = portfolio.trading_capacity
    buy_amount = min(portfolio.cash * fraction * capacity, available_for_stock)
    if buy_amount < price:
        return 0.0  # Can't afford even 1 share

    # Apply slippage and costs
    effective_price = price * (1 + config["slippage"])
    cost = buy_amount * config["transaction_cost"]
    qty = int((buy_amount - cost) / effective_price)
    if qty <= 0:
        return 0.0

    total_cost = qty * effective_price + cost
    portfolio.cash -= total_cost

    if symbol in portfolio.positions:
        old = portfolio.positions[symbol]
        new_qty = old["qty"] + qty
        new_avg = (old["qty"] * old["avg_price"] + qty * effective_price) / new_qty
        portfolio.positions[symbol] = {
            "qty": new_qty,
            "avg_price": new_avg,
            "entry_day": old.get("entry_day", current_day),
        }
    else:
        portfolio.positions[symbol] = {
            "qty": qty,
            "avg_price": effective_price,
            "entry_day": current_day,
        }

    portfolio.trades_today += 1
    portfolio.total_trades += 1
    portfolio.trade_log.append({
        "day": current_day,
        "action": "BUY",
        "symbol": symbol,
        "qty": qty,
        "price": effective_price,
    })

    return 0.0  # Neutral — reward comes from P&L


def execute_sell(
    portfolio: Portfolio,
    symbol: str,
    fraction: float,
    prices: dict[str, float],
    config: dict,
    current_day: int,
) -> float:
    """Execute a sell order. Records trade outcome for streak tracking."""
    if symbol not in portfolio.positions:
        return -0.05  # Penalty for trying to sell what you don't own

    pos = portfolio.positions[symbol]
    price = prices[symbol]

    # Compute quantity to sell
    sell_qty = int(pos["qty"] * fraction)
    if sell_qty <= 0:
        return 0.0

    # Apply slippage and costs
    effective_price = price * (1 - config["slippage"])
    proceeds = sell_qty * effective_price
    cost = proceeds * config["transaction_cost"]
    net_proceeds = proceeds - cost

    portfolio.cash += net_proceeds

    # Calculate trade P&L and record outcome
    trade_pnl = (
        (effective_price - pos["avg_price"]) / pos["avg_price"]
        if pos["avg_price"] > 0 else 0.0
    )
    portfolio.record_trade_outcome(won=trade_pnl > 0)

    portfolio.trade_log.append({
        "day": current_day,
        "action": "SELL",
        "symbol": symbol,
        "qty": sell_qty,
        "price": effective_price,
        "pnl_pct": round(trade_pnl * 100, 2),
    })

    # Update or remove position
    remaining = pos["qty"] - sell_qty
    if remaining > 0:
        portfolio.positions[symbol] = {
            "qty": remaining,
            "avg_price": pos["avg_price"],
            "entry_day": pos.get("entry_day", current_day),
        }
    else:
        del portfolio.positions[symbol]

    portfolio.trades_today += 1
    portfolio.total_trades += 1

    return 0.0
