"""Portfolio tracking — cash, positions, risk metrics, and trade history.

Extracted from environment.py for single-responsibility and scalability.
Tracks position age, drawdown-based trading capacity, and recent trade
outcomes for streak-aware risk management.
"""

from __future__ import annotations

from collections import deque

from models import PositionInfo

# Drawdown thresholds for trading capacity scaling
DRAWDOWN_CAPACITY = [
    (0.03, 1.00),   # drawdown < 3%: full capacity
    (0.05, 0.75),   # drawdown 3-5%: 75% capacity
    (0.08, 0.50),   # drawdown 5-8%: 50% capacity
    (1.00, 0.25),   # drawdown > 8%: survival mode
]

POSITION_AGE_THRESHOLD = 5
POSITION_AGE_COST_PER_DAY = 0.02

STREAK_WINDOW = 5
LOSING_STREAK_THRESHOLD = 3


class Portfolio:
    """Tracks cash, positions, risk metrics, and trade history."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # Positions: {symbol: {qty, avg_price, entry_day}}
        self.positions: dict[str, dict] = {}

        self.trade_log: list[dict] = []
        self.daily_values: list[float] = [initial_capital]
        self.daily_returns: list[float] = []

        # Risk tracking
        self.risk_violations: int = 0
        self.regime_gated_days: int = 0
        self.regime_respected: int = 0
        self.trades_today: int = 0
        self.total_trades: int = 0
        self.peak_value: float = initial_capital
        self.max_drawdown: float = 0.0

        # Trade outcome tracking (for streak awareness)
        self._recent_outcomes: deque[bool] = deque(maxlen=STREAK_WINDOW)

    def get_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value (cash + positions)."""
        position_value = sum(
            pos["qty"] * prices.get(sym, pos["avg_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value

    def get_position_info(self, prices: dict[str, float], current_day: int = 0) -> list[PositionInfo]:
        """Get position details for observation."""
        result = []
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos["avg_price"])
            pnl_pct = (
                (price - pos["avg_price"]) / pos["avg_price"] * 100
                if pos["avg_price"] > 0 else 0.0
            )
            result.append(PositionInfo(
                symbol=sym,
                quantity=pos["qty"],
                avg_price=round(pos["avg_price"], 2),
                current_price=round(price, 2),
                pnl_percent=round(pnl_pct, 2),
                market_value=round(pos["qty"] * price, 2),
            ))
        return result

    def record_daily(self, prices: dict[str, float]) -> None:
        """Record end-of-day portfolio value and update drawdown."""
        value = self.get_value(prices)
        if len(self.daily_values) > 0:
            prev = self.daily_values[-1]
            daily_ret = (value - prev) / prev if prev > 0 else 0.0
            self.daily_returns.append(daily_ret)
        self.daily_values.append(value)
        self.trades_today = 0

        if value > self.peak_value:
            self.peak_value = value
        dd = (self.peak_value - value) / self.peak_value if self.peak_value > 0 else 0.0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    # --- Risk properties ---

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak (0.0 to 1.0)."""
        if not self.daily_values or self.peak_value <= 0:
            return 0.0
        current = self.daily_values[-1]
        return (self.peak_value - current) / self.peak_value

    @property
    def trading_capacity(self) -> float:
        """Scaling factor (0.25-1.0) based on current drawdown.

        Reduces the agent's ability to deploy capital when losing.
        Mirrors real-world risk management where position sizing
        decreases as drawdown increases.
        """
        dd = self.current_drawdown
        for threshold, capacity in DRAWDOWN_CAPACITY:
            if dd < threshold:
                return capacity
        return DRAWDOWN_CAPACITY[-1][1]

    @property
    def recent_win_rate(self) -> float | None:
        """Win rate of recent trades. None if no trades yet."""
        if not self._recent_outcomes:
            return None
        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    @property
    def losing_streak(self) -> int:
        """Number of consecutive losses from most recent trade."""
        streak = 0
        for outcome in reversed(self._recent_outcomes):
            if not outcome:
                streak += 1
            else:
                break
        return streak

    def record_trade_outcome(self, won: bool) -> None:
        """Record whether a completed trade was profitable."""
        self._recent_outcomes.append(won)

    def get_position_age(self, symbol: str, current_day: int) -> int:
        """Days held for a position. Returns 0 if no position."""
        if symbol not in self.positions:
            return 0
        entry_day = self.positions[symbol].get("entry_day", current_day)
        return max(0, current_day - entry_day)

    def compute_holding_cost(self, current_day: int) -> float:
        """Total holding cost for all aged positions.

        Positions held longer than POSITION_AGE_THRESHOLD days
        incur a daily cost, forcing the agent to learn exit timing.
        """
        total_cost = 0.0
        for sym, pos in self.positions.items():
            age = self.get_position_age(sym, current_day)
            if age > POSITION_AGE_THRESHOLD:
                excess_days = age - POSITION_AGE_THRESHOLD
                total_cost += POSITION_AGE_COST_PER_DAY * excess_days
        return total_cost
