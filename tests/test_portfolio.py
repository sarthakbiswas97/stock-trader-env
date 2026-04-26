"""Tests for Portfolio class — cash tracking, position math, daily returns."""

import pytest

from server.portfolio import Portfolio


class TestPortfolioInit:
    """Initial state should be clean."""

    def test_initial_cash(self):
        p = Portfolio(100_000)
        assert p.cash == 100_000

    def test_initial_value_equals_cash(self):
        p = Portfolio(100_000)
        assert p.get_value({}) == 100_000

    def test_no_initial_positions(self):
        p = Portfolio(100_000)
        assert p.positions == {}
        assert p.total_trades == 0
        assert p.risk_violations == 0


class TestPortfolioValue:
    """Portfolio value = cash + sum of (qty * current_price) for all positions."""

    def test_value_with_one_position(self):
        p = Portfolio(100_000)
        p.cash = 50_000
        p.positions = {"RELIANCE": {"qty": 10, "avg_price": 2000}}
        # Current price is 2500, so position value = 10 * 2500 = 25000
        assert p.get_value({"RELIANCE": 2500}) == 75_000

    def test_value_with_multiple_positions(self):
        p = Portfolio(100_000)
        p.cash = 20_000
        p.positions = {
            "RELIANCE": {"qty": 10, "avg_price": 2000},
            "INFY": {"qty": 20, "avg_price": 1500},
        }
        # RELIANCE: 10 * 2500 = 25000, INFY: 20 * 1800 = 36000
        value = p.get_value({"RELIANCE": 2500, "INFY": 1800})
        assert value == 81_000

    def test_value_uses_avg_price_as_fallback(self):
        """If current price missing, fall back to avg_price."""
        p = Portfolio(100_000)
        p.cash = 50_000
        p.positions = {"RELIANCE": {"qty": 10, "avg_price": 2000}}
        # No price provided — should use avg_price (2000)
        assert p.get_value({}) == 70_000


class TestPositionInfo:
    """Position info returned to the agent in observations."""

    def test_pnl_percent_positive(self):
        p = Portfolio(100_000)
        p.positions = {"RELIANCE": {"qty": 10, "avg_price": 2000}}
        info = p.get_position_info({"RELIANCE": 2200})
        assert len(info) == 1
        assert info[0].symbol == "RELIANCE"
        assert info[0].pnl_percent == 10.0  # (2200-2000)/2000 * 100

    def test_pnl_percent_negative(self):
        p = Portfolio(100_000)
        p.positions = {"RELIANCE": {"qty": 10, "avg_price": 2000}}
        info = p.get_position_info({"RELIANCE": 1800})
        assert info[0].pnl_percent == -10.0

    def test_market_value(self):
        p = Portfolio(100_000)
        p.positions = {"RELIANCE": {"qty": 10, "avg_price": 2000}}
        info = p.get_position_info({"RELIANCE": 2500})
        assert info[0].market_value == 25_000


class TestDailyRecording:
    """Daily value recording, returns calculation, drawdown tracking."""

    def test_daily_return_calculation(self):
        p = Portfolio(100_000)
        # Day 1: value goes to 105000 (5% gain)
        p.cash = 105_000
        p.record_daily({})
        assert len(p.daily_returns) == 1
        assert p.daily_returns[0] == pytest.approx(0.05)

    def test_drawdown_tracking(self):
        p = Portfolio(100_000)
        # Day 1: value rises to 110000
        p.cash = 110_000
        p.record_daily({})
        assert p.peak_value == 110_000
        assert p.max_drawdown == 0.0

        # Day 2: value drops to 99000
        p.cash = 99_000
        p.record_daily({})
        # Drawdown = (110000 - 99000) / 110000 = 0.1
        assert p.max_drawdown == pytest.approx(0.1, abs=1e-4)

    def test_trades_today_resets_on_record(self):
        p = Portfolio(100_000)
        p.trades_today = 5
        p.record_daily({})
        assert p.trades_today == 0
