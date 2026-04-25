"""Tests for action parsing logic."""

import pytest

from server.action_parser import parse_action
from server.tasks import TASK_CONFIGS


@pytest.fixture
def single_config():
    return TASK_CONFIGS["single_stock"]


@pytest.fixture
def portfolio_config():
    return TASK_CONFIGS["portfolio"]


class TestSingleStockParsing:
    """Single-stock task: bare BUY/SELL should resolve to the default symbol."""

    def test_hold(self, single_config):
        result = parse_action("HOLD", single_config)
        assert result == {"type": "HOLD"}

    def test_buy_bare(self, single_config):
        """Bare 'BUY' should default to the only symbol (RELIANCE)."""
        result = parse_action("BUY", single_config)
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"
        assert result["fraction"] == 1.0

    def test_sell_bare(self, single_config):
        """Bare 'SELL' should default to the only symbol."""
        result = parse_action("SELL", single_config)
        assert result["type"] == "SELL"
        assert result["symbol"] == "RELIANCE"

    def test_buy_with_symbol(self, single_config):
        """Explicit symbol should work even in single-stock mode."""
        result = parse_action("BUY RELIANCE", single_config)
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"

    def test_buy_with_fraction(self, single_config):
        result = parse_action("BUY RELIANCE 0.5", single_config)
        assert result["fraction"] == 0.5

    def test_case_insensitive(self, single_config):
        """Actions should be case-insensitive."""
        result = parse_action("buy reliance 0.3", single_config)
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"
        assert result["fraction"] == 0.3


class TestPortfolioParsing:
    """Multi-stock task: symbol is required for BUY/SELL."""

    def test_buy_requires_symbol(self, portfolio_config):
        """Bare 'BUY' without symbol should default to HOLD in multi-stock."""
        result = parse_action("BUY", portfolio_config)
        assert result["type"] == "HOLD"

    def test_buy_with_valid_symbol(self, portfolio_config):
        result = parse_action("BUY INFY", portfolio_config)
        assert result["type"] == "BUY"
        assert result["symbol"] == "INFY"

    def test_buy_invalid_symbol(self, portfolio_config):
        """Symbol not in task config should default to HOLD."""
        result = parse_action("BUY GOOGL", portfolio_config)
        assert result["type"] == "HOLD"

    def test_sell_with_fraction(self, portfolio_config):
        result = parse_action("SELL TCS 0.5", portfolio_config)
        assert result["type"] == "SELL"
        assert result["symbol"] == "TCS"
        assert result["fraction"] == 0.5


class TestEdgeCases:
    """Malformed inputs, boundary conditions."""

    def test_empty_string(self, single_config):
        result = parse_action("", single_config)
        assert result["type"] == "HOLD"

    def test_whitespace_only(self, single_config):
        result = parse_action("   ", single_config)
        assert result["type"] == "HOLD"

    def test_invalid_action_word(self, single_config):
        result = parse_action("YOLO RELIANCE", single_config)
        assert result["type"] == "HOLD"

    def test_fraction_clamped_above_one(self, single_config):
        """Fraction > 1.0 should be clamped to 1.0."""
        result = parse_action("BUY RELIANCE 5.0", single_config)
        assert result["fraction"] == 1.0

    def test_fraction_clamped_below_zero(self, single_config):
        """Negative fraction should be clamped to 0.0."""
        result = parse_action("BUY RELIANCE -0.5", single_config)
        assert result["fraction"] == 0.0

    def test_invalid_fraction_defaults_to_one(self, single_config):
        """Non-numeric fraction should default to 1.0."""
        result = parse_action("BUY RELIANCE abc", single_config)
        assert result["fraction"] == 1.0

    def test_extra_whitespace(self, single_config):
        """Extra whitespace should be handled."""
        result = parse_action("  BUY   RELIANCE   0.3  ", single_config)
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"
