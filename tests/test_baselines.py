"""Tests for baseline agents."""

import os

import mlflow
import pytest

from baselines.hold_agent import hold_agent
from baselines.rule_based_agent import rule_based_agent, _parse_stocks, _parse_positions
from training.gym_wrapper import StockTradingGymEnv


@pytest.fixture(autouse=True)
def isolate_mlflow(tmp_path):
    tracking_uri = str(tmp_path / "mlruns")
    os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    yield


class TestHoldAgent:

    def test_always_returns_hold(self):
        assert hold_agent("any observation text") == "HOLD"
        assert hold_agent("") == "HOLD"

    def test_completes_episode(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        obs, _ = env.reset()
        steps = 0
        while True:
            action = hold_agent(obs)
            obs, _, terminated, _, _ = env.step(action)
            steps += 1
            if terminated:
                break
        assert steps == 20
        env.close()


class TestRuleBasedAgent:

    def test_returns_valid_action(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        obs, _ = env.reset()
        action = rule_based_agent(obs)
        assert action in ("HOLD", "BUY", "SELL", "BUY RELIANCE", "SELL RELIANCE")
        env.close()

    def test_completes_single_stock_episode(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        obs, _ = env.reset()
        actions = []
        while True:
            action = rule_based_agent(obs)
            actions.append(action)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        assert len(actions) == 20
        # All actions should be valid
        assert all(a in ("HOLD", "BUY", "SELL") for a in actions)
        env.close()

    def test_completes_portfolio_episode(self):
        env = StockTradingGymEnv(task_id="portfolio", seed=42, obs_mode="text")
        obs, _ = env.reset()
        while True:
            action = rule_based_agent(obs)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        env.close()

    def test_buys_when_oversold(self):
        """Observation with RSI < 35 and non-bearish trend should trigger BUY."""
        obs = """Day 5 of 20 | Cash: Rs100,000 | Portfolio: Rs100,000 | Return: +0.0%

RELIANCE: Rs1,050 (-2.0% today)
  RSI: 25 (oversold) | MACD: bullish
  Trend: bullish | Bollinger: lower_band (oversold)
  Volume: 1.5x avg (high) | Volatility: moderate
  Momentum: down (-3.0%)"""
        action = rule_based_agent(obs)
        assert "BUY" in action

    def test_holds_when_neutral(self):
        """RSI around 50 with no positions should HOLD."""
        obs = """Day 5 of 20 | Cash: Rs100,000 | Portfolio: Rs100,000 | Return: +0.0%

RELIANCE: Rs1,200 (+0.5% today)
  RSI: 50 (neutral) | MACD: bearish
  Trend: sideways | Bollinger: middle
  Volume: 1.0x avg (normal) | Volatility: moderate
  Momentum: flat (+0.1%)"""
        action = rule_based_agent(obs)
        assert action == "HOLD"

    def test_sells_on_profit_target(self):
        """Position with P&L > 3% should trigger SELL."""
        obs = """Day 10 of 20 | Cash: Rs10,000 | Portfolio: Rs105,000 | Return: +5.0%

RELIANCE: Rs1,300 (+1.0% today)
  RSI: 55 (neutral) | MACD: bullish
  Trend: bullish | Bollinger: above_middle
  Volume: 1.0x avg (normal) | Volatility: moderate
  Momentum: up (+2.0%)

Your Positions:
  RELIANCE: 72 shares @ Rs1,250 | Current: Rs1,300 | P&L: +4.0%"""
        action = rule_based_agent(obs)
        assert "SELL" in action


class TestParseStocks:

    def test_parses_single_stock(self):
        text = """RELIANCE: Rs1,179 (+0.4% today)
  RSI: 34 (neutral) | MACD: bearish
  Trend: bearish | Bollinger: lower_band"""
        stocks = _parse_stocks(text)
        assert "RELIANCE" in stocks
        assert stocks["RELIANCE"]["rsi"] == 34
        assert stocks["RELIANCE"]["trend"] == "bearish"

    def test_parses_multiple_stocks(self):
        text = """RELIANCE: Rs1,179 (+0.4% today)
  RSI: 34 (neutral) | MACD: bearish
  Trend: bullish | Bollinger: lower_band
INFY: Rs1,779 (-0.5% today)
  RSI: 64 (neutral) | MACD: bullish
  Trend: bullish | Bollinger: upper_band"""
        stocks = _parse_stocks(text)
        assert len(stocks) == 2
        assert stocks["RELIANCE"]["rsi"] == 34
        assert stocks["INFY"]["rsi"] == 64


class TestParsePositions:

    def test_parses_position(self):
        text = """Your Positions:
  RELIANCE: 84 shares @ Rs1,179 | Current: Rs1,200 | P&L: +1.8%"""
        positions = _parse_positions(text)
        assert "RELIANCE" in positions
        assert positions["RELIANCE"] == pytest.approx(1.8)

    def test_no_positions(self):
        text = "Day 1 of 20 | Cash: Rs100,000"
        positions = _parse_positions(text)
        assert positions == {}

    def test_negative_pnl(self):
        text = """Your Positions:
  RELIANCE: 84 shares @ Rs1,200 | Current: Rs1,150 | P&L: -4.2%"""
        positions = _parse_positions(text)
        assert positions["RELIANCE"] == pytest.approx(-4.2)
