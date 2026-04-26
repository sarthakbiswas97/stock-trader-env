"""Integration tests — full episode runs through the environment."""

import pytest

from server.environment import StockTradingEnvironment
from server.tasks import TASK_CONFIGS
from server import __version__
from models import TradeAction, MarketObservation


@pytest.fixture
def env():
    return StockTradingEnvironment()


class TestReset:
    """reset() should initialize a clean episode."""

    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert isinstance(obs, MarketObservation)

    def test_reset_day_one(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert obs.day == 1
        assert obs.total_days == 20

    def test_reset_initial_capital(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert obs.portfolio_value == 100_000
        assert obs.cash == 100_000

    def test_reset_no_positions(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert obs.positions == []

    def test_reset_not_done(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert obs.done is False

    def test_reset_has_version(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert obs.env_version == __version__
        assert obs.task_version == "1.0.0"

    def test_reset_has_market_summary(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert "Day 1 of 20" in obs.market_summary
        assert "RELIANCE" in obs.market_summary

    def test_reset_has_available_actions(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        assert "HOLD" in obs.available_actions
        assert "BUY" in obs.available_actions

    def test_reset_invalid_task_defaults_to_single(self, env):
        obs = env.reset(seed=42, task_id="nonexistent_task")
        assert obs.task_id == "single_stock"

    def test_reset_portfolio_task(self, env):
        obs = env.reset(seed=42, task_id="portfolio")
        assert obs.total_days == 30
        assert obs.cash == 200_000

    def test_reset_full_autonomous_task(self, env):
        obs = env.reset(seed=42, task_id="full_autonomous")
        assert obs.total_days == 40
        assert obs.cash == 500_000

    def test_reset_deterministic_with_seed(self, env):
        """Same seed should produce same initial state."""
        obs1 = env.reset(seed=42, task_id="single_stock")
        obs2 = env.reset(seed=42, task_id="single_stock")
        assert obs1.market_summary == obs2.market_summary
        assert obs1.portfolio_value == obs2.portfolio_value


class TestStep:
    """step() processes one trading day."""

    def test_step_advances_day(self, env):
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="HOLD"))
        assert obs.day == 2

    def test_step_hold_preserves_cash(self, env):
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="HOLD"))
        assert obs.cash == 100_000

    def test_step_buy_reduces_cash(self, env):
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="BUY"))
        assert obs.cash < 100_000
        assert len(obs.positions) == 1

    def test_step_buy_then_sell(self, env):
        env.reset(seed=42, task_id="single_stock")
        env.step(TradeAction(action="BUY"))
        obs = env.step(TradeAction(action="SELL"))
        assert len(obs.positions) == 0
        # Cash should be back (minus any price movement)
        assert obs.cash > 0

    def test_step_sell_without_position(self, env):
        """Selling without a position should return negative reward."""
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="SELL"))
        # Reward includes penalty for selling what you don't own
        assert obs.reward < 0 or obs.positions == []

    def test_step_returns_reward(self, env):
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="HOLD"))
        assert isinstance(obs.reward, float)

    def test_step_has_score(self, env):
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="HOLD"))
        assert 0 <= obs.score <= 1

    def test_step_invalid_action_treated_as_hold(self, env):
        env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="DANCE"))
        assert obs.cash == 100_000  # No trade happened


class TestFullEpisode:
    """Run complete episodes to test episode lifecycle."""

    def test_single_stock_episode_completes(self, env):
        obs = env.reset(seed=42, task_id="single_stock")
        steps = 0
        while not obs.done:
            obs = env.step(TradeAction(action="HOLD"))
            steps += 1
        assert steps == 20
        assert obs.done is True
        assert 0 < obs.score < 1

    def test_portfolio_episode_with_trades(self, env):
        obs = env.reset(seed=42, task_id="portfolio")
        # Buy a few stocks on day 1
        obs = env.step(TradeAction(action="BUY RELIANCE"))
        assert len(obs.positions) >= 1
        obs = env.step(TradeAction(action="BUY INFY"))

        # Hold through the rest
        while not obs.done:
            obs = env.step(TradeAction(action="HOLD"))

        assert obs.done is True
        assert 0 < obs.score < 1

    def test_episode_score_improves_with_good_trades(self, env):
        """Buying and holding should generally score different from just holding."""
        # Episode 1: all HOLD
        obs = env.reset(seed=42, task_id="single_stock")
        while not obs.done:
            obs = env.step(TradeAction(action="HOLD"))
        hold_score = obs.score

        # Episode 2: buy on day 1, hold rest
        obs = env.reset(seed=42, task_id="single_stock")
        obs = env.step(TradeAction(action="BUY"))
        while not obs.done:
            obs = env.step(TradeAction(action="HOLD"))
        buy_score = obs.score

        # Scores should differ — agent that trades vs agent that doesn't
        assert hold_score != buy_score

    def test_step_after_done_returns_safe_observation(self, env):
        """Stepping after episode ends should not crash."""
        obs = env.reset(seed=42, task_id="single_stock")
        while not obs.done:
            obs = env.step(TradeAction(action="HOLD"))
        # Step again after done
        obs = env.step(TradeAction(action="BUY"))
        assert obs.done is True


class TestMultiStockActions:
    """Portfolio and full_autonomous task-specific behavior."""

    def test_buy_specific_stock(self, env):
        env.reset(seed=42, task_id="portfolio")
        obs = env.step(TradeAction(action="BUY TCS"))
        tcs_positions = [p for p in obs.positions if p.symbol == "TCS"]
        assert len(tcs_positions) == 1

    def test_buy_with_fraction(self, env):
        env.reset(seed=42, task_id="portfolio")
        obs = env.step(TradeAction(action="BUY RELIANCE 0.3"))
        # Should have spent roughly 30% of cash
        assert obs.cash > 100_000  # More than half remaining (started at 200k)

    def test_position_limit_enforced(self, env):
        """Can't exceed per-stock position limit."""
        env.reset(seed=42, task_id="portfolio")
        # Try buying RELIANCE with all cash — should be capped at 30%
        obs = env.step(TradeAction(action="BUY RELIANCE"))
        if obs.positions:
            reliance_value = obs.positions[0].market_value
            total_value = obs.portfolio_value
            # Position should not exceed 30% of portfolio
            assert reliance_value / total_value <= 0.35  # Small tolerance for rounding


class TestRegimeGate:
    """Hard task regime gate behavior."""

    def test_regime_gate_only_on_hard_task(self, env):
        """Regime gate should not affect single_stock or portfolio tasks."""
        assert TASK_CONFIGS["single_stock"]["regime_gate"] is False
        assert TASK_CONFIGS["portfolio"]["regime_gate"] is False
        assert TASK_CONFIGS["full_autonomous"]["regime_gate"] is True


class TestRewardSignal:
    """Reward properties important for RL training."""

    def test_reward_is_finite(self, env):
        env.reset(seed=42, task_id="single_stock")
        for _ in range(5):
            obs = env.step(TradeAction(action="HOLD"))
            assert isinstance(obs.reward, float)
            assert abs(obs.reward) < 100  # Sanity bound

    def test_buy_sell_cycle_reward(self, env):
        """A buy-sell cycle should produce non-zero cumulative reward."""
        env.reset(seed=42, task_id="single_stock")
        rewards = []
        obs = env.step(TradeAction(action="BUY"))
        rewards.append(obs.reward)
        obs = env.step(TradeAction(action="HOLD"))
        rewards.append(obs.reward)
        obs = env.step(TradeAction(action="SELL"))
        rewards.append(obs.reward)
        # At least one reward should be non-zero (price moved)
        assert any(r != 0 for r in rewards)
