"""Tests for the neural market simulator."""

from __future__ import annotations

import pytest

from server.neural_simulator import CHECKPOINT_DIR, NeuralSimulator

_has_checkpoint = any(
    (CHECKPOINT_DIR / name).exists()
    for name in ["best_transformer.pt", "best_cnn-gru.pt", "best_model.pt"]
)

pytestmark = pytest.mark.skipif(not _has_checkpoint, reason="No world model checkpoint available")


@pytest.fixture
def simulator():
    return NeuralSimulator(task_id="single_stock", seed=42, temperature=0.8)


class TestNeuralSimulatorInterface:
    """Verify NeuralSimulator has the same interface as MarketSimulator."""

    def test_reset_sets_state(self, simulator):
        simulator.reset()
        assert simulator.current_day == 0
        assert not simulator.is_done

    def test_advance_day(self, simulator):
        simulator.reset()
        simulator.advance_day()
        assert simulator.current_day == 1

    def test_episode_completes(self, simulator):
        simulator.reset()
        for _ in range(simulator.episode_days):
            assert not simulator.is_done
            simulator.advance_day()
        assert simulator.is_done

    def test_get_price_returns_float(self, simulator):
        simulator.reset()
        price = simulator.get_price("RELIANCE")
        assert isinstance(price, float)
        assert price > 0

    def test_get_daily_change(self, simulator):
        simulator.reset()
        simulator.advance_day()
        change = simulator.get_daily_change("RELIANCE")
        assert isinstance(change, float)
        assert -50 < change < 50

    def test_get_lookback_data_shape(self, simulator):
        simulator.reset()
        lookback = simulator.get_lookback_data("RELIANCE", lookback=50)
        assert len(lookback) == 51
        assert "close" in lookback.columns
        assert "open" in lookback.columns
        assert "volume" in lookback.columns

    def test_get_market_breadth(self, simulator):
        simulator.reset()
        breadth = simulator.get_market_breadth()
        assert "avg_change" in breadth
        assert "advancing" in breadth
        assert "declining" in breadth
        assert "breadth_weak" in breadth

    def test_get_5day_trend(self, simulator):
        simulator.reset()
        for _ in range(5):
            simulator.advance_day()
        trend = simulator.get_5day_trend("RELIANCE")
        assert isinstance(trend, float)

    def test_get_current_date(self, simulator):
        simulator.reset()
        d = simulator.get_current_date()
        assert d is not None


class TestNeuralGeneration:
    """Verify the neural simulator generates plausible data."""

    def test_prices_are_positive(self, simulator):
        simulator.reset()
        for _ in range(simulator.episode_days):
            price = simulator.get_price("RELIANCE")
            assert price > 0, f"Price should be positive, got {price}"
            simulator.advance_day()

    def test_stochastic_episodes(self):
        """Same seed should differ across resets due to model stochasticity."""
        sim1 = NeuralSimulator(task_id="single_stock", seed=42)
        sim2 = NeuralSimulator(task_id="single_stock", seed=99)
        sim1.reset()
        sim2.reset()

        prices1 = [sim1.get_price("RELIANCE")]
        prices2 = [sim2.get_price("RELIANCE")]
        for _ in range(5):
            sim1.advance_day()
            sim2.advance_day()
            prices1.append(sim1.get_price("RELIANCE"))
            prices2.append(sim2.get_price("RELIANCE"))

        assert prices1 != prices2, "Different seeds should produce different episodes"


class TestGroundTruthComparison:
    """Verify ground truth storage and comparison."""

    def test_ground_truth_exists(self, simulator):
        simulator.reset()
        truth = simulator.get_ground_truth("RELIANCE")
        assert not truth.empty
        assert len(truth) == simulator.episode_days

    def test_prediction_error_metrics(self, simulator):
        simulator.reset()
        errors = simulator.compute_prediction_error("RELIANCE")
        assert "mae_returns" in errors
        assert "direction_accuracy" in errors
        assert "volatility_ratio" in errors
        assert "gen_total_return" in errors
        assert "true_total_return" in errors
        assert 0 <= errors["direction_accuracy"] <= 1
        assert errors["volatility_ratio"] > 0
