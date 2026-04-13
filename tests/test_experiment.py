"""Tests for experiment tracking and evaluation harness."""

import os

import mlflow
import pytest

from training.experiment import ExperimentTracker
from training.evaluate import evaluate_agent


@pytest.fixture(autouse=True)
def isolate_mlflow(tmp_path):
    """Each test gets its own MLflow directory. Reset state between tests."""
    tracking_uri = str(tmp_path / "mlruns")
    os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    # Ensure no active run leaks between tests
    if mlflow.active_run():
        mlflow.end_run()
    yield tracking_uri
    if mlflow.active_run():
        mlflow.end_run()


class TestExperimentTracker:
    """MLflow experiment tracking wrapper."""

    def test_creates_run(self):
        with ExperimentTracker(
            run_name="test_run",
            task_id="single_stock",
            agent_type="test",
        ) as tracker:
            assert tracker.run_id is not None

    def test_logs_episode_metrics(self):
        with ExperimentTracker(
            run_name="test_metrics",
            task_id="single_stock",
            agent_type="test",
        ) as tracker:
            tracker.log_episode(score=0.5, total_return=0.02)
            tracker.log_episode(score=0.6, total_return=0.03)

        run = mlflow.get_run(tracker.run_id)
        assert run.data.metrics["mean_score"] == pytest.approx(0.55)
        assert run.data.metrics["episodes_completed"] == 2

    def test_logs_parameters(self):
        with ExperimentTracker(
            run_name="test_params",
            task_id="portfolio",
            agent_type="ppo",
            split="val",
            seed=42,
        ) as tracker:
            pass

        run = mlflow.get_run(tracker.run_id)
        assert run.data.params["task_id"] == "portfolio"
        assert run.data.params["agent_type"] == "ppo"
        assert run.data.params["split"] == "val"
        assert run.data.params["seed"] == "42"

    def test_logs_extra_params(self):
        with ExperimentTracker(
            run_name="test_extra",
            task_id="single_stock",
            agent_type="ppo",
            extra_params={"learning_rate": "0.001", "batch_size": "64"},
        ) as tracker:
            pass

        run = mlflow.get_run(tracker.run_id)
        assert run.data.params["learning_rate"] == "0.001"


class TestEvaluateAgent:
    """Standardized evaluation harness."""

    def test_hold_agent(self):
        results = evaluate_agent(
            agent_fn=lambda obs: "HOLD",
            task_id="single_stock",
            split="train",
            n_episodes=3,
            seed=42,
            agent_name="hold_baseline",
            log_to_mlflow=True,
        )
        assert results.episodes == 3
        assert len(results.scores) == 3
        assert all(0 < s < 1 for s in results.scores)

    def test_eval_results_frozen(self):
        results = evaluate_agent(
            agent_fn=lambda obs: "HOLD",
            task_id="single_stock",
            split="train",
            n_episodes=2,
            seed=42,
            agent_name="test",
            log_to_mlflow=False,
        )
        with pytest.raises(AttributeError):
            results.mean_score = 999

    def test_eval_without_mlflow(self):
        results = evaluate_agent(
            agent_fn=lambda obs: "HOLD",
            task_id="single_stock",
            split="train",
            n_episodes=2,
            seed=42,
            agent_name="test",
            log_to_mlflow=False,
        )
        assert results.mean_score > 0
