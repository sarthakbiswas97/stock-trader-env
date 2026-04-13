"""MLflow experiment tracking — standardized logging for all training runs.

Enforces a consistent schema so every run (PPO, DQN, LLM, rule-based)
logs the same parameters, metrics, and artifacts. This makes the MLflow
dashboard comparable across agent types.

Usage:
    with ExperimentTracker("ppo_baseline", task_id="single_stock") as tracker:
        for episode in range(100):
            score, total_return = run_episode(...)
            tracker.log_episode(score, total_return)
        tracker.log_model("checkpoints/best.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow

from server import __version__
from server.tasks import TASK_CONFIGS

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wraps MLflow with a consistent schema for trading experiments."""

    def __init__(
        self,
        run_name: str,
        task_id: str,
        agent_type: str,
        obs_mode: str = "text",
        split: str = "train",
        seed: int | None = None,
        experiment_name: str = "stock-trader",
        extra_params: dict[str, Any] | None = None,
    ):
        self._run_name = run_name
        self._task_id = task_id
        self._agent_type = agent_type
        self._obs_mode = obs_mode
        self._split = split
        self._seed = seed
        self._experiment_name = experiment_name
        self._extra_params = extra_params or {}

        self._scores: list[float] = []
        self._returns: list[float] = []
        self._run = None

    def __enter__(self) -> "ExperimentTracker":
        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run(run_name=self._run_name)

        task_config = TASK_CONFIGS.get(self._task_id, {})

        # Log fixed parameters
        mlflow.log_params({
            "env_version": __version__,
            "task_id": self._task_id,
            "task_version": task_config.get("version", "unknown"),
            "agent_type": self._agent_type,
            "obs_mode": self._obs_mode,
            "split": self._split,
            "seed": self._seed or "none",
            "episode_days": task_config.get("episode_days", 0),
            "initial_capital": task_config.get("initial_capital", 0),
            **self._extra_params,
        })

        # Tags for filtering in MLflow UI
        mlflow.set_tags({
            "agent_type": self._agent_type,
            "task_difficulty": task_config.get("difficulty", "unknown"),
        })

        logger.info(
            "Started MLflow run '%s' (experiment: %s)",
            self._run_name, self._experiment_name,
        )
        return self

    def __exit__(self, *args: Any) -> None:
        # Log final aggregate metrics
        if self._scores:
            import numpy as np
            scores = np.array(self._scores)
            returns = np.array(self._returns)

            mlflow.log_metrics({
                "best_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "episodes_completed": len(self._scores),
            })

            # Sharpe-like ratio
            if np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns))
                mlflow.log_metric("sharpe_ratio", sharpe)

        mlflow.end_run()
        logger.info("Ended MLflow run '%s'", self._run_name)

    def log_episode(
        self,
        score: float,
        total_return: float,
        step: int | None = None,
    ) -> None:
        """Log metrics from one completed episode."""
        self._scores.append(score)
        self._returns.append(total_return)

        episode_num = len(self._scores)
        mlflow.log_metrics(
            {
                "episode_score": score,
                "episode_return": total_return,
            },
            step=step or episode_num,
        )

    def log_model(self, path: str | Path) -> None:
        """Log a model checkpoint as an artifact."""
        mlflow.log_artifact(str(path))
        logger.info("Logged model artifact: %s", path)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log a config dict as a YAML artifact."""
        import yaml
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(config, f, default_flow_style=False)
            f.flush()
            mlflow.log_artifact(f.name, "config")

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None
