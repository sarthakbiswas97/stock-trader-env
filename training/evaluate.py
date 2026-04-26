"""Standardized evaluation harness — run N episodes and aggregate metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from training.gym_wrapper import StockTradingGymEnv
from training.experiment import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResults:
    """Immutable evaluation results."""

    mean_score: float
    std_score: float
    mean_return: float
    std_return: float
    sharpe: float
    episodes: int
    scores: tuple[float, ...]
    returns: tuple[float, ...]


def evaluate_agent(
    agent_fn: Callable[[str], str],
    task_id: str = "single_stock",
    split: str = "test",
    n_episodes: int = 50,
    seed: int = 42,
    agent_name: str = "unnamed",
    log_to_mlflow: bool = True,
    simulator_mode: str = "replay",
) -> EvalResults:
    """Run agent through N episodes and return aggregate EvalResults."""
    scores: list[float] = []
    returns: list[float] = []

    for i in range(n_episodes):
        episode_seed = seed + i
        env = StockTradingGymEnv(
            task_id=task_id,
            seed=episode_seed,
            obs_mode="text",
            split=split,
            simulator_mode=simulator_mode,
        )

        obs, info = env.reset()
        initial_value = info["portfolio_value"]

        while True:
            action = agent_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        final_value = info["portfolio_value"]
        episode_return = (final_value - initial_value) / initial_value
        scores.append(info["score"])
        returns.append(episode_return)
        env.close()

        if (i + 1) % 10 == 0:
            logger.info(
                "Eval %d/%d — mean score: %.3f",
                i + 1, n_episodes, np.mean(scores),
            )

    scores_arr = np.array(scores)
    returns_arr = np.array(returns)
    sharpe = (
        float(np.mean(returns_arr) / np.std(returns_arr))
        if np.std(returns_arr) > 0 else 0.0
    )

    results = EvalResults(
        mean_score=float(np.mean(scores_arr)),
        std_score=float(np.std(scores_arr)),
        mean_return=float(np.mean(returns_arr)),
        std_return=float(np.std(returns_arr)),
        sharpe=sharpe,
        episodes=n_episodes,
        scores=tuple(scores),
        returns=tuple(returns),
    )

    if log_to_mlflow:
        with ExperimentTracker(
            run_name=f"eval_{agent_name}_{task_id}_{split}",
            task_id=task_id,
            agent_type=agent_name,
            split=split,
            seed=seed,
        ) as tracker:
            for i, (s, r) in enumerate(zip(scores, returns)):
                tracker.log_episode(s, r, step=i + 1)

    logger.info(
        "Evaluation complete: %d episodes, mean_score=%.3f, sharpe=%.2f",
        n_episodes, results.mean_score, results.sharpe,
    )
    return results
