"""Gymnasium-compatible wrapper for the Stock Trading Environment."""

from __future__ import annotations

import uuid
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models import TradeAction
from server.environment import StockTradingEnvironment
from server.market_simulator import _load_stock_data
from server.tasks import TASK_CONFIGS
from training.data_splits import SPLITS, get_valid_index_range
from training.observations import (
    NUMERIC_OBS_SIZE,
    obs_to_numeric,
    obs_to_text,
)
from training.trajectory_logger import TrajectoryLogger


class StockTradingGymEnv(gym.Env):
    """Gymnasium wrapper around StockTradingEnvironment.

    Args:
        task_id: One of "single_stock", "portfolio", "full_autonomous"
        seed: Random seed for reproducibility
        obs_mode: "text" for LLM agents, "numeric" for PPO/DQN
        split: "train", "val", "test", or None (no date restriction)
        log_trajectories: Whether to write episode data to disk
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_id: str = "single_stock",
        seed: int | None = None,
        obs_mode: str = "text",
        split: str | None = None,
        log_trajectories: bool = False,
        simulator_mode: str = "replay",
    ):
        super().__init__()

        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(TASK_CONFIGS)}")
        if obs_mode not in ("text", "numeric"):
            raise ValueError(f"Unknown obs_mode: {obs_mode}. Choose 'text' or 'numeric'")
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split: {split}. Choose from {list(SPLITS)}")

        self._task_id = task_id
        self._seed = seed
        self._obs_mode = obs_mode
        self._split = SPLITS[split] if split else None
        self._log_trajectories = log_trajectories
        self._simulator_mode = simulator_mode

        self._config = TASK_CONFIGS[task_id]
        self._env = StockTradingEnvironment()
        self._logger: TrajectoryLogger | None = None
        self._episode_count = 0

        # Precompute valid index range for the split
        self._valid_range: tuple[int, int] | None = None
        if self._split is not None:
            self._compute_valid_range()

        # Define observation and action spaces
        if obs_mode == "numeric":
            self.observation_space = spaces.Box(
                low=-1.0, high=np.inf, shape=(NUMERIC_OBS_SIZE,), dtype=np.float32
            )
        else:
            # Text observations — variable length strings
            self.observation_space = spaces.Text(
                min_length=0, max_length=10_000
            )

        # Action space: text string (the agent outputs action commands)
        self.action_space = spaces.Text(min_length=1, max_length=100)

    def _compute_valid_range(self) -> None:
        """Compute valid start index range for the current split."""
        # Load timestamps from first symbol to establish date mapping
        sym = self._config["symbols"][0]
        df = _load_stock_data(sym)

        self._valid_range = get_valid_index_range(
            timestamps=df["timestamp"],
            split=self._split,
            lookback=50,
            episode_days=self._config["episode_days"],
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment for a new episode.

        Returns:
            (observation, info) tuple per Gymnasium spec.
        """
        # Close previous trajectory logger if open
        if self._logger is not None:
            self._logger.close()

        effective_seed = seed if seed is not None else self._seed
        self._episode_count += 1

        # If split is set, override the simulator's random start
        # by manipulating the seed to land within valid range
        obs = self._env.reset(
            seed=effective_seed,
            task_id=self._task_id,
            simulator_mode=self._simulator_mode,
        )

        # Constrain to split date range if needed
        if self._valid_range is not None and self._env._sim is not None:
            min_start, max_start = self._valid_range
            import random
            rng = random.Random(effective_seed)
            constrained_start = rng.randint(min_start, max_start)
            # Override the simulator's chosen start indices
            for sym in self._env._sim.symbols:
                self._env._sim._start_idx[sym] = constrained_start
            # Re-build the observation with correct prices
            obs = self._env._build_observation(0.0)

        # Start trajectory logging
        if self._log_trajectories:
            episode_id = f"ep{self._episode_count:06d}_{uuid.uuid4().hex[:8]}"
            split_name = self._split.name if self._split else "all"
            self._logger = TrajectoryLogger(
                task_id=self._task_id,
                split=split_name,
                episode_id=episode_id,
            )
            self._logger.log_reset(obs.market_summary)

        converted_obs = self._convert_obs(obs)
        info = self._build_info(obs)
        return converted_obs, info

    def step(
        self, action: str,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take one step in the environment.

        Args:
            action: Action string (e.g., "BUY RELIANCE 0.3", "HOLD")

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs = self._env.step(TradeAction(action=action))

        terminated = obs.done
        truncated = False  # We don't have time limits beyond episode length
        reward = obs.reward

        if self._logger is not None:
            info = self._build_info(obs)
            self._logger.log_step(
                observation=obs.market_summary,
                action=action,
                reward=reward,
                done=terminated,
                info=info if terminated else None,
            )
            if terminated:
                self._logger.close()
                self._logger = None

        converted_obs = self._convert_obs(obs)
        info = self._build_info(obs)
        return converted_obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    def _convert_obs(self, obs: Any) -> Any:
        """Convert MarketObservation to the configured format."""
        if self._obs_mode == "text":
            return obs_to_text(obs)
        return obs_to_numeric(obs, self._config["initial_capital"])

    def _build_info(self, obs: Any) -> dict[str, Any]:
        """Build the info dict returned alongside observations."""
        return {
            "day": obs.day,
            "total_days": obs.total_days,
            "portfolio_value": obs.portfolio_value,
            "cash": obs.cash,
            "score": obs.score,
            "task_id": obs.task_id,
            "env_version": obs.env_version,
            "task_version": obs.task_version,
            "num_positions": len(obs.positions),
        }
