"""PPO agent — trains a neural network policy via Stable-Baselines3.

Uses numeric observations (fixed-size float vector) and discrete actions.
The action space is simplified: HOLD, BUY (all-in), SELL (all).
For multi-stock tasks, one action per step targeting the "best" stock
by cycling through them.

This baseline demonstrates what traditional RL can achieve with limited
numeric observations — the gap vs LLM agents (which read rich text)
is part of the project narrative.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from training.gym_wrapper import StockTradingGymEnv
from training.observations import NUMERIC_OBS_SIZE

logger = logging.getLogger(__name__)


class DiscreteTradingEnv(gym.Env):
    """Wraps StockTradingGymEnv with a discrete action space for SB3.

    Action mapping:
        0 = HOLD
        1 = BUY (first available stock, full allocation)
        2 = SELL (first held stock, full position)

    For single_stock tasks, this maps directly to BUY/SELL/HOLD.
    For multi-stock tasks, it cycles through stocks each step.
    """

    def __init__(self, task_id: str = "single_stock", seed: int | None = None, split: str | None = None):
        super().__init__()
        self._inner = StockTradingGymEnv(
            task_id=task_id, seed=seed, obs_mode="numeric", split=split,
        )
        self._task_id = task_id
        self._symbols = self._inner._config["symbols"]
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(NUMERIC_OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        self._step_count = 0
        return self._inner.reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        self._step_count += 1
        # Pick which stock to target (cycle through)
        symbol = self._symbols[self._step_count % len(self._symbols)]

        if action == 0:
            action_str = "HOLD"
        elif action == 1:
            if len(self._symbols) == 1:
                action_str = "BUY"
            else:
                action_str = f"BUY {symbol}"
        else:
            if len(self._symbols) == 1:
                action_str = "SELL"
            else:
                action_str = f"SELL {symbol}"

        return self._inner.step(action_str)

    def close(self) -> None:
        self._inner.close()


def train_ppo(
    task_id: str = "single_stock",
    total_timesteps: int = 20_000,
    seed: int = 42,
    split: str = "train",
    save_path: str | None = None,
) -> PPO:
    """Train a PPO agent and return it.

    Args:
        task_id: Which task to train on.
        total_timesteps: Total environment steps for training.
        seed: Random seed.
        split: Data split for training.
        save_path: Where to save the trained model. If None, doesn't save.

    Returns:
        Trained PPO model.
    """
    env = DiscreteTradingEnv(task_id=task_id, seed=seed, split=split)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    logger.info("Training PPO for %d timesteps on %s...", total_timesteps, task_id)
    model.learn(total_timesteps=total_timesteps)
    logger.info("PPO training complete.")

    if save_path:
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

    env.close()
    return model


def make_ppo_agent(model: PPO, task_id: str = "single_stock") -> callable:
    """Create an agent function from a trained PPO model.

    Returns a callable that takes a text observation and returns an action string.
    This allows PPO to be evaluated through the same evaluate_agent() harness
    as the other baselines.

    Note: PPO was trained on numeric obs but evaluate_agent passes text obs.
    We re-create numeric obs from a fresh env step — this is a limitation
    of the current design, traded off for evaluation consistency.
    """
    from server.tasks import TASK_CONFIGS
    symbols = TASK_CONFIGS[task_id]["symbols"]

    # Internal env for numeric conversion
    _converter = StockTradingGymEnv(task_id=task_id, obs_mode="numeric")
    _step = [0]

    def agent_fn(observation: str) -> str:
        # Get numeric obs from the internal env's last state
        numeric_obs = _converter._convert_obs(_converter._env._build_observation(0.0))
        action_int, _ = model.predict(numeric_obs, deterministic=True)

        _step[0] += 1
        symbol = symbols[_step[0] % len(symbols)]

        if action_int == 0:
            return "HOLD"
        elif action_int == 1:
            return f"BUY {symbol}" if len(symbols) > 1 else "BUY"
        else:
            return f"SELL {symbol}" if len(symbols) > 1 else "SELL"

    return agent_fn
