"""Tests for the Gymnasium wrapper and training infrastructure."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from training.data_splits import SPLITS, get_valid_index_range
from training.observations import obs_to_text, obs_to_numeric, NUMERIC_OBS_SIZE
from training.trajectory_logger import TrajectoryLogger
from training.gym_wrapper import StockTradingGymEnv
from server.market_simulator import _load_stock_data


# --- Data Splits ---


class TestDataSplits:
    """Date-based split definitions and index range computation."""

    def test_all_splits_defined(self):
        assert set(SPLITS.keys()) == {"train", "val", "test"}

    def test_splits_are_non_overlapping(self):
        train, val, test = SPLITS["train"], SPLITS["val"], SPLITS["test"]
        assert train.end_date < val.start_date
        assert val.end_date < test.start_date

    def test_splits_cover_data_range(self):
        """Splits should span from 2020 to 2026."""
        assert SPLITS["train"].start_date.year == 2020
        assert SPLITS["test"].end_date.year == 2026

    def test_get_valid_index_range_train(self):
        df = _load_stock_data("RELIANCE")
        ts = df["timestamp"]
        min_start, max_start = get_valid_index_range(
            ts, SPLITS["train"], lookback=50, episode_days=20,
        )
        assert min_start >= 0
        assert max_start >= min_start
        # Verify the trading window falls within train dates
        trading_start = ts.iloc[min_start + 50]
        trading_end = ts.iloc[max_start + 50 + 19]
        assert trading_start >= SPLITS["train"].start_date
        assert trading_end <= SPLITS["train"].end_date

    def test_get_valid_index_range_val(self):
        df = _load_stock_data("RELIANCE")
        ts = df["timestamp"]
        min_start, max_start = get_valid_index_range(
            ts, SPLITS["val"], lookback=50, episode_days=20,
        )
        assert max_start >= min_start

    def test_splits_are_frozen(self):
        """DataSplit should be immutable."""
        with pytest.raises(AttributeError):
            SPLITS["train"].name = "hacked"


# --- Observations ---


class TestObservations:
    """Observation conversion for text and numeric modes."""

    @pytest.fixture
    def sample_obs(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        _, info = env.reset()
        # Get raw observation from internal env
        return env._env._build_observation(0.0)

    def test_text_obs_is_string(self, sample_obs):
        text = obs_to_text(sample_obs)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_numeric_obs_shape(self, sample_obs):
        vec = obs_to_numeric(sample_obs, 100_000)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (NUMERIC_OBS_SIZE,)
        assert vec.dtype == np.float32

    def test_numeric_obs_normalized(self, sample_obs):
        vec = obs_to_numeric(sample_obs, 100_000)
        # Day progress should be between 0 and 1
        assert 0 <= vec[0] <= 1
        # Cash ratio should be around 1.0 at start (no trades yet)
        assert 0.9 <= vec[1] <= 1.1
        # Portfolio value ratio should be around 1.0 at start
        assert 0.9 <= vec[2] <= 1.1


# --- Trajectory Logger ---


TEMP_DIR = Path(__file__).parent / "_tmp_trajectories"


class TestTrajectoryLogger:
    """JSONL trajectory logging."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    def test_creates_file(self):
        with TrajectoryLogger("test_task", "train", "ep001", TEMP_DIR) as tl:
            tl.log_reset("initial obs")
        assert tl.path.exists()

    def test_logs_reset_and_steps(self):
        with TrajectoryLogger("test_task", "train", "ep001", TEMP_DIR) as tl:
            tl.log_reset("initial obs")
            tl.log_step("obs1", "BUY", 0.1, False)
            tl.log_step("obs2", "HOLD", 0.05, True, {"score": 0.7})

        lines = tl.path.read_text().strip().split("\n")
        assert len(lines) == 3

        reset_entry = json.loads(lines[0])
        assert reset_entry["step"] == 0
        assert reset_entry["action"] is None

        step_entry = json.loads(lines[1])
        assert step_entry["step"] == 1
        assert step_entry["action"] == "BUY"

        final_entry = json.loads(lines[2])
        assert final_entry["done"] is True
        assert final_entry["info"]["score"] == 0.7


# --- Gym Wrapper ---


class TestGymWrapper:
    """Full Gymnasium-compatible wrapper."""

    def test_text_mode_reset(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert "Day 1" in obs
        assert info["day"] == 1
        env.close()

    def test_numeric_mode_reset(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="numeric")
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (NUMERIC_OBS_SIZE,)
        env.close()

    def test_step_returns_gym_tuple(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        env.reset()
        result = env.step("HOLD")
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert isinstance(info, dict)
        env.close()

    def test_full_episode(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step("HOLD")
            total_reward += reward
            steps += 1
            if terminated:
                break
        assert steps == 20
        assert 0 < info["score"] < 1
        env.close()

    def test_split_constrains_dates(self):
        """Train split should produce different episodes than test split."""
        env_train = StockTradingGymEnv(
            task_id="single_stock", seed=42, obs_mode="text", split="train",
        )
        env_test = StockTradingGymEnv(
            task_id="single_stock", seed=42, obs_mode="text", split="test",
        )
        obs_train, _ = env_train.reset()
        obs_test, _ = env_test.reset()
        # Different splits should produce different market data
        assert obs_train != obs_test
        env_train.close()
        env_test.close()

    def test_trajectory_logging(self):
        env = StockTradingGymEnv(
            task_id="single_stock", seed=42,
            obs_mode="text", log_trajectories=True,
        )
        env.reset()
        env.step("BUY")
        env.step("HOLD")
        env.step("SELL")
        # Run to completion
        while True:
            _, _, terminated, _, _ = env.step("HOLD")
            if terminated:
                break
        env.close()

        # Check trajectory files exist
        traj_dir = Path(__file__).parent.parent / "data" / "trajectories"
        assert traj_dir.exists()
        jsonl_files = list(traj_dir.rglob("*.jsonl"))
        assert len(jsonl_files) >= 1

        # Verify content
        content = jsonl_files[0].read_text().strip().split("\n")
        assert len(content) > 0
        first = json.loads(content[0])
        assert "observation" in first

        # Cleanup
        shutil.rmtree(traj_dir / "single_stock")

    def test_portfolio_task(self):
        env = StockTradingGymEnv(task_id="portfolio", seed=42, obs_mode="text")
        obs, info = env.reset()
        assert info["total_days"] == 30
        obs, reward, terminated, truncated, info = env.step("BUY RELIANCE")
        assert info["num_positions"] >= 1
        env.close()

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            StockTradingGymEnv(task_id="nonexistent")

    def test_invalid_obs_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown obs_mode"):
            StockTradingGymEnv(obs_mode="invalid")

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Unknown split"):
            StockTradingGymEnv(split="future")

    def test_deterministic_with_seed(self):
        env1 = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        env2 = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert obs1 == obs2
        env1.close()
        env2.close()

    def test_info_contains_version(self):
        env = StockTradingGymEnv(task_id="single_stock", seed=42, obs_mode="text")
        _, info = env.reset()
        assert "env_version" in info
        assert "task_version" in info
        assert info["env_version"] == "1.1.0"
        env.close()
