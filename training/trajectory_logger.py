"""Episode trajectory logger — writes step-by-step data to JSONL files.

Used for:
    1. SFT data collection (observation, action pairs from expert agents)
    2. Debugging training runs (replay episodes to see agent behavior)
    3. Offline RL (train on logged data without running the environment)

Format: one JSON object per line, one file per episode.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"


class TrajectoryLogger:
    """Logs a single episode's trajectory to a JSONL file."""

    def __init__(
        self,
        task_id: str,
        split: str,
        episode_id: str,
        output_dir: Path | None = None,
    ):
        self._dir = output_dir or TRAJECTORY_DIR / task_id / split
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{episode_id}.jsonl"
        self._file = open(self._path, "w")
        self._step = 0

    def log_reset(self, observation: str) -> None:
        """Log the initial observation after reset."""
        self._write({
            "step": 0,
            "observation": observation,
            "action": None,
            "reward": 0.0,
            "done": False,
        })

    def log_step(
        self,
        observation: str,
        action: str,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Log one step of the episode."""
        self._step += 1
        entry: dict[str, Any] = {
            "step": self._step,
            "observation": observation,
            "action": action,
            "reward": round(reward, 4),
            "done": done,
        }
        if info:
            # Include score and other metadata on terminal step
            entry["info"] = {
                k: v for k, v in info.items()
                if isinstance(v, (int, float, str, bool))
            }
        self._write(entry)

    def close(self) -> None:
        """Flush and close the trajectory file."""
        if not self._file.closed:
            self._file.close()
            logger.debug("Trajectory saved to %s", self._path)

    @property
    def path(self) -> Path:
        return self._path

    def _write(self, data: dict) -> None:
        self._file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
