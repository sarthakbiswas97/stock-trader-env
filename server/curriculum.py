"""Adaptive curriculum manager for automatic difficulty escalation.

Tracks agent performance and promotes/demotes through difficulty tiers
based on rolling score windows. The agent drives its own capability
growth — no manual intervention needed.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from server.tasks import CURRICULUM_ORDER

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierConfig:
    """Promotion and demotion thresholds for a difficulty tier."""

    promote_score: float
    promote_window: int
    demote_score: float
    demote_window: int


TIER_THRESHOLDS: dict[str, TierConfig] = {
    "single_stock": TierConfig(
        promote_score=0.60, promote_window=5,
        demote_score=0.0, demote_window=0,
    ),
    "single_stock_costs": TierConfig(
        promote_score=0.55, promote_window=5,
        demote_score=0.30, demote_window=3,
    ),
    "multi_stock_3": TierConfig(
        promote_score=0.50, promote_window=5,
        demote_score=0.25, demote_window=3,
    ),
    "portfolio": TierConfig(
        promote_score=0.50, promote_window=5,
        demote_score=0.25, demote_window=3,
    ),
    "full_autonomous": TierConfig(
        promote_score=1.0, promote_window=0,
        demote_score=0.20, demote_window=3,
    ),
}


@dataclass
class Transition:
    """Record of a tier promotion or demotion."""

    episode: int
    from_tier: str
    to_tier: str
    direction: str
    trigger_score: float


@dataclass
class CurriculumManager:
    """Tracks agent scores and manages automatic difficulty progression.

    Usage:
        cm = CurriculumManager()
        task_id = cm.current_tier           # "single_stock"
        cm.record_score(0.65)               # Record episode score
        task_id = cm.current_tier           # "single_stock_costs" (promoted!)
    """

    current_tier: str = field(default_factory=lambda: CURRICULUM_ORDER[0])
    _scores: dict[str, deque] = field(default_factory=dict, repr=False)
    _episode_count: int = field(default=0, repr=False)
    _transitions: list[Transition] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        max_window = max(t.promote_window for t in TIER_THRESHOLDS.values())
        max_window = max(max_window, max(t.demote_window for t in TIER_THRESHOLDS.values()))
        for tier in CURRICULUM_ORDER:
            self._scores[tier] = deque(maxlen=max_window)

    def record_score(self, score: float) -> str | None:
        """Record an episode score and check for promotion/demotion.

        Returns the new tier name if a transition occurred, None otherwise.
        """
        self._episode_count += 1
        self._scores[self.current_tier].append(score)

        transition = self._check_promotion() or self._check_demotion()
        if transition:
            self._transitions.append(transition)
            logger.info(
                f"Curriculum {transition.direction}: "
                f"{transition.from_tier} → {transition.to_tier} "
                f"(episode {transition.episode}, score {transition.trigger_score:.3f})"
            )
            self.current_tier = transition.to_tier
            return transition.to_tier

        return None

    def _check_promotion(self) -> Transition | None:
        tier_idx = CURRICULUM_ORDER.index(self.current_tier)
        if tier_idx >= len(CURRICULUM_ORDER) - 1:
            return None

        config = TIER_THRESHOLDS[self.current_tier]
        scores = self._scores[self.current_tier]

        if len(scores) < config.promote_window:
            return None

        recent = list(scores)[-config.promote_window:]
        mean_score = sum(recent) / len(recent)

        if mean_score >= config.promote_score:
            next_tier = CURRICULUM_ORDER[tier_idx + 1]
            return Transition(
                episode=self._episode_count,
                from_tier=self.current_tier,
                to_tier=next_tier,
                direction="promotion",
                trigger_score=mean_score,
            )

        return None

    def _check_demotion(self) -> Transition | None:
        tier_idx = CURRICULUM_ORDER.index(self.current_tier)
        if tier_idx <= 0:
            return None

        config = TIER_THRESHOLDS[self.current_tier]
        scores = self._scores[self.current_tier]

        if config.demote_window == 0 or len(scores) < config.demote_window:
            return None

        recent = list(scores)[-config.demote_window:]
        mean_score = sum(recent) / len(recent)

        if mean_score < config.demote_score:
            prev_tier = CURRICULUM_ORDER[tier_idx - 1]
            return Transition(
                episode=self._episode_count,
                from_tier=self.current_tier,
                to_tier=prev_tier,
                direction="demotion",
                trigger_score=mean_score,
            )

        return None

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def tier_index(self) -> int:
        return CURRICULUM_ORDER.index(self.current_tier)

    def summary(self) -> dict:
        """Summary for logging and visualization."""
        return {
            "current_tier": self.current_tier,
            "tier_index": self.tier_index,
            "episode_count": self._episode_count,
            "promotions": sum(1 for t in self._transitions if t.direction == "promotion"),
            "demotions": sum(1 for t in self._transitions if t.direction == "demotion"),
            "transitions": [
                {
                    "episode": t.episode,
                    "from": t.from_tier,
                    "to": t.to_tier,
                    "direction": t.direction,
                    "score": t.trigger_score,
                }
                for t in self._transitions
            ],
        }
