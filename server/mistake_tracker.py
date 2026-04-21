"""Tracks trading mistakes for analysis and training signal.

Detects common errors (regime violations, overbought buys, holding losers)
and exposes them via the environment info dict. Accumulated counts are
logged to MLflow for learning curve visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MistakeType(str, Enum):
    REGIME_VIOLATION = "regime_violation"
    OVERBOUGHT_BUY = "overbought_buy"
    OVERSOLD_SELL = "oversold_sell"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    TRADE_LIMIT_BREACH = "trade_limit_breach"
    LOSS_HOLD = "loss_hold"
    MISSED_OPPORTUNITY = "missed_opportunity"


@dataclass
class Mistake:
    """Single mistake record."""

    type: MistakeType
    day: int
    symbol: str
    action: str
    detail: str


@dataclass
class MistakeTracker:
    """Tracks mistakes within an episode and across episodes."""

    _episode_mistakes: list[Mistake] = field(default_factory=list, repr=False)
    _cumulative_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _total_episodes: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        for mt in MistakeType:
            self._cumulative_counts.setdefault(mt.value, 0)

    def reset_episode(self) -> None:
        """Clear episode mistakes for a new episode."""
        self._episode_mistakes = []
        self._total_episodes += 1

    def record(self, mistake_type: MistakeType, day: int, symbol: str, action: str, detail: str) -> None:
        mistake = Mistake(type=mistake_type, day=day, symbol=symbol, action=action, detail=detail)
        self._episode_mistakes.append(mistake)
        self._cumulative_counts[mistake_type.value] += 1

    def detect_mistakes(
        self,
        day: int,
        action_type: str,
        symbol: str,
        rsi: float | None,
        regime_blocked: bool,
        position_pnl: float | None,
        trades_today: int,
        max_trades: int,
        exposure_pct: float,
        max_exposure: float,
    ) -> list[Mistake]:
        """Detect mistakes from the current step context."""
        detected: list[Mistake] = []

        if regime_blocked and action_type in ("BUY", "SELL"):
            self.record(MistakeType.REGIME_VIOLATION, day, symbol, action_type, "Traded during regime gate")
            detected.append(self._episode_mistakes[-1])

        if action_type == "BUY" and rsi is not None and rsi > 70:
            self.record(MistakeType.OVERBOUGHT_BUY, day, symbol, action_type, f"Bought at RSI {rsi:.0f}")
            detected.append(self._episode_mistakes[-1])

        if action_type == "SELL" and rsi is not None and rsi < 30:
            self.record(MistakeType.OVERSOLD_SELL, day, symbol, action_type, f"Sold at RSI {rsi:.0f}")
            detected.append(self._episode_mistakes[-1])

        if trades_today >= max_trades and action_type in ("BUY", "SELL"):
            self.record(MistakeType.TRADE_LIMIT_BREACH, day, symbol, action_type, f"Trade limit {max_trades} reached")
            detected.append(self._episode_mistakes[-1])

        if exposure_pct > max_exposure and action_type == "BUY":
            self.record(MistakeType.POSITION_LIMIT_BREACH, day, symbol, action_type, f"Exposure {exposure_pct:.0%} > {max_exposure:.0%}")
            detected.append(self._episode_mistakes[-1])

        if action_type == "HOLD" and position_pnl is not None and position_pnl < -5.0:
            self.record(MistakeType.LOSS_HOLD, day, symbol, "HOLD", f"Holding at {position_pnl:+.1f}% P&L")
            detected.append(self._episode_mistakes[-1])

        if action_type == "HOLD" and rsi is not None and (rsi < 25 or rsi > 75):
            self.record(MistakeType.MISSED_OPPORTUNITY, day, symbol, "HOLD", f"No action at RSI {rsi:.0f}")
            detected.append(self._episode_mistakes[-1])

        return detected

    @property
    def episode_mistakes(self) -> list[Mistake]:
        return list(self._episode_mistakes)

    @property
    def episode_count(self) -> int:
        return len(self._episode_mistakes)

    @property
    def episode_counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {mt.value: 0 for mt in MistakeType}
        for m in self._episode_mistakes:
            counts[m.type.value] += 1
        return counts

    @property
    def cumulative_counts(self) -> dict[str, int]:
        return dict(self._cumulative_counts)

    def avg_mistakes_per_episode(self) -> float:
        total = sum(self._cumulative_counts.values())
        return total / max(self._total_episodes, 1)
