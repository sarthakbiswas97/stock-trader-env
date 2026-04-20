"""Data contracts for the Stock Trading Environment (OpenEnv-compliant)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


class TradeAction(BaseAction):
    """Action the agent takes each trading day."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    action: str = Field(
        default="HOLD",
        description="One of: BUY, SELL, HOLD. For multi-stock tasks, use 'BUY SYMBOL' or 'SELL SYMBOL'.",
    )


class PositionInfo(BaseModel):
    """Information about a single open position."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl_percent: float
    market_value: float


class MarketObservation(BaseObservation):
    """What the agent sees after each step."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    day: int = Field(default=0, description="Current trading day (1-indexed)")
    total_days: int = Field(default=0, description="Total days in this episode")
    portfolio_value: float = Field(default=0.0, description="Total portfolio value (cash + positions)")
    cash: float = Field(default=0.0, description="Available cash")
    positions: list[PositionInfo] = Field(default_factory=list, description="Open positions")
    market_summary: str = Field(default="", description="Human-readable market state for the LLM agent")
    available_actions: list[str] = Field(default_factory=list, description="Valid actions the agent can take")
    task_id: str = Field(default="", description="Current task: single_stock, portfolio, full_autonomous")
    score: float = Field(default=0.0, description="Current grader score (0.0-1.0)")

    env_version: str = Field(default="", description="Environment version (semver)")
    task_version: str = Field(default="", description="Task-specific version (semver)")


class TradingState(BaseState):
    """Episode metadata."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    task_id: str = Field(default="", description="Current task identifier")
    initial_capital: float = Field(default=0.0, description="Starting capital")
    current_value: float = Field(default=0.0, description="Current portfolio value")
