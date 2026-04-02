"""
Client for the Stock Trading Environment.
Connects to the deployed environment via WebSocket using openenv-core.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import TradeAction, MarketObservation, TradingState, PositionInfo


class StockTraderClient(EnvClient[TradeAction, MarketObservation, TradingState]):
    """Typed client for the Stock Trading Environment."""

    def _step_payload(self, action: TradeAction) -> Dict[str, Any]:
        """Convert TradeAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MarketObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)

        # Parse positions
        positions = [
            PositionInfo(**p) for p in obs_data.get("positions", [])
        ]

        observation = MarketObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            day=obs_data.get("day", 0),
            total_days=obs_data.get("total_days", 0),
            portfolio_value=obs_data.get("portfolio_value", 0.0),
            cash=obs_data.get("cash", 0.0),
            positions=positions,
            market_summary=obs_data.get("market_summary", ""),
            available_actions=obs_data.get("available_actions", ["HOLD"]),
            task_id=obs_data.get("task_id", ""),
            score=obs_data.get("score", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TradingState:
        """Parse state response."""
        return TradingState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            initial_capital=payload.get("initial_capital", 0.0),
            current_value=payload.get("current_value", 0.0),
        )
