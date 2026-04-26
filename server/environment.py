"""Stock Trading Environment — OpenEnv-compliant orchestrator.

Delegates to focused modules:
  - portfolio.py: cash, positions, risk tracking
  - action_parser.py: parse BUY/SELL/HOLD strings
  - execution.py: order execution with capacity scaling
  - reward.py: decomposed reward computation
  - observation_builder.py: text observation construction
  - curriculum.py: adaptive difficulty escalation
  - mistake_tracker.py: 7-type error detection
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import TradeAction, MarketObservation, TradingState
from server.action_parser import parse_action
from server.curriculum import CurriculumManager
from server.execution import execute_buy, execute_sell
from server.feature_engine import compute_rsi
from server.market_simulator import MarketSimulator
from server.mistake_tracker import MistakeTracker, MistakeType
from server.neural_simulator import NeuralSimulator
from server.observation_builder import build_observation
from server.portfolio import Portfolio
from server.reward import (
    compute_holding_cost,
    compute_streak_penalty,
    evaluate_hold,
    get_worst_position_pnl,
)
from server.tasks import (
    TASK_CONFIGS,
    grade_full_autonomous,
    grade_portfolio,
    grade_single_stock,
)


class StockTradingEnvironment(Environment[TradeAction, MarketObservation, TradingState]):
    """OpenEnv-compliant stock trading environment.

    Orchestrates the trading loop: observation → action → execution → reward.
    Supports static CSV replay and neural world model simulation.
    Optional curriculum manager for adaptive difficulty escalation.
    """

    def __init__(self):
        super().__init__()
        self._state = TradingState()
        self._sim: MarketSimulator | None = None
        self._portfolio: Portfolio | None = None
        self._task_config: dict = {}
        self._done = False
        self._last_reward = 0.0
        self._current_task = "single_stock"
        self._mistake_tracker = MistakeTracker()
        self._curriculum: CurriculumManager | None = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MarketObservation:
        """Start a new trading episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier
            task_id: Task difficulty tier (default: "single_stock")
            simulator_mode: "replay" (CSV) or "neural" (world model)
            use_curriculum: Enable adaptive difficulty escalation
        """
        use_curriculum = kwargs.get("use_curriculum", False)

        # Curriculum-driven task selection
        if use_curriculum:
            if self._curriculum is None:
                self._curriculum = CurriculumManager()
            task_id = self._curriculum.current_tier
        else:
            task_id = kwargs.get("task_id", "single_stock")

        if task_id not in TASK_CONFIGS:
            task_id = "single_stock"

        simulator_mode = kwargs.get("simulator_mode", "replay")

        self._current_task = task_id
        self._task_config = TASK_CONFIGS[task_id]

        if simulator_mode == "neural":
            self._sim = NeuralSimulator(task_id, seed=seed)
        else:
            self._sim = MarketSimulator(task_id, seed=seed)
        self._sim.reset()
        self._portfolio = Portfolio(self._task_config["initial_capital"])
        self._mistake_tracker.reset_episode()
        self._done = False
        self._last_reward = 0.0

        self._state = TradingState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            current_value=self._task_config["initial_capital"],
        )

        return self._build_observation(0.0)

    def step(
        self,
        action: TradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MarketObservation:
        """Process one trading day."""
        if self._done or self._sim is None or self._portfolio is None:
            return self._build_observation(0.0)

        self._state.step_count += 1
        reward = 0.0

        # Parse action
        parsed = parse_action(action.action, self._task_config)
        action_type = parsed["type"]
        symbol = parsed.get("symbol", "")
        fraction = parsed.get("fraction", 1.0)

        # Get current prices
        prices = {sym: self._sim.get_price(sym) for sym in self._task_config["symbols"]}
        value_before = self._portfolio.get_value(prices)

        # Check regime gate (hard task only)
        regime_blocked = False
        if self._task_config["regime_gate"]:
            breadth = self._sim.get_market_breadth()
            if breadth["market_down"] or breadth["breadth_weak"]:
                regime_blocked = True
                self._portfolio.regime_gated_days += 1
                if action_type in ("BUY", "SELL"):
                    self._portfolio.regime_respected += 0
                    reward -= 0.3
                else:
                    self._portfolio.regime_respected += 1

        # Streak penalty — penalize buying on a losing streak
        reward += compute_streak_penalty(self._portfolio, action_type)

        # Execute action
        if action_type == "BUY" and symbol and not regime_blocked:
            reward += execute_buy(
                self._portfolio, symbol, fraction, prices,
                self._task_config, self._sim.current_day,
            )
        elif action_type == "SELL" and symbol:
            reward += execute_sell(
                self._portfolio, symbol, fraction, prices,
                self._task_config, self._sim.current_day,
            )
        elif action_type == "HOLD":
            reward += evaluate_hold(
                self._portfolio, prices, self._task_config, self._get_rsi,
            )

        # Advance day
        self._sim.advance_day()

        # Get new prices after day advance
        if not self._sim.is_done:
            new_prices = {sym: self._sim.get_price(sym) for sym in self._task_config["symbols"]}
        else:
            new_prices = prices

        # Record daily portfolio value
        self._portfolio.record_daily(new_prices)

        # Position age holding cost — stale positions are expensive
        reward -= compute_holding_cost(self._portfolio, self._sim.current_day)

        # Compute step reward
        value_after = self._portfolio.get_value(new_prices)
        pnl_reward = (value_after - value_before) / value_before * 10 if value_before > 0 else 0.0
        reward += pnl_reward

        # Risk discipline bonus
        total_exposure = sum(
            pos["qty"] * new_prices.get(sym, pos["avg_price"])
            for sym, pos in self._portfolio.positions.items()
        )
        exposure_pct = total_exposure / value_after if value_after > 0 else 0
        if exposure_pct <= self._task_config["max_position_pct"] and len(self._portfolio.positions) > 0:
            reward += 0.02

        self._last_reward = round(reward, 4)

        # Detect trading mistakes and apply penalties
        rsi = self._get_rsi(symbol) if symbol else None
        worst_pnl = get_worst_position_pnl(self._portfolio, new_prices)
        detected = self._mistake_tracker.detect_mistakes(
            day=self._sim.current_day,
            action_type=action_type,
            symbol=symbol or "",
            rsi=rsi,
            regime_blocked=regime_blocked,
            position_pnl=worst_pnl,
            trades_today=self._portfolio.trades_today,
            max_trades=self._task_config["max_trades_per_day"],
            exposure_pct=exposure_pct,
            max_exposure=self._task_config["max_position_pct"],
        )

        for mistake in detected:
            if mistake.type == MistakeType.OVERBOUGHT_BUY:
                reward -= 0.15
            elif mistake.type == MistakeType.OVERSOLD_SELL:
                reward -= 0.15

        if self._sim.is_done:
            self._done = True
            # Record score for curriculum progression
            if self._curriculum is not None:
                score = self._compute_score()
                self._curriculum.record_score(score)

        self._state.current_value = round(value_after, 2)

        return self._build_observation(self._last_reward)

    @property
    def state(self) -> TradingState:
        return self._state

    @property
    def mistake_tracker(self) -> MistakeTracker:
        return self._mistake_tracker

    # --- Private helpers ---

    def _get_rsi(self, symbol: str) -> float | None:
        if not self._sim or not symbol:
            return None
        try:
            lookback = self._sim.get_lookback_data(symbol)
            return compute_rsi(lookback["close"])
        except (IndexError, KeyError, ValueError):
            return None

    def _build_observation(self, reward: float) -> MarketObservation:
        """Delegate to observation_builder."""
        score = self._compute_score()
        return build_observation(
            sim=self._sim,
            portfolio=self._portfolio,
            config=self._task_config,
            task_id=self._current_task,
            score=score,
            reward=reward,
            done=self._done,
        )

    def _compute_score(self) -> float:
        """Compute current grader score."""
        if self._portfolio is None or self._sim is None:
            return 0.0

        config = self._task_config
        prices = {sym: self._sim.get_price(sym) for sym in config["symbols"]}
        final_value = self._portfolio.get_value(prices)
        initial = config["initial_capital"]

        if self._current_task in ("single_stock", "single_stock_costs"):
            bh_return = 0.0
            if self._done and len(self._portfolio.daily_values) > 1:
                first_val = self._portfolio.daily_values[0]
                bh_return = (final_value - first_val) / first_val
            return grade_single_stock(final_value, initial, bh_return, self._portfolio.total_trades)

        elif self._current_task in ("portfolio", "multi_stock_3"):
            return grade_portfolio(
                final_value, initial,
                self._portfolio.daily_returns,
                self._portfolio.risk_violations,
                self._portfolio.total_trades,
            )

        elif self._current_task == "full_autonomous":
            return grade_full_autonomous(
                final_value, initial,
                self._portfolio.daily_returns,
                self._portfolio.risk_violations,
                self._portfolio.regime_gated_days,
                self._portfolio.regime_respected,
                self._portfolio.total_trades,
                self._portfolio.max_drawdown,
            )

        return 0.0
