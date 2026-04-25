"""Stock Trading Environment — core OpenEnv implementation."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import TradeAction, MarketObservation, TradingState
from server.market_simulator import MarketSimulator
from server.neural_simulator import NeuralSimulator
from server.feature_engine import compute_all_features, compute_rsi, features_to_text
from server.mistake_tracker import MistakeTracker, MistakeType
from server.macro_data import macro_to_text
from server.action_parser import parse_action
from server.execution import execute_buy, execute_sell
from server.portfolio import Portfolio
from server.reward import evaluate_hold, compute_holding_cost, compute_streak_penalty, get_worst_position_pnl
from server.tasks import TASK_CONFIGS, grade_single_stock, grade_portfolio, grade_full_autonomous
from server import __version__


class StockTradingEnvironment(Environment[TradeAction, MarketObservation, TradingState]):
    """
    OpenEnv-compliant stock trading environment.

    Implements reset(), step(), and state property.
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

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MarketObservation:
        """Start a new trading episode."""
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
            step_count=0,
            task_id=task_id,
            initial_capital=self._task_config["initial_capital"],
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
                    self._portfolio.regime_respected += 0  # Violated
                    reward -= 0.3  # Penalty for trading during regime gate
                else:
                    self._portfolio.regime_respected += 1  # Respected

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
            reward += 0.02  # Small bonus for staying within limits (only with active positions)

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

        # Penalize mistakes not already handled by action execution
        for mistake in detected:
            if mistake.type == MistakeType.OVERBOUGHT_BUY:
                reward -= 0.15
            elif mistake.type == MistakeType.OVERSOLD_SELL:
                reward -= 0.15

        if self._sim.is_done:
            self._done = True

        self._state.current_value = round(value_after, 2)

        return self._build_observation(self._last_reward)

    @property
    def state(self) -> TradingState:
        return self._state

    @property
    def mistake_tracker(self) -> MistakeTracker:
        return self._mistake_tracker

    def _get_rsi(self, symbol: str) -> float | None:
        if not self._sim or not symbol:
            return None
        try:
            lookback = self._sim.get_lookback_data(symbol)
            return compute_rsi(lookback["close"])
        except (IndexError, KeyError, ValueError):
            return None


    # --- Private methods ---


    def _build_observation(self, reward: float) -> MarketObservation:
        """Build the observation for the current state."""
        sim = self._sim
        portfolio = self._portfolio
        config = self._task_config

        if sim is None or portfolio is None:
            return MarketObservation(
                done=True, reward=0.0, day=0, total_days=0,
                portfolio_value=0, cash=0, positions=[],
                market_summary="No episode active.", available_actions=["HOLD"],
                task_id="", score=0.0,
                env_version=__version__,
                task_version="",
            )

        prices = {}
        for sym in config["symbols"]:
            try:
                prices[sym] = sim.get_price(sym)
            except (IndexError, KeyError):
                prices[sym] = 0.0

        value = portfolio.get_value(prices)
        positions = portfolio.get_position_info(prices)

        # Build market summary text
        summary_lines = []
        summary_lines.append(
            f"Day {sim.current_day + 1} of {config['episode_days']} | "
            f"Cash: Rs{portfolio.cash:,.0f} | Portfolio: Rs{value:,.0f} | "
            f"Return: {(value - config['initial_capital']) / config['initial_capital'] * 100:+.1f}%"
        )

        # Regime gate warning (hard task)
        if config["regime_gate"] and not self._done:
            breadth = sim.get_market_breadth()
            if breadth["market_down"] or breadth["breadth_weak"]:
                summary_lines.append(
                    f"\nWARNING: REGIME GATE ACTIVE — Market avg {breadth['avg_change']:+.1f}%, "
                    f"{breadth['declining']}/{len(config['symbols'])} stocks declining. "
                    f"Trading restricted — HOLD recommended."
                )

        summary_lines.append("")

        # Macro context (VIX, USD/INR, crude, sectors, RBI rate)
        macro_snap = sim.get_macro_snapshot_data()
        if macro_snap:
            summary_lines.append(macro_to_text(macro_snap))
            summary_lines.append("")

        # Stock data with features
        for sym in config["symbols"]:
            try:
                lookback = sim.get_lookback_data(sym)
                features = compute_all_features(lookback)
                price = prices[sym]
                change = sim.get_daily_change(sym)
                summary_lines.append(features_to_text(sym, price, change, features))
            except (IndexError, KeyError, ValueError):
                summary_lines.append(f"{sym}: Data unavailable")

        # Position summary
        if positions:
            summary_lines.append("\nYour Positions:")
            for p in positions:
                summary_lines.append(
                    f"  {p.symbol}: {p.quantity} shares @ Rs{p.avg_price:,.0f} | "
                    f"Current: Rs{p.current_price:,.0f} | P&L: {p.pnl_percent:+.1f}%"
                )

        # Constraints reminder (medium/hard)
        if config["transaction_cost"] > 0:
            summary_lines.append(
                f"\nConstraints: {config['transaction_cost']*100:.1f}% cost + "
                f"{config['slippage']*100:.1f}% slippage per trade | "
                f"Max {config['position_limit_per_stock']*100:.0f}% per stock | "
                f"Trades today: {portfolio.trades_today}/{config['max_trades_per_day']}"
            )

        # Available actions
        available = ["HOLD"]
        if not self._done:
            for sym in config["symbols"]:
                if portfolio.cash > prices.get(sym, float("inf")):
                    if len(config["symbols"]) == 1:
                        available.append("BUY")
                    else:
                        available.append(f"BUY {sym}")
                if sym in portfolio.positions:
                    if len(config["symbols"]) == 1:
                        available.append("SELL")
                    else:
                        available.append(f"SELL {sym}")

        # Compute current score
        score = self._compute_score()

        return MarketObservation(
            done=self._done,
            reward=round(reward, 4),
            day=sim.current_day + 1,
            total_days=config["episode_days"],
            portfolio_value=round(value, 2),
            cash=round(portfolio.cash, 2),
            positions=positions,
            market_summary="\n".join(summary_lines),
            available_actions=list(set(available)),
            task_id=self._current_task,
            score=round(score, 4),
            env_version=__version__,
            task_version=config["version"],
        )

    def _compute_score(self) -> float:
        """Compute current grader score."""
        if self._portfolio is None or self._sim is None:
            return 0.0

        config = self._task_config
        prices = {sym: self._sim.get_price(sym) for sym in config["symbols"]}
        final_value = self._portfolio.get_value(prices)
        initial = config["initial_capital"]

        if self._current_task == "single_stock":
            bh_return = 0.0
            if self._done and len(self._portfolio.daily_values) > 1:
                first_val = self._portfolio.daily_values[0]
                bh_return = (final_value - first_val) / first_val
            return grade_single_stock(final_value, initial, bh_return, self._portfolio.total_trades)

        elif self._current_task == "portfolio":
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
