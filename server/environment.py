"""Stock Trading Environment — core OpenEnv implementation."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import TradeAction, MarketObservation, TradingState, PositionInfo
from server.market_simulator import MarketSimulator
from server.neural_simulator import NeuralSimulator
from server.feature_engine import compute_all_features, compute_rsi, features_to_text
from server.mistake_tracker import MistakeTracker, MistakeType
from server.macro_data import macro_to_text
from server.tasks import TASK_CONFIGS, grade_single_stock, grade_portfolio, grade_full_autonomous
from server import __version__


class Portfolio:
    """Tracks cash, positions, and trade history."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, dict] = {}  # {symbol: {qty, avg_price}}
        self.trade_log: list[dict] = []
        self.daily_values: list[float] = [initial_capital]
        self.daily_returns: list[float] = []
        self.risk_violations: int = 0
        self.regime_gated_days: int = 0
        self.regime_respected: int = 0
        self.trades_today: int = 0
        self.total_trades: int = 0
        self.peak_value: float = initial_capital
        self.max_drawdown: float = 0.0

    def get_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value."""
        position_value = sum(
            pos["qty"] * prices.get(sym, pos["avg_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value

    def get_position_info(self, prices: dict[str, float]) -> list[PositionInfo]:
        """Get position details for observation."""
        result = []
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos["avg_price"])
            pnl_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] > 0 else 0.0
            result.append(PositionInfo(
                symbol=sym,
                quantity=pos["qty"],
                avg_price=round(pos["avg_price"], 2),
                current_price=round(price, 2),
                pnl_percent=round(pnl_pct, 2),
                market_value=round(pos["qty"] * price, 2),
            ))
        return result

    def record_daily(self, prices: dict[str, float]) -> None:
        """Record end-of-day portfolio value."""
        value = self.get_value(prices)
        if len(self.daily_values) > 0:
            prev = self.daily_values[-1]
            daily_ret = (value - prev) / prev if prev > 0 else 0.0
            self.daily_returns.append(daily_ret)
        self.daily_values.append(value)
        self.trades_today = 0

        # Track drawdown
        if value > self.peak_value:
            self.peak_value = value
        dd = (self.peak_value - value) / self.peak_value
        if dd > self.max_drawdown:
            self.max_drawdown = dd


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
        parsed = self._parse_action(action.action)
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

        # Execute action
        if action_type == "BUY" and symbol and not regime_blocked:
            reward += self._execute_buy(symbol, fraction, prices)
        elif action_type == "SELL" and symbol:
            reward += self._execute_sell(symbol, fraction, prices)
        elif action_type == "HOLD":
            reward += self._evaluate_hold(prices)

        # Advance day
        self._sim.advance_day()

        # Get new prices after day advance
        if not self._sim.is_done:
            new_prices = {sym: self._sim.get_price(sym) for sym in self._task_config["symbols"]}
        else:
            new_prices = prices

        # Record daily portfolio value
        self._portfolio.record_daily(new_prices)

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
        worst_pnl = self._get_worst_position_pnl(new_prices)
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

    def _get_worst_position_pnl(self, prices: dict[str, float]) -> float | None:
        if not self._portfolio or not self._portfolio.positions:
            return None
        worst = 0.0
        for sym, pos in self._portfolio.positions.items():
            price = prices.get(sym, pos["avg_price"])
            pnl = (price - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] > 0 else 0.0
            worst = min(worst, pnl)
        return worst

    def _evaluate_hold(self, prices: dict[str, float]) -> float:
        """Evaluate HOLD quality based on market signals and portfolio state.

        Returns a small reward/penalty:
          - Missed opportunity (strong signal ignored): -0.15
          - Holding a losing position: -0.1
          - Justified patience (no clear signal): +0.01
        """
        has_positions = bool(self._portfolio and self._portfolio.positions)
        worst_pnl = self._get_worst_position_pnl(prices)

        # Check RSI across all task symbols for actionable signals
        extreme_rsi = False
        for sym in self._task_config["symbols"]:
            rsi = self._get_rsi(sym)
            if rsi is not None and (rsi < 25 or rsi > 75):
                extreme_rsi = True
                break

        # Holding a losing position (> 5% drawdown) — should cut losses
        if has_positions and worst_pnl is not None and worst_pnl < -5.0:
            return -0.1

        # Strong signal ignored — missed opportunity
        if extreme_rsi:
            return -0.15

        # No strong signal — justified patience
        return 0.01

    # --- Private methods ---

    def _parse_action(self, action_str: str) -> dict:
        """Parse action string like 'BUY RELIANCE 0.3' or 'HOLD'."""
        parts = action_str.strip().upper().split()
        if not parts:
            return {"type": "HOLD"}

        action_type = parts[0]
        if action_type not in ("BUY", "SELL", "HOLD"):
            return {"type": "HOLD"}

        if action_type == "HOLD":
            return {"type": "HOLD"}

        # For single_stock task, default symbol
        if len(parts) == 1 and len(self._task_config["symbols"]) == 1:
            symbol = self._task_config["symbols"][0]
        elif len(parts) >= 2:
            symbol = parts[1]
        else:
            return {"type": "HOLD"}

        # Validate symbol
        if symbol not in self._task_config["symbols"]:
            return {"type": "HOLD"}

        fraction = 1.0
        if len(parts) >= 3:
            try:
                fraction = float(parts[2])
                fraction = max(0.0, min(1.0, fraction))
            except ValueError:
                fraction = 1.0

        return {"type": action_type, "symbol": symbol, "fraction": fraction}

    def _execute_buy(self, symbol: str, fraction: float, prices: dict) -> float:
        """Execute a buy order. Returns reward adjustment."""
        config = self._task_config
        portfolio = self._portfolio
        price = prices[symbol]

        # Check trade limit
        if portfolio.trades_today >= config["max_trades_per_day"]:
            portfolio.risk_violations += 1
            return -0.1

        # Check position limit per stock
        current_value = portfolio.get_value(prices)
        existing_value = 0
        if symbol in portfolio.positions:
            existing_value = portfolio.positions[symbol]["qty"] * price
        max_for_stock = current_value * config["position_limit_per_stock"]
        available_for_stock = max(0, max_for_stock - existing_value)

        # Calculate order size
        buy_amount = min(portfolio.cash * fraction, available_for_stock)
        if buy_amount < price:
            return 0.0  # Can't afford even 1 share

        # Apply slippage and costs
        effective_price = price * (1 + config["slippage"])
        cost = buy_amount * config["transaction_cost"]
        qty = int((buy_amount - cost) / effective_price)
        if qty <= 0:
            return 0.0

        total_cost = qty * effective_price + cost
        portfolio.cash -= total_cost

        if symbol in portfolio.positions:
            old = portfolio.positions[symbol]
            new_qty = old["qty"] + qty
            new_avg = (old["qty"] * old["avg_price"] + qty * effective_price) / new_qty
            portfolio.positions[symbol] = {"qty": new_qty, "avg_price": new_avg}
        else:
            portfolio.positions[symbol] = {"qty": qty, "avg_price": effective_price}

        portfolio.trades_today += 1
        portfolio.total_trades += 1
        portfolio.trade_log.append({
            "day": self._sim.current_day,
            "action": "BUY",
            "symbol": symbol,
            "qty": qty,
            "price": effective_price,
        })

        return 0.0  # Neutral — reward comes from P&L

    def _execute_sell(self, symbol: str, fraction: float, prices: dict) -> float:
        """Execute a sell order. Fraction controls how much of the position to sell."""
        config = self._task_config
        portfolio = self._portfolio

        if symbol not in portfolio.positions:
            return -0.05  # Penalty for trying to sell what you don't own

        pos = portfolio.positions[symbol]
        price = prices[symbol]

        # Compute quantity to sell
        sell_qty = int(pos["qty"] * fraction)
        if sell_qty <= 0:
            return 0.0  # Nothing to sell at this fraction

        # Apply slippage and costs
        effective_price = price * (1 - config["slippage"])
        proceeds = sell_qty * effective_price
        cost = proceeds * config["transaction_cost"]
        net_proceeds = proceeds - cost

        portfolio.cash += net_proceeds

        # Calculate trade P&L for logging
        trade_pnl = (effective_price - pos["avg_price"]) / pos["avg_price"] if pos["avg_price"] > 0 else 0.0

        portfolio.trade_log.append({
            "day": self._sim.current_day,
            "action": "SELL",
            "symbol": symbol,
            "qty": sell_qty,
            "price": effective_price,
            "pnl_pct": round(trade_pnl * 100, 2),
        })

        # Update or remove position
        remaining = pos["qty"] - sell_qty
        if remaining > 0:
            portfolio.positions[symbol] = {"qty": remaining, "avg_price": pos["avg_price"]}
        else:
            del portfolio.positions[symbol]

        portfolio.trades_today += 1
        portfolio.total_trades += 1

        return 0.0

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
