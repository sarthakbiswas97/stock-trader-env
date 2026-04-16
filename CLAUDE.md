# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An OpenEnv-compliant RL environment that simulates daily stock trading on Indian equity markets (NIFTY stocks) using real historical OHLCV data. An LLM agent connects via HTTP/WebSocket, receives market observations with technical indicators, and responds with trade actions (BUY, SELL, HOLD).

## Commands

```bash
# Run the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run baseline inference (requires OPENAI_API_KEY env var)
OPENAI_API_KEY=... python inference.py

# Install dependencies
pip install -r requirements.txt

# Docker
docker build -t stock-trader-env .
docker run -p 8000:8000 stock-trader-env
```

## Architecture

**Client-Server RL loop:** The server hosts the environment; agents interact via REST (`/reset`, `/step`, `/state`) or WebSocket (`/ws`). Sessions are stored in-memory keyed by UUID.

**Core flow:**
1. `server/app.py` — FastAPI endpoints, session management
2. `server/environment.py` — `StockTradingEnvironment` with `reset()`/`step()` implementing the RL loop. Contains `Portfolio` class for position/cash tracking
3. `server/market_simulator.py` — Replays historical CSV data from `data/ohlcv/`. Picks random start windows (with 50-day lookback buffer for indicators)
4. `server/feature_engine.py` — Computes technical indicators (RSI, MACD, Bollinger, volume spike, trend, momentum, volatility) and converts them to human-readable text for LLM consumption
5. `server/tasks.py` — Task configs and grading functions for 3 difficulty levels

**Data contracts** (`models.py`): `TradeAction` (input), `MarketObservation` (output), `TradingState` (metadata), `PositionInfo` — all Pydantic v2 models.

**Client** (`client.py`): `StockTraderClient` wraps httpx for the REST API.

**Inference** (`inference.py`): Baseline agent using OpenAI API. Runs all 3 tasks sequentially with seed=42.

## Three Task Difficulty Levels

| Task | Stocks | Days | Capital | Key Constraints |
|------|--------|------|---------|-----------------|
| `single_stock` (easy) | RELIANCE | 20 | 100K | None — no costs, no limits |
| `portfolio` (medium) | 5 stocks | 30 | 200K | 0.1% tx cost + slippage, max 40% per stock, 10 trades/day |
| `full_autonomous` (hard) | 10 stocks | 40 | 500K | Regime gate, 0.2% slippage, max 20% per stock, 5 trades/day |

## Grading

Each grader in `server/tasks.py` returns 0.0–1.0:
- **single_stock**: Agent return vs buy-and-hold benchmark
- **portfolio**: 60% Sharpe-like + 25% discipline + 15% activity balance
- **full_autonomous**: 35% return + 25% risk-adjusted + 25% regime discipline + 15% risk management

## Action Format

Actions are strings: `HOLD`, `BUY`, `SELL`, `BUY <SYMBOL>`, `SELL <SYMBOL>`, `BUY <SYMBOL> <FRACTION>`. Single-stock tasks accept bare `BUY`/`SELL`; multi-stock tasks require the symbol. Invalid actions default to `HOLD`.

## Key Design Details

- Market data lives in `data/ohlcv/{SYMBOL}_daily.csv` with columns: timestamp, open, high, low, close, volume
- The simulator needs 50 days of lookback before the episode window for indicator computation
- Regime gate (hard task only): blocks BUY/SELL when market breadth is weak (>70% declining) or avg change < -0.5%
- Sells always liquidate the entire position in a symbol
- Session cleanup: sessions are deleted from memory when episodes end (`done=True`) or on server shutdown

## Progress Tracking

Detailed progress is tracked in `.progress/` (gitignored, local only).

**On new session:**
1. Read `.progress/index.md` first — it has the current status overview
2. Only read the specific domain file relevant to the current task (e.g., `sft-training.md` if working on SFT)
3. Never load all domain files at once — read on demand

**After any change:**
- Update the relevant `.progress/` domain file with: what changed, bugs hit, solutions applied
- Update the timestamp at the top of the domain file
- Update `index.md` status line if the overall state changed

**Domain files:** `sft-training.md`, `grpo-training.md`, `environment.md`, `infrastructure.md`, `eval-results.md`
