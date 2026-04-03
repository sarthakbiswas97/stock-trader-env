---
title: Stock Trading RL Environment
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - finance
  - trading
---

# Stock Trading RL Environment

A real-world OpenEnv environment that simulates daily stock trading on Indian equity markets (NIFTY stocks) using real historical OHLCV data. LLM agents connect via HTTP/WebSocket, receive market observations with technical indicators, and respond with trade actions.

## Why This Environment?

Stock trading is one of the most genuine real-world decision-making tasks — it requires reading market signals, managing risk, sizing positions, and knowing when NOT to act. This environment captures that complexity across 3 difficulty levels, using real price data from 68 NIFTY stocks (NIFTY 50 + select NIFTY 100) spanning ~5 years of market history.

Unlike toy environments, agents must deal with:
- **Transaction costs and slippage** that eat into returns
- **Position limits** that force diversification
- **Regime gates** that penalize trading during market-wide downturns
- **Meaningful reward shaping** — not just end-of-episode binary scores

## Action Space

Actions are plain text strings that any LLM can produce:

| Action | Example | Description |
|--------|---------|-------------|
| `HOLD` | `HOLD` | Do nothing this day |
| `BUY` | `BUY` | Buy the stock (single-stock task only) |
| `SELL` | `SELL` | Sell entire position (single-stock task only) |
| `BUY <SYMBOL>` | `BUY RELIANCE` | Buy a specific stock (multi-stock tasks) |
| `SELL <SYMBOL>` | `SELL INFY` | Sell entire position in a stock |
| `BUY <SYMBOL> <FRACTION>` | `BUY TCS 0.3` | Buy using 30% of available cash |

Invalid actions default to `HOLD` — the environment never crashes on bad input.

## Observation Space

Each step returns a `MarketObservation` (Pydantic model):

```python
class MarketObservation(BaseModel):
    done: bool                    # Whether the episode has ended
    reward: float                 # Reward for the last action
    day: int                      # Current trading day (1-indexed)
    total_days: int               # Total days in this episode
    portfolio_value: float        # Total value (cash + positions)
    cash: float                   # Available cash
    positions: list[PositionInfo] # Open positions with P&L
    market_summary: str           # Human-readable market state for LLM
    available_actions: list[str]  # Valid actions the agent can take
    task_id: str                  # Current task identifier
    score: float                  # Current grader score (0.0-1.0)
```

The `market_summary` field provides a rich text description including:
- Current prices and daily changes for each stock
- Technical indicators: RSI, MACD, Bollinger Band position, trend, momentum, volatility
- Volume analysis (spike detection relative to 20-day average)
- Position summary with P&L percentages
- Constraint reminders (costs, limits, trades remaining)
- Regime gate warnings when market breadth is weak

## Tasks

### Task 1: `single_stock` (Easy)

Trade a single stock (RELIANCE) over 20 days with no constraints.

| Parameter | Value |
|-----------|-------|
| Stocks | RELIANCE |
| Episode length | 20 days |
| Initial capital | Rs 100,000 |
| Transaction cost | 0% |
| Slippage | 0% |
| Position limit | 100% (can go all-in) |
| Max trades/day | Unlimited |

**Grader:** Compares agent return against buy-and-hold benchmark. Score 0.5+ means matching the market; 0.8+ means significantly beating it.

### Task 2: `portfolio` (Medium)

Manage a 10-stock portfolio over 30 days with realistic trading costs and position limits.

| Parameter | Value |
|-----------|-------|
| Stocks | RELIANCE, INFY, TCS, HDFCBANK, SBIN, ICICIBANK, BHARTIARTL, ITC, KOTAKBANK, LT |
| Episode length | 30 days |
| Initial capital | Rs 200,000 |
| Transaction cost | 0.1% per trade |
| Slippage | 0.1% |
| Position limit | Max 30% in any single stock |
| Max trades/day | 10 |

**Grader:** Weighted composite — 60% risk-adjusted return (Sharpe-like), 25% discipline (no violations), 15% activity balance (not too passive, not overtrading).

### Task 3: `full_autonomous` (Hard)

Trade 25 stocks over 40 days with regime gate, tight position limits, and realistic costs. The agent must learn WHEN NOT to trade.

| Parameter | Value |
|-----------|-------|
| Stocks | 25 NIFTY stocks (RELIANCE, INFY, TCS, HDFCBANK, SBIN, ICICIBANK, BHARTIARTL, ITC, KOTAKBANK, LT, AXISBANK, BAJFINANCE, SUNPHARMA, TITAN, HINDUNILVR, HCLTECH, WIPRO, NTPC, POWERGRID, ADANIENT, TATASTEEL, JSWSTEEL, COALINDIA, ONGC, MARUTI) |
| Episode length | 40 days |
| Initial capital | Rs 500,000 |
| Transaction cost | 0.1% per trade |
| Slippage | 0.2% |
| Position limit | Max 15% in any single stock |
| Max trades/day | 5 |
| Regime gate | Active — blocks trading when >70% stocks declining or avg change < -0.5% |

**Grader:** Weighted composite — 35% absolute return, 25% risk-adjusted return, 25% regime discipline (respected gate when active), 15% risk management (violations + max drawdown).

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Install

```bash
pip install -r requirements.txt
```

### Run locally

```bash
# 1. Build the Docker image
docker build -t stock-trader-env .

# 2. Run the baseline agent (launches the Docker container automatically)
IMAGE_NAME=stock-trader-env HF_TOKEN=your-key-here python inference.py
```

Or without Docker (for development):

```bash
# Start the environment server directly
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` / `API_KEY` | API key for the LLM endpoint | (required) |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier (baseline tested with `gpt-4o-mini`) | `Qwen/Qwen2.5-72B-Instruct` |
| `IMAGE_NAME` | Docker image name for the environment (built locally or pulled from HF) | `stock-trader-env` |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode (JSON body: `{"task_id": "single_stock", "seed": 42}`) |
| `POST` | `/step` | Take an action (JSON body: `{"action": {"action": "BUY RELIANCE"}}`) |
| `GET` | `/state` | Get episode metadata |
| `GET` | `/schema` | Action/observation JSON schemas |
| `WebSocket` | `/ws` | Persistent session (recommended for agents) |

## Reward Design

Rewards provide signal at every step, not just end-of-episode:

- **PnL reward** — Scaled daily portfolio return (primary signal)
- **Risk discipline bonus** — Small positive reward for staying within position limits
- **Regime gate penalty** — Negative reward for trading during market-wide downturns (hard task)
- **Trade limit violation** — Penalty for exceeding daily trade limits
- **Invalid sell penalty** — Penalty for attempting to sell a stock not held

## Market Data

Real historical daily OHLCV data for 68 NIFTY stocks (NIFTY 50 + select NIFTY 100) stored in `data/ohlcv/`, spanning ~5 years of market history. Each episode picks a random start window (with 50-day lookback buffer for indicator computation), ensuring diverse market conditions across runs.

## Baseline Scores

*Scores produced with seed=42 using gpt-4o-mini via the baseline inference script.*

| Task | Score | Steps | Agent Behavior |
|------|-------|-------|----------------|
| `single_stock` | 0.600 | 20 | Two buy-sell cycles on RELIANCE, captured upside on second entry |
| `portfolio` | 0.376 | 30 | Traded across 6 of 10 stocks, struggled with position sizing across larger universe |
| `full_autonomous` | 0.667 | 40 | Good diversification across 8 of 25 stocks with fractional sizing, some regime penalties |

## Project Structure

```
stock-trader-env/
├── server/
│   ├── app.py              # OpenEnv server (via create_app)
│   ├── environment.py      # Core RL environment (reset/step/state)
│   ├── market_simulator.py # Historical data replay
│   ├── feature_engine.py   # Technical indicators → text
│   └── tasks.py            # Task configs + grading functions
├── data/ohlcv/             # 68 NIFTY stock CSVs (~5 years daily)
├── models.py               # Pydantic data contracts
├── client.py               # Typed OpenEnv client
├── inference.py            # Baseline LLM agent
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # Container definition
└── requirements.txt        # Python dependencies
```

## License

MIT
