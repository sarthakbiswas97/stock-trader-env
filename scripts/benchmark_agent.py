"""
Benchmark agent: runs the proven daily reversal strategy against the environment.

Strategy (from the trader project):
- Rank stocks by worst 5d/10d/21d returns (biggest losers)
- Buy top losers (reversal signal)
- Hold for 5 days, then sell
- Respect regime gate warnings
- No ML/LLM — pure rule-based

Usage:
    # Start server first: uvicorn server.app:app --host 0.0.0.0 --port 8000
    python scripts/benchmark_agent.py
"""

import asyncio
import re
from collections import defaultdict
from typing import List, Optional

from openenv.core.generic_client import GenericEnvClient


SEED = 42
HOLD_PERIOD = 5  # sell after 5 days
MAX_POSITIONS = 3  # max concurrent positions per task scaling
BUY_FRACTION = 0.15  # use 15% of cash per buy


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_prices_from_summary(market_summary: str) -> dict[str, float]:
    """Extract current prices from market_summary text.

    Looks for patterns like: 'RELIANCE: Rs2,480 (+1.2% today)'
    """
    prices = {}
    pattern = r"([A-Z&-]+): Rs([\d,]+(?:\.\d+)?)"
    for match in re.finditer(pattern, market_summary):
        symbol = match.group(1)
        price_str = match.group(2).replace(",", "")
        try:
            prices[symbol] = float(price_str)
        except ValueError:
            pass
    return prices


def parse_regime_warning(market_summary: str) -> bool:
    """Check if regime gate warning is active."""
    return "REGIME GATE ACTIVE" in market_summary


def compute_reversal_scores(price_history: dict[str, list[float]], symbols: list[str]) -> list[tuple[str, float]]:
    """Rank stocks by worst recent returns (reversal signal).

    Returns list of (symbol, score) sorted by highest score (biggest losers first).
    """
    scores = []
    for sym in symbols:
        history = price_history.get(sym, [])
        if len(history) < 6:  # need at least 6 days
            continue

        current = history[-1]

        # 5-day return
        ret_5d = (current - history[-6]) / history[-6] if len(history) >= 6 else 0
        # 10-day return (or whatever we have)
        lookback_10 = min(11, len(history))
        ret_10d = (current - history[-lookback_10]) / history[-lookback_10]
        # 21-day return (or whatever we have)
        lookback_21 = min(22, len(history))
        ret_21d = (current - history[-lookback_21]) / history[-lookback_21]

        # Reversal score: average of negative returns (bigger loser = higher score)
        avg_return = (ret_5d + ret_10d + ret_21d) / 3
        reversal_score = -avg_return  # negate: worst returns get highest score

        scores.append((sym, reversal_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def decide_action(
    obs: dict,
    price_history: dict[str, list[float]],
    held_since: dict[str, int],
    current_day: int,
    task_id: str,
) -> str:
    """Apply reversal strategy to decide action."""
    market_summary = obs.get("market_summary", "")
    available_actions = obs.get("available_actions", ["HOLD"])
    positions = obs.get("positions", [])

    # Parse current prices
    prices = parse_prices_from_summary(market_summary)

    # Record prices for history
    for sym, price in prices.items():
        price_history[sym].append(price)

    # Get symbols we can trade
    buyable = [a.replace("BUY ", "").split()[0] for a in available_actions if a.startswith("BUY")]
    sellable = [a.replace("SELL ", "").split()[0] for a in available_actions if a.startswith("SELL")]

    # Check regime
    regime_blocked = parse_regime_warning(market_summary)

    # SELL: positions held for 5+ days
    for sym in sellable:
        if sym in held_since and (current_day - held_since[sym]) >= HOLD_PERIOD:
            del held_since[sym]
            return f"SELL {sym}"

    # SELL: stop loss — sell if P&L < -5%
    for pos in positions:
        sym = pos.get("symbol", "")
        pnl = pos.get("pnl_percent", 0)
        if pnl < -5.0 and sym in sellable:
            if sym in held_since:
                del held_since[sym]
            return f"SELL {sym}"

    # Don't buy during regime gate
    if regime_blocked:
        return "HOLD"

    # Max positions based on task
    max_pos = {"single_stock": 1, "portfolio": 4, "full_autonomous": 6}.get(task_id, 3)
    current_positions = len(positions)

    if current_positions >= max_pos:
        return "HOLD"

    # Not enough history yet — hold
    if current_day < 6:
        return "HOLD"

    # BUY: top reversal candidates not already held
    held_symbols = {pos.get("symbol", "") for pos in positions}
    scores = compute_reversal_scores(price_history, buyable)

    for sym, score in scores:
        if sym not in held_symbols and score > 0:
            # Single stock task: bare BUY
            if task_id == "single_stock":
                held_since[sym] = current_day
                return "BUY"
            else:
                held_since[sym] = current_day
                return f"BUY {sym} {BUY_FRACTION}"

    return "HOLD"


async def run_benchmark():
    env = GenericEnvClient(base_url="http://localhost:8000")

    results = []

    for task_id in ["single_stock", "portfolio", "full_autonomous"]:
        await env.connect()
        log_start(task=task_id, env="stock-trader-env", model="reversal-benchmark")

        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        price_history: dict[str, list[float]] = defaultdict(list)
        held_since: dict[str, int] = {}

        try:
            result = await env.reset(task_id=task_id, seed=SEED)

            step = 0
            while not result.done:
                step += 1
                obs = result.observation

                action = decide_action(obs, price_history, held_since, step, task_id)

                result = await env.step({"action": action})
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action, reward=reward, done=result.done, error=None)

            score = result.observation.get("score", 0.0)
            success = score >= 0.3

        except Exception as e:
            print(f"[DEBUG] Error: {e}", flush=True)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            results.append({"task": task_id, "score": score, "steps": steps_taken})
            await env.close()

    # Summary comparison
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS: Reversal Strategy vs LLM Baseline (gpt-4o-mini)")
    print("=" * 60)

    llm_scores = {"single_stock": 0.600, "portfolio": 0.376, "full_autonomous": 0.667}
    for r in results:
        task = r["task"]
        bench = r["score"]
        llm = llm_scores.get(task, 0)
        diff = bench - llm
        indicator = "▲" if diff > 0 else "▼" if diff < 0 else "="
        print(f"  {task:20s}  Benchmark: {bench:.3f}  LLM: {llm:.3f}  {indicator} {diff:+.3f}")

    avg_bench = sum(r["score"] for r in results) / len(results)
    avg_llm = sum(llm_scores.values()) / len(llm_scores)
    print(f"\n  {'Average':20s}  Benchmark: {avg_bench:.3f}  LLM: {avg_llm:.3f}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
