"""
Baseline inference script for the Stock Trading Environment.

Uses the OpenAI-compatible API client to run an LLM agent against all 3 tasks.
Produces reproducible baseline scores with mandatory structured stdout logging.

Environment variables:
    HF_TOKEN / API_KEY    — API key for the LLM endpoint
    API_BASE_URL          — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME            — Model to use (default: Qwen/Qwen2.5-72B-Instruct)
    IMAGE_NAME            — Docker image name for the environment
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient


# --- Configuration ---
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "stock-trader-env"
SEED = 42
TASKS = ["single_stock", "portfolio", "full_autonomous"]
SUCCESS_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert stock trader operating in the Indian equity market (NIFTY stocks).

    You will receive market observations with technical indicators and portfolio state.
    Your job is to decide whether to BUY, SELL, or HOLD each day.

    RULES:
    - Respond with ONLY your action. No explanation.
    - Valid actions: HOLD, BUY, SELL, BUY <SYMBOL>, SELL <SYMBOL>, BUY <SYMBOL> <FRACTION>
    - For single stock tasks: just use BUY, SELL, or HOLD
    - For multi-stock tasks: specify the symbol, e.g., BUY RELIANCE or SELL INFY
    - The fraction (0.0-1.0) controls how much of your cash to use (default: all available)

    STRATEGY GUIDELINES:
    - Buy when RSI < 35 (oversold) and trend is not strongly bearish
    - Sell when RSI > 70 (overbought) or P&L > +3%
    - HOLD when uncertain — avoiding bad trades is as important as making good ones
    - Respect regime gate warnings — when the market is broadly declining, HOLD
    - Don't overtrade — transaction costs eat into returns
    - Diversify across stocks (for portfolio tasks)
    - Cut losses early — sell if position P&L drops below -3%

    RESPONSE FORMAT:
    Respond with exactly one action, e.g.:
    HOLD
    BUY
    SELL INFY
    BUY TCS 0.3
""").strip()


# --- Structured Logging ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --- Agent ---

def get_action(client: OpenAI, market_summary: str, available_actions: list) -> str:
    """Get action from LLM given current observation."""
    user_message = f"""Current market state:

{market_summary}

Available actions: {', '.join(available_actions)}

What is your action?"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=20,
            temperature=0.1,
        )
        action = response.choices[0].message.content.strip()
        return action.split("\n")[0].strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "HOLD"


# --- Main ---

async def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        print("Set it with: export HF_TOKEN=your-key-here")
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await GenericEnvClient.from_docker_image(IMAGE_NAME)

    try:
        for task_id in TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            try:
                result = await env.reset(task_id=task_id, seed=SEED)

                step = 0
                while not result.done:
                    step += 1
                    obs = result.observation

                    market_summary = obs.get("market_summary", "")
                    available_actions = obs.get("available_actions", ["HOLD"])

                    action = get_action(llm, market_summary, available_actions)

                    result = await env.step({"action": action})

                    reward = result.reward or 0.0
                    rewards.append(reward)
                    steps_taken = step

                    log_step(
                        step=step,
                        action=action,
                        reward=reward,
                        done=result.done,
                        error=None,
                    )

                score = result.observation.get("score", 0.0)
                success = score >= SUCCESS_THRESHOLD

            except Exception as e:
                print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
