"""
Baseline inference script for the Stock Trading Environment.

Uses the OpenAI-compatible API client to run an LLM agent against all 3 tasks.
Produces reproducible baseline scores with mandatory structured stdout logging.

Environment variables:
    HF_TOKEN              — API key for the LLM endpoint
    API_BASE_URL          — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME            — Model to use (default: Qwen/Qwen2.5-72B-Instruct)
    LOCAL_IMAGE_NAME      — Docker image name for the environment (optional, for from_docker_image)
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient


# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "stock-trader-env"
SEED = 42
TASKS = ["single_stock", "portfolio", "full_autonomous"]
SUCCESS_THRESHOLD = 0.3
MAX_RETRIES = 3
SIMULATOR_MODE = os.getenv("SIMULATOR_MODE", "replay")

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

def get_observation_field(obs: object, field: str, default: object = None) -> object:
    """Safely extract a field from an observation (dict or Pydantic model)."""
    if obs is None:
        return default
    if isinstance(obs, dict):
        return obs.get(field, default)
    return getattr(obs, field, default)


def get_action(client: OpenAI, market_summary: str, available_actions: list) -> str:
    """Get action from LLM given current observation."""
    user_message = f"""Current market state:

{market_summary}

Available actions: {', '.join(available_actions)}

What is your action?"""

    for attempt in range(MAX_RETRIES):
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
            content = response.choices[0].message.content
            if content is None:
                print(f"[DEBUG] LLM returned None content (attempt {attempt + 1})", flush=True)
                continue
            action = content.strip().split("\n")[0].strip()
            if action:
                return action
        except Exception as e:
            print(f"[DEBUG] LLM error (attempt {attempt + 1}): {e}", flush=True)

    return "HOLD"


# --- Main ---

async def run_task(env: GenericEnvClient, llm: OpenAI, task_id: str) -> None:
    """Run a single task to completion."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(task_id=task_id, seed=SEED, simulator_mode=SIMULATOR_MODE)

        step = 0
        max_steps = 200  # Safety limit to prevent infinite loops
        while not result.done and step < max_steps:
            step += 1
            obs = result.observation

            market_summary = str(get_observation_field(obs, "market_summary", ""))
            raw_actions = get_observation_field(obs, "available_actions", ["HOLD"])
            available_actions = raw_actions if isinstance(raw_actions, list) else ["HOLD"]

            action = get_action(llm, market_summary, available_actions)

            result = await env.step({"action": action})

            reward = result.reward if result.reward is not None else 0.0
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=result.done,
                error=None,
            )

        if step >= max_steps:
            print(f"[DEBUG] Task {task_id} hit max_steps safety limit ({max_steps})", flush=True)

        final_obs = result.observation if result else None
        score_val = get_observation_field(final_obs, "score", 0.0)
        score = float(score_val) if score_val is not None else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", flush=True)
        raise RuntimeError("HF_TOKEN environment variable not set")

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Connect to the environment:
    # - If LOCAL_IMAGE_NAME is set, spin up a new Docker container
    # - Otherwise, connect directly to an already-running container (validator mode)
    env: GenericEnvClient
    if LOCAL_IMAGE_NAME:
        try:
            env = await GenericEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception as e:
            print(f"ERROR: Failed to start environment from image '{LOCAL_IMAGE_NAME}': {e}", flush=True)
            raise
    else:
        env_url = os.getenv("ENV_URL", "http://localhost:8000")
        print(f"[DEBUG] Connecting directly to environment at {env_url}", flush=True)
        try:
            env = GenericEnvClient(base_url=env_url)
            await env.connect()
        except Exception as e:
            print(f"ERROR: Failed to connect to environment at {env_url}: {e}", flush=True)
            raise

    try:
        for task_id in TASKS:
            await run_task(env, llm, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[DEBUG] Interrupted by user", flush=True)
    except Exception as e:
        print(f"ERROR: Unhandled exception: {e}", flush=True)
        sys.exit(1)
