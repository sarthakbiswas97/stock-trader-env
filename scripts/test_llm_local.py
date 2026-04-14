"""Quick smoke test — run a small LLM against the environment locally."""

import logging

from baselines.llm_agent import create_local_agent
from training.evaluate import evaluate_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

agent = create_local_agent("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")

results = evaluate_agent(
    agent_fn=agent,
    task_id="single_stock",
    split="test",
    n_episodes=3,
    seed=42,
    agent_name="qwen_0.5b_zeroshot",
    log_to_mlflow=False,
)

print(f"Score: {results.mean_score:.3f}, Return: {results.mean_return:.4f}, Sharpe: {results.sharpe:.2f}")
