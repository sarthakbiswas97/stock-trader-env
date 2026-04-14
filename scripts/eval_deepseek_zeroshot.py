"""Evaluate DeepSeek-R1 7B zero-shot via HuggingFace Inference API."""

import logging
import os

from baselines.llm_agent import create_api_agent
from training.evaluate import evaluate_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

agent = create_api_agent(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    api_token=os.environ.get("HF_TOKEN"),
)

results = evaluate_agent(
    agent_fn=agent,
    task_id="single_stock",
    split="test",
    n_episodes=5,
    seed=42,
    agent_name="deepseek_7b_zeroshot",
    log_to_mlflow=False,
)

print(f"\nDeepSeek-R1 7B Zero-Shot Results:")
print(f"Score: {results.mean_score:.3f}, Return: {results.mean_return:.4f}, Sharpe: {results.sharpe:.2f}")
