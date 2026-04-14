"""Run all baseline agents through the evaluation harness and log to MLflow.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --task single_stock --episodes 20
    python scripts/run_baselines.py --train-ppo --ppo-timesteps 50000

After running, view results with:
    mlflow ui
"""

from __future__ import annotations

import argparse
import logging

from baselines.hold_agent import hold_agent
from baselines.rule_based_agent import rule_based_agent
from baselines.ppo_agent import train_ppo, make_ppo_agent
from training.evaluate import evaluate_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline agents")
    parser.add_argument("--task", default="single_stock", help="Task ID")
    parser.add_argument("--split", default="test", help="Data split for evaluation")
    parser.add_argument("--episodes", type=int, default=50, help="Eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ppo", action="store_true", help="Train PPO before eval")
    parser.add_argument("--ppo-timesteps", type=int, default=20_000, help="PPO training steps")
    args = parser.parse_args()

    logger.info("=== Baseline Evaluation: task=%s, split=%s, episodes=%d ===", args.task, args.split, args.episodes)

    # 1. Hold agent
    logger.info("--- Hold Agent ---")
    hold_results = evaluate_agent(
        agent_fn=hold_agent,
        task_id=args.task,
        split=args.split,
        n_episodes=args.episodes,
        seed=args.seed,
        agent_name="hold",
    )
    logger.info("Hold: score=%.3f, return=%.4f, sharpe=%.2f",
                hold_results.mean_score, hold_results.mean_return, hold_results.sharpe)

    # 2. Rule-based agent
    logger.info("--- Rule-Based Agent ---")
    rule_results = evaluate_agent(
        agent_fn=rule_based_agent,
        task_id=args.task,
        split=args.split,
        n_episodes=args.episodes,
        seed=args.seed,
        agent_name="rule_based",
    )
    logger.info("Rule-based: score=%.3f, return=%.4f, sharpe=%.2f",
                rule_results.mean_score, rule_results.mean_return, rule_results.sharpe)

    # 3. PPO agent (optional training)
    if args.train_ppo:
        logger.info("--- PPO Agent (training %d steps) ---", args.ppo_timesteps)
        model = train_ppo(
            task_id=args.task,
            total_timesteps=args.ppo_timesteps,
            seed=args.seed,
            split="train",
        )
        ppo_fn = make_ppo_agent(model, task_id=args.task)
        ppo_results = evaluate_agent(
            agent_fn=ppo_fn,
            task_id=args.task,
            split=args.split,
            n_episodes=args.episodes,
            seed=args.seed,
            agent_name="ppo",
        )
        logger.info("PPO: score=%.3f, return=%.4f, sharpe=%.2f",
                     ppo_results.mean_score, ppo_results.mean_return, ppo_results.sharpe)

    # Summary
    logger.info("")
    logger.info("=== Results Summary ===")
    logger.info("%-15s  Score   Return  Sharpe", "Agent")
    logger.info("%-15s  %.3f   %.4f  %.2f", "hold", hold_results.mean_score, hold_results.mean_return, hold_results.sharpe)
    logger.info("%-15s  %.3f   %.4f  %.2f", "rule_based", rule_results.mean_score, rule_results.mean_return, rule_results.sharpe)
    if args.train_ppo:
        logger.info("%-15s  %.3f   %.4f  %.2f", "ppo", ppo_results.mean_score, ppo_results.mean_return, ppo_results.sharpe)
    logger.info("")
    logger.info("View detailed results: mlflow ui")


if __name__ == "__main__":
    main()
