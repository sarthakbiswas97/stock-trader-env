"""RAFT: Rejection sampling fine-tuning against the neural environment.

Collects episodes, keeps winners, formats as SFT data, trains on them.
The model learns from its own successful trading episodes.

Usage:
    PYTHONPATH=. python scripts/run_raft.py --checkpoint /workspace/sft-v3-checkpoint/checkpoint-200
    PYTHONPATH=. python scripts/run_raft.py --checkpoint sarthakbiswas/stock-trader-sft-v3-model --episodes 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

from baselines.llm_agent import SYSTEM_PROMPT  # noqa: E402 — single source of truth

OUTPUT_DIR = Path("/workspace/raft-data")


def collect_episodes(
    checkpoint: str,
    n_episodes: int,
    simulator_mode: str,
    task_id: str,
    seed: int,
) -> list[dict]:
    """Run model against env, collect full episode trajectories."""
    import re
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from training.gym_wrapper import StockTradingGymEnv

    logger.info(f"Loading model from {checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(checkpoint, max_seq_length=1024, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)

    def get_action(observation: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is today's market data:\n\n{observation}\n\nWhat is your trading action?"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        lines = response.strip().split("\n")
        action = "HOLD"
        for line in reversed(lines):
            match = re.match(r"(BUY|SELL|HOLD)", line.strip().upper())
            if match:
                action = line.strip()
                break
        return action, response

    episodes = []
    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = StockTradingGymEnv(task_id=task_id, seed=ep_seed, obs_mode="text", simulator_mode=simulator_mode)
        obs, info = env.reset()
        initial_value = info["portfolio_value"]

        steps = []
        while True:
            action, response = get_action(obs)
            steps.append({"observation": obs, "action": action, "response": response})
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        env.close()

        final_value = info["portfolio_value"]
        episode_return = (final_value - initial_value) / initial_value
        score = info["score"]
        mistakes = info.get("mistakes_total", 0)

        episodes.append({
            "seed": ep_seed,
            "score": score,
            "return": episode_return,
            "mistakes": mistakes,
            "n_steps": len(steps),
            "steps": steps,
        })

        if (ep + 1) % 10 == 0:
            scores = [e["score"] for e in episodes]
            logger.info(f"Episode {ep + 1}/{n_episodes} — mean score: {sum(scores)/len(scores):.3f}, last: {score:.3f}")

    return episodes


def filter_winners(episodes: list[dict], threshold: float) -> list[dict]:
    """Keep episodes above score threshold."""
    winners = [e for e in episodes if e["score"] >= threshold]
    logger.info(f"Filtered: {len(winners)}/{len(episodes)} episodes above {threshold} threshold")
    return winners


def format_sft_data(winners: list[dict]) -> list[dict]:
    """Convert winning episodes to SFT training format."""
    sft_records = []
    for episode in winners:
        for step in episode["steps"]:
            response = step["response"]
            if "<think>" not in response:
                response = f"<think>Based on current market conditions.</think>\n{step['action']}"

            sft_records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here is today's market data:\n\n{step['observation']}\n\nWhat is your trading action?"},
                    {"role": "assistant", "content": response},
                ]
            })
    logger.info(f"Formatted {len(sft_records)} SFT examples from {len(winners)} winning episodes")
    return sft_records


def main() -> None:
    parser = argparse.ArgumentParser(description="RAFT: Rejection sampling against neural env")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--simulator-mode", default="neural", choices=["replay", "neural"])
    parser.add_argument("--task-id", default="single_stock")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for winners")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect
    logger.info(f"=== RAFT: Collecting {args.episodes} episodes ({args.simulator_mode} mode) ===")
    start = time.time()
    episodes = collect_episodes(args.checkpoint, args.episodes, args.simulator_mode, args.task_id, args.seed)

    scores = [e["score"] for e in episodes]
    logger.info(f"Collection done in {time.time()-start:.0f}s — mean: {sum(scores)/len(scores):.3f}, max: {max(scores):.3f}, min: {min(scores):.3f}")

    # Save raw episodes
    with open(OUTPUT_DIR / "episodes_raw.jsonl", "w") as f:
        for ep in episodes:
            summary = {k: v for k, v in ep.items() if k != "steps"}
            summary["n_steps"] = len(ep["steps"])
            f.write(json.dumps(summary) + "\n")

    # Step 2: Filter
    winners = filter_winners(episodes, args.threshold)
    if not winners:
        logger.warning(f"No episodes above {args.threshold}. Try lowering --threshold.")
        # Fallback: keep top 30%
        sorted_eps = sorted(episodes, key=lambda e: e["score"], reverse=True)
        winners = sorted_eps[:max(1, len(sorted_eps) // 3)]
        logger.info(f"Fallback: keeping top {len(winners)} episodes (top 33%)")

    # Step 3: Format
    sft_data = format_sft_data(winners)

    output_path = OUTPUT_DIR / "raft_winners.jsonl"
    with open(output_path, "w") as f:
        for record in sft_data:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(sft_data)} SFT examples to {output_path}")
    logger.info(f"Winner scores: {[round(w['score'], 3) for w in winners[:10]]}...")
    logger.info("")
    logger.info("Next: train on winners with conservative SFT config:")
    logger.info(f"  PYTHONPATH=. python3 scripts/train_sft.py --dataset {output_path} --lr 2e-6 --epochs 1 --output-dir /workspace/raft-checkpoint")


if __name__ == "__main__":
    main()
