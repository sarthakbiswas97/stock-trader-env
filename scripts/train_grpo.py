"""GRPO Training — Reinforce trading decisions using environment rewards.

Uses counterfactual rewards: for each observation, the model generates N candidate
actions. Each is scored based on what would have happened (next-day price change)
without actually running the environment.

Usage:
    cd /workspace/stock-trader-env
    pip install -q --upgrade typing_extensions
    pip install -q unsloth unsloth_zoo trl datasets mlflow huggingface_hub
    pip install -q -e .

    python scripts/train_grpo.py --prompts-dataset data/grpo/grpo_prompts_single_stock_train_200ep.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

import mlflow
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sft_checkpoint": "sarthakbiswas/stock-trader-sft-v2-model",
    "prompts_dataset": "",
    "max_seq_length": 1024,
    "max_completion_length": 200,
    "num_generations": 4,
    "max_steps": 300,
    "batch_size": 1,
    "grad_accum": 4,
    "lr": 1e-6,
    "beta": 0.0,
    "temperature": 0.8,
    "output_dir": "/workspace/grpo-checkpoint",
}


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def is_valid_format(text: str) -> bool:
    """Check if completion follows <think>...</think>\\nACTION format."""
    has_think = "<think>" in text and "</think>" in text
    lines = text.strip().split("\n")
    last_line = lines[-1].strip().upper() if lines else ""
    has_action = last_line.startswith(("BUY", "SELL", "HOLD"))
    return has_think and has_action


def parse_action_word(text: str) -> str:
    """Extract the action word (BUY/SELL/HOLD) from completion text."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip().upper()
        if line.startswith(("BUY", "SELL", "HOLD")):
            match = re.match(r"(BUY|SELL|HOLD)", line)
            if match:
                return match.group(1)
    return "HOLD"


# Global metadata store -- populated before training, accessed by reward function
_metadata_store: list[dict] = []
_action_counter: Counter = Counter()


def trading_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward function for GRPO.

    Computes counterfactual P&L based on next-day price change.
    Metadata is accessed via global store indexed by prompt content.
    """
    prompts = kwargs.get("prompts", [])
    rewards = []

    for prompt_msgs, completion in zip(prompts, completions):
        # Extract the observation text from the prompt to find matching metadata
        metadata = _find_metadata(prompt_msgs)

        if metadata is None:
            rewards.append(0.0)
            continue

        # Extract text from completion (may be string or list of message dicts)
        if isinstance(completion, list):
            completion_text = ""
            for msg in completion:
                if isinstance(msg, dict):
                    completion_text += msg.get("content", "")
                else:
                    completion_text += str(msg)
        else:
            completion_text = str(completion)

        # Parse action from completion
        action = parse_action_word(completion_text)
        _action_counter[action] += 1

        # Format compliance reward
        format_ok = is_valid_format(completion_text)
        format_score = 0.0 if format_ok else -1.0

        # Counterfactual trading reward
        current_prices = metadata["current_prices"]
        next_prices = metadata["next_prices"]
        has_position = metadata["has_position"]

        # Single stock: use first symbol
        symbol = list(current_prices.keys())[0]
        current_price = current_prices[symbol]
        next_price = next_prices[symbol]
        price_change_pct = (next_price - current_price) / current_price if current_price > 0 else 0.0

        if action == "BUY" and not has_position:
            trading_score = price_change_pct * 10
        elif action == "SELL" and has_position:
            trading_score = -price_change_pct * 10
        elif action == "HOLD":
            trading_score = price_change_pct * 10 if has_position else 0.0
        else:
            trading_score = -0.05

        reward = 0.7 * trading_score + 0.3 * format_score
        rewards.append(float(reward))

    # Log action distribution periodically
    total = sum(_action_counter.values())
    if total > 0 and total % 50 < len(completions):
        logger.info(
            "Actions so far (%d): %s",
            total,
            {k: f"{v / total * 100:.0f}%" for k, v in _action_counter.most_common()},
        )

    return rewards


def _find_metadata(prompt: object) -> dict | None:
    """Find metadata matching a prompt from the global store."""
    # The prompt from GRPOTrainer can be a string or list of dicts.
    # We match by searching for the observation text snippet in our stored prompts.
    if isinstance(prompt, str):
        search_text = prompt
    elif isinstance(prompt, list):
        # List of message dicts -- get the user message content
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                search_text = msg.get("content", "")
                break
        else:
            return None
    else:
        return None

    # Search metadata store for matching observation
    # Use first 150 chars of observation as key (unique enough per step)
    key = search_text[:150]
    for item in _metadata_store:
        if item["_key"] == key:
            return item["metadata"]

    return None


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Run GRPO training."""

    # ------------------------------------------------------------------
    # 1. Environment check
    # ------------------------------------------------------------------
    logger.info("=== GRPO Training ===")
    logger.info("CUDA: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("")

    # ------------------------------------------------------------------
    # 2. Load SFT model with Unsloth
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logger.info("Loading SFT checkpoint: %s", args.sft_checkpoint)
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.sft_checkpoint,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    logger.info("Model loaded")

    # ------------------------------------------------------------------
    # 3. Load prompts dataset
    # ------------------------------------------------------------------
    if not args.prompts_dataset:
        logger.error("--prompts-dataset required. Run collect_grpo_prompts.py first.")
        return

    logger.info("Loading prompts from: %s", args.prompts_dataset)

    prompts_data = []
    prompts_path = Path(args.prompts_dataset)
    if prompts_path.exists():
        with open(prompts_path) as f:
            for line in f:
                prompts_data.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset(args.prompts_dataset, split="train")
        prompts_data = list(ds)

    logger.info("Loaded %d prompts", len(prompts_data))

    # Build dataset with prompt messages (let trainer handle chat template)
    # Also populate global metadata store for reward function
    global _metadata_store
    _metadata_store.clear()

    dataset_rows = []
    for item in prompts_data:
        prompt_messages = item["prompt"]

        # Find user message for metadata key
        user_content = ""
        for msg in prompt_messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        _metadata_store.append({
            "_key": user_content[:150],
            "metadata": item["metadata"],
        })

        dataset_rows.append({"prompt": prompt_messages})

    from datasets import Dataset
    train_dataset = Dataset.from_list(dataset_rows)
    logger.info("Dataset ready: %d prompts", len(train_dataset))

    # ------------------------------------------------------------------
    # 4. MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("file:///workspace/mlruns")
    mlflow.set_experiment("grpo-training")

    run_name = f"grpo-gen{args.num_generations}-lr{args.lr}-steps{args.max_steps}"
    mlflow_run = mlflow.start_run(run_name=run_name)

    mlflow.log_params({
        "sft_checkpoint": args.sft_checkpoint,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "num_prompts": len(train_dataset),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    })

    logger.info("MLflow run: %s (ID: %s)", run_name, mlflow_run.info.run_id)

    # ------------------------------------------------------------------
    # 5. GRPO Training
    # ------------------------------------------------------------------
    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        seed=42,
        report_to="mlflow",
        temperature=args.temperature,
        loss_type="grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=trading_reward,
        processing_class=tokenizer,
    )

    start_time = time.time()
    logger.info("Starting GRPO training...")
    logger.info("  Prompts: %d", len(train_dataset))
    logger.info("  Max steps: %d", args.max_steps)
    logger.info("  Generations per prompt: %d", args.num_generations)
    logger.info("  Temperature: %.1f", args.temperature)
    logger.info("  Beta (KL): %.4f", args.beta)
    logger.info("")

    trainer.train()
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("Training complete in %.1f minutes", elapsed / 60)
    logger.info("Final action distribution: %s", dict(_action_counter.most_common()))

    mlflow.log_metrics({"training_time_minutes": round(elapsed / 60, 1)})

    # ------------------------------------------------------------------
    # 6. Save checkpoint
    # ------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Checkpoint saved to %s", args.output_dir)

    # ------------------------------------------------------------------
    # 7. Push to HF Hub (optional)
    # ------------------------------------------------------------------
    if args.push_to_hub and args.hf_repo:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping push to hub.")
        else:
            logger.info("Pushing model to HF Hub: %s", args.hf_repo)
            model.push_to_hub(args.hf_repo, token=hf_token)
            tokenizer.push_to_hub(args.hf_repo, token=hf_token)

            # Push MLflow data
            mlflow.end_run()
            mlruns_dir = "/workspace/mlruns"
            if os.path.exists(mlruns_dir):
                import tarfile
                mlruns_tar = os.path.join(args.output_dir, "mlruns.tar.gz")
                with tarfile.open(mlruns_tar, "w:gz") as tar:
                    tar.add(mlruns_dir, arcname="mlruns")

                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=mlruns_tar,
                    path_in_repo="mlruns.tar.gz",
                    repo_id=args.hf_repo,
                    repo_type="model",
                    token=hf_token,
                )
                logger.info("MLflow data pushed to HF Hub")

            logger.info("Done: https://huggingface.co/%s", args.hf_repo)
    else:
        mlflow.end_run()

    logger.info("")
    logger.info("=== Done ===")
    logger.info("Checkpoint: %s", args.output_dir)
    logger.info("Evaluate: PYTHONPATH=. python scripts/eval_sft.py --checkpoint %s", args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--sft-checkpoint", default=DEFAULTS["sft_checkpoint"])
    parser.add_argument("--prompts-dataset", default=DEFAULTS["prompts_dataset"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--max-completion-length", type=int, default=DEFAULTS["max_completion_length"])
    parser.add_argument("--num-generations", type=int, default=DEFAULTS["num_generations"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo", default="sarthakbiswas/stock-trader-grpo-model")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
