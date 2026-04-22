"""SFT Training — Fine-tune DeepSeek-R1 7B on oracle-labeled trading data.

Uses Unsloth for 2-3x faster training with 40% less VRAM.
Includes automatic plateau detection to stop training when loss improvement
becomes negligible -- prevents overfitting and saves compute.

Usage:
    cd /workspace/stock-trader-env
    pip install -q --upgrade typing_extensions
    pip install -q unsloth unsloth_zoo trl datasets mlflow huggingface_hub
    pip install -q -e .

    python scripts/train_sft.py
    python scripts/train_sft.py --push-to-hub --hf-repo sarthakbiswas/stock-trader-sft-v2-model
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import mlflow
import numpy as np
import torch
from transformers import TrainerCallback

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plateau detection callback
# ---------------------------------------------------------------------------

class TrainLossPlateauCallback(TrainerCallback):
    """Stop training when relative loss improvement drops below threshold.

    Compares average loss of the current window vs the previous window.
    If improvement is below `rel_threshold` for `patience` consecutive
    checks, sets `control.should_training_stop = True`.
    """

    def __init__(
        self,
        window: int = 30,
        rel_threshold: float = 0.015,
        patience: int = 3,
    ):
        self.window = window  # Number of log events per comparison window
        self.rel_threshold = rel_threshold
        self.patience = patience
        self._losses: list[float] = []
        self._plateau_count = 0
        self._check_count = 0
        self._stopped_at_step: int | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        self._losses.append(logs["loss"])

        # Need at least 2 full windows to compare
        if len(self._losses) < 2 * self.window:
            return

        # Only check at window boundaries
        if len(self._losses) % self.window != 0:
            return

        prev_avg = float(np.mean(self._losses[-2 * self.window : -self.window]))
        curr_avg = float(np.mean(self._losses[-self.window :]))

        rel_improvement = (prev_avg - curr_avg) / prev_avg if prev_avg > 0 else 0.0
        self._check_count += 1

        logger.info(
            "Plateau check #%d: prev_avg=%.4f, curr_avg=%.4f, rel_improvement=%.3f%% (threshold=%.1f%%, patience=%d/%d)",
            self._check_count, prev_avg, curr_avg, rel_improvement * 100,
            self.rel_threshold * 100, self._plateau_count, self.patience,
        )

        if rel_improvement < self.rel_threshold:
            self._plateau_count += 1
            if self._plateau_count >= self.patience:
                self._stopped_at_step = state.global_step
                logger.info(
                    "PLATEAU DETECTED at step %d. Loss %.4f, improvement %.3f%% < %.1f%% for %d consecutive checks. Stopping.",
                    state.global_step, curr_avg, rel_improvement * 100,
                    self.rel_threshold * 100, self.patience,
                )
                control.should_training_stop = True
        else:
            self._plateau_count = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "dataset": "sarthakbiswas/stock-trader-sft-v3",
    "max_seq_length": 1024,
    "epochs": 1,
    "batch_size": 4,
    "grad_accum": 8,
    "lr": 5e-6,
    "warmup_ratio": 0.10,
    "weight_decay": 0.05,
    "lora_rank": 16,
    "lora_alpha": 32,
    "neftune_alpha": 5.0,
    "output_dir": "/workspace/sft-v3-checkpoint",
    "save_steps": 100,
    "plateau_window": 30,
    "plateau_threshold": 0.015,
    "plateau_patience": 3,
}


def train(args: argparse.Namespace) -> None:
    """Run SFT training with Unsloth + plateau detection."""

    # ------------------------------------------------------------------
    # 1. Environment check
    # ------------------------------------------------------------------
    logger.info("=== SFT Training v3 (distilled reasoning) ===")
    logger.info("CUDA: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("")

    # ------------------------------------------------------------------
    # 2. Load model with Unsloth
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logger.info("Loading model: %s", args.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", trainable_params, total_params, trainable_params / total_params * 100)

    # ------------------------------------------------------------------
    # 3. Load dataset from HF Hub
    # ------------------------------------------------------------------
    from datasets import load_dataset

    logger.info("Loading dataset: %s", args.dataset)
    dataset = load_dataset(args.dataset)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    logger.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # ------------------------------------------------------------------
    # 4. MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("file:///workspace/mlruns")
    mlflow.set_experiment("sft-training")

    run_name = f"sft-v3-r{args.lora_rank}-lr{args.lr}-ep{args.epochs}"
    mlflow_run = mlflow.start_run(run_name=run_name)

    mlflow.log_params({
        "model_name": args.model_name,
        "dataset": args.dataset,
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "quantization": "4bit-nf4",
        "precision": "bf16",
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "trainable_params": trainable_params,
        "plateau_window": args.plateau_window,
        "plateau_threshold": args.plateau_threshold,
        "plateau_patience": args.plateau_patience,
    })

    logger.info("MLflow run: %s (ID: %s)", run_name, mlflow_run.info.run_id)

    # ------------------------------------------------------------------
    # 5. Training with plateau detection
    # ------------------------------------------------------------------
    from trl import SFTTrainer, SFTConfig

    total_steps = len(train_ds) // (args.batch_size * args.grad_accum) * args.epochs

    logger.info("Effective batch size: %d", args.batch_size * args.grad_accum)
    logger.info("Total steps (ceiling): ~%d", total_steps)
    logger.info("Plateau detection: window=%d, threshold=%.1f%%, patience=%d",
                args.plateau_window, args.plateau_threshold * 100, args.plateau_patience)
    logger.info("")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=int(total_steps * args.warmup_ratio),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="no",
        bf16=True,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        neftune_noise_alpha=args.neftune_alpha,
        report_to="mlflow",
        seed=42,
        optim="adamw_torch_fused",
    )

    # Pre-convert messages to text
    def convert_to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    logger.info("Converting messages to chat format...")
    train_text = train_ds.map(convert_to_text, remove_columns=["messages"])
    val_text = val_ds.map(convert_to_text, remove_columns=["messages"])
    logger.info("Conversion done. Train: %d, Val: %d", len(train_text), len(val_text))

    # Plateau callback
    plateau_callback = TrainLossPlateauCallback(
        window=args.plateau_window,
        rel_threshold=args.plateau_threshold,
        patience=args.plateau_patience,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_text,
        eval_dataset=val_text,
        processing_class=tokenizer,
        callbacks=[plateau_callback],
    )

    start_time = time.time()
    logger.info("Starting training (plateau callback will auto-stop)...")
    train_result = trainer.train()
    elapsed = time.time() - start_time

    stopped_at = plateau_callback._stopped_at_step
    final_step = trainer.state.global_step

    logger.info("")
    logger.info("Training ended at step %d (%.1f minutes)", final_step, elapsed / 60)
    if stopped_at:
        logger.info("Plateau detected at step %d", stopped_at)
    else:
        logger.info("Completed full epoch (no plateau detected)")
    logger.info("Final train loss: %.4f", train_result.training_loss)

    # ------------------------------------------------------------------
    # 6. Run eval once
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Running evaluation on val set...")
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss", -1.0)
    logger.info("Eval loss: %.4f", eval_loss)

    # Log final metrics
    mlflow.log_metrics({
        "final_train_loss": train_result.training_loss,
        "eval_loss": eval_loss,
        "training_time_minutes": round(elapsed / 60, 1),
        "stopped_at_step": stopped_at or final_step,
        "plateau_detected": 1 if stopped_at else 0,
    })

    # ------------------------------------------------------------------
    # 7. Save checkpoint
    # ------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Checkpoint saved to %s", args.output_dir)

    # ------------------------------------------------------------------
    # 8. Sanity test
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=== Sanity Test ===")
    FastLanguageModel.for_inference(model)

    test_obs = """Day 5 of 20 | Cash: Rs85,000 | Portfolio: Rs103,500 | Return: +3.5%

Market Context:
  India VIX: 14.2 (normal, below 15-day avg of 15.1)
  USD/INR: 83.20 (-0.2% today, strengthening)
  Brent Crude: $78.5 (+0.8% today)
  Sectors: Bank +0.5% | IT -0.3% | Pharma +0.2% (cyclicals leading)
  RBI Repo Rate: 6.00% (last change: Apr 2025)

RELIANCE: Rs1,250 (-1.5% today)
  RSI: 28 (oversold) | MACD: bullish (CROSSOVER)
  Trend: bullish | Bollinger: lower_band (oversold)
  Volume: 1.8x avg (high) | Volatility: moderate
  Momentum: down (-2.0%)
  Candle: hammer (bullish reversal)"""

    messages = [
        {"role": "system", "content": "You are an expert stock trader operating in the Indian equity market.\nYou receive daily market observations with technical indicators and must decide on a trading action.\n\nRules:\n- Respond with EXACTLY one action on the last line\n- Valid actions: HOLD, BUY, SELL, BUY <SYMBOL>, SELL <SYMBOL>, BUY <SYMBOL> <FRACTION>\n- Before your action, briefly explain your reasoning in 1-2 sentences inside <think> tags\n\nExample response:\n<think>RSI is 25 (oversold) and MACD just turned bullish. Volume is spiking. Good entry point.</think>\nBUY RELIANCE 0.5"},
        {"role": "user", "content": f"Here is today's market data:\n\n{test_obs}\n\nWhat is your trading action?"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info("Test response: %s", response[:300])
    mlflow.log_text(response, "sanity_test_response.txt")

    # ------------------------------------------------------------------
    # 9. Push to HF Hub (optional)
    # ------------------------------------------------------------------
    if args.push_to_hub and args.hf_repo:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping push to hub.")
        else:
            logger.info("Pushing model to HF Hub: %s", args.hf_repo)
            model.push_to_hub(args.hf_repo, token=hf_token)
            tokenizer.push_to_hub(args.hf_repo, token=hf_token)

            # Push MLflow experiment data alongside the model
            mlflow.end_run()
            mlruns_dir = "/workspace/mlruns"
            if os.path.exists(mlruns_dir):
                import tarfile
                mlruns_tar = os.path.join(args.output_dir, "mlruns.tar.gz")
                logger.info("Archiving MLflow data to %s", mlruns_tar)
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

            logger.info("Pushed to: https://huggingface.co/%s", args.hf_repo)
    else:
        mlflow.end_run()

    logger.info("")
    logger.info("=== Done ===")
    logger.info("Checkpoint: %s", args.output_dir)
    if not (args.push_to_hub and args.hf_repo):
        logger.info("MLflow: /workspace/mlruns (push with --push-to-hub to persist)")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Training v3 (distilled reasoning)")
    parser.add_argument("--model-name", default=DEFAULTS["model_name"])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--neftune-alpha", type=float, default=DEFAULTS["neftune_alpha"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--save-steps", type=int, default=DEFAULTS["save_steps"])
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo", default="sarthakbiswas/stock-trader-sft-v3-model")
    parser.add_argument("--plateau-window", type=int, default=DEFAULTS["plateau_window"])
    parser.add_argument("--plateau-threshold", type=float, default=DEFAULTS["plateau_threshold"])
    parser.add_argument("--plateau-patience", type=int, default=DEFAULTS["plateau_patience"])
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
