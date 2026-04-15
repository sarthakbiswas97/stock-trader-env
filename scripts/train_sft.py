"""SFT Training — Fine-tune DeepSeek-R1 7B on oracle-labeled trading data.

Runs on RunPod A4500 (or any Ampere GPU with 20GB+ VRAM).
Uses Unsloth for 2-3x faster training with 40% less VRAM.
All hyperparameters logged to MLflow for experiment tracking.

Usage:
    # On RunPod after uploading project:
    cd /workspace/stock-trader-env
    pip install -q --upgrade typing_extensions
    pip install -q unsloth unsloth_zoo trl datasets mlflow huggingface_hub
    pip install -q -e .

    python scripts/train_sft.py
    python scripts/train_sft.py --epochs 3 --push-to-hub --hf-repo sarthakbiswas/stock-trader-sft-v2-model
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import mlflow
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "dataset": "sarthakbiswas/stock-trader-sft-v2",
    "max_seq_length": 1024,
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 8,
    "lr": 2e-5,
    "warmup_ratio": 0.05,
    "lora_rank": 64,
    "lora_alpha": 16,
    "output_dir": "/workspace/sft-v2-checkpoint",
}


def train(args: argparse.Namespace) -> None:
    """Run SFT training with Unsloth."""

    # ------------------------------------------------------------------
    # 1. Environment check
    # ------------------------------------------------------------------
    logger.info("=== SFT Training v2 ===")
    logger.info("CUDA: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)
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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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

    run_name = f"sft-v2-r{args.lora_rank}-ep{args.epochs}-bs{args.batch_size * args.grad_accum}"
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
    })

    logger.info("MLflow run: %s (ID: %s)", run_name, mlflow_run.info.run_id)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    from trl import SFTTrainer, SFTConfig

    total_steps = len(train_ds) // (args.batch_size * args.grad_accum) * args.epochs
    logger.info("Effective batch size: %d", args.batch_size * args.grad_accum)
    logger.info("Total steps: ~%d", total_steps)
    logger.info("")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        report_to="mlflow",
        seed=42,
        optim="adamw_torch_fused",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    start_time = time.time()
    logger.info("Starting training...")
    train_result = trainer.train()
    elapsed = time.time() - start_time

    logger.info("Training complete in %.1f minutes", elapsed / 60)
    logger.info("Final train loss: %.4f", train_result.training_loss)

    mlflow.log_metrics({
        "final_train_loss": train_result.training_loss,
        "training_time_minutes": round(elapsed / 60, 1),
    })

    # ------------------------------------------------------------------
    # 6. Save checkpoint
    # ------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Checkpoint saved to %s", args.output_dir)

    # ------------------------------------------------------------------
    # 7. Quick sanity test
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
    # 8. Push to HF Hub (optional)
    # ------------------------------------------------------------------
    if args.push_to_hub and args.hf_repo:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping push to hub.")
        else:
            logger.info("Pushing to HF Hub: %s", args.hf_repo)
            model.push_to_hub(args.hf_repo, token=hf_token)
            tokenizer.push_to_hub(args.hf_repo, token=hf_token)
            mlflow.log_param("hf_repo", args.hf_repo)
            logger.info("Pushed to: https://huggingface.co/%s", args.hf_repo)

    mlflow.end_run()
    logger.info("")
    logger.info("=== Done ===")
    logger.info("Checkpoint: %s", args.output_dir)
    logger.info("MLflow: /workspace/mlruns")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Training v2")
    parser.add_argument("--model-name", default=DEFAULTS["model_name"])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--push-to-hub", action="store_true", help="Push adapter to HF Hub after training")
    parser.add_argument("--hf-repo", default="sarthakbiswas/stock-trader-sft-v2-model", help="HF repo for model push")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
