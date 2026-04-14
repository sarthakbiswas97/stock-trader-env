"""SFT (Supervised Fine-Tuning) on expert trading trajectories.

Fine-tunes a base model on (observation → reasoning + action) pairs
collected from the rule-based agent. This teaches the model:
    1. The action format (BUY/SELL/HOLD with symbols)
    2. Chain-of-thought reasoning before acting
    3. Basic trading patterns (oversold entries, profit-taking, stop-losses)

Usage:
    # Local test with small model (validates pipeline)
    PYTHONPATH=. python scripts/train_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 1

    # RunPod with 7B model (real training)
    PYTHONPATH=. python scripts/train_sft.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --use-qlora \
        --epochs 3 \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "sft"
OUTPUT_DIR = Path(__file__).parent.parent / "checkpoints" / "sft"


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training on expert trajectories")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID")
    parser.add_argument("--data", default=None, help="Path to JSONL data. If None, loads all files in data/sft/")
    parser.add_argument("--output", default=None, help="Output directory for checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size (for local testing)")
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    args = parser.parse_args()

    # Determine device — MPS can't handle training on 8GB Mac, use CPU as fallback
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # Load data
    if args.data:
        data_files = [args.data]
    else:
        data_files = sorted(str(p) for p in DATA_DIR.glob("*.jsonl"))
    logger.info("Loading data from: %s", data_files)
    dataset = load_dataset("json", data_files=data_files, split="train")
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info("Truncated dataset to %d examples (--max-samples)", args.max_samples)
    logger.info("Dataset size: %d examples", len(dataset))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading
    model_kwargs = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32}

    peft_config = None
    if args.use_qlora:
        from peft import LoraConfig, TaskType
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info("Using QLoRA: rank=%d, alpha=16", args.lora_rank)

    logger.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Output directory
    model_short = args.model.split("/")[-1]
    output_dir = args.output or str(OUTPUT_DIR / model_short)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=(device == "cuda"),
        fp16=False,
        max_length=args.max_seq_length,
        gradient_checkpointing=True if device == "cuda" else False,
        report_to="none",
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting SFT training: %d examples, %d epochs, batch=%d, lr=%s",
                len(dataset), args.epochs, args.batch_size, args.lr)
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
