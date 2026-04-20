"""Train the neural market world model.

Supports two architectures:
  - transformer: Causal transformer + MDN (~1.2M params)
  - cnn-gru: CNN encoder + GRU dynamics + MDN (~500K-1M params)

Usage:
    PYTHONPATH=. python scripts/train_world_model.py --model transformer --epochs 10
    PYTHONPATH=. python scripts/train_world_model.py --model cnn-gru --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from world_model.data import (
    CausalSequenceDataset,
    MarketSequenceDataset,
    load_all_ohlcv,
)
from world_model.model import (
    CausalTransformerWorldModel,
    MarketWorldModel,
    TransformerConfig,
    WorldModelConfig,
    mdn_loss,
    reconstruction_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "world_model"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transformer(seq_len: int) -> CausalTransformerWorldModel:
    config = TransformerConfig(seq_len=seq_len)
    return CausalTransformerWorldModel(config)


def build_cnn_gru(seq_len: int) -> MarketWorldModel:
    config = WorldModelConfig(
        encoder_channels=(64, 128, 128),
        gru_hidden_dim=256,
        gru_layers=2,
        latent_dim=128,
        seq_len=seq_len,
    )
    return MarketWorldModel(config)


def train_epoch_transformer(
    model: CausalTransformerWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch with causal multi-position loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        pi, mu, sigma = model(sequences)
        loss = mdn_loss(pi, mu, sigma, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_epoch_cnn_gru(
    model: MarketWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recon_weight: float = 0.1,
) -> float:
    """Train one epoch with single-step MDN + reconstruction loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        _, pi, mu, sigma, recon = model(sequences)
        loss = mdn_loss(pi, mu, sigma, targets) + recon_weight * reconstruction_loss(recon, sequences)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, is_transformer: bool) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        if is_transformer:
            pi, mu, sigma = model(sequences)
        else:
            _, pi, mu, sigma, _ = model(sequences)

        total_loss += mdn_loss(pi, mu, sigma, targets).item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural market world model")
    parser.add_argument("--model", choices=["transformer", "cnn-gru"], default="transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    is_transformer = args.model == "transformer"

    logger.info(f"Training {args.model} on {device}")

    # Load data
    symbols_data = load_all_ohlcv()

    if is_transformer:
        dataset = CausalSequenceDataset(symbols_data, seq_len=args.seq_len, stride=args.stride)
        model = build_transformer(args.seq_len)
    else:
        dataset = MarketSequenceDataset(symbols_data, seq_len=args.seq_len, stride=args.stride)
        model = build_cnn_gru(args.seq_len)

    model = model.to(device)
    logger.info(f"Model: {model.count_parameters():,} params, {len(dataset):,} sequences")

    # Split
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {"model": args.model, "epochs": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if is_transformer:
            train_loss = train_epoch_transformer(model, train_loader, optimizer, device)
        else:
            train_loss = train_epoch_cnn_gru(model, train_loader, optimizer, device)

        val_loss = validate(model, val_loader, device, is_transformer)
        scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - start_time
        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"train: {train_loss:.4f}, val: {val_loss:.4f}, "
            f"lr: {lr_current:.6f}, time: {elapsed:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_type": args.model,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "seq_len": args.seq_len,
            }, CHECKPOINT_DIR / f"best_{args.model}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    total_time = time.time() - start_time
    history["total_time_s"] = total_time
    history["best_val_loss"] = best_val_loss
    history["params"] = model.count_parameters()

    with open(RESULTS_DIR / f"training_{args.model}.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"Done in {total_time:.0f}s — best val: {best_val_loss:.4f}, "
        f"saved to checkpoints/world_model/best_{args.model}.pt"
    )


if __name__ == "__main__":
    main()
