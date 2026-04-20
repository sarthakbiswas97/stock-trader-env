"""Training loop for the neural market world model.

Trains the V-M encoder + dynamics (GRU+MDN) + decoder on historical
OHLCV data. Optimizes MDN loss (next-day prediction) + reconstruction
loss (autoencoder regularization).

Can train on Mac MPS (~1 hour for 50 epochs on 97 stocks).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from world_model.data import (
    MarketSequenceDataset,
    load_all_ohlcv,
)
from world_model.model import (
    MarketWorldModel,
    WorldModelConfig,
    mdn_loss,
    reconstruction_loss,
)

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "world_model"


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_world_model(
    data_dir: Path | None = None,
    config: WorldModelConfig | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    recon_weight: float = 0.1,
    checkpoint_dir: Path | None = None,
    seed: int = 42,
) -> tuple[MarketWorldModel, dict]:
    """Train the world model on historical OHLCV data.

    Args:
        data_dir: Path to OHLCV CSVs. Defaults to data/ohlcv/.
        config: Model configuration. Defaults to WorldModelConfig().
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        val_split: Fraction of data for validation.
        recon_weight: Weight for reconstruction loss.
        checkpoint_dir: Where to save checkpoints.
        seed: Random seed for reproducibility.

    Returns:
        Trained model and training history dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config or WorldModelConfig()
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Training on device: {device}")

    # Load data
    logger.info("Loading OHLCV data...")
    symbols_data = load_all_ohlcv(data_dir)

    # Create dataset
    dataset = MarketSequenceDataset(symbols_data, seq_len=config.seq_len)
    stats = dataset.compute_stats()
    logger.info(f"Dataset: {len(dataset)} sequences")

    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Create model
    model = MarketWorldModel(config).to(device)
    n_params = model.count_parameters()
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mdn_loss": [],
        "train_recon_loss": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "config": asdict(config),
        "n_params": n_params,
        "device": str(device),
    }

    start_time = time.time()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_mdn = 0.0
        epoch_recon = 0.0
        n_batches = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            _, pi, mu, sigma, recon = model(sequences)

            loss_mdn = mdn_loss(pi, mu, sigma, targets)
            loss_recon = reconstruction_loss(recon, sequences)
            loss = loss_mdn + recon_weight * loss_recon

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_mdn += loss_mdn.item()
            epoch_recon += loss_recon.item()
            n_batches += 1

        scheduler.step()

        avg_mdn = epoch_mdn / n_batches
        avg_recon = epoch_recon / n_batches
        avg_train = avg_mdn + recon_weight * avg_recon

        # --- Validate ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                _, pi, mu, sigma, recon = model(sequences)

                loss_mdn = mdn_loss(pi, mu, sigma, targets)
                loss_recon = reconstruction_loss(recon, sequences)
                val_loss_total += (loss_mdn + recon_weight * loss_recon).item()
                val_batches += 1

        avg_val = val_loss_total / val_batches if val_batches > 0 else float("inf")

        # Record history
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_mdn_loss"].append(avg_mdn)
        history["train_recon_loss"].append(avg_recon)

        # Save best
        if avg_val < history["best_val_loss"]:
            history["best_val_loss"] = avg_val
            history["best_epoch"] = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "stats": {
                    "feature_means": stats.feature_means.tolist(),
                    "feature_stds": stats.feature_stds.tolist(),
                },
                "epoch": epoch,
                "val_loss": avg_val,
            }, checkpoint_dir / "best_model.pt")

        elapsed = time.time() - start_time
        lr_current = scheduler.get_last_lr()[0]

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} — "
                f"train: {avg_train:.4f} (mdn: {avg_mdn:.4f}, recon: {avg_recon:.4f}) | "
                f"val: {avg_val:.4f} | lr: {lr_current:.6f} | "
                f"time: {elapsed:.0f}s"
            )

    total_time = time.time() - start_time
    history["total_time_s"] = total_time

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "stats": {
            "feature_means": stats.feature_means.tolist(),
            "feature_stds": stats.feature_stds.tolist(),
        },
        "epoch": epochs,
        "history": history,
    }, checkpoint_dir / "final_model.pt")

    # Save history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"Training complete in {total_time:.0f}s — "
        f"best val_loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch'] + 1}"
    )

    return model, history


def load_world_model(
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[MarketWorldModel, dict]:
    """Load a trained world model from checkpoint.

    Returns:
        (model, metadata) tuple.
    """
    checkpoint_path = checkpoint_path or CHECKPOINT_DIR / "best_model.pt"
    device = device or get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = WorldModelConfig(**checkpoint["config"])
    model = MarketWorldModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded world model from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss {checkpoint.get('val_loss', '?')})"
    )

    return model, checkpoint
