"""Train the neural market world model on historical OHLCV data.

Usage:
    PYTHONPATH=. python scripts/train_world_model.py
    PYTHONPATH=. python scripts/train_world_model.py --epochs 100 --batch-size 128
"""

from __future__ import annotations

import argparse
import logging

from world_model.model import WorldModelConfig
from world_model.trainer import train_world_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural market world model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--n-gaussians", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = WorldModelConfig(
        latent_dim=args.latent_dim,
        gru_hidden_dim=args.gru_hidden,
        n_gaussians=args.n_gaussians,
        seq_len=args.seq_len,
    )

    logger.info("Starting world model training")
    logger.info(f"Config: latent={config.latent_dim}, gru={config.gru_hidden_dim}, "
                f"gaussians={config.n_gaussians}, seq_len={config.seq_len}")

    model, history = train_world_model(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Best val loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch'] + 1}")
    logger.info(f"Checkpoint saved to: checkpoints/world_model/")


if __name__ == "__main__":
    main()
