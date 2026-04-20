"""Co-evolution: fine-tune the world model against ground truth via autoregressive training.

The model generates multi-step rollouts using its own predictions, then
compares with real data and backpropagates. This closes the train-eval gap
where single-step training doesn't improve autoregressive generation.

Usage:
    PYTHONPATH=. python scripts/coevolve_world_model.py
    PYTHONPATH=. python scripts/coevolve_world_model.py --iterations 20 --episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from server.market_simulator import TASK_STOCKS, _load_stock_data
from server.neural_simulator import NeuralSimulator
from world_model.data import N_FEATURES, N_PRICE_FEATURES, ohlcv_to_features
from world_model.model import mdn_loss
from world_model.trainer import CHECKPOINT_DIR, get_device, load_world_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
TRAIN_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "INFY", "ICICIBANK", "ITC", "KOTAKBANK"]
ROLLOUT_STEPS = 10


def evaluate_model(
    model: torch.nn.Module,
    device: torch.device,
    n_episodes: int = 50,
    temperature: float = 0.3,
    seed: int = 200,
) -> dict[str, float]:
    """Measure autoregressive generation quality against ground truth."""
    maes, dirs, vols = [], [], []

    for i in range(n_episodes):
        sim = NeuralSimulator("single_stock", seed=seed + i, temperature=temperature)
        sim._model = model
        sim._device = device
        sim.reset()

        for _ in range(sim.episode_days):
            sim.advance_day()

        errors = sim.compute_prediction_error("RELIANCE")
        if errors:
            maes.append(errors["mae_returns"])
            dirs.append(errors["direction_accuracy"])
            vols.append(errors["volatility_ratio"])

    return {
        "mae_returns": float(np.mean(maes)),
        "direction_accuracy": float(np.mean(dirs)),
        "volatility_ratio": float(np.mean(vols)),
    }


def fine_tune_autoregressive(
    model: torch.nn.Module,
    device: torch.device,
    features_by_symbol: dict[str, np.ndarray],
    n_windows: int = 50,
    lr: float = 1e-5,
    seed: int = 42,
) -> float:
    """Fine-tune using autoregressive rollouts against ground truth."""
    rng = np.random.default_rng(seed)
    seq_len = model.config.seq_len

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_loss = 0.0
    n_samples = 0

    for sym, features in features_by_symbol.items():
        max_start = len(features) - seq_len - ROLLOUT_STEPS
        if max_start < 1:
            continue

        starts = rng.integers(0, max_start, size=n_windows)

        for start in starts:
            seq = torch.from_numpy(features[start : start + seq_len]).unsqueeze(0).to(device)
            losses = []
            current = seq.clone()

            for step in range(ROLLOUT_STEPS):
                target_idx = start + seq_len + step
                if target_idx >= len(features):
                    break

                target = torch.from_numpy(
                    features[target_idx, :N_PRICE_FEATURES]
                ).unsqueeze(0).to(device)

                _, pi, mu, sigma, _ = model(current)
                losses.append(mdn_loss(pi, mu, sigma, target))

                # Autoregressive: feed model's own prediction with derived features
                with torch.no_grad():
                    predicted = model.dynamics.sample(pi, mu, sigma, temperature=0.3)
                p = predicted[0]
                intraday_range = max(float(p[1] - p[2]), 0.001)
                body = abs(float(p[3] - p[0]))
                body_ratio = min(body / intraday_range, 0.5)

                new_feat = torch.zeros(1, 1, N_FEATURES, device=device)
                new_feat[0, 0, :N_PRICE_FEATURES] = p
                new_feat[0, 0, N_PRICE_FEATURES] = intraday_range
                new_feat[0, 0, N_PRICE_FEATURES + 1] = body_ratio
                current = torch.cat([current[:, 1:, :], new_feat], dim=1)

            if losses:
                loss = sum(losses) / len(losses)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                n_samples += 1

    model.eval()
    return total_loss / max(n_samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-evolve world model against ground truth")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--windows", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    model, _ = load_world_model(CHECKPOINT_DIR / "best_model.pt", device)
    logger.info(f"Loaded model: {model.count_parameters():,} params on {device}")

    # Load training data
    features_by_symbol = {}
    for sym in TRAIN_SYMBOLS:
        try:
            features_by_symbol[sym] = ohlcv_to_features(_load_stock_data(sym))
        except FileNotFoundError:
            logger.warning(f"Skipping {sym}: data not found")
    logger.info(f"Training on {len(features_by_symbol)} symbols")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline
    baseline = evaluate_model(model, device, n_episodes=args.episodes, temperature=args.temperature)
    logger.info(
        f"Baseline — dir: {baseline['direction_accuracy']:.3f}, "
        f"vol: {baseline['volatility_ratio']:.2f}x, mae: {baseline['mae_returns']:.4f}"
    )

    history = {"iterations": [0], "metrics": [baseline], "train_loss": []}
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        ft_loss = fine_tune_autoregressive(
            model, device, features_by_symbol,
            n_windows=args.windows, lr=args.lr,
            seed=args.seed + iteration * 1000,
        )
        history["train_loss"].append(ft_loss)

        if iteration % args.eval_every == 0 or iteration == 1:
            metrics = evaluate_model(
                model, device, n_episodes=args.episodes, temperature=args.temperature,
            )
            history["iterations"].append(iteration)
            history["metrics"].append(metrics)

            elapsed = time.time() - start_time
            logger.info(
                f"Iter {iteration:2d} — loss: {ft_loss:.4f}, "
                f"dir: {metrics['direction_accuracy']:.3f}, "
                f"vol: {metrics['volatility_ratio']:.2f}x, "
                f"mae: {metrics['mae_returns']:.4f} — {elapsed:.0f}s"
            )

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
    }, CHECKPOINT_DIR / "coevolved_model.pt")

    with open(RESULTS_DIR / "coevolution_history.json", "w") as f:
        json.dump(history, f, indent=2)

    final = history["metrics"][-1]
    logger.info("=== Summary ===")
    logger.info(f"  Direction: {baseline['direction_accuracy']:.3f} → {final['direction_accuracy']:.3f}")
    logger.info(f"  Volatility: {baseline['volatility_ratio']:.2f}x → {final['volatility_ratio']:.2f}x")
    logger.info(f"  MAE: {baseline['mae_returns']:.4f} → {final['mae_returns']:.4f}")
    logger.info(f"  Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
