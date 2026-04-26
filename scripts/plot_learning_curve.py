"""Plot the co-evolution learning curve for the world model.

Shows how the neural simulator improves at predicting real market
dynamics over fine-tuning iterations.

Usage:
    PYTHONPATH=. python scripts/plot_learning_curve.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_learning_curve(history_path: Path | None = None) -> None:
    """Generate learning curve plots from co-evolution history."""
    history_path = history_path or RESULTS_DIR / "coevolution_history.json"

    with open(history_path) as f:
        history = json.load(f)

    iterations = history["iterations"]
    metrics = history["metrics"]

    dir_acc = [m["direction_accuracy"] for m in metrics]
    vol_ratio = [m["volatility_ratio"] for m in metrics]
    mae = [m["mae_returns"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Neural Environment Co-Evolution: Learning Curve", fontsize=14, fontweight="bold")

    # Direction accuracy
    axes[0].plot(iterations, dir_acc, "o-", color="#2196F3", linewidth=2, markersize=6)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Direction Accuracy")
    axes[0].set_title("Market Direction Prediction")
    axes[0].set_ylim(0.3, 0.8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Volatility ratio (lower = better, target = 1.0)
    axes[1].plot(iterations, vol_ratio, "o-", color="#FF5722", linewidth=2, markersize=6)
    axes[1].axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect (1.0x)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Volatility Ratio (gen/real)")
    axes[1].set_title("Volatility Calibration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # MAE returns (lower = better)
    axes[2].plot(iterations, mae, "o-", color="#4CAF50", linewidth=2, markersize=6)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("MAE (Daily Returns)")
    axes[2].set_title("Prediction Error")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = RESULTS_DIR / "learning_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved learning curve to {output_path}")
    plt.close(fig)

    # Also save a summary comparison table
    baseline = metrics[0]
    final = metrics[-1]
    improvement = {
        "direction_accuracy": f"{baseline['direction_accuracy']:.3f} → {final['direction_accuracy']:.3f}",
        "volatility_ratio": f"{baseline['volatility_ratio']:.2f}x → {final['volatility_ratio']:.2f}x",
        "mae_returns": f"{baseline['mae_returns']:.4f} → {final['mae_returns']:.4f}",
    }

    summary_path = RESULTS_DIR / "coevolution_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "baseline": baseline,
            "final": final,
            "improvement": improvement,
            "iterations": len(iterations) - 1,
        }, f, indent=2)

    logger.info(f"Saved summary to {summary_path}")
    for k, v in improvement.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    plot_learning_curve()
