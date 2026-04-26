"""Generate comparison visuals for the hackathon pitch.

Charts:
1. Training loss curve (transformer 20 epochs)
2. Architecture comparison (transformer vs CNN+GRU)
3. Generated vs real price path (one episode)
4. Temperature sweep (volatility calibration)

Usage:
    PYTHONPATH=. python scripts/plot_results.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_training_curve(ax: plt.Axes) -> None:
    """Training loss curve — transformer 20 epochs on A5000."""
    epochs = list(range(1, 21))
    val_loss = [
        -9.89, -10.86, -12.01, -12.17, -12.51,
        -12.39, -12.44, -12.47, -12.64, -12.65,
        -12.66, -12.70, -12.70, -12.71, -12.72,
        -12.72, -12.73, -12.73, -12.73, -12.73,
    ]

    ax.plot(epochs, val_loss, "o-", color="#2196F3", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MDN NLL)")
    ax.set_title("World Model Training")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"Best: {val_loss[-1]:.2f}",
        xy=(20, val_loss[-1]),
        xytext=(15, val_loss[0] + 0.5),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )


def plot_architecture_comparison(ax: plt.Axes) -> None:
    """Bar chart: transformer vs CNN+GRU on key metrics."""
    metrics = ["Vol Ratio\n(target: 1.0x)", "MAE Returns\n(lower=better)", "Direction\nAccuracy"]
    transformer = [0.94, 0.0167, 0.492]
    cnn_gru = [3.15, 0.0436, 0.501]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, transformer, width, label="Transformer (1.2M)", color="#2196F3")
    bars2 = ax.bar(x + width / 2, cnn_gru, width, label="CNN+GRU (1M)", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Architecture Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, transformer):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}" if val < 1 else f"{val:.2f}x", ha="center", fontsize=8)
    for bar, val in zip(bars2, cnn_gru):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}" if val < 1 else f"{val:.2f}x", ha="center", fontsize=8)


def plot_price_path(ax: plt.Axes) -> None:
    """Generated vs real price path for one episode."""
    from server.neural_simulator import NeuralSimulator, _get_device, _load_model, CHECKPOINT_DIR
    from server.market_simulator import MarketSimulator

    device = _get_device()
    model = _load_model(CHECKPOINT_DIR / "best_transformer.pt", device)

    sim = NeuralSimulator("single_stock", seed=10, temperature=1.0, model=model, device=device)
    sim.reset()

    gen_prices = []
    for _ in range(sim.episode_days):
        gen_prices.append(sim.get_price("RELIANCE"))
        sim.advance_day()

    truth = sim.get_ground_truth("RELIANCE")
    true_prices = truth["close"].values[:len(gen_prices)]

    days = range(1, len(gen_prices) + 1)
    ax.plot(days, true_prices, "o-", color="#4CAF50", linewidth=2, markersize=4, label="Real Market")
    ax.plot(days, gen_prices, "s--", color="#2196F3", linewidth=2, markersize=4, label="Neural Env")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price (Rs)")
    ax.set_title("Generated vs Real Price Path")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_temperature_sweep(ax: plt.Axes) -> None:
    """Volatility ratio vs temperature."""
    temps = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    vol_ratios = [0.39, 0.52, 0.59, 0.76, 1.02, 1.26]

    ax.plot(temps, vol_ratios, "o-", color="#FF5722", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect (1.0x)")
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Volatility Ratio (gen/real)")
    ax.set_title("Temperature Calibration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "Optimal: 1.0",
        xy=(1.0, 1.02),
        xytext=(1.2, 0.6),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Neural Market Environment — World Model Results", fontsize=16, fontweight="bold")

    plot_training_curve(axes[0, 0])
    plot_architecture_comparison(axes[0, 1])
    plot_price_path(axes[1, 0])
    plot_temperature_sweep(axes[1, 1])

    plt.tight_layout()
    output_path = RESULTS_DIR / "comparison_results.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved comparison results to {output_path}")


if __name__ == "__main__":
    main()
