"""Generate publication-quality training curves from saved training logs.

Reads logs from results/ directory (extracted from HF checkpoint trainer_state.json)
and produces a 4-panel figure showing the complete training story.

Usage:
    PYTHONPATH=. python scripts/plot_training_curves.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# -- Color palette (consistent across all visuals) --
C_SFT = "#3498db"       # blue for SFT
C_GRPO = "#2ecc71"      # green for GRPO success
C_FAIL = "#e74c3c"      # red for failures/limits
C_NEURAL = "#2ecc71"    # green for neural env
C_STATIC = "#3498db"    # blue for static env
C_GRAY = "#95a5a6"      # gray for annotations
C_BG = "#fafafa"        # off-white background


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_sft_loss(ax: plt.Axes, sft_log: list[dict]) -> None:
    """Panel A: SFT v3 loss curve with best/overtrained markers."""
    steps = [e["step"] for e in sft_log]
    losses = [e["loss"] for e in sft_log]

    ax.plot(steps, losses, color=C_SFT, linewidth=2.5, zorder=3)
    ax.fill_between(steps, losses, alpha=0.08, color=C_SFT)

    # Best checkpoint marker (step 200, score 0.399)
    best_idx = next(i for i, e in enumerate(sft_log) if e["step"] == 200)
    ax.axvline(x=200, color=C_GRPO, linestyle="--", alpha=0.7, linewidth=1.5)
    ax.plot(200, losses[best_idx], "o", color=C_GRPO, markersize=10, zorder=5)
    ax.annotate(
        f"Best checkpoint\nscore 0.399",
        xy=(200, losses[best_idx]), xytext=(120, losses[best_idx] + 0.15),
        fontsize=8.5, fontweight="bold", color=C_GRPO,
        arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.2),
    )

    # Overtrained marker (step 352 would be here, but log only goes to 200)
    # Add annotation pointing to the right edge
    ax.annotate(
        "Step 352: loss 2.01\nbut score dropped to 0.383",
        xy=(200, losses[-1]), xytext=(140, 2.65),
        fontsize=7.5, color=C_FAIL, style="italic",
        arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=1, alpha=0.6),
    )

    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("SFT: Teaching the Model to Trade", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)


def plot_kl_divergence(ax: plt.Axes, grpo_log: list[dict]) -> None:
    """Panel B: GRPO KL divergence stability."""
    steps = [e["step"] for e in grpo_log]
    kl = [e["kl"] for e in grpo_log]

    ax.plot(steps, kl, color=C_GRPO, linewidth=2, zorder=3)
    ax.fill_between(steps, kl, alpha=0.1, color=C_GRPO)

    # Danger zone from previous runs
    ax.axhline(y=3.0, color=C_FAIL, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        155, 3.15, "GRPO v3 collapsed here (KL=4.2)",
        fontsize=8, color=C_FAIL, fontweight="bold", ha="center",
    )

    # Max KL annotation
    max_kl = max(kl)
    max_step = steps[kl.index(max_kl)]
    ax.annotate(
        f"Max: {max_kl:.2f}",
        xy=(max_step, max_kl), xytext=(max_step - 60, max_kl + 0.4),
        fontsize=8, color=C_GRPO,
        arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1),
    )

    # Safe zone label
    ax.fill_between([0, 300], [0, 0], [0.5, 0.5], alpha=0.05, color=C_GRPO)
    ax.text(
        150, 0.05, "Safe zone (beta=0.05)",
        fontsize=8, color=C_GRPO, alpha=0.7, ha="center",
    )

    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("KL Divergence", fontsize=10)
    ax.set_title("GRPO: KL Divergence Under Control", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 3.8)
    ax.tick_params(labelsize=9)


def plot_trading_reward(ax: plt.Axes, grpo_log: list[dict]) -> None:
    """Panel C: Trading reward signal with variance band."""
    steps = [e["step"] for e in grpo_log]
    reward_mean = [e["rewards/trading_reward/mean"] for e in grpo_log]
    reward_std = [e["rewards/trading_reward/std"] for e in grpo_log]

    upper = [m + s for m, s in zip(reward_mean, reward_std)]
    lower = [m - s for m, s in zip(reward_mean, reward_std)]

    ax.fill_between(steps, lower, upper, alpha=0.15, color=C_GRPO, label="Reward std")
    ax.plot(steps, reward_mean, color=C_GRPO, linewidth=2, label="Trading reward", zorder=3)
    ax.axhline(y=0, color=C_GRAY, linestyle="-", linewidth=0.8, alpha=0.5)

    # Smoothed trend line
    window = 5
    if len(reward_mean) >= window:
        smoothed = np.convolve(reward_mean, np.ones(window) / window, mode="valid")
        smooth_steps = steps[window - 1:]
        ax.plot(smooth_steps, smoothed, color="#27ae60", linewidth=2.5,
                linestyle="-", alpha=0.8, label="Trend (5-step avg)", zorder=4)

    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("Trading Reward", fontsize=10)
    ax.set_title("GRPO: Trading Reward Signal", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.tick_params(labelsize=9)


def plot_learning_curve(ax: plt.Axes) -> None:
    """Panel D: Final scoreboard — Base vs SFT vs GRPO."""
    models = ["Base\nDeepSeek 7B", "SFT v3\n(distilled)", "GRPO Neural\n(BEST)"]
    static = [0.300, 0.399, 0.470]
    neural = [0.298, 0.417, 0.537]

    x = np.arange(len(models))
    width = 0.32

    bars1 = ax.bar(x - width / 2, static, width, label="Static Env",
                   color=C_STATIC, edgecolor="white", linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + width / 2, neural, width, label="Neural Env",
                   color=C_NEURAL, edgecolor="white", linewidth=1.5, zorder=3)

    # HOLD floor
    ax.axhline(y=0.300, color=C_FAIL, linestyle="--", linewidth=1.2, alpha=0.5)
    ax.text(2.45, 0.305, "HOLD floor", fontsize=7.5, color=C_FAIL, alpha=0.7)

    # Score labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold",
                color=C_STATIC)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold",
                color="#1a8a4a")

    # Improvement arrow
    ax.annotate(
        "", xy=(2.16, 0.53), xytext=(0.16, 0.30),
        arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=2, connectionstyle="arc3,rad=0.15"),
    )
    ax.text(1.1, 0.48, "+79%", fontsize=11, fontweight="bold", color=C_GRAY, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Result: Base 0.300 → GRPO 0.537", fontsize=11, fontweight="bold")
    ax.set_ylim(0.2, 0.62)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=9)


def main() -> None:
    sft_path = RESULTS_DIR / "sft_v3_training_log.json"
    grpo_path = RESULTS_DIR / "grpo_neural_training_log.json"

    if not sft_path.exists() or not grpo_path.exists():
        logger.error("Training logs not found in results/. Run extraction first.")
        return

    sft_log = load_json(sft_path)
    grpo_log = load_json(grpo_path)

    logger.info("SFT log: %d entries, GRPO log: %d entries", len(sft_log), len(grpo_log))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("white")

    for ax in axes.flat:
        ax.set_facecolor(C_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_sft_loss(axes[0, 0], sft_log)
    plot_kl_divergence(axes[0, 1], grpo_log)
    plot_trading_reward(axes[1, 0], grpo_log)
    plot_learning_curve(axes[1, 1])

    fig.suptitle(
        "Training the Trading Agent: SFT → GRPO Against Neural Environment",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = RESULTS_DIR / "training_curves_final.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info("Saved to %s", output_path)

    plt.show()


if __name__ == "__main__":
    main()
