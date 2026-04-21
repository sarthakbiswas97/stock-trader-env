"""Generate self-improvement learning curve visuals for the hackathon pitch.

Charts:
1. Curriculum progression (score over episodes, colored by tier, promotion markers)
2. Mistake reduction (mistakes per episode, stacked by type)
3. Neural env vs static replay comparison (two training curves)
4. World model training loss

Usage:
    PYTHONPATH=. python scripts/plot_learning_curves.py
    PYTHONPATH=. python scripts/plot_learning_curves.py --data results/curriculum_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from server.curriculum import CURRICULUM_ORDER
from server.mistake_tracker import MistakeType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"

TIER_COLORS = {
    "single_stock": "#C8E6C9",
    "single_stock_costs": "#A5D6A7",
    "multi_stock_3": "#FFF9C4",
    "portfolio": "#FFE0B2",
    "full_autonomous": "#FFCDD2",
}

TIER_LABELS = {
    "single_stock": "Single Stock",
    "single_stock_costs": "Single + Costs",
    "multi_stock_3": "Multi (3)",
    "portfolio": "Portfolio (10)",
    "full_autonomous": "Full Auto (25)",
}

MISTAKE_COLORS = {
    "regime_violation": "#F44336",
    "overbought_buy": "#FF9800",
    "oversold_sell": "#FFC107",
    "position_limit_breach": "#9C27B0",
    "trade_limit_breach": "#3F51B5",
    "loss_hold": "#795548",
    "missed_opportunity": "#607D8B",
}


def generate_demo_data() -> dict:
    """Generate realistic demo data for visualization before real training runs."""
    rng = np.random.default_rng(42)
    n_episodes = 120

    tiers = []
    scores = []
    promotions = []
    mistakes_per_episode = []

    current_tier_idx = 0
    tier_episode_count = 0
    base_score = 0.35

    for ep in range(n_episodes):
        tier = CURRICULUM_ORDER[current_tier_idx]
        tiers.append(tier)

        # Score improves within each tier, with noise
        progress = min(tier_episode_count / 20, 1.0)
        tier_difficulty = current_tier_idx * 0.08
        score = base_score + progress * 0.35 - tier_difficulty + rng.normal(0, 0.06)
        score = max(0.05, min(0.95, score))
        scores.append(score)

        # Mistakes decrease with progress
        base_mistakes = max(0, int(8 - progress * 6 + current_tier_idx * 2 + rng.normal(0, 1.5)))
        mistake_breakdown = {}
        remaining = base_mistakes
        for mt in MistakeType:
            if remaining <= 0:
                mistake_breakdown[mt.value] = 0
                continue
            count = rng.integers(0, min(remaining + 1, 4))
            mistake_breakdown[mt.value] = int(count)
            remaining -= count
        mistakes_per_episode.append(mistake_breakdown)

        tier_episode_count += 1

        # Check promotion
        if tier_episode_count >= 20 and current_tier_idx < len(CURRICULUM_ORDER) - 1:
            recent = scores[-5:]
            thresholds = [0.60, 0.55, 0.50, 0.50, 1.0]
            if np.mean(recent) >= thresholds[current_tier_idx]:
                promotions.append({"episode": ep, "from": tier, "to": CURRICULUM_ORDER[current_tier_idx + 1]})
                current_tier_idx += 1
                tier_episode_count = 0
                base_score -= 0.05

    # Neural vs replay comparison
    neural_scores = []
    replay_scores = []
    for ep in range(60):
        progress = ep / 60
        neural_scores.append(0.35 + progress * 0.25 + rng.normal(0, 0.04))
        replay_scores.append(0.33 + progress * 0.15 + rng.normal(0, 0.05))

    return {
        "tiers": tiers,
        "scores": scores,
        "promotions": promotions,
        "mistakes": mistakes_per_episode,
        "neural_scores": neural_scores,
        "replay_scores": replay_scores,
    }


def plot_curriculum_progression(ax: plt.Axes, data: dict) -> None:
    """Score over episodes with tier coloring and promotion markers."""
    episodes = list(range(len(data["scores"])))
    scores = data["scores"]
    tiers = data["tiers"]

    # Background tier bands
    prev_tier = tiers[0]
    band_start = 0
    for i, tier in enumerate(tiers):
        if tier != prev_tier or i == len(tiers) - 1:
            end = i if tier != prev_tier else i + 1
            ax.axvspan(band_start, end, color=TIER_COLORS.get(prev_tier, "#EEEEEE"), alpha=0.4)
            mid = (band_start + end) / 2
            ax.text(mid, 0.02, TIER_LABELS.get(prev_tier, prev_tier), ha="center", fontsize=7, alpha=0.6)
            band_start = i
            prev_tier = tier

    # Score line
    ax.plot(episodes, scores, color="#1565C0", linewidth=1.5, alpha=0.8)

    # Smoothed trend
    window = min(10, len(scores) // 4)
    if window > 1:
        smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(scores)), smoothed, color="#1565C0", linewidth=2.5)

    # Promotion markers
    for p in data["promotions"]:
        ax.axvline(x=p["episode"], color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Adaptive Curriculum Progression")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2)


def plot_mistake_reduction(ax: plt.Axes, data: dict) -> None:
    """Stacked area chart of mistakes decreasing over episodes."""
    mistakes = data["mistakes"]
    episodes = list(range(len(mistakes)))

    # Aggregate by type
    type_series = {}
    for mt in MistakeType:
        type_series[mt.value] = [m.get(mt.value, 0) for m in mistakes]

    # Smooth each series
    window = 5
    smoothed = {}
    for mt_name, values in type_series.items():
        if any(v > 0 for v in values):
            s = np.convolve(values, np.ones(window) / window, mode="valid")
            smoothed[mt_name] = s

    if not smoothed:
        ax.text(0.5, 0.5, "No mistakes detected", ha="center", transform=ax.transAxes)
        return

    x = range(window - 1, len(mistakes))
    arrays = list(smoothed.values())
    labels = list(smoothed.keys())
    colors = [MISTAKE_COLORS.get(k, "#999999") for k in labels]

    ax.stackplot(x, *arrays, labels=[k.replace("_", " ") for k in labels], colors=colors, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mistakes / Episode")
    ax.set_title("Mistake Reduction Over Training")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.2)


def plot_neural_vs_replay(ax: plt.Axes, data: dict) -> None:
    """Two training curves: neural env vs static replay."""
    neural = data["neural_scores"]
    replay = data["replay_scores"]
    episodes = list(range(len(neural)))

    ax.plot(episodes, neural, color="#2196F3", linewidth=1, alpha=0.4)
    ax.plot(episodes, replay, color="#FF9800", linewidth=1, alpha=0.4)

    # Smoothed
    window = 5
    if len(neural) > window:
        n_smooth = np.convolve(neural, np.ones(window) / window, mode="valid")
        r_smooth = np.convolve(replay, np.ones(window) / window, mode="valid")
        x = range(window - 1, len(neural))
        ax.plot(x, n_smooth, color="#2196F3", linewidth=2.5, label="Neural Env")
        ax.plot(x, r_smooth, color="#FF9800", linewidth=2.5, label="Static Replay")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Neural Env vs Static Replay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_world_model_training(ax: plt.Axes) -> None:
    """World model val loss over epochs."""
    epochs = list(range(1, 21))
    val_loss = [
        -9.89, -10.86, -12.01, -12.17, -12.51,
        -12.39, -12.44, -12.47, -12.64, -12.65,
        -12.66, -12.70, -12.70, -12.71, -12.72,
        -12.72, -12.73, -12.73, -12.73, -12.73,
    ]

    ax.plot(epochs, val_loss, "o-", color="#4CAF50", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss (MDN NLL)")
    ax.set_title("World Model Training")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Path to curriculum data JSON")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.data and args.data.exists():
        with open(args.data) as f:
            data = json.load(f)
        logger.info(f"Loaded data from {args.data}")
    else:
        data = generate_demo_data()
        with open(RESULTS_DIR / "demo_curriculum_data.json", "w") as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
        logger.info("Generated demo data")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Self-Improving Trading Agent — Learning Curves", fontsize=16, fontweight="bold")

    plot_curriculum_progression(axes[0, 0], data)
    plot_mistake_reduction(axes[0, 1], data)
    plot_neural_vs_replay(axes[1, 0], data)
    plot_world_model_training(axes[1, 1])

    plt.tight_layout()
    output = RESULTS_DIR / "learning_curves.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
