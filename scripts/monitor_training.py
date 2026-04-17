"""Live training monitor — clean dashboard for GRPO training.

Reads MLflow metrics and displays a compact, auto-refreshing summary.
Run in a second terminal on the same machine while training is active.

Usage:
    python scripts/monitor_training.py                    # auto-detect latest run
    python scripts/monitor_training.py --run-id <id>      # specific run
    python scripts/monitor_training.py --interval 10      # refresh every 10s
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class TrainingSnapshot:
    step: int
    max_steps: int
    loss: float | None
    reward_mean: float | None
    reward_std: float | None
    kl: float | None
    lr: float | None
    grad_norm: float | None
    epoch: float | None
    elapsed_min: float
    run_name: str
    status: str


def get_latest_run(client: MlflowClient, experiment_name: str) -> str | None:
    """Find the most recent active or finished run."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def fetch_latest_metric(
    client: MlflowClient, run_id: str, key: str,
) -> float | None:
    """Get the most recent value for a metric key."""
    try:
        history = client.get_metric_history(run_id, key)
        if history:
            return history[-1].value
    except Exception:
        pass
    return None


def fetch_snapshot(client: MlflowClient, run_id: str) -> TrainingSnapshot:
    """Build a training snapshot from MLflow metrics."""
    run = client.get_run(run_id)
    params = run.data.params
    max_steps = int(params.get("max_steps", 500))
    run_name = run.info.run_name or run_id[:8]
    status = run.info.status

    start_time = run.info.start_time / 1000  # ms -> s
    elapsed_min = (time.time() - start_time) / 60

    # TRL/HF Trainer logs these metric keys
    metric_keys = {
        "loss": ["loss", "train/loss"],
        "reward_mean": ["reward", "rewards/mean", "reward_mean"],
        "reward_std": ["reward_std", "rewards/std"],
        "kl": ["kl", "kl_divergence"],
        "lr": ["learning_rate"],
        "grad_norm": ["grad_norm"],
        "epoch": ["epoch"],
    }

    metrics: dict[str, float | None] = {}
    for field, candidates in metric_keys.items():
        val = None
        for key in candidates:
            val = fetch_latest_metric(client, run_id, key)
            if val is not None:
                break
        metrics[field] = val

    # Infer step from epoch or from metric history length
    step = 0
    if metrics["epoch"] is not None:
        step = int(metrics["epoch"] * max_steps)
    else:
        # Fall back to counting loss entries
        try:
            history = client.get_metric_history(run_id, "loss")
            if history:
                step = int(history[-1].step)
        except Exception:
            pass

    return TrainingSnapshot(
        step=step,
        max_steps=max_steps,
        loss=metrics["loss"],
        reward_mean=metrics["reward_mean"],
        reward_std=metrics["reward_std"],
        kl=metrics["kl"],
        lr=metrics["lr"],
        grad_norm=metrics["grad_norm"],
        epoch=metrics["epoch"],
        elapsed_min=elapsed_min,
        run_name=run_name,
        status=status,
    )


def format_val(val: float | None, fmt: str = ".4f") -> str:
    if val is None:
        return "  --  "
    return f"{val:{fmt}}"


def render(snap: TrainingSnapshot) -> str:
    """Render a compact dashboard string."""
    pct = snap.step / max(snap.max_steps, 1) * 100
    bar_width = 30
    filled = int(bar_width * snap.step / max(snap.max_steps, 1))
    bar = "#" * filled + "-" * (bar_width - filled)

    # ETA
    if snap.step > 0 and snap.elapsed_min > 0:
        rate = snap.elapsed_min / snap.step  # min/step
        remaining = (snap.max_steps - snap.step) * rate
        eta = f"{remaining:.0f}m"
    else:
        eta = "--"

    lines = [
        "",
        f"  GRPO Training Monitor  |  {snap.run_name}  |  {snap.status}",
        f"  {'=' * 58}",
        f"  Step: {snap.step}/{snap.max_steps}  [{bar}]  {pct:.1f}%",
        f"  Elapsed: {snap.elapsed_min:.1f}m  |  ETA: {eta}",
        f"  {'─' * 58}",
        f"  Loss:      {format_val(snap.loss)}",
        f"  Reward:    {format_val(snap.reward_mean)}  (std: {format_val(snap.reward_std)})",
        f"  KL:        {format_val(snap.kl)}",
        f"  LR:        {format_val(snap.lr, '.2e')}",
        f"  Grad norm: {format_val(snap.grad_norm)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO Training Monitor")
    parser.add_argument("--run-id", help="MLflow run ID (auto-detects latest if omitted)")
    parser.add_argument(
        "--experiment", default="grpo-v2.1-training",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri", default="file:///workspace/mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Refresh interval in seconds (default: 30)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    run_id = args.run_id
    if not run_id:
        run_id = get_latest_run(client, args.experiment)
        if not run_id:
            print(f"No runs found in experiment '{args.experiment}'")
            print(f"Tracking URI: {args.tracking_uri}")
            sys.exit(1)

    print(f"Monitoring run: {run_id}")
    print(f"Refresh every {args.interval}s  (Ctrl+C to stop)")

    try:
        while True:
            snap = fetch_snapshot(client, run_id)
            # Clear screen
            os.system("clear" if os.name != "nt" else "cls")
            print(render(snap))

            if snap.status == "FINISHED":
                print("  Training complete.")
                break
            if snap.status == "FAILED":
                print("  Training FAILED. Check logs.")
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
