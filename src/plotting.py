"""Figure generation: training curves (mean +/- std across seeds) and a
baseline-vs-agent bar chart whose error bars are +/- 1 std across individual
evaluation episodes for both series, regenerated from the committed
results/logs and results/metrics data (never from re-running training/eval).
"""
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "results" / "logs"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"


def load_monitor_log(seed: int, logs_dir: Path = LOGS_DIR) -> pd.DataFrame:
    """Load an SB3 Monitor CSV (first line is a JSON header comment)."""
    path = logs_dir / f"seed{seed}.monitor.csv"
    return pd.read_csv(path, skiprows=1)


def training_curve_data(seeds: List[int], logs_dir: Path = LOGS_DIR) -> Dict[int, pd.DataFrame]:
    curves = {}
    for seed in seeds:
        df = load_monitor_log(seed, logs_dir=logs_dir)
        df["cumulative_steps"] = df["l"].cumsum()
        curves[seed] = df
    return curves


def interpolate_curves(curves: Dict[int, pd.DataFrame], grid: np.ndarray) -> np.ndarray:
    return np.array([np.interp(grid, df["cumulative_steps"], df["r"]) for df in curves.values()])


def plot_training_curves(
    seeds: List[int],
    logs_dir: Path = LOGS_DIR,
    output_path: Path = None,
) -> Path:
    output_path = output_path or FIGURES_DIR / "training_curves.png"
    curves = training_curve_data(seeds, logs_dir=logs_dir)

    max_steps = min(df["cumulative_steps"].iloc[-1] for df in curves.values())
    grid = np.linspace(0, max_steps, 200)
    interpolated = interpolate_curves(curves, grid)
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mean, label="Mean episode return")
    ax.fill_between(grid, mean - std, mean + std, alpha=0.3, label="+/- 1 std across seeds")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("SAC training return (mean +/- std across 3 seeds)")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_baseline_comparison(summary_path: Path = None, output_path: Path = None) -> Path:
    summary_path = summary_path or METRICS_DIR / "summary.json"
    output_path = output_path or FIGURES_DIR / "baseline_comparison.png"

    with open(summary_path) as f:
        summary = json.load(f)

    # Both series use the same statistic: spread across individual evaluation
    # episodes. SAC's seed-level std (summary["sac"]) is a different statistic
    # (n=3 per-seed means) and is deliberately not plotted against the
    # baseline's, which has no seed dimension. See README's results section.
    sac = summary["sac_window_level"]
    baseline = summary["baseline"]

    metric_keys = list(sac.keys())
    sac_means = [sac[k]["mean"] for k in metric_keys]
    sac_stds = [sac[k]["std"] for k in metric_keys]
    baseline_means = [baseline[k]["mean"] for k in metric_keys]
    baseline_stds = [baseline[k]["std"] for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, sac_means, width, yerr=sac_stds, capsize=4, label="SAC (all seeds x windows)")
    ax.bar(
        x + width / 2,
        baseline_means,
        width,
        yerr=baseline_stds,
        capsize=4,
        label="Rule-based baseline (windows)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=30, ha="right")
    ax.set_title(
        "SAC vs. rule-based baseline\n"
        "bars: mean; error bars: +/- 1 std across individual held-out evaluation episodes"
    )
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    seeds = [0, 1, 2]
    training_curves_path = plot_training_curves(seeds)
    comparison_path = plot_baseline_comparison()
    print(f"Wrote {training_curves_path}")
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
