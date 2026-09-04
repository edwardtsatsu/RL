import json

import numpy as np
import pandas as pd
import pytest

from src.plotting import (
    interpolate_curves,
    plot_baseline_comparison,
    plot_training_curves,
)


def test_interpolate_curves_matches_known_points():
    df0 = pd.DataFrame({"cumulative_steps": [0, 10, 20], "r": [0.0, 5.0, 10.0]})
    df1 = pd.DataFrame({"cumulative_steps": [0, 10, 20], "r": [0.0, 10.0, 20.0]})
    grid = np.array([0, 10, 20])

    result = interpolate_curves({0: df0, 1: df1}, grid)

    assert result.shape == (2, 3)
    assert list(result[0]) == [0.0, 5.0, 10.0]
    assert list(result[1]) == [0.0, 10.0, 20.0]


def _write_fake_monitor_csv(path, rewards):
    with open(path, "w") as f:
        f.write('#{"t_start": 0.0, "env_id": "None"}\n')
        f.write("r,l,t\n")
        for r in rewards:
            f.write(f"{r},100,1.0\n")


def test_plot_training_curves_writes_a_png(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    _write_fake_monitor_csv(logs_dir / "seed0.monitor.csv", [1.0, 2.0, 3.0])
    _write_fake_monitor_csv(logs_dir / "seed1.monitor.csv", [1.5, 2.5, 3.5])
    output_path = tmp_path / "training_curves.png"

    result_path = plot_training_curves([0, 1], logs_dir=logs_dir, output_path=output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_baseline_comparison_writes_a_png(tmp_path):
    summary = {
        "sac": {"grid_consumption_kwh": {"mean": 10.0, "std": 1.0}},
        "baseline": {"grid_consumption_kwh": {"mean": 12.0, "std": 0.0}},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    output_path = tmp_path / "baseline_comparison.png"

    result_path = plot_baseline_comparison(summary_path=summary_path, output_path=output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
