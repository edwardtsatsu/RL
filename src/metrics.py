"""Metric computation and cross-run aggregation for the evaluation harness.

Used identically for the trained SAC agent and the rule-based baseline, so
the two are compared with the same metric code, per the exam's requirement.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def compute_episode_metrics(building) -> Dict[str, float]:
    """Compute the required evaluation metrics from a completed CityLearn episode."""
    consumption = np.array(building.net_electricity_consumption, dtype=float)
    pricing = np.array(building.pricing.electricity_pricing[: len(consumption)], dtype=float)
    # CityLearn's Building.solar_generation is a negative-value convention (generation reduces net consumption); take magnitude for a physical kW figure.
    solar = np.abs(np.array(building.solar_generation, dtype=float))

    grid_import = np.clip(consumption, 0, None)
    total_solar = solar.sum()
    # Self-consumption = generation actually used on-site (directly or via battery
    # buffering) rather than exported to the grid. This depends on the battery's
    # charge/discharge action (via `consumption`, which already reflects it),
    # unlike a naive min(solar, demand) overlap, which is policy-invariant and
    # cannot distinguish a trained agent from an untrained baseline.
    grid_export = np.clip(-consumption, 0, None)
    self_consumed_solar = total_solar - grid_export.sum()

    return {
        "grid_consumption_kwh": float(grid_import.sum()),
        "solar_self_consumption_pct": float(self_consumed_solar / total_solar * 100) if total_solar > 0 else 0.0,
        "electricity_cost": float((consumption * pricing).sum()),
        "peak_demand_kw": float(grid_import.max()),
    }


def aggregate_metrics(episode_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of each metric across a list of episodes."""
    if not episode_metrics:
        raise ValueError("episode_metrics must be non-empty")

    keys = episode_metrics[0].keys()
    aggregate = {}
    for key in keys:
        values = np.array([m[key] for m in episode_metrics], dtype=float)
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return aggregate


def aggregate_across_seeds(seed_means: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation, across seeds, of each metric's per-seed mean."""
    if not seed_means:
        raise ValueError("seed_means must be non-empty")

    keys = seed_means[0].keys()
    aggregate = {}
    for key in keys:
        values = np.array([sm[key]["mean"] for sm in seed_means], dtype=float)
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return aggregate


def save_metrics(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
