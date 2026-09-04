import json

import numpy as np
import pytest

from src.metrics import (
    aggregate_across_seeds,
    aggregate_metrics,
    compute_episode_metrics,
    save_metrics,
)


class _StubPricing:
    def __init__(self, prices):
        self.electricity_pricing = prices


class _StubBuilding:
    def __init__(self, consumption, prices, solar, demand):
        self.net_electricity_consumption = consumption
        self.pricing = _StubPricing(prices)
        self.solar_generation = solar
        self.non_shiftable_load = demand


def test_compute_episode_metrics_known_values():
    building = _StubBuilding(
        consumption=[2.0, -1.0, 3.0],
        prices=[0.1, 0.2, 0.3],
        solar=[0.0, 3.0, 0.0],
        demand=[2.0, 2.0, 3.0],
    )

    metrics = compute_episode_metrics(building)

    assert metrics["grid_consumption_kwh"] == pytest.approx(2.0 + 0.0 + 3.0)
    assert metrics["peak_demand_kw"] == pytest.approx(3.0)
    assert metrics["electricity_cost"] == pytest.approx(2.0 * 0.1 + (-1.0) * 0.2 + 3.0 * 0.3)
    # self-consumption: (total solar - total grid export) / total solar generated.
    # Grid export per step = max(0, -consumption): [0, 1.0, 0] here, summing to 1.0.
    expected_self_consumption_pct = (3.0 - 1.0) / 3.0 * 100
    assert metrics["solar_self_consumption_pct"] == pytest.approx(expected_self_consumption_pct)


def test_compute_episode_metrics_handles_citylearns_negative_solar_convention():
    # CityLearn's real Building.solar_generation is negative-valued (generation
    # reduces net consumption). This reproduces that convention directly.
    building = _StubBuilding(
        consumption=[2.0, -1.0, 3.0],
        prices=[0.1, 0.2, 0.3],
        solar=[0.0, -3.0, 0.0],
        demand=[2.0, 2.0, 3.0],
    )

    metrics = compute_episode_metrics(building)

    # Same physical scenario as test_compute_episode_metrics_known_values (solar
    # magnitude 3.0 at index 1), just expressed in CityLearn's real sign convention.
    expected_self_consumption_pct = (0.0 + 2.0 + 0.0) / 3.0 * 100
    assert metrics["solar_self_consumption_pct"] == pytest.approx(expected_self_consumption_pct)
    assert metrics["solar_self_consumption_pct"] > 0.0


def test_compute_episode_metrics_handles_zero_solar_without_dividing_by_zero():
    building = _StubBuilding(
        consumption=[1.0, 1.0],
        prices=[0.1, 0.1],
        solar=[0.0, 0.0],
        demand=[1.0, 1.0],
    )

    metrics = compute_episode_metrics(building)

    assert metrics["solar_self_consumption_pct"] == 0.0


def test_compute_episode_metrics_credits_battery_charging_as_self_consumption():
    # Demand is 0 but net consumption is 0 too (the battery fully absorbs the
    # generated solar with no grid export). A naive demand-overlap formula
    # would wrongly say 0% self-consumption here (min(solar, demand=0) = 0);
    # the correct (export-based) formula must recognise the battery-absorbed
    # solar as fully self-consumed.
    building = _StubBuilding(
        consumption=[0.0],
        prices=[0.1],
        solar=[-5.0],  # CityLearn's negative-value convention; magnitude 5.0
        demand=[0.0],
    )

    metrics = compute_episode_metrics(building)

    assert metrics["solar_self_consumption_pct"] == pytest.approx(100.0)


def test_aggregate_metrics_computes_mean_and_std():
    episode_metrics = [
        {"grid_consumption_kwh": 10.0, "peak_demand_kw": 2.0},
        {"grid_consumption_kwh": 20.0, "peak_demand_kw": 4.0},
    ]

    aggregate = aggregate_metrics(episode_metrics)

    assert aggregate["grid_consumption_kwh"]["mean"] == pytest.approx(15.0)
    assert aggregate["grid_consumption_kwh"]["std"] == pytest.approx(np.std([10.0, 20.0]))
    assert aggregate["peak_demand_kw"]["mean"] == pytest.approx(3.0)


def test_aggregate_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_metrics([])


def test_aggregate_across_seeds_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_across_seeds([])


def test_aggregate_across_seeds_computes_mean_and_std_of_seed_means():
    seed_means = [
        {"grid_consumption_kwh": {"mean": 10.0, "std": 1.0}},
        {"grid_consumption_kwh": {"mean": 20.0, "std": 1.0}},
        {"grid_consumption_kwh": {"mean": 30.0, "std": 1.0}},
    ]

    aggregate = aggregate_across_seeds(seed_means)

    assert aggregate["grid_consumption_kwh"]["mean"] == pytest.approx(20.0)
    assert aggregate["grid_consumption_kwh"]["std"] == pytest.approx(np.std([10.0, 20.0, 30.0]))


def test_save_metrics_writes_readable_json(tmp_path):
    path = tmp_path / "nested" / "metrics.json"

    save_metrics(path, {"a": 1})

    assert json.loads(path.read_text()) == {"a": 1}
