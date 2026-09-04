from src.reward import SolarBatteryReward


def _reward(central_agent=True, **weights):
    return SolarBatteryReward({"central_agent": central_agent}, **weights)


def test_pure_grid_import_is_penalized_by_cost_and_peak_terms():
    reward_fn = _reward(cost_weight=1.0, peak_penalty_weight=0.1, self_consumption_weight=0.05)
    observations = [{
        "net_electricity_consumption": 2.0,
        "electricity_pricing": 0.2,
        "solar_generation": 0.0,
        "non_shiftable_load": 2.0,
    }]

    reward = reward_fn.calculate(observations)

    expected = -(2.0 * 0.2) - 0.1 * (2.0 ** 2) + 0.05 * min(0.0, 2.0)
    assert reward == [expected]


def test_solar_self_consumption_increases_reward():
    reward_fn = _reward(cost_weight=1.0, peak_penalty_weight=0.1, self_consumption_weight=0.05)
    obs_no_solar = [{"net_electricity_consumption": 2.0, "electricity_pricing": 0.2,
                      "solar_generation": 0.0, "non_shiftable_load": 2.0}]
    obs_with_solar = [{"net_electricity_consumption": 0.0, "electricity_pricing": 0.2,
                        "solar_generation": 2.0, "non_shiftable_load": 2.0}]

    r_no_solar = reward_fn.calculate(obs_no_solar)[0]
    r_with_solar = reward_fn.calculate(obs_with_solar)[0]

    assert r_with_solar > r_no_solar


def test_grid_export_is_not_penalized_by_the_peak_term():
    reward_fn = _reward(cost_weight=1.0, peak_penalty_weight=0.1, self_consumption_weight=0.05)
    observations = [{
        "net_electricity_consumption": -1.5,
        "electricity_pricing": 0.2,
        "solar_generation": 3.0,
        "non_shiftable_load": 1.5,
    }]

    reward = reward_fn.calculate(observations)

    expected = -(-1.5 * 0.2) - 0.1 * 0.0 + 0.05 * min(3.0, 1.5)
    assert reward == [expected]


def test_central_agent_sums_across_buildings():
    reward_fn = _reward(central_agent=True, cost_weight=1.0, peak_penalty_weight=0.0, self_consumption_weight=0.0)
    observations = [
        {"net_electricity_consumption": 1.0, "electricity_pricing": 0.1, "solar_generation": 0.0, "non_shiftable_load": 1.0},
        {"net_electricity_consumption": 2.0, "electricity_pricing": 0.1, "solar_generation": 0.0, "non_shiftable_load": 2.0},
    ]

    reward = reward_fn.calculate(observations)

    assert reward == [-(1.0 * 0.1) - (2.0 * 0.1)]
