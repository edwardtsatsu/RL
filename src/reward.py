"""Original reward function for the SAC-1 residential solar/battery project.

Attribution: subclasses citylearn.reward_function.RewardFunction (CityLearn
2.5.0, https://github.com/intelligent-environments-lab/CityLearn), which
supplies the env_metadata/central_agent plumbing. The reward equation and
term weighting below are original work.
"""
from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class SolarBatteryReward(RewardFunction):
    """r_t = -(e_t * p_t) - beta * max(0, e_t)^2 + gamma_sc * min(pv_t, d_t)

    e_t: net electricity consumption (kWh, +import/-export)
    p_t: electricity price ($/kWh)
    pv_t: solar generation (kW)
    d_t: household non-shiftable demand (kW)
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        cost_weight: float = 1.0,
        peak_penalty_weight: float = 0.1,
        self_consumption_weight: float = 0.05,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.cost_weight = cost_weight
        self.peak_penalty_weight = peak_penalty_weight
        self.self_consumption_weight = self_consumption_weight

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        rewards = []

        for observation in observations:
            consumption = observation["net_electricity_consumption"]
            price = observation["electricity_pricing"]
            solar = observation["solar_generation"]
            demand = observation["non_shiftable_load"]

            cost_term = -(consumption * price) * self.cost_weight
            peak_term = -self.peak_penalty_weight * max(0.0, consumption) ** 2
            self_consumption_term = self.self_consumption_weight * min(solar, demand)

            rewards.append(cost_term + peak_term + self_consumption_term)

        if self.central_agent:
            return [sum(rewards)]

        return rewards
