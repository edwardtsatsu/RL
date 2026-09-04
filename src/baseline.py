"""Original rule-based battery controller — the exam's required SAC-1 baseline.

Not derived from CityLearn's own RBC classes: this is a from-scratch policy
with the same act(...) -> action interface the evaluation harness uses for
the trained SAC agent, so both run through identical evaluation code.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RuleBasedBatteryController:
    """Charge from excess solar; discharge to cover a demand deficit while
    the state of charge is above a minimum reserve; otherwise hold."""

    min_soc_reserve: float = 0.1

    def act(self, building) -> np.ndarray:
        observation = building.observations()
        solar = observation["solar_generation"]
        demand = observation["non_shiftable_load"]
        soc = observation["electrical_storage_soc"]
        nominal_power = building.electrical_storage.nominal_power

        net = solar - demand

        if net > 0:
            action = min(1.0, net / nominal_power)
        elif soc > self.min_soc_reserve:
            action = -min(1.0, abs(net) / nominal_power)
        else:
            action = 0.0

        return np.array([action], dtype=np.float32)
