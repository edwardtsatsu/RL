import numpy as np

from src.baseline import RuleBasedBatteryController


class _StubStorage:
    def __init__(self, nominal_power):
        self.nominal_power = nominal_power


class _StubBuilding:
    def __init__(self, pv, demand, soc, nominal_power=5.0):
        self._obs = {
            "solar_generation": pv,
            "non_shiftable_load": demand,
            "electrical_storage_soc": soc,
        }
        self.electrical_storage = _StubStorage(nominal_power)

    def observations(self):
        return self._obs


def test_charges_when_solar_exceeds_demand():
    controller = RuleBasedBatteryController()
    building = _StubBuilding(pv=3.0, demand=1.0, soc=0.5, nominal_power=5.0)

    action = controller.act(building)

    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert action[0] == np.float32(min(1.0, 2.0 / 5.0))


def test_discharges_when_demand_exceeds_solar_and_soc_above_reserve():
    controller = RuleBasedBatteryController(min_soc_reserve=0.1)
    building = _StubBuilding(pv=0.5, demand=2.0, soc=0.5, nominal_power=5.0)

    action = controller.act(building)

    assert action[0] == np.float32(-min(1.0, 1.5 / 5.0))


def test_holds_when_demand_exceeds_solar_and_soc_at_or_below_reserve():
    controller = RuleBasedBatteryController(min_soc_reserve=0.1)
    building = _StubBuilding(pv=0.5, demand=2.0, soc=0.05, nominal_power=5.0)

    action = controller.act(building)

    assert action[0] == 0.0


def test_action_magnitude_clips_to_one():
    controller = RuleBasedBatteryController()
    building = _StubBuilding(pv=10.0, demand=0.0, soc=0.5, nominal_power=5.0)

    action = controller.act(building)

    assert action[0] == 1.0
