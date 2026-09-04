"""CityLearn environment construction for the SAC-1 solar/battery project.

Single building (Building_1) from the citylearn_challenge_2022_phase_1
dataset, central_agent=True (one continuous battery-rate action). The
active_observations list below is CityLearn's own default set for this
dataset/building, written out explicitly (rather than left implicit) so the
state representation is a reviewable, justified choice per the MDP
formulation in docs/mdp_formulation.md.
"""
from typing import List, Tuple

from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from src.reward import SolarBatteryReward

DATASET_NAME = "citylearn_challenge_2022_phase_1"
BUILDING = "Building_1"

TRAIN_START_TIME_STEP = 0
TRAIN_END_TIME_STEP = 6551
EVAL_START_TIME_STEP = 6552
EVAL_WINDOW_STEPS = 72
EVAL_NUM_WINDOWS = 30

ACTIVE_OBSERVATIONS = [
    "month", "day_type", "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_1",
    "outdoor_relative_humidity_predicted_2",
    "outdoor_relative_humidity_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
]

REWARD_KWARGS_DEFAULT = {
    "cost_weight": 1.0,
    "peak_penalty_weight": 0.1,
    "self_consumption_weight": 0.05,
}


def build_env(
    start_time_step: int,
    end_time_step: int,
    reward_kwargs: dict = None,
    wrap_for_sb3: bool = True,
):
    """Build a single-building CityLearnEnv for a fixed [start, end] time-step window."""
    reward_kwargs = reward_kwargs if reward_kwargs is not None else REWARD_KWARGS_DEFAULT

    # Pass a resolved schema dict, not the dataset name string — CityLearnEnv._load validates a string schema against DataSet.get_dataset_names(), which has a caching bug in citylearn==2.5.0 that hits the GitHub API unconditionally on every call and exhausts the unauthenticated rate limit.
    schema = DataSet().get_schema(DATASET_NAME)
    env = CityLearnEnv(
        schema,
        central_agent=True,
        buildings=[BUILDING],
        active_observations=ACTIVE_OBSERVATIONS,
        simulation_start_time_step=start_time_step,
        simulation_end_time_step=end_time_step,
        reward_function=SolarBatteryReward,
        reward_function_kwargs=reward_kwargs,
    )
    env = NormalizedObservationWrapper(env)

    if wrap_for_sb3:
        env = StableBaselines3Wrapper(env)

    return env


def build_training_env(reward_kwargs: dict = None):
    return build_env(TRAIN_START_TIME_STEP, TRAIN_END_TIME_STEP, reward_kwargs=reward_kwargs)


def eval_window_bounds(
    window_index: int,
    eval_start_time_step: int = EVAL_START_TIME_STEP,
    window_steps: int = EVAL_WINDOW_STEPS,
) -> Tuple[int, int]:
    """Inclusive (start, end) time-step bounds for held-out eval window `window_index` (0-indexed)."""
    start = eval_start_time_step + window_index * window_steps
    end = start + window_steps - 1
    return start, end


def build_eval_windows(
    num_windows: int = EVAL_NUM_WINDOWS,
    eval_start_time_step: int = EVAL_START_TIME_STEP,
    window_steps: int = EVAL_WINDOW_STEPS,
    reward_kwargs: dict = None,
) -> List:
    """Build the fixed, non-overlapping evaluation-window environments."""
    envs = []
    for window_index in range(num_windows):
        start, end = eval_window_bounds(window_index, eval_start_time_step, window_steps)
        envs.append(build_env(start, end, reward_kwargs=reward_kwargs))
    return envs
