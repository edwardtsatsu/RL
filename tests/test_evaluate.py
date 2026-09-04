from src.baseline import RuleBasedBatteryController
from src.evaluate import baseline_act_fn, evaluate_policy, run_episode
from src.environment import build_env, eval_window_bounds


def test_run_episode_returns_the_required_metric_keys():
    start, end = eval_window_bounds(0)
    env = build_env(start, end)
    act_fn = baseline_act_fn(RuleBasedBatteryController())

    metrics = run_episode(env, act_fn)

    assert set(metrics.keys()) == {
        "grid_consumption_kwh",
        "solar_self_consumption_pct",
        "electricity_cost",
        "peak_demand_kw",
    }


def test_baseline_act_fn_returns_action_within_bounds():
    start, end = eval_window_bounds(0)
    env = build_env(start, end)
    act_fn = baseline_act_fn(RuleBasedBatteryController())
    obs, info = env.reset()

    action = act_fn(env, obs)

    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0


def test_evaluate_policy_runs_the_requested_number_of_windows():
    act_fn = baseline_act_fn(RuleBasedBatteryController())

    results = evaluate_policy(act_fn, num_windows=2)

    assert len(results) == 2
    for episode_metrics in results:
        assert episode_metrics["grid_consumption_kwh"] >= 0
