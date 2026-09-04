from src.environment import build_env, build_training_env, eval_window_bounds, build_eval_windows


def test_training_env_has_expected_observation_and_action_spaces():
    env = build_training_env()

    assert env.observation_space.shape == (31,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_training_env_runs_to_natural_end_after_6551_steps():
    env = build_training_env()
    obs, info = env.reset()

    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps == 6551
    assert terminated is True


def test_eval_window_bounds_are_contiguous_non_overlapping_72_step_windows():
    bounds = [eval_window_bounds(i) for i in range(30)]

    assert bounds[0] == (6552, 6623)
    assert bounds[1] == (6624, 6695)
    for (start_a, end_a), (start_b, _end_b) in zip(bounds, bounds[1:]):
        assert end_a == start_b - 1
    assert bounds[-1][1] <= 8759


def test_eval_window_env_runs_exactly_71_steps_to_termination():
    start, end = eval_window_bounds(0)
    env = build_env(start, end)
    obs, info = env.reset()

    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps == 71
    assert terminated is True


def test_build_eval_windows_returns_requested_count():
    envs = build_eval_windows(num_windows=3)

    assert len(envs) == 3
    for env in envs:
        assert env.observation_space.shape == (31,)
