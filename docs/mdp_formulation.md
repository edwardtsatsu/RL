# MDP Formulation — SAC-1: Residential Solar and Battery Energy Management

This document is the standalone, citable expansion of design-spec §3
(`docs/superpowers/specs/2026-09-04-sac1-solar-battery-design.md`), written
against the real, committed implementation (`src/environment.py`,
`src/reward.py`, `configs/sac_config.yaml`) rather than a generic
CityLearn/SAC description. It is intended to be cited directly from the
exam report's "Problem formulation" section.

Notation: time step `t`, state `s_t`, action `a_t`, reward `r_t`, discount
factor `γ`. The environment is a single-building CityLearn simulation
(`Building_1`, dataset `citylearn_challenge_2022_phase_1`), run with
`central_agent=True`, so the whole problem is a single-agent MDP even
though CityLearn is designed for multi-building settings.

## 1. State space

The raw, per-step observation is CityLearn's `active_observations` list,
written out explicitly in `src/environment.py::ACTIVE_OBSERVATIONS` (28
entries) rather than left implicit, so every included signal is a
reviewable, justified choice:

| Group | Observations |
|---|---|
| Calendar / time | `month`, `day_type`, `hour` |
| Weather (current + 1/2/3-step-ahead forecast) | `outdoor_dry_bulb_temperature` (+3 forecasts), `outdoor_relative_humidity` (+3 forecasts) |
| Solar irradiance (current + forecast) | `diffuse_solar_irradiance` (+3 forecasts), `direct_solar_irradiance` (+3 forecasts) |
| Grid / carbon | `carbon_intensity` |
| Building state | `non_shiftable_load`, `solar_generation`, `electrical_storage_soc`, `net_electricity_consumption` |
| Price (current + forecast) | `electricity_pricing` (+3 forecasts) |

That is 3 + 8 + 8 + 1 + 4 + 4 = 28 raw observations.

`build_env` (`src/environment.py`) wraps the raw `CityLearnEnv` in
`NormalizedObservationWrapper` (min-max normalisation of continuous
observations into `[0, 1]`, and a sine/cosine pair encoding of the three
periodic calendar observations — `month`, `day_type`, `hour` — so their
cyclical structure, e.g. hour 23 being adjacent to hour 0, is represented
correctly rather than as a raw integer discontinuity) and then, for
SB3-facing use, in `StableBaselines3Wrapper` (flattens the per-building
observation dict into a single array). The three periodic observations
each expand from 1 raw value to a 2-value encoding, so the flattened,
normalised observation space is

```
Box(0, 1, (31,))       # 28 raw observations - 3 periodic + 3*2 sin/cos dims = 31
```

confirmed directly by `tests/test_environment.py::test_training_env_has_expected_observation_and_action_spaces`
(and repeated for eval-window environments in
`test_build_eval_windows_returns_requested_count`), both asserting
`env.observation_space.shape == (31,)`.

Forecast components (1/2/3-step-ahead temperature, humidity, irradiance,
price) are included specifically because the agent needs to anticipate
near-future solar and price conditions to decide *now* whether to charge
or discharge — a purely present-instant state would make good
charge/discharge timing unobservable from `s_t` alone (see §6, Markov
property).

## 2. Action space

```
Box(-1, 1, (1,))
```

One continuous dimension: the fraction of the battery's nominal
charge/discharge power to apply this step. `a_t > 0` requests charging,
`a_t < 0` requests discharging, at that fraction of nominal power;
`a_t = 0` is hold. CityLearn's own `Battery`/`Building` internals clip the
requested action to the battery's physical charge/discharge power limit
and state-of-charge bounds before applying it, so `a_t` is a *request*,
not a guaranteed physical delta — the environment, not the agent, enforces
feasibility.

## 3. Reward function

Implemented as `SolarBatteryReward(citylearn.reward_function.RewardFunction)`
in `src/reward.py`, an original subclass (CityLearn supplies only the
`env_metadata`/`central_agent` plumbing via the base class):

```
r_t = -(e_t · p_t) − β · max(0, e_t)² + γ_sc · min(pv_t, d_t)
```

where, per step `t` and per the raw CityLearn observation dict:

- `e_t` = `net_electricity_consumption` (kWh; positive = import from grid,
  negative = export to grid)
- `p_t` = `electricity_pricing` ($/kWh)
- `pv_t` = `solar_generation` (kW)
- `d_t` = `non_shiftable_load`, i.e. household demand (kW)

Weights actually used for the real training run
(`configs/sac_config.yaml`, `reward:` block, identical to
`src/environment.py::REWARD_KWARGS_DEFAULT`):

| Weight | Value | Term |
|---|---|---|
| `cost_weight` | 1.0 | `-(e_t · p_t)`, negative grid electricity cost |
| `peak_penalty_weight` (`β`) | 0.1 | `-β · max(0, e_t)²`, quadratic penalty on grid *import* only (export, `e_t < 0`, is never penalised by this term) — a per-step, non-lookahead proxy for the episode-level peak-demand objective, since true peak demand is not a well-defined per-step reward |
| `self_consumption_weight` (`γ_sc`) | 0.05 | `γ_sc · min(pv_t, d_t)`, intended to reward solar self-consumption |

These are the weights actually passed into training (Task 7); no
retuning pass happened between the values chosen at design time and the
real training run, so "tuned" here means "the values training was
actually run with," not the result of a separate hyperparameter search.

### Honest limitation: the self-consumption term is policy-invariant

`self_consumption_term = γ_sc · min(pv_t, d_t)` is computed purely from
`solar_generation` and `non_shiftable_load` — both exogenous data at time
`t` that do not depend on `e_t`, the battery's action, or anything else
the agent controls. Charging the battery from excess solar versus letting
that same solar spill to the grid produces an *identical* value of
`min(pv_t, d_t)` at time `t`, because the term never looks at
`net_electricity_consumption`. Consequently this term cannot distinguish
a good battery decision from a bad (or absent) one, and it contributed
**no training signal** to the real 150,000-step-per-seed training run
whose models are committed under `results/models/`. This was discovered
during evaluation-metric work (Task 9/hotfix), root-caused in
`src/reward.py`, and discussed with the project owner, who decided: do
not retrain (the real training run stands as-is), but document the flaw
honestly here rather than silently leaving it implicit or working around
it only in the metric.

Note this is *not* the same formula as the evaluation metric
`solar_self_consumption_pct` (`src/metrics.py`), which was separately
fixed, post-training, to be genuinely policy-dependent — see
`src/metrics.py`'s own comment and §7's evaluation note below. The reward
function itself was deliberately left unchanged; only the reported metric
was corrected. The two are no longer the same self-consumption formula,
and only the metric one is policy-sensitive.

## 4. Termination and truncation

CityLearn signals the end of an episode for this environment via
`terminated=True` — **not** `truncated=True` — once the configured
time-step window is exhausted, even though the cause is a fixed-horizon
cutoff rather than a domain-specific failure/absorbing state. This is a
CityLearn-specific implementation detail (not the general Gymnasium
convention that a fixed-horizon cutoff should be reported as
`truncated`), and it was verified empirically rather than assumed:

- `tests/test_environment.py::test_training_env_runs_to_natural_end_after_6551_steps` —
  runs the training environment (window `[0, 6551]`) to completion under
  random actions and asserts `steps == 6551` and `terminated is True`.
- `tests/test_environment.py::test_eval_window_env_runs_exactly_71_steps_to_termination` —
  same check for a 72-step evaluation window (`[6552, 6623]`), asserting
  `steps == 71` and `terminated is True`.

Stated separately, as the exam requires:

- **Termination** (in the sense of a domain-specific end-state): none.
  The domain has no failure or absorbing state — the battery operates
  entirely within CityLearn's own physical clipping (§2), never triggering
  an episode-ending fault. CityLearn's `terminated=True` here is
  reporting *window exhaustion*, not a domain failure condition.
- **Truncation** (fixed-horizon cutoff): the actual mechanism ending
  every episode in this project. The training episode is a single pass
  over time steps `[0, 6551]` (6552 steps, ≈9 months); each evaluation
  episode is a single pass over its 72-step window (§5). Because CityLearn
  reports this via `terminated=True` rather than `truncated=True`, code
  that consumes these environments (`src/train.py`'s SB3 `Monitor`,
  `src/evaluate.py::run_episode`) must check `terminated or truncated`
  to detect episode end — checking `truncated` alone would silently never
  fire.

## 5. Train / evaluation split

Fully time-based, no shuffling, defined in `src/environment.py`:

- **Training window**: time steps `TRAIN_START_TIME_STEP=0` to
  `TRAIN_END_TIME_STEP=6551` inclusive (6552 steps ≈ 273 days ≈ 9 months),
  one continuous episode per training pass.
- **Held-out evaluation period**: time steps `6552`–`8759` (2208 steps ≈
  92 days), never seen during training.
- **Evaluation episodes**: the held-out period is sliced into
  `EVAL_NUM_WINDOWS=30` fixed, non-overlapping, contiguous windows of
  `EVAL_WINDOW_STEPS=72` steps each (`30 × 72 = 2160` of the 2208
  available steps; the trailing 48 steps are unused), computed by
  `eval_window_bounds(window_index)`. Window 0 is `[6552, 6623]`; window
  29 ends at or before step 8759. Each window is built as its own
  `CityLearnEnv` instance (`build_env(start, end)`), so the same 30
  windows are run, identically, for every SAC seed and for the rule-based
  baseline — the "same episodes, seeds and metric code" comparison the
  exam requires.
- **Evaluation mode**: SAC is evaluated with `model.predict(...,
  deterministic=True)` (policy mean, no exploration noise); the baseline
  is deterministic by construction (a fixed rule, no learned stochastic
  policy).

### Honest note on the reported evaluation `std`

`results/metrics/summary.json` reports `mean` and `std` for both `sac`
and `baseline` under the same field names, but these two `std` values are
**not the same statistic** and are not directly comparable despite
sharing a JSON key:

- **SAC's `std`** (`src/metrics.py::aggregate_across_seeds`) is the
  standard deviation, across the 3 trained seeds, of each seed's own
  per-seed mean over its 30 eval windows (n=3) — genuine seed-to-seed
  training variability.
- **Baseline's `std`** (`src/metrics.py::aggregate_metrics`) is the
  standard deviation across the 30 held-out evaluation windows for the
  single, deterministic baseline policy (n=30) — window-to-window
  variability, not seed variability, because the baseline has no learned
  parameters and therefore no seed dimension by construction; it was only
  ever evaluated once.

Reading both numbers as "the same kind of spread" would be a
methodological error. This asymmetry is inherent to comparing a
stochastically-trained agent against a deterministic rule-based baseline
under the exam's own required protocol, not a bug, but it must be flagged
wherever these numbers are quoted (see also `README.md`'s results
section).

## 6. Discount factor

`γ = 0.99` (`configs/sac_config.yaml::sac.gamma`, the Stable-Baselines3
SAC default, used as-is rather than overridden). Effective horizon
`1/(1-γ) = 100` hours ≈ 4.2 days. This is deliberately close to, and
slightly longer than, the diurnal solar/demand cycle (24 hours) and the
short-horizon (1–3 step-ahead) price/weather forecasts already included
in the state — long enough for the agent to value an overnight-charge /
morning-discharge strategy spanning roughly one full day-night cycle plus
margin, without requiring credit assignment across the entire ~6552-step
training trace, which would make the value function needlessly hard to
learn at this training budget (§7 below).

## 7. Training budget and seeds (for context)

Not part of the MDP itself, but recorded here because it affects what
"the trained policy" actually saw: `configs/sac_config.yaml` sets
`total_timesteps: 150000` per seed and `seeds: [0, 1, 2]`. One training
episode takes exactly 6551 `env.step()` calls to reach `terminated=True`
(§4; the window covers 6552 time steps, but `reset()` already returns the
observation for the first one, so only 6551 further steps are needed to
reach the last). `results/logs/seed{N}.monitor.csv` confirms each logged
episode has length 6551. 150,000 steps is therefore 22 complete episodes
per seed (22 × 6551 = 144,122 logged steps), plus a final partial episode
(5,878 steps) that SB3 does not log to `Monitor` because it never reaches
`terminated`/`truncated`. This training budget is explicitly scoped down
from published-benchmark scale, per the exam's own compute-constraint
allowance, and is not hidden in the report.

## 8. Markov property

Two design choices are relevant to whether `s_t` is a sufficient
statistic of history (i.e. whether this is genuinely Markovian rather
than only approximately so):

- **Forecast observations are included specifically to preserve the
  Markov property.** Good charge/discharge decisions depend on
  near-future solar, weather and price conditions, not just the current
  instant. If the state carried only *current* values (no forecast
  components), the environment would still be non-stationary from the
  agent's point of view in a way that current state alone cannot resolve
  — the agent would need memory of past observations to infer what is
  coming next, breaking the Markov property. Including the 1/2/3-step-ahead
  forecast components for temperature, humidity, irradiance and price
  (§1) makes the relevant near-future information part of `s_t` itself,
  so the current state, not history, is sufficient for an optimal
  decision at `t`.

- **Battery degradation is not tracked in the state — and its rate is
  small but not exactly zero.** `ACTIVE_OBSERVATIONS` does not include any
  cumulative-cycling or degraded-capacity observation, so if the
  battery's usable capacity changed measurably over the course of an
  episode as a function of how heavily it had been cycled, the state
  would silently become insufficient (identical `s_t` values would no
  longer imply identical future dynamics, since the "true" capacity would
  depend on unobserved cycling history). The dataset's own default
  battery configuration for `Building_1` (`citylearn_challenge_2022_phase_1`
  schema, `electrical_storage.attributes`) sets
  `capacity_loss_coefficient: 1e-05` — a very small but *non-zero* rate —
  and this project's code (`src/environment.py::build_env`) does not
  override it. So, correcting the design spec's earlier stated intention
  ("degradation fixed at 0"): the real, committed configuration leaves
  CityLearn's dataset default in place, `1e-05`, not exactly `0`. Over a
  6552-step training episode or a 72-step evaluation window this rate is
  practically negligible (on the order of hundredths of a percent of
  capacity across the whole training window), so the Markov-property
  violation it introduces is small in practice, but it is not literally
  absent, and this document states that plainly rather than repeating the
  earlier, inaccurate "fixed at 0" framing.

## 9. Summary table

| Element | Value |
|---|---|
| State space | `Box(0, 1, (31,))` — 28 active observations, 3 periodic (month/day_type/hour) sin/cos-encoded |
| Action space | `Box(-1, 1, (1,))` — signed fraction of battery nominal power |
| Reward | `r_t = -(e_t·p_t) − 0.1·max(0,e_t)² + 0.05·min(pv_t,d_t)` |
| Termination | none (no domain failure state); CityLearn reports window-exhaustion as `terminated=True` |
| Truncation | fixed-horizon window cutoff (train: 6552 steps; eval: 72 steps) — the actual end-of-episode mechanism, reported via `terminated`, not `truncated` |
| Discount | `γ = 0.99`, effective horizon ≈ 4.2 days |
| Markov property | preserved by including forecast observations; slightly weakened by a small (1e-05), unmodified default battery degradation rate not tracked in the state |
