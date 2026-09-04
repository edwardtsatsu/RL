# SAC-1: Residential Solar and Battery Energy Management

DSCD614 Reinforcement Learning (University of Ghana) group-exam
implementation of **Option SAC-1 — Residential Solar and Battery Energy
Management**: a Soft Actor-Critic (SAC) agent, trained with
Stable-Baselines3 on a single-building [CityLearn](https://github.com/intelligent-environments-lab/CityLearn)
simulation, learns a continuous battery charge/discharge policy that
coordinates a household's rooftop solar generation and battery storage
against grid electricity price and demand, and is compared against an
original rule-based baseline controller on a held-out evaluation period.
Full problem formulation (state/action/reward/termination/discount/Markov
property) is written up in [`docs/mdp_formulation.md`](docs/mdp_formulation.md).

## Installation

Requires Python 3.12+ (developed and run on macOS/arm64).

CityLearn must be installed in **two steps**, in this order, because of a
packaging problem in CityLearn's own dependency declaration:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps citylearn==2.5.0
```

**Why two steps, not one `pip install citylearn`:** CityLearn ≥2.2.0
declares `doe-xstock>=1.1.0` as a hard dependency, which in turn requires
`openstudio<=3.3.0` — a package with no installable wheel for
macOS/Python 3.12 (arm64). `doe-xstock`/`openstudio` are only used by
CityLearn's ResStock *dataset-building* pipeline (constructing new
building datasets from scratch); this project only *loads* CityLearn's
already-packaged `citylearn_challenge_2022_phase_1` dataset and never
touches that pipeline, so the dependency is unnecessary here but still
breaks a plain `pip install citylearn==2.5.0` on this platform. The fix is
to install CityLearn's actually-required runtime dependencies explicitly
first (`requirements.txt` pins every one of them, resolved and verified
working: `numpy`, `gymnasium==0.28.1`, `pandas`, `scikit-learn`, `scipy`,
`PyYAML`, `simplejson`, `matplotlib`, `torch`, `stable_baselines3`, etc.),
then install `citylearn==2.5.0` itself with `--no-deps` so pip never
attempts to resolve `doe-xstock`/`openstudio` at all.

Verify the install:

```bash
python -m pytest tests/ -v
```

### Note: dataset loading avoids CityLearn's own GitHub rate-limit bug

CityLearn's usual documented pattern is
`CityLearnEnv("citylearn_challenge_2022_phase_1", ...)` — passing the
dataset **name as a string**. This project's `src/environment.py`
(`build_env`) does **not** do that; it instead pre-resolves the dataset
with `DataSet().get_schema(DATASET_NAME)` and passes the resulting
**schema dict** to `CityLearnEnv`. This is a deliberate workaround, not a
stylistic choice: `CityLearnEnv._load`'s string-schema code path validates
the name against `DataSet.get_dataset_names()`, which — due to a caching
bug in the installed `citylearn==2.5.0` — calls the GitHub API
*unconditionally* on every invocation regardless of its own local cache.
Since every `CityLearnEnv(...)` construction in this project (training,
each of the 30 evaluation windows, per seed, per policy) would otherwise
trigger one such call, this exhausts an unauthenticated GitHub API rate
limit quickly once training/evaluation run repeatedly. Passing a
pre-resolved schema dict instead of the string skips that validation path
entirely — same resulting environment, no network dependency. If you
copy the string-based pattern from CityLearn's own docs/examples instead
of using `src/environment.py::build_env`, you will hit this rate limit
differently (and possibly less predictably, since it depends on how much
of the unauthenticated GitHub quota is already used from other activity
on the same IP).

## Training

Train all 3 seeds for the full configured budget (150,000 environment
steps each, `configs/sac_config.yaml`):

```bash
python -m src.train
```

Fast smoke check (one seed, 200 steps — checks the training pipeline
runs end-to-end without waiting for a real training run):

```bash
python -m src.train --seeds 0 --total-timesteps 200
```

Trained models are written to `results/models/sac_seed{N}.zip`; per-step
episode returns are logged to `results/logs/seed{N}.monitor.csv` via SB3's
`Monitor` wrapper.

**Note:** the models and logs already committed in `results/` are the
real 3-seed, 150,000-step-per-seed training run this project reports on
(see Results below). Re-running `python -m src.train` overwrites those
files with a new run — expect different, seed-driven numbers, not a
byte-for-byte reproduction.

## Evaluation

```bash
python -m src.evaluate
```

Evaluates the rule-based baseline (`src/baseline.py`) and each trained SAC
seed (`results/models/sac_seed{N}.zip`, loaded with
`model.predict(..., deterministic=True)`) over the same 30 fixed,
non-overlapping, held-out 72-step windows (`src/environment.py`'s
`eval_window_bounds`), through identical evaluation and metric code
(`src/evaluate.py`, `src/metrics.py`). Writes
`results/metrics/baseline.json`, `results/metrics/sac_seed{N}.json`, and
the aggregated `results/metrics/summary.json`.

## Reproducing the figures

```bash
python -m src.plotting
```

Regenerates both figures directly from already-committed result files
(never by re-running training or evaluation):

| Figure | Traces back to |
|---|---|
| `results/figures/training_curves.png` — mean ± std episode return across the 3 seeds over training | `results/logs/seed0.monitor.csv`, `seed1.monitor.csv`, `seed2.monitor.csv` |
| `results/figures/baseline_comparison.png` — SAC vs. rule-based baseline, all 4 metrics, mean ± std error bars | `results/metrics/summary.json` |

## One-command reproduction

```bash
./scripts/run_all.sh
```

Creates a fresh virtualenv, installs dependencies (including the
`--no-deps` CityLearn workaround above), runs the test suite, then trains
all 3 seeds, evaluates, and regenerates both figures — end to end, from a
clean environment. As with `python -m src.train` above, this **overwrites**
the committed `results/models/`, `results/logs/`, `results/metrics/`, and
`results/figures/` with a fresh run; it reproduces the *pipeline*, not
necessarily the exact numbers below (SAC training is stochastic even with
fixed seeds, across machines/library-version drift).

## Results overview

Real evaluation results from the committed run (`results/metrics/summary.json`),
SAC (mean ± std across 3 seeds) vs. rule-based baseline (mean ± std across
the 30 held-out evaluation windows):

| Metric | SAC | Baseline |
|---|---|---|
| Electricity cost | 9.11 ± 0.41 | 10.60 ± 5.68 |
| Grid consumption (kWh) | 56.40 ± 0.89 | 54.36 ± 13.67 |
| Peak demand (kW) | 4.83 ± 0.28 | 4.66 ± 1.69 |
| Solar self-consumption (%) | 51.50 ± 0.93 | 62.84 ± 8.90 |

Read honestly: at this training budget (150,000 steps/seed, explicitly
scoped down from published-benchmark scale — see
`docs/mdp_formulation.md` §7), SAC does not clearly outperform the
rule-based baseline on every metric — it is lower on electricity cost, but
higher on grid consumption and does not beat the baseline's mean
self-consumption. What SAC does show consistently is **markedly lower
variance** across seeds than the baseline shows across windows on 3 of
the 4 metrics (cost, grid consumption, peak demand). This is reported as
found, not adjusted to look better.

**Two caveats needed to read these numbers correctly:**

1. **The `std` columns above are not the same statistic.** SAC's `std` is
   the standard deviation, across the 3 trained seeds, of each seed's own
   mean over its 30 eval windows (n=3) — seed-to-seed variability. The
   baseline's `std` is the standard deviation across the 30 held-out
   windows for one deterministic policy (n=30) — window-to-window
   variability, because the baseline has no seed dimension by
   construction (it was only evaluated once). They share a JSON field
   name (`std`) but are not directly comparable spreads. Full detail in
   `docs/mdp_formulation.md` §5.
2. **The reward term intended to encourage solar self-consumption
   (`src/reward.py`, `self_consumption_weight · min(solar, demand)`) is
   policy-invariant** — it depends only on exogenous solar/demand data at
   time `t`, never on the battery's action, so it could not have taught
   the agent anything about self-consumption during training. The
   `solar_self_consumption_pct` metric reported above uses a *different*,
   genuinely policy-dependent formula (generation minus grid export,
   `src/metrics.py`) fixed after training; the reward function itself was
   deliberately left unchanged and untrained on this signal. Full detail
   in `docs/mdp_formulation.md` §3.

## Repository layout

```
reinforcementProject/
├── README.md                    (this file)
├── spec.md                      (original exam prompt handed to the implementation)
├── requirements.txt             (pinned runtime dependencies; excludes citylearn, see Installation)
├── DSCD614 - Reinforcement Learning 3 (1).pdf   (exam brief)
├── src/
│   ├── environment.py           (CityLearnEnv factory: dataset load, single-building
│   │                              restriction, wrapper stack, train/eval-window construction)
│   ├── reward.py                 (SolarBatteryReward — original RewardFunction subclass)
│   ├── baseline.py               (RuleBasedBatteryController — original rule-based policy)
│   ├── train.py                  (SAC training entry point; loops configured seeds)
│   ├── evaluate.py               (evaluation harness: 30 fixed windows, deterministic eval,
│   │                              SAC and baseline through identical code path)
│   ├── metrics.py                (per-episode metric computation + cross-seed aggregation)
│   └── plotting.py               (training-curve and baseline-comparison figures)
├── configs/
│   └── sac_config.yaml           (hyperparameter source of truth: seeds, total_timesteps,
│                                   every SB3 SAC constructor argument, reward weights, split bounds)
├── scripts/
│   └── run_all.sh                (single entry point: install → test → train → evaluate → plot)
├── tests/                        (pytest suite for every src/ module)
├── results/
│   ├── logs/                     (SB3 Monitor CSVs per seed, committed — real training run)
│   ├── models/                   (trained SAC weights per seed, committed — real training run)
│   ├── metrics/                  (per-seed and aggregated evaluation metrics, committed)
│   └── figures/                  (generated plots, regenerable from results/logs + results/metrics)
└── docs/
    └── mdp_formulation.md         (full state/action/reward/termination/discount/Markov write-up)
```

## Attribution

- **[CityLearn](https://github.com/intelligent-environments-lab/CityLearn) 2.5.0**
  — the building/district energy simulation environment this project
  builds on (`citylearn.citylearn.CityLearnEnv`, `citylearn.data.DataSet`,
  `citylearn.wrappers.NormalizedObservationWrapper`,
  `citylearn.wrappers.StableBaselines3Wrapper`,
  `citylearn.reward_function.RewardFunction` as the base class subclassed
  by `src/reward.py`). Not modified; used as an installed library.
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 2.3.2**
  — the `SAC` implementation (`"MlpPolicy"`, default 2×256-unit MLP
  actor/critics) used for training (`src/train.py`) and deterministic
  policy evaluation (`src/evaluate.py`). Not modified; used as an
  installed library.
- **Original work**: the reward function (`src/reward.py`), the
  rule-based baseline controller (`src/baseline.py`, not derived from
  CityLearn's own RBC classes), the environment factory and train/eval
  window construction (`src/environment.py`), the metric computation and
  aggregation (`src/metrics.py`), the training/evaluation/plotting entry
  points (`src/train.py`, `src/evaluate.py`, `src/plotting.py`), and this
  documentation.
