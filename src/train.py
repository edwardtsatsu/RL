"""SAC training entry point.

Trains one Stable-Baselines3 SAC model per configured seed on the training-
period CityLearn environment, logging per-episode return via SB3's Monitor
wrapper so training curves can be plotted later (src/plotting.py).
"""
import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.environment import build_training_env

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sac_config.yaml"
MODELS_DIR = ROOT / "results" / "models"
LOGS_DIR = ROOT / "results" / "logs"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_seed(seed: int, config: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = build_training_env(reward_kwargs=config["reward"])
    env = Monitor(env, filename=str(LOGS_DIR / f"seed{seed}"))

    sac_kwargs = config["sac"]
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_kwargs["learning_rate"],
        buffer_size=sac_kwargs["buffer_size"],
        learning_starts=sac_kwargs["learning_starts"],
        batch_size=sac_kwargs["batch_size"],
        tau=sac_kwargs["tau"],
        gamma=sac_kwargs["gamma"],
        train_freq=sac_kwargs["train_freq"],
        gradient_steps=sac_kwargs["gradient_steps"],
        ent_coef=sac_kwargs["ent_coef"],
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=config["total_timesteps"])

    model_path = MODELS_DIR / f"sac_seed{seed}.zip"
    model.save(model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train SAC on the SAC-1 solar/battery environment.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    seeds = args.seeds if args.seeds is not None else config["seeds"]

    for seed in seeds:
        print(f"=== Training SAC seed {seed} ({config['total_timesteps']} steps) ===")
        model_path = train_one_seed(seed, config)
        print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
