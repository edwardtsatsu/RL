#!/usr/bin/env bash
# Single entry point reproducing the headline result from a clean environment.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements.txt
pip install -q --no-deps citylearn==2.5.0

python -m pytest tests/ -v

python -m src.train
python -m src.evaluate
python -m src.plotting

echo "Done. See results/metrics/summary.json and results/figures/*.png"
