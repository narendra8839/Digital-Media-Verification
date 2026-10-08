#!/bin/bash
set -e
source ~/venvs/dmv-gpu/bin/activate
echo "[$(date)] Starting full 700-video deepfake test evaluation..."
PYTHONPATH=. python -u scripts/phase4_evaluate.py 2>&1 | tee phase4_eval.log
echo "[$(date)] Starting balanced deepfake robustness evaluation..."
PYTHONPATH=. python -u scripts/phase4_robustness.py 2>&1 | tee phase4_robustness.log
echo "[$(date)] All Phase 4 evaluations completed successfully!"
