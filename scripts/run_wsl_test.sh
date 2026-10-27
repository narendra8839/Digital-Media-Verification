#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
~/venvs/dmv-gpu/bin/python scripts/test_wsl_gpu_inference.py
