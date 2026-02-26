#!/usr/bin/env python3
"""
Run Experiment Part 1 (CIFAR-10 cGAN ablation).
Usage (from repo root):
  python run_experiment/run_part1.py

Runs experiment_part1/test.py with cwd=experiment_part1. Outputs go to
outputs/part1/. Seed: 17.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PART1_DIR = REPO_ROOT / "experiment_part1"
TEST_SCRIPT = PART1_DIR / "test.py"


def main():
    if not TEST_SCRIPT.exists():
        print(f"Error: Part 1 script not found: {TEST_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not (PART1_DIR / "data").exists():
        print("Note: experiment_part1/data/ not found. Place CIFAR-10 data there (e.g. cifar-10-batches-py/).", file=sys.stderr)
    print(f"Running Experiment Part 1 (cwd={PART1_DIR})...")
    sys.exit(subprocess.run([sys.executable, str(TEST_SCRIPT)], cwd=str(PART1_DIR)).returncode)


if __name__ == "__main__":
    main()
