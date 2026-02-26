#!/usr/bin/env python3
"""
Run Experiment Part 2–3 (multi-dataset cGAN pipeline).
Usage (from repo root):
  python run_experiment/run_part23.py --dataset_key mnist
  python run_experiment/run_part23.py --dataset_key sports_ball --label_noise_ps "0.0,0.1,0.2,0.3"

Forwards all arguments to tools/run_pipeline.py. Outputs go to outputs/part23/. Seed: 42.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = REPO_ROOT / "tools" / "run_pipeline.py"


def main():
    if not PIPELINE_SCRIPT.exists():
        print(f"Error: Part 2–3 pipeline not found: {PIPELINE_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) == 1:
        print("Usage: python run_experiment/run_part23.py --dataset_key {mnist|sports_ball|animals} [options]")
        print("Example: python run_experiment/run_part23.py --dataset_key mnist --out_root results")
        sys.exit(1)
    cmd = [sys.executable, str(PIPELINE_SCRIPT)] + sys.argv[1:]
    sys.exit(subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode)


if __name__ == "__main__":
    main()
