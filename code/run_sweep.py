"""
Run all 12 experiment combinations:
  3 tasks × (1 full fine-tune + 3 LoRA ranks) = 12 runs

Usage:
    python code/run_sweep.py [--seed 42] [--verify]
"""

import argparse
import itertools
import subprocess
import sys


TASKS = ["sst2", "qnli", "rte"]
RANKS = [4, 8, 16]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="Pass --verify flag to each LoRA run for sanity checks")
    return parser.parse_args()


def main():
    args = parse_args()

    runs = []
    for task in TASKS:
        # Full fine-tuning baseline
        runs.append({"task": task, "mode": "full", "rank": None})
        # LoRA rank sweep
        for r in RANKS:
            runs.append({"task": task, "mode": "lora", "rank": r})

    print(f"Total runs: {len(runs)}")

    for i, run in enumerate(runs, 1):
        task, mode, rank = run["task"], run["mode"], run["rank"]
        label = f"{task}_{mode}" + (f"_r{rank}" if rank else "")
        print(f"\n{'='*60}")
        print(f"[{i}/{len(runs)}] Starting: {label}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "code/run_experiment.py",
            "--task", task,
            "--mode", mode,
            "--seed", str(args.seed),
        ]
        if mode == "lora":
            cmd += ["--rank", str(rank)]
            if args.verify:
                cmd += ["--verify"]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: run {label} failed with code {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
