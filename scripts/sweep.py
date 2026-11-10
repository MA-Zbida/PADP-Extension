"""Run every variant x seed of an experiment, several at a time.

Examples:
    python scripts/sweep.py configs/experiments/e1_mechanism.yaml --dry-run
    python scripts/sweep.py configs/experiments/e1_mechanism.yaml --parallel 4
    python scripts/sweep.py configs/experiments/e1_mechanism.yaml --only coupled-k2-long indep-k2-long --seeds 0 1

Finished runs (final.pt present) are skipped and interrupted ones resume, so you can stop with
Ctrl+C and rerun the same command later. Each run's console output goes to its stdout.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from padp.utils.config_io import REPO_ROOT, load_experiment

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--parallel", type=int, default=4, help="runs at the same time (each uses one CPU thread)")
    parser.add_argument("--only", nargs="*", help="restrict to these variant names")
    parser.add_argument("--seeds", nargs="*", type=int, help="restrict to these seeds")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra overrides for every run")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--dry-run", action="store_true", help="list runs without starting them")
    args = parser.parse_args(argv)

    experiment = load_experiment(args.config)
    variants = [v for v, _ in experiment.variants() if not args.only or v in args.only]
    seeds = args.seeds if args.seeds else experiment.seeds
    runs = [(v, s) for v in variants for s in seeds]
    todo = [(v, s) for v, s in runs if not (experiment.run_dir(v, s, args.runs_root) / "final.pt").exists()]

    print(f"{experiment.name}: {len(runs)} runs ({len(variants)} variants x {len(seeds)} seeds), "
          f"{len(runs) - len(todo)} already finished, {len(todo)} to run")
    if args.dry_run:
        for v, s in runs:
            done = (experiment.run_dir(v, s, args.runs_root) / "final.pt").exists()
            print(f"  {'done' if done else 'todo'}  {v}  seed {s}")
        return 0

    def run(variant: str, seed: int) -> int:
        run_dir = experiment.run_dir(variant, seed, args.runs_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(TRAIN_SCRIPT), "--config", str(args.config), "--variant", variant,
               "--seed", str(seed), "--runs-root", str(args.runs_root)]
        if args.set:
            cmd += ["--set", *args.set]
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode

    start, failed = time.time(), []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(run, v, s): (v, s) for v, s in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            v, s = futures[future]
            code = future.result()
            status = "ok" if code == 0 else f"FAILED (exit {code}, see stdout.log)"
            if code != 0:
                failed.append((v, s))
            print(f"[{done}/{len(todo)}] {v} seed {s}: {status} | {(time.time() - start) / 60:.0f} min elapsed", flush=True)

    if failed:
        print(f"{len(failed)} run(s) failed; rerun the same command to retry them.")
        return 1
    print(f"All runs finished. Next: python scripts/analyze.py {experiment.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
