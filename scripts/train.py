"""Train one run.

Examples:
    python scripts/train.py --config configs/experiments/smoke.yaml
    python scripts/train.py --config configs/experiments/e1_mechanism.yaml --variant coupled-k2-long --seed 3
    python scripts/train.py --config configs/base.yaml --set train.sampler=coupled env.n_agents=3 --name my_test

Outputs go to runs/<experiment>/<variant>/seed<seed>/. Rerunning an interrupted run resumes it
from latest.pt; a finished run (final.pt present) is skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from padp.marl.trainer import Trainer
from padp.utils.config_io import REPO_ROOT, build_configs, load_experiment


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="experiment YAML or a plain config such as configs/base.yaml")
    parser.add_argument("--variant", help="variant name (required if the experiment has several)")
    parser.add_argument("--seed", type=int, default=None, help="seed (default: the experiment's first seed)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings, e.g. train.lr=1e-4")
    parser.add_argument("--name", help="experiment name for ad-hoc runs (default: the config's name)")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--force", action="store_true", help="delete an existing run directory and start over")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    experiment = load_experiment(args.config)
    if args.name:
        experiment.name = args.name
    variants = [name for name, _ in experiment.variants()]
    variant = args.variant or (variants[0] if len(variants) == 1 else None)
    if variant is None:
        print(f"{experiment.name} has {len(variants)} variants; pass --variant. Choices:\n  " + "\n  ".join(variants))
        return 2
    seed = experiment.seeds[0] if args.seed is None else args.seed

    config = experiment.resolve(variant, seed, args.set)
    env_cfg, train_cfg = build_configs(config)
    run_dir = experiment.run_dir(variant, seed, args.runs_root)

    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "final.pt").exists():
        print(f"Already finished: {run_dir} (use --force to retrain)")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "latest.pt").exists():
        # A run that stopped before its first checkpoint starts over: drop its partial logs so the
        # new rows are not appended after them.
        for stale in ("metrics.csv", "eval.csv"):
            (run_dir / stale).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "tb", ignore_errors=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"experiment": experiment.name, "variant": variant, "seed": seed, **config}, f, sort_keys=False)

    trainer = Trainer(env_cfg, train_cfg, run_dir, tensorboard=not args.no_tensorboard)
    if (run_dir / "latest.pt").exists():
        trainer.load(run_dir / "latest.pt")
        print(f"Resuming from step {trainer.global_step:,}")
    finished = trainer.train()
    return 0 if finished else 130


if __name__ == "__main__":
    sys.exit(main())
