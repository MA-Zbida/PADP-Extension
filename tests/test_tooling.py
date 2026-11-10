"""Experiment configs resolve correctly and the analysis script summarizes runs."""
import csv
from pathlib import Path

import pytest

from padp.utils.config_io import REPO_ROOT, build_configs, load_experiment, parse_sets

EXPERIMENTS = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_every_variant_builds_valid_configs(path):
    experiment = load_experiment(path)
    for variant, _ in experiment.variants():
        env_cfg, train_cfg = build_configs(experiment.resolve(variant, seed=0))
        assert env_cfg.n_agents >= 1 and train_cfg.total_steps > 0


def test_e1_grid_and_resolution():
    experiment = load_experiment(REPO_ROOT / "configs/experiments/e1_mechanism.yaml")
    names = [v for v, _ in experiment.variants()]
    assert len(names) == 12 and "coupled-k2-long" in names
    config = experiment.resolve("coupled-k2-long", seed=3, sets=["train.lr=0.001"])
    assert config["train"]["sampler"] == "coupled" and config["env"]["weights"] == [90]
    assert config["env"]["carry_distance"] == [0.45, 0.55]
    assert config["train"]["seed"] == 3 and config["train"]["lr"] == 0.001
    assert experiment.variant_levels("coupled-k2-long") == {"sampler": "coupled", "k": "k2", "carry": "long"}


def test_sparse_ablation_inherits_main_experiment():
    config = load_experiment(REPO_ROOT / "configs/experiments/e4_sparse.yaml").resolve("coupled", seed=0)
    assert config["env"]["layout"] == "warehouse" and config["env"]["reward_mode"] == "sparse"
    assert config["train"]["rack_curriculum_steps"] == 250000


def test_parse_sets():
    assert parse_sets(["env.weights=[90, 140]", "train.optimistic=true"]) == {
        "env": {"weights": [90, 140]}, "train": {"optimistic": True}}


def _fake_run(run_dir: Path, success: float) -> None:
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "success", "carry_agreement", "q_agree"])
        for step in range(0, 100_001, 20_000):
            w.writerow([step, success * step / 100_000, 0.5, 0.4])
    with (run_dir / "eval.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "stochastic_success", "greedy_success"])
        w.writerow([100_000, success, success])
    (run_dir / "final.pt").write_bytes(b"")


def test_analyze_writes_summary_and_figures(tmp_path):
    import scripts.analyze as analyze

    config = REPO_ROOT / "configs/experiments/e2_entropy.yaml"
    experiment = load_experiment(config)
    for seed in (0, 1, 2):
        _fake_run(experiment.run_dir("indep-ent005", seed, tmp_path / "runs"), 0.2 + 0.05 * seed)
        _fake_run(experiment.run_dir("coupled-ent005", seed, tmp_path / "runs"), 0.7 + 0.05 * seed)
    text = analyze.analyze(config, tmp_path / "runs", tmp_path / "results")
    out = tmp_path / "results" / "e2_entropy"
    assert (out / "summary.md").exists() and (out / "learning_curves.png").exists() and (out / "final_metrics.png").exists()
    assert "| ent005 | 0.25 -> 0.75 | 0.100 |" in text
    assert "| coupled-ent005 | 2 | 0.80 |" in text  # every seed is listed
