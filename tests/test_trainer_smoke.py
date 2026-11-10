"""The training loop runs end to end, writes its outputs and resumes (a few seconds, not real training)."""
import csv

import pytest

from padp.env.config import EnvConfig
from padp.marl.config import TrainConfig
from padp.marl.trainer import Trainer

ENV = dict(n_agents=3, n_objects=1, n_racks=0, layout="open", spawn="near_objects", max_steps=20, weights=[90.0])
TRAIN = dict(n_envs=2, rollout_len=16, hidden=32, log_every=32, eval_every=64, save_every=32, eval_episodes=1)


@pytest.mark.parametrize("sampler,critic", [("coupled", "central"), ("independent", "local")])
def test_train_writes_outputs_and_resumes(tmp_path, sampler, critic):
    cfg = TrainConfig(total_steps=64, sampler=sampler, critic=critic, **TRAIN)
    assert Trainer(EnvConfig(**ENV), cfg, tmp_path, tensorboard=False).train()
    for name in ("metrics.csv", "eval.csv", "final.pt", "latest.pt"):
        assert (tmp_path / name).exists(), name
    with (tmp_path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and "q_agree" in rows[0] and "carry_agreement" in rows[0]

    longer = TrainConfig(total_steps=128, sampler=sampler, critic=critic, **TRAIN)
    resumed = Trainer(EnvConfig(**ENV), longer, tmp_path, tensorboard=False)
    resumed.load(tmp_path / "latest.pt")
    assert resumed.global_step == 64
    assert resumed.train() and resumed.global_step == 128


def test_restart_without_checkpoint_drops_partial_logs(tmp_path):
    import scripts.train as train
    from padp.utils.config_io import REPO_ROOT

    config = REPO_ROOT / "configs/experiments/smoke.yaml"
    sets =["train.total_steps=64", "train.n_envs=2", "train.rollout_len=16", "train.hidden=32", "train.log_every=32",
            "train.eval_every=64", "train.save_every=1000", "train.eval_episodes=1", "env.max_steps=20"]
    args = ["--config", str(config), "--seed", "0", "--runs-root", str(tmp_path), "--no-tensorboard", "--set", *sets]
    # A crashed run: logs exist but no checkpoint was saved.
    assert train.main(args) == 0
    run_dir = next(tmp_path.glob("*/*/seed0"))
    for name in ("final.pt", "latest.pt"):
        (run_dir / name).unlink()
    with (run_dir / "metrics.csv").open("a", newline="") as f:
        f.write("999999" + "," * 3 + "\n")
    assert train.main(args) == 0
    with (run_dir / "metrics.csv").open() as f:
        steps = [int(row["step"]) for row in csv.DictReader(f)]
    assert steps == sorted(set(steps)) and 999999 not in steps
