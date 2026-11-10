"""The simulator must reproduce its reference trajectories bit for bit.

The digests were recorded with the original numpy implementation that produced the E1-E4
results. A change that alters any observation, state, reward, anchor or episode statistic, even
in the last bit, fails here; so does a NumPy or Python upgrade that changes float behaviour.
If a change is meant to alter the simulator, document why and update DIGESTS.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from padp.env.config import EnvConfig
from padp.env.warehouse import WarehouseEnv
from padp.utils.config_io import REPO_ROOT, build_configs, load_experiment

CASES = {
    "e4_coalition": ("e4_coalition.yaml", "mappo", []),
    "e1_k3_long": ("e1_mechanism.yaml", "coupled-k3-long", []),
    "e3_average": ("e3_physics.yaml", "coupled-average", []),
    "e3_leader": ("e3_physics.yaml", "coupled-leader", []),
    "e4_main_racks": ("e4_main.yaml", "coupled", []),
    "simultaneous": ("e4_coalition.yaml", "mappo", ["env.attach_mode=simultaneous"]),
    "sparse": ("e1_sparse.yaml", "coupled-k2", []),
}

DIGESTS = {
    "e4_coalition": "2983ac8e1788e61bd320ff3748f807d8647e65a731fa9b535cccffdf0a0b34df",
    "e1_k3_long": "9f6b930d5ddc442099f908e031a6d4cc589272899619f41bec19bfc2f8586111",
    "e3_average": "7d5b1ec6d608a9366c66717e04a3c03352c2a227b19789daed1b50d1a6be0248",
    "e3_leader": "80b636d1e51d2d3b5748aa64ae261e84eff2b42e2685e5c5a3ebb6bece6fba8b",
    "e4_main_racks": "c6bcb9cdbcff1fcf9a39fe68e62b1410152be519863432f9f74dd28f744e17d4",
    "simultaneous": "7cc07d89655dbc74eb21b5b2b691c4eb814c00c834ba88700a9f165652cfdf59",
    "sparse": "9378d614644cd4ee9751fbca842b8d19d53d6690c759aefe35f6d151e240a72b",
}


def scripted_policy(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reads only the observation, and lifts, carries and delivers often enough to cover every rule."""
    acts = np.zeros((obs.shape[0], 2), dtype=np.int64)
    shared = int(rng.integers(5))
    for i, o in enumerate(obs):
        held, lifted, can_attach, can_drop = o[3] > 0, o[4] > 0, o[17] > 0, o[18] > 0
        r = rng.random()
        if lifted:
            toward = (4 if o[14] > 0 else 3) if abs(o[14]) > abs(o[15]) else (2 if o[15] > 0 else 1)
            move = toward if r < 0.7 else (shared if r < 0.9 else int(rng.integers(5)))
        elif not held:
            toward = (4 if o[12] > 0 else 3) if abs(o[12]) > abs(o[13]) else (2 if o[13] > 0 else 1)
            move = toward if r < 0.6 else int(rng.integers(5))
        else:
            move = int(rng.integers(5))
        q = rng.random()
        if can_drop:
            interaction = 2 if q < 0.85 else 0
        elif can_attach:
            interaction = 1 if q < 0.8 else (2 if q < 0.85 else 0)
        elif held:
            interaction = 2 if q < 0.03 else (1 if q < 0.06 else 0)
        else:
            interaction = 1 if q < 0.05 else (2 if q < 0.08 else 0)
        acts[i] = (move, interaction)
    return acts


def trajectory_digest(env_cfg: EnvConfig, episodes: int = 12, seed: int = 5) -> str:
    env = WarehouseEnv(env_cfg)
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    for ep in range(episodes):
        if env_cfg.layout == "warehouse":
            env.set_n_racks(ep % (env_cfg.n_racks + 1))
        obs, info = env.reset(seed=seed * 1000 + ep)
        done = False
        while True:
            for arr in (obs, info["anchors"], env.state(), *env.agent_positions, *env.object_positions):
                h.update(arr.dtype.str.encode())
                h.update(arr.tobytes())
            if done:
                h.update(repr(sorted((k, float(v)) for k, v in info["episode"].items())).encode())
                break
            obs, reward, terminated, truncated, info = env.step(scripted_policy(obs, rng))
            h.update(repr((float(reward), terminated, truncated, env.last_events)).encode())
            done = terminated or truncated
    return h.hexdigest()


def case_config(name: str) -> EnvConfig:
    file, variant, overrides = CASES[name]
    experiment = load_experiment(REPO_ROOT / "configs" / "experiments" / file)
    env_cfg, _ = build_configs(experiment.resolve(variant, 0, overrides))
    return env_cfg


@pytest.mark.parametrize("name", sorted(CASES))
def test_trajectories_match_reference(name):
    assert trajectory_digest(case_config(name)) == DIGESTS[name], (
        f"simulator trajectories changed for {name}; see this file's docstring")
