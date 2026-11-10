"""Identical seeds give identical episodes."""
import numpy as np

from padp.env.config import EnvConfig
from padp.env.warehouse import WarehouseEnv


def _rollout(seed: int, steps: int = 60):
    env = WarehouseEnv(EnvConfig(n_agents=4, n_objects=2, layout="warehouse", n_racks=4, capacities=[30, 50, 80],
                                 weights=[40, 90, 140]))
    obs, _ = env.reset(seed=seed)
    action_rng = np.random.default_rng(123)
    trace = [obs]
    for _ in range(steps):
        actions = np.stack([action_rng.integers(0, 5, env.n_agents), action_rng.integers(0, 3, env.n_agents)], axis=1)
        obs, reward, terminated, truncated, _ = env.step(actions)
        trace.append(obs)
        trace.append(np.array([reward]))
        if terminated or truncated:
            obs, _ = env.reset()
    return np.concatenate([t.ravel() for t in trace])


def test_same_seed_same_episode():
    np.testing.assert_array_equal(_rollout(7), _rollout(7))


def test_different_seed_different_episode():
    assert not np.array_equal(_rollout(7), _rollout(8))


def test_two_objects_at_carry_distance_always_place():
    # The old placement raised about once per 1,000 resets with two objects; this crashed pilot runs.
    env = WarehouseEnv(EnvConfig(n_agents=4, n_objects=2, layout="open", n_racks=0, spawn="near_objects",
                                 carry_distance=[0.2, 0.3], capacities=[50], weights=[90]))
    for seed in range(3000):
        env.reset(seed=seed)
        assert len(env.object_positions) == 2
