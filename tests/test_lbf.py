"""Level-Based Foraging adapter: shared anchors from observations, forced cooperation, training."""
import numpy as np
import pytest

pytest.importorskip("lbforaging")

from padp.env.lbf import LOAD, LBFConfig, LBFEnv, anchor_from_obs  # noqa: E402
from padp.marl.trainer import EPISODE_KEYS  # noqa: E402


def _place(env: LBFEnv, food, level, positions, levels):
    """Put one food item and the agents at given cells, then refresh the package's observations."""
    inner = env.env
    inner.field[:] = 0
    inner.field[food] = level
    inner._food_spawned = level
    for player, pos, lv in zip(inner.players, positions, levels):
        player.position, player.level = pos, lv
    inner._gen_valid_moves()
    env._raw = np.stack(inner._make_gym_obs()).astype(np.float32)
    env._food_total = 1
    return env._raw


def test_partial_observability_is_rejected():
    with pytest.raises(ValueError):
        LBFConfig(field_size=8, sight=2)


def test_full_observability_gives_absolute_coordinates():
    env = LBFEnv(LBFConfig(players=3, field_size=10, max_num_food=3))
    env.reset(seed=1)
    for i, player in enumerate(env.env.players):
        own = env._raw[i, 3 * 3: 3 * 3 + 2]
        assert tuple(int(v) for v in own) == tuple(player.position)


def test_agents_next_to_the_same_food_compute_the_same_anchor_from_their_own_observation():
    env = LBFEnv(LBFConfig(players=3, field_size=8, max_num_food=2))
    env.reset(seed=0)
    raw = _place(env, food=(3, 3), level=4, positions=[(3, 2), (2, 3), (6, 6)], levels=[2, 2, 1])
    anchors = [anchor_from_obs(raw[i], 2, 8) for i in range(3)]
    assert anchors[0] == anchors[1] == 3 * 8 + 3  # two different sides of the same food
    assert anchors[2] == -1                     # not adjacent to any food
    np.testing.assert_array_equal(env._anchors()[:, 0], anchors)


def test_forced_cooperation_needs_every_agent_to_load_in_the_same_step():
    env = LBFEnv(LBFConfig(players=2, field_size=8, max_num_food=2, force_coop=True))
    env.reset(seed=3)
    levels = [p.level for p in env.env.players]
    assert all(level == sum(levels) for level in env.env.field[env.env.field > 0])

    _place(env, food=(3, 3), level=3, positions=[(3, 2), (2, 3)], levels=[1, 2])
    _, reward, terminated, _, _ = env.step(np.array([[LOAD, 0], [0, 0]]))     # one agent loads
    assert env.env.field[3, 3] == 3 and reward == 0.0 and not terminated
    _, reward, terminated, _, info = env.step(np.array([[LOAD, 0], [LOAD, 0]]))  # both load
    assert env.env.field[3, 3] == 0 and terminated
    assert reward == pytest.approx(1.0)  # normalized team reward: all food collected
    ep = info["episode"]
    assert ep["success"] == 1.0 and ep["deliveries_k2"] == 1.0
    assert ep["carry_agreement"] == pytest.approx(0.5) and ep["drop_disagreements"] == 1.0


def test_episode_summary_and_seeding():
    def rollout(seed):
        env = LBFEnv(LBFConfig(players=2, field_size=8, max_num_food=2))
        obs, _ = env.reset(seed=seed)
        rng, total, done = np.random.default_rng(0), [obs.sum()], False
        while not done:
            obs, r, term, trunc, info = env.step(np.stack([rng.integers(0, 6, 2), np.zeros(2, int)], 1))
            total.append(obs.sum() + r)
            done = term or trunc
        return np.array(total), info["episode"]

    a, ep = rollout(5)
    b, _ = rollout(5)
    np.testing.assert_array_equal(a, b)
    assert set(EPISODE_KEYS) <= set(ep)
    assert ep["steps"] <= 50 and 0.0 <= ep["return"] <= 1.0 + 1e-6


@pytest.mark.parametrize("sampler", ["independent", "coupled"])
def test_training_runs_on_lbf(tmp_path, sampler):
    from padp.marl.config import TrainConfig
    from padp.marl.trainer import Trainer

    trainer = Trainer(LBFConfig(players=2, field_size=6, max_num_food=1),
                      TrainConfig(total_steps=4096, n_envs=4, rollout_len=64, log_every=2048, eval_every=4096,
                                  eval_episodes=2, save_every=4096, sampler=sampler),
                      tmp_path, tensorboard=False)
    assert trainer.train()
    assert (tmp_path / "metrics.csv").exists() and (tmp_path / "final.pt").exists()
