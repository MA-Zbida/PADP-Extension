"""The shaped reward cannot be farmed and stays policy-invariant."""
import numpy as np
import pytest

from padp.env.actions import INTERACT_DROP, INTERACT_NONE, INTERACT_PICK, MOVE_RIGHT, MOVE_STAY
from tests.conftest import act, make_env, place, slot

PICK = (MOVE_STAY, INTERACT_PICK)
DROP = (MOVE_STAY, INTERACT_DROP)
RIGHT = (MOVE_RIGHT, INTERACT_NONE)


def _scene(**overrides):
    env = make_env(**overrides)
    obj = (0.5, 0.5)
    place(env, [obj], [(0.70, 0.5)], {0: slot(obj, "left"), 1: slot(obj, "right")})
    return env


def _cycle_rewards(env, steps):
    rewards = []
    for t in range(steps):
        a = {0: PICK, 1: PICK} if t % 2 == 0 else {0: DROP, 1: DROP}
        rewards.append(env.step(act(env, a))[1])
    return rewards


def _delivery_rewards(env):
    plan = [{0: PICK, 1: PICK}] + [{0: RIGHT, 1: RIGHT}] * 8 + [{0: DROP, 1: DROP}]
    rewards = []
    for a in plan:
        _, r, terminated, _, _ = env.step(act(env, a))
        rewards.append(r)
    assert terminated, "the scripted carry should deliver the object"
    return rewards


def test_pick_drop_cycling_does_not_pay():
    env = _scene()
    rewards = _cycle_rewards(env, 40)
    assert env.tracker.totals.releases_outside_goal == 20
    assert sum(rewards) < 0


def test_delivering_beats_cycling():
    deliver = sum(_delivery_rewards(_scene()))
    cycle = sum(_cycle_rewards(_scene(), 10))
    assert deliver > 0 > cycle


def test_shaping_telescopes():
    """Discounted shaping over an episode equals gamma^T * Phi(s_T) - Phi(s_0) (Ng et al., 1999)."""
    env = _scene()
    gamma = env.cfg.reward.gamma
    phi0 = env._phi
    shaped = 0.0
    for t, a in enumerate([{0: PICK, 1: PICK}, {0: RIGHT, 1: RIGHT}, {0: DROP, 1: DROP}, {0: PICK, 1: PICK}]):
        env.step(act(env, a))
        shaped += gamma ** t * env.last_reward_parts["shaping"]
    assert shaped == pytest.approx(gamma ** 4 * env._phi - phi0, abs=1e-9)


def test_sparse_mode_pays_only_step_cost_and_delivery():
    env = _scene(reward_mode="sparse")
    rewards = _cycle_rewards(env, 6)
    assert np.allclose(rewards, env.cfg.reward.step)
    rewards = _delivery_rewards(_scene(reward_mode="sparse"))
    assert rewards[-1] == pytest.approx(env.cfg.reward.delivery + env.cfg.reward.step)
