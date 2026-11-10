"""Lift, carry and drop rules for k-agent coalitions."""
import numpy as np
import pytest

from padp.env.actions import INTERACT_DROP, INTERACT_NONE, INTERACT_PICK, MOVE_RIGHT, MOVE_STAY, MOVE_UP
from padp.env.warehouse import MOVE_STEP, OBS_DIM
from tests.conftest import act, make_env, place, slot

PICK = (MOVE_STAY, INTERACT_PICK)
DROP = (MOVE_STAY, INTERACT_DROP)
RIGHT = (MOVE_RIGHT, INTERACT_NONE)
UP = (MOVE_UP, INTERACT_NONE)


def test_light_object_is_lifted_by_one_agent():
    env = make_env(weights=[40.0])
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left")})
    env.step(act(env, {0: PICK}))
    assert env.lifted[0] and env.holders[0] == [0]
    env.step(act(env, {0: RIGHT}))
    assert env.object_positions[0][0] == pytest.approx(0.5 + MOVE_STEP)


def test_two_agents_need_opposite_grips():
    env = make_env()
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left"), 1: slot(obj, "top")})
    env.step(act(env, {0: PICK, 1: PICK}))
    assert sorted(env.holders[0]) == [0, 1] and not env.lifted[0]


def test_opposite_pair_lifts(pair_scene):
    env = pair_scene
    env.step(act(env, {0: PICK, 1: PICK}))
    assert env.lifted[0]


def test_three_agents_lift_heavy_object():
    env = make_env(weights=[140.0])
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left"), 1: slot(obj, "right"), 2: slot(obj, "top")})
    env.step(act(env, {0: PICK, 1: PICK}))
    assert not env.lifted[0], "100 kg of capacity must not lift 140 kg"
    env.step(act(env, {2: PICK}))
    assert env.lifted[0] and len(env.holders[0]) == 3


def test_attached_agent_waits_in_place():
    env = make_env()
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left")})
    env.step(act(env, {0: PICK}))
    before = env.agent_positions[0].copy()
    env.step(act(env, {0: RIGHT}))
    np.testing.assert_allclose(env.agent_positions[0], before)
    env.step(act(env, {0: DROP}))
    assert env.agent_holding[0] is None and env.holders[0] == []


def test_strict_rule_needs_identical_moves(pair_scene):
    env = pair_scene
    env.step(act(env, {0: PICK, 1: PICK}))
    start = env.object_positions[0].copy()
    env.step(act(env, {0: RIGHT, 1: UP}))
    np.testing.assert_allclose(env.object_positions[0], start)
    env.step(act(env, {0: RIGHT, 1: RIGHT}))
    np.testing.assert_allclose(env.object_positions[0], start + [MOVE_STEP, 0.0], atol=1e-6)


def test_average_rule_moves_by_mean_delta():
    env = make_env(coupling_rule="average")
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left"), 1: slot(obj, "right")})
    env.step(act(env, {0: PICK, 1: PICK}))
    env.step(act(env, {0: RIGHT, 1: UP}))
    np.testing.assert_allclose(env.object_positions[0], [0.5 + MOVE_STEP / 2, 0.5 - MOVE_STEP / 2], atol=1e-6)


def test_leader_rule_follows_lowest_index():
    env = make_env(coupling_rule="leader")
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left"), 1: slot(obj, "right")})
    env.step(act(env, {0: PICK, 1: PICK}))
    env.step(act(env, {0: RIGHT, 1: UP}))
    np.testing.assert_allclose(env.object_positions[0], [0.5 + MOVE_STEP, 0.5], atol=1e-6)


def test_simultaneous_mode_requires_joint_pick():
    env = make_env(attach_mode="simultaneous")
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left"), 1: slot(obj, "right")})
    env.step(act(env, {0: PICK}))
    assert env.holders[0] == [], "a lone pick must not attach in simultaneous mode"
    env.step(act(env, {0: PICK, 1: PICK}))
    assert env.lifted[0]


def test_joint_drop_in_goal_delivers():
    env = make_env()
    obj = (0.5, 0.5)
    place(env, [obj], [obj], {0: slot(obj, "left"), 1: slot(obj, "right")})
    env.step(act(env, {0: PICK, 1: PICK}))
    _, reward, terminated, _, info = env.step(act(env, {0: DROP, 1: DROP}))
    assert env.delivered[0] and terminated
    assert info["episode"]["success"] == 1.0 and info["episode"]["deliveries_k2"] == 1.0
    assert reward > 0


def test_partial_drop_is_a_disagreement(pair_scene):
    env = pair_scene
    env.step(act(env, {0: PICK, 1: PICK}))
    env.step(act(env, {0: DROP}))
    assert env.lifted[0] and env.last_events.drop_disagreements == 1


def test_anchors_mark_coalitions(pair_scene):
    env = pair_scene
    _, _, _, _, info = env.step(act(env, {0: PICK, 1: PICK}))
    anchors = info["anchors"]
    assert list(anchors[0]) == [0, 0] and list(anchors[1]) == [0, 0]
    assert list(anchors[2]) == [-1, -1]


@pytest.mark.parametrize("layout,n_racks", [("open", 0), ("warehouse", 4)])
def test_observation_and_state_shapes(layout, n_racks):
    env = make_env(layout=layout, n_racks=n_racks, n_agents=4, n_objects=2, spawn="depot")
    obs, _ = env.reset(seed=3)
    assert obs.shape == (4, OBS_DIM)
    assert env.state().shape == (env.state_dim,)
    assert np.all(np.abs(obs) <= 1.0) and np.all(np.abs(env.state()) <= 1.0)
