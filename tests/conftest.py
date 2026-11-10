"""Helpers for building deterministic warehouse scenes."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

from padp.env.actions import INTERACT_NONE, MOVE_STAY
from padp.env.config import EnvConfig
from padp.env.geometry import grip_points
from padp.env.warehouse import AGENT_RADIUS, OBJECT_RADIUS, WarehouseEnv

PARKING = [(0.08, 0.92), (0.16, 0.92), (0.24, 0.92), (0.08, 0.84), (0.16, 0.84), (0.24, 0.84)]


def make_env(**overrides) -> WarehouseEnv:
    params = dict(n_agents=3, n_objects=1, n_racks=0, layout="open", spawn="near_objects",
                  max_steps=200, capacities=[50.0], weights=[90.0])
    params.update(overrides)
    env = WarehouseEnv(EnvConfig(**params))
    env.reset(seed=0)
    return env


def slot(obj: Sequence[float], side: str) -> np.ndarray:
    return grip_points(np.asarray(obj, dtype=np.float32), OBJECT_RADIUS, AGENT_RADIUS)[side].copy()


def place(env: WarehouseEnv, objects: Sequence[Tuple[float, float]], goals: Sequence[Tuple[float, float]],
          agents: Dict[int, np.ndarray]) -> None:
    """Override positions after reset. Agents not listed are parked in the depot corner."""
    env.object_positions = [np.array(p, dtype=np.float32) for p in objects]
    env.goal_positions = [np.array(p, dtype=np.float32) for p in goals]
    for i in range(env.n_agents):
        pos = agents.get(i, PARKING[i % len(PARKING)])
        env.agent_positions[i] = np.array(pos, dtype=np.float32)
    env.refresh_derived_state()


def act(env: WarehouseEnv, per_agent: Optional[Dict[int, Tuple[int, int]]] = None) -> np.ndarray:
    actions = np.tile([MOVE_STAY, INTERACT_NONE], (env.n_agents, 1))
    for i, a in (per_agent or {}).items():
        actions[i] = a
    return actions


@pytest.fixture
def pair_scene():
    """Two agents at the left/right grips of a 90 kg object (capacity 50 each), goal far away."""
    env = make_env()
    obj = (0.5, 0.5)
    place(env, [obj], [(0.85, 0.15)], {0: slot(obj, "left"), 1: slot(obj, "right")})
    return env
