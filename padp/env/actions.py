"""Action constants and helpers for the continuous warehouse environment."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


MOVE_STAY = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4

INTERACT_NONE = 0
INTERACT_PICK = 1
INTERACT_DROP = 2

MOVE_NAMES = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]
INTERACTION_NAMES = ["NONE", "PICK", "DROP"]

NUM_MOVE_ACTIONS = len(MOVE_NAMES)
NUM_INTERACTION_ACTIONS = len(INTERACTION_NAMES)
NUM_FLAT_ACTIONS = NUM_MOVE_ACTIONS * NUM_INTERACTION_ACTIONS

MOVE_DELTAS = {
    MOVE_STAY: np.array([0.0, 0.0], dtype=np.float32),
    MOVE_UP: np.array([0.0, -1.0], dtype=np.float32),
    MOVE_DOWN: np.array([0.0, 1.0], dtype=np.float32),
    MOVE_LEFT: np.array([-1.0, 0.0], dtype=np.float32),
    MOVE_RIGHT: np.array([1.0, 0.0], dtype=np.float32),
}


def flatten_action(move_action: int, interaction_action: int) -> int:
    return int(move_action) * NUM_INTERACTION_ACTIONS + int(interaction_action)


def unflatten_action(action: int) -> Tuple[int, int]:
    action = int(action)
    return action // NUM_INTERACTION_ACTIONS, action % NUM_INTERACTION_ACTIONS


def parse_factorized_actions(actions: "np.ndarray | Sequence[object]", n_agents: int) -> np.ndarray:
    """Return actions as an ``(n_agents, 2)`` movement/interaction array."""
    return _parse(actions, n_agents)[0]


def parse_action_rows(actions: "np.ndarray | Sequence[object]", n_agents: int) -> List[List[int]]:
    """Same as parse_factorized_actions, as a list of [movement, interaction] rows (fast path)."""
    return _parse(actions, n_agents)[1]


def _parse(actions, n_agents: int) -> Tuple[np.ndarray, List[List[int]]]:
    arr = np.asarray(actions, dtype=np.int32)

    if arr.shape == (n_agents, 2):
        parsed = arr.copy()
    else:
        flat = arr.reshape(-1)
        if flat.size == n_agents * 2:
            parsed = flat.reshape(n_agents, 2).copy()
        elif flat.size == n_agents:
            parsed = np.zeros((n_agents, 2), dtype=np.int32)
            for idx, action in enumerate(flat):
                if 0 <= int(action) < NUM_MOVE_ACTIONS:
                    parsed[idx] = [int(action), INTERACT_NONE]
                elif 0 <= int(action) < NUM_FLAT_ACTIONS:
                    parsed[idx] = unflatten_action(int(action))
                else:
                    raise ValueError("Action entries must be valid movement/interaction actions.")
        else:
            raise ValueError("Action must provide one entry or one pair per agent.")

    rows = parsed.tolist()  # checking 2 x n_agents Python ints is faster than numpy reductions
    if any(not 0 <= move < NUM_MOVE_ACTIONS for move, _ in rows):
        raise ValueError("Movement actions are out of range.")
    if any(not 0 <= interaction < NUM_INTERACTION_ACTIONS for _, interaction in rows):
        raise ValueError("Interaction actions are out of range.")

    return parsed, rows
