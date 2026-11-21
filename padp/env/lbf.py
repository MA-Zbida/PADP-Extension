"""Level-Based Foraging (Papoudakis et al., 2021) behind the trainer's environment interface.

Uses the official `lbforaging` package (v2, gymnasium). Agents move on a grid and collect food by
choosing LOAD next to it; a food item is collected only if the levels of the agents that LOAD in the
same step sum to at least the food's level. With `force_coop`, every food's level is the sum of the
three lowest agent levels, so with two or three agents all of them must LOAD in the same step: a
one-step agreement-gated transition.

Anchors for coupled sampling come only from each agent's own observation: the grid position of the
food it is adjacent to. This requires full observability (the package's standard tasks, sight =
grid size), where observed coordinates are absolute and therefore identical for all agents; with
partial observability the package reports coordinates relative to each agent, so agents next to the
same food would compute different anchors. Partial observability is therefore rejected.

The trainer's policy has two action heads; LBF has one 6-way action. The second head has a single
action (log-probability 0, entropy 0), so the policy is exactly LBF's 6-way categorical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

LOAD = 5
N_ACTIONS = 6


@dataclass
class LBFConfig:
    players: int = 2
    field_size: int = 8
    max_num_food: int = 2
    min_player_level: int = 1
    max_player_level: int = 2
    min_food_level: int = 1
    max_food_level: Optional[int] = None   # None = the package default (sum of the three lowest agent levels)
    sight: Optional[int] = None            # None = full observability (sight = field_size), as in the standard tasks
    max_episode_steps: int = 50
    force_coop: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sight is not None and self.sight < self.field_size:
            raise ValueError("anchors must come from observations: partial observability gives agent-relative "
                             "coordinates in lbforaging, so use full observability (sight = None)")

    @property
    def n_agents(self) -> int:
        return self.players


def anchor_from_obs(obs: np.ndarray, n_food: int, n_cols: int) -> int:
    """Anchor of one agent, computed from its own raw observation only.

    The observation lists foods as (row, col, level), absent foods as (-1, -1, 0), then agents as
    (row, col, level) with the observing agent first. The anchor is the grid cell of the adjacent food
    with the highest level (ties: smallest row, then column), or -1 if no food is adjacent.
    """
    row, col = obs[3 * n_food], obs[3 * n_food + 1]
    best = None
    for i in range(n_food):
        fr, fc, level = obs[3 * i], obs[3 * i + 1], obs[3 * i + 2]
        if level > 0 and abs(fr - row) + abs(fc - col) == 1:
            key = (-level, fr, fc)
            if best is None or key < best:
                best = key
    return -1 if best is None else int(best[1] * n_cols + best[2])


class LBFEnv:
    """Adapter with the interface the trainer expects (reset, step, state, anchors, episode summary)."""

    n_move_actions = N_ACTIONS
    n_interact_actions = 1
    n_racks = 0

    def __init__(self, cfg: LBFConfig) -> None:
        from lbforaging.foraging import ForagingEnv

        self.cfg = cfg
        self.n_agents = cfg.players
        self.env = ForagingEnv(
            players=cfg.players, min_player_level=cfg.min_player_level, max_player_level=cfg.max_player_level,
            min_food_level=cfg.min_food_level, max_food_level=cfg.max_food_level,
            field_size=(cfg.field_size, cfg.field_size), max_num_food=cfg.max_num_food, sight=cfg.field_size,
            max_episode_steps=cfg.max_episode_steps, force_coop=cfg.force_coop, normalize_reward=True,
            grid_observation=False, observe_agent_levels=True, penalty=0.0,
        )
        self.obs_dim = int(self.env.observation_space[0].shape[0])
        self.state_dim = self.n_agents * self.obs_dim
        # Network inputs are the observations scaled to about [-1, 1]; anchors use the raw values.
        coord = max(1, cfg.field_size - 1)
        scale = []
        for _ in range(cfg.max_num_food):
            scale += [coord, coord, 3 * cfg.max_player_level]
        for _ in range(cfg.players):
            scale += [coord, coord, cfg.max_player_level]
        self._scale = np.asarray(scale, dtype=np.float32)
        self.episode_idx = -1
        self.t = 0

    # ------------------------------------------------------------------ interface

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw, _ = self.env.reset(seed=seed)
        self.episode_idx += 1
        self.t = 0
        self._raw = np.stack(raw).astype(np.float32)
        self._food_total = int((self.env.field > 0).sum())
        self._stats = dict(ret=0.0, events=0, agreed=0, invalid=0, loads=[], collected=0)
        return self._observations(), {"anchors": self._anchors()}

    def step(self, actions: np.ndarray):
        moves = [int(a) for a in np.asarray(actions).reshape(self.n_agents, -1)[:, 0]]
        if any(not 0 <= m < N_ACTIONS for m in moves):
            raise ValueError("actions must be in [0, 6)")
        env, st = self.env, self._stats
        players = env.players
        valid = [m if m in [a.value for a in env._valid_actions[p]] else 0 for p, m in zip(players, moves)]
        st["invalid"] += sum(m != v for m, v in zip(moves, valid))

        # Load agreement: food cells where the adjacent agents' levels suffice and at least one of them loads.
        before = env.field.copy()
        attempts = []
        for r, c in zip(*np.nonzero(before)):
            adjacent = env.adjacent_players(r, c)
            loaders = [p for p in adjacent if valid[players.index(p)] == LOAD]
            if loaders and sum(p.level for p in adjacent) >= before[r, c]:
                attempts.append((r, c, len(loaders)))

        raw, rewards, done, _, _ = env.step(valid)
        self.t += 1
        self._raw = np.stack(raw).astype(np.float32)
        reward = float(sum(rewards))
        st["ret"] += reward
        for r, c, n in attempts:
            st["events"] += 1
            if env.field[r, c] == 0:
                st["agreed"] += 1
        for r, c in zip(*np.nonzero(before)):
            if env.field[r, c] == 0:  # collected this step
                st["collected"] += 1
                st["loads"].append(sum(1 for p in env.adjacent_players(r, c) if valid[players.index(p)] == LOAD))

        terminated = int(env.field.sum()) == 0
        truncated = bool(done) and not terminated
        info: Dict[str, Any] = {"anchors": self._anchors()}
        if terminated or truncated:
            info["episode"] = self._summary(terminated)
        return self._observations(), reward, terminated, truncated, info

    def state(self) -> np.ndarray:
        return self._observations().reshape(-1)

    # ------------------------------------------------------------------ helpers

    def _observations(self) -> np.ndarray:
        return np.clip(self._raw / self._scale, -1.0, 1.0).astype(np.float32)

    def _anchors(self) -> np.ndarray:
        anchors = np.full((self.n_agents, 2), -1, dtype=np.int64)
        for i in range(self.n_agents):
            anchors[i, 0] = anchor_from_obs(self._raw[i], self.cfg.max_num_food, self.cfg.field_size)
        return anchors

    def _summary(self, success: bool) -> Dict[str, float]:
        st, nan = self._stats, math.nan
        loads = st["loads"]
        return {
            "success": float(success),
            "completion": st["collected"] / max(1, self._food_total),
            "steps": float(self.t),
            "return": st["ret"],
            # Gated-event agreement: of the load attempts where enough agents were adjacent, the
            # fraction that collected the food. Reported in the "carry_agreement" column.
            "carry_agreement": st["agreed"] / st["events"] if st["events"] else nan,
            "drop_disagreements": float(st["events"] - st["agreed"]),  # miscoordinated load attempts
            "coalition_size": float(np.mean(loads)) if loads else nan,  # agents loading per collected food
            "deliveries_k1": float(sum(1 for n in loads if n == 1)),
            "deliveries_k2": float(sum(1 for n in loads if n == 2)),
            "deliveries_k3plus": float(sum(1 for n in loads if n >= 3)),
            "invalid_actions": float(st["invalid"]),
            "help_wait": nan, "spare_capacity": nan, "collisions": nan, "releases_outside_goal": nan,
        }
