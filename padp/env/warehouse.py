"""Continuous 2D warehouse with weighted objects, agent capacities and k-agent coalitions.

An agent attaches to a free grip slot of an object. The object is lifted once the attached
capacity covers its weight and, for two or more holders, the holders include an opposite pair
of grips. A lifted object moves as a rigid group according to ``coupling_rule`` and is
delivered when every holder drops it inside a free goal.

Positions are float32 arrays (``agent_positions``, ``object_positions``, ``goal_positions``).
The per-step code works on exact Python-float copies of them (``_apos``, ``_opos``, ``_gpos``)
because numpy is slow on 2-vectors. Every float32 operation of the reference implementation is
reproduced exactly: sums and differences are rounded to float32 where the reference computed
them in float32, and comparisons against float32 positions use thresholds rounded to float32
(numpy compares a float32 with a Python float in float32). The trajectories are bit-for-bit
those of the plain numpy version.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from padp.env.actions import (
    INTERACT_DROP,
    INTERACT_PICK,
    MOVE_DELTAS,
    MOVE_STAY,
    NUM_INTERACTION_ACTIONS,
    NUM_MOVE_ACTIONS,
    parse_action_rows,
)
from padp.env.config import EnvConfig
from padp.env.geometry import (
    circle_any_rect_overlap,
    circle_in_bounds,
    circle_overlap,
    distance,
    grip_points,
    has_opposite_pair,
    nearest_rect_distance,
    normalize_pos,
    segment_distance,
)
from padp.env.layout import make_warehouse_layout
from padp.env.rewards import EpisodeTracker, StepEvents, compute_reward

Vec = np.ndarray

AGENT_RADIUS = 0.025
OBJECT_RADIUS = 0.035
GOAL_RADIUS = 0.065
MOVE_STEP = 0.025
PICKUP_RADIUS = 0.04
NUM_SLOTS = 4

OBS_DIM = 27
OBJECT_FEATURES = 8
GOAL_FEATURES = 3
RACK_FEATURES = 4

_hypot = math.hypot


def _f32(x: float) -> float:
    return float(np.float32(x))


# circle_in_bounds(pos, r) on a float32 position compares in float32: these are its thresholds.
_AGENT_LO, _AGENT_HI = _f32(AGENT_RADIUS), _f32(1.0 - AGENT_RADIUS)
_OBJECT_LO, _OBJECT_HI = _f32(OBJECT_RADIUS), _f32(1.0 - OBJECT_RADIUS)
# circle_overlap(a, ra, b, rb) is distance(a, b) < ra + rb - 1e-6.
_AA = AGENT_RADIUS + AGENT_RADIUS - 1e-6
_AO = AGENT_RADIUS + OBJECT_RADIUS - 1e-6
_OO = OBJECT_RADIUS + OBJECT_RADIUS - 1e-6
_REACH = 2 * AGENT_RADIUS + 2 * MOVE_STEP + 1e-6  # agents farther apart cannot touch after one step
_CROSS = 2 * AGENT_RADIUS
_STEP_DELTAS = {m: d * MOVE_STEP for m, d in MOVE_DELTAS.items()}  # float32, as MOVE_DELTAS[m] * MOVE_STEP


def _rect_hit(x: float, y: float, radius: float, racks) -> bool:
    """circle_any_rect_overlap for a position given as floats."""
    for rect in racks:
        cx = min(max(x, rect.x1), rect.x2)
        cy = min(max(y, rect.y1), rect.y2)
        if _hypot(x - cx, y - cy) < radius:
            return True
    return False


class WarehouseEnv:
    """Cooperative transport task. Actions are ``(n_agents, 2)``: movement and interaction."""

    n_move_actions = NUM_MOVE_ACTIONS
    n_interact_actions = NUM_INTERACTION_ACTIONS
    obs_dim = OBS_DIM

    agent_radius = AGENT_RADIUS
    object_radius = OBJECT_RADIUS
    goal_radius = GOAL_RADIUS
    move_step = MOVE_STEP
    pickup_radius = PICKUP_RADIUS

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.cfg = config or EnvConfig()
        self.n_agents = self.cfg.n_agents
        self.n_objects = self.cfg.n_objects
        self.n_goals = self.cfg.n_objects
        self.n_racks = self.cfg.n_racks
        self.state_dim = (
            self.cfg.max_agents * OBS_DIM
            + 1
            + self.cfg.max_objects * (OBJECT_FEATURES + GOAL_FEATURES)
            + self.cfg.max_racks * RACK_FEATURES
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.episode_idx = -1
        self.t = 0
        self.renderer = None
        self.last_events = StepEvents()
        # Free grip slots and slot-reach checks are cached once movement is resolved (hot path);
        # None while a step is being processed, so they are recomputed from the live state.
        self._free_cache: Optional[List[List[Tuple[str, float, float]]]] = None
        self._attempt_cache: Optional[List[Optional[Tuple[int, str, float]]]] = None
        self._grip_cache: Dict[int, Tuple[Tuple[float, float], list]] = {}

    # ------------------------------------------------------------------ public API

    def set_n_racks(self, n_racks: int) -> None:
        if not 0 <= n_racks <= self.cfg.max_racks:
            raise ValueError(f"n_racks must be in [0, {self.cfg.max_racks}]")
        self.n_racks = n_racks

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_idx += 1
        self.t = 0

        n_racks = 0 if self.cfg.layout == "open" else self.n_racks
        self.layout = make_warehouse_layout(n_racks, self.cfg.max_racks)
        self.racks = list(self.layout.racks)

        # Resample until every object can be lifted by at most four of the sampled agents.
        for _ in range(1000):
            self.capacities = self.rng.choice(np.asarray(self.cfg.capacities, dtype=float), size=self.n_agents)
            self.weights = self.rng.choice(np.asarray(self.cfg.weights, dtype=float), size=self.n_objects)
            strongest = np.sort(self.capacities)[::-1][:4].sum()
            if self.weights.max() <= strongest:
                break

        # A carry-distance placement can rarely leave no room for a later object (about 1 reset in
        # 1,000 with two objects); draw new goals and objects instead of failing.
        for _ in range(100):
            self.goal_positions = self._sample_goals()
            objects = self._sample_objects()
            if objects is not None:
                break
        else:
            raise RuntimeError("could not place the objects at the requested carry distance")
        self.object_positions = objects
        self.agent_positions = self._sample_agents()

        self.goal_used = [False] * self.n_goals
        self.delivered = [False] * self.n_objects
        self.holders: List[List[int]] = [[] for _ in range(self.n_objects)]
        self.lifted = [False] * self.n_objects
        self.first_attach_step: List[Optional[int]] = [None] * self.n_objects
        self.agent_holding: List[Optional[int]] = [None] * self.n_agents
        self.agent_side: List[Optional[str]] = [None] * self.n_agents

        self.tracker = EpisodeTracker()
        self.last_events = StepEvents()
        self.last_reward_parts: Dict[str, float] = {}
        self.refresh_derived_state()
        return self._obs, {"anchors": self._anchors()}

    def step(self, actions: Union[np.ndarray, Sequence[Sequence[int]]]):
        acts = parse_action_rows(actions, self.n_agents)
        self.t += 1
        events = StepEvents()
        self._free_cache = None  # attaches/drops/movement change the slots
        self._attempt_cache = None

        self._process_drops(acts, events)
        self._process_attaches(acts, events)
        lifted_now = self._process_lifts(events)
        self._resolve_movement(acts, events, lifted_now)
        self._settle()

        terminated = all(self.delivered)
        truncated = (not terminated) and self.t >= self.cfg.max_steps
        phi_next = 0.0 if terminated else self._potential()
        shaping = self.cfg.reward.gamma * phi_next - self._phi
        self._phi = phi_next

        reward, parts = compute_reward(events, shaping, self.cfg.reward, self.cfg.reward_mode)
        self.tracker.add(events, reward)
        self.last_events = events
        self.last_reward_parts = parts

        info: Dict[str, Any] = {"anchors": self._anchors()}
        if terminated or truncated:
            info["episode"] = self.tracker.summary(self.n_objects, self.t)
        self._obs = self._observations()
        return self._obs, reward, terminated, truncated, info

    def state(self) -> np.ndarray:
        """Global state for a centralized critic (fixed size)."""
        cfg = self.cfg
        state = np.full(self.state_dim, -1.0, dtype=np.float32)
        n_obs = self.n_agents * OBS_DIM
        state[:n_obs] = self._obs.ravel()  # cached by reset()/step()
        k = cfg.max_agents * OBS_DIM
        state[k] = self.t / cfg.max_steps
        k += 1

        rows: List[float] = []
        for o in range(self.n_objects):
            x, y = self._opos[o]
            goal_vec, _ = self._nearest_free_goal(self._opos[o])
            rows += [x * 2.0 - 1.0, y * 2.0 - 1.0, goal_vec[0], goal_vec[1], self._w[o] / cfg.max_weight,
                     self._attached_capacity(o) / cfg.max_weight, float(self.lifted[o]), float(self.delivered[o])]
        state[k:k + len(rows)] = rows
        k += cfg.max_objects * OBJECT_FEATURES

        rows = []
        for g in range(self.n_goals):
            x, y = self._gpos[g]
            rows += [x * 2.0 - 1.0, y * 2.0 - 1.0, float(self.goal_used[g])]
        state[k:k + len(rows)] = rows
        k += cfg.max_objects * GOAL_FEATURES

        state[k:k + self._rack_state.size] = self._rack_state
        return np.clip(state, -1.0, 1.0)

    def render(self, debug: bool = False) -> None:
        if self.renderer is None:
            from padp.env.renderer import WarehouseRenderer  # pygame is only needed for rendering

            self.renderer = WarehouseRenderer()
        self.renderer.render(self, debug=debug)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def refresh_derived_state(self) -> None:
        """Recompute cached grip slots, the shaping potential and observations.

        Called by reset()/step(); call it yourself after editing positions by hand (tests, demos).
        Positions must be float32 arrays.
        """
        for p in (*self.agent_positions, *self.object_positions, *self.goal_positions):
            if np.asarray(p).dtype != np.float32:
                raise TypeError("positions must be float32 arrays")
        self._rack_state = np.array(
            [[*normalize_pos(rack.center), rack.width, rack.height] for rack in self.racks], dtype=np.float32
        ).reshape(-1)
        self._sync()
        self._settle()
        self._phi = self._potential()
        self._obs = self._observations()

    def required_coalition_size(self, obj: int) -> int:
        """Smallest number of agents (by capacity, largest first) able to lift ``obj``."""
        total = 0.0
        for size, cap in enumerate(sorted(self.capacities, reverse=True), start=1):
            total += cap
            if total >= self.weights[obj]:
                return size
        return self.n_agents + 1

    def _settle(self) -> None:
        """Cache what the rest of a step reads once movement is resolved: free grip slots, the
        objects that still need help, each agent's distance to their free slots, and slot reach."""
        self._free_cache = None
        free = self._free_cache = [self._free_points(o) for o in range(self.n_objects)]
        needy = self._needy = [o for o in range(self.n_objects) if self._needs_help(o)]
        self._needy_index = {o: k for k, o in enumerate(needy)}
        slot_d, attempts = [], []
        for px, py in self._apos:
            per_object, best = [], None
            for o in needy:
                ds = [_hypot(px - sx, py - sy) for _, sx, sy in free[o]]
                per_object.append(ds)
                for (side, _, _), d in zip(free[o], ds):
                    if d <= PICKUP_RADIUS and (best is None or d < best[2]):
                        best = (o, side, d)
            slot_d.append(per_object)
            attempts.append(best)  # what _slot_attempt computes
        self._slot_d = slot_d  # [agent][k]: distances to the free slots of needy[k], in slot order
        self._attempt_cache = attempts

    def _sync(self) -> None:
        """Exact float copies of positions, capacities and weights for the per-step code.

        Called by refresh_derived_state(); step() keeps the copies up to date itself.
        """
        self._apos = [p.tolist() for p in self.agent_positions]
        self._opos = [p.tolist() for p in self.object_positions]
        self._gpos = [p.tolist() for p in self.goal_positions]
        self._caps = np.asarray(self.capacities, dtype=float).tolist()
        self._w = np.asarray(self.weights, dtype=float).tolist()

    # ------------------------------------------------------------------ spawning

    def _sample_goals(self) -> List[Vec]:
        points = [p.copy() for p in self.layout.goal_points]
        self.rng.shuffle(points)
        goals = points[: self.n_goals]
        shipping = self.layout.shipping
        while len(goals) < self.n_goals:
            goals.append(self._uniform_in(shipping.x1, shipping.y1, shipping.x2, shipping.y2, GOAL_RADIUS * 0.5))
        return goals

    def _sample_objects(self) -> Optional[List[Vec]]:
        """Object positions, or None when the carry-distance placement found no room."""
        objects: List[Vec] = []
        if self.cfg.carry_distance is not None:
            lo, hi = self.cfg.carry_distance
            for o in range(self.n_objects):
                goal = self.goal_positions[o % self.n_goals]
                for _ in range(2000):
                    d = self.rng.uniform(lo, hi)
                    angle = self.rng.uniform(0.0, 2.0 * math.pi)
                    point = (goal + d * np.array([math.cos(angle), math.sin(angle)])).astype(np.float32)
                    if self._valid_object_spawn(point, objects):
                        objects.append(point)
                        break
                else:
                    return None
            return objects

        candidates = [p.copy() for p in self.layout.pick_faces]
        self.rng.shuffle(candidates)
        for point in candidates:
            if len(objects) >= self.n_objects:
                break
            if self._valid_object_spawn(point, objects):
                objects.append(point)
        while len(objects) < self.n_objects:
            point = self._uniform_in(0.15, 0.15, 0.85, 0.85, 0.0)
            if self._valid_object_spawn(point, objects):
                objects.append(point)
        return objects

    def _sample_agents(self) -> List[Vec]:
        agents: List[Vec] = []
        if self.cfg.spawn == "near_objects":
            for i in range(self.n_agents):
                center = self.object_positions[i % self.n_objects]
                for _ in range(2000):
                    r = self.rng.uniform(0.10, 0.18)
                    angle = self.rng.uniform(0.0, 2.0 * math.pi)
                    point = (center + r * np.array([math.cos(angle), math.sin(angle)])).astype(np.float32)
                    if self._valid_agent_spawn(point, agents):
                        agents.append(point)
                        break
            if len(agents) == self.n_agents:
                return agents
            agents = []  # fall back to the depot if the area around objects is too crowded

        points = [p.copy() for p in self.layout.spawn_points]
        self.rng.shuffle(points)
        for point in points:
            if len(agents) >= self.n_agents:
                break
            if self._valid_agent_spawn(point, agents):
                agents.append(point)
        depot = self.layout.depot
        while len(agents) < self.n_agents:
            point = self._uniform_in(depot.x1, depot.y1, depot.x2, depot.y2, AGENT_RADIUS)
            if self._valid_agent_spawn(point, agents):
                agents.append(point)
        return agents

    def _uniform_in(self, x1: float, y1: float, x2: float, y2: float, margin: float) -> Vec:
        return np.array(
            [self.rng.uniform(x1 + margin, x2 - margin), self.rng.uniform(y1 + margin, y2 - margin)],
            dtype=np.float32,
        )

    def _valid_object_spawn(self, point: Vec, existing: List[Vec]) -> bool:
        # Keep all four grip slots reachable: the grip ring must fit inside the bounds and clear of racks.
        ring = OBJECT_RADIUS + 2.0 * AGENT_RADIUS + 0.02
        if not (ring <= point[0] <= 1.0 - ring and ring <= point[1] <= 1.0 - ring):
            return False
        if circle_any_rect_overlap(point, ring, self.racks):
            return False
        if any(distance(point, other) < 2.0 * ring for other in existing):
            return False
        return not any(distance(point, goal) < GOAL_RADIUS + ring for goal in self.goal_positions)

    def _valid_agent_spawn(self, point: Vec, existing: List[Vec]) -> bool:
        if not circle_in_bounds(point, AGENT_RADIUS):
            return False
        if circle_any_rect_overlap(point, AGENT_RADIUS, self.racks):
            return False
        if any(circle_overlap(point, AGENT_RADIUS, obj, OBJECT_RADIUS) for obj in self.object_positions):
            return False
        return not any(circle_overlap(point, AGENT_RADIUS, other, AGENT_RADIUS) for other in existing)

    # ------------------------------------------------------------------ interaction

    def _process_drops(self, acts: List[List[int]], events: StepEvents) -> None:
        for i in range(self.n_agents):
            if acts[i][1] == INTERACT_DROP and self.agent_holding[i] is None:
                events.invalid_actions += 1

        for o in range(self.n_objects):
            if self.delivered[o] or not self.holders[o]:
                continue
            dropping = [h for h in self.holders[o] if acts[h][1] == INTERACT_DROP]
            if not dropping:
                continue
            if self.lifted[o]:
                if len(dropping) < len(self.holders[o]):
                    events.drop_disagreements += 1
                    continue
                goal = self._goal_containing(self._opos[o])
                if goal is None:
                    events.releases_outside_goal += 1
                else:
                    self.delivered[o] = True
                    self.goal_used[goal] = True
                    self.object_positions[o] = self.goal_positions[goal].copy()
                    self._opos[o] = list(self._gpos[goal])
                    self.tracker.record_delivery(len(self.holders[o]))
                    events.deliveries += 1
                self._release(o)
            else:
                for h in dropping:
                    self._detach(h)
                events.detaches += len(dropping)

    def _process_attaches(self, acts: List[List[int]], events: StepEvents) -> None:
        candidates: List[Tuple[float, int, int, str]] = []
        for i in range(self.n_agents):
            if acts[i][1] != INTERACT_PICK:
                continue
            attempt = self._slot_attempt(i) if self.agent_holding[i] is None else None
            if attempt is None:
                events.invalid_actions += 1
                continue
            obj, side, dist = attempt
            candidates.append((dist, i, obj, side))
        candidates.sort()

        if self.cfg.attach_mode == "incremental":
            for _, i, obj, side in candidates:
                if side in self._occupied_sides(obj):
                    events.invalid_actions += 1
                    continue
                self._attach(i, obj, side)
                events.attaches += 1
            return

        # Simultaneous: a group attaches only if it can lift the object right now.
        by_object: Dict[int, Dict[str, int]] = {}
        for _, i, obj, side in candidates:
            slots = by_object.setdefault(obj, {})
            if side in slots:
                events.invalid_actions += 1
            else:
                slots[side] = i
        for obj, slots in by_object.items():
            capacity = sum(self.capacities[i] for i in slots.values())
            if self._can_lift(capacity, self.weights[obj], slots.keys()):
                for side, i in slots.items():
                    self._attach(i, obj, side)
                    events.attaches += 1
            else:
                events.invalid_actions += len(slots)

    def _process_lifts(self, events: StepEvents) -> set:
        """Lift every object whose holders can carry it. Returns the objects lifted this step."""
        lifted_now = set()
        for o in range(self.n_objects):
            if self.delivered[o] or self.lifted[o] or not self.holders[o]:
                continue
            capacity = self._attached_capacity(o)
            if self._can_lift(capacity, self.weights[o], self._occupied_sides(o)):
                self.lifted[o] = True
                lifted_now.add(o)
                events.lifts += 1
                wait = self.t - (self.first_attach_step[o] if self.first_attach_step[o] is not None else self.t)
                self.tracker.record_lift(wait, capacity - self.weights[o], len(self.holders[o]))
        return lifted_now

    @staticmethod
    def _can_lift(capacity: float, weight: float, sides) -> bool:
        sides = list(sides)
        return capacity >= weight and (len(sides) == 1 or has_opposite_pair(sides))

    def _attach(self, agent: int, obj: int, side: str) -> None:
        self.holders[obj].append(agent)
        self.agent_holding[agent] = obj
        self.agent_side[agent] = side
        if self.first_attach_step[obj] is None:
            self.first_attach_step[obj] = self.t

    def _detach(self, agent: int) -> None:
        obj = self.agent_holding[agent]
        if obj is None:
            return
        self.holders[obj].remove(agent)
        self.agent_holding[agent] = None
        self.agent_side[agent] = None
        if not self.holders[obj]:
            self.first_attach_step[obj] = None

    def _release(self, obj: int) -> None:
        for agent in list(self.holders[obj]):
            self.agent_holding[agent] = None
            self.agent_side[agent] = None
        self.holders[obj] = []
        self.lifted[obj] = False
        self.first_attach_step[obj] = None

    def _occupied_sides(self, obj: int) -> List[str]:
        return [self.agent_side[h] for h in self.holders[obj]]

    def _attached_capacity(self, obj: int) -> float:
        # Plain left-to-right addition, as sum() does over np.float64 items (sum() over Python
        # floats would use compensated summation and could round differently).
        caps = self._caps
        total = 0.0
        for h in self.holders[obj]:
            total += caps[h]
        return total

    def _grip(self, obj: int) -> List[Tuple[str, Vec, List[float]]]:
        """Grip slots of ``obj`` in the order left, right, top, bottom (cached per position)."""
        key = tuple(self._opos[obj])
        cached = self._grip_cache.get(obj)
        if cached is None or cached[0] != key:
            slots = grip_points(self.object_positions[obj], OBJECT_RADIUS, AGENT_RADIUS)
            cached = (key, [(side, p, p.tolist()) for side, p in slots.items()])
            self._grip_cache[obj] = cached
        return cached[1]

    def _free_points(self, obj: int) -> List[Tuple[str, float, float]]:
        """Free grip slots as (side, x, y)."""
        if self._free_cache is not None:
            return self._free_cache[obj]
        occupied = {self.agent_side[h] for h in self.holders[obj]}
        return [(side, xy[0], xy[1]) for side, _, xy in self._grip(obj) if side not in occupied]

    def _free_slots(self, obj: int) -> Dict[str, Vec]:
        """Free grip slots as float32 positions (for the renderer)."""
        occupied = {self.agent_side[h] for h in self.holders[obj]}
        return {side: p for side, p, _ in self._grip(obj) if side not in occupied}

    def _needs_help(self, obj: int) -> bool:
        return not self.delivered[obj] and not self.lifted[obj] and len(self.holders[obj]) < NUM_SLOTS

    def _slot_attempt(self, agent: int) -> Optional[Tuple[int, str, float]]:
        """Nearest free grip slot within reach of ``agent``: (object, side, distance)."""
        if self._attempt_cache is not None:
            return self._attempt_cache[agent]
        px, py = self._apos[agent]
        best: Optional[Tuple[int, str, float]] = None
        for o in range(self.n_objects):
            if not self._needs_help(o):
                continue
            for side, sx, sy in self._free_points(o):
                d = _hypot(px - sx, py - sy)
                if d <= PICKUP_RADIUS and (best is None or d < best[2]):
                    best = (o, side, d)
        return best

    def _goal_containing(self, pos: Sequence[float]) -> Optional[int]:
        px, py = pos[0], pos[1]
        for g, (gx, gy) in enumerate(self._gpos):
            if not self.goal_used[g] and _hypot(px - gx, py - gy) <= GOAL_RADIUS:
                return g
        return None

    # ------------------------------------------------------------------ movement

    def _resolve_movement(self, acts: List[List[int]], events: StepEvents, lifted_now: set = frozenset()) -> None:
        """Propose every move, reject the groups that collide, apply the rest.

        Group ids: agent ``i`` alone is ``i``; the coalition carrying object ``o`` is ``n_agents + o``.
        """
        n_a = self.n_agents
        old_arr_a, old_arr_o = self.agent_positions, self.object_positions
        old_a, old_o = self._apos, self._opos
        new_arr_a, new_arr_o = list(old_arr_a), list(old_arr_o)
        new_a, new_o = list(old_a), list(old_o)

        group_of = list(range(n_a))
        rejected: set = set()

        for o in range(self.n_objects):
            if self.delivered[o] or not self.lifted[o]:
                continue
            holders = self.holders[o]
            gid = n_a + o
            for h in holders:
                group_of[h] = gid
            if o in lifted_now:
                continue  # lifting takes the step; the coalition starts moving next step

            moves = [acts[h][0] for h in holders]
            events.carry_steps += 1
            if all(m == moves[0] for m in moves):
                events.carry_agreements += 1
            delta = self._group_delta(moves, holders)
            if delta is None:
                continue
            for h in holders:
                p = old_arr_a[h] + delta
                new_arr_a[h], new_a[h] = p, p.tolist()
            p = old_arr_o[o] + delta
            new_arr_o[o], new_o[o] = p, p.tolist()
            # Objects later in this loop have not moved yet, exactly as in a sequential update.
            reason = self._validate_group(o, new_a, new_o)
            if reason is not None:
                self._count_collision(reason, events)
                rejected.add(gid)

        for i in range(n_a):
            if self.agent_holding[i] is not None:
                continue  # carrying (handled above) or attached and waiting (anchored in place)
            move = acts[i][0]
            if move == MOVE_STAY:
                continue
            p = old_arr_a[i] + _STEP_DELTAS[move]
            new_arr_a[i], new_a[i] = p, p.tolist()
            reason = self._validate_free_agent(new_a[i], new_o)
            if reason is not None:
                self._count_collision(reason, events)
                rejected.add(group_of[i])

        self._resolve_agent_collisions(old_a, new_a, group_of, rejected, events)
        self._resolve_agent_object_collisions(new_a, new_o, group_of, rejected, events)

        for gid in rejected:
            if gid < n_a:
                new_arr_a[gid], new_a[gid] = old_arr_a[gid], old_a[gid]
            else:
                o = gid - n_a
                for h in self.holders[o]:
                    new_arr_a[h], new_a[h] = old_arr_a[h], old_a[h]
                new_arr_o[o], new_o[o] = old_arr_o[o], old_o[o]

        self.agent_positions, self.object_positions = new_arr_a, new_arr_o
        self._apos, self._opos = new_a, new_o

    def _group_delta(self, moves: List[int], holders: List[int]) -> Optional[Vec]:
        rule = self.cfg.coupling_rule
        if rule == "strict":
            if any(m != moves[0] for m in moves) or moves[0] == MOVE_STAY:
                return None
            return _STEP_DELTAS[moves[0]]
        if rule == "average":
            delta = np.mean([MOVE_DELTAS[m] for m in moves], axis=0) * MOVE_STEP
            return None if float(np.linalg.norm(delta)) < 1e-9 else delta.astype(np.float32)
        leader_move = moves[holders.index(min(holders))]
        return None if leader_move == MOVE_STAY else _STEP_DELTAS[leader_move]

    def _validate_group(self, obj: int, new_a: List[List[float]], new_o: List[List[float]]) -> Optional[str]:
        x, y = new_o[obj]
        if not (_OBJECT_LO <= x <= _OBJECT_HI and _OBJECT_LO <= y <= _OBJECT_HI):
            return "bound"
        if self.racks and _rect_hit(x, y, OBJECT_RADIUS, self.racks):
            return "rack"
        for other, (ox, oy) in enumerate(new_o):
            if other != obj and not self.delivered[other] and _hypot(x - ox, y - oy) < _OO:
                return "object"
        for h in self.holders[obj]:
            reason = self._validate_static(new_a[h])
            if reason is not None:
                return reason
        return None

    def _validate_free_agent(self, pos: List[float], new_o: List[List[float]]) -> Optional[str]:
        reason = self._validate_static(pos)
        if reason is not None:
            return reason
        x, y = pos
        for o, (ox, oy) in enumerate(new_o):
            if not self.delivered[o] and _hypot(x - ox, y - oy) < _AO:
                return "object"
        return None

    def _validate_static(self, pos: List[float]) -> Optional[str]:
        x, y = pos
        if not (_AGENT_LO <= x <= _AGENT_HI and _AGENT_LO <= y <= _AGENT_HI):
            return "bound"
        if self.racks and _rect_hit(x, y, AGENT_RADIUS, self.racks):
            return "rack"
        return None

    def _resolve_agent_collisions(self, old_a, new_a, group_of, rejected, events) -> None:
        n = self.n_agents
        moved = [_hypot(old_a[i][0] - new_a[i][0], old_a[i][1] - new_a[i][1]) > 1e-6 for i in range(n)]
        for i in range(n):
            oix, oiy = old_a[i]
            for j in range(i + 1, n):
                if group_of[i] == group_of[j]:
                    continue
                ojx, ojy = old_a[j]
                if _hypot(oix - ojx, oiy - ojy) > _REACH:
                    continue
                overlap = _hypot(new_a[i][0] - new_a[j][0], new_a[i][1] - new_a[j][1]) < _AA
                if overlap or ((moved[i] or moved[j])
                               and segment_distance(old_a[i], new_a[i], old_a[j], new_a[j]) < _CROSS):
                    rejected.add(group_of[i])
                    rejected.add(group_of[j])
                    events.agent_collisions += 1

    def _resolve_agent_object_collisions(self, new_a, new_o, group_of, rejected, events) -> None:
        n_a = self.n_agents
        for i, (x, y) in enumerate(new_a):
            for o, (ox, oy) in enumerate(new_o):
                if self.delivered[o] or self.agent_holding[i] == o:
                    continue
                if not _hypot(x - ox, y - oy) < _AO:
                    continue
                rejected.add(group_of[i])
                if self.lifted[o]:
                    rejected.add(n_a + o)
                events.object_collisions += 1

    @staticmethod
    def _count_collision(reason: str, events: StepEvents) -> None:
        if reason == "rack":
            events.rack_collisions += 1
        elif reason == "bound":
            events.bound_collisions += 1
        else:
            events.object_collisions += 1

    # ------------------------------------------------------------------ shaping potential

    def _potential(self) -> float:
        if self.cfg.reward_mode != "shaped":
            return 0.0
        rc = self.cfg.reward
        phi = 0.0
        for o in range(self.n_objects):
            if self.delivered[o]:
                continue  # the delivery reward replaces the object's potential
            if self.lifted[o]:
                _, goal_dist = self._nearest_free_goal(self._opos[o])
                phi += rc.lift_potential + rc.carry_potential * (1.0 - goal_dist)
            else:
                # self.weights holds np.float64: keep it, so phi has the reference's type. The type
                # matters: sum() in compute_reward rounds differently for float and np.float64 items.
                covered = min(1.0, self._attached_capacity(o) / self.weights[o])
                phi += rc.attach_potential * covered

        if any(self._free_cache[o] for o in self._needy):
            for i in range(self.n_agents):
                if self.agent_holding[i] is None:
                    # Nearest free slot of any object that needs help.
                    phi -= rc.approach_potential * min(min(ds) for ds in self._slot_d[i] if ds)
        return phi

    # ------------------------------------------------------------------ observations

    def _anchors(self) -> np.ndarray:
        """Per agent: [move anchor, interaction anchor] object ids (-1 = none).

        Both are locally observable: an agent knows which object it holds and whether it is
        standing at an object's grip slot.
        """
        anchors = np.full((self.n_agents, 2), -1, dtype=np.int64)
        for i in range(self.n_agents):
            held = self.agent_holding[i]
            if held is not None:
                anchors[i, 1] = held
                if self.lifted[held]:
                    anchors[i, 0] = held
            else:
                attempt = self._slot_attempt(i)
                if attempt is not None:
                    anchors[i, 1] = attempt[0]
        return anchors

    def _observations(self) -> np.ndarray:
        cfg = self.cfg
        max_w, max_c = cfg.max_weight, cfg.max_capacity
        apos, opos = self._apos, self._opos
        n = self.n_agents
        needy = self._needy
        time_frac = self.t / cfg.max_steps
        rows = []
        with_target = []  # agents whose target_dist is filled in below
        for i in range(n):
            px, py = apos[i]
            held = self.agent_holding[i]
            lifted = held is not None and self.lifted[held]

            target = held
            if target is None and needy:
                best_d = None
                for o in needy:  # first nearest, as min(..., key=distance)
                    d = _hypot(px - opos[o][0], py - opos[o][1])
                    if best_d is None or d < best_d:
                        target, best_d = o, d

            tvx = tvy = svx = svy = 0.0
            target_dist = 1.0
            target_weight = remaining = free_frac = 0.0
            if target is not None:
                tvx, tvy = opos[target][0] - px, opos[target][1] - py
                with_target.append(i)
                target_weight = self._w[target] / max_w
                remaining = max(0.0, self._w[target] - self._attached_capacity(target)) / max_w
                free = self._free_points(target)
                free_frac = len(free) / NUM_SLOTS
                if held is None and free:  # target is then one of the needy objects
                    best_d = None
                    for (_, sx, sy), d in zip(free, self._slot_d[i][self._needy_index[target]]):
                        if best_d is None or d < best_d:
                            best_d, svx, svy = d, sx - px, sy - py

            goal_vec, goal_dist = self._nearest_free_goal(opos[held] if lifted else apos[i])
            can_attach = float(held is None and self._slot_attempt(i) is not None)
            can_drop = float(lifted and self._goal_containing(opos[held]) is not None)

            if n > 1:
                best_d, near = None, None
                for j in range(n):
                    if j != i:
                        d = _hypot(px - apos[j][0], py - apos[j][1])
                        if best_d is None or d < best_d:
                            best_d, near = d, j
                ovx, ovy = apos[near][0] - px, apos[near][1] - py
                other_dist, other_cap = min(1.0, best_d), self._caps[near] / max_c
            else:
                ovx = ovy = 0.0
                other_dist, other_cap = 1.0, 0.0
            if held is not None:
                hvx, hvy = opos[held][0] - px, opos[held][1] - py
            else:
                hvx = hvy = 0.0

            rows.append([
                px * 2.0 - 1.0, py * 2.0 - 1.0,
                self._caps[i] / max_c,
                float(held is not None),
                float(lifted),
                (len(self.holders[held]) if held is not None else 0) / NUM_SLOTS,
                tvx, tvy,
                target_dist,
                target_weight,
                remaining,
                free_frac,
                svx, svy,
                goal_vec[0], goal_vec[1],
                goal_dist,
                can_attach,
                can_drop,
                min(1.0, nearest_rect_distance(apos[i], self.racks)),
                ovx, ovy,
                other_dist,
                other_cap,
                hvx, hvy,
                time_frac,
            ])
        if with_target:
            # The reference used min(1, np.linalg.norm(v)) on the float32 vector v. For a 2-vector
            # that is float32 x*x + y*y (OpenBLAS sdot, no fused multiply-add) and a float32 sqrt,
            # which this computes for all agents at once.
            v = np.array([rows[i][6:8] for i in with_target], dtype=np.float32)
            for i, d in zip(with_target, np.sqrt((v * v).sum(axis=1)).tolist()):
                rows[i][8] = min(1.0, d)
        # Rounding each exact float64 value to float32 here equals the reference's float32 arithmetic.
        return np.clip(np.array(rows, dtype=np.float32), -1.0, 1.0)

    def _nearest_free_goal(self, pos: Sequence[float]) -> Tuple[Tuple[float, float], float]:
        px, py = float(pos[0]), float(pos[1])
        best = None
        gpos = self._gpos
        for g in range(self.n_goals):
            if self.goal_used[g]:
                continue
            dx, dy = gpos[g][0] - px, gpos[g][1] - py
            d = _hypot(dx, dy)
            if best is None or d < best[2]:
                best = (dx, dy, d)
        if best is None:
            return (0.0, 0.0), 0.0
        return (best[0], best[1]), min(1.0, best[2])
