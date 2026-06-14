"""Continuous collaborative warehouse environment."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env, spaces

from environment.actions import (
    INTERACT_DROP,
    INTERACT_PICK,
    MOVE_DELTAS,
    MOVE_STAY,
    NUM_FLAT_ACTIONS,
    NUM_INTERACTION_ACTIONS,
    NUM_MOVE_ACTIONS,
    parse_factorized_actions,
)
from environment.geometry import (
    Rect,
    circle_any_rect_overlap,
    circle_in_bounds,
    circle_overlap,
    distance,
    grip_points,
    nearest_rect_distance,
    normalize_pos,
    opposite_sides,
    segment_distance,
)
from environment.layout import WarehouseLayout, make_warehouse_layout
from environment.renderer import WarehouseRenderer
from environment.rewards import DEFAULT_REWARD_CONFIG, EpisodeMetrics, StepEvents, compute_reward


Vec = np.ndarray


class CollaborativeCarryEnv(Env):
    """Cooperative continuous 2D warehouse carrying task."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        grid_size: int = 8,
        n_agents: int = 4,
        n_objects: Optional[int] = None,
        n_goals: Optional[int] = None,
        n_obstacles: int = 4,
        max_agents: int = 10,
        max_objects: Optional[int] = None,
        max_goals: Optional[int] = None,
        max_obstacles: int = 6,
        max_grid_size: int = 10,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        debug_render: bool = False,
    ) -> None:
        super().__init__()

        derived_objects = n_objects if n_objects is not None else max(1, int(np.ceil(n_agents / 2)))
        derived_goals = n_goals if n_goals is not None else derived_objects

        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_objects = derived_objects
        self.n_goals = derived_goals
        self.n_obstacles = n_obstacles
        self.max_agents = max(max_agents, n_agents)
        self.max_objects = max(max_objects or derived_objects, derived_objects)
        self.max_goals = max(max_goals or derived_goals, derived_goals)
        self.max_obstacles = max(max_obstacles, n_obstacles)
        self.max_grid_size = max(max_grid_size, grid_size)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.debug_render = debug_render

        self.agent_radius = 0.025
        self.object_radius = 0.035
        self.goal_radius = 0.065
        self.move_step = 0.025
        self.pickup_radius = 0.04

        self.n_move_actions = NUM_MOVE_ACTIONS
        self.n_interaction_actions = NUM_INTERACTION_ACTIONS
        self.n_actions = NUM_FLAT_ACTIONS
        self.action_space = spaces.MultiDiscrete(np.tile([NUM_MOVE_ACTIONS, NUM_INTERACTION_ACTIONS], self.n_agents))

        self.obs_dim_per_agent = 24
        self.obs_dim_shared = 3 + self.max_obstacles * 4 + self.max_objects * 7 + self.max_goals * 3
        obs_space_dict = {
            f"agent_{i + 1}": spaces.Box(
                low=-1.0, high=1.0, shape=(self.obs_dim_per_agent,), dtype=np.float32
            )
            for i in range(self.max_agents)
        }
        obs_space_dict["shared"] = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim_shared,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(obs_space_dict)

        self.reward_config: Dict[str, float] = dict(DEFAULT_REWARD_CONFIG)
        self.metrics = EpisodeMetrics()

        self.layout: WarehouseLayout = make_warehouse_layout(self.n_obstacles, self.max_obstacles)
        self.racks: List[Rect] = []
        self.agent_positions: List[Vec] = []
        self.object_positions: List[Vec] = []
        self.goal_positions: List[Vec] = []
        self.goal_used: List[bool] = []
        self.delivered: List[bool] = []
        self.object_holders: List[List[int]] = []
        self.agent_holding: List[Optional[int]] = []
        self.agent_grip_sides: List[Optional[str]] = []
        self.current_step = 0

        self._prev_agent_positions: List[Vec] = []
        self._prev_object_positions: List[Vec] = []
        self._prev_delivered: List[bool] = []
        self.last_events = StepEvents()
        self.last_reward_breakdown: Dict[str, float] = {}
        self.renderer = WarehouseRenderer(fps=self.metadata["render_fps"])

    def set_n_obstacles(self, n_obstacles: int) -> None:
        if n_obstacles > self.max_obstacles:
            raise ValueError(f"n_obstacles ({n_obstacles}) cannot exceed max_obstacles ({self.max_obstacles})")
        self.n_obstacles = n_obstacles

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_events = StepEvents()
        self.last_reward_breakdown = {}
        self.metrics.reset()

        self.layout = make_warehouse_layout(self.n_obstacles, self.max_obstacles)
        self.racks = list(self.layout.racks)
        self.goal_used = [False] * self.n_goals
        self.delivered = [False] * self.n_objects
        self.object_holders = [[] for _ in range(self.n_objects)]
        self.agent_holding = [None for _ in range(self.n_agents)]
        self.agent_grip_sides = [None for _ in range(self.n_agents)]

        self.agent_positions = self._sample_agents()
        self.goal_positions = self._sample_goals()
        self.object_positions = self._sample_objects()

        self._prev_agent_positions = [p.copy() for p in self.agent_positions]
        self._prev_object_positions = [p.copy() for p in self.object_positions]
        self._prev_delivered = list(self.delivered)

        return self._get_obs(), {}

    def step(self, action: Sequence[object]):
        actions = parse_factorized_actions(action, self.n_agents)
        self.current_step += 1
        self._prev_agent_positions = [p.copy() for p in self.agent_positions]
        self._prev_object_positions = [p.copy() for p in self.object_positions]
        self._prev_delivered = list(self.delivered)

        events = StepEvents(idle_actions=int(np.sum(actions[:, 0] == MOVE_STAY)))
        prev_potential = self._potential()
        prev_grip_potential = self._grip_potential()

        self._process_drops(actions, events)
        self._process_pickups(actions, events)
        self._resolve_movement(actions, events)

        events.carrier_pair_steps = sum(1 for holders in self.object_holders if len(holders) == 2)
        events.can_pick_agents = self._count_can_pick_agents()
        events.opposite_grip_pairs = self._count_opposite_grip_pairs()
        events.blocking += self._count_blocking_agents()
        events.grip_progress = 0.99 * self._grip_potential() - prev_grip_potential
        events.shaping = 0.99 * self._potential() - prev_potential

        self.metrics.update(events)
        reward, reward_info = compute_reward(events, self.reward_config)
        self.last_events = events
        self.last_reward_breakdown = reward_info

        terminated = all(self.delivered)
        truncated = self.current_step >= self.max_steps
        info = {
            "delivered": list(self.delivered),
            "reward_breakdown": reward_info,
            "metrics": self._compute_metrics(events),
            "step": self.current_step,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return self._get_obs()
        self.renderer.render(self, debug=self.debug_render)

    def close(self):
        self.renderer.close()

    def get_flat_obs(self) -> np.ndarray:
        obs = self._get_obs()
        agents = [obs[f"agent_{i + 1}"] for i in range(self.n_agents)]
        return np.concatenate(agents + [obs["shared"]])

    def get_agent_obs(self, agent_id: int) -> np.ndarray:
        obs = self._get_obs()
        return np.concatenate([obs[f"agent_{agent_id + 1}"], obs["shared"]])

    def _sample_agents(self) -> List[Vec]:
        points = [p.copy() for p in self.layout.spawn_points]
        self.np_random.shuffle(points)
        agents: List[Vec] = []
        for point in points:
            if len(agents) >= self.n_agents:
                break
            if self._valid_agent_spawn(point, agents):
                agents.append(point)
        while len(agents) < self.n_agents:
            point = self._sample_point_in_rect(self.layout.depot, self.agent_radius)
            if self._valid_agent_spawn(point, agents):
                agents.append(point)
        return agents

    def _sample_goals(self) -> List[Vec]:
        points = [p.copy() for p in self.layout.goal_points]
        self.np_random.shuffle(points)
        goals = points[: self.n_goals]
        while len(goals) < self.n_goals:
            goals.append(self._sample_point_in_rect(self.layout.shipping, self.goal_radius * 0.5))
        return goals

    def _sample_objects(self) -> List[Vec]:
        candidates = [p.copy() for p in self.layout.pick_faces]
        self.np_random.shuffle(candidates)
        objects: List[Vec] = []
        for point in candidates:
            if len(objects) >= self.n_objects:
                break
            if self._valid_object_spawn(point, objects):
                objects.append(point)
        while len(objects) < self.n_objects:
            point = np.array(
                [
                    float(self.np_random.uniform(0.18, 0.82)),
                    float(self.np_random.uniform(0.18, 0.82)),
                ],
                dtype=np.float32,
            )
            if self._valid_object_spawn(point, objects):
                objects.append(point)
        return objects

    def _sample_point_in_rect(self, rect: Rect, radius: float) -> Vec:
        return np.array(
            [
                float(self.np_random.uniform(rect.x1 + radius, rect.x2 - radius)),
                float(self.np_random.uniform(rect.y1 + radius, rect.y2 - radius)),
            ],
            dtype=np.float32,
        )

    def _valid_agent_spawn(self, point: Vec, existing: List[Vec]) -> bool:
        if not circle_in_bounds(point, self.agent_radius):
            return False
        if circle_any_rect_overlap(point, self.agent_radius, self.racks):
            return False
        return not any(circle_overlap(point, self.agent_radius, other, self.agent_radius) for other in existing)

    def _valid_object_spawn(self, point: Vec, existing: List[Vec]) -> bool:
        if not circle_in_bounds(point, self.object_radius):
            return False
        if circle_any_rect_overlap(point, self.object_radius, self.racks):
            return False
        if any(circle_overlap(point, self.object_radius, other, self.object_radius) for other in existing):
            return False
        if any(circle_overlap(point, self.object_radius, agent, self.agent_radius) for agent in self.agent_positions):
            return False
        return not any(distance(point, goal) < self.goal_radius + self.object_radius for goal in self.goal_positions)

    def _process_drops(self, actions: np.ndarray, events: StepEvents) -> None:
        for obj_idx, holders in enumerate(list(self.object_holders)):
            if self.delivered[obj_idx] or len(holders) != 2:
                continue
            drop_holders = [agent for agent in holders if actions[agent, 1] == INTERACT_DROP]
            if len(drop_holders) == 2:
                goal_idx = self._goal_containing_object(obj_idx)
                events.valid_drops += 2
                if goal_idx is not None:
                    self.delivered[obj_idx] = True
                    self.goal_used[goal_idx] = True
                    self.object_positions[obj_idx] = self.goal_positions[goal_idx].copy()
                    events.deliveries += 1
                else:
                    events.invalid_drops += 2
                self._release_object(obj_idx)
            elif len(drop_holders) == 1:
                events.invalid_drops += 1

    def _process_pickups(self, actions: np.ndarray, events: StepEvents) -> None:
        attempts: Dict[int, List[Tuple[int, str]]] = {}
        for agent_idx in range(self.n_agents):
            if actions[agent_idx, 1] != INTERACT_PICK:
                continue
            if self.agent_holding[agent_idx] is not None:
                events.invalid_pickups += 1
                continue

            result = self._nearest_grip_attempt(agent_idx)
            if result is None:
                events.invalid_pickups += 1
                continue
            obj_idx, side = result
            if len(self.object_holders[obj_idx]) >= 2:
                events.invalid_pickups += 1
                events.overstaffing += 1
                continue
            attempts.setdefault(obj_idx, []).append((agent_idx, side))

        for obj_idx, object_attempts in attempts.items():
            if len(object_attempts) != 2:
                events.invalid_pickups += len(object_attempts)
                events.overstaffing += max(0, len(object_attempts) - 2)
                continue
            (agent_a, side_a), (agent_b, side_b) = object_attempts
            if not opposite_sides(side_a, side_b):
                events.invalid_pickups += 2
                continue
            self.object_holders[obj_idx] = [agent_a, agent_b]
            self.agent_holding[agent_a] = obj_idx
            self.agent_holding[agent_b] = obj_idx
            self.agent_grip_sides[agent_a] = side_a
            self.agent_grip_sides[agent_b] = side_b
            events.valid_pickups += 2

    def _resolve_movement(self, actions: np.ndarray, events: StepEvents) -> None:
        old_agents = [p.copy() for p in self.agent_positions]
        old_objects = [p.copy() for p in self.object_positions]
        proposed_agents = [p.copy() for p in self.agent_positions]
        proposed_objects = [p.copy() for p in self.object_positions]

        group_for_agent: Dict[int, Tuple[str, int]] = {i: ("agent", i) for i in range(self.n_agents)}
        group_members: Dict[Tuple[str, int], List[int]] = {("agent", i): [i] for i in range(self.n_agents)}
        group_object: Dict[Tuple[str, int], int] = {}
        rejected: set[Tuple[str, int]] = set()
        candidate_moved_objects: set[int] = set()

        for obj_idx, holders in enumerate(self.object_holders):
            if self.delivered[obj_idx] or len(holders) != 2:
                continue

            group_id = ("object", obj_idx)
            group_members[group_id] = list(holders)
            group_object[group_id] = obj_idx
            for agent_idx in holders:
                group_for_agent[agent_idx] = group_id

            move_a, move_b = int(actions[holders[0], 0]), int(actions[holders[1], 0])
            if move_a != move_b:
                events.incompatible_carry_actions += 1
                continue
            if move_a == MOVE_STAY:
                continue

            delta = MOVE_DELTAS[move_a] * self.move_step
            for agent_idx in holders:
                proposed_agents[agent_idx] = old_agents[agent_idx] + delta
            proposed_objects[obj_idx] = old_objects[obj_idx] + delta
            candidate_moved_objects.add(obj_idx)

            reason = self._validate_carrier_group(obj_idx, holders, proposed_agents, proposed_objects)
            if reason is not None:
                self._count_static_collision(reason, events)
                rejected.add(group_id)

        for agent_idx in range(self.n_agents):
            if group_for_agent[agent_idx][0] == "object":
                continue
            move = int(actions[agent_idx, 0])
            if move == MOVE_STAY:
                continue
            proposed_agents[agent_idx] = old_agents[agent_idx] + MOVE_DELTAS[move] * self.move_step
            reason = self._validate_independent_agent(agent_idx, proposed_agents[agent_idx], proposed_objects)
            if reason is not None:
                self._count_static_collision(reason, events)
                rejected.add(group_for_agent[agent_idx])

        self._resolve_agent_collisions(old_agents, proposed_agents, group_for_agent, group_members, rejected, events)
        self._resolve_agent_object_collisions(proposed_agents, proposed_objects, group_for_agent, rejected, events)

        for group_id in rejected:
            for agent_idx in group_members.get(group_id, []):
                proposed_agents[agent_idx] = old_agents[agent_idx].copy()
            if group_id in group_object:
                obj_idx = group_object[group_id]
                proposed_objects[obj_idx] = old_objects[obj_idx].copy()

        self.agent_positions = proposed_agents
        self.object_positions = proposed_objects

        for obj_idx in candidate_moved_objects:
            if ("object", obj_idx) in rejected or self.delivered[obj_idx]:
                continue
            events.carrier_move_steps += 1
            events.carry_progress += self._nearest_free_goal_distance(old_objects[obj_idx]) - self._nearest_free_goal_distance(
                self.object_positions[obj_idx]
            )

    def _validate_carrier_group(
        self,
        obj_idx: int,
        holders: List[int],
        proposed_agents: List[Vec],
        proposed_objects: List[Vec],
    ) -> Optional[str]:
        obj_pos = proposed_objects[obj_idx]
        if not circle_in_bounds(obj_pos, self.object_radius):
            return "bound"
        if circle_any_rect_overlap(obj_pos, self.object_radius, self.racks):
            return "rack"
        for other_idx, other_pos in enumerate(proposed_objects):
            if other_idx == obj_idx or self.delivered[other_idx]:
                continue
            if circle_overlap(obj_pos, self.object_radius, other_pos, self.object_radius):
                return "object"
        for agent_idx in holders:
            reason = self._validate_agent_static(proposed_agents[agent_idx])
            if reason is not None:
                return reason
        return None

    def _validate_independent_agent(self, agent_idx: int, proposed_pos: Vec, proposed_objects: List[Vec]) -> Optional[str]:
        reason = self._validate_agent_static(proposed_pos)
        if reason is not None:
            return reason
        for obj_idx, obj_pos in enumerate(proposed_objects):
            if self.delivered[obj_idx]:
                continue
            if self.agent_holding[agent_idx] == obj_idx:
                continue
            if circle_overlap(proposed_pos, self.agent_radius, obj_pos, self.object_radius):
                return "object"
        return None

    def _validate_agent_static(self, proposed_pos: Vec) -> Optional[str]:
        if not circle_in_bounds(proposed_pos, self.agent_radius):
            return "bound"
        if circle_any_rect_overlap(proposed_pos, self.agent_radius, self.racks):
            return "rack"
        return None

    def _resolve_agent_collisions(
        self,
        old_agents: List[Vec],
        proposed_agents: List[Vec],
        group_for_agent: Dict[int, Tuple[str, int]],
        group_members: Dict[Tuple[str, int], List[int]],
        rejected: set[Tuple[str, int]],
        events: StepEvents,
    ) -> None:
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                group_i = group_for_agent[i]
                group_j = group_for_agent[j]
                if group_i == group_j:
                    continue

                final_overlap = circle_overlap(proposed_agents[i], self.agent_radius, proposed_agents[j], self.agent_radius)
                pass_through = (
                    segment_distance(old_agents[i], proposed_agents[i], old_agents[j], proposed_agents[j])
                    < self.agent_radius * 2.0
                    and (distance(old_agents[i], proposed_agents[i]) > 1e-6 or distance(old_agents[j], proposed_agents[j]) > 1e-6)
                )
                if final_overlap or pass_through:
                    rejected.add(group_i)
                    rejected.add(group_j)
                    events.agent_collisions += 1

    def _resolve_agent_object_collisions(
        self,
        proposed_agents: List[Vec],
        proposed_objects: List[Vec],
        group_for_agent: Dict[int, Tuple[str, int]],
        rejected: set[Tuple[str, int]],
        events: StepEvents,
    ) -> None:
        for agent_idx, agent_pos in enumerate(proposed_agents):
            for obj_idx, obj_pos in enumerate(proposed_objects):
                if self.delivered[obj_idx] or self.agent_holding[agent_idx] == obj_idx:
                    continue
                if not circle_overlap(agent_pos, self.agent_radius, obj_pos, self.object_radius):
                    continue
                rejected.add(group_for_agent[agent_idx])
                if len(self.object_holders[obj_idx]) == 2:
                    rejected.add(("object", obj_idx))
                events.object_collisions += 1

    def _count_static_collision(self, reason: str, events: StepEvents) -> None:
        if reason == "rack":
            events.rack_collisions += 1
        elif reason == "bound":
            events.bound_collisions += 1
        else:
            events.object_collisions += 1

    def _nearest_grip_attempt(self, agent_idx: int) -> Optional[Tuple[int, str]]:
        agent_pos = self.agent_positions[agent_idx]
        best: Optional[Tuple[float, int, str]] = None
        for obj_idx, obj_pos in enumerate(self.object_positions):
            if self.delivered[obj_idx]:
                continue
            for side, grip in grip_points(obj_pos, self.object_radius, self.agent_radius).items():
                d = distance(agent_pos, grip)
                if d <= self.pickup_radius and (best is None or d < best[0]):
                    best = (d, obj_idx, side)
        if best is None:
            return None
        return best[1], best[2]

    def _goal_containing_object(self, obj_idx: int) -> Optional[int]:
        obj_pos = self.object_positions[obj_idx]
        for goal_idx, goal_pos in enumerate(self.goal_positions):
            if self.goal_used[goal_idx]:
                continue
            if distance(obj_pos, goal_pos) <= self.goal_radius:
                return goal_idx
        return None

    def _release_object(self, obj_idx: int) -> None:
        for agent_idx in list(self.object_holders[obj_idx]):
            self.agent_holding[agent_idx] = None
            self.agent_grip_sides[agent_idx] = None
        self.object_holders[obj_idx] = []

    def _count_blocking_agents(self) -> int:
        blocking = 0
        for obj_idx, holders in enumerate(self.object_holders):
            if len(holders) != 2 or self.delivered[obj_idx]:
                continue
            for agent_idx, pos in enumerate(self.agent_positions):
                if agent_idx in holders:
                    continue
                if distance(pos, self.object_positions[obj_idx]) < self.object_radius + self.agent_radius + 0.05:
                    blocking += 1
        return blocking

    def _nearest_free_goal_distance(self, pos: Vec) -> float:
        distances = [
            distance(pos, goal_pos)
            for goal_idx, goal_pos in enumerate(self.goal_positions)
            if not self.goal_used[goal_idx]
        ]
        return min(distances) if distances else 0.0

    def _potential(self) -> float:
        needy_objects = [
            obj_pos
            for obj_idx, obj_pos in enumerate(self.object_positions)
            if not self.delivered[obj_idx] and len(self.object_holders[obj_idx]) < 2
        ]
        if not needy_objects:
            return 0.0
        potential = 0.0
        for agent_idx, agent_pos in enumerate(self.agent_positions):
            if self.agent_holding[agent_idx] is not None:
                continue
            potential -= min(distance(agent_pos, obj_pos) for obj_pos in needy_objects)
        return potential

    def _grip_potential(self) -> float:
        candidate_grips: List[Vec] = []
        for obj_idx, obj_pos in enumerate(self.object_positions):
            if self.delivered[obj_idx] or len(self.object_holders[obj_idx]) >= 2:
                continue
            candidate_grips.extend(grip_points(obj_pos, self.object_radius, self.agent_radius).values())
        if not candidate_grips:
            return 0.0

        potential = 0.0
        for agent_idx, agent_pos in enumerate(self.agent_positions):
            if self.agent_holding[agent_idx] is not None:
                continue
            potential -= min(distance(agent_pos, grip) for grip in candidate_grips)
        return potential

    def _count_can_pick_agents(self) -> int:
        count = 0
        for agent_idx in range(self.n_agents):
            if self.agent_holding[agent_idx] is None and self._nearest_grip_attempt(agent_idx) is not None:
                count += 1
        return count

    def _count_opposite_grip_pairs(self) -> int:
        pair_count = 0
        for obj_idx, obj_pos in enumerate(self.object_positions):
            if self.delivered[obj_idx] or len(self.object_holders[obj_idx]) >= 2:
                continue
            sides: Dict[str, int] = {}
            for agent_idx, agent_pos in enumerate(self.agent_positions):
                if self.agent_holding[agent_idx] is not None:
                    continue
                for side, grip in grip_points(obj_pos, self.object_radius, self.agent_radius).items():
                    if distance(agent_pos, grip) <= self.pickup_radius:
                        sides[side] = sides.get(side, 0) + 1
                        break
            pair_count += min(sides.get("left", 0), sides.get("right", 0))
            pair_count += min(sides.get("top", 0), sides.get("bottom", 0))
        return pair_count

    def _compute_metrics(self, events: StepEvents) -> Dict[str, float]:
        delivered_count = int(sum(self.delivered))
        completion_ratio = delivered_count / max(1, self.n_objects)
        carrier_pairs = sum(1 for holders in self.object_holders if len(holders) == 2)
        needy_objects = [
            obj_pos
            for obj_idx, obj_pos in enumerate(self.object_positions)
            if not self.delivered[obj_idx] and len(self.object_holders[obj_idx]) < 2
        ]
        mean_agent_object_distance = 0.0
        if needy_objects:
            mean_agent_object_distance = float(
                np.mean([min(distance(agent, obj) for obj in needy_objects) for agent in self.agent_positions])
            )
        remaining_goal_distance = sum(
            self._nearest_free_goal_distance(obj_pos)
            for obj_idx, obj_pos in enumerate(self.object_positions)
            if not self.delivered[obj_idx]
        )

        totals = self.metrics.totals
        safety_events = totals["agent_collisions"] + totals["rack_collisions"] + totals["bound_collisions"] + totals["object_collisions"]
        return {
            "success": float(all(self.delivered)),
            "completion_ratio": float(completion_ratio),
            "delivered_count": float(delivered_count),
            "new_deliveries": float(events.deliveries),
            "steps": float(self.current_step),
            "remaining_object_goal_distance": float(remaining_goal_distance),
            "mean_agent_object_distance": float(mean_agent_object_distance),
            "carrier_pairs": float(carrier_pairs),
            "valid_pickups": float(events.valid_pickups),
            "invalid_pickups": float(events.invalid_pickups),
            "valid_drops": float(events.valid_drops),
            "invalid_drops": float(events.invalid_drops),
            "agent_collisions": float(events.agent_collisions),
            "rack_collisions": float(events.rack_collisions),
            "bound_collisions": float(events.bound_collisions),
            "object_collisions": float(events.object_collisions),
            "overstaffing": float(events.overstaffing),
            "blocking": float(events.blocking),
            "idle_actions": float(events.idle_actions),
            "can_pick_agents": float(events.can_pick_agents),
            "opposite_grip_pairs": float(events.opposite_grip_pairs),
            "grip_progress": float(events.grip_progress),
            "carrier_move_steps": float(events.carrier_move_steps),
            "cumulative_valid_pickups": totals["valid_pickups"],
            "cumulative_invalid_pickups": totals["invalid_pickups"],
            "cumulative_valid_drops": totals["valid_drops"],
            "cumulative_invalid_drops": totals["invalid_drops"],
            "cumulative_agent_collisions": totals["agent_collisions"],
            "cumulative_rack_collisions": totals["rack_collisions"],
            "cumulative_bound_collisions": totals["bound_collisions"],
            "cumulative_object_collisions": totals["object_collisions"],
            "cumulative_overstaffing": totals["overstaffing"],
            "cumulative_blocking": totals["blocking"],
            "cumulative_idle_actions": totals["idle_actions"],
            "cumulative_can_pick_agents": totals["can_pick_agents"],
            "cumulative_opposite_grip_pairs": totals["opposite_grip_pairs"],
            "cumulative_carrier_pair_steps": totals["carrier_pair_steps"],
            "cumulative_carrier_move_steps": totals["carrier_move_steps"],
            "cumulative_safety_events": safety_events,
            "cumulative_obstacle_hits": totals["rack_collisions"] + totals["bound_collisions"],
            "cumulative_wall_collisions": totals["agent_collisions"] + totals["object_collisions"],
        }

    def _get_obs(self):
        agent_obs: List[np.ndarray] = []
        for agent_idx in range(self.n_agents):
            agent_pos = self.agent_positions[agent_idx]
            held_obj = self.agent_holding[agent_idx]
            nearest_obj_vec, nearest_obj_dist = self._nearest_needy_object_features(agent_pos)
            nearest_goal_vec, nearest_goal_dist = self._nearest_goal_features(agent_pos)
            nearest_agent_vec, nearest_agent_dist = self._nearest_agent_features(agent_idx)
            can_pick = 1.0 if held_obj is None and self._nearest_grip_attempt(agent_idx) is not None else 0.0
            can_drop = 1.0 if held_obj is not None and self._goal_containing_object(held_obj) is not None else 0.0
            partner_vec = np.zeros(2, dtype=np.float32)
            has_partner = 0.0
            if held_obj is not None and len(self.object_holders[held_obj]) == 2:
                partner = next(a for a in self.object_holders[held_obj] if a != agent_idx)
                partner_vec = self.agent_positions[partner] - agent_pos
                has_partner = 1.0
            density = sum(
                1 for other_idx, other_pos in enumerate(self.agent_positions)
                if other_idx != agent_idx and distance(agent_pos, other_pos) <= 0.12
            ) / max(1, self.n_agents - 1)
            holding_flag = 1.0 if held_obj is not None else 0.0
            held_obj_norm = 0.0 if held_obj is None else (held_obj + 1) / max(1, self.max_objects)
            nearest_rack_dist = min(1.0, nearest_rect_distance(agent_pos, self.racks))

            obs = np.array(
                [
                    *normalize_pos(agent_pos),
                    nearest_obj_vec[0],
                    nearest_obj_vec[1],
                    nearest_obj_dist,
                    nearest_goal_vec[0],
                    nearest_goal_vec[1],
                    nearest_goal_dist,
                    nearest_rack_dist,
                    holding_flag,
                    held_obj_norm,
                    has_partner,
                    can_pick,
                    can_drop,
                    partner_vec[0],
                    partner_vec[1],
                    nearest_agent_vec[0],
                    nearest_agent_vec[1],
                    nearest_agent_dist,
                    density,
                    float(any(not d and len(h) < 2 for d, h in zip(self.delivered, self.object_holders))),
                    float(self.last_events.overstaffing > 0),
                    float(self.last_events.blocking > 0),
                    0.0,
                ],
                dtype=np.float32,
            )
            agent_obs.append(np.clip(obs, -1.0, 1.0))

        while len(agent_obs) < self.max_agents:
            agent_obs.append(np.full(self.obs_dim_per_agent, -1.0, dtype=np.float32))

        shared_obs = self._get_shared_obs()
        return {
            **{f"agent_{i + 1}": agent_obs[i] for i in range(self.max_agents)},
            "shared": shared_obs,
        }

    def _nearest_needy_object_features(self, pos: Vec) -> Tuple[np.ndarray, float]:
        candidates = [
            obj
            for obj_idx, obj in enumerate(self.object_positions)
            if not self.delivered[obj_idx] and len(self.object_holders[obj_idx]) < 2
        ]
        if not candidates:
            return np.zeros(2, dtype=np.float32), 1.0
        target = min(candidates, key=lambda obj: distance(pos, obj))
        vec = target - pos
        return np.clip(vec, -1.0, 1.0).astype(np.float32), min(1.0, distance(pos, target))

    def _nearest_goal_features(self, pos: Vec) -> Tuple[np.ndarray, float]:
        candidates = [goal for idx, goal in enumerate(self.goal_positions) if not self.goal_used[idx]]
        if not candidates:
            return np.zeros(2, dtype=np.float32), 1.0
        target = min(candidates, key=lambda goal: distance(pos, goal))
        vec = target - pos
        return np.clip(vec, -1.0, 1.0).astype(np.float32), min(1.0, distance(pos, target))

    def _nearest_agent_features(self, agent_idx: int) -> Tuple[np.ndarray, float]:
        pos = self.agent_positions[agent_idx]
        candidates = [other for idx, other in enumerate(self.agent_positions) if idx != agent_idx]
        if not candidates:
            return np.zeros(2, dtype=np.float32), 1.0
        target = min(candidates, key=lambda other: distance(pos, other))
        vec = target - pos
        return np.clip(vec, -1.0, 1.0).astype(np.float32), min(1.0, distance(pos, target))

    def _get_shared_obs(self) -> np.ndarray:
        rack_flat: List[float] = []
        for idx in range(self.max_obstacles):
            if idx < len(self.racks):
                rack = self.racks[idx]
                rack_flat.extend([*normalize_pos(rack.center), rack.width, rack.height])
            else:
                rack_flat.extend([-1.0, -1.0, -1.0, -1.0])

        objects_flat: List[float] = []
        for idx in range(self.max_objects):
            if idx < self.n_objects:
                obj = self.object_positions[idx]
                goal_vec, _ = self._nearest_goal_features(obj)
                goal_pos = np.clip(obj + goal_vec, 0.0, 1.0)
                objects_flat.extend(
                    [
                        *normalize_pos(obj),
                        *normalize_pos(goal_pos),
                        1.0 if len(self.object_holders[idx]) == 2 else 0.0,
                        1.0 if self.delivered[idx] else 0.0,
                        len(self.object_holders[idx]) / 2.0,
                    ]
                )
            else:
                objects_flat.extend([-1.0] * 7)

        goals_flat: List[float] = []
        for idx in range(self.max_goals):
            if idx < self.n_goals:
                goals_flat.extend([*normalize_pos(self.goal_positions[idx]), 1.0 if self.goal_used[idx] else 0.0])
            else:
                goals_flat.extend([-1.0, -1.0, -1.0])

        shared = np.array(
            [
                self.current_step / max(1, self.max_steps),
                self.n_agents / max(1, self.max_agents),
                self.n_objects / max(1, self.max_objects),
                *rack_flat,
                *objects_flat,
                *goals_flat,
            ],
            dtype=np.float32,
        )
        return np.clip(shared, -1.0, 1.0)
