"""Reward and metric bookkeeping for continuous collaborative carrying."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StepEvents:
    deliveries: int = 0
    valid_pickups: int = 0
    invalid_pickups: int = 0
    valid_drops: int = 0
    invalid_drops: int = 0
    agent_collisions: int = 0
    rack_collisions: int = 0
    bound_collisions: int = 0
    object_collisions: int = 0
    overstaffing: int = 0
    blocking: int = 0
    carrier_pair_steps: int = 0
    carrier_move_steps: int = 0
    incompatible_carry_actions: int = 0
    idle_actions: int = 0
    can_pick_agents: int = 0
    opposite_grip_pairs: int = 0
    carry_progress: float = 0.0
    grip_progress: float = 0.0
    shaping: float = 0.0


@dataclass
class EpisodeMetrics:
    totals: Dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.totals = {
            "deliveries": 0.0,
            "valid_pickups": 0.0,
            "invalid_pickups": 0.0,
            "valid_drops": 0.0,
            "invalid_drops": 0.0,
            "agent_collisions": 0.0,
            "rack_collisions": 0.0,
            "bound_collisions": 0.0,
            "object_collisions": 0.0,
            "overstaffing": 0.0,
            "blocking": 0.0,
            "idle_actions": 0.0,
            "can_pick_agents": 0.0,
            "opposite_grip_pairs": 0.0,
            "carrier_pair_steps": 0.0,
            "carrier_move_steps": 0.0,
            "incompatible_carry_actions": 0.0,
        }

    def update(self, events: StepEvents) -> None:
        self.totals["deliveries"] += events.deliveries
        self.totals["valid_pickups"] += events.valid_pickups
        self.totals["invalid_pickups"] += events.invalid_pickups
        self.totals["valid_drops"] += events.valid_drops
        self.totals["invalid_drops"] += events.invalid_drops
        self.totals["agent_collisions"] += events.agent_collisions
        self.totals["rack_collisions"] += events.rack_collisions
        self.totals["bound_collisions"] += events.bound_collisions
        self.totals["object_collisions"] += events.object_collisions
        self.totals["overstaffing"] += events.overstaffing
        self.totals["blocking"] += events.blocking
        self.totals["idle_actions"] += events.idle_actions
        self.totals["can_pick_agents"] += events.can_pick_agents
        self.totals["opposite_grip_pairs"] += events.opposite_grip_pairs
        self.totals["carrier_pair_steps"] += events.carrier_pair_steps
        self.totals["carrier_move_steps"] += events.carrier_move_steps
        self.totals["incompatible_carry_actions"] += events.incompatible_carry_actions


DEFAULT_REWARD_CONFIG = {
    "alpha_del": 100.0,
    "alpha_step": -0.08,
    "alpha_valid_pickup": 5.0,
    "alpha_invalid_pickup": -0.05,
    "alpha_valid_drop": 2.0,
    "alpha_invalid_drop": -0.05,
    "alpha_collision": -0.75,
    "alpha_overstaffing": -0.2,
    "alpha_blocking": -0.5,
    "alpha_can_pick": 0.1,
    "alpha_opposite_grip_pair": 1.0,
    "alpha_grip_progress": 8.0,
    "alpha_carry_progress": 40.0,
    "alpha_approach": 1.0,
}


def compute_reward(events: StepEvents, reward_config: Dict[str, float] | None = None) -> tuple[float, Dict[str, float]]:
    cfg = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
    breakdown: Dict[str, float] = {}

    breakdown["step"] = cfg["alpha_step"]
    breakdown["delivery"] = cfg["alpha_del"] * events.deliveries
    breakdown["valid_pickup"] = cfg["alpha_valid_pickup"] * events.valid_pickups
    breakdown["invalid_pickup"] = cfg["alpha_invalid_pickup"] * events.invalid_pickups
    breakdown["valid_drop"] = cfg["alpha_valid_drop"] * events.valid_drops
    breakdown["invalid_drop"] = cfg["alpha_invalid_drop"] * events.invalid_drops
    breakdown["collision"] = cfg["alpha_collision"] * (
        events.agent_collisions + events.rack_collisions + events.bound_collisions + events.object_collisions
    )
    breakdown["overstaffing"] = cfg["alpha_overstaffing"] * events.overstaffing
    breakdown["blocking"] = cfg["alpha_blocking"] * events.blocking
    breakdown["can_pick"] = cfg["alpha_can_pick"] * events.can_pick_agents
    breakdown["opposite_grip_pair"] = cfg["alpha_opposite_grip_pair"] * events.opposite_grip_pairs
    breakdown["grip_progress"] = cfg["alpha_grip_progress"] * events.grip_progress
    breakdown["carry_progress"] = cfg["alpha_carry_progress"] * events.carry_progress
    breakdown["shaping"] = cfg["alpha_approach"] * events.shaping

    total = float(sum(breakdown.values()))
    return total, {key: value for key, value in breakdown.items() if abs(value) > 1e-9}
