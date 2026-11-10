"""Reward definition and episode bookkeeping.

The shaped reward adds only two kinds of terms on top of the delivery reward and step cost:
negative-only penalties, and potential-based shaping gamma * Phi(s') - Phi(s) (Ng et al., 1999).
Neither can be farmed, so the reward-optimal behaviour stays "deliver every object".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RewardConfig:
    delivery: float = 10.0
    step: float = -0.01
    collision: float = -0.05
    invalid: float = -0.01
    # Potential coefficients (shaped mode only).
    attach_potential: float = 0.5     # times the fraction of the weight covered by attached capacity
    lift_potential: float = 1.0       # once the object is lifted
    carry_potential: float = 3.0      # times progress of a lifted object toward its nearest free goal
    approach_potential: float = 1.0   # times (minus) each free agent's distance to the nearest free grip slot
    gamma: float = 0.99               # must match the trainer's discount for policy invariance


@dataclass
class StepEvents:
    deliveries: int = 0
    attaches: int = 0
    detaches: int = 0
    lifts: int = 0
    releases_outside_goal: int = 0
    drop_disagreements: int = 0
    invalid_actions: int = 0
    agent_collisions: int = 0
    rack_collisions: int = 0
    bound_collisions: int = 0
    object_collisions: int = 0
    carry_steps: int = 0
    carry_agreements: int = 0

    @property
    def collisions(self) -> int:
        return self.agent_collisions + self.rack_collisions + self.bound_collisions + self.object_collisions


_EVENT_FIELDS = tuple(StepEvents.__dataclass_fields__)


def compute_reward(events: StepEvents, shaping: float, cfg: RewardConfig, mode: str) -> Tuple[float, Dict[str, float]]:
    parts = {"delivery": cfg.delivery * events.deliveries, "step": cfg.step}
    if mode == "shaped":
        parts["collision"] = cfg.collision * events.collisions
        parts["invalid"] = cfg.invalid * events.invalid_actions
        parts["shaping"] = shaping
    return float(sum(parts.values())), parts


@dataclass
class EpisodeTracker:
    """Accumulates step events into the end-of-episode summary."""

    totals: StepEvents = field(default_factory=StepEvents)
    episode_return: float = 0.0
    delivery_sizes: List[int] = field(default_factory=list)
    lift_waits: List[int] = field(default_factory=list)
    lift_spares: List[float] = field(default_factory=list)
    lift_sizes: List[int] = field(default_factory=list)

    def add(self, events: StepEvents, reward: float) -> None:
        t = self.totals
        for name in _EVENT_FIELDS:
            setattr(t, name, getattr(t, name) + getattr(events, name))
        self.episode_return += reward

    def record_lift(self, wait: int, spare: float, size: int) -> None:
        self.lift_waits.append(wait)
        self.lift_spares.append(spare)
        self.lift_sizes.append(size)

    def record_delivery(self, size: int) -> None:
        self.delivery_sizes.append(size)

    def summary(self, n_objects: int, steps: int) -> Dict[str, float]:
        t = self.totals
        delivered = len(self.delivery_sizes)
        return {
            "success": float(delivered == n_objects),
            "completion": delivered / max(1, n_objects),
            "steps": float(steps),
            "return": self.episode_return,
            "deliveries": float(delivered),
            "deliveries_k1": float(sum(1 for s in self.delivery_sizes if s == 1)),
            "deliveries_k2": float(sum(1 for s in self.delivery_sizes if s == 2)),
            "deliveries_k3plus": float(sum(1 for s in self.delivery_sizes if s >= 3)),
            "carry_agreement": t.carry_agreements / t.carry_steps if t.carry_steps else math.nan,
            "help_wait": _mean(self.lift_waits),
            "spare_capacity": _mean(self.lift_spares),
            "coalition_size": _mean(self.lift_sizes),
            "lifts": float(t.lifts),
            "releases_outside_goal": float(t.releases_outside_goal),
            "drop_disagreements": float(t.drop_disagreements),
            "detaches": float(t.detaches),
            "invalid_actions": float(t.invalid_actions),
            "collisions": float(t.collisions),
        }


def _mean(values) -> float:
    return float(sum(values) / len(values)) if values else math.nan
