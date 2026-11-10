"""Environment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from padp.env.rewards import RewardConfig

LAYOUTS = ("warehouse", "open")
SPAWNS = ("depot", "near_objects")
ATTACH_MODES = ("incremental", "simultaneous")
COUPLING_RULES = ("strict", "average", "leader")
REWARD_MODES = ("shaped", "sparse")


@dataclass
class EnvConfig:
    n_agents: int = 4
    n_objects: int = 2
    n_racks: int = 4
    layout: str = "warehouse"                     # warehouse: racks + pick faces; open: no racks
    spawn: str = "depot"                          # depot | near_objects
    carry_distance: Optional[List[float]] = None  # [min, max] object-to-goal distance; None = layout default
    max_steps: int = 200
    capacities: List[float] = field(default_factory=lambda: [50.0])  # sampled per agent each episode
    weights: List[float] = field(default_factory=lambda: [90.0])     # sampled per object each episode
    attach_mode: str = "incremental"              # incremental: attach and wait; simultaneous: all at once
    coupling_rule: str = "strict"                 # strict | average | leader
    reward_mode: str = "shaped"                   # shaped | sparse
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Padding sizes keep observation/state dimensions fixed across settings.
    max_agents: int = 6
    max_objects: int = 3
    max_racks: int = 6
    max_capacity: float = 100.0
    max_weight: float = 200.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _check("layout", self.layout, LAYOUTS)
        _check("spawn", self.spawn, SPAWNS)
        _check("attach_mode", self.attach_mode, ATTACH_MODES)
        _check("coupling_rule", self.coupling_rule, COUPLING_RULES)
        _check("reward_mode", self.reward_mode, REWARD_MODES)
        if not 1 <= self.n_agents <= self.max_agents:
            raise ValueError(f"n_agents must be in [1, max_agents={self.max_agents}]")
        if not 1 <= self.n_objects <= self.max_objects:
            raise ValueError(f"n_objects must be in [1, max_objects={self.max_objects}]")
        if not 0 <= self.n_racks <= self.max_racks:
            raise ValueError(f"n_racks must be in [0, max_racks={self.max_racks}]")
        if self.carry_distance is not None and len(self.carry_distance) != 2:
            raise ValueError("carry_distance must be [min, max]")
        # Each object has four grip slots, so at most four agents can lift it.
        if max(self.weights) > max(self.capacities) * min(4, self.n_agents):
            raise ValueError("some weight can never be lifted, even by the strongest possible coalition")


def _check(name: str, value: str, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r}; expected one of {allowed}")
