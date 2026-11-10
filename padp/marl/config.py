"""Trainer configuration."""
from __future__ import annotations

from dataclasses import dataclass

from padp.marl.sampler import SAMPLERS

CRITICS = ("central", "local")


@dataclass
class TrainConfig:
    total_steps: int = 2_000_000       # environment steps, summed over parallel environments
    n_envs: int = 16
    rollout_len: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    hidden: int = 256
    sampler: str = "independent"       # independent | coupled (object-anchored shared noise) | global | coupled_move
    anchor_error: float = 0.0          # coupled samplers: probability per agent and step of using a wrong anchor
    critic: str = "central"            # central = MAPPO, local = IPPO
    optimistic: bool = False           # clip advantages at zero (OptiMAPPO-style baseline)
    rack_curriculum_steps: int = 0     # 0 = off; otherwise add one rack every N steps up to env.n_racks
    log_every: int = 20_000
    eval_every: int = 200_000
    eval_episodes: int = 20
    save_every: int = 200_000
    seed: int = 0
    device: str = "cpu"
    threads: int = 1                   # torch threads per run; keep 1 when running sweeps in parallel

    def __post_init__(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler={self.sampler!r}; expected one of {SAMPLERS}")
        if self.critic not in CRITICS:
            raise ValueError(f"critic={self.critic!r}; expected one of {CRITICS}")
