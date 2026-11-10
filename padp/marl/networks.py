"""Shared actor with movement and interaction heads, plus a critic."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        _init(nn.Linear(in_dim, hidden), np.sqrt(2)), nn.ReLU(),
        _init(nn.Linear(hidden, hidden // 2), np.sqrt(2)), nn.ReLU(),
    )


class ActorCritic(nn.Module):
    """Parameters are shared across agents. The critic sees the global state (MAPPO) or the
    agent's own observation (IPPO), depending on what the trainer feeds it."""

    def __init__(self, obs_dim: int, critic_dim: int, n_move: int, n_interact: int, hidden: int) -> None:
        super().__init__()
        self.actor = _mlp(obs_dim, hidden)
        self.move_head = _init(nn.Linear(hidden // 2, n_move), 0.01)
        self.interact_head = _init(nn.Linear(hidden // 2, n_interact), 0.01)
        self.critic = nn.Sequential(_mlp(critic_dim, hidden), _init(nn.Linear(hidden // 2, 1), 1.0))

    def logits(self, obs: torch.Tensor):
        features = self.actor(obs)
        return self.move_head(features), self.interact_head(features)

    def value(self, critic_in: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_in).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Joint log-probability and entropy of factorized actions ``(B, 2)``."""
        move_logits, interact_logits = self.logits(obs)
        move, interact = Categorical(logits=move_logits), Categorical(logits=interact_logits)
        log_prob = move.log_prob(actions[:, 0]) + interact.log_prob(actions[:, 1])
        return log_prob, move.entropy() + interact.entropy()
