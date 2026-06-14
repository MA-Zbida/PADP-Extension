import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def orthogonal_init(layer, gain=1.0):
    """Apply orthogonal initialization to a layer."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ActorCritic(nn.Module):
    """Shared MAPPO actor-critic with factorized movement and interaction heads."""

    def __init__(self, obs_dim: int, state_dim: int, n_actions, hidden_dim: int):
        super().__init__()
        if isinstance(n_actions, (tuple, list)):
            self.n_move_actions = int(n_actions[0])
            self.n_interaction_actions = int(n_actions[1])
        else:
            self.n_move_actions = int(n_actions)
            self.n_interaction_actions = 1

        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.move_head = nn.Linear(hidden_dim // 2, self.n_move_actions)
        self.interaction_head = nn.Linear(hidden_dim // 2, self.n_interaction_actions)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.actor_body.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=np.sqrt(2))
        orthogonal_init(self.move_head, gain=0.01)
        orthogonal_init(self.interaction_head, gain=0.01)

        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=np.sqrt(2))
        orthogonal_init(self.critic[-1], gain=1.0)

    def _dists(self, obs: torch.Tensor):
        features = self.actor_body(obs)
        move_dist = Categorical(logits=self.move_head(features))
        interaction_dist = Categorical(logits=self.interaction_head(features))
        return move_dist, interaction_dist

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        move_dist, interaction_dist = self._dists(obs)

        if deterministic:
            move_action = move_dist.logits.argmax(dim=-1)
            interaction_action = interaction_dist.logits.argmax(dim=-1)
        else:
            move_action = move_dist.sample()
            interaction_action = interaction_dist.sample()

        action = torch.stack([move_action, interaction_action], dim=-1)
        log_prob = move_dist.log_prob(move_action) + interaction_dist.log_prob(interaction_action)
        entropy = move_dist.entropy() + interaction_dist.entropy()
        return action, log_prob, entropy

    def get_value(self, state: torch.Tensor):
        return self.critic(state).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        move_actions = actions[:, 0]
        if actions.shape[-1] > 1:
            interaction_actions = actions[:, 1]
        else:
            interaction_actions = torch.zeros_like(move_actions)

        move_dist, interaction_dist = self._dists(obs)
        log_prob = move_dist.log_prob(move_actions) + interaction_dist.log_prob(interaction_actions)
        entropy = move_dist.entropy() + interaction_dist.entropy()
        return log_prob, entropy
