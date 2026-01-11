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
    """Shared actor-critic network for MAPPO with improved architecture."""
    
    def __init__(self, obs_dim: int, state_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        
        # Actor (decentralized - uses local obs) - 3 hidden layers
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        
        # Critic (centralized - uses global state) - 3 hidden layers
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Apply orthogonal initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.actor.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=np.sqrt(2))
        # Last layer of actor should have smaller gain
        orthogonal_init(self.actor[-1], gain=0.01)
        
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=np.sqrt(2))
        # Last layer of critic should have gain=1
        orthogonal_init(self.critic[-1], gain=1.0)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def get_value(self, state: torch.Tensor):
        return self.critic(state).squeeze(-1)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, n_envs: int, n_agents: int, episode_limit: int, obs_dim: int, state_dim: int):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        self.reset()
    
    def reset(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, obs, states, actions, log_probs, rewards, dones, values):
        self.obs.append(obs)
        self.states.append(states)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)
    
    def compute_returns(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE returns."""
        n_steps = len(self.rewards)
        
        returns = np.zeros((n_steps, self.n_envs))
        advantages = np.zeros((n_steps, self.n_envs))
        
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - np.array(self.dones[t])
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - np.array(self.dones[t])
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + np.array(self.values)
        
        return returns, advantages
    
    def get_batches(self, returns: np.ndarray, advantages: np.ndarray, num_minibatches: int):
        """Generate minibatches for training."""
        n_steps = len(self.obs)
        batch_size = n_steps * self.n_envs
        minibatch_size = batch_size // num_minibatches
        
        # Flatten all data
        obs_flat = []
        states_flat = []
        actions_flat = []
        log_probs_flat = []
        returns_flat = []
        advantages_flat = []
        
        for t in range(n_steps):
            for e in range(self.n_envs):
                for a in range(self.n_agents):
                    obs_flat.append(self.obs[t][e][a])
                    states_flat.append(self.states[t][e])
                    actions_flat.append(self.actions[t][e][a])
                    log_probs_flat.append(self.log_probs[t][e][a])
                    returns_flat.append(returns[t, e])
                    advantages_flat.append(advantages[t, e])
        
        obs_flat = np.array(obs_flat)
        states_flat = np.array(states_flat)
        actions_flat = np.array(actions_flat)
        log_probs_flat = np.array(log_probs_flat)
        returns_flat = np.array(returns_flat)
        advantages_flat = np.array(advantages_flat)
        
        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # Random permutation
        indices = np.random.permutation(len(obs_flat))
        
        for start in range(0, len(obs_flat), minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]
            
            yield (
                obs_flat[batch_indices],
                states_flat[batch_indices],
                actions_flat[batch_indices],
                log_probs_flat[batch_indices],
                returns_flat[batch_indices],
                advantages_flat[batch_indices],
            )