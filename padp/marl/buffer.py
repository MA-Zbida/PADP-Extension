"""Preallocated rollout storage with generalized advantage estimation."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


class RolloutBuffer:
    """Arrays are ``(T, E, N, ...)``: time, environment, agent.

    Rewards are stored per agent so that the time-limit bootstrap (gamma * V(s_T), added to the
    last reward of a truncated episode) can use per-agent values from a local critic.
    """

    def __init__(self, T: int, E: int, N: int, obs_dim: int, critic_dim: int) -> None:
        self.T, self.E, self.N = T, E, N
        self.obs = np.zeros((T, E, N, obs_dim), dtype=np.float32)
        self.critic_in = np.zeros((T, E, N, critic_dim), dtype=np.float32)
        self.actions = np.zeros((T, E, N, 2), dtype=np.int64)
        self.log_probs = np.zeros((T, E, N), dtype=np.float32)
        self.values = np.zeros((T, E, N), dtype=np.float32)
        self.rewards = np.zeros((T, E, N), dtype=np.float32)
        self.dones = np.zeros((T, E), dtype=np.float32)

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(self.rewards)
        gae = np.zeros((self.E, self.N), dtype=np.float32)
        for t in reversed(range(self.T)):
            next_values = last_values if t == self.T - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t][:, None]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae
        return advantages, advantages + self.values

    def minibatches(self, advantages: np.ndarray, returns: np.ndarray, n: int, rng: np.random.Generator) -> Iterator:
        flat = lambda x: x.reshape(self.T * self.E * self.N, *x.shape[3:])  # noqa: E731
        data = [flat(self.obs), flat(self.critic_in), flat(self.actions), flat(self.log_probs), flat(advantages), flat(returns)]
        order = rng.permutation(len(data[0]))
        for chunk in np.array_split(order, n):
            yield tuple(d[chunk] for d in data)
