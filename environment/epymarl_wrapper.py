"""EPyMARL-compatible wrapper for CollaborativeCarryEnv."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from environment.env import CollaborativeCarryEnv


class CollaborativeCarryMARL:
    """
    EPyMARL-compatible wrapper.
    
    EPyMARL expects:
    - get_obs(): returns list of obs per agent
    - get_state(): returns global state
    - get_avail_actions(): returns available actions per agent
    - step(actions): takes list of actions
    - reset(): resets environment
    - Various properties: n_agents, n_actions, episode_limit, etc.
    """

    def __init__(self, **kwargs):
        self.env = CollaborativeCarryEnv(
            grid_size=kwargs.get("grid_size", 8),
            n_agents=kwargs.get("n_agents", 4),
            n_objects=kwargs.get("n_objects", None),
            n_goals=kwargs.get("n_goals", None),
            n_obstacles=kwargs.get("n_obstacles", 4),
            max_agents=kwargs.get("max_agents", 10),
            max_objects=kwargs.get("max_objects", None),
            max_goals=kwargs.get("max_goals", None),
            max_obstacles=kwargs.get("max_obstacles", 6),
            max_grid_size=kwargs.get("max_grid_size", 10),
            max_steps=kwargs.get("episode_limit", 100),
            render_mode=kwargs.get("render_mode", None),
        )
        
        self.n_agents = self.env.n_agents
        self.n_actions = len(self.env._action_to_delta)
        self.episode_limit = self.env.max_steps
        
        self._obs = None
        self._state = None
        self._episode_steps = 0
    
    def set_n_obstacles(self, n_obstacles: int):
        """Change number of obstacles for curriculum learning."""
        self.env.set_n_obstacles(n_obstacles)

    def reset(self):
        """Reset the environment and return initial observations."""
        obs_dict, _ = self.env.reset()
        self._episode_steps = 0
        self._cache_obs(obs_dict)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """
        Execute actions for all agents.
        
        Args:
            actions: list of action indices, one per agent
            
        Returns:
            reward: shared team reward (float)
            terminated: whether episode ended
            info: additional information
        """
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        self._episode_steps += 1
        self._cache_obs(obs_dict)
        
        done = terminated or truncated
        info["episode_steps"] = self._episode_steps
        
        return reward, done, info

    def _cache_obs(self, obs_dict):
        """Cache observations for get_obs() and get_state()."""
        self._obs = [obs_dict[f"agent_{i+1}"] for i in range(self.n_agents)]
        self._state = np.concatenate(self._obs + [obs_dict["shared"]])

    def get_obs(self):
        """Returns all agent observations as a list."""
        return [obs.copy() for obs in self._obs]

    def get_obs_agent(self, agent_id):
        """Returns observation for a specific agent."""
        return self._obs[agent_id].copy()

    def get_obs_size(self):
        """Returns the size of the observation for each agent."""
        return self.env.obs_dim_per_agent

    def get_state(self):
        """Returns the global state (centralized critic input)."""
        return self._state.copy()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.env.obs_dim_per_agent * self.n_agents + self.env.obs_dim_shared

    def get_avail_actions(self):
        """Returns available actions for all agents (all actions always available)."""
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns available actions for a specific agent."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions per agent."""
        return self.n_actions

    def render(self):
        """Render the environment."""
        self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def seed(self, seed=None):
        """Set the random seed."""
        if seed is not None:
            self.env.reset(seed=seed)

    def get_env_info(self):
        """Returns environment information dict required by EPyMARL."""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }


# Factory function for EPyMARL registration
def env_fn(**kwargs) -> CollaborativeCarryMARL:
    return CollaborativeCarryMARL(**kwargs)
