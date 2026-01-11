from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environment.epymarl_wrapper import CollaborativeCarryMARL
from mappo.actor_critic import ActorCritic
from mappo.buffer import RolloutBuffer

@dataclass
class Config:
    # Environment
    n_envs: int = 16
    episode_limit: int = 100
    grid_size: int = 8
    n_agents: int = 4
    n_obstacles: int = 4
    max_agents: int = 10
    max_objects: Optional[int] = None
    max_goals: Optional[int] = None
    max_obstacles: int = 6  # Fixed obs space size for consistent network dims
    max_grid_size: int = 10

    # Run naming
    run_name: Optional[str] = None
    
    # Training (stabilized for reduced variance rewards)
    total_timesteps: int = 10_000_000
    lr: float = 2.5e-4            # Reduced LR to stabilize with high loss
    gamma: float = 0.99           # Standard discount
    gae_lambda: float = 0.95      # GAE parameter
    
    # PPO (tuned for stability)
    clip_param: float = 0.2
    entropy_coef: float = 0.01    # Lower entropy for more exploitation
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 5           # Fewer epochs to prevent overfitting per batch
    num_minibatches: int = 4      # Larger minibatches for stable gradients
    
    # Network
    hidden_dim: int = 256         # Larger network for complex role understanding
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50000
    eval_interval: int = 20000
    eval_episodes: int = 5
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "checkpoints"



class MAPPOTrainer:
    """Multi-Agent PPO trainer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create environments
        self.envs = [
            CollaborativeCarryMARL(
                grid_size=config.grid_size,
                n_agents=config.n_agents,
                n_obstacles=config.n_obstacles,
                max_agents=config.max_agents,
                max_objects=config.max_objects,
                max_goals=config.max_goals,
                max_obstacles=config.max_obstacles,
                max_grid_size=config.max_grid_size,
                episode_limit=config.episode_limit,
            ) for _ in range(config.n_envs)
        ]
        
        # Set obstacle count
        for env in self.envs:
            env.set_n_obstacles(config.n_obstacles)
        print(f"Training with {config.n_obstacles} obstacle(s)")
        
        # Get dimensions
        env_info = self.envs[0].get_env_info()
        self.obs_dim = env_info["obs_shape"]
        self.state_dim = env_info["state_shape"]
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        
        # Create shared network (parameter sharing across agents)
        self.network = ActorCritic(
            self.obs_dim, self.state_dim, self.n_actions, config.hidden_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)  # Track success (all objects delivered)
        self.global_step = 0
        self.reward_history: List[Tuple[int, float]] = []  # (global_step, mean_reward)
        self.success_history: List[Tuple[int, float]] = []  # (global_step, success_rate)
        self.moving_avg_history: List[Tuple[int, float]] = []  # (global_step, moving_avg_reward)

        # Checkpoint naming
        self.checkpoint_prefix = self.config.run_name or "mappo"
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def collect_rollouts(self, n_steps: int) -> Tuple[RolloutBuffer, List[float]]:
        """Collect rollout data from all environments."""
        buffer = RolloutBuffer(
            self.config.n_envs, self.n_agents, n_steps, self.obs_dim, self.state_dim
        )
        
        episode_rewards_list = []
        
        # Initialize if needed
        if not hasattr(self, 'current_obs'):
            self.current_obs = []
            self.current_states = []
            self.ep_rewards = [0.0] * self.config.n_envs
            self.ep_lengths = [0] * self.config.n_envs
            self.ep_successes = [False] * self.config.n_envs  # Track if episode succeeded
            
            for env in self.envs:
                obs, state = env.reset()
                self.current_obs.append(obs)
                self.current_states.append(state)
        
        for _ in range(n_steps):
            obs_batch = np.array(self.current_obs)  # (n_envs, n_agents, obs_dim)
            states_batch = np.array(self.current_states)  # (n_envs, state_dim)
            
            # Get actions for all agents in all envs
            actions_batch = []
            log_probs_batch = []
            
            with torch.no_grad():
                for e in range(self.config.n_envs):
                    actions_e = []
                    log_probs_e = []
                    for a in range(self.n_agents):
                        obs_tensor = torch.FloatTensor(obs_batch[e, a]).unsqueeze(0).to(self.device)
                        action, log_prob, _ = self.network.get_action(obs_tensor)
                        actions_e.append(action.item())
                        log_probs_e.append(log_prob.item())
                    actions_batch.append(actions_e)
                    log_probs_batch.append(log_probs_e)
                
                # Get values
                states_tensor = torch.FloatTensor(states_batch).to(self.device)
                values = self.network.get_value(states_tensor).cpu().numpy()
            
            # Step environments
            rewards = []
            dones = []
            next_obs = []
            next_states = []
            
            for e, env in enumerate(self.envs):
                reward, done, info = env.step(actions_batch[e])
                rewards.append(reward)
                dones.append(done)
                
                self.ep_rewards[e] += reward
                self.ep_lengths[e] += 1
                
                if done:
                    episode_rewards_list.append(self.ep_rewards[e])
                    self.episode_rewards.append(self.ep_rewards[e])
                    self.episode_lengths.append(self.ep_lengths[e])
                    
                    # Track success: all objects delivered (check info dict)
                    success = all(info.get("delivered", [])) if info.get("delivered") else False
                    self.episode_successes.append(success)
                    
                    self.ep_rewards[e] = 0.0
                    self.ep_lengths[e] = 0
                    
                    obs, state = env.reset()
                    next_obs.append(obs)
                    next_states.append(state)
                else:
                    next_obs.append(env.get_obs())
                    next_states.append(env.get_state())
            
            buffer.add(
                obs_batch.copy(),
                states_batch.copy(),
                actions_batch,
                log_probs_batch,
                rewards,
                dones,
                values,
            )
            
            self.current_obs = next_obs
            self.current_states = next_states
            self.global_step += self.config.n_envs
        
        # Get last values for GAE
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(self.current_states)).to(self.device)
            last_values = self.network.get_value(states_tensor).cpu().numpy()
        
        returns, advantages = buffer.compute_returns(
            last_values, self.config.gamma, self.config.gae_lambda
        )
        
        return buffer, returns, advantages, episode_rewards_list
    
    def train_step(self, buffer: RolloutBuffer, returns: np.ndarray, advantages: np.ndarray):
        """Perform PPO update."""
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.config.ppo_epochs):
            for batch in buffer.get_batches(returns, advantages, self.config.num_minibatches):
                obs, states, actions, old_log_probs, batch_returns, batch_advantages = batch
                
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                states_tensor = torch.FloatTensor(states).to(self.device)
                actions_tensor = torch.LongTensor(actions).to(self.device)
                old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
                returns_tensor = torch.FloatTensor(batch_returns).to(self.device)
                advantages_tensor = torch.FloatTensor(batch_advantages).to(self.device)
                
                # Get new log probs and entropy
                new_log_probs, entropy = self.network.evaluate_actions(obs_tensor, actions_tensor)
                
                # Get new values and obstacle distances
                values = self.network.get_value(states_tensor)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss and obstacle distance loss
                value_loss = 0.5 * ((values - returns_tensor) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.value_coef * value_loss 
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def evaluate(self, n_episodes: int = 5, render: bool = False) -> float:
        """Evaluate the current policy."""
        eval_env = CollaborativeCarryMARL(
            grid_size=self.config.grid_size,
            n_agents=self.config.n_agents,
            n_obstacles=self.config.n_obstacles,
            max_agents=self.config.max_agents,
            max_objects=self.config.max_objects,
            max_goals=self.config.max_goals,
            max_obstacles=self.config.max_obstacles,
            max_grid_size=self.config.max_grid_size,
            episode_limit=self.config.episode_limit,
            render_mode="human" if render else None,
        )
        
        total_rewards = []
        
        for _ in range(n_episodes):
            obs, state = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                actions = []
                with torch.no_grad():
                    for a in range(self.n_agents):
                        obs_tensor = torch.FloatTensor(obs[a]).unsqueeze(0).to(self.device)
                        action, _, _ = self.network.get_action(obs_tensor, deterministic=True)
                        actions.append(action.item())
                
                reward, done, info = eval_env.step(actions)
                episode_reward += reward
                
                if render:
                    eval_env.render()
                
                obs = eval_env.get_obs()
            
            total_rewards.append(episode_reward)
        
        eval_env.close()
        return np.mean(total_rewards)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        print(f"Saved checkpoint to {path}")

    def save_reward_plot(self, path: str):
        """Save reward vs global_step plot to the given path."""
        if len(self.reward_history) < 2:
            return  # Not enough points to plot
        steps, rewards = zip(*self.reward_history)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, rewards, label="Mean Reward (last 100 eps)", alpha=0.6)
        
        # Add moving average line if available
        if len(self.moving_avg_history) >= 2:
            ma_steps, ma_rewards = zip(*self.moving_avg_history)
            plt.plot(ma_steps, ma_rewards, label="Moving Avg Reward (EMA)", linewidth=2)
        
        plt.xlabel("Global Step")
        plt.ylabel("Reward")
        plt.title("MAPPO Training Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    def save_metrics_plot(self, path: str):
        """Save comprehensive metrics plot with reward and success rate."""
        if len(self.reward_history) < 2:
            return  # Not enough points to plot
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Rewards
        steps, rewards = zip(*self.reward_history)
        axes[0].plot(steps, rewards, label="Mean Reward (last 100 eps)", alpha=0.6, color='blue')
        if len(self.moving_avg_history) >= 2:
            ma_steps, ma_rewards = zip(*self.moving_avg_history)
            axes[0].plot(ma_steps, ma_rewards, label="Moving Avg (EMA)", linewidth=2, color='darkblue')
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Performance")
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Success Rate
        if len(self.success_history) >= 2:
            s_steps, s_rates = zip(*self.success_history)
            axes[1].plot(s_steps, s_rates, label="Success Rate (%)", color='green', linewidth=2)
            axes[1].fill_between(s_steps, 0, s_rates, alpha=0.2, color='green')
        axes[1].set_xlabel("Global Step")
        axes[1].set_ylabel("Success Rate (%)")
        axes[1].set_ylim(0, 105)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        print(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting MAPPO training on {self.config.device}")
        print(f"Obs dim: {self.obs_dim}, State dim: {self.state_dim}, Actions: {self.n_actions}")
        print(f"Obstacles: {self.config.n_obstacles}, Timesteps: {self.config.total_timesteps:,}")
        print("-" * 60)
        
        n_steps_per_update = 128
        n_updates = 0
        
        while self.global_step < self.config.total_timesteps:
            # Collect rollouts
            buffer, returns, advantages, ep_rewards = self.collect_rollouts(n_steps_per_update)
            
            # Train
            train_info = self.train_step(buffer, returns, advantages)
            n_updates += 1
            
            # Logging
            if n_updates % self.config.log_interval == 0:
                max_reward = max(self.episode_rewards) if self.episode_rewards else 0
                min_reward = min(self.episode_rewards) if self.episode_rewards else 0
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                # Compute moving average reward (exponential moving average)
                if not hasattr(self, '_ema_reward'):
                    self._ema_reward = mean_reward
                else:
                    alpha = 0.05  # Smoothing factor for EMA
                    self._ema_reward = alpha * mean_reward + (1 - alpha) * self._ema_reward
                moving_avg_reward = self._ema_reward
                
                # Compute success rate (percentage of episodes where all objects delivered)
                success_rate = np.mean(self.episode_successes) * 100 if self.episode_successes else 0
                
                # Track and plot rewards
                self.reward_history.append((self.global_step, mean_reward))
                self.success_history.append((self.global_step, success_rate))
                self.moving_avg_history.append((self.global_step, moving_avg_reward))
                
                # Save plots
                self.save_reward_plot(os.path.join(self.config.save_dir, "reward.png"))
                self.save_metrics_plot(os.path.join(self.config.save_dir, "metrics.png"))

                print(f"Step {self.global_step:>8} | "
                      f"Reward: {mean_reward:>7.2f} | "
                      f"MovAvg: {moving_avg_reward:>7.2f} | "
                      f"Success: {success_rate:>5.1f}% | "
                      f"Length: {mean_length:>5.1f} | "
                      f"Loss: {train_info['loss']:.4f} | "
                      f"Entropy: {train_info['entropy']:.4f}")
            
            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                eval_reward = self.evaluate(self.config.eval_episodes)
                print(f">>> Evaluation: {eval_reward:.2f}")
            
            # Save
            if self.global_step % self.config.save_interval == 0:
                ckpt_path = os.path.join(self.config.save_dir, f"{self.checkpoint_prefix}_{self.global_step}.pt")
                self.save(ckpt_path)
        
        # Final save
        final_path = os.path.join(self.config.save_dir, f"{self.checkpoint_prefix}_final.pt")
        self.save(final_path)
        # Final plots
        self.save_reward_plot(os.path.join(self.config.save_dir, "reward.png"))
        self.save_metrics_plot(os.path.join(self.config.save_dir, "metrics.png"))
        
        # Print final summary
        final_success_rate = np.mean(self.episode_successes) * 100 if self.episode_successes else 0
        final_mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Final Success Rate: {final_success_rate:.1f}%")
        print(f"Final Mean Reward: {final_mean_reward:.2f}")
        print(f"Final Moving Avg Reward: {self._ema_reward:.2f}" if hasattr(self, '_ema_reward') else "")
        print(f"{'='*60}")