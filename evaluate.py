"""Evaluate and visualize trained MAPPO agents."""
from __future__ import annotations

import sys
import time
from pathlib import Path
import argparse

import numpy as np
import torch
import pygame

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mappo.mappo_trainer import ActorCritic, Config
from environment.epymarl_wrapper import CollaborativeCarryMARL


def evaluate(
    checkpoint_path: str,
    n_episodes: int = 10,
    render: bool = True,
    step_delay: float = 2.0,
    grid_size: int = 8,
    n_agents: int = 4,
    n_obstacles: int = 4,
    max_agents: int = 10,
    max_objects: int | None = None,
    max_goals: int | None = None,
    max_obstacles: int = 6,
    max_grid_size: int = 10,
):
    """Run evaluation with visualization (multi-agent, variable grid/obstacles)."""
    config = Config()
    device = torch.device(config.device)
    
    env = CollaborativeCarryMARL(
        grid_size=grid_size,
        n_agents=n_agents,
        n_obstacles=n_obstacles,
        max_agents=max_agents,
        max_objects=max_objects,
        max_goals=max_goals,
        max_obstacles=max_obstacles,
        max_grid_size=max_grid_size,
        episode_limit=config.episode_limit,
        render_mode="human" if render else None,
    )
    
    env_info = env.get_env_info()
    
    # Create and load network
    network = ActorCritic(
        env_info["obs_shape"],
        env_info["state_shape"],
        env_info["n_actions"],
        config.hidden_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Trained for {checkpoint['global_step']} steps")
    print(f"Step delay: {step_delay}s (press Q to quit, SPACE to pause)")
    print("-" * 40)
    
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    
    total_rewards = []
    successes = 0
    paused = False
    
    for ep in range(n_episodes):
        obs, state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n=== Episode {ep + 1} ===")
        
        if render:
            env.render()
            time.sleep(step_delay)
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        env.close()
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
            
            if paused:
                time.sleep(0.1)
                continue
            
            actions = []
            with torch.no_grad():
                for a in range(env_info["n_agents"]):
                    obs_tensor = torch.FloatTensor(obs[a]).unsqueeze(0).to(device)
                    action, _, _ = network.get_action(obs_tensor, deterministic=True)
                    actions.append(action.item())

            reward, done, info = env.step(actions)
            episode_reward += reward
            step += 1
            
            # Print step info for all agents
            action_str = " | ".join([
                f"A{idx+1}:{ACTION_NAMES[act]:<5}" for idx, act in enumerate(actions)
            ])
            print(f"  Step {step:>3}: {action_str} | R={reward:>+6.2f}")
            
            if render:
                env.render()
                time.sleep(step_delay)
            
            obs = env.get_obs()
        
        total_rewards.append(episode_reward)
        
        # Check if succeeded (high reward means delivery)
        if episode_reward > 5:
            successes += 1
        
        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    env.close()
    
    print("-" * 40)
    print(f"Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO agents")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mappo_final.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between steps (default: 2.0 for slow-mo)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of agents")
    parser.add_argument("--obstacles", type=int, default=4,
                        help="Number of obstacles")
    parser.add_argument("--grid", type=int, default=8,
                        help="Grid size")
    parser.add_argument("--max-agents", type=int, default=10,
                        help="Max agents for padding")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Max objects for padding (default: derived)")
    parser.add_argument("--max-goals", type=int, default=None,
                        help="Max goals for padding (default: derived)")
    parser.add_argument("--max-obstacles", type=int, default=6,
                        help="Max obstacles for padding")
    parser.add_argument("--max-grid", type=int, default=10,
                        help="Max grid size for padding")
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        render=not args.no_render,
        step_delay=args.delay,
        grid_size=args.grid,
        n_agents=args.agents,
        n_obstacles=args.obstacles,
        max_agents=args.max_agents,
        max_objects=args.max_objects,
        max_goals=args.max_goals,
        max_obstacles=args.max_obstacles,
        max_grid_size=args.max_grid,
    )


if __name__ == "__main__":
    main()
