import numpy as np
import torch
import time

import argparse
from mappo.mappo_trainer import Config, MAPPOTrainer

def main():
    parser = argparse.ArgumentParser(description="Train MAPPO on CollaborativeCarryEnv")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of agents (default: 4)")
    parser.add_argument("--obstacles", type=int, default=4,
                        help="Number of obstacles (default: 4)")
    parser.add_argument("--grid", type=int, default=8,
                        help="Grid size (default: 8)")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total training timesteps (default: 10M)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name prefix for checkpoints (default: timestamped)")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="PPO running on the device")
    parser.add_argument("--n-envs", type=int, default=32,
                        help="How many parallel envs to run (depends heavily on your device)")
    args = parser.parse_args()
    
    config = Config()
    config.n_agents = args.agents
    config.n_obstacles = args.obstacles
    config.grid_size = args.grid
    config.total_timesteps = args.timesteps
    config.save_dir = args.save_dir
    config.run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    config.device = args.device
    config.n_envs = args.n_envs
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    trainer = MAPPOTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load(args.checkpoint)
        print(f"Resuming from step {trainer.global_step}")
    
    trainer.train()


if __name__ == "__main__":
    main()