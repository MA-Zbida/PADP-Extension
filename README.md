# miniRL - Multi-Agent Cooperative Learning

A reinforcement learning project implementing Multi-Agent Proximal Policy Optimization (MAPPO) for a collaborative object-carrying task in a grid-based environment.

## Overview

This project trains multiple agents to collaboratively pick up and deliver objects to goal locations in a grid world. The agents must work together to carry objects that require multiple agents. The implementation uses MAPPO, a state-of-the-art multi-agent reinforcement learning algorithm.

## Project Structure

```
miniRL/
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation and visualization script
├── config/                     # Configuration files
│   ├── algs/                   # Algorithm configurations
│   │   └── mappo_collab.yaml   # MAPPO hyperparameters
│   └── envs/                   # Environment configurations
│       └── collaborative_carry.yaml
├── environment/                # Custom environment implementation
│   ├── env.py                  # CollaborativeCarryEnv class
│   ├── epymarl_wrapper.py      # Multi-agent wrapper
│   └── __init__.py
├── mappo/                      # MAPPO trainer and network implementation
│   ├── mappo_trainer.py        # MAPPOTrainer class
│   ├── actor_critic.py         # Actor-Critic network
│   └── buffer.py               # Rollout buffer
├── epymarl/                    # Extended PyMARL framework
└── checkpoints/                # Saved model checkpoints
```

## Environment Description

The Collaborative Carry environment is a multi-agent cooperative task where:

- Agents spawn in a grid world with obstacles
- Objects are placed randomly and need to be carried to goal locations
- Objects require multiple agents (typically 2) to be carried
- Agents receive rewards for delivering objects to goals
- Observations include agent positions, nearby objects, obstacles, and state information

### Key Parameters

- `grid_size`: Size of the grid (default: 8)
- `n_agents`: Number of agents (default: 4)
- `n_obstacles`: Number of obstacles (default: 4)
- `episode_limit`: Maximum steps per episode (default: 100)

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pygame (for visualization)

### Setup

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install torch numpy pygame
   ```

## Usage

### Training

To train a MAPPO agent on the collaborative carry task:

```bash
python train.py
```

#### Training Arguments

- `--agents`: Number of agents (default: 4)
- `--obstacles`: Number of obstacles (default: 4)
- `--grid`: Grid size (default: 8)
- `--timesteps`: Total training timesteps (default: 10,000,000)
- `--device`: Device to use - 'cpu' or 'cuda' (default: 'cpu')
- `--n-envs`: Number of parallel environments (default: 32)
- `--save-dir`: Directory to save checkpoints (default: 'checkpoints')
- `--run-name`: Custom run name for checkpoints
- `--checkpoint`: Path to checkpoint to resume training from

#### Example Commands

Train with custom configuration:
```bash
python train.py --agents 4 --obstacles 4 --grid 8 --timesteps 5000000 --device cuda --n-envs 16
```

Resume from checkpoint:
```bash
python train.py --checkpoint checkpoints/run_20240115_120000.pt --timesteps 15000000
```

### Evaluation

To evaluate a trained model with visualization:

```bash
python evaluate.py
```

#### Evaluation Arguments

- `--checkpoint`: Path to the checkpoint file (positional argument)
- `--episodes`: Number of episodes to evaluate (default: 10)
- `--delay`: Delay between steps in seconds (default: 2.0)
- `--grid-size`: Grid size for evaluation (default: 8)
- `--n-agents`: Number of agents (default: 4)
- `--n-obstacles`: Number of obstacles (default: 4)

#### Example Commands

Evaluate with visualization:
```bash
python evaluate.py checkpoints/t4_o4_g8_final.pt --render --n-episodes 5 --step-delay 1.0
```

Evaluate without visualization:
```bash
python evaluate.py checkpoints/t4_o4_g8_final.pt --n-episodes 20
```

## Configuration Files

### Algorithm Configuration (mappo_collab.yaml)

Contains MAPPO hyperparameters:
- Learning rate: 0.0005
- PPO clipping: 0.2
- Entropy coefficient: 0.01
- GAE lambda: 0.95
- Network hidden dimension: 256

### Environment Configuration (collaborative_carry.yaml)

Specifies environment parameters like grid size, number of obstacles, and render mode.

## Pre-trained Checkpoints

The `checkpoints/` directory contains pre-trained models:

- `t4_o4_g8_final.pt`: Trained on 4 agents, 4 obstacles, 8x8 grid
- `t6_o6_g12_final.pt`: Trained on 6 agents, 6 obstacles, 12x12 grid

## Performance Monitoring

During training, the system logs:
- Episode rewards
- Average episode length
- Policy loss
- Value loss
- Training progress

Logs are saved in the save directory and can be used to monitor training progress.

## Advanced Usage

### Modifying Hyperparameters

Edit `mappo/mappo_trainer.py` Config class to adjust:
- Network architecture (hidden_dim)
- Learning rate (lr)
- PPO parameters (clip_param, ppo_epochs)
- Training duration (total_timesteps)

### Custom Environments

To use the framework with other environments, modify the environment wrapper in `environment/epymarl_wrapper.py`.

## Project Notes

This is a PADP (Pick and Delivery Problem) extension where multiple agents must cooperatively pick up and deliver objects within a grid environment. The implementation uses modern deep reinforcement learning techniques to enable agents to learn complex collaborative behaviors.

## References

- MAPPO: The Multi-Agent PPO algorithm
- PyTorch: Deep learning framework
- EPyMARL: Extended PyMARL multi-agent framework
