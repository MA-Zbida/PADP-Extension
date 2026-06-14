# Continuous Warehouse MARL for Cooperative PADP

This project studies a fully Multi-Agent Reinforcement Learning (MARL) version of the Pick and Delivery Problem (PADP). Agents operate in a continuous 2D warehouse, where each object must be picked by exactly two agents from opposite grip zones, carried as a rigid pair, and dropped into a shipping/staging goal.

The implementation uses MAPPO with centralized training and decentralized execution.

## Current Environment

- Continuous normalized warehouse plane: `[0, 1] x [0, 1]`.
- Depot/charging zone, shipping zone, rack obstacles, and pick-face object spawns.
- Physical agent-agent collisions: agents cannot overlap or pass through each other.
- Explicit factorized actions per agent:
  - Movement: `stay`, `up`, `down`, `left`, `right`.
  - Interaction: `none`, `pick`, `drop`.
- Opposite-side pickup:
  - Valid pairs are `left+right` or `top+bottom`.
  - Third agents cannot attach to an already held object.
- Rigid two-agent carrying:
  - Both holders must choose the same movement action.
  - If the pair collides with racks, bounds, objects, or another agent, the full group movement is rejected.
- Delivery:
  - Both holders must select `drop` while the object is inside an unused goal radius.

## Project Structure

```text
environment/
  actions.py          # Factorized action constants and flattening helpers
  geometry.py         # Continuous geometry, collision, grip helpers
  layout.py           # Warehouse depot/rack/pick-face/shipping layout
  rewards.py          # Reward and metric bookkeeping
  renderer.py         # Smooth Pygame warehouse renderer
  env.py              # CollaborativeCarryEnv public environment class
  epymarl_wrapper.py  # EPyMARL-compatible wrapper

mappo/
  actor_critic.py     # Two-head MAPPO actor and centralized critic
  buffer.py           # Rollout buffer for factorized actions
  mappo_trainer.py    # Custom MAPPO training loop

train.py              # Training entry point
evaluate.py           # Checkpoint evaluation and visualization
visualize_env.py      # Standalone environment rendering preview
Project_State.md      # Research-facing project state and formulation
```

## Training

```bash
python train.py --agents 4 --obstacles 4 --grid 8 --timesteps 10000000 --device cpu --n-envs 16
```

Useful arguments:

- `--agents`: number of warehouse agents.
- `--obstacles`: number of rack obstacles.
- `--timesteps`: total environment steps.
- `--device`: `cpu` or `cuda`.
- `--n-envs`: number of parallel rollout environments.
- `--checkpoint`: resume from a checkpoint.

Existing grid-world checkpoints are legacy and are not expected to load into the new continuous/factorized architecture.

If you stop training with `Ctrl+C`, the trainer now saves an interrupt checkpoint before exiting:

```bash
checkpoints/<run-name>_interrupt_<step>.pt
```

Resume with:

```bash
python train.py --checkpoint checkpoints/<run-name>_interrupt_<step>.pt --agents 4 --obstacles 4 --grid 8 --timesteps 10000000 --device cpu --n-envs 16
```

By default, training now uses an obstacle curriculum. If `--obstacles 4` is requested, the run starts with easier layouts and increases the rack count every curriculum stage until it reaches 4. This is intentional: the continuous pickup task is sparse, and the curriculum helps agents discover grip formation before learning obstacle avoidance in dense layouts.

Useful debugging signal in the training log:

- `CanPick` should rise first: agents are reaching valid grip zones.
- `OppGrip` should rise next: two agents are occupying opposite grips.
- `Pickups` should become nonzero before `PairSteps`.
- `PairSteps` should become nonzero once agents learn to hold and move objects together.

To disable the curriculum:

```bash
python train.py --agents 4 --obstacles 4 --timesteps 10000000 --no-curriculum
```

## Visualizing the Environment First

To inspect the renderer without training or loading a checkpoint:

```bash
python visualize_env.py --mode scripted --agents 4 --obstacles 4 --debug --delay 0.15
```

The scripted preview places two agents on opposite grip zones, shows a valid pickup, flashes a blocked carrier move, moves the blocker aside, carries the crate to shipping, and drops it on the goal pad.

For a noisy random-action preview of the current reset distribution:

```bash
python visualize_env.py --mode random --agents 4 --obstacles 4 --debug --delay 0.1
```

Close either window with `Q`, `Esc`, or the window close button.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/<checkpoint>.pt --episodes 5 --delay 0.2
```

Use `--no-render` for headless evaluation.

The evaluator reports:

- Mean reward.
- Success rate.
- Completion ratio.
- Steps to success.
- Collision counts.
- Idle actions.
- Carrier-pair coordination steps.

## Research Metrics

The environment returns rich metrics through `info["metrics"]`, including:

- Valid/invalid pickups and drops.
- Agent-agent, rack, bound, and object collisions.
- Carrier-pair steps and carrier movement steps.
- Overstaffing and blocking.
- Completion ratio and success.
- Remaining object-goal distance.

These metrics are intended to support research reporting beyond episode return and success rate.
