# Project State: Continuous MARL Warehouse PADP

## Description

This project extends prior Pick and Delivery Problem (PADP) research from a hybrid CBS + RL approach into a fully Multi-Agent Reinforcement Learning (MARL) formulation. The task now requires physical cooperation: each object must be picked from opposite grip zones by exactly two agents, carried as a rigid pair, and dropped in a shipping/staging goal.

The current implementation is no longer a grid-world renderer. It is a continuous 2D warehouse abstraction with normalized `(x, y)` positions, depot and shipping zones, rack obstacles, pick-face object placement, explicit `pick/drop` interactions, physical agent-agent collisions, and smooth Pygame visualization. The public environment class remains `CollaborativeCarryEnv`, but the internals are split into action, geometry, layout, reward/metrics, rendering, and environment modules.

The central research question is:

Can a fully learned MARL policy solve a warehouse-style cooperative PADP where objects require synchronized two-agent opposite-side gripping, while remaining efficient, safe, and scalable under partial observability?

## Environment Setup

The environment is a cooperative partially observable stochastic game/POMDP:

`M = <N, S, {A_i}, P, {O_i}, O, R, gamma, H>`

- `N`: homogeneous warehouse agents.
- `S`: global continuous state with agent positions, object positions, goal positions, rack rectangles, object holders, grip sides, delivered flags, used-goal flags, and time step.
- `A_i`: factorized discrete action for each agent: movement and interaction.
- `P`: continuous transition dynamics with collision resolution and rigid two-agent carrying.
- `O_i`: compact local observation for agent `i`.
- `R`: shared team reward.
- `H`: finite episode horizon.

### Geometry

- Warehouse bounds: normalized `[0, 1] x [0, 1]`.
- Agent radius: `0.025`.
- Object radius: `0.035`.
- Goal radius: `0.065`.
- Movement step: `0.025`.
- Depot zone: lower-left.
- Shipping zone: upper-right.
- Racks: rectangular continuous obstacles.
- Pick faces: object spawn candidates around racks.

Agents cannot overlap with other agents, racks, bounds, or objects they are not holding. Independent agents that collide or try to swap through each other both stay. If a carried group collides, the whole group movement is rejected.

### Actions

Each agent chooses:

- Movement: `stay`, `up`, `down`, `left`, `right`.
- Interaction: `none`, `pick`, `drop`.

The custom MAPPO trainer uses this factorized action directly. The EPyMARL wrapper also exposes a flattened categorical action space of size `15` for compatibility.

### Pickup, Carry, and Drop

Pickup succeeds only when:

- The agent selects `pick`.
- The agent is close to a grip zone of an undelivered object.
- Exactly two agents pick the same object in the same step.
- Their grip zones are opposite: `left+right` or `top+bottom`.

Carry dynamics:

- A held object has exactly two holders.
- Holders and object move as a rigid group.
- Group movement succeeds only when both holders choose the same movement action.
- Third agents cannot attach to an already held object.

Drop dynamics:

- Delivery succeeds when both holders select `drop` while the object is inside an unused goal radius.
- Dropping outside a goal releases the object but does not deliver it.
- A single holder trying to drop is counted as an invalid drop.

### Observations

Each active agent receives a 24-dimensional local observation containing:

- Own continuous position.
- Vector and distance to nearest object needing help.
- Vector and distance to nearest unused goal.
- Nearest rack distance.
- Holding/object-partner state.
- Whether it can currently pick or drop.
- Partner vector when carrying.
- Nearest-agent vector and distance.
- Local agent density.
- Whether work remains.
- Recent overstaffing/blocking indicators.

The centralized critic receives the concatenated per-agent observations plus shared state features:

- Time, number of agents, number of objects.
- Padded rack rectangle descriptors.
- Padded object descriptors.
- Padded goal descriptors.

## Reward Function

The reward remains shared and delivery-focused, with small interpretable terms for interaction and safety:

`R_t = R_step + R_delivery + R_pick/drop + R_collision + R_coordination + R_progress + R_potential`

Current reward coefficients are defined in `environment/rewards.py`:

- Delivery: `+100.0` per delivered object.
- Step cost: `-0.08`.
- Valid pickup: `+5.0`.
- Invalid pickup: `-0.05`.
- Valid drop: `+2.0`.
- Invalid drop: `-0.05`.
- Collision: `-0.75`.
- Overstaffing: `-0.2`.
- Blocking: `-0.5`.
- Can-pick state: `+0.1` per free agent at a valid grip zone.
- Opposite-grip formation: `+1.0` per object with free agents at opposite grip zones.
- Grip progress: `+8.0 * grip-distance improvement`.
- Carry progress: `+40.0 * object-goal distance improvement`.
- Potential shaping: `+1.0 * (0.99 * Phi(s') - Phi(s))`.

Useful post-delivery help is not penalized. The environment penalizes only noisy help: overstaffing a held object, blocking carrier pairs, invalid pick/drop attempts, and physical collision attempts.

## Metrics

The environment reports rich metrics through `info["metrics"]`:

- Success and completion ratio.
- Steps and delivered count.
- Valid/invalid pickups.
- Valid/invalid drops.
- Agent-agent collisions.
- Rack, bound, and object collisions.
- Overstaffing and blocking.
- Idle actions.
- Carrier-pair steps.
- Carrier-move steps.
- Can-pick agents and opposite-grip pairs.
- Remaining object-goal distance.
- Mean agent-object distance.

The trainer also logs completion, safety events, can-pick agents, opposite-grip pairs, valid pickups, and carrier-pair coordination steps in addition to reward and success. Obstacle curriculum is enabled by default so agents learn grip formation before dense rack avoidance.

## The Solution

The solution is MAPPO with centralized training and decentralized execution.

- The actor is shared across agents.
- The actor now has two heads: movement and interaction.
- The critic receives centralized global state.
- PPO log probability is the sum of movement and interaction log probabilities.
- Entropy is the sum of both action-head entropies.
- The rollout buffer stores factorized actions per agent.

This keeps the learned policy interpretable: one head learns where to move, and the other learns when to pick or drop.

## Current Strengths

- The task now has physical cooperation instead of visual-only overlap.
- Pickup is interpretable because two agents must grip opposite sides.
- Agent-agent collisions prevent unrealistic pass-through motion.
- Pygame visualization now shows carrier links, held crates, depot, racks, and staging pads.
- The environment is modular and easier to debug.
- Metrics are suitable for research reporting beyond reward and success.

## Current Limitations and Next Steps

- Existing checkpoints are legacy and incompatible with the new architecture.
- Communication is still implicit through observations and shared reward; explicit learned communication remains future work.
- A formal baseline suite is still needed: random, greedy heuristic, IPPO, MAPPO without shaping, and MAPPO without collision/grip constraints.
- Training should be rerun with multiple seeds and reported with confidence intervals.
- A scripted visual demo should be recorded for paper/presentation material.
