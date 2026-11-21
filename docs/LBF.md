# Level-Based Foraging setup

The adapter in [padp/env/lbf.py](../padp/env/lbf.py) runs the official `lbforaging` package
(version 2.0.0, gymnasium) behind the trainer's environment interface. Select it with
`env.kind: lbf` and an `env.lbf` block (fields of `LBFConfig`).

## Task

| Item | Setting |
|---|---|
| Grid | `field_size` x `field_size` |
| Agents | `players`, levels drawn uniformly from 1-2 each episode |
| Food | `max_num_food` items; no two adjacent; placed away from the border |
| Food levels | package defaults: with `force_coop: true`, every food's level is the sum of the three lowest agent levels (with 2 or 3 agents: all agents must load in the same step); with `force_coop: false`, levels are drawn between 1 and that sum |
| Collection | a food is collected when the agents choosing LOAD next to it in the same step have levels summing to at least its level |
| Episode | ends when all food is collected or after `max_episode_steps` (50; the time limit is treated as truncation) |
| Actions | 6: none, north, south, west, east, load (invalid actions become none, as in the package) |

## Observation, state and reward

- **Observation** (package vector observation, full observability, sight = grid size): for each food
  (row, col, level), absent foods (-1, -1, 0); then for each agent (row, col, level), the observing
  agent first. With full observability the coordinates are absolute. The network receives these
  values divided by fixed scales (coordinates by `field_size - 1`, food levels by `3 x max_player_level`,
  agent levels by `max_player_level`).
- **Global state** for the centralized critic: all agents' scaled observations, concatenated.
- **Reward**: team reward = sum of the package's normalized per-agent rewards (`normalize_reward=True`,
  no penalty). An episode's total is 1 when all food is collected.

## Anchors for coupled sampling

Computed by `anchor_from_obs` from each agent's **own raw observation only**: the grid cell
(row x `field_size` + col) of the adjacent food (Manhattan distance 1) with the highest level, ties
broken by row then column; -1 if no food is adjacent. Agents next to the same food therefore compute
the same anchor without any simulator state; `tests/test_lbf.py` checks this on package
observations. Partial observability is rejected, because the package then reports coordinates relative
to each agent.

The trainer's policy has two heads; for LBF the second has a single action (log-probability 0,
entropy 0), so the policy is the package's 6-way categorical and coupling applies to it.

## Metrics

The run summaries reuse the transport metric names:

| Column | Meaning in LBF |
|---|---|
| Success, completion, steps, return | all food collected; fraction of food collected; episode length; team return |
| Carry agreement | **load agreement**: of the load attempts where the adjacent agents' levels sufficed and at least one of them chose LOAD, the fraction that collected the food |
| Ignored drops | miscoordinated load attempts (the complement of the above, as a count) |
| Coalition size, deliveries k=1 / 2 / 3+ | agents loading per collected food |
| Help wait, spare capacity, collisions, releases outside goal | not defined (empty) |
