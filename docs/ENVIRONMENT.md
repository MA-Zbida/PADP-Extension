# Environment specification

A reference for `padp/env/`.

## Formal model

A Dec-POMDP `<N, S, A, P, O, R, gamma, H>`:

| Symbol | Meaning |
|---|---|
| N | 1 to 6 homogeneous agents with sampled capacities |
| S | Positions, capacities, weights, holders, lifted/delivered flags, used goals, time |
| A_i | Movement {stay, up, down, left, right} x interaction {none, pick, drop} |
| P | Deterministic given the joint action |
| O_i | 27 features |
| R | Shared team reward |
| gamma | 0.99 |
| H | `max_steps` (time limit = truncation) |

## World

| Item | Value |
|---|---|
| Arena | [0, 1] x [0, 1] |
| Agent radius | 0.025 |
| Object radius | 0.035 |
| Goal radius | 0.065 |
| Move step | 0.025 |
| Pickup radius | 0.04 |
| Grip slots | 4 (left, right, top, bottom) at offset 0.074 from the object center |

**Layouts.**
- `warehouse`: depot, shipping zone, up to 6 racks with aisles of at least 0.22, pick faces 0.11 from racks.
- `open`: no racks.

**Episode sampling.**
- Capacities and weights are drawn from the configured lists and redrawn until every object can be
  lifted by its four strongest possible holders.
- Objects are placed at pick faces or at `carry_distance` from their goal. A clear ring of 0.105
  keeps all slots reachable.
- Agents start in the depot or 0.10-0.18 from an object (`spawn: near_objects`).

## Step order

1. **Drops.**
   - All holders of a lifted object drop: delivered if the object is inside a free goal, otherwise put down.
   - Only some holders drop: drop disagreement.
   - An attached agent of an object that is not lifted detaches.
   - Dropping while holding nothing is invalid.
2. **Attaches.** `pick` attaches to the nearest free slot within reach of an object that is not lifted.
   - Slot conflicts go to the nearest agent.
   - `simultaneous` mode attaches a group only if it can lift in the same step.
3. **Lift check.** An object lifts when attached capacity >= weight and, for two or more holders, the
   holders include an opposite pair. Lifting takes the step.
4. **Movement.** A lifted coalition moves by the coupling rule:
   - `strict`: identical moves only;
   - `average`: mean displacement;
   - `leader`: the lowest-index holder steers.

   Attached agents of an object that is not lifted stay in place. Collisions (bounds, racks,
   objects, agents including swept paths) reject a whole group's move.
5. **Reward, termination** (all delivered) **or truncation** (time limit).
6. **Observations and anchors.**

## Reward

    sparse: r = 10 * deliveries - 0.01
    shaped: r = 10 * deliveries - 0.01 - 0.05 * collisions - 0.01 * invalid_actions + gamma * Phi(s') - Phi(s)

The potential Phi sums over undelivered objects:

| Object state | Term |
|---|---|
| Not lifted | 0.5 * min(1, attached capacity / weight) |
| Lifted | 1.0 + 3.0 * (1 - min(1, distance to nearest free goal)) |

It also subtracts `1.0 * distance` from each free agent to the nearest free slot (when one exists).
Delivered objects contribute 0, and Phi = 0 at termination.

## Observation (27 values, clipped to [-1, 1])

| Index | Feature |
|---|---|
| 0-1 | Own position (2x - 1) |
| 2 | Own capacity / 100 |
| 3 | Attached |
| 4 | Lifted |
| 5 | Coalition size / 4 |
| 6-7 | Vector to target object (held, else nearest needing help) |
| 8 | Distance to target |
| 9 | Target weight / 200 |
| 10 | Capacity still needed / 200 |
| 11 | Free slots / 4 |
| 12-13 | Vector to nearest free slot |
| 14-15 | Vector to nearest free goal (from the held object when carrying) |
| 16 | Distance to that goal |
| 17 | Can attach |
| 18 | Can drop in goal |
| 19 | Distance to nearest rack |
| 20-21 | Vector to nearest agent |
| 22 | Distance to nearest agent |
| 23 | Capacity of nearest agent / 100 |
| 24-25 | Vector to held object |
| 26 | t / max_steps |

## Global state (220 values, padded with -1)

| Block | Size | Content |
|---|---|---|
| Agent observations | 6 x 27 | Every agent's observation |
| Time | 1 | t / max_steps |
| Objects | 3 x 8 | Position, vector to nearest free goal, weight, attached capacity, lifted, delivered |
| Goals | 3 x 3 | Position, used |
| Racks | 6 x 4 | Center, width, height |

## Anchors

Per agent, `info["anchors"]` returns `[move_anchor, interaction_anchor]`:

- **move anchor:** id of the lifted object the agent carries;
- **interaction anchor:** the object the agent is attached to or whose free slot it stands at.

Either is -1 when there is none. Used only by the coupled sampler.

## Episode summary (`info["episode"]`)

| Field | Meaning |
|---|---|
| `success`, `completion` | All objects delivered; fraction delivered |
| `steps`, `return` | Episode length and total reward |
| `deliveries_k1`, `deliveries_k2`, `deliveries_k3plus` | Deliveries by coalition size |
| `carry_agreement` | Fraction of carrying steps with identical moves |
| `help_wait` | Steps from first attach to lift |
| `spare_capacity` | Attached capacity minus weight at lift |
| `coalition_size` | Holders at lift |
| `lifts`, `releases_outside_goal`, `drop_disagreements`, `detaches` | Interaction counts |
| `invalid_actions`, `collisions` | Rejected or impossible actions |
