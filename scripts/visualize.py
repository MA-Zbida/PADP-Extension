"""Watch the environment rules in action, without training.

Scenarios:
    k1       one agent lifts a 40 kg object alone and delivers it
    k2       one agent attaches and waits; a second joins on the opposite side; they lift and deliver
    k3       a 140 kg object needs three agents: two attach, the third completes the lift
    blocked  a carried object is blocked by another agent until it steps aside
    random   random actions in the full warehouse (mixed capacities and weights)

Examples:
    python scripts/visualize.py --scenario k3
    python scripts/visualize.py --scenario random --delay 0.05

Reading the picture: numbers on objects are weights (kg), numbers under agents are capacities.
The bar above an object fills as capacity attaches (orange) and turns green when it can be lifted.
Grey links = attached and waiting; blue links = lifted and carrying. A red frame flashes on collisions.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Iterator, List, Tuple

import numpy as np

from padp.env.actions import (
    INTERACT_DROP, INTERACT_NONE, INTERACT_PICK, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY, MOVE_UP,
)
from padp.env.config import EnvConfig
from padp.env.geometry import grip_points
from padp.env.warehouse import AGENT_RADIUS, MOVE_STEP, OBJECT_RADIUS, WarehouseEnv

OBJECT = np.array([0.45, 0.55], dtype=np.float32)
GOAL = np.array([0.80, 0.175], dtype=np.float32)
PARKING = [(0.08, 0.92), (0.16, 0.92), (0.24, 0.92), (0.08, 0.84)]
Step = Tuple[str, Dict[int, Tuple[int, int]], int]  # label, per-agent (move, interact), repeats


def slot(side: str) -> np.ndarray:
    return grip_points(OBJECT, OBJECT_RADIUS, AGENT_RADIUS)[side].copy()


def scene(weight: float, holders: Dict[int, str], n_agents: int = 3) -> WarehouseEnv:
    env = WarehouseEnv(EnvConfig(n_agents=n_agents, n_objects=1, n_racks=0, layout="open", spawn="near_objects",
                                 capacities=[50.0], weights=[weight], max_steps=400))
    env.reset(seed=0)
    env.object_positions = [OBJECT.copy()]
    env.goal_positions = [GOAL.copy()]
    for i in range(n_agents):
        env.agent_positions[i] = slot(holders[i]) if i in holders else np.array(PARKING[i], dtype=np.float32)
    env.refresh_derived_state()
    return env


def carry(holders: List[int]) -> Iterator[Step]:
    dx, dy = GOAL - OBJECT
    horizontal = MOVE_RIGHT if dx > 0 else MOVE_LEFT
    vertical = MOVE_DOWN if dy > 0 else MOVE_UP
    yield "carry toward shipping", {h: (horizontal, INTERACT_NONE) for h in holders}, int(round(abs(dx) / MOVE_STEP))
    yield "align with the goal", {h: (vertical, INTERACT_NONE) for h in holders}, int(round(abs(dy) / MOVE_STEP))
    yield "joint drop: delivered", {h: (MOVE_STAY, INTERACT_DROP) for h in holders}, 1


def script(name: str) -> Tuple[WarehouseEnv, Iterator[Step]]:
    pick = (MOVE_STAY, INTERACT_PICK)
    if name == "k1":
        def steps():
            yield "attach: 50 kg covers 40 kg, lift", {0: pick}, 1
            yield from carry([0])
        return scene(40.0, {0: "left"}), steps()
    if name == "k2":
        def steps():
            yield "agent 1 attaches", {0: pick}, 1
            yield "agent 1 waits for help", {}, 3
            yield "agent 2 joins opposite: lift", {1: pick}, 1
            yield from carry([0, 1])
        return scene(90.0, {0: "left", 1: "right"}), steps()
    if name == "k3":
        def steps():
            yield "two attach: 100 kg < 140 kg", {0: pick, 1: pick}, 1
            yield "waiting for a third agent", {}, 3
            yield "third joins: lift", {2: pick}, 1
            yield from carry([0, 1, 2])
        return scene(140.0, {0: "left", 1: "right", 2: "top"}), steps()
    if name == "blocked":
        env = scene(90.0, {0: "left", 1: "right"})
        env.agent_positions[2] = slot("right") + np.array([0.06, 0.0], dtype=np.float32)
        env.refresh_derived_state()

        def steps():
            yield "lift", {0: pick, 1: pick}, 1
            yield "blocked: move rejected", {0: (MOVE_RIGHT, INTERACT_NONE), 1: (MOVE_RIGHT, INTERACT_NONE)}, 2
            yield "blocker steps aside", {2: (MOVE_DOWN, INTERACT_NONE)}, 4
            yield from carry([0, 1])
        return env, steps()
    raise ValueError(name)


def wait(env: WarehouseEnv, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        env.render()
        time.sleep(0.02)


def run_scripted(name: str, delay: float, hold: float) -> None:
    env, steps = script(name)
    env.render()
    wait(env, delay * 4)
    try:
        for label, per_agent, repeats in steps:
            for r in range(repeats):
                actions = np.tile([MOVE_STAY, INTERACT_NONE], (env.n_agents, 1))
                for i, a in per_agent.items():
                    actions[i] = a
                _, reward, terminated, truncated, _ = env.step(actions)
                print(f"{label:<34} {r + 1:>2}/{repeats:<2} reward {reward:+.3f} | lifted {env.lifted[0]} | "
                      f"delivered {env.delivered[0]}")
                env.render()
                wait(env, delay)
                if terminated or truncated:
                    break
        wait(env, hold)
    finally:
        env.close()


def run_random(delay: float, seed: int) -> None:
    env = WarehouseEnv(EnvConfig(n_agents=4, n_objects=2, n_racks=4, capacities=[30, 50, 80], weights=[40, 90, 140]))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    try:
        while True:
            actions = np.stack([rng.integers(0, 5, env.n_agents), rng.integers(0, 3, env.n_agents)], axis=1)
            _, _, terminated, truncated, _ = env.step(actions)
            env.render()
            time.sleep(delay)
            if terminated or truncated:
                env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=["k1", "k2", "k3", "blocked", "random"], default="k3")
    parser.add_argument("--delay", type=float, default=0.12, help="seconds between steps")
    parser.add_argument("--hold", type=float, default=3.0, help="seconds to keep the final frame")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.scenario == "random":
        run_random(args.delay, args.seed)
    else:
        run_scripted(args.scenario, args.delay, args.hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
