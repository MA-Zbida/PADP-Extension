"""Standalone Pygame visualizer for the continuous warehouse environment."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pygame

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environment.actions import (
    INTERACT_DROP,
    INTERACT_NONE,
    INTERACT_PICK,
    MOVE_RIGHT,
    MOVE_STAY,
    MOVE_UP,
)
from environment.env import CollaborativeCarryEnv
from environment.geometry import grip_points


def make_actions(
    n_agents: int,
    overrides: dict[int, tuple[int, int]] | None = None,
) -> list[list[int]]:
    actions = [[MOVE_STAY, INTERACT_NONE] for _ in range(n_agents)]
    for agent_idx, action in (overrides or {}).items():
        if 0 <= agent_idx < n_agents:
            actions[agent_idx] = [int(action[0]), int(action[1])]
    return actions


def handle_window_events(env: CollaborativeCarryEnv) -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            raise SystemExit
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
            env.close()
            raise SystemExit


def render_pause(env: CollaborativeCarryEnv, seconds: float) -> None:
    end_time = time.time() + max(0.0, seconds)
    while time.time() < end_time:
        env.render()
        handle_window_events(env)


def print_step(label: str, reward: float, info: dict[str, object]) -> None:
    metrics = info.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    collisions = (
        metrics.get("agent_collisions", 0.0)
        + metrics.get("rack_collisions", 0.0)
        + metrics.get("bound_collisions", 0.0)
        + metrics.get("object_collisions", 0.0)
    )
    print(
        f"{label:<26} | R={reward:>+6.2f} | "
        f"done={100 * metrics.get('completion_ratio', 0.0):>5.1f}% | "
        f"pairs={metrics.get('carrier_pairs', 0.0):.0f} | "
        f"collisions={collisions:.0f}"
    )


def step_and_render(
    env: CollaborativeCarryEnv,
    actions: list[list[int]],
    label: str,
    delay: float,
) -> bool:
    _, reward, terminated, truncated, info = env.step(actions)
    print_step(label, reward, info)
    env.render()
    handle_window_events(env)
    render_pause(env, delay)
    return terminated or truncated


def configure_scripted_scene(env: CollaborativeCarryEnv) -> None:
    object_pos = np.array([0.45, 0.50], dtype=np.float32)
    goal_pos = np.array([0.80, 0.175], dtype=np.float32)

    env.object_positions[0] = object_pos
    env.goal_positions[0] = goal_pos
    env.goal_used = [False for _ in env.goal_positions]
    env.delivered = [False for _ in env.object_positions]
    env.object_holders = [[] for _ in env.object_positions]
    env.agent_holding = [None for _ in env.agent_positions]
    env.agent_grip_sides = [None for _ in env.agent_positions]

    grips = grip_points(object_pos, env.object_radius, env.agent_radius)
    env.agent_positions[0] = grips["left"].copy()
    env.agent_positions[1] = grips["right"].copy()

    if env.n_agents >= 3:
        env.agent_positions[2] = np.array([0.585, 0.50], dtype=np.float32)

    spare_positions = [
        np.array([0.12, 0.86], dtype=np.float32),
        np.array([0.19, 0.86], dtype=np.float32),
        np.array([0.26, 0.86], dtype=np.float32),
        np.array([0.12, 0.91], dtype=np.float32),
        np.array([0.19, 0.91], dtype=np.float32),
        np.array([0.26, 0.91], dtype=np.float32),
    ]
    for agent_idx, pos in zip(range(3, env.n_agents), spare_positions):
        env.agent_positions[agent_idx] = pos.copy()

    env._prev_agent_positions = [pos.copy() for pos in env.agent_positions]
    env._prev_object_positions = [pos.copy() for pos in env.object_positions]
    env._prev_delivered = list(env.delivered)


def scripted_actions(env: CollaborativeCarryEnv) -> Iterable[tuple[str, list[list[int]], int]]:
    pick = make_actions(
        env.n_agents,
        {
            0: (MOVE_STAY, INTERACT_PICK),
            1: (MOVE_STAY, INTERACT_PICK),
        },
    )
    yield "opposite-side pickup", pick, 1

    if env.n_agents >= 3:
        blocked_move = make_actions(
            env.n_agents,
            {
                0: (MOVE_RIGHT, INTERACT_NONE),
                1: (MOVE_RIGHT, INTERACT_NONE),
            },
        )
        yield "blocked carrier move", blocked_move, 1

        clear_blocker = make_actions(env.n_agents, {2: (MOVE_RIGHT, INTERACT_NONE)})
        yield "move blocker aside", clear_blocker, 14

    carry_right = make_actions(
        env.n_agents,
        {
            0: (MOVE_RIGHT, INTERACT_NONE),
            1: (MOVE_RIGHT, INTERACT_NONE),
        },
    )
    yield "carry toward shipping", carry_right, 14

    carry_up = make_actions(
        env.n_agents,
        {
            0: (MOVE_UP, INTERACT_NONE),
            1: (MOVE_UP, INTERACT_NONE),
        },
    )
    yield "align with goal pad", carry_up, 13

    drop = make_actions(
        env.n_agents,
        {
            0: (MOVE_STAY, INTERACT_DROP),
            1: (MOVE_STAY, INTERACT_DROP),
        },
    )
    yield "coordinated drop", drop, 1


def run_scripted(args: argparse.Namespace) -> None:
    env = CollaborativeCarryEnv(
        grid_size=args.grid,
        n_agents=max(2, args.agents),
        n_objects=1,
        n_goals=1,
        n_obstacles=args.obstacles,
        max_agents=max(10, args.agents),
        max_obstacles=max(6, args.obstacles),
        max_grid_size=max(10, args.grid),
        max_steps=200,
        render_mode="human",
        debug_render=args.debug,
    )
    env.reset(seed=args.seed)
    configure_scripted_scene(env)

    print("Scripted warehouse demo")
    print("Controls: Q or Esc closes the window.")
    print("Watch for: opposite grip pickup, blue carrier link, red collision flash, then delivery.")
    env.render()
    render_pause(env, args.delay)

    try:
        for label, actions, repeats in scripted_actions(env):
            for repeat_idx in range(repeats):
                repeated_label = label if repeats == 1 else f"{label} {repeat_idx + 1}/{repeats}"
                done = step_and_render(env, actions, repeated_label, args.delay)
                if done:
                    break
        print("Demo complete. Press Q/Esc or close the window.")
        render_pause(env, args.hold)
    finally:
        env.close()


def run_random(args: argparse.Namespace) -> None:
    env = CollaborativeCarryEnv(
        grid_size=args.grid,
        n_agents=args.agents,
        n_obstacles=args.obstacles,
        max_agents=max(10, args.agents),
        max_obstacles=max(6, args.obstacles),
        max_grid_size=max(10, args.grid),
        max_steps=args.max_steps,
        render_mode="human",
        debug_render=args.debug,
    )
    env.reset(seed=args.seed)

    print("Random-action warehouse preview")
    print("Controls: Q or Esc closes the window.")

    try:
        while True:
            env.render()
            handle_window_events(env)
            actions = env.action_space.sample().reshape(env.n_agents, 2).astype(np.int64).tolist()
            _, reward, terminated, truncated, info = env.step(actions)
            print_step("random step", reward, info)
            render_pause(env, args.delay)
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the continuous warehouse Pygame renderer")
    parser.add_argument("--mode", choices=["scripted", "random"], default="scripted",
                        help="scripted shows pickup/carry/drop; random samples environment actions")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of agents to visualize")
    parser.add_argument("--obstacles", type=int, default=4,
                        help="Number of rack obstacles")
    parser.add_argument("--grid", type=int, default=8,
                        help="Compatibility grid parameter")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="Seconds to pause after each rendered transition")
    parser.add_argument("--hold", type=float, default=5.0,
                        help="Seconds to keep the final scripted frame open")
    parser.add_argument("--seed", type=int, default=7,
                        help="Environment reset seed")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Episode length for random mode")
    parser.add_argument("--debug", action="store_true",
                        help="Show grip zones around unheld objects")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "scripted":
        run_scripted(args)
    else:
        run_random(args)


if __name__ == "__main__":
    main()
