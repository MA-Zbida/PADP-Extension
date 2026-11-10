"""Evaluate a trained run (optionally rendered), or a random-policy baseline.

Examples:
    python scripts/evaluate.py runs/e1_mechanism/coupled-k2-long/seed0
    python scripts/evaluate.py runs/e1_mechanism/coupled-k2-long/seed0 --render --episodes 3 --delay 0.05
    python scripts/evaluate.py runs/e4_main/mappo/seed0 --policy random --episodes 50

Episodes use the same fixed seeds as the trainer's evaluation, so numbers are comparable across runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from padp.env.warehouse import WarehouseEnv
from padp.marl.networks import ActorCritic
from padp.marl.sampler import NoiseSource, gumbel_argmax
from padp.marl.trainer import EVAL_SEED_BASE, nanmean
from padp.utils.config_io import build_configs, load_yaml

REPORT_KEYS = ("success", "completion", "steps", "carry_agreement", "help_wait", "coalition_size", "collisions")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="final.pt", help="file inside run_dir (final.pt or latest.pt)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy", choices=["trained", "random"], default="trained")
    parser.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    parser.add_argument("--sampler", choices=["trained", "independent", "coupled", "global", "coupled_move"], default="trained",
                        help="noise used at test time (default: the one used in training); e.g. evaluate a "
                             "coupled-trained policy with independent noise to measure its reliance on shared noise")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between rendered steps")
    args = parser.parse_args(argv)

    config = load_yaml(args.run_dir / "config.yaml")
    env_cfg, train_cfg = build_configs(config)
    env = WarehouseEnv(env_cfg)
    heads = (env.n_move_actions, env.n_interact_actions)
    sampler = train_cfg.sampler if args.sampler == "trained" else args.sampler
    noise = NoiseSource(sampler, train_cfg.seed + 7777, heads)
    rng = np.random.default_rng(0)

    net = None
    if args.policy == "trained":
        path = args.run_dir / args.checkpoint
        if not path.exists():
            path = args.run_dir / "latest.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        critic_dim = env.state_dim if train_cfg.critic == "central" else env.obs_dim
        net = ActorCritic(env.obs_dim, critic_dim, *heads, train_cfg.hidden)
        net.load_state_dict(ckpt["net"])
        net.eval()
        print(f"Loaded {path} (step {ckpt['global_step']:,}) | trained with {train_cfg.sampler} | "
              f"{'greedy' if args.greedy else f'stochastic, {sampler} noise'}")

    summaries = []
    try:
        for k in range(args.episodes):
            obs, info = env.reset(seed=EVAL_SEED_BASE + k)
            done = False
            while not done:
                if net is None:
                    actions = np.stack([rng.integers(0, heads[0], env.n_agents), rng.integers(0, heads[1], env.n_agents)], 1)
                else:
                    with torch.no_grad():
                        ml, il = net.logits(torch.as_tensor(obs))
                    ml, il = ml.numpy(), il.numpy()
                    if args.greedy:
                        actions = np.stack([ml.argmax(-1), il.argmax(-1)], 1)
                    else:
                        mn, inn = noise.sample(info["anchors"], 0, env.episode_idx, env.t)
                        actions = np.stack([gumbel_argmax(ml, mn), gumbel_argmax(il, inn)], 1)
                obs, _, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                if args.render:
                    env.render()
                    time.sleep(args.delay)
            summary = info["episode"]
            summaries.append(summary)
            print(f"episode {k + 1:>3}: success {summary['success']:.0f} | completion {summary['completion']:.2f} | "
                  f"steps {summary['steps']:.0f} | carry agreement {summary['carry_agreement']:.2f}")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()

    if summaries:
        print("-" * 60)
        for key in REPORT_KEYS:
            print(f"{key:>16}: {nanmean(s[key] for s in summaries):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
