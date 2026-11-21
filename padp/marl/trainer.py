"""PPO-family multi-agent trainer: MAPPO or IPPO, with independent or coupled action sampling."""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from padp.env.config import EnvConfig
from padp.env.factory import make_env
from padp.marl.buffer import RolloutBuffer
from padp.marl.config import TrainConfig
from padp.marl.logger import RunLogger
from padp.marl.networks import ActorCritic
from padp.marl.sampler import NoiseSource, coalition_stats, gumbel_argmax, log_softmax

EPISODE_KEYS = (
    "success", "completion", "return", "steps", "carry_agreement", "help_wait", "spare_capacity",
    "coalition_size", "collisions", "invalid_actions", "drop_disagreements", "releases_outside_goal",
    "deliveries_k1", "deliveries_k2", "deliveries_k3plus",
)
EVAL_KEYS = ("success", "completion", "steps", "carry_agreement", "collisions")
EVAL_SEED_BASE = 1_000_000  # evaluation episodes are identical across runs and variants


def nanmean(values) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else math.nan


class Trainer:
    def __init__(self, env_cfg: EnvConfig, cfg: TrainConfig, run_dir: Path, tensorboard: bool = True) -> None:
        self.env_cfg, self.cfg = env_cfg, cfg
        self.run_dir = Path(run_dir)
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)

        self.envs = [make_env(env_cfg) for _ in range(cfg.n_envs)]
        self.n_agents = env_cfg.n_agents
        probe = self.envs[0]
        self.heads = (probe.n_move_actions, probe.n_interact_actions)
        self.obs_dim = probe.obs_dim
        self.critic_dim = probe.state_dim if cfg.critic == "central" else probe.obs_dim
        self.net = ActorCritic(self.obs_dim, self.critic_dim, *self.heads, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.noise = NoiseSource(cfg.sampler, cfg.seed, self.heads, cfg.anchor_error)
        self.logger = RunLogger(self.run_dir, tensorboard)

        self.global_step = 0
        self.episodes = 0
        self.recent: deque = deque(maxlen=100)
        self._q_sum = self._tv_sum = 0.0
        self._coalitions = 0
        self._apply_curriculum()
        self._reset_envs(seed_offset=0)

    # ------------------------------------------------------------------ setup helpers

    def _reset_envs(self, seed_offset: int) -> None:
        E, N = self.cfg.n_envs, self.n_agents
        self.obs = np.zeros((E, N, self.obs_dim), dtype=np.float32)
        self.anchors = np.full((E, N, 2), -1, dtype=np.int64)
        for e, env in enumerate(self.envs):
            obs, info = env.reset(seed=self.cfg.seed * 10_000 + e + seed_offset)
            self.obs[e], self.anchors[e] = obs, info["anchors"]

    def _apply_curriculum(self) -> None:
        if self.cfg.rack_curriculum_steps <= 0:
            return
        n_racks = min(self.env_cfg.n_racks, self.global_step // self.cfg.rack_curriculum_steps)
        for env in self.envs:
            env.set_n_racks(n_racks)

    def _critic_input(self) -> np.ndarray:
        if self.cfg.critic == "local":
            return self.obs.copy()
        states = np.stack([env.state() for env in self.envs])
        return np.repeat(states[:, None, :], self.n_agents, axis=1)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ rollout

    def collect(self, buf: RolloutBuffer) -> np.ndarray:
        cfg, E, N = self.cfg, self.cfg.n_envs, self.n_agents
        for t in range(cfg.rollout_len):
            critic_in = self._critic_input()
            buf.obs[t], buf.critic_in[t] = self.obs, critic_in  # copied before the environments step
            with torch.no_grad():
                move_logits, inter_logits = self.net.logits(self._tensor(self.obs.reshape(E * N, -1)))
                values = self.net.value(self._tensor(critic_in.reshape(E * N, -1))).view(E, N).cpu().numpy()
            move_logits = move_logits.cpu().numpy().reshape(E, N, -1)
            inter_logits = inter_logits.cpu().numpy().reshape(E, N, -1)

            move_noise, inter_noise = np.empty_like(move_logits), np.empty_like(inter_logits)
            for e, env in enumerate(self.envs):
                move_noise[e], inter_noise[e] = self.noise.sample(self.anchors[e], e, env.episode_idx, env.t)
            moves, inters = gumbel_argmax(move_logits, move_noise), gumbel_argmax(inter_logits, inter_noise)
            move_lp, inter_lp = log_softmax(move_logits), log_softmax(inter_logits)
            log_probs = (np.take_along_axis(move_lp, moves[..., None], -1)[..., 0]
                         + np.take_along_axis(inter_lp, inters[..., None], -1)[..., 0])

            move_probs = np.exp(move_lp)
            for e in range(E):
                q, tv, count = coalition_stats(move_probs[e], self.anchors[e])
                self._q_sum += q
                self._tv_sum += tv
                self._coalitions += count

            actions = np.stack([moves, inters], axis=-1)
            rewards = np.zeros((E, N), dtype=np.float32)
            dones = np.zeros(E, dtype=np.float32)
            truncated: List[Tuple[int, np.ndarray]] = []
            for e, env in enumerate(self.envs):
                obs, reward, terminated, trunc, info = env.step(actions[e])
                rewards[e] = reward
                if terminated or trunc:
                    dones[e] = 1.0
                    self.recent.append(info["episode"])
                    self.episodes += 1
                    if trunc:  # time limit: bootstrap from the final state instead of treating it as terminal
                        final = obs if self.cfg.critic == "local" else np.repeat(env.state()[None], N, axis=0)
                        truncated.append((e, final))
                    obs, info = env.reset()
                self.obs[e], self.anchors[e] = obs, info["anchors"]
            if truncated:
                with torch.no_grad():
                    finals = self._tensor(np.stack([f for _, f in truncated]).reshape(len(truncated) * N, -1))
                    boot = self.net.value(finals).view(len(truncated), N).cpu().numpy()
                for k, (e, _) in enumerate(truncated):
                    rewards[e] += cfg.gamma * boot[k]

            buf.actions[t], buf.log_probs[t], buf.values[t] = actions, log_probs, values
            buf.rewards[t], buf.dones[t] = rewards, dones
            self.global_step += E

        with torch.no_grad():
            last = self.net.value(self._tensor(self._critic_input().reshape(E * N, -1))).view(E, N).cpu().numpy()
        return last

    # ------------------------------------------------------------------ PPO update

    def update(self, buf: RolloutBuffer, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if cfg.optimistic:
            advantages = np.maximum(advantages, 0.0)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(cfg.epochs):
            for obs, critic_in, actions, old_lp, adv, ret in buf.minibatches(advantages, returns, cfg.minibatches, self.rng):
                obs_t, critic_t = self._tensor(obs), self._tensor(critic_in)
                actions_t = torch.as_tensor(actions, device=self.device)
                old_lp_t, adv_t, ret_t = self._tensor(old_lp), self._tensor(adv), self._tensor(ret)

                new_lp, entropy = self.net.evaluate(obs_t, actions_t)
                values = self.net.value(critic_t)
                ratio = torch.exp(new_lp - old_lp_t)
                policy_loss = -torch.min(ratio * adv_t, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_t).mean()
                value_loss = 0.5 * ((values - ret_t) ** 2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append((old_lp_t - new_lp).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------ evaluation

    def evaluate(self) -> Dict[str, float]:
        """Fixed-seed episodes with stochastic (configured sampler) and greedy actions."""
        env = make_env(self.env_cfg)
        noise = NoiseSource(self.cfg.sampler, self.cfg.seed + 7777, self.heads, self.cfg.anchor_error)
        results: Dict[str, float] = {}
        for mode in ("stochastic", "greedy"):
            summaries = []
            for k in range(self.cfg.eval_episodes):
                obs, info = env.reset(seed=EVAL_SEED_BASE + k)
                done = False
                while not done:
                    with torch.no_grad():
                        ml, il = self.net.logits(self._tensor(obs))
                    ml, il = ml.cpu().numpy(), il.cpu().numpy()
                    if mode == "greedy":
                        moves, inters = ml.argmax(-1), il.argmax(-1)
                    else:
                        mn, inn = noise.sample(info["anchors"], 0, env.episode_idx, env.t)
                        moves, inters = gumbel_argmax(ml, mn), gumbel_argmax(il, inn)
                    obs, _, terminated, truncated, info = env.step(np.stack([moves, inters], axis=-1))
                    done = terminated or truncated
                summaries.append(info["episode"])
            for key in EVAL_KEYS:
                results[f"{mode}_{key}"] = nanmean(s[key] for s in summaries)
        return results

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str) -> Path:
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "global_step": self.global_step,
            "episodes": self.episodes,
            "env_config": asdict(self.env_cfg),
            "train_config": asdict(self.cfg),
        }, tmp)
        tmp.replace(path)
        return path

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.global_step = int(ckpt["global_step"])
        self.episodes = int(ckpt["episodes"])
        self.logger.truncate_after(self.global_step)
        self._apply_curriculum()
        self._reset_envs(seed_offset=self.global_step)

    # ------------------------------------------------------------------ main loop

    def _rolling(self) -> Dict[str, float]:
        return {k: nanmean(ep[k] for ep in self.recent) for k in EPISODE_KEYS}

    def _log(self, train_stats: Dict[str, float], sps: float) -> None:
        row = self._rolling()
        row["q_agree"] = self._q_sum / self._coalitions if self._coalitions else math.nan
        row["policy_tv"] = self._tv_sum / self._coalitions if self._coalitions else math.nan
        self._q_sum = self._tv_sum = 0.0
        self._coalitions = 0
        row.update(train_stats)
        row.update({"episodes": self.episodes, "racks": self.envs[0].n_racks, "steps_per_sec": sps})
        self.logger.log("metrics", self.global_step, row)
        print(
            f"step {self.global_step:>10,} | success {_pct(row['success'])} | completion {_pct(row['completion'])}"
            f" | agree {_num(row['carry_agreement'])} | q {_num(row['q_agree'])} | tv {_num(row['policy_tv'])}"
            f" | coll {_num(row['collisions'], 1)} | entropy {_num(row['entropy'])} | {sps:,.0f} steps/s",
            flush=True,
        )

    def train(self) -> bool:
        """Run until ``total_steps``. Returns False if interrupted (``latest.pt`` is saved either way)."""
        cfg = self.cfg
        buf = RolloutBuffer(cfg.rollout_len, cfg.n_envs, self.n_agents, self.obs_dim, self.critic_dim)
        next_log = _next_multiple(self.global_step, cfg.log_every)
        next_eval = _next_multiple(self.global_step, cfg.eval_every)
        next_save = _next_multiple(self.global_step, cfg.save_every)
        window_start, window_steps = time.time(), self.global_step
        last_eval = None
        print(f"Training {self.run_dir} | sampler={cfg.sampler} critic={cfg.critic} "
              f"optimistic={cfg.optimistic} | {cfg.total_steps:,} steps", flush=True)
        try:
            while self.global_step < cfg.total_steps:
                self._apply_curriculum()
                last_values = self.collect(buf)
                advantages, returns = buf.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)
                train_stats = self.update(buf, advantages, returns)

                if self.global_step >= next_log:
                    elapsed = max(time.time() - window_start, 1e-9)
                    self._log(train_stats, (self.global_step - window_steps) / elapsed)
                    window_start, window_steps = time.time(), self.global_step
                    next_log = _next_multiple(self.global_step, cfg.log_every)
                if self.global_step >= next_eval:
                    last_eval = (self.global_step, self.evaluate())
                    self.logger.log("eval", *last_eval)
                    next_eval = _next_multiple(self.global_step, cfg.eval_every)
                if self.global_step >= next_save:
                    self.save("latest.pt")
                    next_save = _next_multiple(self.global_step, cfg.save_every)
        except KeyboardInterrupt:
            self.save("latest.pt")
            print(f"\nInterrupted at step {self.global_step:,}. Saved {self.run_dir / 'latest.pt'}; "
                  "rerun the same command to resume.", flush=True)
            self.logger.close()
            return False

        if last_eval is not None and last_eval[0] == self.global_step:
            final_eval = last_eval[1]  # the last scheduled evaluation already ran at this step
        else:
            final_eval = self.evaluate()
            self.logger.log("eval", self.global_step, final_eval)
        self.save("latest.pt")
        self.save("final.pt")
        self.logger.close()
        print(f"Done: {self.global_step:,} steps | eval success stochastic {_pct(final_eval['stochastic_success'])}"
              f" greedy {_pct(final_eval['greedy_success'])}", flush=True)
        return True


def _next_multiple(step: int, every: int) -> int:
    return (step // every + 1) * every


def _pct(x: float) -> str:
    return "  -  " if math.isnan(x) else f"{100 * x:5.1f}%"


def _num(x: float, digits: int = 2) -> str:
    return "  -  " if math.isnan(x) else f"{x:.{digits}f}"
