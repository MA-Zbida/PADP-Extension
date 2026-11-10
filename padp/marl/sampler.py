"""Action sampling with independent or object-anchored coupled Gumbel noise.

Every agent samples ``argmax(log pi(a) + g)`` (the Gumbel-max trick), so each agent's action
follows its own policy exactly in both modes. In ``coupled`` mode, agents anchored to the same
object draw ``g`` from a generator seeded by facts they all know (run, episode, object, step,
action head). No message is exchanged, yet the members' samples agree whenever their policies
are close: with identical policies they always pick the same action.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

SAMPLERS = ("independent", "coupled", "global", "coupled_move")
# Ablations of the coupled sampler:
#   global        every agent shares one noise stream per (env, episode, step, head), ignoring anchors;
#   coupled_move  coupling on the movement head only (interaction noise stays private).
GLOBAL_ANCHOR = 2**31 - 1  # anchor value used by the global sampler; no object has this id


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def gumbel_argmax(logits: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return np.argmax(log_softmax(logits) + noise, axis=-1)


class NoiseSource:
    """Gumbel noise per agent and action head; shared within an anchored coalition when coupled."""

    def __init__(self, mode: str, run_seed: int, head_sizes: Sequence[int], anchor_error: float = 0.0) -> None:
        if mode not in SAMPLERS:
            raise ValueError(f"sampler={mode!r}; expected one of {SAMPLERS}")
        if not 0.0 <= anchor_error <= 1.0:
            raise ValueError("anchor_error must be in [0, 1]")
        self.mode = mode
        self.run_seed = int(run_seed)
        self.head_sizes = tuple(head_sizes)
        self.private = np.random.default_rng([self.run_seed, 0x5EED])
        # Robustness test of the common-knowledge assumption: with this probability, an agent uses a
        # wrong anchor in a step, so its noise no longer matches its teammates'. A separate generator
        # keeps every other random stream unchanged.
        self.anchor_error = float(anchor_error)
        self.errors = np.random.default_rng([self.run_seed, 0xBAD]) if anchor_error > 0 else None

    def shared(self, env_idx: int, episode_idx: int, anchor: int, t: int, head: int) -> np.ndarray:
        """The noise every member of coalition ``anchor`` derives independently."""
        key = (self.run_seed, env_idx, episode_idx, anchor, t, head)
        if all(0 <= k < 2**32 for k in key):
            # Each such int is one uint32 word of seed entropy, so this is the same seed as the
            # list, without NumPy's slow conversion of a Python list.
            key = np.array(key, dtype=np.uint32)
        return np.random.default_rng(key).gumbel(size=self.head_sizes[head])

    def sample(self, anchors: np.ndarray, env_idx: int, episode_idx: int, t: int) -> Tuple[np.ndarray, ...]:
        """Noise for one environment. ``anchors`` is ``(n_agents, n_heads)``; -1 means no anchor."""
        n_agents = anchors.shape[0]
        out = []
        for head, size in enumerate(self.head_sizes):
            noise = self.private.gumbel(size=(n_agents, size))
            if self.mode == "global":
                noise[:] = self.shared(env_idx, episode_idx, GLOBAL_ANCHOR, t, head)
            elif self.mode == "coupled" or (self.mode == "coupled_move" and head == 0):
                cache: Dict[int, np.ndarray] = {}
                for i in range(n_agents):
                    anchor = int(anchors[i, head])
                    if anchor >= 0 and self.errors is not None and self.errors.random() < self.anchor_error:
                        anchor = GLOBAL_ANCHOR - 1 - i  # a wrong anchor that no teammate shares
                    if anchor >= 0:
                        if anchor not in cache:
                            cache[anchor] = self.shared(env_idx, episode_idx, anchor, t, head)
                        noise[i] = cache[anchor]
            out.append(noise)
        return tuple(out)


def coalition_stats(probs: np.ndarray, anchors: np.ndarray) -> Tuple[float, float, int]:
    """Agreement diagnostics for lifted coalitions in one environment.

    ``probs`` is ``(n_agents, n_moves)``; ``anchors[:, 0]`` holds each agent's lifted-object id.
    Returns (sum of q_k, sum of mean pairwise TV, number of coalitions), where
    q_k = sum_a prod_i pi_i(a) is the agreement probability under independent sampling.
    """
    q_sum = tv_sum = 0.0
    count = 0
    if anchors[:, 0].max() < 0:
        return q_sum, tv_sum, count  # no lifted coalition (the common case)
    for anchor in np.unique(anchors[:, 0]):
        if anchor < 0:
            continue
        members = probs[anchors[:, 0] == anchor]
        if len(members) < 2:
            continue
        q_sum += float(np.prod(members, axis=0).sum())
        pairs = [0.5 * np.abs(members[a] - members[b]).sum() for a in range(len(members)) for b in range(a + 1, len(members))]
        tv_sum += float(np.mean(pairs))
        count += 1
    return q_sum, tv_sum, count
