"""Stage 0 go/no-go check: does shared (coupled) sampling noise help agents learn joint actions?

The toy game strips the warehouse down to its coordination core. k agents each have their own
softmax policy over 5 moves. Each step all agents act; the team makes progress only when *every*
agent picks the target move. The team succeeds after L such steps within H = 3L steps. Policies
are trained with REINFORCE (batch-mean baseline, entropy bonus, Adam), either with independent
Gumbel noise per agent or with one noise vector shared by all agents.

Run (about 2 to 4 minutes on a laptop CPU):
    python scripts/toy_agreement_game.py

Outputs in results/stage0_toy/: summary.md (paste it back), success_vs_L.png,
success_vs_entropy.png, learning_curves.png.

The gate below was fixed before any result was seen:
  G1  coupled sampling leaves each agent's action frequencies equal to its policy (error < 0.01);
  G2  at k=2, L=10, entropy 0.02 (dense reward), coupled final success exceeds independent by at
      least 0.2 with non-overlapping 95% CIs, or reaches 50% success in at most half the iterations;
  G3  the coupled-minus-independent gap is larger at k=3 than at k=2 (same setting).
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
N_ACTIONS, TARGET = 5, 1
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SAMPLER_COLOR = {"independent": BLUE, "coupled": ORANGE}


def log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def train(k: int, L: int, entropy: float, sampler: str, reward: str, n_seeds: int, iters: int, batch: int,
          lr: float, seed: int) -> Dict[str, np.ndarray]:
    """Train n_seeds independent teams in parallel. Returns per-seed learning curves."""
    rng = np.random.default_rng([seed, k, L, int(round(entropy * 1e4)), int(sampler == "coupled"), int(reward == "dense")])
    S, B, A, H = n_seeds, batch, N_ACTIONS, 3 * L
    theta = rng.normal(0.0, 0.1, size=(S, k, A))  # separate policies that start slightly different
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    eye = np.eye(A)
    success_curve = np.zeros((S, iters))
    agree_curve = np.zeros((S, iters))
    for it in range(iters):
        logp = log_softmax(theta)
        pi = np.exp(logp)
        progress = np.zeros((S, B))
        active = np.ones((S, B), dtype=bool)
        steps = np.zeros((S, B))
        counts = np.zeros((S, B, k, A))
        ret = np.zeros((S, B))
        agreements = np.zeros((S, B))
        for _ in range(H):
            noise_shape = (S, B, 1, A) if sampler == "coupled" else (S, B, k, A)
            acts = np.argmax(logp[:, None] + rng.gumbel(size=noise_shape), axis=-1)  # (S, B, k)
            agree = np.all(acts == acts[..., :1], axis=-1) & active
            step_progress = agree & (acts[..., 0] == TARGET)
            counts += eye[acts] * active[..., None, None]
            steps += active
            agreements += agree
            progress += step_progress
            if reward == "dense":
                ret += step_progress / L
            finished = active & (progress >= L)
            ret += finished
            active &= ~finished
            if not active.any():
                break
        success_curve[:, it] = (progress >= L).mean(1)
        agree_curve[:, it] = (agreements.sum(1) / steps.sum(1))

        adv = (ret - ret.mean(1, keepdims=True)) / (ret.std(1, keepdims=True) + 1e-8)
        score = (counts - steps[..., None, None] * pi[:, None]) / steps[..., None, None]  # per-step-normalized
        grad = (adv[..., None, None] * score).mean(1)
        ent = -(pi * logp).sum(-1, keepdims=True)
        grad += entropy * (-pi * (logp + ent))  # gradient of the entropy bonus
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta += lr * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
    return {"success": success_curve, "agreement": agree_curve, "target_prob": np.exp(log_softmax(theta))[..., TARGET].mean(1)}


def marginal_check(seed: int = 0, draws: int = 200_000) -> Tuple[float, float, float]:
    """Max error between coupled samples' frequencies and the policies; agreement vs the TV bound."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, size=(2, N_ACTIONS))
    g = rng.gumbel(size=(draws, 1, N_ACTIONS))
    acts = np.argmax(log_softmax(logits)[None] + g, axis=-1)
    pi = np.exp(log_softmax(logits))
    freq = np.stack([np.bincount(acts[:, i], minlength=N_ACTIONS) / draws for i in range(2)])
    tv = 0.5 * np.abs(pi[0] - pi[1]).sum()
    return float(np.abs(freq - pi).max()), float(np.mean(acts[:, 0] == acts[:, 1])), float((1 - tv) / (1 + tv))


def final_success(curve: np.ndarray) -> np.ndarray:
    tail = max(1, curve.shape[1] // 10)
    return curve[:, -tail:].mean(1)


def iters_to(curve: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    smooth = np.apply_along_axis(lambda c: np.convolve(c, np.ones(5) / 5, mode="same"), 1, curve)
    out = np.full(curve.shape[0], np.nan)
    for s in range(curve.shape[0]):
        hit = np.nonzero(smooth[s] >= threshold)[0]
        if len(hit):
            out[s] = hit[0]
    return out


def median_iters_to(curve: np.ndarray, threshold: float = 0.5) -> float:
    """Median over seeds of the iterations needed to reach ``threshold``; inf if no seed reached it."""
    hits = iters_to(curve, threshold)
    return float(np.nanmedian(hits)) if np.any(~np.isnan(hits)) else math.inf


def mean_ci(values: np.ndarray, n: int = 2000) -> Tuple[float, float, float]:
    rng = np.random.default_rng(0)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return float(values.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ---------------------------------------------------------------------- figures

def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=9, fontweight="bold")


def _line_panel(ax, xs, results, key_fn, xlabel):
    ends = {}
    for sampler in ("independent", "coupled"):
        stats = [mean_ci(final_success(results[key_fn(x, sampler)]["success"])) for x in xs]
        mean = [s[0] for s in stats]
        err = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
        positions = range(len(xs))
        ax.errorbar(positions, mean, yerr=err, color=SAMPLER_COLOR[sampler], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, capsize=0, label=sampler)
        ends[sampler] = mean[-1]
    # Direct labels at the right end, pushed apart when the two lines finish close together.
    close = abs(ends["coupled"] - ends["independent"]) < 0.08
    for sampler, other in (("coupled", "independent"), ("independent", "coupled")):
        above = ends[sampler] > ends[other] or (ends[sampler] == ends[other] and sampler == "coupled")
        offset = (7 if above else -7) if close else 0
        ax.annotate(sampler, (len(xs) - 1, ends[sampler]), xytext=(6, offset), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_xlim(-0.3, len(xs) - 0.2 + 0.9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)


def make_figures(results, grid, out: Path) -> None:
    ks, Ls, ents, rewards = grid["k"], grid["L"], grid["entropy"], grid["reward"]
    ent_mid = 0.02 if 0.02 in ents else ents[len(ents) // 2]
    L_mid = 10 if 10 in Ls else Ls[len(Ls) // 2]

    fig, axes = plt.subplots(len(rewards), len(ks), figsize=(3.3 * len(ks), 2.6 * len(rewards) + 0.5), dpi=150,
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for r, reward in enumerate(rewards):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            _style(ax, f"k = {k} agents, {reward} reward")
            _line_panel(ax, Ls, results, lambda L, s: (k, L, ent_mid, s, reward), "Matched moves needed (L)")
    axes[0, 0].set_ylabel("Final success", color=INK2, fontsize=8)
    fig.suptitle(f"Success vs task length (entropy {ent_mid}; mean of seeds, 95% CI)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "success_vs_L.png", facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 3.0), dpi=150, squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for c, k in enumerate(ks):
        ax = axes[0, c]
        _style(ax, f"k = {k} agents, L = {L_mid}, dense reward")
        _line_panel(ax, ents, results, lambda e, s: (k, L_mid, e, s, "dense"), "Entropy bonus")
    axes[0, 0].set_ylabel("Final success", color=INK2, fontsize=8)
    fig.suptitle("Success vs exploration (entropy-agreement tension)", x=0.01, ha="left", fontsize=10,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "success_vs_entropy.png", facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(len(rewards), len(ks), figsize=(3.3 * len(ks), 2.6 * len(rewards) + 0.8), dpi=150,
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    shown = 0
    for r, reward in enumerate(rewards):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            _style(ax, f"k = {k} agents, L = {L_mid}, {reward} reward")
            for sampler in ("independent", "coupled"):
                curve = results[(k, L_mid, ent_mid, sampler, reward)]["success"]
                shown = curve.shape[1] if curve.shape[1] < 200 else 100  # the dynamics happen early
                x = np.arange(curve.shape[1])
                ax.plot(x, curve.mean(0), color=SAMPLER_COLOR[sampler], linewidth=2, label=sampler)
                ax.fill_between(x, np.quantile(curve, 0.1, 0), np.quantile(curve, 0.9, 0),
                                color=SAMPLER_COLOR[sampler], alpha=0.15, linewidth=0)
            ax.set_xlim(0, shown)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xlabel(f"Training iteration (first {shown})", color=INK2, fontsize=8)
        axes[r, 0].set_ylabel("Success rate", color=INK2, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.suptitle(f"Learning curves (entropy {ent_mid}; mean, band = 10th-90th percentile of seeds)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "learning_curves.png", facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- gate and summary

def evaluate_gate(results, grid, marginal_error) -> List[Tuple[str, bool, str]]:
    ent = 0.02 if 0.02 in grid["entropy"] else grid["entropy"][len(grid["entropy"]) // 2]
    L = 10 if 10 in grid["L"] else grid["L"][len(grid["L"]) // 2]
    gates = [("G1 marginals preserved", marginal_error < 0.01, f"max frequency error {marginal_error:.4f} (< 0.01)")]

    def gap(k):
        ind = final_success(results[(k, L, ent, "independent", "dense")]["success"])
        cpl = final_success(results[(k, L, ent, "coupled", "dense")]["success"])
        return ind, cpl

    ind, cpl = gap(2)
    m_i, lo_i, hi_i = mean_ci(ind)
    m_c, lo_c, hi_c = mean_ci(cpl)
    t_i = median_iters_to(results[(2, L, ent, "independent", "dense")]["success"])
    t_c = median_iters_to(results[(2, L, ent, "coupled", "dense")]["success"])
    by_success = (m_c - m_i >= 0.2) and lo_c > hi_i
    by_speed = t_c <= 0.5 * t_i if math.isfinite(t_c) else False
    gates.append((f"G2 coupled helps at k=2, L={L}, entropy {ent}", by_success or by_speed,
                  f"success {m_i:.2f} [{lo_i:.2f}, {hi_i:.2f}] -> {m_c:.2f} [{lo_c:.2f}, {hi_c:.2f}]; "
                  f"iterations to 50%: {_fmt_iter(t_i)} -> {_fmt_iter(t_c)}"))
    if 3 in grid["k"]:
        ind3, cpl3 = gap(3)
        g2, g3 = cpl.mean() - ind.mean(), cpl3.mean() - ind3.mean()
        gates.append(("G3 gap grows with k", g3 > g2, f"gap k=2: {g2:+.2f}, gap k=3: {g3:+.2f}"))
    return gates


def _fmt_iter(x: float) -> str:
    return "never" if not math.isfinite(x) else f"{x:.0f}"


def write_summary(results, grid, gates, marginal, settings, out: Path, elapsed: float) -> str:
    err, agree, bound = marginal
    passed = all(ok for _, ok, _ in gates)
    lines = [
        "# Stage 0: toy agreement game",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} | {settings['seeds']} seeds | {settings['iters']} iterations x "
        f"{settings['batch']} episodes | {elapsed / 60:.1f} min",
        "",
        f"## Gate verdict: {'PASS' if passed else 'FAIL'}",
        "",
        "| Criterion | Result | Evidence |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {'pass' if ok else 'FAIL'} | {evidence} |" for name, ok, evidence in gates]
    lines += [
        "",
        f"Coupled agreement for two different random policies: {agree:.3f} (theoretical lower bound {bound:.3f}).",
        "",
        "## All settings (final success = mean of the last 10% of iterations; mean of seeds [95% CI])",
        "",
        "| k | L | entropy | reward | independent | coupled | gap | agreement indep -> coupled |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, L, ent, reward in itertools.product(grid["k"], grid["L"], grid["entropy"], grid["reward"]):
        ind = results[(k, L, ent, "independent", reward)]
        cpl = results[(k, L, ent, "coupled", reward)]
        mi, li, hi = mean_ci(final_success(ind["success"]))
        mc, lc, hc = mean_ci(final_success(cpl["success"]))
        ai, ac = ind["agreement"][:, -1].mean(), cpl["agreement"][:, -1].mean()
        lines.append(f"| {k} | {L} | {ent} | {reward} | {mi:.2f} [{li:.2f}, {hi:.2f}] | {mc:.2f} [{lc:.2f}, {hc:.2f}] | "
                     f"{mc - mi:+.2f} | {ai:.2f} -> {ac:.2f} |")
    lines += ["", "Figures: `success_vs_L.png`, `success_vs_entropy.png`, `learning_curves.png`", ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64, help="episodes per iteration")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "stage0_toy")
    parser.add_argument("--quick", action="store_true", help="tiny run to check the script works (not a result)")
    args = parser.parse_args(argv)

    grid = {"k": [2, 3], "L": [5, 10, 20], "entropy": [0.0, 0.02, 0.05], "reward": ["dense", "sparse"]}
    if args.quick:
        args.seeds, args.iters, args.batch = 2, 40, 16
        grid = {"k": [2, 3], "L": [5, 10], "entropy": [0.02], "reward": ["dense"]}
        if args.out == REPO_ROOT / "results" / "stage0_toy":
            args.out = REPO_ROOT / "results" / "_smoke_toy"
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    marginal = marginal_check(args.seed)
    print(f"Marginal check: max frequency error {marginal[0]:.4f}; coupled agreement {marginal[1]:.3f} "
          f"(bound {marginal[2]:.3f})")
    combos = list(itertools.product(grid["k"], grid["L"], grid["entropy"], ["independent", "coupled"], grid["reward"]))
    results = {}
    for n, combo in enumerate(combos, start=1):
        k, L, ent, sampler, reward = combo
        results[combo] = train(k, L, ent, sampler, reward, args.seeds, args.iters, args.batch, args.lr, args.seed)
        print(f"[{n}/{len(combos)}] k={k} L={L} entropy={ent} {sampler:<11} {reward:<6} final success "
              f"{final_success(results[combo]['success']).mean():.2f}", flush=True)

    gates = evaluate_gate(results, grid, marginal[0])
    make_figures(results, grid, args.out)
    text = write_summary(results, grid, gates, marginal,
                         {"seeds": args.seeds, "iters": args.iters, "batch": args.batch}, args.out, time.time() - start)
    print()
    print(text)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
