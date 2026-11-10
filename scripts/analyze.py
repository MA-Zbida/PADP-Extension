"""Summarize an experiment's runs into results/<experiment>/summary.md plus figures.

Examples:
    python scripts/analyze.py e1_mechanism
    python scripts/analyze.py --all

Paste the printed summary (also saved as summary.md) back into the conversation.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from padp.utils.config_io import REPO_ROOT, Experiment, load_experiment  # noqa: E402
from padp.utils.stats import bootstrap_ci, iqm, mann_whitney, normalized_auc, steps_to_threshold  # noqa: E402

# Reference categorical palette (fixed order). Baseline = slot 1, treatment = slot 2.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASELINE_AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


@dataclass
class Run:
    seed: int
    metrics: pd.DataFrame
    final_eval: Dict[str, float]
    finished: bool


def load_runs(experiment: Experiment, runs_root: Path) -> Dict[str, List[Run]]:
    out: Dict[str, List[Run]] = {}
    for variant, _ in experiment.variants():
        runs = []
        for seed in experiment.seeds:
            run_dir = experiment.run_dir(variant, seed, runs_root)
            if not (run_dir / "metrics.csv").exists():
                continue
            metrics = pd.read_csv(run_dir / "metrics.csv")
            final_eval: Dict[str, float] = {}
            if (run_dir / "eval.csv").exists():
                ev = pd.read_csv(run_dir / "eval.csv")
                if len(ev):
                    final_eval = ev.iloc[-1].to_dict()
            runs.append(Run(seed, metrics, final_eval, (run_dir / "final.pt").exists()))
        out[variant] = runs
    return out


def _fmt(x: float, digits: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _fmt_steps(x: float) -> str:
    if math.isnan(x):
        return "-"
    if math.isinf(x):
        return "never"
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"


TAIL = 0.1  # end-of-training metrics average the last 10% of training (one log row is noisy)
STAGE_COLUMNS = ("help_wait", "coalition_size", "spare_capacity", "carry_agreement", "drop_disagreements",
                 "releases_outside_goal", "collisions", "deliveries_k1", "deliveries_k2", "deliveries_k3plus")


def _tail(run: Run, column: str) -> float:
    m = run.metrics
    if column not in m or not len(m):
        return math.nan
    return _nan_stat(np.mean, m.loc[m["step"] >= m["step"].max() * (1 - TAIL), column])


def _nan_stat(fn, values) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(fn(vals)) if vals else math.nan


def _reached_50(run: Run) -> float:
    """First step at which training success reached 50%; inf if it never did."""
    step = steps_to_threshold(run.metrics["step"], run.metrics["success"], 0.5)
    return math.inf if math.isnan(step) else step


def summarize_variant(runs: List[Run]) -> Dict[str, float]:
    success = [r.final_eval.get("stochastic_success", math.nan) for r in runs]
    steps = [r.final_eval.get("stochastic_steps", math.nan) for r in runs]
    auc = [normalized_auc(r.metrics["step"], r.metrics["success"]) for r in runs]
    lo, hi = bootstrap_ci(success)
    steps_lo, steps_hi = bootstrap_ci(steps)
    return {
        "success_iqm": iqm(success), "success_lo": lo, "success_hi": hi,
        "solved": sum(v >= 0.5 for v in success if not math.isnan(v)),
        "greedy": _nan_stat(np.mean, [r.final_eval.get("greedy_success", math.nan) for r in runs]),
        "steps_iqm": iqm(steps), "steps_lo": steps_lo, "steps_hi": steps_hi,
        "agreement": _nan_stat(np.mean, [_tail(r, "carry_agreement") for r in runs]),
        "q_agree": _nan_stat(np.mean, [_tail(r, "q_agree") for r in runs]),
        # Seeds that never reached 50% count as infinitely late instead of being left out.
        "steps_to_50": float(np.median([_reached_50(r) for r in runs])) if runs else math.nan,
        "auc": iqm(auc),
        "success_values": success, "steps_values": steps, "auc_values": auc,
    }


def comparisons(experiment: Experiment, stats: Dict[str, Dict]) -> List[Dict]:
    if not experiment.compare:
        return []
    factor, base, treat = experiment.compare["factor"], experiment.compare["baseline"], experiment.compare["treatment"]
    rows = []
    for variant in stats:
        levels = experiment.variant_levels(variant)
        if levels.get(factor) != base:
            continue
        partner = "-".join(treat if name == factor else level for name, level in levels.items())
        if partner not in stats:
            continue
        condition = "-".join(level for name, level in levels.items() if name != factor) or "all"
        a, b = stats[variant], stats[partner]
        row = {"condition": condition, "agree_base": a["agreement"], "agree_treat": b["agreement"]}
        for metric, key in (("success", "success_iqm"), ("auc", "auc"), ("steps", "steps_iqm")):
            row[metric] = (a[key], b[key], mann_whitney(a[f"{metric}_values"], b[f"{metric}_values"]))
        rows.append(row)
    return rows


def write_summary(experiment: Experiment, runs: Dict[str, List[Run]], stats: Dict[str, Dict],
                  comps: List[Dict], out_dir: Path) -> str:
    total = len(experiment.variants()) * len(experiment.seeds)
    found = sum(len(r) for r in runs.values())
    finished = sum(sum(x.finished for x in r) for r in runs.values())
    lines = [
        f"# {experiment.name}: results summary",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} | runs finished {finished}/{total} (started {found}) | "
        f"seeds {experiment.seeds}",
        "",
        experiment.description.strip(),
        "",
        "## Final performance",
        "",
        "- **Success**: final evaluation on 20 fixed episodes, stochastic policy with the training sampler. "
        "IQM across seeds with a 95% bootstrap CI. **Solved**: seeds with final success of at least 0.5.",
        "- **Steps/episode**: mean episode length in the same evaluation (the episode limit when the task is "
        "not done), so lower is faster delivery.",
        "- **AUC**: training success averaged over the whole run (IQM across seeds); it measures learning speed.",
        "- **Steps to 50%**: median over seeds of the first step where training success reached 50%. Seeds that "
        "never got there count as \"never\", so the median is \"never\" when half of the seeds or more failed.",
        f"- **Carry agreement**, **q_k**, **help wait**: training averages over the last {TAIL:.0%} of steps.",
        "",
        "| Variant | Runs | Solved | Success IQM [95% CI] | Greedy | Steps/episode IQM [95% CI] | AUC | "
        "Steps to 50% | Carry agreement | q_k |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, s in stats.items():
        n_done = sum(r.finished for r in runs[variant])
        lines.append(
            f"| {variant} | {n_done}/{len(experiment.seeds)} | {s['solved']}/{len(runs[variant])} | "
            f"{_fmt(s['success_iqm'])} [{_fmt(s['success_lo'])}, {_fmt(s['success_hi'])}] | {_fmt(s['greedy'])} | "
            f"{_fmt(s['steps_iqm'], 1)} [{_fmt(s['steps_lo'], 1)}, {_fmt(s['steps_hi'], 1)}] | {_fmt(s['auc'])} | "
            f"{_fmt_steps(s['steps_to_50'])} | {_fmt(s['agreement'])} | {_fmt(s['q_agree'])} |"
        )
    if comps:
        c = experiment.compare
        lines += [
            "",
            f"## {c['treatment']} vs {c['baseline']} (same other settings)",
            "",
            f"Each cell reads {c['baseline']} -> {c['treatment']}; p is a two-sided Mann-Whitney test across seeds "
            "(with 5 seeds per side the smallest possible p is 0.008).",
            "",
            "| Condition | Success | p | AUC | p | Steps/episode | p | Carry agreement |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in comps:
            cells = []
            for metric, digits in (("success", 2), ("auc", 2), ("steps", 1)):
                base, treat, p = row[metric]
                cells += [f"{_fmt(base, digits)} -> {_fmt(treat, digits)}", _fmt(p, 3)]
            lines.append(f"| {row['condition']} | {' | '.join(cells)} | "
                         f"{_fmt(row['agree_base'])} -> {_fmt(row['agree_treat'])} |")
    lines += [
        "",
        "## Coordination stages",
        "",
        f"Per-episode training averages over the last {TAIL:.0%} of steps, IQM across seeds. Help wait: steps "
        "from the first attach to the lift. Coalition size and spare capacity (kg above the weight): at the lift. "
        "Ignored drops: a holder dropped while the others did not. Deliveries are split by coalition size.",
        "",
        "| Variant | Help wait | Coalition size | Spare capacity | Carry agreement | Ignored drops | "
        "Releases outside goal | Collisions | Deliveries k=1 / 2 / 3+ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, variant_runs in runs.items():
        stage = {c: iqm([_tail(r, c) for r in variant_runs]) for c in STAGE_COLUMNS}
        deliveries = " / ".join(_fmt(stage[c]) for c in ("deliveries_k1", "deliveries_k2", "deliveries_k3plus"))
        lines.append(
            f"| {variant} | {_fmt(stage['help_wait'], 1)} | {_fmt(stage['coalition_size'])} | "
            f"{_fmt(stage['spare_capacity'], 0)} | {_fmt(stage['carry_agreement'])} | "
            f"{_fmt(stage['drop_disagreements'], 1)} | {_fmt(stage['releases_outside_goal'])} | "
            f"{_fmt(stage['collisions'], 1)} | {deliveries} |"
        )
    lines += [
        "",
        "## Every seed",
        "",
        "| Variant | Seed | Success | Greedy | Steps/episode | AUC | Reached 50% at | Carry agreement | Help wait |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, variant_runs in runs.items():
        for r in variant_runs:
            ev = r.final_eval
            lines.append(
                f"| {variant} | {r.seed} | {_fmt(ev.get('stochastic_success', math.nan))} | "
                f"{_fmt(ev.get('greedy_success', math.nan))} | {_fmt(ev.get('stochastic_steps', math.nan), 1)} | "
                f"{_fmt(normalized_auc(r.metrics['step'], r.metrics['success']))} | {_fmt_steps(_reached_50(r))} | "
                f"{_fmt(_tail(r, 'carry_agreement'))} | {_fmt(_tail(r, 'help_wait'), 1)} |"
            )
    lines += ["", "Figures: `learning_curves.png`, `final_metrics.png`", ""]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------- figures

def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE_AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=9, fontweight="bold")


def _mean_curve(runs: List[Run], column: str, grid: np.ndarray):
    curves = []
    for r in runs:
        if column in r.metrics and len(r.metrics) > 1:
            curves.append(np.interp(grid, r.metrics["step"], r.metrics[column].fillna(0.0)))
    if not curves:
        return None
    curves = np.array(curves)
    rng = np.random.default_rng(0)
    boots = np.array([curves[rng.integers(0, len(curves), len(curves))].mean(0) for _ in range(500)])
    return curves.mean(0), np.quantile(boots, 0.025, 0), np.quantile(boots, 0.975, 0)


def _colors(experiment: Experiment, variants: List[str]) -> Dict[str, str]:
    """Color follows the compared factor's level (baseline blue, treatment orange) or the variant order."""
    if experiment.compare:
        factor = experiment.compare["factor"]
        order = list(next(f for f in experiment.factors if f.name == factor).levels)
        order.sort(key=lambda lvl: (lvl != experiment.compare["baseline"], lvl != experiment.compare["treatment"]))
        return {v: PALETTE[order.index(experiment.variant_levels(v)[factor]) % len(PALETTE)] for v in variants}
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variants)}


def plot_learning_curves(experiment: Experiment, runs: Dict[str, List[Run]], out_dir: Path) -> None:
    variants = [v for v, r in runs.items() if r]
    if not variants:
        return
    max_step = max(float(r.metrics["step"].max()) for v in variants for r in runs[v] if len(r.metrics))
    grid = np.linspace(0, max_step, 120)
    colors = _colors(experiment, variants)

    factor = experiment.compare["factor"] if experiment.compare else None
    panels: Dict[str, List[str]] = {}
    for v in variants:
        levels = experiment.variant_levels(v)
        key = "-".join(l for n, l in levels.items() if n != factor) or experiment.name
        panels.setdefault(key, []).append(v)

    n = len(panels)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    # At least 6.8 in wide, so the title and a one-line legend fit even with a single panel.
    fig, axes = plt.subplots(rows, cols, figsize=(max(3.4 * cols, 6.8), 2.5 * rows + 0.5), dpi=150,
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (panel, members) in zip(axes.ravel(), panels.items()):
        _style(ax, "" if panel == experiment.name else panel)  # a lone panel needs no title of its own
        for v in members:
            curve = _mean_curve(runs[v], "success", grid)
            if curve is None:
                continue
            mean, lo, hi = curve
            label = experiment.variant_levels(v).get(factor, v) if factor else v
            ax.fill_between(grid, lo, hi, color=colors[v], alpha=0.15, linewidth=0)
            ax.plot(grid, mean, color=colors[v], linewidth=2, label=label)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Environment steps", color=INK2, fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Training success rate", color=INK2, fontsize=8)
    # Title, then the legend on its own line below it (fixed margins in inches, so narrow figures don't collide).
    height = fig.get_figheight()
    fig.suptitle(f"{experiment.name}: success during training (mean across seeds, 95% CI)", x=0.01,
                 y=1 - 0.08 / height, ha="left", va="top", fontsize=10, fontweight="bold", color=INK)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.32 / height), frameon=False,
               fontsize=8, labelcolor=INK2, ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / height))
    fig.savefig(out_dir / "learning_curves.png", facecolor=SURFACE)
    plt.close(fig)


def plot_final_metrics(experiment: Experiment, stats: Dict[str, Dict], out_dir: Path) -> None:
    """Final success and steps per episode, one row per variant (IQM with 95% CI)."""
    variants = [v for v, s in stats.items() if not math.isnan(s["success_iqm"])]
    if not variants:
        return
    if experiment.compare:  # put the compared variants of each condition next to each other
        factor = experiment.compare["factor"]

        def condition(v: str) -> str:
            return "-".join(l for n, l in experiment.variant_levels(v).items() if n != factor)

        conditions = list(dict.fromkeys(condition(v) for v in variants))
        variants.sort(key=lambda v: conditions.index(condition(v)))
    colors = _colors(experiment, variants)
    panels = [("success", "Success rate", "{:.2f}"), ("steps", "Steps per episode (lower is faster)", "{:.0f}")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.32 * len(variants) + 1.2), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, xlabel, label_fmt) in zip(axes, panels):
        _style(ax, "")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        for y, v in enumerate(variants):
            mid, lo, hi = (stats[v][f"{key}_{part}"] for part in ("iqm", "lo", "hi"))
            if math.isnan(mid):
                continue
            ax.plot([lo, hi], [y, y], color=colors[v], linewidth=2, solid_capstyle="round")
            ax.plot(mid, y, "o", color=colors[v], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(label_fmt.format(mid), (mid, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
        ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    axes[0].set_xlim(-0.02, 1.02)
    axes[1].set_xlim(left=0)
    axes[0].set_yticks(range(len(variants)), variants, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"{experiment.name}: final evaluation (IQM across seeds, 95% CI)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "final_metrics.png", facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- entry point

def analyze(config_path: Path, runs_root: Path, results_root: Path) -> Optional[str]:
    experiment = load_experiment(config_path)
    runs = load_runs(experiment, runs_root)
    if not any(runs.values()):
        print(f"No runs found for {experiment.name} under {runs_root / experiment.name}")
        return None
    out_dir = results_root / experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {v: summarize_variant(r) for v, r in runs.items()}
    text = write_summary(experiment, runs, stats, comparisons(experiment, stats), out_dir)
    plot_learning_curves(experiment, runs, out_dir)
    plot_final_metrics(experiment, stats, out_dir)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="experiment name (configs/experiments/<name>.yaml) or a path")
    parser.add_argument("--all", action="store_true", help="analyze every experiment that has runs")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    exp_dir = REPO_ROOT / "configs" / "experiments"
    if args.all:
        paths = sorted(exp_dir.glob("*.yaml"))
    elif args.experiment:
        candidate = Path(args.experiment)
        paths = [candidate if candidate.suffix == ".yaml" else exp_dir / f"{args.experiment}.yaml"]
    else:
        parser.error("give an experiment name or --all")
    for path in paths:
        text = analyze(path, args.runs_root, args.results_root)
        if text:
            print(text)
            print(f"Saved to {args.results_root / load_experiment(path).name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
