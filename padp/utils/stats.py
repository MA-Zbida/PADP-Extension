"""Small statistics helpers for reporting RL results across seeds (Agarwal et al., 2021)."""
from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% of values (robust to outlier seeds)."""
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(x) == 0:
        return math.nan
    return float(sp_stats.trim_mean(x, 0.25))


def bootstrap_ci(values: Sequence[float], stat: Callable = iqm, n: int = 2000, level: float = 0.95,
                 seed: int = 0) -> Tuple[float, float]:
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(x) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    alpha = (1 - level) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def steps_to_threshold(steps: Sequence[float], values: Sequence[float], threshold: float) -> float:
    for s, v in zip(steps, values):
        if not math.isnan(v) and v >= threshold:
            return float(s)
    return math.nan


def normalized_auc(steps: Sequence[float], values: Sequence[float]) -> float:
    """Area under a learning curve divided by its step range (= time-averaged value)."""
    s = np.asarray(steps, dtype=float)
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    if len(s) < 2 or s[-1] == s[0]:
        return math.nan
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 only has trapz
    return float(trapezoid(v, s) / (s[-1] - s[0]))


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided p-value; NaN if either group has fewer than two values."""
    a = [v for v in a if not math.isnan(v)]
    b = [v for v in b if not math.isnan(v)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    if len(set(a) | set(b)) == 1:
        return 1.0
    return float(sp_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
