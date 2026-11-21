"""YAML configs, command-line overrides and experiment grids.

An experiment file looks like::

    name: e1_mechanism
    base: ../base.yaml          # defaults, relative to this file
    seeds: [0, 1, 2, 3, 4]
    overrides:                  # applied to every variant
      env: {layout: open}
    grid:                       # variants = cartesian product of factor levels
      - name: sampler
        key: train.sampler      # each level sets this key...
        levels: {indep: independent, coupled: coupled}
      - name: method            # ...or, without `key`, each level is a dict of dotted keys
        levels:
          ippo: {train.critic: local}
    compare: {factor: sampler, baseline: indep, treatment: coupled}

Variant names join the level names, e.g. ``coupled-k2-short``.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from padp.env.config import EnvConfig
from padp.env.rewards import RewardConfig
from padp.marl.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(config: Dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = config
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def parse_sets(items: Optional[List[str]]) -> Dict[str, Any]:
    """``["train.lr=1e-4", "env.weights=[90]"]`` -> nested dict (values parsed as YAML)."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def _dataclass_from(cls, values: Dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**values)


def build_configs(config: Dict[str, Any]) -> Tuple[Any, TrainConfig]:
    """Environment config (EnvConfig, or LBFConfig when env.kind is "lbf") and TrainConfig."""
    env = dict(config.get("env", {}))
    train = _dataclass_from(TrainConfig, dict(config.get("train", {})))
    kind = env.pop("kind", "warehouse")
    lbf = env.pop("lbf", None)
    if kind == "lbf":
        from padp.env.lbf import LBFConfig

        return _dataclass_from(LBFConfig, dict(lbf or {})), train
    if kind != "warehouse":
        raise ValueError(f"env.kind={kind!r}; expected 'warehouse' or 'lbf'")
    env["reward"] = _dataclass_from(RewardConfig, env.get("reward", {}) or {})
    return _dataclass_from(EnvConfig, env), train


@dataclass
class Factor:
    name: str
    levels: Dict[str, Dict[str, Any]]  # level name -> {dotted key: value}


@dataclass
class Experiment:
    name: str
    description: str
    config: Dict[str, Any]
    factors: List[Factor] = field(default_factory=list)
    seeds: List[int] = field(default_factory=lambda: [0])
    compare: Optional[Dict[str, str]] = None
    path: Optional[Path] = None

    def variants(self) -> List[Tuple[str, Dict[str, Any]]]:
        if not self.factors:
            return [("default", {})]
        out = []
        for combo in itertools.product(*[list(f.levels.items()) for f in self.factors]):
            name = "-".join(level for level, _ in combo)
            settings: Dict[str, Any] = {}
            for _, assignments in combo:
                settings.update(assignments)
            out.append((name, settings))
        return out

    def variant_levels(self, variant: str) -> Dict[str, str]:
        if not self.factors:
            return {}
        return dict(zip([f.name for f in self.factors], variant.split("-")))

    def resolve(self, variant: str, seed: int, sets: Optional[List[str]] = None) -> Dict[str, Any]:
        lookup = dict(self.variants())
        if variant not in lookup:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(lookup)}")
        config = copy.deepcopy(self.config)
        for dotted, value in lookup[variant].items():
            set_dotted(config, dotted, copy.deepcopy(value))
        config = deep_merge(config, parse_sets(sets))
        set_dotted(config, "train.seed", seed)
        return config

    def run_dir(self, variant: str, seed: int, runs_root: Path = REPO_ROOT / "runs") -> Path:
        return Path(runs_root) / self.name / variant / f"seed{seed}"


def resolve_config(path: Path) -> Dict[str, Any]:
    """The env/train settings of a file, following its ``base:`` chain (base first, then overrides)."""
    path = Path(path).resolve()
    raw = load_yaml(path)
    config: Dict[str, Any] = resolve_config(path.parent / raw["base"]) if "base" in raw else {}
    config = deep_merge(config, {k: raw[k] for k in ("env", "train") if k in raw})
    return deep_merge(config, raw.get("overrides", {}))


def load_experiment(path: Path) -> Experiment:
    """Load an experiment file, or a plain config (one default variant, named after the file)."""
    path = Path(path)
    raw = load_yaml(path)
    config = resolve_config(path)
    factors = []
    for spec in raw.get("grid", []) or []:
        levels = spec["levels"]
        if not isinstance(levels, dict):
            levels = {str(v): v for v in levels}
        if "key" in spec:
            levels = {str(name): {spec["key"]: value} for name, value in levels.items()}
        for name in levels:
            if "-" in str(name):
                raise ValueError(f"level name {name!r} must not contain '-' (used to join variant names)")
        factors.append(Factor(spec["name"], {str(k): dict(v) for k, v in levels.items()}))
    return Experiment(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        config=config,
        factors=factors,
        seeds=list(raw.get("seeds", [0])),
        compare=raw.get("compare"),
        path=path,
    )
