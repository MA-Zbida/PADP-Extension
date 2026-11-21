"""Build the environment that matches an environment config."""
from __future__ import annotations

from padp.env.lbf import LBFConfig, LBFEnv
from padp.env.warehouse import WarehouseEnv


def make_env(cfg):
    """WarehouseEnv for an EnvConfig, LBFEnv (Level-Based Foraging) for an LBFConfig."""
    if isinstance(cfg, LBFConfig):
        return LBFEnv(cfg)
    return WarehouseEnv(cfg)
