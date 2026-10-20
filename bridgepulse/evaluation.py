"""Policy evaluation with common random numbers, bootstrap confidence intervals and paired comparisons.

A policy is any callable `policy(env, obs) -> action`. Seed ranges are disjoint so that tuning,
validation and final testing never reuse the same episodes.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from bridgepulse.env import BridgeNetworkEnv

Policy = Callable[[BridgeNetworkEnv, np.ndarray], int]

TUNE_SEEDS = range(700_000, 700_064)
VALIDATION_SEEDS = range(500_000, 500_040)
TEST_SEEDS = range(900_000, 900_200)


def run_episode(env: BridgeNetworkEnv, policy: Policy, seed: int) -> dict:
    obs, _ = env.reset(seed=int(seed))
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(env, obs))
        done = terminated or truncated
    out = dict(info["episode"])
    out["seed"] = int(seed)
    return out


def evaluate(env: BridgeNetworkEnv, policy: Policy, seeds: Iterable[int]) -> pd.DataFrame:
    return pd.DataFrame([run_episode(env, policy, s) for s in seeds])


def bootstrap_ci(x, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, metric: str = "true_cost") -> tuple[float, float, float]:
    """Mean of (a - b) over episodes matched by seed, with a bootstrap CI."""
    m = a[["seed", metric]].merge(b[["seed", metric]], on="seed", suffixes=("_a", "_b"))
    return bootstrap_ci(m[f"{metric}_a"] - m[f"{metric}_b"])


def summarise(results: dict[str, pd.DataFrame], metrics=("true_cost", "agency", "closure", "risk", "salvage",
                                                        "closure_months", "inspections", "minor", "major")) -> pd.DataFrame:
    rows = []
    for name, df in results.items():
        row = {"policy": name, "episodes": len(df)}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(df[m])
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)
