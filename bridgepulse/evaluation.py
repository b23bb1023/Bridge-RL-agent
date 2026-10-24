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


def bootstrap_ci_clustered(x, clusters, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a CI that resamples whole clusters (e.g. all episodes played on one bridge network)."""
    x, clusters = np.asarray(x, dtype=np.float64), np.asarray(clusters)
    keys = np.unique(clusters)
    sums = np.array([x[clusters == k].sum() for k in keys])
    counts = np.array([(clusters == k).sum() for k in keys])
    idx = np.random.default_rng(seed).integers(0, keys.size, size=(n_boot, keys.size))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_difference_clustered(a: pd.DataFrame, b: pd.DataFrame, cluster: str, metric: str = "true_cost") -> tuple[float, float, float]:
    """Like paired_difference, but resampling whole clusters of episodes instead of single episodes."""
    m = a[["seed", cluster, metric]].merge(b[["seed", metric]], on="seed", suffixes=("_a", "_b"))
    return bootstrap_ci_clustered(m[f"{metric}_a"] - m[f"{metric}_b"], m[cluster])


def paired_difference_over_seeds(members: pd.DataFrame, reference: pd.DataFrame, metric: str = "true_cost",
                                 n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference of a multi-seed policy against a reference, with a CI that resamples both the trained
    models (members) and the test episodes. It covers the uncertainty of training another set of seeds, which an
    episode-only bootstrap of the seed-averaged policy does not."""
    table = members.pivot_table(index="member", columns="seed", values=metric)
    ref = reference.set_index("seed")[metric].reindex(table.columns).to_numpy()
    diff = table.to_numpy() - ref[None, :]
    rng = np.random.default_rng(seed)
    n_members, n_episodes = diff.shape
    means = np.array([diff[np.ix_(rng.integers(0, n_members, n_members), rng.integers(0, n_episodes, n_episodes))].mean()
                      for _ in range(n_boot)])
    return float(diff.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


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
