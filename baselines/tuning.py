"""Grid search over heuristic thresholds on dedicated tuning episodes."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from baselines.heuristics import GRIDS, build
from bridgepulse.config import load_env_config
from bridgepulse.evaluation import TUNE_SEEDS, evaluate
from bridgepulse.env import BridgeNetworkEnv


def _score(job) -> tuple[str, dict, float, float]:
    name, params, overrides, seeds = job
    env = BridgeNetworkEnv(load_env_config(), overrides=overrides)
    df = evaluate(env, build(name, params), seeds)
    return name, params, float(df["true_cost"].mean()), float(df["agency"].mean())


def grid_search(names, overrides: dict | None = None, seeds=TUNE_SEEDS, workers: int = 4) -> dict[str, dict]:
    """Return the best parameters per heuristic, with their tuning cost and spend."""
    jobs = []
    for name in names:
        grid = GRIDS.get(name, {})
        keys = list(grid)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]
        jobs += [(name, combo, overrides or {}, list(seeds)) for combo in combos]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score, jobs))
    best: dict[str, dict] = {}
    for name, params, cost, spend in results:
        if name not in best or cost < best[name]["tune_true_cost"]:
            best[name] = {"params": params, "tune_true_cost": cost, "tune_agency": spend}
    for name in best:
        best[name]["grid_size"] = sum(1 for r in results if r[0] == name)
        best[name]["tune_true_cost_worst"] = float(np.max([r[2] for r in results if r[0] == name]))
    return best
