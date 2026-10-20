"""Grid-search every rule-based baseline under the calibrated funding and save the best thresholds.

Writes configs/heuristics.yaml. Use --budget-scale to tune for a different funding level
(the budget sweep stores one entry per scale).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.tuning import grid_search  # noqa: E402

OUT = ROOT / "configs" / "heuristics.yaml"
NAMES = ["condition_based", "risk_greedy", "entropy_triggered", "oracle_threshold"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    args = parser.parse_args()

    overrides = {"stress": {"budget_scale": args.budget_scale}}
    best = grid_search(NAMES, overrides, workers=args.workers)
    store = yaml.safe_load(OUT.read_text()) if OUT.exists() else {}
    key = f"budget_scale_{args.budget_scale:g}"
    store[key] = {name: {"params": v["params"], "tune_true_cost": round(v["tune_true_cost"], -3),
                         "grid_size": v["grid_size"]} for name, v in best.items()}
    OUT.write_text(yaml.safe_dump(store, sort_keys=False))
    for name, v in best.items():
        print(f"{name:18s} {v['params']}  tuning cost ${v['tune_true_cost'] / 1e6:8.1f}M  "
              f"(worst in grid ${v['tune_true_cost_worst'] / 1e6:8.1f}M, {v['grid_size']} configs)")


if __name__ == "__main__":
    main()
