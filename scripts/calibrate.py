"""Calibrate yearly maintenance funding, then report sanity checks on the calibrated problem.

1. Tune the rule-based policies with unlimited money; record the best cost C_inf and its yearly spend U.
2. Sufficiency level F*: the lowest yearly funding (searched over multiples of U) at which the best
   tuned rule stays within 10% of C_inf.
3. Funding = fraction_of_need x F*. With half the money needed to keep every bridge in service,
   choosing what to fix, when, and what to save for decides the outcome.

The result is written to configs/calibration.yaml, which load_env_config merges automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.heuristics import build  # noqa: E402
from baselines.tuning import grid_search  # noqa: E402
from bridgepulse.config import load_env_config  # noqa: E402
from bridgepulse.env import BridgeNetworkEnv  # noqa: E402
from bridgepulse.evaluation import TUNE_SEEDS, evaluate  # noqa: E402

CALIBRATION_FILE = ROOT / "configs" / "calibration.yaml"
RULES = ["condition_based", "risk_greedy", "entropy_triggered"]
MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def best_cost(funding: float | None, workers: int) -> tuple[float, str, dict, float]:
    best = grid_search(RULES, {"budget": {"annual": funding}}, workers=workers)
    name = min(best, key=lambda k: best[k]["tune_true_cost"])
    return best[name]["tune_true_cost"], name, best[name]["params"], best[name]["tune_agency"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()

    if CALIBRATION_FILE.exists():
        CALIBRATION_FILE.unlink()
    cfg = load_env_config()
    years = cfg["horizon_months"] / 12

    c_inf, leader, params, spend = best_cost(None, args.workers)
    yearly_spend = spend / years
    print(f"unlimited money: best rule {leader} {params}  cost ${c_inf / 1e6:.1f}M, spend ${yearly_spend / 1e6:.2f}M/yr")

    sufficiency = None
    for m in MULTIPLES:
        cost, name, _, _ = best_cost(m * yearly_spend, args.workers)
        print(f"  funding {m:>4.1f} x U = ${m * yearly_spend / 1e6:5.2f}M/yr -> best rule cost ${cost / 1e6:8.1f}M ({name})")
        if cost <= (1 + args.tolerance) * c_inf:
            sufficiency = m * yearly_spend
            break
    if sufficiency is None:
        sufficiency = MULTIPLES[-1] * yearly_spend
    funding = cfg["budget"]["fraction_of_need"] * sufficiency
    CALIBRATION_FILE.write_text(yaml.safe_dump(
        {"budget": {"annual": round(float(funding), -3)},
         "calibration": {"unlimited_cost": round(c_inf, -3), "unlimited_yearly_spend": round(yearly_spend, -3),
                         "sufficiency_funding": round(sufficiency, -3)}}, sort_keys=False))
    print(f"sufficiency F* = ${sufficiency / 1e6:.2f}M/yr -> funding ${funding / 1e6:.2f}M/yr")

    # Sanity checks on the calibrated problem.
    env = BridgeNetworkEnv(load_env_config())
    seeds = list(TUNE_SEEDS)
    reached = []
    for s in seeds[:32]:
        env.reset(seed=s)
        hit = np.zeros(env.n, dtype=bool)
        for _ in range(env.T):
            env.step(0)
            hit |= env.state == 4
        reached.append(hit.sum())
    nothing = evaluate(env, build("do_nothing"), seeds)
    tuned = grid_search(RULES + ["oracle_threshold"], workers=args.workers)
    rule = min(RULES, key=lambda k: tuned[k]["tune_true_cost"])
    print(f"do nothing: cost ${nothing['true_cost'].mean() / 1e6:.1f}M, bridges reaching Critical {np.mean(reached):.1f}/10")
    print(f"best belief rule at calibrated funding: {rule} ${tuned[rule]['tune_true_cost'] / 1e6:.1f}M")
    print(f"oracle threshold rule: ${tuned['oracle_threshold']['tune_true_cost'] / 1e6:.1f}M")
    print(f"largest single major repair: ${env.costs.major.max() / 1e6:.2f}M = "
          f"{env.costs.major.max() / funding:.1f} years of funding")


if __name__ == "__main__":
    main()
