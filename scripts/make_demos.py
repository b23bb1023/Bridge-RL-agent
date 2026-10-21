"""Record demonstration episodes of the lookahead planner (baselines/rollout.py) for the DQN warm start.

The planner is slow (it simulates every candidate work order to the end of the horizon), so its
episodes are recorded once per setting and shared by all training seeds. One file per funding level
and information setting:

  python scripts/make_demos.py                                  # main setting
  python scripts/make_demos.py --budget-scale 0.5
  python scripts/make_demos.py --obs-mode oracle                # planner that sees the true condition
  python scripts/make_demos.py --no-inspection

Writes data/demos/<name>.npz with the observations, actions, per-bridge rewards and feasibility masks.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgepulse.config import load_env_config  # noqa: E402
from bridgepulse.env import BridgeNetworkEnv  # noqa: E402

DEMO_DIR = ROOT / "data" / "demos"
DEMO_SEEDS = 32_000_000


def demo_name(budget_scale: float = 1.0, obs_mode: str = "belief", allow_inspection: bool = True) -> str:
    return f"planner_budget{budget_scale:g}_{obs_mode}" + ("" if allow_inspection else "_noinsp")


def env_overrides(budget_scale: float, obs_mode: str, allow_inspection: bool) -> dict:
    return {"stress": {"budget_scale": budget_scale},
            "agent_interface": {"obs_mode": obs_mode, "allow_inspection": allow_inspection}}


def base_rule(budget_scale: float, obs_mode: str, allow_inspection: bool):
    """The rule the planner improves: the best tuned rule for this setting, chosen on validation episodes."""
    from baselines.heuristics import build
    from bridgepulse.evaluation import VALIDATION_SEEDS, evaluate

    table = yaml.safe_load((ROOT / "configs" / "heuristics.yaml").read_text())[f"budget_scale_{budget_scale:g}"]
    if obs_mode == "oracle":
        return "oracle_threshold", build("oracle_threshold", table["oracle_threshold"]["params"])
    env = BridgeNetworkEnv(load_env_config(), overrides=env_overrides(budget_scale, obs_mode, allow_inspection))
    scores = {k: evaluate(env, build(k, table[k]["params"]), VALIDATION_SEEDS)["true_cost"].mean()
              for k in table if k != "oracle_threshold"}
    name = min(scores, key=scores.get)
    return name, build(name, table[name]["params"])


def record(job: tuple) -> dict:
    from baselines.rollout import RolloutPlanner

    k, budget_scale, obs_mode, allow_inspection, rule_name, rollouts = job
    from baselines.heuristics import build
    table = yaml.safe_load((ROOT / "configs" / "heuristics.yaml").read_text())[f"budget_scale_{budget_scale:g}"]
    rule = build(rule_name, table[rule_name]["params"])
    env = BridgeNetworkEnv(load_env_config(), overrides=env_overrides(budget_scale, obs_mode, allow_inspection))
    planner = RolloutPlanner(rule, n_rollouts=rollouts, seed=DEMO_SEEDS + k)
    obs, _ = env.reset(seed=DEMO_SEEDS + k)
    observations, masks, actions, rewards = [obs], [env.action_masks()], [], []
    done = False
    while not done:
        a = planner(env, obs)
        obs, _, done, _, info = env.step(a)
        observations.append(obs)
        masks.append(env.action_masks())
        actions.append(a)
        rewards.append(info["reward_vector"])
    return {"obs": np.stack(observations).astype(np.float32), "mask": np.stack(masks), "action": np.array(actions, dtype=np.int16),
            "reward": np.stack(rewards).astype(np.float32), "true_cost": info["episode"]["true_cost"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--obs-mode", default="belief", choices=["belief", "oracle", "last_obs"])
    parser.add_argument("--no-inspection", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    allow = not args.no_inspection
    rule_name, _ = base_rule(args.budget_scale, args.obs_mode, allow)
    jobs = [(k, args.budget_scale, args.obs_mode, allow, rule_name, args.rollouts) for k in range(args.episodes)]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        eps = list(pool.map(record, jobs))
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    name = demo_name(args.budget_scale, args.obs_mode, allow)
    np.savez_compressed(DEMO_DIR / f"{name}.npz", **{k: np.stack([e[k] for e in eps]) for k in eps[0]},
                        base_rule=rule_name, rollouts=args.rollouts)
    cost = np.mean([e["true_cost"] for e in eps])
    print(f"{name}: {args.episodes} planner episodes on {rule_name}, mean cost ${cost / 1e6:.2f}M ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
