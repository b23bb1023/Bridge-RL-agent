"""Train a maskable PPO agent on the same environment, with the same validation protocol as the DQN runs.

Example:
  python scripts/train_ppo.py --seed 0 --steps 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgepulse.config import load_env_config, load_yaml  # noqa: E402
from bridgepulse.env import BridgeNetworkEnv  # noqa: E402
from bridgepulse.evaluation import VALIDATION_SEEDS, evaluate  # noqa: E402


def ppo_policy(model: MaskablePPO):
    def act(env, obs):
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(action)
    return act


class Validation(BaseCallback):
    def __init__(self, env: BridgeNetworkEnv, every: int, out_dir: Path, quiet: bool) -> None:
        super().__init__()
        self.val_env, self.every, self.out_dir, self.quiet = env, every, out_dir, quiet
        self.next_eval = 0
        self.rows: list[dict] = []
        self.best = np.inf
        self.recent: list[float] = []
        self.start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_summary" in info:
                self.recent.append(info["episode_summary"]["true_cost"])
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval += self.every
        return True

    def evaluate(self) -> None:
        df = evaluate(self.val_env, ppo_policy(self.model), VALIDATION_SEEDS)
        row = {
            "step": self.num_timesteps,
            "train_true_cost": float(np.mean(self.recent)) if self.recent else np.nan,
            "val_true_cost": float(df["true_cost"].mean()),
            "val_closure_months": float(df["closure_months"].mean()),
        }
        self.recent = []
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out_dir / "log.csv", index=False)
        self.model.save(self.out_dir / "last")
        if row["val_true_cost"] < self.best:
            self.best = row["val_true_cost"]
            self.model.save(self.out_dir / "best")
        if not self.quiet:
            print(f"[ppo] step {row['step']:>8d}  val ${row['val_true_cost'] / 1e6:7.1f}M  ({time.time() - self.start:.0f}s)", flush=True)


class EpisodeSummary(BridgeNetworkEnv):
    """Copies the end-of-episode summary under a key that survives vectorised auto-reset."""

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if term or trunc:
            info["episode_summary"] = info["episode"]
            del info["episode"]
        return obs, r, term, trunc, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    spec = load_yaml("configs/ppo.yaml")
    hp = dict(spec["ppo"])
    total = args.steps or int(hp.pop("total_steps"))
    hp.pop("total_steps", None)
    n_envs = int(hp.pop("n_envs"))
    every = int(hp.pop("eval_every"))
    net_arch = hp.pop("net_arch")

    out_dir = ROOT / args.runs_dir / "ppo" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"seed": args.seed, "ppo": spec["ppo"], "steps": total}, indent=2))

    cfg = load_env_config()
    vec = DummyVecEnv([lambda: EpisodeSummary(cfg) for _ in range(n_envs)])
    vec.seed(20_000_000 + args.seed * 100_000)
    model = MaskablePPO("MlpPolicy", vec, seed=args.seed, policy_kwargs={"net_arch": net_arch}, verbose=0, **hp)
    callback = Validation(BridgeNetworkEnv(cfg), every, out_dir, args.quiet)
    model.learn(total_timesteps=total, callback=callback)
    callback.evaluate()


if __name__ == "__main__":
    main()
