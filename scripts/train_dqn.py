"""Train a DQN variant (configs/dqn.yaml) on the bridge network.

Example:
  python scripts/train_dqn.py --variant main --seed 0
  python scripts/train_dqn.py --variant no_demos --seed 1
  python scripts/train_dqn.py --variant main --seed 0 --budget-scale 0.5 --tag budget0.5

Writes runs/<variant>[-<tag>]/seed<k>/: log.csv (training + validation curve), best.pt (lowest
validation cost), last.pt, config.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from agents.replay import NStepCollector, PrioritizedReplayBuffer, ReplayBuffer  # noqa: E402
from baselines.heuristics import build as build_heuristic  # noqa: E402
from bridgepulse.config import deep_update, load_env_config, load_yaml  # noqa: E402
from bridgepulse.env import BridgeNetworkEnv  # noqa: E402
from bridgepulse.evaluation import VALIDATION_SEEDS, evaluate  # noqa: E402
from scripts.make_demos import DEMO_DIR, demo_name  # noqa: E402


def training_seed(run_seed: int, episode: int) -> int:
    return 10_000_000 + run_seed * 100_000 + episode


def build(variant: str, steps: int | None, env_extra: dict | None = None, agent_extra: dict | None = None):
    spec = load_yaml("configs/dqn.yaml")
    v = spec["variants"][variant]
    agent_kwargs = {**spec["agent"], **v.get("agent", {}), **(agent_extra or {})}
    if steps is not None:
        agent_kwargs["total_steps"] = steps
    defaults = DQNConfig()
    for key, value in agent_kwargs.items():
        default = getattr(defaults, key, None)
        if isinstance(default, bool):
            agent_kwargs[key] = value if isinstance(value, bool) else str(value).lower() == "true"
        elif isinstance(default, (int, float)):
            agent_kwargs[key] = type(default)(float(value))
    env_overrides = deep_update(v.get("env", {}), env_extra)
    return DQNConfig(**agent_kwargs), env_overrides


def validate(agent: DQNAgent, env: BridgeNetworkEnv) -> dict:
    df = evaluate(env, agent.policy(), VALIDATION_SEEDS)
    return {"val_true_cost": float(df["true_cost"].mean()), "val_closure_months": float(df["closure_months"].mean())}


def load_planner_demonstrations(cfg: DQNConfig, env: BridgeNetworkEnv, budget_scale: float, reward_dim: int, seed: int) -> ReplayBuffer:
    """Recorded episodes of the lookahead planner (scripts/make_demos.py), stored with n-step returns and masks."""
    name = demo_name(budget_scale, env.obs_mode, env.allow_inspection)
    path = DEMO_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; record it first with scripts/make_demos.py")
    data = np.load(path)
    episodes = min(cfg.demo_episodes, len(data["action"]))
    print(f"demonstrations: {episodes} planner episodes from {path.name} (planner on {data['base_rule']})", flush=True)
    demos = ReplayBuffer(episodes * env.T, env.obs_dim, env.num_actions, seed=seed + 1, reward_dim=reward_dim)
    collector = NStepCollector(cfg.n_step, cfg.gamma, demos)
    for e in range(episodes):
        obs, mask = data["obs"][e].astype(np.float32), data["mask"][e]
        T = len(data["action"][e])
        for t in range(T):
            r = data["reward"][e, t]
            collector.push(obs[t], int(data["action"][e, t]), r if reward_dim > 1 else float(r.sum()), obs[t + 1],
                           mask[t + 1], t == T - 1, mask[t])
    return demos


def collect_demonstrations(cfg: DQNConfig, env: BridgeNetworkEnv, budget_scale: float, reward_dim: int, decomposed: bool, seed: int) -> ReplayBuffer:
    """Episodes of a tuned rule, stored with n-step returns and feasibility masks."""
    if cfg.demo_policy == "planner":
        return load_planner_demonstrations(cfg, env, budget_scale, reward_dim, seed)
    table = load_yaml("configs/heuristics.yaml")[f"budget_scale_{budget_scale:g}"]
    name = cfg.demo_policy
    if name == "best":   # the tuned belief-based rule with the lowest validation cost
        scores = {k: evaluate(env, build_heuristic(k, table[k]["params"]), VALIDATION_SEEDS)["true_cost"].mean()
                  for k in table if k != "oracle_threshold"}
        name = min(scores, key=scores.get)
    rule = build_heuristic(name, table[name]["params"])
    print(f"demonstrations from rule: {name}", flush=True)
    demos = ReplayBuffer(cfg.demo_episodes * env.T, env.obs_dim, env.num_actions, seed=seed + 1, reward_dim=reward_dim)
    collector = NStepCollector(cfg.n_step, cfg.gamma, demos)
    for k in range(cfg.demo_episodes):
        obs, _ = env.reset(seed=30_000_000 + seed * 100_000 + k)
        done = False
        while not done:
            mask = env.action_masks()
            a = rule(env, obs)
            nxt, r, done, _, info = env.step(a)
            collector.push(obs, a, info["reward_vector"] if decomposed else r, nxt, env.action_masks(), done, mask)
            obs = nxt
    return demos


def mixed_batch(agent_buffer: ReplayBuffer, demos: ReplayBuffer, batch_size: int, ratio: float) -> dict:
    n_demo = int(round(batch_size * ratio))
    a, d = agent_buffer.sample(batch_size - n_demo), demos.sample(n_demo)
    batch = {k: np.concatenate([a[k], d[k]]) for k in a}
    batch["is_demo"] = np.concatenate([np.zeros(batch_size - n_demo), np.ones(n_demo)])
    return batch


def train(variant: str, seed: int, steps: int | None, out_dir: Path, env_extra: dict | None = None,
          quiet: bool = False, agent_extra: dict | None = None) -> Path:
    torch.set_num_threads(1)
    cfg, env_overrides = build(variant, steps, env_extra, agent_extra)
    base = load_env_config()
    envs = [BridgeNetworkEnv(base, overrides=env_overrides) for _ in range(cfg.n_envs)]
    val_env = BridgeNetworkEnv(base, overrides=env_overrides)
    probe = envs[0]
    costs = base["costs"]
    cfg.cost_ratios = (costs["inspection_per_m2"] / costs["major_per_m2"], costs["minor_per_m2"] / costs["major_per_m2"], 1.0)
    decomposed = cfg.arch == "decomposed"
    agent = DQNAgent(probe.obs_dim, probe.num_actions, probe.n, cfg, seed=seed)
    capacity = min(cfg.buffer_size, cfg.total_steps)
    reward_dim = probe.n if decomposed else 1
    if cfg.per:
        buffer = PrioritizedReplayBuffer(capacity, probe.obs_dim, probe.num_actions, alpha=cfg.per_alpha, seed=seed, reward_dim=reward_dim)
    else:
        buffer = ReplayBuffer(capacity, probe.obs_dim, probe.num_actions, seed=seed, reward_dim=reward_dim)
    collectors = [NStepCollector(cfg.n_step, cfg.gamma, buffer) for _ in envs]

    demos = None
    if cfg.demo_policy:
        if cfg.per:
            raise ValueError("demonstrations are used with uniform replay only")
        budget_scale = float((env_overrides.get("stress") or {}).get("budget_scale", 1.0))
        demos = collect_demonstrations(cfg, probe, budget_scale, reward_dim, decomposed, seed)
        for _ in range(cfg.demo_pretrain_updates):
            batch = demos.sample(cfg.batch_size)
            batch["is_demo"] = np.ones(cfg.batch_size)
            agent.update(batch, margin_weight=cfg.margin_weight)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"variant": variant, "seed": seed, "agent": asdict(cfg), "env_overrides": env_overrides}, indent=2))

    log, losses, recent_cost = [], [], []
    episodes_started = 0

    def next_seed() -> int:
        nonlocal episodes_started
        episodes_started += 1
        return training_seed(seed, episodes_started - 1)

    obs = np.stack([e.reset(seed=next_seed())[0] for e in envs])
    best = validate(agent, val_env)
    log.append({"step": 0, "episodes": 0, "epsilon": cfg.eps_start, "loss": np.nan, "train_true_cost": np.nan, **best})
    best_cost = best["val_true_cost"]
    agent.save(out_dir / "best.pt", {"step": 0, **best})
    start = time.time()

    step, update_credit, next_eval, finished = 0, 0.0, cfg.eval_every, 0
    while step < cfg.total_steps:
        eps = cfg.epsilon(step)
        masks = np.stack([e.action_masks() for e in envs])
        actions = agent.act_batch(obs, masks, eps)
        for k, e in enumerate(envs):
            nxt, r, terminated, truncated, info = e.step(int(actions[k]))
            done = terminated or truncated
            collectors[k].push(obs[k].copy(), int(actions[k]), info["reward_vector"] if decomposed else r, nxt, e.action_masks(), done)
            if done:
                recent_cost.append(info["episode"]["true_cost"])
                finished += 1
                nxt, _ = e.reset(seed=next_seed())
            obs[k] = nxt
        step += len(envs)

        if step >= cfg.learning_starts:
            update_credit += len(envs) / cfg.train_every
            while update_credit >= 1.0:
                if cfg.per:
                    beta = cfg.per_beta_start + (1.0 - cfg.per_beta_start) * min(1.0, step / cfg.total_steps)
                    batch = buffer.sample(cfg.batch_size, beta)
                    losses.append(agent.update(batch))
                    buffer.update_priorities(batch["indices"], agent.last_td_error)
                elif demos is not None:
                    fade = max(cfg.margin_floor, 1.0 - step / max(1, cfg.margin_decay_steps))
                    batch = mixed_batch(buffer, demos, cfg.batch_size, cfg.demo_ratio)
                    losses.append(agent.update(batch, margin_weight=cfg.margin_weight * fade))
                else:
                    losses.append(agent.update(buffer.sample(cfg.batch_size)))
                update_credit -= 1.0

        if step >= next_eval or step >= cfg.total_steps:
            next_eval += cfg.eval_every
            val = validate(agent, val_env)
            row = {
                "step": step, "episodes": finished, "epsilon": eps,
                "loss": float(np.mean(losses)) if losses else np.nan,
                "train_true_cost": float(np.mean(recent_cost)) if recent_cost else np.nan,
                **val,
            }
            log.append(row)
            losses, recent_cost = [], []
            agent.save(out_dir / "last.pt", {"step": step, **val})
            if val["val_true_cost"] < best_cost:
                best_cost = val["val_true_cost"]
                agent.save(out_dir / "best.pt", {"step": step, **val})
            if not quiet:
                print(f"[{variant} s{seed}] step {step:>8d}  eps {eps:.3f}  loss {row['loss']:.4f}  "
                      f"train ${row['train_true_cost'] / 1e6:7.1f}M  val ${val['val_true_cost'] / 1e6:7.1f}M  "
                      f"({time.time() - start:.0f}s)", flush=True)
            pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override agent hyperparameters")
    parser.add_argument("--env-set", nargs="*", default=[], metavar="A.B=VALUE", help="override environment settings")
    args = parser.parse_args()

    env_extra: dict = {}
    if args.budget_scale is not None:
        env_extra["stress"] = {"budget_scale": args.budget_scale}
    for item in args.env_set:
        path, value = item.split("=", 1)
        node = env_extra
        *parents, leaf = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    agent_extra = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    name = args.variant + (f"-{args.tag}" if args.tag else "")
    train(args.variant, args.seed, args.steps, ROOT / args.runs_dir / name / f"seed{args.seed}", env_extra, args.quiet, agent_extra)


if __name__ == "__main__":
    main()
