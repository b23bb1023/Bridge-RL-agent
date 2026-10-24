"""Final evaluation on held-out test episodes with common random numbers.

For every policy and scenario, the same 200 test seeds are used, so differences between
policies are paired episode by episode. RL variants are evaluated for every training seed and
averaged per episode.

  python scripts/evaluate.py                 # nominal comparison + information ladder
  python scripts/evaluate.py --ood           # adds out-of-distribution scenarios
  python scripts/evaluate.py --budget-sweep  # funding levels 0.5x .. 2x

Outputs go to results/: episodes/*.csv (per episode), summary_*.csv (means with 95% CIs),
paired_*.csv (differences against the main agent), timeline.json (one annotated episode).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from baselines.heuristics import build  # noqa: E402
from bridgepulse import belief as B  # noqa: E402
from bridgepulse.config import deep_update, load_env_config  # noqa: E402
from bridgepulse.env import BridgeNetworkEnv  # noqa: E402
from bridgepulse.evaluation import (TEST_SEEDS, bootstrap_ci, bootstrap_ci_clustered, evaluate,  # noqa: E402
                                   paired_difference, paired_difference_clustered, paired_difference_over_seeds, summarise)
from bridgepulse.network import network_from_frame  # noqa: E402

RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"
MAIN = "DQN (ours)"
ENSEMBLE = "DQN (ours, 5-seed ensemble)"
PLANNER = "Lookahead planner (teacher)"

DQN_LABELS = {
    "main": MAIN,
    "no_demos": "DQN, no demonstrations",
    "rule_demos": "DQN, rule demonstrations",
    "pretrain_only": "DQN, offline pre-training only",
    "shared": "DQN, shared network",
    "flat": "DQN, flat network",
    "vanilla": "Vanilla DQN",
    "oracle": "DQN, true condition",
    "last_obs": "DQN, last inspection only",
    "no_inspection": "DQN, no extra inspections",
}
HEURISTIC_LABELS = {
    "do_nothing": "Do nothing",
    "condition_based": "Rule: condition-based",
    "risk_greedy": "Rule: risk-greedy",
    "entropy_triggered": "Rule: entropy-triggered",
    "oracle_threshold": "Rule: oracle threshold",
}
SCENARIOS = {
    "nominal": {},
    # Sensitivity to the unmaintained-deterioration assumption (factor 3 nominally): 1.5 and 5.
    "slower_decay": {"stress": {"physics_decay_scale": 0.5, "filter_decay_scale": 0.5}},
    "faster_decay": {"stress": {"physics_decay_scale": 5 / 3, "filter_decay_scale": 5 / 3}},
    # Deterioration 50% faster than the agent's model assumes (model misspecification).
    "unmodelled_decay": {"stress": {"physics_decay_scale": 1.5, "filter_decay_scale": 1.0}},
    "budget_cut": {"stress": {"budget_scale": 0.75}},
    # Routine inspections as variable as different inspectors' ratings, instead of the fitted consistency.
    "noisy_routine": {"inspection": {"routine_sigma": 0.65}},
}


# ---------------------------------------------------------------------------- policies
def ppo_policy(path: Path):
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(path, device="cpu")

    def act(env, obs):
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(action)

    return act


def make_policy(spec: dict):
    kind = spec["kind"]
    if kind == "heuristic":
        return build(spec["name"], spec.get("params"))
    if kind == "planner":
        from baselines.rollout import RolloutPlanner
        return RolloutPlanner(build(spec["name"], spec["params"]), n_rollouts=spec["rollouts"], z=spec["z"], seed=spec["seed"])
    if kind == "dqn":
        return DQNAgent.load(spec["path"]).policy()
    if kind == "dqn_ensemble":
        agents = [DQNAgent.load(p) for p in spec["paths"]]

        def act(env, obs):
            q = np.mean([a.q_values(obs)[0] for a in agents], axis=0)
            return int(np.argmax(np.where(env.action_masks(), q, -1e9)))

        return act
    if kind == "dqn_random":
        agent = DQNAgent(103, 31, 10, DQNConfig(arch="decomposed"), seed=spec.get("seed", 0))
        return agent.policy()
    if kind == "ppo":
        return ppo_policy(Path(spec["path"]))
    raise ValueError(kind)


def code_and_data_fingerprint() -> str:
    """Hash of everything besides the job itself that determines an episode: the environment and policy code,
    the configuration files, the bridge network and the fitted deterioration model."""
    import hashlib
    files = sorted([*(ROOT / "bridgepulse").glob("*.py"), *(ROOT / "baselines").glob("*.py"), *(ROOT / "configs").glob("*.yaml"),
                    ROOT / "data" / "processed" / "network_10.csv", ROOT / "data" / "processed" / "transition_fit.json"])
    h = hashlib.sha1()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _cache_path(job: dict) -> Path | None:
    """Slow planner episodes are cached on disk, keyed by everything that determines them."""
    if job["policy"]["kind"] != "planner" or job.get("network"):
        return None
    import hashlib
    resolved = deep_update(load_env_config(), deep_update(job["env_overrides"], job["scenario_overrides"]))
    key = json.dumps([job["policy"], resolved, job["seeds"], job["fingerprint"]], sort_keys=True, default=str)
    return Path(job["results_dir"]) / "episodes" / "planner_cache" / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.csv"


def _job(job: dict) -> pd.DataFrame:
    torch.set_num_threads(1)
    cache = _cache_path(job)
    if cache is not None and cache.exists():
        df = pd.read_csv(cache)
        df["policy"], df["scenario"], df["member"] = job["label"], job["scenario"], job.get("member", 0)
        df["network_id"] = job.get("network_id", -1)
        return df
    cfg = load_env_config()
    network = network_from_frame(pd.DataFrame(job["network"])) if job.get("network") else None
    env = BridgeNetworkEnv(cfg, network=network, overrides=deep_update(job["env_overrides"], job["scenario_overrides"]))
    df = evaluate(env, make_policy(job["policy"]), job["seeds"])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    df["policy"], df["scenario"], df["member"] = job["label"], job["scenario"], job.get("member", 0)
    df["network_id"] = job.get("network_id", -1)
    return df


# ---------------------------------------------------------------------------- discovery
def heuristic_specs(budget_scale: float = 1.0) -> list[tuple[str, dict, dict]]:
    tuned = yaml.safe_load((ROOT / "configs" / "heuristics.yaml").read_text())[f"budget_scale_{budget_scale:g}"]
    out = [(HEURISTIC_LABELS["do_nothing"], {"kind": "heuristic", "name": "do_nothing"}, {})]
    for name in ("condition_based", "risk_greedy", "entropy_triggered", "oracle_threshold"):
        env = {"agent_interface": {"obs_mode": "oracle"}} if name == "oracle_threshold" else {}
        out.append((HEURISTIC_LABELS[name], {"kind": "heuristic", "name": name, "params": tuned[name]["params"]}, env))
    return out


def planner_spec(budget_scale: float = 1.0, rollouts: int = 8) -> tuple[str, list[dict], dict]:
    """The lookahead planner on the rule it improves (the same rule its demonstrations were recorded with)."""
    from scripts.make_demos import base_rule
    name, rule = base_rule(budget_scale, "belief", True)
    spec = {"kind": "planner", "name": name, "params": rule.params, "rollouts": rollouts, "z": 1.0, "seed": 0, "chunks": 20}
    return PLANNER, [spec], {}


def _env_overrides_for(run: str) -> dict:
    """The environment settings a DQN variant was trained with: from its run folder, its archived log, or the config."""
    for folder in (RUNS / run, ROOT / "results" / "training_logs" / run):
        configs = sorted(folder.glob("seed*/config.json"))
        if configs:
            return json.loads(configs[0].read_text()).get("env_overrides", {})
    from scripts.train_dqn import build
    return build(run.split("-")[0], None)[1]


def rl_specs(run_names: list[str], allow_missing: bool = False) -> list[tuple[str, list[dict], dict]]:
    """Trained agents to evaluate. Each run is looked up in runs/<run>/seed*/ (fresh training) and, failing that,
    in the shipped models/ folder. A requested run with no models stops the evaluation unless allow_missing."""
    out, missing = [], []
    for run in run_names:
        variant = run.split("-")[0]
        if variant == "ppo":
            paths = [s / "best.zip" for s in sorted((RUNS / run).glob("seed*")) if (s / "best.zip").exists()]
            paths = paths or sorted(MODELS.glob("ppo_seed*.zip"))
            members = [{"kind": "ppo", "path": str(p)} for p in paths]
            env, label = {}, "Maskable PPO"
        else:
            paths = [s / "best.pt" for s in sorted((RUNS / run).glob("seed*")) if (s / "best.pt").exists()]
            paths = paths or sorted(MODELS.glob(f"dqn_{run}_seed*.pt"))
            members = [{"kind": "dqn", "path": str(p)} for p in paths]
            env, label = (_env_overrides_for(run) if members else {}), DQN_LABELS.get(variant, variant)
            if variant == "main" and len(members) > 1:
                out.append((ENSEMBLE, [{"kind": "dqn_ensemble", "paths": [m["path"] for m in members]}], env))
        if not members:
            missing.append(run)
            continue
        print(f"  {label}: {len(members)} trained model(s) from {Path(members[0]['path']).parent.relative_to(ROOT)}", flush=True)
        out.append((label, members, env))
    if missing and not allow_missing:
        raise SystemExit(f"no trained models found for {missing} (looked in {RUNS.relative_to(ROOT)}/ and models/). "
                         "Train them first, or leave them out with --runs.")
    return out


MEMBER_ROWS: list[pd.DataFrame] = []   # every evaluated episode of every trained model, before averaging seeds


def run_jobs(jobs: list[dict], workers: int) -> pd.DataFrame:
    fingerprint = code_and_data_fingerprint()
    for job in jobs:
        job["fingerprint"], job["results_dir"] = fingerprint, str(RESULTS)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(_job, jobs))
    df = pd.concat(frames, ignore_index=True)
    MEMBER_ROWS.append(df)
    # Average RL training seeds per test episode so every policy has one row per episode.
    num = df.select_dtypes("number").columns.drop(["seed", "member", "network_id"], errors="ignore")
    return df.groupby(["scenario", "policy", "seed", "network_id"], as_index=False)[list(num)].mean()


def jobs_for(policies, scenario: str, overrides: dict, seeds, network=None, network_id: int = -1) -> list[dict]:
    jobs = []
    for label, members, env in policies:
        members = members if isinstance(members, list) else [members]
        for k, spec in enumerate(members):
            parts = np.array_split(np.asarray(seeds), min(spec.get("chunks", 1), len(seeds)))
            for c, part in enumerate(parts):   # slow policies are spread over several workers
                job_spec = {**spec, "seed": spec.get("seed", 0) + c} if spec["kind"] == "planner" else spec
                jobs.append({"label": label, "policy": job_spec, "env_overrides": env, "scenario": scenario,
                             "scenario_overrides": overrides, "seeds": [int(x) for x in part], "member": k, "network": network,
                             "network_id": network_id})
    return jobs


def unseen_network_jobs(policies, seeds, n_networks: int = 20) -> list[dict]:
    """Networks drawn from the held-out pool (3 arterial, 4 collector, 3 local), funding scaled to their repair costs."""
    n_networks = min(n_networks, len(seeds))   # every network needs at least one episode
    pool = pd.read_csv(ROOT / "data" / "processed" / "pool.csv")
    base_cfg = load_env_config()
    base_env = BridgeNetworkEnv(base_cfg)
    per = max(1, len(seeds) // n_networks)
    rng = np.random.default_rng(12345)
    jobs = []
    for j in range(n_networks):
        parts = [pool[pool["archetype"] == a].sample(n=k, random_state=int(rng.integers(1 << 31)))
                 for a, k in (("arterial", 3), ("collector", 4), ("local", 3))]
        frame = pd.concat(parts, ignore_index=True)
        net_env = BridgeNetworkEnv(base_cfg, network=network_from_frame(frame))
        ratio = net_env.costs.major.sum() / base_env.costs.major.sum()
        overrides = {"budget": {"annual": base_cfg["budget"]["annual"] * ratio}}
        jobs += jobs_for(policies, "unseen_networks", overrides, seeds[j * per:(j + 1) * per], network=frame.to_dict("list"),
                         network_id=j)
    return jobs


def write_summary(df: pd.DataFrame, name: str, reference: str = MAIN) -> pd.DataFrame:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for scenario, part in df.groupby("scenario"):
        res = {p: g.reset_index(drop=True) for p, g in part.groupby("policy")}
        summary = summarise(res)
        summary.insert(0, "scenario", scenario)
        # Episodes played on the same sampled network are not independent: resample whole networks there.
        clustered = (part["network_id"] >= 0).any()
        if clustered:
            for p, g in res.items():
                _, lo, hi = bootstrap_ci_clustered(g["true_cost"], g["network_id"])
                summary.loc[summary["policy"] == p, ["true_cost_lo", "true_cost_hi"]] = [lo, hi]
        summary["ci_resamples"] = "networks" if clustered else "episodes"
        if reference in res:
            for p in res:
                if clustered:
                    d, lo, hi = paired_difference_clustered(res[p], res[reference], "network_id")
                else:
                    d, lo, hi = paired_difference(res[p], res[reference])
                summary.loc[summary["policy"] == p, ["diff_vs_main", "diff_lo", "diff_hi"]] = [d, lo, hi]
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True).sort_values(["scenario", "true_cost"])
    out.to_csv(RESULTS / f"summary_{name}.csv", index=False)
    return out


def write_seed_uncertainty(nominal: pd.DataFrame, members: pd.DataFrame, rl) -> None:
    """Each multi-seed agent against the best tuned rule and the planner, with CIs over training seeds and episodes."""
    means = nominal.groupby("policy")["true_cost"].mean()
    rules = [p for p in means.index if p.startswith("Rule:") and "oracle" not in p]
    references = [min(rules, key=means.get)] + ([PLANNER] if PLANNER in means.index else [])
    rows = []
    for label, specs, _ in rl:
        if len(specs) < 2:
            continue
        own = members[members["policy"] == label]
        for ref in references:
            d, lo, hi = paired_difference_over_seeds(own, nominal[nominal["policy"] == ref])
            rows.append({"policy": label, "reference": ref, "seeds": len(specs), "episodes": own["seed"].nunique(),
                         "diff": d, "diff_lo": lo, "diff_hi": hi})
    pd.DataFrame(rows).to_csv(RESULTS / "summary_seed_uncertainty.csv", index=False)


def record_timeline(spec: dict, env_overrides: dict, seed: int) -> dict:
    env = BridgeNetworkEnv(load_env_config(), overrides=env_overrides)
    policy = make_policy(spec)
    obs, _ = env.reset(seed=seed)
    tl = {"bridge_id": list(env.net.bridge_id), "archetype": list(env.net.archetype), "months": []}
    done = False
    while not done:
        a = policy(env, obs)
        month = {"entropy": B.entropy(env.belief).round(4).tolist(), "expected": B.expected_state(env.belief).round(4).tolist(),
                 "true": env.state.tolist(), "closed": env.closed.tolist(), "balance": float(env.budget)}
        obs, _, done, _, info = env.step(a)
        month.update({"action_kind": int(info["action_kind"]), "bridge": int(info["bridge"])})
        tl["months"].append(month)
    tl["episode"] = info["episode"]
    return tl


def decision_times(specs: list[tuple[str, dict, dict]], seed: int, months: int = 24) -> dict:
    """Wall-clock seconds per decision on one CPU thread, over the first months of one test episode."""
    import time

    torch.set_num_threads(1)
    out = {}
    for label, spec, env_over in specs:
        env = BridgeNetworkEnv(load_env_config(), overrides=env_over)
        policy = make_policy(spec)
        obs, _ = env.reset(seed=seed)
        start = time.perf_counter()
        for _ in range(months):
            obs, *_ = env.step(policy(env, obs))
        out[label] = (time.perf_counter() - start) / months
    return out


# ---------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=len(TEST_SEEDS))
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--budget-sweep", action="store_true")
    parser.add_argument("--runs", nargs="*", default=["main", "pretrain_only", "no_demos", "rule_demos", "shared", "flat", "vanilla", "oracle", "last_obs", "no_inspection", "ppo"])
    parser.add_argument("--no-planner", action="store_true", help="skip the slow lookahead planner")
    parser.add_argument("--planner-only", action="store_true", help="only run (and cache) the planner's test episodes")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    global RUNS, RESULTS
    RUNS, RESULTS = ROOT / args.runs_dir, ROOT / args.results_dir
    seeds = list(TEST_SEEDS)[: args.episodes]
    (RESULTS / "episodes").mkdir(parents=True, exist_ok=True)

    if args.planner_only:
        run_jobs(jobs_for([planner_spec()], "nominal", {}, seeds), args.workers)
        return
    heur = heuristic_specs()
    rl = rl_specs(args.runs)
    random_init = [("DQN (random init, untrained)", [{"kind": "dqn_random", "seed": s} for s in range(3)], {})]

    teacher = [] if args.no_planner else [planner_spec()]
    nominal = run_jobs(jobs_for(teacher + heur + rl + random_init, "nominal", {}, seeds), args.workers)
    nominal_members = MEMBER_ROWS[-1]
    nominal.to_csv(RESULTS / "episodes" / "nominal.csv", index=False)
    nominal_members.to_csv(RESULTS / "episodes" / "nominal_members.csv", index=False)
    summary = write_summary(nominal, "nominal")
    write_seed_uncertainty(nominal, nominal_members, rl)
    cols = ["policy", "true_cost", "true_cost_lo", "true_cost_hi", "closure_months", "inspections", "diff_vs_main", "diff_lo", "diff_hi"]
    print(summary[[c for c in cols if c in summary]].to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    main_runs = [r for r in rl if r[0] == MAIN]
    if main_runs:
        spec = main_runs[0][1][0]
        (RESULTS / "timeline.json").write_text(json.dumps(record_timeline(spec, main_runs[0][2], seeds[0])))
        timed = [(MAIN, spec, main_runs[0][2])] + [(label, members[0], env) for label, members, env in teacher]
        (RESULTS / "decision_time.json").write_text(json.dumps(decision_times(timed, seeds[0]), indent=2))

    if args.ood:
        keep = {MAIN, ENSEMBLE, "Maskable PPO"}
        pols = [p for p in heur if p[0] != HEURISTIC_LABELS["do_nothing"]] + [p for p in rl if p[0] in keep]
        frames = [run_jobs(jobs_for(pols, sc, ov, seeds), args.workers) for sc, ov in SCENARIOS.items() if sc != "nominal"]
        frames.append(run_jobs(unseen_network_jobs(pols, seeds), args.workers))
        ood = pd.concat(frames + [nominal[nominal["policy"].isin([p[0] for p in pols])]], ignore_index=True)
        ood.to_csv(RESULTS / "episodes" / "ood.csv", index=False)
        pd.concat(MEMBER_ROWS[-len(frames):], ignore_index=True).to_csv(RESULTS / "episodes" / "ood_members.csv", index=False)
        write_summary(ood, "ood")

    if args.budget_sweep:
        frames = []
        for scale in (0.5, 0.75, 1.0, 1.5, 2.0):
            ov = {"stress": {"budget_scale": scale}}
            # Rules are re-tuned and the agent retrained for each funding level.
            pols = [p for p in heuristic_specs(scale) if p[0] != HEURISTIC_LABELS["do_nothing"]]
            pols += [p for p in rl_specs(["main" if scale == 1.0 else f"main-budget{scale:g}"], allow_missing=True) if p[0] == MAIN]
            part = run_jobs(jobs_for(pols, f"budget_{scale:g}", ov, seeds), args.workers)
            part["budget_scale"] = scale
            frames.append(part)
        sweep = pd.concat(frames, ignore_index=True)
        sweep.to_csv(RESULTS / "episodes" / "budget_sweep.csv", index=False)
        rows = []
        for (scale, policy), g in sweep.groupby(["budget_scale", "policy"]):
            mean, lo, hi = bootstrap_ci(g["true_cost"])
            rows.append({"budget_scale": scale, "policy": policy, "true_cost": mean, "lo": lo, "hi": hi})
        pd.DataFrame(rows).to_csv(RESULTS / "summary_budget_sweep.csv", index=False)


if __name__ == "__main__":
    main()
