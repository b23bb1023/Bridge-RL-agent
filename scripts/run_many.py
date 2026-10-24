"""Run many training jobs in parallel (one CPU thread each) and skip runs that already finished.

  python scripts/run_many.py --jobs 4 --dqn shared flat --ppo --seeds 0 1 2 3 4
  python scripts/run_many.py --jobs 4 --dqn shared --budget-scales 0.5 0.75 1.5 2 --seeds 0 1 2
  python scripts/run_many.py --jobs 8 --dqn main:0-4 no_demos:0-1     # per-variant seeds
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def finished(run_dir: Path, target: int | None) -> bool:
    log = run_dir / "log.csv"
    if not log.exists():
        return False
    last = log.read_text().strip().splitlines()[-1].split(",")[0]
    return target is not None and last.isdigit() and int(last) >= target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dqn", nargs="*", default=[])
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--budget-scales", nargs="*", type=float, default=[])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    import yaml
    dqn_spec = yaml.safe_load((ROOT / "configs" / "dqn.yaml").read_text())
    dqn_steps = args.steps or dqn_spec["agent"]["total_steps"]
    ppo_steps = args.steps or yaml.safe_load((ROOT / "configs" / "ppo.yaml").read_text())["ppo"]["total_steps"]

    def seeds_of(item: str) -> tuple[str, list[int]]:
        """'main' uses --seeds; 'main:0-4' or 'main:0,2' gives that variant its own seeds."""
        if ":" not in item:
            return item, args.seeds
        name, spec = item.split(":", 1)
        if "-" in spec:
            lo, hi = spec.split("-")
            return name, list(range(int(lo), int(hi) + 1))
        return name, [int(x) for x in spec.split(",")]

    commands: list[tuple[str, list[str], Path, int]] = []
    for item in args.dqn:
        variant, variant_seeds = seeds_of(item)
        for scale in args.budget_scales or [None]:
            tag = [] if scale is None else ["--budget-scale", f"{scale:g}", "--tag", f"budget{scale:g}"]
            name = variant if scale is None else f"{variant}-budget{scale:g}"
            for seed in variant_seeds:
                cmd = [sys.executable, "scripts/train_dqn.py", "--variant", variant, "--seed", str(seed), "--quiet",
                       "--runs-dir", args.runs_dir, *tag]
                if args.steps:
                    cmd += ["--steps", str(args.steps)]
                if args.set:
                    cmd += ["--set", *args.set]
                target = args.steps or (dqn_spec["variants"][variant] or {}).get("agent", {}).get("total_steps", dqn_steps)
                commands.append((f"{name}/seed{seed}", cmd, ROOT / args.runs_dir / name / f"seed{seed}", int(target)))
    if args.ppo:
        for seed in args.seeds:
            cmd = [sys.executable, "scripts/train_ppo.py", "--seed", str(seed), "--quiet", "--runs-dir", args.runs_dir]
            if args.steps:
                cmd += ["--steps", str(args.steps)]
            commands.append((f"ppo/seed{seed}", cmd, ROOT / args.runs_dir / "ppo" / f"seed{seed}", ppo_steps))

    todo = [c for c in commands if not finished(c[2], c[3])]
    print(f"{len(commands)} runs requested, {len(commands) - len(todo)} already complete, {len(todo)} to run", flush=True)

    def run(item):
        name, cmd, out_dir, _ = item
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with open(out_dir / "stdout.log", "w") as f:
            code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"  {'done' if code == 0 else 'FAILED'}  {name}  ({(time.time() - start) / 60:.1f} min)", flush=True)
        return code

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(run, todo))
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
