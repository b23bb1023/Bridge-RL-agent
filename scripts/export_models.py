"""Copy the selected checkpoint of every finished run into models/ (the trained agents shipped with the repository),
and its training log and settings into results/training_logs/ (used by the learning-curve figure).

Runs are read from runs/<variant>/seed<k>/. Models already in models/ are kept, so agents that were not retrained
(for example PPO) stay available. models/index.json always lists exactly the files present in models/.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS, MODELS, LOGS = ROOT / "runs", ROOT / "models", ROOT / "results" / "training_logs"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    copied = 0
    for run_dir in sorted(p for p in RUNS.glob("*/seed*") if p.is_dir()):
        variant, seed = run_dir.parent.name, run_dir.name
        if (run_dir / "best.pt").exists():
            shutil.copy(run_dir / "best.pt", MODELS / f"dqn_{variant}_{seed}.pt")
        elif (run_dir / "best.zip").exists():
            shutil.copy(run_dir / "best.zip", MODELS / f"ppo_{seed}.zip")
        else:
            continue
        copied += 1
        log_dir = LOGS / variant / seed
        log_dir.mkdir(parents=True, exist_ok=True)
        for name in ("log.csv", "config.json"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, log_dir / name)
    if copied == 0:
        print(f"no finished runs under {RUNS.relative_to(ROOT)}/; models/ left unchanged")

    index = []
    for f in sorted(MODELS.iterdir()):
        m = re.fullmatch(r"dqn_(.+)_(seed\d+)\.pt", f.name) or re.fullmatch(r"(ppo)_(seed\d+)\.zip", f.name)
        if m:
            index.append({"model": f.name, "run": f"{m.group(1)}/{m.group(2)}"})
    (MODELS / "index.json").write_text(json.dumps(index, indent=2))
    print(f"exported {copied} run(s); models/index.json lists {len(index)} model(s)")


if __name__ == "__main__":
    main()
