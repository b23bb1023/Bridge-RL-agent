"""Copy the selected checkpoint of every finished run into models/ (the trained agents shipped with the repository),
and its training log and settings into results/training_logs/ (used by the learning-curve figure)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS, MODELS, LOGS = ROOT / "runs", ROOT / "models", ROOT / "results" / "training_logs"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    index = []
    for run_dir in sorted(p for p in RUNS.glob("*/seed*") if p.is_dir()):
        variant, seed = run_dir.parent.name, run_dir.name
        if (run_dir / "best.pt").exists():
            dst = MODELS / f"dqn_{variant}_{seed}.pt"
            shutil.copy(run_dir / "best.pt", dst)
        elif (run_dir / "best.zip").exists():
            dst = MODELS / f"ppo_{seed}.zip"
            shutil.copy(run_dir / "best.zip", dst)
        else:
            continue
        index.append({"model": dst.name, "run": f"{variant}/{seed}"})
        log_dir = LOGS / variant / seed
        log_dir.mkdir(parents=True, exist_ok=True)
        for name in ("log.csv", "config.json"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, log_dir / name)
    (MODELS / "index.json").write_text(json.dumps(index, indent=2))
    print(f"exported {len(index)} models to {MODELS}")


if __name__ == "__main__":
    main()
