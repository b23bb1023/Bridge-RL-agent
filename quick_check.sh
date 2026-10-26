#!/bin/bash
# A few-minute rehearsal of the whole pipeline with tiny settings, run in a temporary copy of this folder,
# so nothing here is touched. Run it once before ./run_all.sh.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
PY="${PY:-$HERE/.venv/bin/python}"
[ -x "$PY" ] || { echo "error: $PY not found; create the virtual environment first (see the run instructions)"; exit 1; }
JOBS="${JOBS:-$("$PY" -c 'import os; print(max(1, (os.cpu_count() or 4) - 1))')}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar cf - --exclude ./.venv --exclude ./runs --exclude ./run_logs . | (cd "$TMP" && tar xf -)
cd "$TMP"
echo "rehearsal in $TMP with $JOBS processes"

"$PY" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ needed, found " + sys.version'
"$PY" -m pytest -p no:cacheprovider -x 2>&1 | tail -1
rm -f data/demos/planner_budget1_oracle.npz data/demos/planner_budget1_last_obs.npz
for mode in oracle last_obs; do
  "$PY" scripts/make_demos.py --obs-mode "$mode" --episodes 2 --rollouts 2 --workers "$JOBS"
done
"$PY" scripts/run_many.py --jobs "$JOBS" --steps 3000 --dqn main:0-1 pretrain_only:0 no_demos:0 vanilla:0 oracle:0 last_obs:0 \
  --set eval_every=1500 learning_starts=500 demo_pretrain_updates=100 demo_episodes=2
"$PY" scripts/evaluate.py --workers "$JOBS" --episodes 4 --ood --budget-sweep \
  --runs main pretrain_only no_demos vanilla oracle last_obs ppo
"$PY" scripts/export_models.py
"$PY" scripts/make_figures.py
"$PY" scripts/make_report_assets.py
for f in results/summary_nominal.csv results/summary_ood.csv results/summary_budget_sweep.csv \
         results/summary_seed_uncertainty.csv results/figures/fig1_learning.pdf report/generated/numbers.tex; do
  [ -s "$f" ] || { echo "QUICK CHECK FAILED: $f missing"; exit 1; }
done
echo
echo "QUICK CHECK PASSED - now run ./run_all.sh"
