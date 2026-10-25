#!/bin/bash
# Re-run every experiment after the fixes, then package the outputs.
#
#   ./run_all.sh            (from the project folder, with the virtual environment in .venv)
#
# Safe to stop and start again: finished training runs and recorded demonstrations are skipped.
# Settings (optional): JOBS=<parallel processes>  PY=<python>  ABL=0 (one seed per ablation instead of two)  RETRAIN_PPO=1

set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "error: $PY not found; create the virtual environment first (see the run instructions)"; exit 1; }
JOBS="${JOBS:-$("$PY" -c 'import os; print(max(1, (os.cpu_count() or 4) - 1))')}"
ABL="${ABL:-0-1}"
mkdir -p run_logs
START=$(date +%s)

step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
elapsed() { echo "    (elapsed $(( ($(date +%s) - START) / 60 )) min)"; }

step "0/6  checks (Python, packages, unit tests) - using $JOBS parallel processes"
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ needed, found " + sys.version'
"$PY" -c 'import numpy, scipy, pandas, torch, gymnasium, sb3_contrib, matplotlib, yaml; print("    packages ok, torch", torch.__version__)'
"$PY" -m pytest -q -p no:cacheprovider > run_logs/tests_before.log 2>&1 || true
tail -1 run_logs/tests_before.log
if ! grep -Eq "[0-9]+ passed" run_logs/tests_before.log || grep -Eq "[0-9]+ (failed|error)" run_logs/tests_before.log; then
  echo "error: unit tests failed; send run_logs/tests_before.log"; exit 1
fi

step "1/6  planner demonstrations for the oracle and last-inspection agents (re-recorded with the reward fix)"
for mode in oracle last_obs; do
  f="data/demos/planner_budget1_${mode}.npz"
  if [ -f "$f" ]; then echo "    $f already recorded, skipping"
  else
    "$PY" scripts/make_demos.py --obs-mode "$mode" --workers "$JOBS" 2>&1 | tee "run_logs/demos_${mode}.log" || true
    [ -f "$f" ] || { echo "error: recording $f failed; send run_logs/demos_${mode}.log"; exit 1; }
  fi
done
elapsed

step "2/6  training the DQN agents (main and offline-only: seeds 0-4; ablations: seeds $ABL)"
"$PY" scripts/run_many.py --jobs "$JOBS" --dqn main:0-4 pretrain_only:0-4 \
  no_demos:$ABL rule_demos:$ABL shared:$ABL flat:$ABL vanilla:$ABL oracle:$ABL last_obs:$ABL no_inspection:$ABL \
  2>&1 | tee run_logs/train_dqn.log || true
if [ "${RETRAIN_PPO:-0}" = "1" ]; then
  "$PY" scripts/run_many.py --jobs "$JOBS" --ppo --seeds 0 1 2 3 4 2>&1 | tee run_logs/train_ppo.log || true
fi
if grep -q "FAILED" run_logs/train_dqn.log run_logs/train_ppo.log 2>/dev/null; then
  echo "error: a training run failed (marked FAILED above); send runs/<variant>/seed<k>/stdout.log for it"; exit 1
fi
elapsed

step "3/6  evaluation: 200 test episodes, planner, stress tests, unseen networks, funding sweep"
"$PY" scripts/evaluate.py --workers "$JOBS" --ood --budget-sweep 2>&1 | tee run_logs/evaluate.log || true
[ -f results/summary_budget_sweep.csv ] && [ results/summary_budget_sweep.csv -nt run_logs/train_dqn.log ] \
  || { echo "error: evaluation did not finish; send run_logs/evaluate.log"; exit 1; }
elapsed

step "4/6  models, figures, report numbers and README table"
"$PY" scripts/export_models.py 2>&1 | tee run_logs/export.log
"$PY" scripts/make_figures.py 2>&1 | tee run_logs/figures.log
"$PY" scripts/make_report_assets.py 2>&1 | tee run_logs/report_assets.log

step "5/6  unit tests again"
"$PY" -m pytest -q -p no:cacheprovider > run_logs/tests_after.log 2>&1 || true
tail -1 run_logs/tests_after.log

step "6/6  packaging the outputs to send back"
rm -f outputs_for_review.zip
zip -qr outputs_for_review.zip results report/generated report/figures models/index.json run_logs README.md \
  -x "results/episodes/planner_cache/*"
ls -lh outputs_for_review.zip
elapsed
echo
echo "Done. Send outputs_for_review.zip back."
