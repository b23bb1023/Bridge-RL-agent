PY ?= python
JOBS ?= 4
SEEDS ?= 0 1 2 3 4
ABLATION_SEEDS ?= 0 1
SWEEP ?= 0.5 2

.PHONY: help install test data calibrate demos train-main train-ablations train-sweep evaluate figures report all smoke clean-runs

help:
	@echo "make install      install pinned dependencies"
	@echo "make test         run the unit tests"
	@echo "make data         build network + deterioration fit from NBI files in data/raw (see README)"
	@echo "make calibrate    calibrate yearly funding and tune the rule-based baselines"
	@echo "make demos        record lookahead-planner demonstrations for the DQN warm start"
	@echo "make train-main   main DQN agent, its offline-only version and maskable PPO, $(words $(SEEDS)) seeds each"
	@echo "make train-ablations  architecture, demonstration and information ablations"
	@echo "make train-sweep  optional: DQN retrained at other funding levels"
	@echo "make evaluate     final test evaluation, stress scenarios and funding sweep"
	@echo "make figures      export trained agents to models/, figures and report tables"
	@echo "make smoke        a 3-minute end-to-end check"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

data:
	$(PY) scripts/prepare_nbi.py --release-a data/raw/PA22.txt --release-b data/raw/PA24.txt --as-of 2024-03 --fit-noise

calibrate:
	$(PY) scripts/calibrate.py --workers $(JOBS)
	$(PY) scripts/tune_heuristics.py --workers $(JOBS)
	for s in 0.5 0.75 1.5 2; do $(PY) scripts/tune_heuristics.py --workers $(JOBS) --budget-scale $$s; done

demos:
	$(PY) scripts/make_demos.py --workers $(JOBS)
	$(PY) scripts/make_demos.py --workers $(JOBS) --obs-mode oracle
	$(PY) scripts/make_demos.py --workers $(JOBS) --obs-mode last_obs
	$(PY) scripts/make_demos.py --workers $(JOBS) --no-inspection

train-main:
	$(PY) scripts/run_many.py --jobs $(JOBS) --dqn main pretrain_only --ppo --seeds $(SEEDS)

train-ablations:
	$(PY) scripts/run_many.py --jobs $(JOBS) --dqn no_demos rule_demos shared flat vanilla oracle last_obs no_inspection --seeds $(ABLATION_SEEDS)

train-sweep:
	for s in $(SWEEP); do $(PY) scripts/make_demos.py --workers $(JOBS) --budget-scale $$s; done
	$(PY) scripts/run_many.py --jobs $(JOBS) --dqn main --budget-scales $(SWEEP) --seeds $(ABLATION_SEEDS)

evaluate:
	$(PY) scripts/evaluate.py --workers $(JOBS) --ood --budget-sweep

figures:
	$(PY) scripts/export_models.py
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_report_assets.py

all: test calibrate demos train-main train-ablations evaluate figures

smoke:
	$(PY) -m pytest -x
	$(PY) scripts/train_dqn.py --variant main --seed 0 --steps 20000 --runs-dir runs_smoke --set eval_every=10000 learning_starts=2000 demo_episodes=20 demo_pretrain_updates=1000
	$(PY) scripts/train_ppo.py --seed 0 --steps 20000 --runs-dir runs_smoke

clean-runs:
	rm -rf runs runs_smoke
