# BridgePulse-RL

**The problem.** A road agency has to keep ten aging bridges in service for ten years, with half
the money it needs and without ever seeing their true condition. Each month it issues one work
order:

- wait,
- buy an in-depth inspection,
- carry out a minor repair, or
- carry out a major repair.

Each work order targets one bridge. Money comes from a maintenance account that is topped up
monthly and carries savings forward.

**What the agent sees and pays for.** The agent sees an exact Bayesian belief over every bridge's
condition, plus the account balance. It is judged on the full 10-year cost to society:

- agency spending,
- road-user detour costs while a bridge is closed,
- structural risk,
- the cost of handing over bridges in poor shape at the end.

**Where the model comes from.** The bridges are real Pennsylvania bridges from the National Bridge
Inventory. The deterioration model is fitted by maximum likelihood to 15,190 pairs of real
inspections.

**The agent.** The main agent is a **deep Q-network built for this problem**:

- a per-bridge value decomposition linked by a learned price of money;
- action masking in acting and in the bootstrap target;
- double Q-learning and 3-step returns;
- a warm start from demonstrations of a lookahead planner that improves the best tuned rule by
  simulation; their influence fades during training.

It is compared on the same held-out test episodes, with common random numbers, against:

- four tuned rule-based policies;
- the lookahead planner it learned from;
- Maskable PPO;
- ablated DQNs: rule instead of planner demonstrations, no demonstrations, an undecomposed network,
  a flat network, and vanilla DQN;
- agents that see more or less information.

## Results

All numbers are on 200 held-out test episodes. Every policy faces the same random draws in each
episode, so the comparisons are paired, with 95% bootstrap confidence intervals. "Δ vs ours" is how
much more a policy costs than our agent (negative = cheaper).

| Policy | 10-year cost, $M [95% CI] | Δ vs ours, $M [95% CI] | Bridge-months closed | Inspections |
|---|---|---|---|---|
| **DQN (ours)** | 8.28 [7.24, 9.51] | — | 5.7 | 1.7 |
| DQN (ours, 5-seed ensemble) | 8.06 [7.09, 9.16] | -0.22 [-0.42, -0.07] | 5.4 | 0.7 |
| Lookahead planner (teacher) | 8.61 [7.61, 9.82] | +0.33 [-0.21, +1.02] | 5.9 | 4.2 |
| DQN, offline pre-training only | 8.26 [7.28, 9.35] | -0.02 [-0.24, +0.14] | 6.4 | 1.0 |
| DQN, rule demonstrations | 8.96 [8.11, 9.99] | +0.68 [+0.21, +1.07] | 4.7 | 0.0 |
| DQN, no demonstrations | 13.47 [12.32, 14.77] | +5.19 [+4.54, +5.97] | 3.4 | 44.6 |
| DQN, shared network | 8.76 [7.47, 10.26] | +0.48 [+0.14, +0.88] | 6.2 | 0.6 |
| DQN, flat network | 10.84 [9.06, 12.85] | +2.56 [+1.65, +3.58] | 10.1 | 2.0 |
| Maskable PPO | 14.93 [13.11, 16.93] | +6.64 [+5.60, +7.80] | 11.6 | 78.1 |
| Vanilla DQN | 158.80 [136.69, 181.90] | +150.52 [+128.12, +173.56] | 88.1 | 71.3 |
| DQN, last inspection only | 8.14 [7.20, 9.22] | -0.14 [-0.37, +0.07] | 5.2 | 0.5 |
| DQN, no extra inspections | 8.41 [7.39, 9.55] | +0.13 [-0.11, +0.43] | 5.9 | 0.0 |
| DQN, true condition | 7.61 [6.58, 8.87] | -0.67 [-1.02, -0.31] | 5.4 | 0.1 |
| Rule: condition-based | 9.37 [8.38, 10.56] | +1.08 [+0.51, +1.58] | 6.0 | 0.0 |
| Rule: entropy-triggered | 12.20 [10.60, 14.04] | +3.92 [+2.95, +4.96] | 13.3 | 1.1 |
| Rule: risk-greedy | 8.96 [8.11, 9.96] | +0.68 [+0.21, +1.08] | 4.8 | 0.0 |
| Rule: oracle threshold | 8.69 [7.78, 9.76] | +0.41 [-0.25, +1.02] | 5.7 | 0.0 |
| DQN (random init, untrained) | 300.66 [252.59, 351.24] | +292.37 [+244.10, +343.20] | 106.3 | 40.0 |
| Do nothing | 300.59 [252.52, 351.17] | +292.31 [+244.03, +343.13] | 106.3 | 0.0 |

### Key findings

- **The DQN beats every tuned rule.** Its 10-year cost of $8.28M is 8% below the best rule
  (risk-greedy, $8.96M): a paired saving of $0.68M [0.21, 1.08]. Averaging the five networks'
  Q-values gives $8.06M.
- **It matches the planner it learned from, about 1,200× faster.** It is $0.33M cheaper than the
  planner, though the CI [-0.21, 1.02] includes zero. It decides in 0.6 ms instead of 750 ms.
- **The gain comes from the planner's demonstrations, learned offline.**
  - Q-learning with the margin on the recorded episodes alone already reaches $8.26M.
  - Online Q-learning keeps that level while the margin lasts. Every seed degrades once the margin
    reaches zero, so the best-validation checkpoint is kept.
  - With rule demonstrations the agent only matches the rule. From scratch it reaches $13.5M.
- **The per-bridge decomposition matters.** The same agent with an undecomposed network costs
  $8.76M, and with a flat MLP $10.84M. Maskable PPO ($14.9M) and textbook DQN ($159M) do not reach the
  rules.
- **How it saves.** It makes about as many repairs as the rule but spends $2.87M instead of $2.37M.
  It cuts structural risk from $4.44M to $3.33M and the final restore cost from $0.68M to $0.12M, at
  slightly more closure.
- **Value of information.** Seeing the true condition is worth $0.67M [0.31, 1.02]. Routine ratings
  are accurate (σ = 0.35), so the Bayesian filter and paid inspections recover little of it.
- **Robustness and its limit.** Without retraining, the agent is still significantly cheaper than the
  best rule under slower decay, faster decay and a 25% funding cut. It is no better when decay is
  faster than its model expects. It is **worse** with noisier routine ratings and on unseen bridge
  networks: it is trained for one network and does not transfer, while the rules do by construction.

| Stress test (no retraining) | DQN (ours) | Best rule | Saving, $M [95% CI] |
|---|---|---|---|
| Slower decay (×0.5) | 5.17 | 6.27 | +1.11 [+1.00, +1.22] |
| Faster decay (×5/3) | 16.05 | 16.83 | +0.77 [+0.03, +1.49] |
| Decay 50% faster than the agent's model | 13.31 | 13.10 | −0.21 [−1.26, +0.66] |
| Funding cut 25% | 15.05 | 28.68 | +13.63 [+7.58, +19.83] |
| Noisy routine ratings (σ = 0.65) | 15.70 | 12.43 | −3.27 [−4.73, −1.90] |
| 20 unseen networks of real bridges | 20.97 | 13.13 | −7.83 [−13.75, −3.20] |

Figures in `results/figures/`, report in `report/main.pdf`:

- `fig1_learning`: learning curves.
- `fig2_information`: value of information and robustness.
- `fig3_timeline`: one annotated test episode.

The funding sweep of the rules (re-tuned at 0.5×–2× funding) is in
`results/summary_budget_sweep.csv`.

## Repository layout

```
bridgepulse/         environment package
  env.py             Gymnasium environment (one work order per month, action masks, common random numbers)
  dynamics.py        deterioration hazards, repair effects, inspection noise
  belief.py          exact Bayesian filter for each bridge's hidden condition
  costs.py           agency, closure, risk and salvage costs
  network.py         bridge attributes and NBI rating mapping
  evaluation.py      evaluation, bootstrap confidence intervals, paired comparisons
agents/              DQN agent, Q-networks (decomposed, shared, flat), replay with n-step returns
baselines/           rule-based policies, their grid search, and the lookahead planner (rollout.py)
scripts/             data preparation, calibration, training, evaluation, figures
configs/             environment, agent, rule and calibration settings
data/processed/      the bridge network, the held-out bridge pool and the fitted deterioration model
data/demos/          recorded planner demonstrations used to warm-start the DQN
tests/               unit tests (filter vs brute force, environment, agent, data pipeline)
models/              final trained agents
results/             evaluation tables, per-episode results, training logs, figures
report/              2-page report (LaTeX)
docs/FORMULATION.md  full problem formulation and algorithm
```

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # about 20 s
```

## Using a trained agent

```python
from agents.dqn import DQNAgent
from bridgepulse import make_env

env = make_env()
agent = DQNAgent.load("models/dqn_main_seed0.pt")
obs, _ = env.reset(seed=123)
done = False
while not done:
    obs, reward, done, _, info = env.step(agent.act(obs, env.action_masks()))
print(f"10-year cost: ${info['episode']['true_cost'] / 1e6:.2f}M")
```

## Reproducing the results

Every number in the report and in this README is regenerated by these commands. They run on a
4-core CPU; no GPU is needed.

| command | what it does | time on 4 cores |
|---|---|---|
| `make data` | optional: rebuild the network and deterioration fit from NBI files (see Data below) | |
| `make calibrate` | funding level, then the rule grids at every funding level | ~1.5 h |
| `make train-main` | main DQN agent, its offline-only version and Maskable PPO, 5 seeds each | ~1.5 h |
| `make demos` | record the planner demonstrations for every training setting (shipped in `data/demos/`) | ~2 h |
| `make train-ablations` | 8 DQN ablations, 2 seeds each | ~2 h |
| `make train-sweep` | optional: DQN retrained at 0.5× and 2× funding | ~1 h |
| `make evaluate` | 200 held-out test episodes (including the slow planner), stress tests, unseen networks, rule funding sweep | ~1.5 h |
| `make figures` | trained agents to `models/`, figures, LaTeX tables and macros in `report/generated` | |
| `make smoke` | a few-minute end-to-end check | |

`configs/calibration.yaml` holds the funding level used for every result. Running `make calibrate`
recomputes it.

Single runs:

```bash
python scripts/train_dqn.py --variant main --seed 0                  # main agent
python scripts/train_dqn.py --variant no_demos --seed 0              # same agent learned from scratch
python scripts/train_dqn.py --variant main --seed 0 --budget-scale 0.5 --tag budget0.5
python scripts/train_ppo.py --seed 0
python scripts/evaluate.py --runs main ppo --episodes 200
```

## Data

The bridge network and the deterioration model come from the U.S. National Bridge Inventory.

1. Download two delimited releases for one state from https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm,
   for example Pennsylvania 2022 and 2024.
2. Save them as `data/raw/PA22.txt` and `data/raw/PA24.txt`.
3. Run `make data`.

This does three things:

- **Fits the deterioration model** by maximum likelihood over paired inspections. It estimates
  monthly deterioration hazards, the effects of age and traffic, the routine rating noise, and the
  mix of hidden conditions. Both ratings in a pair are treated as noisy observations of the hidden
  state.
- **Selects the ten-bridge network.** It picks 3 arterial, 4 collector and 3 local bridges, with
  their real deck areas, traffic, truck shares and detour lengths.
- **Keeps a pool of about 300 held-out bridges.** These are used for the unseen-network tests.

The processed files are included, so the raw downloads are only needed to rebuild them.

## Problem in brief

| | |
|---|---|
| State (103) | per bridge: belief over 5 condition states, closure cost, repair cost, age, months since inspection, closed flag; global: account balance, month of year, time |
| Actions (31) | defer; inspect bridge *i*; minor repair *i*; major repair *i* (infeasible actions are masked) |
| Reward | minus the monthly cost in $M: agency spending + road-user closure cost + structural risk, plus the cost to restore at the end |
| Horizon | 120 months |
| Hidden dynamics | fitted monthly deterioration, closures of Critical bridges, noisy routine and paid inspections |

See `docs/FORMULATION.md` for the full specification.
