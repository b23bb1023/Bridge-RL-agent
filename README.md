# BridgePulse-RL

Reinforcement learning for budget-constrained bridge maintenance. A DQN agent schedules inspections
and repairs across a network of aging bridges whose true condition is hidden, under a yearly budget.
States include the estimated condition of each bridge, its traffic importance, the time since its last
inspection and the remaining funds; actions inspect, repair or defer; rewards penalise failure risk
and spending.

Work in progress.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
