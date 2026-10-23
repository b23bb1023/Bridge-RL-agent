# Problem formulation

A road agency maintains a network of N = 10 bridges for T = 120 months. Each month it can issue
one work order: do nothing, inspect one bridge in depth, or carry out a minor or major repair on
one bridge. The true condition of a bridge is never seen directly. It is only glimpsed through
noisy inspections, and through the closure of a bridge that has become critical. Money comes from
a maintenance account that is topped up every month and carries unspent funds forward.

This is a partially observable Markov decision process (POMDP). The agent acts on the exact Bayesian
belief over each bridge's condition, which turns the POMDP into a fully observed belief MDP.

## Hidden state

Each bridge has a condition state `c_i ∈ {0,1,2,3,4}` (Excellent, Good, Fair, Poor, Critical),
mapped from the lowest of its NBI deck, superstructure and substructure ratings:

| state | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| NBI rating | 8–9 | 6–7 | 5 | 4 | ≤ 3 |

Each bridge also carries an open/closed flag `D_i`, its age, and the months since its last inspection.

## Deterioration

Without repair, a bridge drops one state per month with probability

`q_{i,k}(t) = q_k · exp( β_age (age_i(t) − 50)/50 + β_adt (w_i − 0.5) )`, with `k = c_i < 4`,

where `w_i = log10(ADT_i + 1)/5` measures traffic exposure. Ages, and therefore hazards, are updated
once a year. State 4 is absorbing.

**Fitted from the National Bridge Inventory.** The base hazards `q_k`, `β_age` and `β_adt`, the
routine-inspection noise σ_rt and the distribution of hidden states are fitted by maximum likelihood
(`scripts/prepare_nbi.py`). The data are 15,190 Pennsylvania bridges inspected in both the 2022 and
2024 releases: no reconstruction in between, about 24 months between inspections. The likelihood
treats both recorded ratings as noisy observations of hidden states linked by the monthly chain over
the actual gap (a hidden-Markov likelihood), so rating noise is not mistaken for deterioration.

| fitted quantity | value |
|---|---|
| monthly drop probability q₀ … q₃ | 0.0049, 0.0014, 0.0014, 0.0013 |
| implied mean years in Good / Fair / Poor | 59 / 61 / 64 |
| age effect β_age | +0.63 (older bridges deteriorate faster) |
| traffic effect β_adt | −0.74 (busier bridges, which get more upkeep, change more slowly) |
| routine rating noise σ_rt | 0.35 |
| hidden-state mix (Excellent … Critical) | 0.07, 0.44, 0.35, 0.12, 0.02 |

A fit that ignores rating noise gives hazards about twice as fast (33–40 years per state). The
noise-aware fit reproduces the observed two-year transition frequencies to within about 1.5 percentage
points from the Excellent, Good, Fair and Poor states. From Critical, 13% of the recorded ratings rise
again; the model attributes most of these to rating noise and predicts 8% (see `fit_table` in
`data/processed/transition_fit.json`).

**Unmaintained deterioration (assumption).** Inventory ratings move slowly partly because agencies
keep doing routine upkeep that the inventory does not record as work. In the simulator the network
gets no upkeep except the agent's own work orders, so the fitted hazards are multiplied by
`unmaintained_factor = 3`, giving about 20 years per state. The stress tests repeat the evaluation
with factors 1.5 and 5.

An open bridge in state 4 is closed with probability 0.25 each month. It stays closed until a
major repair.

## Actions (31)

| index | action | effect |
|---|---|---|
| 0 | defer | — |
| 1–10 | in-depth inspection of bridge i | observation `o ~ O_in(·∣c_i)`, σ = 0.25 |
| 11–20 | minor repair of bridge i | `c ∈ {1,2} → c − 1` w.p. 0.9; no effect on states 0, 3, 4 |
| 21–30 | major repair of bridge i | `c → 0` w.p. 0.95, otherwise `c → 1`; reopens the bridge |

A repaired bridge does not deteriorate in the month of the repair. Every 24 months each bridge also
gets a free routine inspection on its regulatory schedule, with the fitted noise σ_rt = 0.35.

**Observation model.** `O(o | c) ∝ exp(−(o − c)² / 2σ²)` over `o ∈ {0..4}`. Routine inspections
(σ = 0.35) report the exact state about 97% of the time for interior states. The paid in-depth
inspection is assumed sharper (σ = 0.25, about 99.9%); its value is mostly in timing, since it
can be bought any month instead of waiting up to two years for the next routine visit.

**Feasibility mask.** An action is allowed only if its cost does not exceed the account balance.
Inspection and minor repair are not allowed on a closed bridge. Defer is always allowed.

## Belief update (exact)

For each bridge, in the order events happen within a month:

1. in-depth inspection: `b ← b ⊙ O_in(o | ·)`, normalised;
2. repair: `b ← b R` (minor or major repair matrix);
3. deterioration, if not repaired: `b'(k) = b(k)(1 − q_k) + b(k−1) q_{k−1}`;
4. closure: a newly closed bridge has `b = δ_4`; a bridge that stays open is weak evidence
   against state 4: `b ← b ⊙ [1, 1, 1, 1, 0.75]`, normalised;
5. routine inspection: `b ← b ⊙ O_rt(o | ·)`, normalised.

The starting condition is drawn from the posterior given the bridge's last recorded rating, rolled
forward by the months since that rating. So the filter is exact from the first month, and the agent
and the simulator never disagree about the model. `tests/test_belief.py` checks the filter against
brute-force enumeration of every hidden trajectory. `tests/test_env.py` checks that beliefs are
calibrated against the simulated true states.

## Observation given to the agent (103 numbers)

Per bridge (10 × 10):

- belief `b_i` (5);
- user cost of a closure (log-scaled);
- age / 100;
- `min(δ_i, 24) / 24`, where `δ_i` is months since the last inspection;
- major-repair cost (log-scaled);
- closed flag.

Global (3):

- account balance relative to the largest single repair;
- month of year;
- `t / T`.

Variants used in the ablations keep the same size:

- **oracle:** the true state one-hot replaces the belief;
- **last inspection:** the one-hot of the last observed state replaces the belief. A closed bridge
  shows as Critical, and a repair moves the recorded state the way the repair would.

## Costs and reward

All costs are in US dollars, scaled by deck area `A_i`, traffic and detour length:

| item | value |
|---|---|
| in-depth inspection | 16 $/m² · A_i |
| minor repair | 160 $/m² · A_i |
| major repair | 1,600 $/m² · A_i |
| closure (road-user cost, per month) | ADT_i · 30.4 · detour_i · (0.35 (1 − f_i) + 1.00 f_i) $/veh-km, where `f_i` is the truck share |
| structural risk (open bridges, per month) | ρ(c_i) · 0.025 · (major_i + closure_i), with ρ = [0, 0.05, 0.2, 0.6, 1] |
| salvage at t = T | cost to restore: minor repair for state 2, major repair for states 3–4 |

The reward is the negative monthly cost in millions of dollars:

`r_t = −( C_agency(a_t) + Σ_i D_i closure_i + Σ_i (1 − D_i) risk_i(c_i) + 1[t = T] Σ_i salvage_i(c_i) ) / 10⁶`

Policies are always compared on the true 10-year cost, which is exactly the negative undiscounted
return times 10⁶.

**Training reward with less noise.** The risk and salvage terms depend on the hidden state, which
the agent cannot see. For training, they are replaced by their expectation under the agent's
belief, `Σ_c b_i(c) risk_i(c)` and `Σ_c b_i(c) salvage_i(c)`. Because the belief is the exact
posterior, this has the same expected return for every policy, but much less variance. The closure
cost and the agency cost are always charged as they happen. Evaluation always uses the realised cost.

**Per-bridge reward.** `info["reward_vector"]` splits the same reward by bridge: each bridge is
charged its own closure, risk and salvage, plus the cost of any work order on it. The vector sums to
the scalar reward and is used by the decomposed Q-network.

Optional potential-based shaping (`Φ(b) = −expected monthly risk`, with `Φ = 0` at the end) is
available but off; it is never used in the comparisons.

## Funding

The account opens with one year of funding and receives 1/12 of a year's funding every month.
The yearly funding level is calibrated (`scripts/calibrate.py`):

1. Tune the rule-based policies with unlimited money. This gives the best cost and its yearly spend U.
2. Find the sufficiency level F*: the smallest of 1, 1.5, 2, 2.5, 3, 4, … × U at which the best tuned
   rule gets within 10% of its unlimited-money cost (here 2.5 U).
3. Set funding to 0.5 · F*.

The network therefore gets half the money it would need to keep every bridge in service. Which
bridges to fix, when, and what to save for is the decision that matters.

## Learning algorithm

The agent is a deep Q-network with four problem-specific parts.

**1. Masking wherever a maximum is taken.** The greedy action, the exploratory action and the
double-DQN bootstrap target all use only feasible actions. The next state's mask is stored in replay.

**2. A per-bridge value decomposition with a learned price of money.** The 31 action values are
built from ten per-bridge cost-to-go terms, `Q(s, a) = Σ_i Q_i(s, a)`. One encoder, shared by all
bridges, embeds each bridge's 10 features together with the 3 global ones. Mean and max pooling over
the embeddings give a network context. From its embedding and the context, each bridge produces:

- a baseline value `base_i`,
- one value for each work order on itself, `own_i(k)`,
- a non-negative shadow price of money, `λ_i`.

Then

- `Q_i(s, defer) = base_i`;
- `Q_i(s, work k on bridge i) = own_i(k)`;
- `Q_i(s, work k on bridge j ≠ i) = base_i − λ_i · cost(k, j)`.

A work order on another bridge affects bridge `i` only by spending money that `i` may need later,
and `λ_i` learns what that money is worth to it. Each `Q_i` is trained by TD on bridge `i`'s share of
the reward (`info["reward_vector"]`). All ten targets are evaluated at the same joint greedy next
action. The structure is permutation-equivariant: relabelling the bridges relabels the actions, and
what is learned about one bridge transfers to the others.

**3. A warm start from a lookahead planner (learning from demonstrations).**

*The planner* (`baselines/rollout.py`) improves the best tuned rule by simulation. Each month it
shortlists a few work orders:

- the rule's choice;
- defer;
- the four repairs with the most expected risk saved per dollar;
- the most informative inspection (belief entropy × exposure).

It scores each candidate by 8 rollouts: the candidate, then the rule until the end of the horizon.
The rollouts start from hidden conditions sampled from the agent's belief, never the true state.
Sampling is stratified per bridge, so a bridge that is Critical with probability 0.2 is Critical in
about 20% of the rollouts. All candidates share the same sampled worlds and random numbers. The
planner leaves the rule's choice only when the paired saving exceeds one standard error. This is one
step of policy improvement on the rule. It works, but it needs hundreds of simulated episodes per
decision.

*The demonstrations.* `scripts/make_demos.py` records 100 planner episodes once per setting. The
setting is the funding level plus what the agent observes; the oracle agent's planner starts its
rollouts from the true condition. All training seeds share the recording.

*Learning from them.* Before RL, the network is pre-trained on the recording for 20k updates. The
loss is TD plus a large-margin term `max_a [Q(s, a) + m·1(a ≠ a_E)] − Q(s, a_E)`, with `m = 0.5`
($0.5M). During RL, a quarter of every batch comes from the demonstrations, and the margin weight
fades linearly to zero over training. The agent starts from the planner's behaviour and is then free
to depart from it wherever its own experience says it should. The `rule_demos` ablation uses
episodes of the best tuned rule instead.

**4. Lower-variance targets.** 3-step returns, γ = 0.99, and the belief-expected risk and salvage
reward described above.

**Training settings.**

| setting | value |
|---|---|
| parallel environments | 8 |
| replay memory | every transition of the run (3·10⁵) |
| batch size | 128 |
| update rate | one gradient step per 4 environment steps |
| optimiser | Adam, 2.5·10⁻⁴; gradient-norm clip 10; Huber loss |
| target network | soft update, τ = 0.005 |
| exploration | ε from 0.3 to 0.05 over the first 20% of training; half of the exploratory actions defer |
| training length | 3·10⁵ environment steps |

Validation runs 40 fixed episodes every 20k steps, and the checkpoint with the lowest validation cost
is kept. The 5-seed ensemble averages the Q-values of the five trained networks before the masked
argmax.

**Ablations.** Each ablation changes one thing at a time:

- *offline pre-training only:* the network after the offline phase on the planner's episodes, with no
  online interaction;
- *rule demonstrations:* warm-started from the best tuned rule instead of the planner;
- *no demonstrations:* learns from scratch, with ε starting at 1;
- *shared network:* the same encoder with ordinary dueling heads, without the decomposition;
- *flat network:* an MLP over the whole observation;
- *vanilla DQN:* flat MLP, no masks (an infeasible action is penalised and replaced by defer),
  no dueling, no double estimator, no demonstrations, hard target updates;
- *information ablations:* the true condition instead of the belief, the last inspection result
  instead of the belief, or no paid inspections. Each is warm-started from a planner that has the
  same information.

Maskable PPO, trained on the same environment and reward for 10⁶ steps, is the policy-gradient
comparison.

## Evaluation protocol

- **Disjoint seed ranges** for training, tuning, validation and testing.
- **Common random numbers.** Every random draw in an episode (deterioration, closures, inspection
  noise, repair outcomes, initial condition) comes from the episode seed, so all policies face the
  same luck. Differences are paired episode by episode.
- **200 test episodes.** RL results are averaged over training seeds per episode. The planner runs on
  the same 200 episodes. 95% CIs come from the bootstrap; paired differences are reported against the
  main agent.
- **Baselines.** The rule-based policies are grid-searched on separate tuning episodes. The
  untrained network is included to show that the learned behaviour comes from training.
