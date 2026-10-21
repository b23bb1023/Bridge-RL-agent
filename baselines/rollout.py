"""One-step lookahead planner that improves a tuned rule by simulation.

Each month the planner shortlists a few candidate work orders. It scores each one by simulating that
order followed by the rule until the end of the horizon. The simulations start from hidden
conditions sampled from the agent's belief, never from the true state, and use fresh random numbers,
so the planner sees exactly what the agent sees. All candidates share the same sampled worlds
(common random numbers), so the comparison between them is far less noisy than each estimate alone.

Two details keep the planner from being fooled by rare events:

- *Stratified worlds.* Each bridge's starting condition is drawn by stratified sampling across the
  rollouts. A bridge that is Critical with probability 0.2 is Critical in about 20% of the rollouts,
  instead of possibly none.
- *Conservative switching.* The planner leaves the rule's choice only when the paired saving is
  larger than `z` standard errors.

The result is one step of policy improvement on the rule.
"""

from __future__ import annotations

import numpy as np

from baselines.heuristics import Heuristic, _urgency
from bridgepulse import actions as A
from bridgepulse import belief as B
from bridgepulse.dynamics import sample_rows
from bridgepulse.env import BridgeNetworkEnv

ARRAYS = ("age", "delta", "state", "belief", "closed", "last_obs", "_mask", "_q_phys", "_q_filter",
          "_u_det", "_u_close", "_u_routine", "_u_inspect", "_u_repair")
NOISE = ("_u_det", "_u_close", "_u_routine", "_u_inspect", "_u_repair")


def snapshot(env: BridgeNetworkEnv) -> dict:
    snap = {k: getattr(env, k).copy() for k in ARRAYS}
    snap.update(t=env.t, budget=env.budget, stats=dict(env.stats))
    return snap


def restore(env: BridgeNetworkEnv, snap: dict) -> None:
    for k in ARRAYS:
        setattr(env, k, snap[k].copy())
    env.t, env.budget, env.stats = snap["t"], snap["budget"], dict(snap["stats"])


class RolloutPlanner(Heuristic):
    def __init__(self, base: Heuristic, n_rollouts: int = 8, n_repairs: int = 4, z: float = 1.0, seed: int = 0) -> None:
        super().__init__(name=f"Lookahead on {base.name}", params=dict(n_rollouts=n_rollouts, n_repairs=n_repairs, z=z))
        self.base, self.rng = base, np.random.default_rng(seed)

    def candidates(self, env: BridgeNetworkEnv) -> list[int]:
        mask = env.action_masks()
        n, b = env.n, env.observed_belief()
        out = [self.base(env, None), 0]
        # The repairs a dollar-for-dollar risk ranking likes best.
        base = _urgency(env, b)
        gain_minor = np.where(env.closed, 0.0, base - _urgency(env, b @ env.R_minor)) / env.costs.minor
        gain_major = (base - _urgency(env, b @ env.R_major) + env.costs.closure(env.closed)) / env.costs.major
        scores = np.concatenate([gain_minor, gain_major])
        actions = np.concatenate([np.arange(n) + 1 + n, np.arange(n) + 1 + 2 * n])
        order = [int(a) for a in actions[np.argsort(-scores)] if mask[a]]
        out += order[: self.params["n_repairs"]]
        # The most informative inspection.
        if mask[1 : 1 + n].any():
            value = np.where(mask[1 : 1 + n], B.entropy(b) * base, -np.inf)
            out.append(A.encode(A.INSPECT, int(np.argmax(value)), n))
        return list(dict.fromkeys(out))

    def act(self, env: BridgeNetworkEnv) -> int:
        cands = self.candidates(env)
        if len(cands) == 1:
            return cands[0]
        snap = snapshot(env)
        known = env.observed_belief().copy()   # what the agent knows about the condition (belief, truth or last result)
        R = self.params["n_rollouts"]
        # Stratified uniforms: rollout r gets one draw from each 1/R slice, shuffled independently per bridge.
        strata = (np.arange(R)[:, None] + self.rng.random((R, env.n))) / R
        strata = np.take_along_axis(strata, self.rng.random((R, env.n)).argsort(axis=0), axis=0)
        costs = np.zeros((R, len(cands)))
        for r in range(R):
            start = sample_rows(known, strata[r]).astype(np.int64)
            noise = {k: self.rng.random(snap[k].shape) for k in NOISE}
            for j, a in enumerate(cands):
                restore(env, snap)
                env.state = start.copy()
                for k, v in noise.items():
                    setattr(env, k, v)
                done, action = False, a
                while not done:
                    _, rew, done, _, _ = env.step(action)
                    costs[r, j] -= rew
                    action = self.base(env, None)
        restore(env, snap)
        saving = costs[:, :1] - costs                      # paired saving of each candidate over the rule's choice
        mean = saving.mean(axis=0)
        se = saving.std(axis=0, ddof=1) / np.sqrt(R) if R > 1 else np.zeros_like(mean)
        best = int(np.argmax(mean))
        return cands[best] if mean[best] > self.params["z"] * se[best] else cands[0]
