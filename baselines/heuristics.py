"""Rule-based maintenance policies used as baselines. Each one has tunable thresholds (see scripts/tune_heuristics.py)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bridgepulse import actions as A
from bridgepulse import belief as B
from bridgepulse.env import BridgeNetworkEnv

# Rough chance that a bridge in each state ends up closed within about a year; used to rank urgency.
CLOSURE_EXPOSURE = np.array([0.0, 0.0, 0.02, 0.15, 1.0])


def _first_allowed(order, kind: int, env: BridgeNetworkEnv, mask: np.ndarray) -> int | None:
    for i in order:
        a = A.encode(kind, int(i), env.n)
        if mask[a]:
            return a
    return None


def _urgency(env: BridgeNetworkEnv, belief: np.ndarray) -> np.ndarray:
    """Expected monthly exposure of each bridge: risk cost plus a closure-weighted user cost."""
    f = env.costs.risk_scores[None, :] * env.costs.risk_base[:, None] + CLOSURE_EXPOSURE[None, :] * env.costs.closure_month[:, None]
    return (belief * f).sum(axis=1)


@dataclass
class Heuristic:
    name: str = "heuristic"
    params: dict = field(default_factory=dict)
    needs_true_state: bool = False

    def __call__(self, env: BridgeNetworkEnv, obs: np.ndarray) -> int:
        return self.act(env)

    def act(self, env: BridgeNetworkEnv) -> int:
        raise NotImplementedError


class DoNothing(Heuristic):
    def __init__(self) -> None:
        super().__init__(name="Do nothing")

    def act(self, env: BridgeNetworkEnv) -> int:
        return 0


class ConditionBased(Heuristic):
    """Fixed inspection cycle plus repair thresholds on the expected condition."""

    def __init__(self, inspect_every: int = 12, major_at: float = 2.6, minor_at: float = 1.6) -> None:
        super().__init__(name="Condition-based", params=dict(inspect_every=inspect_every, major_at=major_at, minor_at=minor_at))

    def act(self, env: BridgeNetworkEnv) -> int:
        p = self.params
        mask = env.action_masks()
        exp_c = B.expected_state(env.observed_belief())
        urgency = _urgency(env, env.observed_belief())

        closed = np.flatnonzero(env.closed)
        a = _first_allowed(closed[np.argsort(-env.costs.closure_month[closed])], A.MAJOR, env, mask)
        if a is not None:
            return a
        severe = np.flatnonzero(exp_c >= p["major_at"])
        a = _first_allowed(severe[np.argsort(-urgency[severe])], A.MAJOR, env, mask)
        if a is not None:
            return a
        if p["inspect_every"] > 0:
            overdue = np.flatnonzero(env.delta >= p["inspect_every"])
            a = _first_allowed(overdue[np.argsort(-env.delta[overdue])], A.INSPECT, env, mask)
            if a is not None:
                return a
        moderate = np.flatnonzero((exp_c >= p["minor_at"]) & (exp_c < p["major_at"]))
        a = _first_allowed(moderate[np.argsort(-urgency[moderate])], A.MINOR, env, mask)
        return a if a is not None else 0


class RiskGreedy(Heuristic):
    """Picks the repair with the largest expected monthly exposure reduction per dollar, if above a threshold."""

    def __init__(self, min_benefit_per_dollar: float = 0.01) -> None:
        super().__init__(name="Risk-greedy", params=dict(min_benefit_per_dollar=min_benefit_per_dollar))

    def act(self, env: BridgeNetworkEnv) -> int:
        mask = env.action_masks()
        n = env.n
        base = _urgency(env, env.observed_belief())
        after_minor = _urgency(env, env.observed_belief() @ env.R_minor)
        after_major = _urgency(env, env.observed_belief() @ env.R_major)
        gain_minor = np.where(env.closed, 0.0, base - after_minor) / env.costs.minor
        gain_major = (base - after_major + env.costs.closure(env.closed)) / env.costs.major
        scores = np.concatenate([gain_minor, gain_major])
        actions = np.concatenate([np.arange(n) + 1 + n, np.arange(n) + 1 + 2 * n])
        scores = np.where(mask[actions], scores, -np.inf)
        best = int(np.argmax(scores))
        return int(actions[best]) if scores[best] >= self.params["min_benefit_per_dollar"] else 0


class EntropyTriggered(Heuristic):
    """Inspects bridges whose condition is uncertain and possibly poor, then repairs on the sharpened belief."""

    def __init__(self, entropy_at: float = 0.5, risk_at: float = 0.3, major_at: float = 0.5, minor_at: float = 0.5) -> None:
        super().__init__(name="Entropy-triggered", params=dict(entropy_at=entropy_at, risk_at=risk_at, major_at=major_at, minor_at=minor_at))

    def act(self, env: BridgeNetworkEnv) -> int:
        p = self.params
        mask = env.action_masks()
        b = env.observed_belief()
        urgency = _urgency(env, b)
        p_poor = b[:, 3:].sum(axis=1)
        p_fair = b[:, 1:3].sum(axis=1)

        closed = np.flatnonzero(env.closed)
        a = _first_allowed(closed[np.argsort(-env.costs.closure_month[closed])], A.MAJOR, env, mask)
        if a is not None:
            return a
        sure_poor = np.flatnonzero(p_poor >= p["major_at"])
        a = _first_allowed(sure_poor[np.argsort(-urgency[sure_poor])], A.MAJOR, env, mask)
        if a is not None:
            return a
        ent = B.entropy(b)
        unsure = np.flatnonzero((ent >= p["entropy_at"]) & (b[:, 2:].sum(axis=1) >= p["risk_at"]))
        a = _first_allowed(unsure[np.argsort(-(urgency * ent)[unsure])], A.INSPECT, env, mask)
        if a is not None:
            return a
        fair = np.flatnonzero((p_fair >= p["minor_at"]) & (B.expected_state(b) >= 1.0))
        a = _first_allowed(fair[np.argsort(-urgency[fair])], A.MINOR, env, mask)
        return a if a is not None else 0


class OracleThreshold(Heuristic):
    """Threshold rule on the true condition (only meaningful when the true state is observable)."""

    def __init__(self, major_at: int = 3, minor_at: int = 2) -> None:
        super().__init__(name="Oracle threshold", params=dict(major_at=major_at, minor_at=minor_at), needs_true_state=True)

    def act(self, env: BridgeNetworkEnv) -> int:
        p = self.params
        mask = env.action_masks()
        onehot = np.eye(5)[env.state]
        urgency = _urgency(env, onehot)
        severe = np.flatnonzero((env.state >= p["major_at"]) | env.closed)
        a = _first_allowed(severe[np.argsort(-urgency[severe])], A.MAJOR, env, mask)
        if a is not None:
            return a
        fair = np.flatnonzero((env.state >= p["minor_at"]) & (env.state <= 2))
        a = _first_allowed(fair[np.argsort(-urgency[fair])], A.MINOR, env, mask)
        return a if a is not None else 0


REGISTRY = {
    "do_nothing": DoNothing,
    "condition_based": ConditionBased,
    "risk_greedy": RiskGreedy,
    "entropy_triggered": EntropyTriggered,
    "oracle_threshold": OracleThreshold,
}

GRIDS = {
    "condition_based": dict(inspect_every=[0, 6, 12, 24], major_at=[1.8, 2.2, 2.6, 3.0, 3.4, 3.8], minor_at=[0.8, 1.2, 1.6, 2.0]),
    "risk_greedy": dict(min_benefit_per_dollar=[0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]),
    "entropy_triggered": dict(entropy_at=[0.3, 0.45, 0.6], risk_at=[0.2, 0.4], major_at=[0.2, 0.3, 0.4, 0.6], minor_at=[0.2, 0.3, 0.4, 0.6]),
    "oracle_threshold": dict(major_at=[3, 4], minor_at=[1, 2, 3]),
}


def build(name: str, params: dict | None = None) -> Heuristic:
    return REGISTRY[name](**(params or {}))
