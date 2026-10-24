"""Gymnasium environment: inspection and maintenance of a bridge network under partial observability.

One decision per month. The agent observes a belief over each bridge's hidden condition
(maintained by an exact Bayes filter), the bridge attributes, and the balance of a maintenance
account that receives funding every month and carries unspent money forward.

Every source of randomness in an episode is drawn up front from the reset seed, so two
policies evaluated with the same seed face the same deterioration, closure, inspection-noise
and repair-outcome draws (common random numbers).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bridgepulse import actions as A
from bridgepulse import belief as B
from bridgepulse.config import deep_update, load_env_config
from bridgepulse.costs import CostModel
from bridgepulse.dynamics import (
    DeteriorationModel,
    major_repair_matrix,
    minor_repair_matrix,
    observation_matrix,
    sample_rows,
)
from bridgepulse.network import NUM_STATES, Network, load_network

K = NUM_STATES
FEATURES_PER_BRIDGE = 10
GLOBAL_FEATURES = 3
_OPEN_LIKELIHOOD_IDX = K - 1


class BridgeNetworkEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict | None = None, network: Network | None = None, overrides: dict | None = None) -> None:
        super().__init__()
        base = load_env_config() if config is None else config
        self.cfg = deep_update(base, overrides)
        cfg = self.cfg

        self.net = network if network is not None else load_network(cfg["network_file"])
        self.n = self.net.n
        self.T = int(cfg["horizon_months"])
        self.weight = self.net.weight

        stress = cfg["stress"]
        unmaintained = float(cfg["deterioration"].get("unmaintained_factor", 1.0))
        self.physics = DeteriorationModel(cfg["deterioration"], scale=unmaintained * stress["physics_decay_scale"])
        self.filter_model = DeteriorationModel(cfg["deterioration"], scale=unmaintained * stress["filter_decay_scale"])

        self.p_close = float(cfg["closure"]["monthly_prob"])
        self.open_likelihood = np.ones(K)
        self.open_likelihood[_OPEN_LIKELIHOOD_IDX] = 1.0 - self.p_close
        self.minor_success = float(cfg["repair"]["minor_success"])
        self.major_success = float(cfg["repair"]["major_success"])
        self.R_minor = minor_repair_matrix(self.minor_success)
        self.R_major = major_repair_matrix(self.major_success)

        insp = cfg["inspection"]
        self.routine_interval = int(insp["routine_interval_months"])
        self.O_routine = observation_matrix(float(insp["routine_sigma"]))
        self.O_indepth = observation_matrix(float(insp["indepth_sigma"]))

        self.costs = CostModel(self.net, cfg["costs"])
        self.reward_scale = self.costs.reward_scale
        bud = cfg["budget"]
        annual = bud["annual"]
        self.annual_budget = np.inf if annual is None else float(annual) * float(stress["budget_scale"])
        self.opening_balance = self.annual_budget * float(bud.get("initial_years", 1.0))
        cap = bud.get("cap_years")
        self.balance_cap = np.inf if cap is None else self.annual_budget * float(cap)
        # The balance feature reads 1.0 once any single repair in the network is affordable.
        self._balance_ref = 1.05 * float(self.costs.major.max())

        ai = cfg["agent_interface"]
        self.obs_mode = str(ai["obs_mode"])
        if self.obs_mode not in ("belief", "oracle", "last_obs"):
            raise ValueError(f"unknown obs_mode {self.obs_mode!r}")
        self.allow_inspection = bool(ai["allow_inspection"])
        self.domain_masks = bool(ai["domain_masks"])
        self.invalid_penalty = float(ai["invalid_action_penalty"])

        # Replacing risk and salvage by their expectation under the belief keeps every policy's expected
        # return unchanged only when the belief is the decision maker's posterior. That holds in belief
        # mode; an agent that sees the true state (or plans from its last result) gets realised costs.
        self.expected_reward = (cfg.get("training_reward", {}).get("risk_and_salvage", "realized") == "expected"
                                and self.obs_mode == "belief")
        sh = cfg["shaping"]
        self.shaping = bool(sh["enabled"])
        self.shaping_eta = float(sh.get("eta", 1.0))
        self.shaping_gamma = float(sh.get("gamma", 0.99))

        self.prior = np.asarray(cfg["initial_state"]["prior"], dtype=np.float64)
        self.perturb_prob = float(cfg["initial_state"].get("perturb_prob", 0.0))

        # Static per-bridge observation features.
        self._importance = np.clip(np.log10(1.0 + self.costs.closure_month / 1e3) / 4.0, 0.0, 1.0)
        self._cost_index = np.clip(np.log10(1.0 + self.costs.major / 1e3) / 4.5, 0.0, 1.0)

        self.num_actions = A.num_actions(self.n)
        self.obs_dim = self.n * FEATURES_PER_BRIDGE + GLOBAL_FEATURES
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_actions)
        self._eye = np.eye(K)
        self._initialised = False

    # ------------------------------------------------------------------ episode set-up
    def _draw_noise(self) -> None:
        rng = self.np_random
        T, n = self.T, self.n
        self._u_init = rng.random(n)
        self._u_det = rng.random((T, n))
        self._u_close = rng.random((T, n))
        self._u_routine = rng.random((T, n))
        self._u_inspect = rng.random(T)
        self._u_repair = rng.random(T)

    def _update_hazards(self) -> None:
        self._q_phys = self.physics.drop_probs(self.age, self.weight)
        self._q_filter = self.filter_model.drop_probs(self.age, self.weight)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._draw_noise()
        self.t = 0
        self.age = self.net.age_years.astype(np.float64).copy()
        self.delta = self.net.months_since_inspection.astype(np.int64).copy()
        rating = self.net.rating_state.astype(np.int64).copy()
        if self.perturb_prob > 0.0:
            rng = self.np_random
            moved = rng.random(self.n) < self.perturb_prob
            rating = np.where(moved, np.clip(rating + rng.choice([-1, 1], self.n), 0, 3), rating)
            self.delta = np.where(moved, rng.integers(0, 24, self.n), self.delta)
        self.closed = np.zeros(self.n, dtype=bool)
        self.last_obs = rating.copy()
        self._update_hazards()

        # The true starting condition is drawn from the same posterior the agent starts with,
        # so the filter is exact from the first month in every observation mode.
        true_prior = B.initial_belief(rating, self.delta, self._q_phys, self.O_routine, self.prior)
        self.state = sample_rows(true_prior, self._u_init).astype(np.int64)
        self.belief = B.initial_belief(rating, self.delta, self._q_filter, self.O_routine, self.prior)

        self.budget = self.opening_balance
        self.stats = {
            "agency": 0.0, "closure": 0.0, "risk": 0.0, "salvage": 0.0, "penalty": 0.0,
            "closure_months": 0, "critical_months": 0,
            "inspections": 0, "minor": 0, "major": 0, "invalid": 0,
            "routine_inspections": 0,
        }
        self._initialised = True
        self._mask = self._compute_mask()
        return self._obs(), {"t": 0}

    # ------------------------------------------------------------------ masks and observations
    def action_masks(self) -> np.ndarray:
        """Boolean mask of actions that are feasible this month (affordable, and sensible for closed bridges)."""
        return self._mask.copy()

    def _compute_mask(self) -> np.ndarray:
        n = self.n
        mask = self.costs.action_costs <= self.budget + 1e-6
        mask[1 : 1 + n] &= ~self.closed
        mask[1 + n : 1 + 2 * n] &= ~self.closed
        if not self.allow_inspection:
            mask[1 : 1 + n] = False
        if self.domain_masks:
            mask[1 : 1 + n] &= self.delta >= 6
            mask[1 + 2 * n :] &= (B.expected_state(self.belief) >= 1.15) | self.closed
        mask[0] = True
        return mask

    def observed_belief(self) -> np.ndarray:
        """Condition information available to the decision maker in the current observation mode."""
        return self._state_features()

    def _state_features(self) -> np.ndarray:
        if self.obs_mode == "belief":
            return self.belief
        if self.obs_mode == "oracle":
            return self._eye[self.state]
        return self._eye[np.where(self.closed, K - 1, self.last_obs)]

    def _obs(self) -> np.ndarray:
        per = np.empty((self.n, FEATURES_PER_BRIDGE), dtype=np.float32)
        per[:, :K] = self._state_features()
        per[:, 5] = self._importance
        per[:, 6] = np.clip(self.age / 100.0, 0.0, 1.0)
        per[:, 7] = np.minimum(self.delta, 24) / 24.0
        per[:, 8] = self._cost_index
        per[:, 9] = self.closed
        balance = 1.0 if not np.isfinite(self.budget) else self.budget / self._balance_ref
        glob = np.array([np.clip(balance, 0.0, 1.0), (self.t % 12) / 12.0, self.t / self.T], dtype=np.float32)
        return np.concatenate([per.ravel(), glob])

    def _potential(self) -> np.ndarray:
        """Per-bridge shaping potential: minus each bridge's expected monthly risk."""
        return -self.costs.expected_risk(self.belief, self.closed) / self.reward_scale

    # ------------------------------------------------------------------ transition
    def step(self, action: int):
        assert self._initialised, "call reset() first"
        a = int(action)
        t, n = self.t, self.n
        penalty = 0.0
        if not self._mask[a]:
            penalty = self.invalid_penalty
            self.stats["invalid"] += 1
            a = 0
        kind, i = A.decode(a, n)
        phi_before = self._potential() if self.shaping else np.zeros(n)

        agency = float(self.costs.action_costs[a])
        self.budget -= agency
        repaired = np.zeros(n, dtype=bool)

        if kind == A.INSPECT:
            o = int(sample_rows(self.O_indepth[self.state[i]], self._u_inspect[t]))
            self.belief[i] = B.condition(self.belief[i : i + 1], self.O_indepth[:, o])[0]
            self.last_obs[i] = o
            self.delta[i] = 0
            self.stats["inspections"] += 1
        elif kind == A.MINOR:
            c = self.state[i]
            if c in (1, 2) and self._u_repair[t] < self.minor_success:
                self.state[i] = c - 1
            self.belief[i] = self.belief[i] @ self.R_minor
            if self.last_obs[i] in (1, 2):
                self.last_obs[i] -= 1
            repaired[i] = True
            self.stats["minor"] += 1
        elif kind == A.MAJOR:
            self.state[i] = 0 if self._u_repair[t] < self.major_success else 1
            self.closed[i] = False
            self.belief[i] = self.belief[i] @ self.R_major
            self.last_obs[i] = 0
            repaired[i] = True
            self.stats["major"] += 1

        # Costs incurred this month, on the post-action condition.
        closure_vec = self.costs.closure(self.closed)
        risk_vec = self.costs.risk(self.state, self.closed)
        expected_risk_vec = self.costs.expected_risk(self.belief, self.closed)
        closure = float(closure_vec.sum())
        risk = float(risk_vec.sum())
        risk_for_reward = float(expected_risk_vec.sum()) if self.expected_reward else risk
        self.stats["closure_months"] += int(self.closed.sum())
        self.stats["critical_months"] += int((self.state == K - 1).sum())

        # Deterioration of bridges that were not repaired this month.
        aging = ~repaired
        drop = aging & (self._u_det[t] < self._q_phys[np.arange(n), self.state])
        self.state = self.state + drop
        self.belief[aging] = B.predict(self.belief[aging], self._q_filter[aging])

        # Closures are public: a newly closed bridge is known to be Critical; staying open is weak evidence.
        newly_closed = ~self.closed & (self.state == K - 1) & (self._u_close[t] < self.p_close)
        was_open = ~self.closed
        self.closed |= newly_closed
        still_open = was_open & ~self.closed
        self.belief[newly_closed] = self._eye[K - 1]
        self.belief[still_open] = B.condition(self.belief[still_open], self.open_likelihood)

        # Routine inspections on the regulatory schedule.
        self.t = t + 1
        self.age += 1.0 / 12.0
        self.delta += 1
        due = self.delta >= self.routine_interval
        if due.any():
            obs = sample_rows(self.O_routine[self.state[due]], self._u_routine[t, due])
            self.belief[due] = B.condition(self.belief[due], self.O_routine[:, obs].T)
            self.last_obs[due] = obs
            self.delta[due] = 0
            self.stats["routine_inspections"] += int(due.sum())

        self.budget = min(self.budget + self.annual_budget / 12.0, self.balance_cap)
        if self.t % 12 == 0:
            self._update_hazards()

        terminated = self.t >= self.T
        salvage = float(self.costs.salvage(self.state).sum()) if terminated else 0.0
        salvage_for_reward = salvage
        if terminated and self.expected_reward:
            salvage_for_reward = float((self.belief * self.costs.salvage_table).sum())

        total = agency + closure + risk + salvage + penalty
        for key, val in (("agency", agency), ("closure", closure), ("risk", risk), ("salvage", salvage), ("penalty", penalty)):
            self.stats[key] += val
        reward = -(agency + closure + risk_for_reward + salvage_for_reward + penalty) / self.reward_scale
        shaping_vec = np.zeros(n)
        if self.shaping:
            phi_after = np.zeros(n) if terminated else self._potential()
            shaping_vec = self.shaping_eta * (self.shaping_gamma * phi_after - phi_before)
            reward += float(shaping_vec.sum())

        # The same reward split by bridge, summing to the scalar reward: the cost of a work order is charged
        # to its bridge, an invalid-action penalty is shared equally, and shaping is per-bridge already.
        per_bridge = closure_vec + (expected_risk_vec if self.expected_reward else risk_vec) + penalty / n
        if i >= 0:
            per_bridge[i] += agency
        if terminated:
            per_bridge += (self.belief * self.costs.salvage_table).sum(axis=1) if self.expected_reward else self.costs.salvage(self.state)
        reward_vector = -per_bridge / self.reward_scale + shaping_vec

        self._mask = self._compute_mask()
        info: dict[str, Any] = {"t": self.t, "cost": total, "action_kind": kind, "bridge": i,
                                "reward_vector": reward_vector.astype(np.float32)}
        if terminated:
            info["episode"] = self.episode_summary()
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ reporting
    def episode_summary(self) -> dict[str, float]:
        s = dict(self.stats)
        s["true_cost"] = s["agency"] + s["closure"] + s["risk"] + s["salvage"]
        return s

    def render(self) -> str:
        lines = [f"month {self.t}/{self.T}  budget left ${self.budget:,.0f}"]
        exp_c = B.expected_state(self.belief)
        ent = B.entropy(self.belief)
        for j in range(self.n):
            lines.append(
                f"  {self.net.bridge_id[j]:>16s}  true={self.state[j]}  E[c]={exp_c[j]:.2f}  H={ent[j]:.2f}  "
                f"months-since-insp={self.delta[j]:2d}  {'CLOSED' if self.closed[j] else 'open'}"
            )
        return "\n".join(lines)


def make_env(overrides: dict | None = None, network: Network | None = None, config_path: str | None = None) -> BridgeNetworkEnv:
    cfg = load_env_config(config_path)
    return BridgeNetworkEnv(cfg, network=network, overrides=overrides)
