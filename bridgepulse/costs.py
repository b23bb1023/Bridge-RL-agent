"""Monetary cost model: agency spending, road-user closure costs, structural risk and end-of-horizon salvage."""

from __future__ import annotations

import numpy as np

from bridgepulse.network import Network

DAYS_PER_MONTH = 30.4


class CostModel:
    def __init__(self, network: Network, cost_cfg: dict) -> None:
        area = network.deck_area_m2
        self.inspect = cost_cfg["inspection_per_m2"] * area
        self.minor = cost_cfg["minor_per_m2"] * area
        self.major = cost_cfg["major_per_m2"] * area

        detour = np.clip(network.detour_km, cost_cfg["min_detour_km"], cost_cfg["max_detour_km"])
        per_km = (1.0 - network.truck_frac) * cost_cfg["car_cost_per_km"] + network.truck_frac * cost_cfg["truck_cost_per_km"]
        self.closure_month = network.adt * DAYS_PER_MONTH * detour * per_km

        self.risk_scores = np.asarray(cost_cfg["risk_scores"], dtype=np.float64)
        self.risk_base = cost_cfg["risk_rate"] * (self.major + self.closure_month)

        # Cost to bring a bridge back to Good condition at the end of the horizon.
        self.salvage_table = np.stack(
            [np.zeros_like(area), np.zeros_like(area), self.minor, self.major, self.major], axis=1
        )
        self.reward_scale = float(cost_cfg["reward_scale"])

        n = network.n
        self.action_costs = np.concatenate([[0.0], self.inspect, self.minor, self.major])
        assert self.action_costs.shape == (1 + 3 * n,)

    def risk(self, state: np.ndarray, closed: np.ndarray) -> np.ndarray:
        """Monthly expected loss of each open bridge given its true state."""
        return self.risk_scores[state] * self.risk_base * (~closed)

    def expected_risk(self, belief: np.ndarray, closed: np.ndarray) -> np.ndarray:
        return (belief @ self.risk_scores) * self.risk_base * (~closed)

    def closure(self, closed: np.ndarray) -> np.ndarray:
        return self.closure_month * closed

    def salvage(self, state: np.ndarray) -> np.ndarray:
        return self.salvage_table[np.arange(state.size), state]
