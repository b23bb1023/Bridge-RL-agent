"""Condition dynamics: deterioration hazards, repair effects and inspection noise.

Deterioration is a one-step monthly Markov chain: an un-repaired bridge in state k
drops to k+1 with probability q_k * h(age, traffic), otherwise stays. State 4 is absorbing.
The same model drives the simulator and the agent's belief filter.
"""

from __future__ import annotations

import numpy as np

from bridgepulse.network import NUM_STATES

K = NUM_STATES


def hazard_multiplier(age_years, weight, age_coef: float, adt_coef: float, age_ref: float) -> np.ndarray:
    """Proportional-hazard multiplier: older and busier bridges deteriorate faster."""
    age = np.asarray(age_years, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    return np.exp(age_coef * (age - age_ref) / age_ref + adt_coef * (w - 0.5))


def monthly_drop_probs(base_drop, multiplier, cap: float = 0.5) -> np.ndarray:
    """Per-bridge drop probabilities, shape (N, K); the last column (absorbing state) is zero."""
    base = np.asarray(base_drop, dtype=np.float64)
    q = np.zeros((np.size(multiplier), K), dtype=np.float64)
    q[:, : K - 1] = np.minimum(cap, base[None, :] * np.asarray(multiplier, dtype=np.float64)[:, None])
    return q


def transition_matrices(q: np.ndarray) -> np.ndarray:
    """Expand drop probabilities (N, K) into row-stochastic matrices (N, K, K)."""
    n = q.shape[0]
    P = np.zeros((n, K, K), dtype=np.float64)
    idx = np.arange(K)
    P[:, idx, idx] = 1.0 - q
    P[:, idx[:-1], idx[1:]] = q[:, :-1]
    return P


def observation_matrix(sigma: float) -> np.ndarray:
    """Inspection likelihood O[c, o] = P(observe o | true state c): a discretised Gaussian around c."""
    idx = np.arange(K, dtype=np.float64)
    O = np.exp(-((idx[None, :] - idx[:, None]) ** 2) / (2.0 * sigma**2))
    return O / O.sum(axis=1, keepdims=True)


def minor_repair_matrix(success: float) -> np.ndarray:
    """Minor repair restores one state for Good/Fair bridges and does nothing otherwise."""
    R = np.eye(K)
    for c in (1, 2):
        R[c, c] = 1.0 - success
        R[c, c - 1] = success
    return R


def major_repair_matrix(success: float) -> np.ndarray:
    """Major repair returns any bridge to Excellent, or to Good if the work falls short."""
    R = np.zeros((K, K))
    R[:, 0] = success
    R[:, 1] = 1.0 - success
    return R


def sample_rows(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling of one category per row of `probs` from uniforms `u`."""
    cdf = np.cumsum(probs, axis=-1)
    cdf[..., -1] = 1.0
    return (np.asarray(u)[..., None] > cdf).sum(axis=-1)


class DeteriorationModel:
    """Holds the hazard parameters and produces per-bridge monthly drop probabilities."""

    def __init__(self, det_cfg: dict, scale: float = 1.0) -> None:
        self.base_drop = np.asarray(det_cfg["base_monthly_drop"], dtype=np.float64)
        self.age_coef = float(det_cfg["age_coef"])
        self.adt_coef = float(det_cfg["adt_coef"])
        self.age_ref = float(det_cfg["age_ref_years"])
        self.cap = float(det_cfg.get("max_monthly_drop", 0.5))
        self.scale = float(scale)

    def drop_probs(self, age_years, weight) -> np.ndarray:
        mult = self.scale * hazard_multiplier(age_years, weight, self.age_coef, self.adt_coef, self.age_ref)
        return monthly_drop_probs(self.base_drop, mult, self.cap)
