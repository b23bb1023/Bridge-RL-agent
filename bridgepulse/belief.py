"""Exact Bayesian filtering of bridge condition beliefs.

Each bridge's condition evolves independently given the actions, so the joint belief
factorises into one categorical distribution per bridge (an (N, K) array).
"""

from __future__ import annotations

import numpy as np

from bridgepulse.network import NUM_STATES

K = NUM_STATES
_STATES = np.arange(K, dtype=np.float64)


def predict(belief: np.ndarray, q: np.ndarray) -> np.ndarray:
    """One month of deterioration for beliefs (M, K) with drop probabilities (M, K)."""
    out = belief * (1.0 - q)
    out[:, 1:] += belief[:, :-1] * q[:, :-1]
    return out


def condition(belief: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Bayes update with a likelihood vector over states; rows with zero evidence are left unchanged."""
    post = belief * likelihood
    z = post.sum(axis=-1, keepdims=True)
    return np.divide(post, z, out=belief.copy(), where=z > 0.0)


def entropy(belief: np.ndarray) -> np.ndarray:
    """Shannon entropy normalised to [0, 1]."""
    p = np.clip(belief, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1) / np.log(K)


def expected_state(belief: np.ndarray) -> np.ndarray:
    return belief @ _STATES


def initial_belief(rating_state, months_since, q: np.ndarray, O_routine: np.ndarray, prior) -> np.ndarray:
    """Belief at t=0: posterior from the last recorded rating, rolled forward by the months elapsed since."""
    rating_state = np.asarray(rating_state)
    months_since = np.asarray(months_since)
    b = np.asarray(prior, dtype=np.float64)[None, :] * O_routine[:, rating_state].T
    b /= b.sum(axis=1, keepdims=True)
    for m in range(int(months_since.max(initial=0))):
        active = months_since > m
        b[active] = predict(b[active], q[active])
    return b
