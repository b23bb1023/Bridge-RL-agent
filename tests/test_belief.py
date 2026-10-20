import numpy as np

from bridgepulse import belief as B
from bridgepulse.dynamics import (
    major_repair_matrix,
    minor_repair_matrix,
    observation_matrix,
    sample_rows,
    transition_matrices,
)

K = 5
P_CLOSE = 0.25


def _random_history(rng, months, q, O_hi, O_lo, R_minor, R_major):
    """Simulate one bridge and return the event history the agent would see."""
    P = transition_matrices(q[None, :])[0]
    c = int(rng.integers(0, 3))
    closed = False
    events = []
    for _ in range(months):
        ev = {"action": rng.choice(["none", "inspect", "minor", "major"], p=[0.4, 0.3, 0.15, 0.15])}
        if closed and ev["action"] in ("inspect", "minor"):
            ev["action"] = "none"
        if ev["action"] == "inspect":
            ev["obs"] = int(sample_rows(O_hi[c], rng.random()))
        M = {"minor": R_minor, "major": R_major}.get(ev["action"], P)
        if ev["action"] == "major":
            closed = False
        c = int(sample_rows(M[c], rng.random()))
        ev["was_open"] = not closed
        if not closed and c == K - 1 and rng.random() < P_CLOSE:
            closed = True
        ev["closed"] = closed
        if rng.random() < 0.3:
            ev["routine"] = int(sample_rows(O_lo[c], rng.random()))
        events.append(ev)
    return events


def _filter(b0, events, q, O_hi, O_lo, R_minor, R_major):
    b = b0[None, :].copy()
    open_lik = np.array([1, 1, 1, 1, 1 - P_CLOSE])
    for ev in events:
        if ev["action"] == "inspect":
            b = B.condition(b, O_hi[:, ev["obs"]])
        if ev["action"] == "minor":
            b = b @ R_minor
        elif ev["action"] == "major":
            b = b @ R_major
        else:
            b = B.predict(b, q[None, :])
        if ev["was_open"]:
            b = np.eye(K)[[K - 1]] if ev["closed"] else B.condition(b, open_lik)
        if "routine" in ev:
            b = B.condition(b, O_lo[:, ev["routine"]])
    return b[0]


def _enumerate(b0, events, q, O_hi, O_lo, R_minor, R_major):
    """Posterior of the final state by summing over every hidden state path."""
    P = transition_matrices(q[None, :])[0]
    m = len(events)
    paths = np.indices((K,) * (m + 1)).reshape(m + 1, -1).T
    w = b0[paths[:, 0]].copy()
    for t, ev in enumerate(events):
        c, c_next = paths[:, t], paths[:, t + 1]
        if ev["action"] == "inspect":
            w *= O_hi[c, ev["obs"]]
        M = {"minor": R_minor, "major": R_major}.get(ev["action"], P)
        w *= M[c, c_next]
        if ev["was_open"]:
            p_close = P_CLOSE * (c_next == K - 1)
            w *= p_close if ev["closed"] else 1.0 - p_close
        if "routine" in ev:
            w *= O_lo[c_next, ev["routine"]]
    post = np.bincount(paths[:, -1], weights=w, minlength=K)
    return post / post.sum()


def test_filter_matches_brute_force_enumeration():
    rng = np.random.default_rng(7)
    q = np.array([0.15, 0.2, 0.25, 0.3, 0.0])
    O_hi, O_lo = observation_matrix(0.45), observation_matrix(0.65)
    R_minor, R_major = minor_repair_matrix(0.9), major_repair_matrix(0.95)
    for _ in range(25):
        b0 = rng.dirichlet(np.ones(K))
        events = _random_history(rng, 6, q, O_hi, O_lo, R_minor, R_major)
        got = _filter(b0, events, q, O_hi, O_lo, R_minor, R_major)
        want = _enumerate(b0, events, q, O_hi, O_lo, R_minor, R_major)
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_inspection_reduces_entropy_on_average():
    rng = np.random.default_rng(3)
    O = observation_matrix(0.45)
    for _ in range(20):
        prior = rng.dirichlet(np.ones(K))
        predictive = prior @ O
        post_entropy = sum(predictive[o] * B.entropy(B.condition(prior[None], O[:, o]))[0] for o in range(K))
        assert post_entropy <= B.entropy(prior[None])[0] + 1e-12


def test_initial_belief_rolls_forward_and_normalises():
    q = np.tile([0.02, 0.02, 0.02, 0.02, 0.0], (3, 1))
    O = observation_matrix(0.65)
    b = B.initial_belief(np.array([1, 2, 3]), np.array([0, 12, 23]), q, O, np.full(K, 0.2))
    np.testing.assert_allclose(b.sum(axis=1), 1.0)
    assert B.expected_state(b)[2] > B.expected_state(b)[1] > B.expected_state(b)[0]
