import numpy as np

from bridgepulse.dynamics import (
    DeteriorationModel,
    major_repair_matrix,
    minor_repair_matrix,
    observation_matrix,
    sample_rows,
    transition_matrices,
)


def test_observation_matrix_rows_and_accuracy():
    O = observation_matrix(0.45)
    np.testing.assert_allclose(O.sum(axis=1), 1.0)
    assert 0.84 < O[2, 2] < 0.87  # in-depth inspection: ~85% exact for interior states
    assert O[0, 0] > O[2, 2]      # edge states have only one neighbour to confuse with
    O_routine = observation_matrix(0.65)
    assert 0.59 < O_routine[2, 2] < 0.64


def test_repair_matrices():
    R_minor = minor_repair_matrix(0.9)
    R_major = major_repair_matrix(0.95)
    for R in (R_minor, R_major):
        np.testing.assert_allclose(R.sum(axis=1), 1.0)
    np.testing.assert_allclose(R_minor[[0, 3, 4]], np.eye(5)[[0, 3, 4]])
    assert R_minor[1, 0] == R_minor[2, 1] == 0.9
    np.testing.assert_allclose(R_major[:, 0], 0.95)


def test_deterioration_matrices_are_stochastic_and_monotone():
    model = DeteriorationModel(
        {"base_monthly_drop": [0.01, 0.005, 0.012, 0.02], "age_coef": 0.4, "adt_coef": 0.3, "age_ref_years": 50}
    )
    q_young = model.drop_probs(np.array([10.0]), np.array([0.5]))
    q_old = model.drop_probs(np.array([90.0]), np.array([0.5]))
    assert np.all(q_old[:, :4] > q_young[:, :4])
    P = transition_matrices(q_old)
    np.testing.assert_allclose(P.sum(axis=2), 1.0)
    assert P[0, 4, 4] == 1.0
    assert np.all(np.triu(P[0], 2) == 0.0) and np.all(np.tril(P[0], -1) == 0.0)


def test_inverse_cdf_sampling_matches_probabilities():
    rng = np.random.default_rng(0)
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    draws = sample_rows(np.tile(p, (200_000, 1)), rng.random(200_000))
    freq = np.bincount(draws, minlength=5) / draws.size
    np.testing.assert_allclose(freq, p, atol=0.005)
