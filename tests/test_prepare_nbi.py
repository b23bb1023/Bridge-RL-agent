"""The NBI pipeline recovers known deterioration parameters from two synthetic releases in NBI file format."""

import numpy as np
import pandas as pd

from bridgepulse.dynamics import hazard_multiplier, monthly_drop_probs, observation_matrix, sample_rows
from scripts.prepare_nbi import build_pairs, clean, fit_transitions, load_release

TRUE_Q = np.array([0.010, 0.006, 0.014, 0.020])
TRUE_AGE, TRUE_ADT, SIGMA = 0.4, 0.3, 0.65
STATE_TO_RATING = np.array([8, 6, 5, 4, 3])


def _write_release(path, ids, ratings, inspected_month, year_built, adt):
    mm = inspected_month % 12 + 1
    yy = (inspected_month // 12) % 100
    df = pd.DataFrame({
        "STATE_CODE_001": "42",
        "STRUCTURE_NUMBER_008": ids,
        "RECORD_TYPE_005A": "1",
        "DETOUR_KILOS_019": 12,
        "YEAR_BUILT_027": year_built,
        "ADT_029": adt,
        "STRUCTURE_KIND_043A": "3",
        "STRUCTURE_LEN_MT_049": 40.0,
        "DECK_WIDTH_MT_052": 12.0,
        "DECK_COND_058": ratings,
        "SUPERSTRUCTURE_COND_059": np.minimum(ratings + 1, 9),
        "SUBSTRUCTURE_COND_060": np.minimum(ratings + 1, 9),
        "DATE_OF_INSPECT_090": [f"{m:02d}{y:02d}" for m, y in zip(mm, yy)],
        "YEAR_RECONSTRUCTED_106": 0,
        "PERCENT_ADT_TRUCK_109": 8,
    })
    df.to_csv(path, index=False, quotechar="'")


def test_noise_aware_fit_recovers_parameters(tmp_path):
    rng = np.random.default_rng(0)
    n = 6000
    ids = np.array([f"{k:015d}" for k in range(n)])
    year_built = rng.integers(1940, 2010, n)
    adt = np.exp(rng.uniform(np.log(200), np.log(80_000), n)).astype(int)
    start = 2022 * 12 + rng.integers(0, 12, n)
    gap = rng.integers(18, 31, n)
    age = start / 12.0 - year_built
    w = np.log10(adt + 1) / 5
    q = monthly_drop_probs(TRUE_Q, hazard_multiplier(age, w, TRUE_AGE, TRUE_ADT, 50.0))

    c0 = rng.choice(5, size=n, p=[0.15, 0.5, 0.25, 0.08, 0.02])
    c = c0.copy()
    for m in range(int(gap.max())):
        active = gap > m
        move = active & (rng.random(n) < q[np.arange(n), c])
        c = c + move
    O = observation_matrix(SIGMA)
    r_a = STATE_TO_RATING[sample_rows(O[c0], rng.random(n))]
    r_b = STATE_TO_RATING[sample_rows(O[c], rng.random(n))]

    _write_release(tmp_path / "A.txt", ids, r_a, start, year_built, adt)
    _write_release(tmp_path / "B.txt", ids, r_b, start + gap, year_built, adt)
    a = clean(load_release(tmp_path / "A.txt"))
    b = clean(load_release(tmp_path / "B.txt"))
    pairs, diag = build_pairs(a, b)
    assert diag["pairs_used"] > 0.95 * n

    fit = fit_transitions(pairs, SIGMA, fit_noise=False)
    got = np.array(fit["base_monthly_drop"])
    # Hazards for the well-populated states come back within 30%, the covariate effects keep their sign,
    # and the hidden-state distribution is recovered despite the rating noise.
    assert np.all(np.abs(got[:3] / TRUE_Q[:3] - 1) < 0.30), got
    assert fit["age_coef"] > 0 and fit["adt_coef"] > 0
    np.testing.assert_allclose(fit["state_distribution"], [0.15, 0.5, 0.25, 0.08, 0.02], atol=0.04)
