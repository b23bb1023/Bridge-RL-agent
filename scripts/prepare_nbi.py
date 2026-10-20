"""Build the bridge network and fit the deterioration model from two National Bridge Inventory releases.

Inputs are two FHWA NBI delimited files for the same state, e.g. PA22.txt and PA24.txt from
https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm (save them under data/raw/).

Outputs (data/processed/):
  transition_fit.json  fitted monthly hazards, age and traffic effects, fit diagnostics
  network_10.csv       the 10-bridge case-study network (3 arterial, 4 collector, 3 local)
  pool.csv             held-out bridges for out-of-distribution network tests

The fit is noise-aware: both recorded ratings are treated as noisy routine inspections of
hidden states, linked by the monthly deterioration chain over the actual gap between the
two inspection dates. This is a hidden-Markov likelihood maximised over the hazard parameters.

Example:
  python scripts/prepare_nbi.py --release-a data/raw/PA22.txt --release-b data/raw/PA24.txt --as-of 2024-03 --fit-noise
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgepulse.belief import predict  # noqa: E402
from bridgepulse.dynamics import hazard_multiplier, monthly_drop_probs, observation_matrix  # noqa: E402
from bridgepulse.network import NUM_STATES, traffic_weight  # noqa: E402

K = NUM_STATES
AGE_REF = 50.0
ITEMS = {
    "id": "008", "record_type": "005A", "detour_km": "019", "year_built": "027", "adt": "029",
    "kind": "043A", "length_m": "049", "width_m": "052", "deck": "058", "superstructure": "059",
    "substructure": "060", "inspected": "090", "year_reconstructed": "106", "truck_pct": "109",
}


# ---------------------------------------------------------------------------- loading
def _find_column(columns: list[str], item: str) -> str:
    pattern = re.compile(rf"(^|_){re.escape(item)}$", re.IGNORECASE)
    hits = [c for c in columns if pattern.search(c.strip())]
    if not hits:
        raise KeyError(f"no column for NBI item {item}")
    return hits[0]


def _parse_inspection_month(code: pd.Series) -> pd.Series:
    """Item 90 is MMYY; return months since year 0 (NaN when unparseable)."""
    s = code.astype(str).str.strip().str.zfill(4)
    mm = pd.to_numeric(s.str[:2], errors="coerce")
    yy = pd.to_numeric(s.str[2:4], errors="coerce")
    year = np.where(yy <= 60, 2000 + yy, 1900 + yy)
    out = year * 12 + (mm - 1)
    return pd.Series(np.where((mm >= 1) & (mm <= 12), out, np.nan), index=code.index)


def load_release(path: str | Path) -> pd.DataFrame:
    # Records with stray unquoted commas in free-text fields cannot be parsed reliably; they are skipped.
    raw = pd.read_csv(path, dtype=str, quotechar="'", low_memory=False, encoding_errors="replace", on_bad_lines="skip")
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f) - 1
    if n_lines != len(raw):
        print(f"{Path(path).name}: skipped {n_lines - len(raw)} malformed record(s)")
    raw.columns = [c.strip().strip("'\"") for c in raw.columns]
    cols = list(raw.columns)
    df = pd.DataFrame({name: raw[_find_column(cols, item)] for name, item in ITEMS.items()})
    df["id"] = df["id"].str.strip()
    for name in ("detour_km", "year_built", "adt", "length_m", "width_m", "year_reconstructed", "truck_pct"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    for name in ("deck", "superstructure", "substructure"):
        df[name] = pd.to_numeric(df[name].str.strip(), errors="coerce")
    df["inspected_month"] = _parse_inspection_month(df["inspected"])
    df["min_rating"] = df[["deck", "superstructure", "substructure"]].min(axis=1, skipna=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep highway bridges (not culverts) with complete ratings and usable geometry."""
    keep = (
        df["record_type"].astype(str).str.strip().eq("1")
        & df["min_rating"].between(0, 9)
        & df["year_built"].between(1850, 2030)
        & (df["adt"] > 0)
        & (df["length_m"] > 6)
        & (df["width_m"] > 3)
        & df["inspected_month"].notna()
    )
    out = df.loc[keep].copy()
    out["deck_area_m2"] = out["length_m"] * out["width_m"]
    return out.drop_duplicates("id")


def rating_state(r: pd.Series) -> np.ndarray:
    r = r.to_numpy(dtype=np.int64)
    return np.select([r >= 8, r >= 6, r == 5, r == 4], [0, 1, 2, 3], default=4)


# ---------------------------------------------------------------------------- transition pairs
def build_pairs(a: pd.DataFrame, b: pd.DataFrame, min_gap: int = 12, max_gap: int = 36) -> tuple[pd.DataFrame, dict]:
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    gap = (m["inspected_month_b"] - m["inspected_month_a"]).astype(float)
    recon_a = m["year_reconstructed_a"].fillna(0)
    recon_b = m["year_reconstructed_b"].fillna(0)
    no_rebuild = recon_a.eq(recon_b)
    ok = gap.between(min_gap, max_gap) & no_rebuild
    pairs = pd.DataFrame({
        "s_a": rating_state(m.loc[ok, "min_rating_a"]),
        "s_b": rating_state(m.loc[ok, "min_rating_b"]),
        "gap": gap[ok].astype(int).to_numpy(),
        "age": (m.loc[ok, "inspected_month_a"] / 12.0 - np.maximum(m.loc[ok, "year_built_a"], recon_a[ok])).to_numpy(),
        "w": traffic_weight(m.loc[ok, "adt_a"].to_numpy()),
    })
    pairs = pairs[pairs["age"].between(0, 150)].reset_index(drop=True)
    diag = {
        "matched_records": int(len(m)),
        "pairs_used": int(len(pairs)),
        "excluded_rebuilt": int((~no_rebuild).sum()),
        "excluded_gap": int((~gap.between(min_gap, max_gap)).sum()),
        "rating_improvements_in_pairs": int((pairs["s_b"] < pairs["s_a"]).sum()),
    }
    return pairs, diag


# ---------------------------------------------------------------------------- likelihood
# Parameter vector: log base hazards (4), age and traffic effects (2), logits of the hidden-state
# distribution at the first inspection (4, the last state is the reference), optional routine noise (1).
def _unpack(theta: np.ndarray, fit_noise: bool, sigma_fixed: float):
    q = np.exp(theta[:4])
    b_age, b_adt = theta[4], theta[5]
    logits = np.concatenate([theta[6:10], [0.0]])
    prior = np.exp(logits - logits.max())
    prior /= prior.sum()
    sigma = theta[10] if fit_noise else sigma_fixed
    return q, b_age, b_adt, prior, sigma


def neg_log_likelihood(theta: np.ndarray, pairs: pd.DataFrame, fit_noise: bool, sigma_fixed: float, noisy: bool = True) -> float:
    """-log P(rating_a, rating_b) summed over pairs: both ratings are noisy views of hidden states."""
    q_base, b_age, b_adt, prior, sigma = _unpack(theta, fit_noise, sigma_fixed)
    mult = hazard_multiplier(pairs["age"].to_numpy(), pairs["w"].to_numpy(), b_age, b_adt, AGE_REF)
    q = monthly_drop_probs(q_base, mult)
    s_a = pairs["s_a"].to_numpy()
    s_b = pairs["s_b"].to_numpy()
    gap = pairs["gap"].to_numpy()

    if noisy:
        O = observation_matrix(sigma)
        dist = prior[None, :] * O[:, s_a].T          # joint P(hidden state, first rating)
    else:
        dist = np.eye(K)[s_a]
    for step in range(int(gap.max())):
        active = gap > step
        dist[active] = predict(dist[active], q[active])
    if noisy:
        lik = (dist * O[:, s_b].T).sum(axis=1)
    else:
        lik = dist[np.arange(len(s_b)), s_b]
    return -float(np.log(np.clip(lik, 1e-300, None)).sum())


def fit_transitions(pairs: pd.DataFrame, sigma_routine: float, fit_noise: bool, noisy: bool = True) -> dict:
    theta0 = np.concatenate([np.log([0.008, 0.005, 0.012, 0.018]), [0.3, 0.2], [2.0, 3.0, 2.5, 1.5]]
                            + ([[sigma_routine]] if fit_noise else []))
    bounds = ([(np.log(1e-5), np.log(0.3))] * 4 + [(-3.0, 3.0), (-3.0, 3.0)] + [(-8.0, 8.0)] * 4
              + ([(0.2, 1.5)] if fit_noise else []))
    res = minimize(neg_log_likelihood, theta0, args=(pairs, fit_noise, sigma_routine, noisy), method="L-BFGS-B", bounds=bounds)
    q, b_age, b_adt, prior, sigma = _unpack(res.x, fit_noise, sigma_routine)
    return {
        "base_monthly_drop": [float(v) for v in q],
        "age_coef": float(b_age),
        "adt_coef": float(b_adt),
        "age_ref_years": AGE_REF,
        "routine_sigma": float(sigma),
        "state_distribution": [float(v) for v in prior],
        "noise_aware": bool(noisy),
        "neg_log_likelihood": float(res.fun),
        "converged": bool(res.success),
        "implied_mean_years_in_state": [float(1.0 / (12.0 * v)) for v in q],
    }


def fit_table(pairs: pd.DataFrame, fit: dict) -> list[dict]:
    """Observed vs model-predicted rating transition frequencies, by starting rating state."""
    rows = []
    O = observation_matrix(fit["routine_sigma"])
    mult = hazard_multiplier(pairs["age"].to_numpy(), pairs["w"].to_numpy(), fit["age_coef"], fit["adt_coef"], AGE_REF)
    q = monthly_drop_probs(np.asarray(fit["base_monthly_drop"]), mult)
    prior = np.asarray(fit["state_distribution"])
    for s in range(K):
        sel = pairs["s_a"].to_numpy() == s
        if sel.sum() < 20:
            continue
        sub = pairs[sel]
        dist = prior[None, :] * O[:, sub["s_a"].to_numpy()].T
        dist /= dist.sum(axis=1, keepdims=True)
        gap = sub["gap"].to_numpy()
        for step in range(int(gap.max())):
            active = gap > step
            dist[active] = predict(dist[active], q[sel][active])
        pred = (dist @ O).mean(axis=0)
        obs = np.bincount(sub["s_b"], minlength=K) / len(sub)
        rows.append({"from_state": s, "n": int(sel.sum()), "observed": obs.round(4).tolist(), "predicted": pred.round(4).tolist()})
    return rows


# ---------------------------------------------------------------------------- network selection
STRATA = {
    "arterial": (lambda d: d["adt"] >= 25_000, 3),
    "collector": (lambda d: (d["adt"] >= 5_000) & (d["adt"] < 25_000) & (d["age_years"] >= 40), 4),
    "local": (lambda d: d["adt"] < 5_000, 3),
}


def select_network(b: pd.DataFrame, as_of_month: int, seed: int, pool_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    d = b.copy()
    d["age_years"] = (as_of_month / 12.0 - np.maximum(d["year_built"], d["year_reconstructed"].fillna(0))).round(1)
    d["months_since_inspection"] = np.clip(as_of_month - d["inspected_month"], 0, 23).astype(int)
    d["state"] = rating_state(d["min_rating"])
    # District maintenance scale: very large structures are usually separate capital projects.
    d = d[d["state"].between(1, 3) & d["age_years"].between(5, 120) & d["deck_area_m2"].between(60, 2_500)]
    # The case study is the set of bridges a district would worry about: Fair and Poor bridges are
    # strongly favoured, so the network starts with a maintenance backlog.
    pick_weight = d["state"].map({1: 0.5, 2: 2.0, 3: 6.0}).to_numpy()

    chosen, pool = [], []
    for name, (rule, count) in STRATA.items():
        cand = d[rule(d)].copy()
        if len(cand) < count:
            raise RuntimeError(f"not enough {name} bridges in the release")
        w = pick_weight[d.index.get_indexer(cand.index)]
        idx = rng.choice(len(cand), size=count, replace=False, p=w / w.sum())
        sel = cand.iloc[idx].assign(archetype=name)
        chosen.append(sel)
        rest = cand.drop(sel.index)
        n_pool = min(len(rest), int(round(pool_size * count / 10)))
        pool.append(rest.sample(n=n_pool, random_state=seed).assign(archetype=name))

    cols = ["id", "archetype", "adt", "truck_pct", "detour_km", "length_m", "width_m", "deck_area_m2",
            "age_years", "min_rating", "months_since_inspection", "year_built"]

    def finish(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[cols].rename(columns={"id": "bridge_id"}).copy()
        out["truck_pct"] = out["truck_pct"].fillna(8.0)
        out["detour_km"] = out["detour_km"].fillna(10.0)
        out["min_rating"] = out["min_rating"].astype(int)
        out["source"] = "FHWA NBI"
        return out.reset_index(drop=True)

    return finish(pd.concat(chosen)), finish(pd.concat(pool))


# ---------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-a", required=True, help="older NBI delimited file")
    parser.add_argument("--release-b", required=True, help="newer NBI delimited file (also used for the network)")
    parser.add_argument("--as-of", default="2024-12", help="date the network snapshot represents (YYYY-MM)")
    parser.add_argument("--routine-sigma", type=float, default=0.65)
    parser.add_argument("--fit-noise", action="store_true", help="also estimate the routine-inspection noise")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pool-size", type=int, default=300)
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    a = clean(load_release(args.release_a))
    b = clean(load_release(args.release_b))
    pairs, diag = build_pairs(a, b)
    print(f"release A: {len(a)} bridges, release B: {len(b)} bridges, transition pairs: {len(pairs)}")

    fit = fit_transitions(pairs, args.routine_sigma, args.fit_noise, noisy=True)
    naive = fit_transitions(pairs[pairs["s_b"] >= pairs["s_a"]], args.routine_sigma, False, noisy=False)
    fit["diagnostics"] = diag
    fit["rating_distribution_release_a"] = (np.bincount(pairs["s_a"], minlength=K) / len(pairs)).round(4).tolist()
    fit["fit_table"] = fit_table(pairs, fit)
    fit["noise_free_fit_for_comparison"] = {k: naive[k] for k in ("base_monthly_drop", "age_coef", "adt_coef", "implied_mean_years_in_state")}
    fit["data"] = {"release_a": Path(args.release_a).name, "release_b": Path(args.release_b).name}

    year, month = (int(x) for x in args.as_of.split("-"))
    network, pool = select_network(b, year * 12 + month - 1, args.seed, args.pool_size)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "transition_fit.json").write_text(json.dumps(fit, indent=2))
    network.to_csv(out / "network_10.csv", index=False)
    pool.to_csv(out / "pool.csv", index=False)
    print(json.dumps({k: fit[k] for k in ("base_monthly_drop", "age_coef", "adt_coef", "routine_sigma", "implied_mean_years_in_state")}, indent=2))
    print(network.to_string(index=False))


if __name__ == "__main__":
    main()
