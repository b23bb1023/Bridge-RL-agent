"""Bridge network description: static attributes of every bridge in the managed portfolio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bridgepulse.config import resolve

NUM_STATES = 5
STATE_NAMES = ("Excellent", "Good", "Fair", "Poor", "Critical")


def nbi_rating_to_state(rating: float) -> int:
    """Map an NBI 0-9 condition rating to the 5 model states (0 = best, 4 = worst)."""
    r = int(rating)
    if r >= 8:
        return 0
    if r >= 6:
        return 1
    if r == 5:
        return 2
    if r == 4:
        return 3
    return 4


def traffic_weight(adt: np.ndarray) -> np.ndarray:
    """Log-scaled traffic exposure in [0, 1]; 100k vehicles/day maps to 1."""
    return np.clip(np.log10(np.asarray(adt, dtype=np.float64) + 1.0) / 5.0, 0.0, 1.0)


@dataclass(frozen=True)
class Network:
    bridge_id: tuple[str, ...]
    archetype: tuple[str, ...]
    adt: np.ndarray
    truck_frac: np.ndarray
    detour_km: np.ndarray
    deck_area_m2: np.ndarray
    age_years: np.ndarray
    rating_state: np.ndarray
    months_since_inspection: np.ndarray
    source: str

    @property
    def n(self) -> int:
        return len(self.bridge_id)

    @property
    def weight(self) -> np.ndarray:
        return traffic_weight(self.adt)

    def subset(self, idx) -> "Network":
        idx = np.asarray(idx)
        return Network(
            bridge_id=tuple(np.asarray(self.bridge_id)[idx]),
            archetype=tuple(np.asarray(self.archetype)[idx]),
            adt=self.adt[idx],
            truck_frac=self.truck_frac[idx],
            detour_km=self.detour_km[idx],
            deck_area_m2=self.deck_area_m2[idx],
            age_years=self.age_years[idx],
            rating_state=self.rating_state[idx],
            months_since_inspection=self.months_since_inspection[idx],
            source=self.source,
        )


REQUIRED_COLUMNS = (
    "bridge_id", "archetype", "adt", "truck_pct", "detour_km", "deck_area_m2",
    "age_years", "min_rating", "months_since_inspection",
)


def network_from_frame(df: pd.DataFrame, source: str = "") -> Network:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"network table is missing columns: {missing}")
    if not source and "source" in df.columns:
        source = str(df["source"].iloc[0])
    return Network(
        bridge_id=tuple(df["bridge_id"].astype(str)),
        archetype=tuple(df["archetype"].astype(str)),
        adt=df["adt"].to_numpy(dtype=np.float64),
        truck_frac=np.clip(df["truck_pct"].to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0),
        detour_km=df["detour_km"].to_numpy(dtype=np.float64),
        deck_area_m2=df["deck_area_m2"].to_numpy(dtype=np.float64),
        age_years=df["age_years"].to_numpy(dtype=np.float64),
        rating_state=np.array([nbi_rating_to_state(r) for r in df["min_rating"]], dtype=np.int64),
        months_since_inspection=df["months_since_inspection"].to_numpy(dtype=np.int64),
        source=source,
    )


def load_network(path: str) -> Network:
    df = pd.read_csv(resolve(path))
    return network_from_frame(df)
