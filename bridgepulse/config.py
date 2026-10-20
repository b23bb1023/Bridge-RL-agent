"""Configuration loading: YAML files, nested overrides and fitted deterioration parameters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, overrides: dict | None) -> dict:
    """Return a copy of `base` with `overrides` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the environment config, then fitted deterioration parameters, the calibrated budget and overrides."""
    cfg = load_yaml(path or "configs/env.yaml")
    calibration = resolve("configs/calibration.yaml")
    if path is None and calibration.exists():
        cfg = deep_update(cfg, load_yaml(calibration))
    fit_file = cfg.get("deterioration", {}).get("fit_file")
    if fit_file and resolve(fit_file).exists():
        with open(resolve(fit_file), "r", encoding="utf-8") as f:
            fit = json.load(f)
        det = cfg["deterioration"]
        det["base_monthly_drop"] = fit["base_monthly_drop"]
        det["age_coef"] = fit["age_coef"]
        det["adt_coef"] = fit["adt_coef"]
        det["age_ref_years"] = fit.get("age_ref_years", det["age_ref_years"])
        det["fitted"] = True
        if "routine_sigma" in fit:
            cfg["inspection"]["routine_sigma"] = fit["routine_sigma"]
        if "state_distribution" in fit:
            cfg["initial_state"]["prior"] = fit["state_distribution"]
    return deep_update(cfg, overrides)
