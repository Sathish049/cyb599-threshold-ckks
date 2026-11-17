"""Synthetic multi-site heart-failure readmission cohort.

No real patient data is used. Each site draws from a slightly shifted
distribution so that the sites are heterogeneous, as real hospitals are.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["age", "prior_admissions", "length_of_stay", "creatinine",
            "hemoglobin", "ejection_fraction", "diabetes"]

# Public clipping bounds agreed in Phase 0 (Part 2, Section 3.1)
LOWER = np.array([30, 0, 1, 0.4, 7, 10, 0], dtype=float)
UPPER = np.array([100, 10, 30, 5.0, 18, 75, 1], dtype=float)

# Ground-truth logistic model used only to generate outcomes
_TRUE_COEF = np.array([-2.6, 0.02, 0.35, 0.05, 0.4, -0.08, -0.015, 0.3])


def make_site(n: int, shift: float, rng: np.random.Generator):
    """Return (X, y) for one site: X is n x 7 raw features, y is 0/1 readmission."""
    age = rng.normal(70 + shift, 10, n)
    prior = rng.poisson(1.2 + 0.1 * shift, n).astype(float)
    los = rng.gamma(2.0, 2.5, n)
    creat = rng.lognormal(0.1, 0.35, n)
    hgb = rng.normal(12.5, 1.8, n)
    ef = rng.normal(40, 12, n)
    dm = (rng.random(n) < 0.35).astype(float)
    X = np.clip(np.c_[age, prior, los, creat, hgb, ef, dm], LOWER, UPPER)
    z = _TRUE_COEF[0] + X @ _TRUE_COEF[1:]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    return X, y


def make_cohort(num_sites: int = 3, n_per_site: int = 5000, seed: int = 2026):
    """List of (X, y) per site. Site shifts spread evenly over [-3, 3] years of age."""
    rng = np.random.default_rng(seed)
    shifts = np.linspace(-3, 3, num_sites) if num_sites > 1 else [0.0]
    return [make_site(n_per_site, float(s), rng) for s in shifts]


def scale_public(X: np.ndarray) -> np.ndarray:
    """Min-max scaling to [0, 1] with the public bounds (Part 2 design, D2)."""
    return (X - LOWER) / (UPPER - LOWER)


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize with the pooled mean/std released by the Mode A round."""
    return (X - mean) / std


def with_intercept(Z: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(Z)), Z]
