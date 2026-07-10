"""Parametric per-compound degradation curves.

ratio(age) = c0 + c1*age + c2*age^2 on fuel-corrected pace ratios
(fuel_corrected_s / reference_s). Live fits are quadratic ridge regressions on
this race's clean laps, shrunk toward a historical (track, compound) prior;
they give the simulator smooth, extrapolatable stint curves.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradationCurve:
    c0: float
    c1: float
    c2: float
    n_laps: int  # live laps behind the fit (0 = pure prior)

    def ratio(self, age: np.ndarray | float) -> np.ndarray | float:
        return self.c0 + self.c1 * np.asarray(age, dtype=float) + self.c2 * np.square(age)

    def coeffs(self) -> np.ndarray:
        return np.array([self.c0, self.c1, self.c2])


_PRIOR_AGES = np.linspace(0.0, 30.0, 13)  # pseudo-observation grid for shrinkage


def _design(ages: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(ages)), ages, ages**2])


def fit_curve(
    ages: np.ndarray,
    ratios: np.ndarray,
    prior: DegradationCurve | None = None,
    prior_weight: float = 8.0,
    ridge_alpha: float = 1.0,
) -> DegradationCurve:
    """Quadratic fit shrunk toward a prior in prediction space.

    The prior enters as pseudo-observations: 13 points sampled from the prior
    curve over ages 0..30 carrying `prior_weight` laps of total weight, so
    "8 pseudo-laps vs n live laps" means exactly that. Without a prior, a
    ridge pull toward "flat at the observed mean" keeps 1-2 laps from making
    a wild parabola. c2 is clipped at >= 0 so extrapolating past observed
    ages never predicts a tyre getting faster forever.
    """
    ages = np.asarray(ages, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    n = len(ages)
    if n == 0:
        if prior is not None:
            return DegradationCurve(prior.c0, prior.c1, prior.c2, 0)
        return DegradationCurve(1.0, 0.0, 0.0, 0)

    x = _design(ages)
    w = np.ones(n)
    if prior is not None:
        x = np.vstack([x, _design(_PRIOR_AGES)])
        ratios = np.concatenate([ratios, np.asarray(prior.ratio(_PRIOR_AGES))])
        w = np.concatenate([w, np.full(len(_PRIOR_AGES), prior_weight / len(_PRIOR_AGES))])
        lhs = x.T @ (x * w[:, None]) + 1e-8 * np.eye(3)
        rhs = x.T @ (w * ratios)
    else:
        beta0 = np.array([float(np.mean(ratios)), 0.0, 0.0])
        lhs = x.T @ x + ridge_alpha * np.eye(3)
        rhs = x.T @ ratios + ridge_alpha * beta0
    c0, c1, c2 = np.linalg.solve(lhs, rhs)
    return DegradationCurve(float(c0), float(c1), float(max(c2, 0.0)), n)


def fit_prior_curves(
    rows: list[tuple[float, float]], min_laps: int = 30, ridge_alpha: float = 1.0
) -> DegradationCurve | None:
    """Historical prior for one (track, compound): plain ridge quadratic over
    (age, ratio) points. None when there is too little data to be meaningful."""
    if len(rows) < min_laps:
        return None
    ages = np.array([r[0] for r in rows])
    ratios = np.array([r[1] for r in rows])
    return fit_curve(ages, ratios, prior=None, ridge_alpha=ridge_alpha)
