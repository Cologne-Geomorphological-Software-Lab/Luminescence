"""Descriptive statistics for equivalent-dose distributions.

Port of ``calc_Statistics`` (R package v0.1.8). Reproduces the R behaviour
verbatim, including two documented quirks:

- a single zero error replaces the *entire* error column with 1e-9;
- the "weighted" skewness/kurtosis are the unweighted values (only the MCM
  variant computes its own).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from luminescence.utils.exceptions import LuminescenceWarning
from luminescence.utils.numerics import weighted_median

__all__ = ["calc_statistics"]

_ZERO_ERROR_SUBSTITUTE = 1e-9


def calc_statistics(
    data: npt.ArrayLike,
    weight_calc: Literal["inverse_var", "inverse_std"] = "inverse_var",
    *,
    digits: int | None = None,
    n_mcm: int | None = None,
    na_rm: bool = True,
    rng: np.random.Generator | int | None = None,
) -> dict[str, dict[str, float]]:
    """Weighted, unweighted, and Monte-Carlo statistics of a De distribution.

    Args:
        data: Two columns: values (De) and their errors. A single column is
            accepted (errors treated as missing).
        weight_calc: ``"inverse_var"`` (1/e^2, default) or ``"inverse_std"``.
        digits: Round all results to this many decimal places.
        n_mcm: Number of Monte-Carlo resamples; ``None`` disables MC (the
            "MCM" entry then reflects the raw values).
        na_rm: Drop rows containing NaN before calculation.
        rng: Seed or generator for the MC resampling.

    Returns:
        ``{"weighted": {...}, "unweighted": {...}, "MCM": {...}}``, each with
        keys n, mean, median, sd.abs, sd.rel, se.abs, se.rel, skewness,
        kurtosis (matching the R result lists).
    """
    arr = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if arr.shape[0] == 1 and arr.shape[1] > 2:
        arr = arr.T
    if arr.shape[1] == 1:
        arr = np.column_stack([arr[:, 0], np.full(arr.shape[0], np.nan)])
    values, errors = arr[:, 0].copy(), arr[:, 1].copy()

    if na_rm:
        keep = ~np.isnan(values)
        values, errors = values[keep], errors[keep]
    if values.size == 0:
        raise ValueError("'data' contains only NA values")

    errors[np.isnan(errors)] = 0.0
    if np.any(errors == 0):
        if np.sum(errors) == 0:
            warnings.warn(
                "All errors are NA or zero, automatically set to 1e-09",
                LuminescenceWarning,
                stacklevel=2,
            )
        errors = np.full_like(errors, _ZERO_ERROR_SUBSTITUTE)

    if weight_calc == "inverse_std":
        w_raw = 1.0 / errors
    elif weight_calc == "inverse_var":
        w_raw = 1.0 / errors**2
    else:
        raise ValueError(f"unknown weight_calc: {weight_calc!r}")
    w = w_raw / np.sum(w_raw)

    n = values.size
    if n_mcm is None:
        mcm = values.reshape(n, 1)
    else:
        generator = np.random.default_rng(rng)
        mcm = generator.normal(
            loc=values[:, np.newaxis], scale=errors[:, np.newaxis], size=(n, n_mcm)
        )

    u_mean = float(np.mean(values))
    u_sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    u_skew = float(np.sum(((values - u_mean) / u_sd) ** 3) / n)
    u_kurt = float(np.sum(((values - u_mean) / u_sd) ** 4) / n)

    w_mean = float(np.sum(w * values) / np.sum(w))
    w_sd = float(np.sqrt(np.sum(w * (values - w_mean) ** 2) / (((n - 1) * np.sum(w)) / n)))

    m_flat = mcm.ravel()
    m_mean = float(np.mean(m_flat))
    m_sd = float(np.std(m_flat, ddof=1)) if m_flat.size > 1 else float("nan")
    m_n = m_flat.size
    m_skew = float(np.sum(((m_flat - m_mean) / m_sd) ** 3) / m_n)
    m_kurt = float(np.sum(((m_flat - m_mean) / m_sd) ** 4) / m_n)

    def summarise(
        mean: float, median: float, sd: float, skewness: float, kurtosis: float
    ) -> dict[str, float]:
        se = sd / np.sqrt(n)
        entry: dict[str, Any] = {
            "n": n,
            "mean": mean,
            "median": median,
            "sd.abs": sd,
            "sd.rel": sd / mean * 100,
            "se.abs": se,
            "se.rel": se / mean * 100,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }
        if digits is not None:
            entry = {k: (v if k == "n" else round(float(v), digits)) for k, v in entry.items()}
        return entry

    return {
        # quirk: weighted skew/kurt are the unweighted values (R L176)
        "weighted": summarise(w_mean, weighted_median(values, w), w_sd, u_skew, u_kurt),
        "unweighted": summarise(u_mean, float(np.median(values)), u_sd, u_skew, u_kurt),
        "MCM": summarise(m_mean, float(np.median(m_flat)), m_sd, m_skew, m_kurt),
    }
