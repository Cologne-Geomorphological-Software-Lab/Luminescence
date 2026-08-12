"""Numerical helpers shared across subsystems (port of internals_RLum.R)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["weighted_median"]


def weighted_median(values: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    """Weighted median, replicating the R package's ``.weighted.median``.

    Sort by value, accumulate normalised weights; the median is the value
    where the cumulative weight passes 0.5, averaging the two neighbouring
    values when it hits the boundary exactly.
    """
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(x)
    x, w = x[order], w[order]
    p = np.cumsum(w) / np.sum(w)
    n = int(np.count_nonzero(p < 0.5))
    if p[n] > 0.5:
        return float(x[n])
    return float((x[n] + x[n + 1]) / 2)
