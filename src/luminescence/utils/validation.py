"""Small runtime-validation helpers.

Most of the R package's ~15 ``.validate_*`` internals collapse into type
hints; what remains here are the checks that guard against silently wrong
science (ranges, shapes) rather than wrong types.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "as_curve_matrix",
    "require_in_range",
    "require_positive",
    "to_channels",
    "validate_integral",
]


def require_positive(value: float, name: str, *, strict: bool = True) -> float:
    """Return ``value`` if positive, else raise ``ValueError``."""
    if strict and value <= 0:
        raise ValueError(f"'{name}' must be > 0, got {value!r}")
    if not strict and value < 0:
        raise ValueError(f"'{name}' must be >= 0, got {value!r}")
    return value


def require_in_range(value: float, name: str, low: float, high: float) -> float:
    """Return ``value`` if within ``[low, high]``, else raise ``ValueError``."""
    if not low <= value <= high:
        raise ValueError(f"'{name}' must be in [{low}, {high}], got {value!r}")
    return value


def to_channels(
    integral: npt.NDArray[np.float64], x_axis: npt.NDArray[np.float64], name: str
) -> npt.NDArray[np.float64]:
    """Convert measurement values (e.g. seconds) to a contiguous 1-based channel range."""
    import warnings

    from luminescence.utils.exceptions import LuminescenceWarning

    integral = integral[~np.isnan(integral)]
    if integral.size == 0:
        raise ValueError(f"'{name}' contains no usable elements")
    if integral.min() > x_axis.max() or integral.max() < x_axis.min():
        warnings.warn(
            f"Conversion of '{name}' from time to channels failed: expected values in"
            f" [{x_axis.min():g}, {x_axis.max():g}]",
            LuminescenceWarning,
            stacklevel=3,
        )
    ch_min = int(np.argmin(np.abs(integral.min() - x_axis))) + 1
    ch_max = int(np.argmin(np.abs(integral.max() - x_axis))) + 1
    return np.arange(ch_min, ch_max + 1, dtype=np.float64)


def validate_integral(
    integral: npt.ArrayLike | None, name: str, low: int, high: int
) -> npt.NDArray[np.int64] | None:
    """Clamp a 1-based channel integral to [low, high]; None passes through."""
    import warnings

    from luminescence.utils.exceptions import LuminescenceWarning

    if integral is None:
        return None
    arr = np.asarray(integral, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    if low > high:
        raise ValueError(f"'{name}' is expected to be at least {low}, but the maximum is {high}")
    keep = arr[(arr >= low) & (arr <= high)]
    if keep.size == 0:
        raise ValueError(f"'{name}' contains no elements between {low} and {high}")
    if keep.size != arr.size:
        warnings.warn(
            f"'{name}' contains out of bounds elements, reset to be between"
            f" {int(keep.min())} and {int(keep.max())}",
            LuminescenceWarning,
            stacklevel=3,
        )
    if np.any(keep != np.round(keep)):
        raise ValueError(f"'{name}' should be a vector of integers")
    return np.unique(keep).astype(np.int64)


def as_curve_matrix(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce input to the canonical (n, 2) float64 curve matrix.

    Accepts anything array-like with two columns (x, y); a 1-D array of
    counts is promoted with a 1-based channel index as x-axis, matching the
    R constructor's behaviour for bare count vectors.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.column_stack([np.arange(1, arr.size + 1, dtype=np.float64), arr])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"curve data must have shape (n, 2), got {arr.shape}")
    return arr
