"""A single measured luminescence curve (port of ``RLum.Data.Curve``)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from luminescence.core.base import Record
from luminescence.utils.validation import as_curve_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

__all__ = ["Curve"]


def _empty_matrix() -> npt.NDArray[np.float64]:
    return np.empty((0, 2), dtype=np.float64)


@dataclass(kw_only=True, eq=False)
class Curve(Record):
    """One measured curve: an (n, 2) matrix of (x, y) values.

    ``x`` is time or temperature depending on ``record_type`` (e.g. "OSL",
    "TL", "IRSL"); ``y`` is the measured signal (counts).
    """

    record_type: str = ""
    curve_type: str = ""
    data: npt.NDArray[np.float64] = field(default_factory=_empty_matrix)

    def __post_init__(self) -> None:
        self.data = as_curve_matrix(self.data)

    # -- basic protocol --------------------------------------------------

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.data[:, 0]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self.data[:, 1]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, key: slice | int) -> Curve:
        """Row-slice the curve, returning a new :class:`Curve`."""
        if not isinstance(key, slice):
            raise TypeError("Curve supports slice indexing only; use .x/.y/.data for values")
        return dataclasses.replace(self, data=self.data[key].copy())

    def __repr__(self) -> str:
        span = f"[{self.x[0]:g}, {self.x[-1]:g}]" if len(self) else "[]"
        return (
            f"<Curve {self.record_type or '?'}/{self.curve_type or '?'}, "
            f"{len(self)} channels, x {span}>"
        )

    def __eq__(self, other: object) -> bool:
        """Payload equality: type, record/curve type, and data (never uid)."""
        if not isinstance(other, Curve):
            return NotImplemented
        return (
            self.record_type == other.record_type
            and self.curve_type == other.curve_type
            and np.array_equal(self.data, other.data)
        )

    __hash__ = None  # type: ignore[assignment]  # mutable container

    # -- arithmetic (y-values; x-axes must be compatible) -----------------

    def _binary_op(self, other: Curve | float, op: Callable[[Any, Any], Any], symbol: str) -> Curve:
        if isinstance(other, Curve):
            if self.data.shape != other.data.shape or not np.allclose(self.x, other.x):
                raise ValueError(
                    f"cannot compute curve {symbol}: x-axes have different"
                    f" resolution ({self.data.shape} vs {other.data.shape})"
                )
            other_y = other.y
        else:
            other_y = other
        new = dataclasses.replace(self, data=np.column_stack([self.x, op(self.y, other_y)]))
        return new.with_parent(self) if isinstance(other, Curve) else new

    def __add__(self, other: Curve | float) -> Curve:
        return self._binary_op(other, lambda a, b: a + b, "+")

    def __sub__(self, other: Curve | float) -> Curve:
        return self._binary_op(other, lambda a, b: a - b, "-")

    def __mul__(self, other: Curve | float) -> Curve:
        return self._binary_op(other, lambda a, b: a * b, "*")

    def __truediv__(self, other: Curve | float) -> Curve:
        return self._binary_op(other, lambda a, b: a / b, "/")

    # -- conversions ------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame({"x": self.x, "y": self.y})
