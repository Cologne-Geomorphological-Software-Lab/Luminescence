"""A measured luminescence spectrum (port of ``RLum.Data.Spectrum``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from luminescence.core.base import Record

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Spectrum"]


def _empty_2d() -> npt.NDArray[np.float64]:
    return np.empty((0, 0), dtype=np.float64)


def _empty_1d() -> npt.NDArray[np.float64]:
    return np.empty(0, dtype=np.float64)


@dataclass(kw_only=True, eq=False)
class Spectrum(Record):
    """A (wavelength x time/frame) signal matrix.

    In R the axis values live in the matrix dimnames; here they are explicit
    vectors. ``data[i, j]`` is the signal at ``wavelengths[i]``, ``times[j]``.
    """

    record_type: str = ""
    curve_type: str = ""
    data: npt.NDArray[np.float64] = field(default_factory=_empty_2d)
    wavelengths: npt.NDArray[np.float64] = field(default_factory=_empty_1d)
    times: npt.NDArray[np.float64] = field(default_factory=_empty_1d)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"spectrum data must be 2-D, got shape {self.data.shape}")
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        if self.wavelengths.size == 0:
            self.wavelengths = np.arange(1, self.data.shape[0] + 1, dtype=np.float64)
        if self.times.size == 0:
            self.times = np.arange(1, self.data.shape[1] + 1, dtype=np.float64)
        if self.wavelengths.size != self.data.shape[0] or self.times.size != self.data.shape[1]:
            raise ValueError(
                f"axis lengths ({self.wavelengths.size}, {self.times.size}) do not match"
                f" data shape {self.data.shape}"
            )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return (
            f"<Spectrum {self.record_type or '?'}, {self.data.shape[0]} wavelengths x "
            f"{self.data.shape[1]} frames>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Spectrum):
            return NotImplemented
        return (
            self.record_type == other.record_type
            and np.array_equal(self.data, other.data)
            and np.array_equal(self.wavelengths, other.wavelengths)
            and np.array_equal(self.times, other.times)
        )

    __hash__ = None  # type: ignore[assignment]

    def to_dataframe(self) -> pd.DataFrame:
        """Long format: one row per (wavelength, time, signal) triple."""
        import pandas as pd

        wl, t = np.meshgrid(self.wavelengths, self.times, indexing="ij")
        return pd.DataFrame(
            {"wavelength": wl.ravel(), "time": t.ravel(), "signal": self.data.ravel()}
        )
