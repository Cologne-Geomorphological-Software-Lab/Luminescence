"""Measured image data, e.g. EMCCD frames (port of ``RLum.Data.Image``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from luminescence.core.base import Record

__all__ = ["ImageData"]


def _empty_3d() -> npt.NDArray[np.float64]:
    return np.empty((0, 0, 0), dtype=np.float64)


@dataclass(kw_only=True, eq=False)
class ImageData(Record):
    """An image stack with shape (rows, columns, frames)."""

    record_type: str = ""
    curve_type: str = ""
    data: npt.NDArray[np.float64] = field(default_factory=_empty_3d)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"image data must be 2-D or 3-D, got shape {arr.shape}")
        self.data = arr

    @property
    def n_frames(self) -> int:
        return self.data.shape[2]

    def __len__(self) -> int:
        return self.n_frames

    def __repr__(self) -> str:
        rows, cols, frames = self.data.shape
        return f"<ImageData {self.record_type or '?'}, {rows}x{cols} px, {frames} frame(s)>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageData):
            return NotImplemented
        return self.record_type == other.record_type and np.array_equal(self.data, other.data)

    __hash__ = None  # type: ignore[assignment]
