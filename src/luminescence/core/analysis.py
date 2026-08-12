"""Container for a measurement sequence (port of ``RLum.Analysis``)."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

from luminescence.core.base import Record
from luminescence.core.curve import Curve

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Analysis"]


@dataclass(kw_only=True, eq=False)
class Analysis(Record):
    """An ordered set of measurement records, e.g. one aliquot's SAR sequence.

    ``records`` holds :class:`~luminescence.core.curve.Curve`,
    :class:`~luminescence.core.spectrum.Spectrum` or
    :class:`~luminescence.core.image.ImageData` objects in measurement order.
    """

    protocol: str = ""
    records: list[Record] = field(default_factory=list)

    # -- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __contains__(self, item: object) -> bool:
        return any(record == item for record in self.records)

    @overload
    def __getitem__(self, key: int) -> Record: ...
    @overload
    def __getitem__(self, key: slice | Sequence[int]) -> Analysis: ...

    def __getitem__(self, key: int | slice | Sequence[int]) -> Record | Analysis:
        """``int`` returns the record; slice or index list a sub-Analysis."""
        if isinstance(key, int):
            return self.records[key]
        selected = self.records[key] if isinstance(key, slice) else [self.records[i] for i in key]
        return dataclasses.replace(self, records=list(selected)).with_parent(self)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for record in self.records:
            label = getattr(record, "record_type", "") or type(record).__name__
            counts[label] = counts.get(label, 0) + 1
        summary = ", ".join(f"{n}x {label}" for label, n in counts.items()) or "empty"
        protocol = f" protocol={self.protocol!r}" if self.protocol else ""
        return f"<Analysis{protocol}, {len(self)} records: {summary}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Analysis):
            return NotImplemented
        return self.protocol == other.protocol and self.records == other.records

    __hash__ = None  # type: ignore[assignment]

    # -- record access (port of get_RLum) -----------------------------------

    @property
    def names(self) -> list[str]:
        """The record types, in measurement order."""
        return [getattr(record, "record_type", "") for record in self.records]

    def get_records(
        self,
        record_type: str | Iterable[str] | None = None,
        *,
        curve_type: str | None = None,
        regex: bool = False,
    ) -> list[Record]:
        """Select records by type, preserving measurement order.

        Args:
            record_type: One or more record types to keep (e.g. ``"OSL"``).
                With ``regex=True`` each entry is a regular expression matched
                against the record type (R's ``get_RLum`` grepl behaviour).
            curve_type: Keep only records with this curve type.
            regex: Interpret ``record_type`` entries as regular expressions.
        """
        wanted: list[str] | None = None
        if record_type is not None:
            wanted = [record_type] if isinstance(record_type, str) else list(record_type)

        def keep(record: Record) -> bool:
            rtype = getattr(record, "record_type", "")
            if wanted is not None:
                if regex:
                    if not any(re.search(pattern, rtype) for pattern in wanted):
                        return False
                elif rtype not in wanted:
                    return False
            return curve_type is None or getattr(record, "curve_type", "") == curve_type

        return [record for record in self.records if keep(record)]

    def subset(
        self,
        record_type: str | Iterable[str] | None = None,
        *,
        curve_type: str | None = None,
        regex: bool = False,
    ) -> Analysis:
        """Like :meth:`get_records` but returns a new :class:`Analysis`."""
        selected = self.get_records(record_type, curve_type=curve_type, regex=regex)
        return dataclasses.replace(self, records=list(selected)).with_parent(self)

    # -- overview (port of structure_RLum) ----------------------------------

    def describe(self) -> pd.DataFrame:
        """One row per record: index, class, types, length, x-range."""
        from collections.abc import Sized

        import numpy as np
        import pandas as pd

        rows = []
        for i, record in enumerate(self.records):
            x_min = x_max = np.nan
            if isinstance(record, Curve) and len(record):
                x_min, x_max = float(record.x[0]), float(record.x[-1])
            rows.append(
                {
                    "id": i,
                    "class": type(record).__name__,
                    "record_type": getattr(record, "record_type", ""),
                    "curve_type": getattr(record, "curve_type", ""),
                    "length": len(record) if isinstance(record, Sized) else np.nan,
                    "x_min": x_min,
                    "x_max": x_max,
                }
            )
        return pd.DataFrame(rows)
