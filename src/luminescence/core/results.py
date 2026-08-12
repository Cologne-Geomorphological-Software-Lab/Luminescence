"""Universal analysis-result container (port of ``RLum.Results``)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from luminescence.core.base import Record

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Results"]


@dataclass(kw_only=True, eq=False)
class Results(Record, Mapping[str, Any]):
    """The return type of every ``analyse_*``/``calc_*``/``fit_*`` function.

    Behaves as a read-only mapping over its ``data`` dict; by convention the
    first entry is the primary result table.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        origin = f" from {self.originator}()" if self.originator else ""
        return f"<Results{origin}: {list(self.data)}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Results):
            return NotImplemented
        return self.originator == other.originator and list(self.data) == list(other.data)

    __hash__ = None  # type: ignore[assignment]

    def to_dataframe(self, key: str | None = None) -> pd.DataFrame:
        """Return entry ``key`` (default: the first entry) as a DataFrame."""
        import pandas as pd

        if not self.data:
            raise ValueError("Results object is empty")
        if key is None:
            key = next(iter(self.data))
        return pd.DataFrame(self.data[key])
