"""Base record type with provenance tracking.

Replaces the R package's virtual ``RLum`` S4 class. Every data object and
analysis result carries a unique ``uid`` and a ``pids`` chain naming the
objects it was derived from — the provenance backbone that in R is
maintained through the ``.uid``/``.pid`` slots.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from typing import Any, Self

__all__ = ["Record"]


def _new_uid() -> str:
    return str(uuid.uuid4())


@dataclass(kw_only=True, eq=False)
class Record:
    """Common metadata carried by every luminescence data object.

    Attributes:
        originator: Name of the function that created the object.
        info: Free-form metadata (instrument settings, comments, ...).
        uid: Unique identifier of this object (never compared or copied).
        pids: uids of the parent objects this one was derived from.
    """

    originator: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    uid: str = field(default_factory=_new_uid)
    pids: tuple[str, ...] = ()

    def with_parent(self, parent: Record) -> Self:
        """Return a copy registering ``parent`` in the provenance chain."""
        return dataclasses.replace(self, uid=_new_uid(), pids=(*self.pids, parent.uid))

    def _meta_repr(self) -> str:
        origin = f" from {self.originator}()" if self.originator else ""
        return f"{type(self).__name__}{origin}"
