"""Exception and warning types for the luminescence package.

Replaces the R package's ``.throw_error``/``.throw_warning`` condition
framework: Python exceptions already carry the originating function in the
traceback, so no function-name stack is needed.
"""

from __future__ import annotations

__all__ = [
    "DataFormatError",
    "FitError",
    "LuminescenceError",
    "LuminescenceWarning",
]


class LuminescenceError(Exception):
    """Base class for all errors raised by the luminescence package."""


class DataFormatError(LuminescenceError):
    """A file or byte stream does not conform to the expected instrument format."""


class FitError(LuminescenceError):
    """A curve fit failed or produced an unusable result."""


class LuminescenceWarning(UserWarning):
    """Base class for all warnings emitted by the luminescence package."""
