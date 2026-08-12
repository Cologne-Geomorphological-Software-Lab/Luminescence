"""In-memory representation of Risø BIN/BINX files (port of ``Risoe.BINfileData``).

The ``metadata`` table mirrors the binary record headers column by column
(80 columns, see ``tools/specs/bin-format.md``); ``data`` holds the raw count
vectors (or ROI definitions for camera records). :meth:`RisoeBINFileData.to_analysis`
bridges into the :class:`~luminescence.core.analysis.Analysis` world.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from luminescence.core.analysis import Analysis
from luminescence.core.curve import Curve
from luminescence.utils.exceptions import LuminescenceWarning

__all__ = ["METADATA_SCHEMA", "RisoeBINFileData", "build_curve_matrix"]

# column name -> (pandas dtype, default). Order matters: it mirrors the R
# METADATA data.frame. Columns whose default is 0 (not NA) are load-bearing:
# RECTYPE stays 0 for V3-V7 files and POSITION/GRAIN 0 marks "not set".
METADATA_SCHEMA: dict[str, tuple[str, Any]] = {
    "ID": ("int64", 0),
    "SEL": ("boolean", True),
    "VERSION": ("Int64", pd.NA),
    "LENGTH": ("int64", 0),
    "PREVIOUS": ("int64", 0),
    "NPOINTS": ("int64", 0),
    "RECTYPE": ("int64", 0),
    "RUN": ("Int64", pd.NA),
    "SET": ("Int64", pd.NA),
    "POSITION": ("int64", 0),
    "GRAIN": ("int64", 0),
    "GRAINNUMBER": ("int64", 0),
    "CURVENO": ("Int64", pd.NA),
    "XCOORD": ("Int64", pd.NA),
    "YCOORD": ("Int64", pd.NA),
    "SAMPLE": ("str", ""),
    "COMMENT": ("str", ""),
    "SYSTEMID": ("Int64", pd.NA),
    "FNAME": ("str", ""),
    "USER": ("str", ""),
    "TIME": ("str", ""),
    "DATE": ("str", ""),
    "DTYPE": ("str", ""),
    "BL_TIME": ("float64", np.nan),
    "BL_UNIT": ("Int64", pd.NA),
    "NORM1": ("float64", np.nan),
    "NORM2": ("float64", np.nan),
    "NORM3": ("float64", np.nan),
    "BG": ("float64", np.nan),
    "SHIFT": ("Int64", pd.NA),
    "TAG": ("Int64", pd.NA),
    "LTYPE": ("str", ""),
    "LIGHTSOURCE": ("str", ""),
    "LPOWER": ("float64", np.nan),
    "LIGHTPOWER": ("float64", np.nan),
    "LOW": ("float64", np.nan),
    "HIGH": ("float64", np.nan),
    "RATE": ("float64", np.nan),
    "TEMPERATURE": ("float64", np.nan),
    "MEASTEMP": ("float64", np.nan),
    "AN_TEMP": ("float64", np.nan),
    "AN_TIME": ("float64", np.nan),
    "TOLDELAY": ("Int64", pd.NA),
    "TOLON": ("Int64", pd.NA),
    "TOLOFF": ("Int64", pd.NA),
    "IRR_TIME": ("float64", np.nan),
    "IRR_TYPE": ("Int64", pd.NA),
    "IRR_UNIT": ("Int64", pd.NA),
    "IRR_DOSERATE": ("float64", np.nan),
    "IRR_DOSERATEERR": ("float64", np.nan),
    "TIMESINCEIRR": ("float64", np.nan),
    "TIMETICK": ("float64", np.nan),
    "ONTIME": ("float64", np.nan),
    "OFFTIME": ("float64", np.nan),
    "STIMPERIOD": ("Int64", pd.NA),
    "GATE_ENABLED": ("float64", np.nan),
    "ENABLE_FLAGS": ("float64", np.nan),
    "GATE_START": ("float64", np.nan),
    "GATE_STOP": ("float64", np.nan),
    "PTENABLED": ("float64", np.nan),
    "DTENABLED": ("float64", np.nan),
    "DEADTIME": ("float64", np.nan),
    "MAXLPOWER": ("float64", np.nan),
    "XRF_ACQTIME": ("float64", np.nan),
    "XRF_HV": ("float64", np.nan),
    "XRF_CURR": ("float64", np.nan),
    "XRF_DEADTIMEF": ("float64", np.nan),
    "DETECTOR_ID": ("Int64", pd.NA),
    "LOWERFILTER_ID": ("Int64", pd.NA),
    "UPPERFILTER_ID": ("Int64", pd.NA),
    "ENOISEFACTOR": ("float64", np.nan),
    "MARKPOS_X1": ("float64", np.nan),
    "MARKPOS_Y1": ("float64", np.nan),
    "MARKPOS_X2": ("float64", np.nan),
    "MARKPOS_Y2": ("float64", np.nan),
    "MARKPOS_X3": ("float64", np.nan),
    "MARKPOS_Y3": ("float64", np.nan),
    "EXTR_START": ("float64", np.nan),
    "EXTR_END": ("float64", np.nan),
    "SEQUENCE": ("str", ""),
}


def _seq_rlum(start: float, stop: float, n: int) -> npt.NDArray[np.float64]:
    """R helper ``seq_RLum``: n values excluding `start`, including `stop`."""
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    by = (stop - start) / n
    return start + by * np.arange(1, n + 1, dtype=np.float64)


def build_curve_matrix(
    data: npt.ArrayLike,
    version: int,
    npoints: int,
    ltype: str,
    low: float,
    high: float,
    an_temp: float,
    toldelay: int,
    tolon: int,
    toloff: int,
) -> npt.NDArray[np.float64]:
    """Build the (x, y) curve matrix from raw BIN record fields.

    Port of ``src_create_RLumDataCurve_matrix.cpp``: TL curves of version >= 4
    get a three-segment temperature axis (ramp, plateau at ``an_temp``, ramp);
    everything else gets a plain sequence from ``low`` to ``high``.
    """
    if npoints <= 0:
        return np.full((1, 2), np.nan)

    y = np.asarray(data, dtype=np.float64)
    if y.size < npoints:
        y = np.concatenate([y, np.full(npoints - y.size, np.nan)])
    y = y[:npoints]

    if ltype == "TL" and version >= 4:
        if toldelay == 0 and tolon == 0 and toloff == 0:
            print("[build_curve_matrix()] BIN/BINX-file non-conform. TL curve may be wrong!")
            toloff = npoints
        x = np.empty(npoints, dtype=np.float64)
        b_start = min(max(toldelay, 0), npoints)
        b_end = min(b_start + max(tolon, 0), npoints)
        x[:b_start] = _seq_rlum(low, an_temp, toldelay)[:b_start]
        x[b_start:b_end] = an_temp
        ramp_end = _seq_rlum(an_temp, high, toloff)
        tail = npoints - b_end
        if ramp_end.size >= tail:
            x[b_end:] = ramp_end[:tail]
        else:  # undersized ramp: pad with the last value (C++ reads OOB here)
            x[b_end : b_end + ramp_end.size] = ramp_end
            fill = ramp_end[-1] if ramp_end.size else np.nan
            x[b_end + ramp_end.size :] = fill
    else:
        x = _seq_rlum(low, high, npoints)

    return np.column_stack([x, y])


@dataclass(eq=False)
class RisoeBINFileData:
    """Parsed contents of a Risø BIN/BINX file.

    ``metadata``, ``data`` and ``reserved`` are parallel: one row / element
    per record kept from the file.
    """

    metadata: pd.DataFrame = field(default_factory=lambda: empty_metadata())
    data: list[Any] = field(default_factory=list)
    reserved: list[Any] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.metadata)

    def __repr__(self) -> str:
        if not len(self):
            return "<RisoeBINFileData: empty>"
        versions = sorted(set(self.metadata["VERSION"].dropna().astype(int).tolist()))
        positions = self.metadata["POSITION"].nunique()
        return (
            f"<RisoeBINFileData: {len(self)} records"
            f" (BIN version {'/'.join(map(str, versions))}, {positions} position(s))>"
        )

    # -- bridge to the RLum world -------------------------------------------

    def record_to_curve(self, row: int) -> Curve:
        """Convert metadata row ``row`` (0-based) + payload into a Curve."""
        meta = self.metadata.iloc[row]
        if int(meta["RECTYPE"]) == 128:
            return Curve(originator="risoe_bin_file_data_to_curve")
        info = {col: _plain(meta[col]) for col in self.metadata.columns}
        matrix = build_curve_matrix(
            self.data[row],
            version=int(meta["VERSION"]),
            npoints=int(meta["NPOINTS"]),
            ltype=str(meta["LTYPE"]),
            low=float(meta["LOW"]),
            high=float(meta["HIGH"]),
            an_temp=float(meta["AN_TEMP"]),
            toldelay=_int_or_zero(meta["TOLDELAY"]),
            tolon=_int_or_zero(meta["TOLON"]),
            toloff=_int_or_zero(meta["TOLOFF"]),
        )
        return Curve(
            record_type=f"{meta['LTYPE']} (PMT)",
            data=matrix,
            info=info,
            originator="risoe_bin_file_data_to_curve",
        )

    def to_analysis(
        self,
        pos: int | list[int] | None = None,
        grain: int | list[int] | None = None,
        run: int | list[int] | None = None,
        set_: int | list[int] | None = None,
        ltype: str | list[str] | None = None,
        dtype: str | list[str] | None = None,
        protocol: str = "unknown",
        keep_empty: bool = True,
    ) -> Analysis | list[Analysis]:
        """Group records into one Analysis per (position, grain) pair.

        ``run``/``set_``/``ltype``/``dtype`` are filters (invalid values are an
        error); invalid ``pos``/``grain`` values are skipped with a warning.
        A single (position, grain) pair returns a bare Analysis, several
        return a flat list ordered position-major.
        """
        meta = self.metadata
        pos_list = _resolve_filter(pos, _col(meta, "POSITION"), "pos", strict=False)
        grain_list = _resolve_filter(grain, _col(meta, "GRAIN"), "grain", strict=False)
        run_list = _resolve_filter(run, _col(meta, "RUN"), "run", strict=True)
        set_list = _resolve_filter(set_, _col(meta, "SET"), "set_", strict=True)
        ltype_list = _resolve_filter(ltype, _col(meta, "LTYPE"), "ltype", strict=True)
        dtype_list = _resolve_filter(dtype, _col(meta, "DTYPE"), "dtype", strict=True)

        base_mask = (
            _col(meta, "RUN").isin(run_list)
            & _col(meta, "LTYPE").isin(ltype_list)
            & _col(meta, "SET").isin(set_list)
            & _col(meta, "DTYPE").isin(dtype_list)
        )

        analyses: list[Analysis] = []
        for p in pos_list:
            for g in grain_list:
                mask = (
                    base_mask
                    & (_col(meta, "POSITION") == p)
                    & (_col(meta, "GRAIN").isna() | (_col(meta, "GRAIN") == g))
                )
                rows = np.flatnonzero(mask.to_numpy())
                if rows.size == 0 and not keep_empty:
                    continue
                analysis = Analysis(
                    protocol=protocol,
                    originator="risoe_bin_file_data_to_analysis",
                    records=[],
                )
                records = [
                    dataclasses.replace(self.record_to_curve(int(r)), pids=(analysis.uid,))
                    for r in rows
                ]
                analysis.records = list(records)
                analyses.append(analysis)

        if len(analyses) == 1:
            return analyses[0]
        return analyses


def empty_metadata() -> pd.DataFrame:
    """An empty METADATA table with the full 80-column schema."""
    return pd.DataFrame({name: pd.Series(dtype=dt) for name, (dt, _) in METADATA_SCHEMA.items()})


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    """Typed single-column access (pandas' __getitem__ union confuses pyright)."""
    return cast("pd.Series", frame[name])


def _plain(value: Any) -> Any:
    """Convert pandas scalars to plain Python types for the info dict."""
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _int_or_zero(value: Any) -> int:
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return 0
    return int(value)


def _resolve_filter(wanted: Any, column: pd.Series, name: str, *, strict: bool) -> list[Any]:
    valid = column.dropna().unique().tolist()
    if wanted is None:
        return valid
    wanted_list = [wanted] if not isinstance(wanted, list | tuple) else list(wanted)
    invalid = [v for v in wanted_list if v not in valid]
    if invalid:
        if strict:
            raise ValueError(f"'{name}' contains invalid values, valid values are: {valid}")
        warnings.warn(
            f"Invalid {name} number skipped: {invalid}",
            LuminescenceWarning,
            stacklevel=3,
        )
    return [v for v in wanted_list if v in valid]
