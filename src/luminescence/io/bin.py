"""Reader for Risø BIN/BINX files (port of ``read_BIN2R``, R package v0.19).

Byte layout per ``tools/specs/bin-format.md`` (versions 3-8, little-endian,
Pascal-style strings). The reader degrades gracefully on truncated payloads,
over-long string length bytes, and unknown lookup codes, exactly like the R
implementation — the edge-case corpus in ``tests/testthat/_data/bin-tests/``
depends on it.
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from luminescence.core.analysis import Analysis
from luminescence.core.risoe import METADATA_SCHEMA, RisoeBINFileData, empty_metadata
from luminescence.utils.exceptions import DataFormatError, LuminescenceWarning

__all__ = ["read_bin"]

_SUPPORTED_VERSIONS = {3, 4, 5, 6, 7, 8}

LTYPE_LOOKUP = {
    0: "TL", 1: "OSL", 2: "IRSL", 3: "M-IR", 4: "M-VIS", 5: "TOL", 6: "TRPOSL",
    7: "RIR", 8: "RBR", 9: "USER", 10: "POSL", 11: "SGOSL", 12: "RL", 13: "XRF",
}  # fmt: skip

DTYPE_LOOKUP = {
    0: "Natural", 1: "N+dose", 2: "Bleach", 3: "Bleach+dose",
    4: "Natural (Bleach)", 5: "N+dose (Bleach)", 6: "Dose", 7: "Background",
}  # fmt: skip

LIGHTSOURCE_LOOKUP = {
    0: "None", 1: "Lamp", 2: "IR diodes/IR Laser", 3: "Calibration LED",
    4: "Blue Diodes", 5: "White light", 6: "Green laser (single grain)",
    7: "IR laser (single grain)",
}  # fmt: skip


def _warn(message: str) -> None:
    warnings.warn(message, LuminescenceWarning, stacklevel=4)


class _Cursor:
    """Sequential little-endian reader over the file buffer."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def _unpack(self, fmt: str, size: int) -> int | float:
        value = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += size
        return value

    def i8(self) -> int:
        return int(self._unpack("<b", 1))

    def u8(self) -> int:
        return int(self._unpack("<B", 1))

    def i16(self) -> int:
        return int(self._unpack("<h", 2))

    def i32(self) -> int:
        return int(self._unpack("<i", 4))

    def f32(self) -> float:
        return float(self._unpack("<f", 4))

    def raw(self, n: int) -> bytes:
        chunk = self.buf[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def skip(self, n: int) -> None:
        self.pos += n

    def string(self, field_length: int, force_size: int | None = None) -> str:
        """Pascal-style length-prefixed string inside a fixed-width field."""
        raw = self.raw(field_length)
        if not raw:
            return ""
        strlen = raw[0] if force_size is None else force_size
        if strlen == 0:
            return ""
        strlen = min(strlen, field_length - 1)
        payload = raw[1 : 1 + strlen].split(b"\x00")[0]
        return payload.decode("latin-1")


def _parse_v34(cur: _Cursor, version: int, rec: dict[str, Any]) -> None:
    rec["LTYPE"] = cur.i8()
    rec["LOW"] = cur.f32()
    rec["HIGH"] = cur.f32()
    rec["RATE"] = cur.f32()
    rec["TEMPERATURE"] = float(cur.i16())
    rec["XCOORD"] = cur.i16()
    rec["YCOORD"] = cur.i16()
    rec["TOLDELAY"] = cur.i16()
    rec["TOLON"] = cur.i16()
    rec["TOLOFF"] = cur.i16()
    rec["POSITION"] = cur.u8()
    rec["RUN"] = cur.u8()
    rec["TIME"] = cur.string(7, force_size=6)
    rec["DATE"] = cur.string(7, force_size=6)
    rec["SEQUENCE"] = cur.string(9)
    rec["USER"] = cur.string(9)
    rec["DTYPE"] = cur.i8()
    rec["IRR_TIME"] = cur.f32()
    rec["IRR_TYPE"] = cur.i8()
    rec["IRR_UNIT"] = cur.i8()
    rec["BL_TIME"] = cur.f32()
    rec["BL_UNIT"] = cur.i8()
    rec["AN_TEMP"] = cur.f32()
    rec["AN_TIME"] = cur.f32()
    rec["NORM1"] = cur.f32()
    rec["NORM2"] = cur.f32()
    rec["NORM3"] = cur.f32()
    rec["BG"] = cur.f32()
    rec["SHIFT"] = cur.i16()
    rec["SAMPLE"] = cur.string(21)
    rec["COMMENT"] = cur.string(81)
    rec["LIGHTSOURCE"] = cur.i8()
    rec["SET"] = cur.i8()
    rec["TAG"] = cur.i8()
    rec["GRAINNUMBER"] = cur.i16()
    rec["LIGHTPOWER"] = cur.f32()
    rec["SYSTEMID"] = cur.i16()
    if version == 3:
        res1 = cur.raw(36)
        rec["ONTIME"] = cur.f32()
        rec["OFFTIME"] = cur.f32()
        rec["GATE_ENABLED"] = float(cur.u8())
        rec["GATE_START"] = cur.f32()
        rec["GATE_STOP"] = cur.f32()
        res2 = cur.raw(1)
    else:
        res1 = cur.raw(20)
        rec["CURVENO"] = cur.i16()
        rec["TIMETICK"] = cur.f32()
        rec["ONTIME"] = float(cur.i32())
        rec["STIMPERIOD"] = cur.i32()
        rec["GATE_ENABLED"] = float(cur.u8())
        rec["GATE_START"] = cur.f32()
        rec["GATE_STOP"] = cur.f32()
        rec["PTENABLED"] = float(cur.u8())
        res2 = cur.raw(10)
    rec["_reserved"] = (res1, res2)


def _parse_v58(cur: _Cursor, version: int, rec: dict[str, Any]) -> None:
    rec["RUN"] = cur.i16()
    rec["SET"] = cur.i16()
    rec["POSITION"] = cur.i16()
    rec["GRAINNUMBER"] = cur.i16()
    rec["CURVENO"] = cur.i16()
    rec["XCOORD"] = cur.i16()
    rec["YCOORD"] = cur.i16()
    rec["SAMPLE"] = cur.string(21)
    rec["COMMENT"] = cur.string(81)
    rec["SYSTEMID"] = cur.i16()
    rec["FNAME"] = cur.string(101)
    rec["USER"] = cur.string(31)
    rec["TIME"] = cur.string(7)
    rec["DATE"] = cur.string(7, force_size=6)
    rec["DTYPE"] = cur.i8()
    rec["BL_TIME"] = cur.f32()
    rec["BL_UNIT"] = cur.i8()
    rec["NORM1"] = cur.f32()
    rec["NORM2"] = cur.f32()
    rec["NORM3"] = cur.f32()
    rec["BG"] = cur.f32()
    rec["SHIFT"] = cur.i16()
    rec["TAG"] = cur.i8()
    res1 = cur.raw(20)
    rec["LTYPE"] = cur.i8()
    rec["LIGHTSOURCE"] = cur.i8()
    rec["LIGHTPOWER"] = cur.f32()
    rec["LOW"] = cur.f32()
    rec["HIGH"] = cur.f32()
    rec["RATE"] = cur.f32()
    rec["TEMPERATURE"] = float(cur.i16())
    rec["MEASTEMP"] = float(cur.i16())
    rec["AN_TEMP"] = cur.f32()
    rec["AN_TIME"] = cur.f32()
    rec["TOLDELAY"] = cur.i16()
    rec["TOLON"] = cur.i16()
    rec["TOLOFF"] = cur.i16()
    rec["IRR_TIME"] = cur.f32()
    rec["IRR_TYPE"] = cur.i8()
    rec["IRR_DOSERATE"] = cur.f32()
    if version >= 6:
        rec["IRR_DOSERATEERR"] = cur.f32()
    rec["TIMESINCEIRR"] = float(cur.i32())
    rec["TIMETICK"] = cur.f32()
    rec["ONTIME"] = float(cur.i32())
    rec["STIMPERIOD"] = cur.i32()
    rec["GATE_ENABLED"] = float(cur.u8())
    rec["GATE_START"] = float(cur.i32())
    rec["GATE_STOP"] = float(cur.i32())
    rec["PTENABLED"] = float(cur.u8())
    rec["DTENABLED"] = float(cur.u8())
    rec["DEADTIME"] = cur.f32()
    rec["MAXLPOWER"] = cur.f32()
    rec["XRF_ACQTIME"] = cur.f32()
    rec["XRF_HV"] = cur.f32()
    rec["XRF_CURR"] = float(cur.i32())
    rec["XRF_DEADTIMEF"] = cur.f32()
    if version == 5:
        res2 = cur.raw(4)
    elif version == 6:
        res2 = cur.raw(24)
    else:
        rec["DETECTOR_ID"] = cur.i8()
        rec["LOWERFILTER_ID"] = cur.i16()
        rec["UPPERFILTER_ID"] = cur.i16()
        rec["ENOISEFACTOR"] = cur.f32()
        if version == 7:
            res2 = cur.raw(15)
        else:
            rec["MARKPOS_X1"] = cur.f32()
            rec["MARKPOS_Y1"] = cur.f32()
            rec["MARKPOS_X2"] = cur.f32()
            rec["MARKPOS_Y2"] = cur.f32()
            rec["MARKPOS_X3"] = cur.f32()
            rec["MARKPOS_Y3"] = cur.f32()
            rec["EXTR_START"] = cur.f32()
            rec["EXTR_END"] = cur.f32()
            res2 = cur.raw(42)
    rec["_reserved"] = (res1, res2)


def _parse_roi_payload(buf: bytes, offset: int, n_rois: int) -> list[dict[str, Any]]:
    rois: list[dict[str, Any]] = []
    for i in range(n_rois):
        base = offset + i * 504
        if base + 504 > len(buf):
            break
        nof_points = struct.unpack_from("<i", buf, base)[0]
        used_for = [b != 0 for b in buf[base + 4 : base + 52]]
        show_for = [b != 0 for b in buf[base + 52 : base + 100]]
        roi_color = struct.unpack_from("<i", buf, base + 100)[0]
        x = np.frombuffer(buf, dtype="<f4", count=50, offset=base + 104).astype(np.float64)
        y = np.frombuffer(buf, dtype="<f4", count=50, offset=base + 304).astype(np.float64)
        rois.append(
            {
                "NOFPOINTS": int(nof_points),
                "USEDFOR": used_for,
                "SHOWFOR": show_for,
                "ROICOLOR": int(roi_color),
                "X": x,
                "Y": y,
            }
        )
    return rois


def _parse_file(
    buf: bytes,
    *,
    n_records: set[int] | None,
    force_version: int | None,
    ignore_rectype: bool | int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Parse all records; returns (records, count of records seen in the file)."""
    records: list[dict[str, Any]] = []
    pos = 0
    file_index = 0  # 1-based file record counter (R's temp.ID)

    while pos < len(buf):
        version_byte = buf[pos]
        version = force_version if force_version is not None else version_byte
        if version not in _SUPPORTED_VERSIONS:
            if records:
                if n_records is not None:
                    _warn(f"BIN-file appears to be corrupt, 'n_records' reset to {len(records)}")
                else:
                    _warn(
                        "BIN-file appears to be corrupt, import limited to the first"
                        f" {len(records)} records"
                    )
                break
            raise DataFormatError(
                f"BIN/BINX format version ({version:02d}) is not supported or file is"
                " broken. Supported version numbers are: 03, 04, 05, 06, 07, 08"
            )

        int_size = 4 if version >= 5 else 2
        header_ints_end = pos + 2 + 3 * int_size
        if pos + 2 + int_size > len(buf):
            if verbose:
                print(f"Record #{file_index + 1} skipped due to wrong record length")
            break
        cur = _Cursor(buf, pos + 2)
        length = cur.i32() if int_size == 4 else cur.i16()
        num_toread = length - int_size - 2
        if num_toread <= 0:
            if verbose:
                print(f"Record #{file_index + 1} skipped due to wrong record length")
            pos = pos + 2 + int_size  # R does not rewind after the failed read
            continue
        if header_ints_end > len(buf):
            break
        previous = cur.i32() if int_size == 4 else cur.i16()
        npoints = cur.i32() if int_size == 4 else cur.i16()
        record_start = pos
        record_end = pos + length

        file_index += 1
        if n_records is not None and file_index not in n_records:
            pos = record_end
            continue

        rec: dict[str, Any] = {
            "VERSION": version,
            "LENGTH": length,
            "PREVIOUS": previous,
            "NPOINTS": npoints,
        }

        # RECTYPE (version 8 only; reset per record for others — fixes an R bug)
        rectype = 0
        if version == 8:
            rectype = cur.u8()
            rec["RECTYPE"] = rectype
            if not isinstance(ignore_rectype, bool) and rectype == int(ignore_rectype):
                if verbose:
                    print(
                        f"Record #{file_index} skipped due to 'ignore_rectype = {ignore_rectype}'"
                    )
                pos = record_end
                continue
            if rectype not in (0, 1, 128):
                if not ignore_rectype:
                    raise DataFormatError(
                        f"Byte RECTYPE = {rectype} is not supported in record"
                        f" #{file_index}, set 'ignore_rectype = True' to skip this record"
                    )
                if verbose:
                    print(
                        f"Byte RECTYPE = {rectype} is not supported in record"
                        f" #{file_index}, record skipped"
                    )
                pos = record_end
                continue

        if rectype == 128:
            # camera/ROI record: only the first 15 header bytes are meaningful
            rec["_data"] = _parse_roi_payload(buf, record_start + 507, npoints)
            rec["_reserved"] = None
        else:
            if version >= 5:
                _parse_v58(cur, version, rec)
            else:
                _parse_v34(cur, version, rec)
            data_offset = cur.pos
            available = max(0, (len(buf) - data_offset) // 4)
            count = min(npoints, available)
            rec["_data"] = np.frombuffer(buf, dtype="<i4", count=count, offset=data_offset).astype(
                np.float64
            )

        records.append(rec)
        pos = record_end

    return records, file_index


def _assemble_metadata(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(records, start=1):
        row = {name: rec.get(name, default) for name, (_, default) in METADATA_SCHEMA.items()}
        row["ID"] = i
        tag = rec.get("TAG")
        row["SEL"] = bool(tag) if tag is not None else pd.NA
        row["GRAIN"] = rec.get("GRAINNUMBER", 0) or 0
        row["GRAINNUMBER"] = rec.get("GRAINNUMBER", 0) or 0
        lightpower = rec.get("LIGHTPOWER", np.nan)
        row["LPOWER"] = lightpower
        row["ENABLE_FLAGS"] = rec.get("GATE_ENABLED", np.nan)
        # LTYPE/DTYPE/LIGHTSOURCE are stringified raw codes before translation
        for col in ("LTYPE", "DTYPE", "LIGHTSOURCE"):
            if col in rec:
                row[col] = str(rec[col])
        rows.append(row)
    if not rows:
        return empty_metadata()
    frame = pd.DataFrame(rows)
    for name, (dtype, _) in METADATA_SCHEMA.items():
        frame[name] = frame[name].astype(dtype)
    return cast("pd.DataFrame", frame.loc[:, list(METADATA_SCHEMA)])


def _translate(metadata: pd.DataFrame, path: Path) -> pd.DataFrame:
    def lookup(column: pd.Series, table: dict[int, str]) -> pd.Series:
        def translate_one(value: str) -> Any:
            try:
                return table.get(int(value), pd.NA)
            except (ValueError, TypeError):
                return pd.NA

        return column.map(translate_one).astype("str").replace("<NA>", "")

    metadata = metadata.copy()
    metadata["LTYPE"] = lookup(cast("pd.Series", metadata["LTYPE"]), LTYPE_LOOKUP)
    metadata["DTYPE"] = lookup(cast("pd.Series", metadata["DTYPE"]), DTYPE_LOOKUP)
    metadata["LIGHTSOURCE"] = lookup(cast("pd.Series", metadata["LIGHTSOURCE"]), LIGHTSOURCE_LOOKUP)

    # V3 quirk: IR-stimulated OSL becomes IRSL when the FIRST record is V3
    if len(metadata) and metadata["VERSION"].iloc[0] == 3:
        mask = (metadata["LTYPE"] == "OSL") & (metadata["LIGHTSOURCE"] == "IR diodes/IR Laser")
        metadata.loc[mask, "LTYPE"] = "IRSL"

    def normalise_time(value: str) -> str:
        if len(value) == 5:
            value = "0" + value
        if len(value) == 6 and value.isdigit():
            return f"{value[0:2]}:{value[2:4]}:{value[4:6]}"
        return value

    metadata["TIME"] = metadata["TIME"].map(normalise_time)

    if not (metadata["FNAME"] != "").any():
        metadata["FNAME"] = path.stem
    return metadata


def read_bin(
    file: str | Path | list[str | Path],
    *,
    show_raw_values: bool = False,
    position: int | list[int] | None = None,
    n_records: int | list[int] | None = None,
    zero_data_rm: bool = True,
    duplicated_rm: bool = False,
    fast_forward: bool = False,
    force_version: int | None = None,
    ignore_rectype: bool | int = False,
    verbose: bool = False,
) -> RisoeBINFileData | Analysis | list[Any] | None:
    """Read one or more Risø BIN/BINX files.

    Args:
        file: Path to a ``.bin``/``.binx`` file, or a list of paths (returns a
            list of per-file results).
        show_raw_values: Keep raw integer codes for LTYPE/DTYPE/LIGHTSOURCE.
        position: Keep only records with these POSITION values (all-or-nothing:
            any invalid position leaves the object unfiltered, with a warning).
        n_records: 1-based file record indices to import.
        zero_data_rm: Drop records with an empty data payload.
        duplicated_rm: Drop records whose payload equals the previous record's.
        fast_forward: Return ``list[Analysis]`` via the Risoe bridge instead.
        force_version: Override the per-record version byte.
        ignore_rectype: ``True`` skips unsupported RECTYPEs; a number
            additionally skips records with exactly that RECTYPE.
        verbose: Print progress/skip messages.

    Returns:
        A :class:`RisoeBINFileData` (or ``list[Analysis]`` with
        ``fast_forward=True``); ``None`` when the file contains no records.
    """
    if isinstance(file, list):
        return [
            read_bin(
                single,
                show_raw_values=show_raw_values,
                position=position,
                n_records=n_records,
                zero_data_rm=zero_data_rm,
                duplicated_rm=duplicated_rm,
                fast_forward=fast_forward,
                force_version=force_version,
                ignore_rectype=ignore_rectype,
                verbose=verbose,
            )
            for single in file
        ]

    path = Path(file)
    if not path.is_file():
        raise FileNotFoundError(f"File '{path}' does not exist")
    if path.suffix.lower() not in (".bin", ".binx"):
        raise ValueError(
            f"File extension '{path.suffix.lstrip('.')}' is not supported, only"
            " 'bin' and 'binx' are valid"
        )
    buf = path.read_bytes()
    if not buf:
        raise DataFormatError(f"File '{path}' is a zero-byte file")

    if force_version is not None and verbose:
        print(
            f"'force_version' set to {force_version:02d}, but this version may not"
            " match your input file"
        )

    n_records_set: set[int] | None = None
    if n_records is not None:
        n_records_set = (
            {int(v) for v in n_records}
            if isinstance(n_records, list | tuple | range)
            else {int(n_records)}
        )

    records, seen = _parse_file(
        buf,
        n_records=n_records_set,
        force_version=force_version,
        ignore_rectype=ignore_rectype,
        verbose=verbose,
    )
    if not records and seen == 0 and n_records_set is None:
        # nothing resembling a record in the file at all
        _warn("0 records read, None returned")
        return None

    metadata = _assemble_metadata(records)
    data: list[Any] = [rec["_data"] for rec in records]
    reserved: list[Any] = [rec.get("_reserved") for rec in records]

    if verbose:
        print(f"\t >> {len(data)} records read successfully")

    # -- position filter (all-or-nothing) --------------------------------------
    if position is not None and len(metadata):
        wanted = [position] if isinstance(position, int) else list(position)
        valid = set(metadata["POSITION"].tolist())
        if all(p in valid for p in wanted):
            keep = metadata["POSITION"].isin(wanted).to_numpy()
            metadata = metadata.loc[keep].reset_index(drop=True)
            data = [d for d, k in zip(data, keep, strict=True) if k]
            reserved = [r for r, k in zip(reserved, keep, strict=True) if k]
            if verbose:
                print(f"Kept records matching 'position': {sorted(set(wanted))}")
        else:
            _warn(
                "At least one position number is not valid, valid position numbers"
                f" are: {sorted(valid)}"
            )

    # -- zero-data removal -------------------------------------------------------
    if zero_data_rm and len(metadata):
        empty_ids = [i for i, d in enumerate(data) if len(d) == 0]
        if empty_ids:
            _warn(
                "Zero-data records detected and removed:"
                f" {', '.join(str(i + 1) for i in empty_ids)}"
            )
            keep_mask = np.ones(len(data), dtype=bool)
            keep_mask[empty_ids] = False
            metadata = metadata.loc[keep_mask].reset_index(drop=True)
            data = [d for d, k in zip(data, keep_mask, strict=True) if k]
            reserved = [r for r, k in zip(reserved, keep_mask, strict=True) if k]

    if not len(metadata):
        if verbose:
            print("Empty object returned")
        return (
            []
            if fast_forward
            else RisoeBINFileData(metadata=empty_metadata(), data=[], reserved=[])
        )

    # -- duplicate check (adjacent records only; skipped when ROIs present) ------
    if len(data) >= 2 and not (metadata["RECTYPE"] == 128).any():
        duplicates = [
            i
            for i in range(1, len(data))
            if len(data[i - 1]) == len(data[i]) and bool(np.all(data[i - 1] == data[i]))
        ]
        if duplicates:
            labels = ", ".join(str(i + 1) for i in duplicates)
            if duplicated_rm:
                keep_mask = np.ones(len(data), dtype=bool)
                keep_mask[duplicates] = False
                metadata = metadata.loc[keep_mask].reset_index(drop=True)
                data = [d for d, k in zip(data, keep_mask, strict=True) if k]
                reserved = [r for r, k in zip(reserved, keep_mask, strict=True) if k]
                if verbose:
                    print(f"Duplicated records detected and removed: {labels}")
            else:
                _warn(
                    f"Duplicated records detected: {labels}\n"
                    " >> You should consider using 'duplicated_rm = True'."
                )

    # -- ID recalculation ----------------------------------------------------------
    metadata = metadata.copy()
    metadata["ID"] = np.arange(1, len(metadata) + 1)

    if not show_raw_values:
        metadata = _translate(metadata, path)

    result = RisoeBINFileData(metadata=metadata, data=data, reserved=reserved)
    if fast_forward:
        analyses = result.to_analysis()
        return analyses if isinstance(analyses, list) else [analyses]
    return result
