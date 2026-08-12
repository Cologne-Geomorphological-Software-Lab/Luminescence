"""Tests for the BIN/BINX reader against the R package's binary test corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luminescence.core import Analysis, Curve
from luminescence.core.risoe import RisoeBINFileData, build_curve_matrix
from luminescence.io import read_bin
from luminescence.utils.exceptions import DataFormatError, LuminescenceWarning


@pytest.fixture(scope="session")
def bin_tests_dir(r_test_data_dir: Path) -> Path:
    return r_test_data_dir / "bin-tests"


class TestVersions:
    @pytest.mark.parametrize(
        ("version", "filename"),
        [
            (3, "BINfile_V3.bin"),
            (4, "BINfile_V4.bin"),
            (5, "BINfile_V5.binx"),
            (6, "BINfile_V6.binx"),
            (7, "BINfile_V7.binx"),
        ],
    )
    def test_read_all_versions(self, r_test_data_dir: Path, version: int, filename: str) -> None:
        result = read_bin(r_test_data_dir / filename)
        assert isinstance(result, RisoeBINFileData)
        assert len(result) > 0
        assert (result.metadata["VERSION"] == version).all()
        assert (result.metadata["NPOINTS"] > 0).all()
        for i in range(len(result)):
            assert len(result.data[i]) == result.metadata["NPOINTS"].iloc[i]

    def test_read_v8(self, extdata_dir: Path) -> None:
        result = read_bin(extdata_dir / "BINfile_V8.binx")
        assert isinstance(result, RisoeBINFileData)
        assert (result.metadata["VERSION"] == 8).all()
        assert bool(result.metadata["RECTYPE"].isin([0, 1]).all())
        assert (result.metadata["FNAME"] != "").any()

    def test_ltype_translated(self, r_test_data_dir: Path) -> None:
        result = read_bin(r_test_data_dir / "BINfile_V3.bin")
        assert isinstance(result, RisoeBINFileData)
        known = {"TL", "OSL", "IRSL", "TOL", "POSL", "SGOSL", "RL", "XRF", ""}
        assert set(result.metadata["LTYPE"]) <= known

    def test_show_raw_values(self, r_test_data_dir: Path) -> None:
        result = read_bin(r_test_data_dir / "BINfile_V3.bin", show_raw_values=True)
        assert isinstance(result, RisoeBINFileData)
        assert result.metadata["LTYPE"].str.isdigit().all()

    def test_v3_fname_fallback(self, r_test_data_dir: Path) -> None:
        result = read_bin(r_test_data_dir / "BINfile_V3.bin")
        assert isinstance(result, RisoeBINFileData)
        assert (result.metadata["FNAME"] == "BINfile_V3").all()

    def test_time_normalised(self, r_test_data_dir: Path) -> None:
        result = read_bin(r_test_data_dir / "BINfile_V5.binx")
        assert isinstance(result, RisoeBINFileData)
        times = [t for t in result.metadata["TIME"] if t]
        assert all(":" in t for t in times)


class TestEdgeCases:
    def test_corrupted_file(self, bin_tests_dir: Path) -> None:
        with pytest.warns(LuminescenceWarning, match="appears to be corrupt"):
            result = read_bin(bin_tests_dir / "corrupted.bin")
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 1

    def test_corrupted_file_with_n_records(self, bin_tests_dir: Path) -> None:
        with pytest.warns(LuminescenceWarning, match="'n_records' reset"):
            result = read_bin(bin_tests_dir / "corrupted.bin", n_records=[1, 2])
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 1

    def test_two_versions(self, bin_tests_dir: Path) -> None:
        result = read_bin(bin_tests_dir / "two-versions.binx")
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 4
        assert result.metadata["VERSION"].tolist() == [3, 3, 8, 8]
        # V3 rows: RECTYPE defaults to 0; V8 rows carry RECTYPE 1
        assert result.metadata["RECTYPE"].tolist() == [0, 0, 1, 1]
        assert (result.metadata["FNAME"].iloc[2:] != "").all()

    def test_rectype_128(self, bin_tests_dir: Path) -> None:
        result = read_bin(bin_tests_dir / "rectype-128.binx")
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 3
        assert result.metadata["RECTYPE"].tolist() == [1, 1, 128]
        rois = result.data[2]
        assert isinstance(rois, list)
        assert len(rois) == 100
        assert set(rois[0]) == {"NOFPOINTS", "USEDFOR", "SHOWFOR", "ROICOLOR", "X", "Y"}
        assert len(rois[0]["USEDFOR"]) == 48

    def test_rectype_128_fast_forward(self, bin_tests_dir: Path) -> None:
        result = read_bin(bin_tests_dir / "rectype-128.binx", fast_forward=True)
        assert isinstance(result, list)
        # ROI record has POSITION 0, curves POSITION 1 -> separate Analysis objects
        assert all(isinstance(a, Analysis) for a in result)
        curves = [r for a in result for r in a.records]
        assert any(isinstance(c, Curve) and len(c) > 1 for c in curves)

    def test_duplicated_records_warn(self, bin_tests_dir: Path) -> None:
        with pytest.warns(LuminescenceWarning, match="Duplicated records detected: 2"):
            result = read_bin(bin_tests_dir / "duplicated-records.binx")
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 3

    def test_duplicated_records_removed(self, bin_tests_dir: Path) -> None:
        result = read_bin(bin_tests_dir / "duplicated-records.binx", duplicated_rm=True)
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 2
        assert len(result.data) == 2
        assert len(result.reserved) == 2

    def test_zero_data_record_removed(self, bin_tests_dir: Path) -> None:
        with pytest.warns(LuminescenceWarning, match="Zero-data records detected"):
            result = read_bin(bin_tests_dir / "zero-data-record.binx")
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 1
        assert len(result.reserved) == 1

    def test_zero_data_record_kept(self, bin_tests_dir: Path) -> None:
        result = read_bin(bin_tests_dir / "zero-data-record.binx", zero_data_rm=False)
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 2
        assert len(result.data[1]) == 0

    def test_zero_data_all(self, bin_tests_dir: Path) -> None:
        with pytest.warns(LuminescenceWarning, match="Zero-data records detected"):
            result = read_bin(bin_tests_dir / "zero-data-all.binx")
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 0


class TestArguments:
    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_bin("does-not-exist.bin")

    def test_wrong_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "file.R"
        bad.write_bytes(b"x")
        with pytest.raises(ValueError, match="not supported"):
            read_bin(bad)

    def test_zero_byte_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        with pytest.raises(DataFormatError, match="zero-byte"):
            read_bin(empty)

    def test_unsupported_version_first_record(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"\x01" + b"\x00" * 20)
        with pytest.raises(DataFormatError, match="not supported or file is broken"):
            read_bin(bad)

    def test_garbage_with_forced_version(self, tmp_path: Path) -> None:
        garbage = tmp_path / "garbage.bin"
        garbage.write_bytes(b"00\n")
        with pytest.warns(LuminescenceWarning, match="0 records read"):
            assert read_bin(garbage, force_version=8) is None

    def test_unknown_rectype(self, tmp_path: Path) -> None:
        header = b"\x08\x00" + (1507).to_bytes(4, "little") + (0).to_bytes(4, "little")
        header += (250).to_bytes(4, "little") + b"\x63"  # RECTYPE 99
        bad = tmp_path / "rectype99.binx"
        bad.write_bytes(header)
        with pytest.raises(DataFormatError, match="RECTYPE = 99"):
            read_bin(bad)
        result = read_bin(bad, ignore_rectype=True)
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 0

    def test_n_records_selection(self, extdata_dir: Path) -> None:
        result = read_bin(extdata_dir / "BINfile_V8.binx", n_records=1)
        assert isinstance(result, RisoeBINFileData)
        assert len(result) == 1

    def test_invalid_position_warns(self, extdata_dir: Path) -> None:
        with pytest.warns(LuminescenceWarning, match="position number is not valid"):
            result = read_bin(extdata_dir / "BINfile_V8.binx", position=99)
        assert isinstance(result, RisoeBINFileData)
        assert len(result) > 0  # nothing filtered

    def test_file_list(self, r_test_data_dir: Path) -> None:
        files = [
            r_test_data_dir / "BINfile_V3.bin",
            r_test_data_dir / "BINfile_V5.binx",
        ]
        results = read_bin(files)  # type: ignore[arg-type]
        assert isinstance(results, list)
        assert len(results) == 2


class TestBridge:
    def test_to_analysis(self, extdata_dir: Path) -> None:
        result = read_bin(extdata_dir / "BINfile_V8.binx")
        assert isinstance(result, RisoeBINFileData)
        analyses = result.to_analysis()
        items = analyses if isinstance(analyses, list) else [analyses]
        assert all(isinstance(a, Analysis) for a in items)
        first = items[0].records[0]
        assert isinstance(first, Curve)
        assert first.record_type.endswith(" (PMT)")
        assert "POSITION" in first.info
        assert first.pids == (items[0].uid,)

    def test_fast_forward_wraps_in_list(self, extdata_dir: Path) -> None:
        result = read_bin(extdata_dir / "BINfile_V8.binx", fast_forward=True)
        assert isinstance(result, list)
        assert all(isinstance(a, Analysis) for a in result)

    def test_invalid_run_raises(self, extdata_dir: Path) -> None:
        result = read_bin(extdata_dir / "BINfile_V8.binx")
        assert isinstance(result, RisoeBINFileData)
        with pytest.raises(ValueError, match="invalid values"):
            result.to_analysis(run=[9999])


class TestCurveMatrix:
    def test_default_axis(self) -> None:
        matrix = build_curve_matrix(
            np.arange(10.0), version=8, npoints=10, ltype="OSL",
            low=0.0, high=10.0, an_temp=0.0, toldelay=0, tolon=0, toloff=0,
        )  # fmt: skip
        assert matrix.shape == (10, 2)
        assert matrix[0, 0] == pytest.approx(1.0)  # excludes `low`
        assert matrix[-1, 0] == pytest.approx(10.0)  # includes `high`

    def test_zero_points(self) -> None:
        matrix = build_curve_matrix(
            np.empty(0), version=8, npoints=0, ltype="OSL",
            low=0.0, high=1.0, an_temp=0.0, toldelay=0, tolon=0, toloff=0,
        )  # fmt: skip
        assert matrix.shape == (1, 2)
        assert np.isnan(matrix).all()

    def test_tl_nonconform_fallback(self, capsys: pytest.CaptureFixture[str]) -> None:
        matrix = build_curve_matrix(
            np.ones(100), version=8, npoints=100, ltype="TL",
            low=20.0, high=450.0, an_temp=220.0, toldelay=0, tolon=0, toloff=0,
        )  # fmt: skip
        captured = capsys.readouterr()
        assert "non-conform" in captured.out
        # fallback collapses to seq(an_temp, high, npoints) -> starts above an_temp
        assert matrix[0, 0] > 220.0
        assert matrix[-1, 0] == pytest.approx(450.0)

    def test_tl_three_segments(self) -> None:
        matrix = build_curve_matrix(
            np.ones(30), version=8, npoints=30, ltype="TL",
            low=20.0, high=450.0, an_temp=220.0, toldelay=10, tolon=5, toloff=15,
        )  # fmt: skip
        x = matrix[:, 0]
        assert x[9] == pytest.approx(220.0)  # end of start ramp
        assert np.all(x[10:15] == 220.0)  # plateau
        assert x[-1] == pytest.approx(450.0)  # end ramp

    def test_v3_tl_uses_plain_axis(self) -> None:
        matrix = build_curve_matrix(
            np.ones(10), version=3, npoints=10, ltype="TL",
            low=0.0, high=100.0, an_temp=50.0, toldelay=2, tolon=2, toloff=2,
        )  # fmt: skip
        assert matrix[-1, 0] == pytest.approx(100.0)
        assert matrix[0, 0] == pytest.approx(10.0)
