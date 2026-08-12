"""Unit tests for the core object model (Record, Curve, Analysis, Results)."""

from __future__ import annotations

import numpy as np
import pytest

from luminescence.core import Analysis, Curve, ImageData, Record, Results, Spectrum


def make_curve(record_type: str = "OSL", n: int = 5, scale: float = 1.0) -> Curve:
    x = np.arange(1, n + 1, dtype=float)
    data = np.column_stack([x, x * scale])
    return Curve(record_type=record_type, curve_type="measured", data=data)


class TestRecord:
    def test_unique_uids(self) -> None:
        assert Record().uid != Record().uid

    def test_with_parent_extends_provenance(self) -> None:
        parent = Record()
        child = Record().with_parent(parent)
        assert child.pids == (parent.uid,)
        grandchild = child.with_parent(child)
        assert grandchild.pids == (parent.uid, child.uid)

    def test_with_parent_creates_new_uid(self) -> None:
        parent = Record()
        child = parent.with_parent(parent)
        assert child.uid != parent.uid


class TestCurve:
    def test_from_two_columns(self) -> None:
        curve = make_curve(n=3)
        assert len(curve) == 3
        assert curve.data.shape == (3, 2)

    def test_bare_counts_get_channel_axis(self) -> None:
        curve = Curve(data=np.array([10.0, 20.0, 30.0]))
        assert curve.x.tolist() == [1.0, 2.0, 3.0]
        assert curve.y.tolist() == [10.0, 20.0, 30.0]

    def test_invalid_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            Curve(data=np.zeros((3, 3)))

    def test_arithmetic_with_curve(self) -> None:
        a = make_curve(scale=2.0)
        b = make_curve(scale=1.0)
        assert np.array_equal((a - b).y, b.y)
        assert np.array_equal((a / b).y, np.full(5, 2.0))
        assert np.array_equal((a + b).y, 3 * b.y)
        assert np.array_equal((a * b).y, 2 * b.y**2)

    def test_arithmetic_with_scalar(self) -> None:
        curve = make_curve(scale=1.0)
        assert np.array_equal((curve * 2).y, 2 * curve.y)
        assert np.array_equal((curve * 2).x, curve.x)

    def test_arithmetic_incompatible_axes(self) -> None:
        with pytest.raises(ValueError, match="different resolution"):
            _ = make_curve(n=5) + make_curve(n=6)

    def test_arithmetic_records_provenance(self) -> None:
        a, b = make_curve(), make_curve()
        assert a.uid in (a + b).pids

    def test_slicing(self) -> None:
        curve = make_curve(n=10)
        head = curve[:3]
        assert isinstance(head, Curve)
        assert len(head) == 3
        assert head.record_type == curve.record_type

    def test_numpy_interop(self) -> None:
        assert np.asarray(make_curve(n=4)).shape == (4, 2)

    def test_payload_equality_ignores_uid(self) -> None:
        assert make_curve() == make_curve()
        assert make_curve(scale=1.0) != make_curve(scale=2.0)

    def test_to_dataframe(self) -> None:
        df = make_curve(n=3).to_dataframe()
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 3


class TestSpectrum:
    def test_default_axes(self) -> None:
        spectrum = Spectrum(data=np.ones((4, 2)))
        assert spectrum.wavelengths.tolist() == [1, 2, 3, 4]
        assert spectrum.times.tolist() == [1, 2]

    def test_axis_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="do not match"):
            Spectrum(data=np.ones((4, 2)), wavelengths=np.arange(3.0))

    def test_long_dataframe(self) -> None:
        df = Spectrum(data=np.arange(6.0).reshape(3, 2)).to_dataframe()
        assert list(df.columns) == ["wavelength", "time", "signal"]
        assert len(df) == 6


class TestImageData:
    def test_single_frame_promoted(self) -> None:
        image = ImageData(data=np.ones((5, 4)))
        assert image.data.shape == (5, 4, 1)
        assert image.n_frames == 1

    def test_invalid_dims(self) -> None:
        with pytest.raises(ValueError, match="2-D or 3-D"):
            ImageData(data=np.ones(5))


class TestAnalysis:
    def make_sequence(self) -> Analysis:
        records = [
            make_curve("TL"),
            make_curve("OSL", scale=3.0),
            make_curve("OSL", scale=1.0),
            make_curve("IRSL"),
        ]
        return Analysis(protocol="SAR", records=list(records))

    def test_container_protocol(self) -> None:
        seq = self.make_sequence()
        assert len(seq) == 4
        assert [r.record_type for r in seq] == ["TL", "OSL", "OSL", "IRSL"]  # type: ignore[attr-defined]
        assert make_curve("TL") in seq

    def test_int_indexing_returns_record(self) -> None:
        assert isinstance(self.make_sequence()[0], Curve)

    def test_slice_returns_analysis_with_provenance(self) -> None:
        seq = self.make_sequence()
        sub = seq[1:3]
        assert isinstance(sub, Analysis)
        assert len(sub) == 2
        assert seq.uid in sub.pids

    def test_index_list(self) -> None:
        sub = self.make_sequence()[[0, 3]]
        assert sub.names == ["TL", "IRSL"]

    def test_get_records_by_type(self) -> None:
        seq = self.make_sequence()
        assert len(seq.get_records("OSL")) == 2
        assert len(seq.get_records(["TL", "IRSL"])) == 2
        assert seq.get_records("nope") == []

    def test_get_records_regex(self) -> None:
        seq = self.make_sequence()
        assert len(seq.get_records("SL$", regex=True)) == 3  # OSL, OSL, IRSL
        assert len(seq.get_records("L$", regex=True)) == 4

    def test_subset(self) -> None:
        sub = self.make_sequence().subset("OSL")
        assert isinstance(sub, Analysis)
        assert sub.names == ["OSL", "OSL"]
        assert sub.protocol == "SAR"

    def test_describe(self) -> None:
        df = self.make_sequence().describe()
        assert len(df) == 4
        assert df["record_type"].tolist() == ["TL", "OSL", "OSL", "IRSL"]
        assert df["x_min"].iloc[0] == 1.0


class TestResults:
    def test_mapping_protocol(self) -> None:
        results = Results(originator="calc_test", data={"summary": {"De": [1.0]}, "args": {}})
        assert list(results) == ["summary", "args"]
        assert results["summary"] == {"De": [1.0]}
        assert results.get("missing") is None
        assert len(results) == 2

    def test_to_dataframe_defaults_to_first_entry(self) -> None:
        results = Results(data={"summary": {"De": [1.0, 2.0]}})
        df = results.to_dataframe()
        assert df["De"].tolist() == [1.0, 2.0]

    def test_to_dataframe_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Results().to_dataframe()
