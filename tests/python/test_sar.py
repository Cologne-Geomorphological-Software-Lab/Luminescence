"""Tests for analyse_sar_cwosl on synthetic SAR sequences."""

from __future__ import annotations

import numpy as np
import pytest

from luminescence.analysis.sar_cwosl import analyse_sar_cwosl
from luminescence.core import Analysis, Curve
from luminescence.utils.exceptions import LuminescenceWarning

D0_TRUE = 1500.0
N_CHANNELS = 100
SIGNAL = range(1, 11)
BACKGROUND = range(51, 101)


def _cw_curve(level: float, irr_time: float, rng: np.random.Generator) -> Curve:
    """A decaying CW-OSL curve whose initial signal encodes the dose response."""
    x = np.arange(1, N_CHANNELS + 1, dtype=float) * 0.1
    decay = np.exp(-np.arange(N_CHANNELS) / 5.0)
    y = 40.0 + level * 4000.0 * decay + rng.normal(0, 3, N_CHANNELS)
    return Curve(
        record_type="OSL",
        curve_type="measured",
        data=np.column_stack([x, y]),
        info={"IRR_TIME": irr_time, "POSITION": 1, "GRAIN": 0},
    )


def make_sar_sequence(
    de_true: float = 1700.0,
    doses: tuple[float, ...] = (0.0, 450.0, 1050.0, 2000.0, 2550.0, 450.0, 0.0),
    seed: int = 3,
) -> Analysis:
    """Alternating Lx, Tx, ... sequence following a saturating exponential."""
    rng = np.random.default_rng(seed)
    records: list[Curve] = []
    for i, dose in enumerate(doses):
        applied = de_true if i == 0 else dose
        lx_level = 1 - np.exp(-applied / D0_TRUE)
        records.append(_cw_curve(lx_level, dose, rng))
        records.append(_cw_curve(0.35, 15.0, rng))  # test dose response
    return Analysis(protocol="SAR", records=list(records))


class TestAnalyseSarCwosl:
    def test_de_recovery(self) -> None:
        seq = make_sar_sequence(de_true=1700.0)
        results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, rng=1, n_mc=50)
        assert results is not None
        row = results["data"].iloc[0]
        assert row["Fit"] == "SSE"
        assert row["De"] == pytest.approx(1700.0, rel=0.05)
        assert row["De.Error"] > 0
        assert row["RC.Status"] == "OK"
        assert row["POS"] == 1.0

    def test_lxtx_table_structure(self) -> None:
        seq = make_sar_sequence()
        results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, rng=1, n_mc=10)
        assert results is not None
        table = results["LnLxTnTx.table"]
        assert list(table["Name"]) == ["Natural", "R1", "R2", "R3", "R4", "R5", "R0"]
        assert list(table["Repeated"]) == [
            False,
            False,
            False,
            False,
            False,
            True,
            False,
        ]
        assert table.columns[:3].tolist() == ["Name", "Repeated", "Dose"]
        assert table["Dose"].iloc[0] == 0.0

    def test_rejection_criteria_present(self) -> None:
        seq = make_sar_sequence()
        results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, rng=1, n_mc=10)
        assert results is not None
        crit = results["rejection.criteria"]
        labels = crit["Criteria"].tolist()
        assert "Recycling ratio (R5/R1)" in labels
        assert "Recuperation rate (Natural) 1" in labels
        assert "Testdose error" in labels
        assert "Palaeodose error" in labels
        assert "De > max. dose point" in labels
        # healthy synthetic data passes everything
        assert (crit["Status"] == "OK").all()

    def test_recycling_ratio_close_to_one(self) -> None:
        seq = make_sar_sequence()
        results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, rng=1, n_mc=10)
        assert results is not None
        crit = results["rejection.criteria"]
        recycling = crit[crit["Criteria"].str.startswith("Recycling")]["Value"].iloc[0]
        assert recycling == pytest.approx(1.0, abs=0.1)

    def test_only_lxtx_table_skips_fit(self) -> None:
        seq = make_sar_sequence()
        results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, only_lxtx_table=True, rng=1)
        assert results is not None
        assert np.isnan(results["data"].iloc[0]["De"])
        labels = results["rejection.criteria"]["Criteria"].tolist()
        assert "Palaeodose error" not in labels

    def test_odd_curve_count_returns_none(self) -> None:
        seq = make_sar_sequence()
        broken = Analysis(protocol="SAR", records=list(seq.records[:-1]))
        with pytest.warns(LuminescenceWarning, match="not a multiple of two"):
            assert analyse_sar_cwosl(broken, SIGNAL, BACKGROUND, rng=1) is None

    def test_no_osl_records_returns_none(self) -> None:
        seq = Analysis(records=[Curve(record_type="TL", data=np.ones(5))])
        with pytest.warns(LuminescenceWarning, match="No record of type"):
            assert analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, rng=1) is None

    def test_dose_points_override(self) -> None:
        seq = make_sar_sequence()
        doses = [0.0, 450.0, 1050.0, 2000.0, 2550.0, 450.0, 0.0]
        results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, dose_points=doses, rng=1, n_mc=10)
        assert results is not None
        assert results["LnLxTnTx.table"]["Dose"].tolist() == doses

    def test_natural_dose_reset_warning(self) -> None:
        seq = make_sar_sequence()
        doses = [100.0, 450.0, 1050.0, 2000.0, 2550.0, 450.0, 0.0]
        with pytest.warns(LuminescenceWarning, match="natural dose was set to 0"):
            results = analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, dose_points=doses, rng=1, n_mc=10)
        assert results is not None
        assert results["LnLxTnTx.table"]["Dose"].iloc[0] == 0.0

    def test_dose_points_length_mismatch(self) -> None:
        seq = make_sar_sequence()
        with pytest.raises(ValueError, match="differs from number of curves"):
            analyse_sar_cwosl(seq, SIGNAL, BACKGROUND, dose_points=[0.0, 1.0], rng=1)

    def test_multi_aliquot_list(self) -> None:
        seqs = [make_sar_sequence(seed=3), make_sar_sequence(seed=4)]
        results = analyse_sar_cwosl(seqs, SIGNAL, BACKGROUND, rng=1, n_mc=10)
        assert results is not None
        assert len(results["data"]) == 2
        assert results["data"]["ALQ"].tolist() == [1, 2]
        assert results["LnLxTnTx.table"]["UID"].nunique() == 2

    def test_underscore_records_removed(self) -> None:
        seq = make_sar_sequence()
        extra = Curve(record_type="_background", data=np.ones(5))
        padded = Analysis(protocol="SAR", records=[extra, *seq.records])
        results = analyse_sar_cwosl(padded, SIGNAL, BACKGROUND, rng=1, n_mc=10)
        assert results is not None
        assert len(results["LnLxTnTx.table"]) == 7


class TestPlotGrowthCurve:
    def test_returns_axes(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from luminescence.fitting.dose_response import fit_dose_response_curve
        from luminescence.plot.growth_curve import plot_growth_curve

        doses = np.array([0.0, 450.0, 1050.0, 2000.0, 2550.0])
        lxtx = 5.0 * (1 - np.exp(-doses / 1500.0))
        lxtx[0] = 5.0 * (1 - np.exp(-1700.0 / 1500.0))
        table = np.column_stack([doses, lxtx, np.maximum(lxtx * 0.02, 0.001)])
        fit = fit_dose_response_curve(table, rng=1, n_mc=10)
        assert fit is not None
        ax = plot_growth_curve(fit, table)
        assert ax.get_xlabel() == "Dose [s]"
        assert len(ax.lines) >= 1
