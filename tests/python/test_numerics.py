"""Tests for calc_statistics, calc_osl_lxtx_ratio, and fit_dose_response_curve.

Numerical parity against R reference values is covered separately by the
fixture-based tests; here we verify behaviour, quirks, and self-consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from luminescence.analysis.lxtx import calc_osl_lxtx_ratio
from luminescence.fitting.dose_response import fit_dose_response_curve
from luminescence.models.statistics import calc_statistics
from luminescence.utils.exceptions import LuminescenceWarning
from luminescence.utils.numerics import weighted_median


class TestWeightedMedian:
    def test_equal_weights_odd(self) -> None:
        assert weighted_median([3, 1, 2], [1, 1, 1]) == 2.0

    def test_dominant_weight(self) -> None:
        assert weighted_median([1, 2, 100], [0.1, 0.1, 10]) == 100.0


class TestCalcStatistics:
    DATA = np.column_stack([[10.0, 12.0, 9.0, 11.0, 10.5], [0.5, 0.6, 0.4, 0.5, 0.55]])

    def test_unweighted_matches_numpy(self) -> None:
        stats = calc_statistics(self.DATA)
        values = self.DATA[:, 0]
        assert stats["unweighted"]["n"] == 5
        assert stats["unweighted"]["mean"] == pytest.approx(values.mean())
        assert stats["unweighted"]["sd.abs"] == pytest.approx(values.std(ddof=1))
        assert stats["unweighted"]["median"] == pytest.approx(np.median(values))

    def test_weighted_mean(self) -> None:
        stats = calc_statistics(self.DATA)
        w = 1 / self.DATA[:, 1] ** 2
        assert stats["weighted"]["mean"] == pytest.approx(np.sum(w * self.DATA[:, 0]) / np.sum(w))

    def test_weighted_skew_kurt_are_unweighted(self) -> None:
        # quirk kept from R
        stats = calc_statistics(self.DATA)
        assert stats["weighted"]["skewness"] == stats["unweighted"]["skewness"]
        assert stats["weighted"]["kurtosis"] == stats["unweighted"]["kurtosis"]

    def test_zero_error_replaces_whole_column(self) -> None:
        data = self.DATA.copy()
        data[0, 1] = 0.0
        stats = calc_statistics(data)  # no warning: sum(errors) != 0
        w_uniform = np.full(5, 1 / 5)
        assert stats["weighted"]["mean"] == pytest.approx(
            np.sum(w_uniform * data[:, 0]) / np.sum(w_uniform)
        )

    def test_all_zero_errors_warn(self) -> None:
        data = np.column_stack([self.DATA[:, 0], np.zeros(5)])
        with pytest.warns(LuminescenceWarning, match="automatically set"):
            calc_statistics(data)

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="only NA"):
            calc_statistics(np.column_stack([[np.nan, np.nan], [1.0, 1.0]]))

    def test_mcm_reproducible_with_seed(self) -> None:
        a = calc_statistics(self.DATA, n_mcm=1000, rng=42)
        b = calc_statistics(self.DATA, n_mcm=1000, rng=42)
        assert a["MCM"] == b["MCM"]
        assert a["MCM"]["mean"] == pytest.approx(a["unweighted"]["mean"], rel=0.01)

    def test_digits(self) -> None:
        stats = calc_statistics(self.DATA, digits=2)
        assert stats["unweighted"]["mean"] == round(float(self.DATA[:, 0].mean()), 2)


def make_pair(
    signal: float = 5000.0, background: float = 100.0, n: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(1, n + 1, dtype=float) * 0.1
    lx = np.full(n, background)
    lx[:10] += signal
    tx = np.full(n, background)
    tx[:10] += signal / 2
    return np.column_stack([x, lx]), np.column_stack([x, tx])


class TestCalcOslLxTxRatio:
    def test_basic_ratio(self) -> None:
        lx, tx = make_pair()
        results = calc_osl_lxtx_ratio(
            lx, tx, signal_integral=range(1, 11), background_integral=range(51, 101)
        )
        table = results["LxTx.table"]
        # constant background subtracts out exactly
        assert table["Net_LnLx"] == pytest.approx(50000.0)
        assert table["Net_TnTx"] == pytest.approx(25000.0)
        assert table["LxTx"] == pytest.approx(2.0)
        assert table["LxTx.Error"] > 0

    def test_alternate_mode(self) -> None:
        lx, tx = make_pair()
        results = calc_osl_lxtx_ratio(lx, tx, signal_integral=None)
        table = results["LxTx.table"]
        assert table["LxTx"] == pytest.approx(np.sum(lx[:, 1]) / np.sum(tx[:, 1]))
        assert table["LxTx.Error"] == 0.0

    def test_channel_mismatch_raises(self) -> None:
        lx, tx = make_pair()
        with pytest.raises(ValueError, match="Different number of channels"):
            calc_osl_lxtx_ratio(
                lx,
                tx[:50],
                signal_integral=range(1, 11),
                background_integral=range(51, 101),
            )

    def test_background_before_signal_raises(self) -> None:
        lx, tx = make_pair()
        with pytest.raises(ValueError, match="no elements between"):
            calc_osl_lxtx_ratio(
                lx, tx, signal_integral=range(1, 11), background_integral=range(5, 8)
            )

    def test_measurement_input_equivalent_to_channels(self) -> None:
        lx, tx = make_pair()
        by_channel = calc_osl_lxtx_ratio(
            lx, tx, signal_integral=range(1, 11), background_integral=range(51, 101)
        )
        by_time = calc_osl_lxtx_ratio(
            lx,
            tx,
            signal_integral=[0.1, 1.0],
            background_integral=[5.1, 10.0],
            integral_input="measurement",
        )
        assert by_time["LxTx.table"] == by_channel["LxTx.table"]

    def test_poisson_vs_nonpoisson(self) -> None:
        rng = np.random.default_rng(7)
        lx, tx = make_pair()
        lx[:, 1] += rng.normal(0, 50, lx.shape[0])  # overdispersed background
        kwargs = {
            "signal_integral": range(1, 11),
            "background_integral": range(31, 101),
        }
        poisson = calc_osl_lxtx_ratio(lx, tx, background_count_distribution="poisson", **kwargs)
        nonpoisson = calc_osl_lxtx_ratio(lx, tx, **kwargs)
        assert nonpoisson["LxTx.table"]["Net_LnLx.Error"] >= poisson["LxTx.table"]["Net_LnLx.Error"]

    def test_missing_tx_yields_nan_not_zero(self) -> None:
        # regression: a missing Tx curve must produce a *missing* ratio (NaN),
        # not 0.0 — R returns NA here; 0.0 would look like a real measurement
        lx, _ = make_pair()
        results = calc_osl_lxtx_ratio(
            lx, None, signal_integral=range(1, 11), background_integral=range(51, 101)
        )
        table = results["LxTx.table"]
        assert np.isnan(table["LxTx"])
        assert np.isnan(table["LxTx.Error"])
        assert np.isnan(table["Net_TnTx"])

    def test_zero_curves_do_not_raise(self) -> None:
        zeros = np.column_stack([np.arange(1, 101, dtype=float), np.zeros(100)])
        results = calc_osl_lxtx_ratio(
            zeros,
            zeros,
            signal_integral=range(1, 11),
            background_integral=range(51, 101),
        )
        assert results["LxTx.table"]["LxTx"] == 0.0

    def test_digits_rounding(self) -> None:
        lx, tx = make_pair()
        results = calc_osl_lxtx_ratio(
            lx,
            tx,
            signal_integral=range(1, 11),
            background_integral=range(51, 101),
            digits=2,
        )
        assert results["LxTx.table"]["LxTx"] == round(results["LxTx.table"]["LxTx"], 2)


def synthetic_sar_table(
    de_true: float = 1700.0, d0: float = 1500.0, n_scale: float = 5.0
) -> np.ndarray:
    """Saturating-exponential SAR table; row 0 is the natural."""
    doses = np.array([0.0, 450.0, 1050.0, 2000.0, 2550.0, 450.0, 0.0])
    lxtx = n_scale * (1 - np.exp(-doses / d0))
    lxtx[0] = n_scale * (1 - np.exp(-de_true / d0))  # natural signal
    lxtx[lxtx == 0] = 0.002
    errors = np.maximum(lxtx * 0.02, 0.001)
    return np.column_stack([doses, lxtx, errors])


class TestFitDoseResponseCurve:
    def test_sse_recovers_de(self) -> None:
        table = synthetic_sar_table(de_true=1700.0)
        results = fit_dose_response_curve(table, rng=1)
        assert results is not None
        de = results["De"]
        assert de["Fit"] == "SSE"
        assert de["De"] == pytest.approx(1700.0, rel=0.01)
        assert 0 < de["De.Error"] < 400
        assert de["D01"] == pytest.approx(1500.0, rel=0.05)

    def test_lin_fit(self) -> None:
        doses = np.array([0.0, 100.0, 200.0, 300.0])
        lxtx = np.array([1.5, 1.0, 2.0, 3.0])  # natural = 1.5 -> De = 150
        table = np.column_stack([doses, lxtx, np.full(4, 0.05)])
        results = fit_dose_response_curve(table, fit_method="LIN", rng=1)
        assert results is not None
        assert results["De"]["De"] == pytest.approx(150.0, rel=1e-6)

    def test_gok_close_to_sse_for_first_order(self) -> None:
        table = synthetic_sar_table(de_true=1700.0)
        results = fit_dose_response_curve(table, fit_method="GOK", rng=1)
        assert results is not None
        assert results["De"]["De"] == pytest.approx(1700.0, rel=0.05)

    def test_same_dose_returns_none(self) -> None:
        table = np.column_stack([np.full(4, 100.0), np.arange(4.0) + 1, np.full(4, 0.1)])
        with pytest.warns(LuminescenceWarning, match="same dose"):
            assert fit_dose_response_curve(table, rng=1) is None

    def test_too_few_points_falls_back_to_lin(self) -> None:
        table = np.column_stack([[0.0, 100.0, 200.0], [0.5, 1.0, 2.0], [0.01, 0.02, 0.04]])
        with pytest.warns(LuminescenceWarning, match="changed to 'LIN'"):
            results = fit_dose_response_curve(table, rng=1)
        assert results is not None
        assert results["De"]["Fit"] == "LIN"

    def test_unimplemented_method_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="not yet ported"):
            fit_dose_response_curve(synthetic_sar_table(), fit_method="DSE", rng=1)

    def test_legacy_name_mapping(self) -> None:
        results = fit_dose_response_curve(synthetic_sar_table(), fit_method="EXP", rng=1)
        assert results is not None
        assert results["De"]["Fit"] == "SSE"

    def test_reproducible_with_seed(self) -> None:
        a = fit_dose_response_curve(synthetic_sar_table(), rng=99)
        b = fit_dose_response_curve(synthetic_sar_table(), rng=99)
        assert a is not None and b is not None
        assert a["De"]["De.Error"] == b["De"]["De.Error"]

    def test_extrapolation_mode(self) -> None:
        doses = np.array([0.0, 100.0, 200.0, 300.0])
        lxtx = np.array([0.5, 1.0, 1.5, 2.0])  # zero-intercept at x = -100
        table = np.column_stack([doses, lxtx, np.full(4, 0.02)])
        results = fit_dose_response_curve(table, mode="extrapolation", fit_method="LIN", rng=1)
        assert results is not None
        assert results["De"]["De"] == pytest.approx(100.0, rel=1e-6)
