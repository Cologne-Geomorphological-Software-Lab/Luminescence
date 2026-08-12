"""Numerical parity tests against R-generated reference fixtures.

The fixtures under ``tests/python/fixtures/`` were produced by
``tools/generate_fixtures.R`` with the CRAN release of the R package
(see MANIFEST.json for provenance). Deterministic quantities are compared
tightly; Monte-Carlo error estimates only statistically (seeds cannot match
across languages).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from luminescence.analysis.lxtx import calc_osl_lxtx_ratio
from luminescence.analysis.sar_cwosl import analyse_sar_cwosl
from luminescence.core import Analysis
from luminescence.core.risoe import METADATA_SCHEMA, RisoeBINFileData
from luminescence.fitting.dose_response import fit_dose_response_curve
from luminescence.models.statistics import calc_statistics

pytestmark = pytest.mark.skipif(
    not (Path(__file__).parent / "fixtures" / "analysis").is_dir(),
    reason="R-generated fixtures not present (run tools/generate_fixtures.R)",
)

FIXTURES = Path(__file__).parent / "fixtures"

# Tolerances (see conftest.py): the LxTx chain is pure arithmetic -> 1e-9;
# fitted parameters go through different optimizers (minpack.lm vs. scipy
# trf) -> 1e-4 on parameters is justified; MC errors are statistical.
RTOL_TABLE = 1e-9
RTOL_FIT = 1e-4


def load_json(relative: str) -> Any:
    return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cwosl_sar_data() -> RisoeBINFileData:
    """Rebuild ExampleData.BINfileData$CWOSL.SAR.Data from exported fixtures."""
    metadata = pd.read_csv(FIXTURES / "io" / "CWOSL.SAR.Data__METADATA.csv")
    for name, (dtype, default) in METADATA_SCHEMA.items():
        if name not in metadata.columns:
            metadata[name] = default
        with contextlib.suppress(ValueError, TypeError):
            # keep the exported dtype where R and the schema disagree
            metadata[name] = metadata[name].astype(dtype)
    data = [
        np.asarray(counts, dtype=np.float64) for counts in load_json("io/CWOSL.SAR.Data__DATA.json")
    ]
    return RisoeBINFileData(metadata=metadata, data=data, reserved=[None] * len(data))


@pytest.fixture(scope="module")
def pos1_analysis(cwosl_sar_data: RisoeBINFileData) -> Analysis:
    analysis = cwosl_sar_data.to_analysis(pos=1)
    assert isinstance(analysis, Analysis)
    return analysis


@pytest.fixture(scope="module")
def sar_reference() -> dict[str, Any]:
    return load_json("analysis/analyse_SAR.CWOSL__pos1_sig1-2_bg900-1000_SSE.json")


@pytest.fixture(scope="module")
def sar_results(pos1_analysis: Analysis) -> Any:
    return analyse_sar_cwosl(
        pos1_analysis,
        signal_integral=range(1, 3),
        background_integral=range(900, 1001),
        rng=1,
    )


class TestSarParity:
    """DoD of phase 1: the full chain reproduces R's De on the example data."""

    def test_lxtx_table_deterministic_columns(
        self, sar_results: Any, sar_reference: dict[str, Any]
    ) -> None:
        assert sar_results is not None
        table = sar_results["LnLxTnTx.table"]
        reference = pd.DataFrame(sar_reference["LnLxTnTx.table"])
        assert len(table) == len(reference) == 7
        assert table["Name"].tolist() == reference["Name"].tolist()
        assert table["Repeated"].tolist() == reference["Repeated"].tolist()
        for column in [
            "Dose", "LnLx", "LnLx.BG", "TnTx", "TnTx.BG",
            "Net_LnLx", "Net_LnLx.Error", "Net_TnTx", "Net_TnTx.Error",
            "SN_RATIO_LnLx", "SN_RATIO_TnTx", "LxTx", "LxTx.Error",
        ]:  # fmt: skip
            np.testing.assert_allclose(
                table[column].to_numpy(),
                reference[column].to_numpy(dtype=np.float64),
                rtol=RTOL_TABLE,
                err_msg=f"column {column}",
            )

    def test_de_matches_r(self, sar_results: Any, sar_reference: dict[str, Any]) -> None:
        assert sar_results is not None
        row = sar_results["data"].iloc[0]
        ref = sar_reference["data"][0]
        assert row["De"] == pytest.approx(ref["De"], rel=RTOL_FIT)
        assert row["D01"] == pytest.approx(ref["D01"], rel=RTOL_FIT)
        assert row["Fit"] == ref["Fit"] == "SSE"
        assert row["RC.Status"] == ref["RC.Status"] == "OK"
        assert row["signal.range"] == "1:2"
        assert row["background.range"] == "900:1000"

    def test_de_error_statistically_compatible(
        self, sar_results: Any, sar_reference: dict[str, Any]
    ) -> None:
        # MC standard errors from different RNG streams (n.MC = 100); the
        # relative sampling error of an sd estimate is ~1/sqrt(2*99) ~ 7%,
        # so 30% covers both draws comfortably.
        assert sar_results is not None
        row = sar_results["data"].iloc[0]
        ref = sar_reference["data"][0]
        assert row["De.Error"] == pytest.approx(ref["De.Error"], rel=0.3)

    def test_rejection_criteria_match(
        self, sar_results: Any, sar_reference: dict[str, Any]
    ) -> None:
        assert sar_results is not None
        crit = sar_results["rejection.criteria"]
        reference = pd.DataFrame(sar_reference["rejection.criteria"])
        assert crit["Criteria"].tolist() == reference["Criteria"].tolist()
        assert crit["Status"].tolist() == reference["Status"].tolist()
        deterministic = ~reference["Criteria"].str.startswith("Palaeodose")
        np.testing.assert_allclose(
            crit.loc[deterministic, "Value"].to_numpy(dtype=np.float64),
            reference.loc[deterministic, "Value"].to_numpy(dtype=np.float64),
            rtol=1e-6,
        )


class TestLxTxParity:
    def test_first_pair(self, pos1_analysis: Analysis) -> None:
        reference = load_json("analysis/calc_OSLLxTxRatio__pos1_first_pair.json")
        ref_row = reference["LxTx.table"][0]
        lx = pos1_analysis.records[1]  # record 0 is TL
        tx = pos1_analysis.records[3]
        results = calc_osl_lxtx_ratio(
            lx,  # type: ignore[arg-type]
            tx,  # type: ignore[arg-type]
            signal_integral=range(1, 3),
            background_integral=range(900, 1001),
        )
        table = results["LxTx.table"]
        for key, ref_value in ref_row.items():
            if key in table:
                assert table[key] == pytest.approx(ref_value, rel=RTOL_TABLE), key


class TestStatisticsParity:
    def test_bt998(self) -> None:
        data = pd.read_csv(FIXTURES / "models" / "ExampleData.DeValues__BT998.csv")
        reference = load_json("models/calc_Statistics__BT998.json")
        stats = calc_statistics(data.to_numpy())
        for variant in ("weighted", "unweighted"):
            for key, ref_value in reference[variant].items():
                assert stats[variant][key] == pytest.approx(
                    ref_value[0] if isinstance(ref_value, list) else ref_value,
                    rel=1e-9,
                ), f"{variant}.{key}"


class TestFitParity:
    def test_sse_fit_on_sar_table(self, sar_reference: dict[str, Any]) -> None:
        reference = load_json("fitting/fit_DoseResponseCurve__pos1_SSE.json")
        ref_de = reference["De"][0]
        table = pd.DataFrame(sar_reference["LnLxTnTx.table"])
        fit_input = pd.DataFrame(
            {
                "Dose": table["Dose"].astype(float),
                "LxTx": table["LxTx"].astype(float),
                "LxTx.Error": table["LxTx.Error"].astype(float),
            }
        )
        results = fit_dose_response_curve(fit_input, fit_method="SSE", rng=7)
        assert results is not None
        row = results["De"]
        assert row["De"] == pytest.approx(ref_de["De"], rel=RTOL_FIT)
        assert row["D01"] == pytest.approx(ref_de["D01"], rel=RTOL_FIT)
        assert row["De.Error"] == pytest.approx(ref_de["De.Error"], rel=0.3)
