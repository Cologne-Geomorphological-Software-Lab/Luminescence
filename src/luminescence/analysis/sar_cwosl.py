"""SAR CW-OSL analysis (port of ``analyse_SAR.CWOSL``, R package).

Phase-1 scope: the numerical protocol — Lx/Tx pairing, LxTx table, rejection
criteria, and De determination. Not yet ported: plotting, OSL components
(``OSLdecomposition``), the Bluszcz ``od_rates`` error model, XSYG
irradiation-time extraction, and channel trimming.
"""

from __future__ import annotations

import dataclasses
import re
import uuid
import warnings
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from luminescence.analysis.lxtx import calc_osl_lxtx_ratio
from luminescence.core.analysis import Analysis
from luminescence.core.curve import Curve
from luminescence.core.results import Results
from luminescence.fitting.dose_response import Mode, fit_dose_response_curve
from luminescence.utils.exceptions import LuminescenceWarning
from luminescence.utils.validation import to_channels, validate_integral

__all__ = ["analyse_sar_cwosl"]

_CW_TYPE_PATTERN = re.compile(r"(P?OSL[a-zA-Z]*|IRSL[a-zA-Z]*)")

_REJECTION_DEFAULTS: dict[str, Any] = {
    "recycling.ratio": 10.0,  # percent
    "recuperation.rate": 10.0,  # percent
    "palaeodose.error": 10.0,  # percent
    "testdose.error": 10.0,  # percent
    "sn.ratio": np.nan,  # absolute; NaN = not evaluated
    "exceed.max.regpoint": True,
    "consider.uncertainties": False,
    "sn_reference": "Natural",
    "recuperation_reference": "Natural",
}

_DE_NA_COLUMNS = [
    "De", "De.Error", "D01", "D01.ERROR", "D02", "D02.ERROR",
    "R", "R.LOWER", "R.UPPER", "Dc", "Dc.LOWER", "Dc.UPPER",
    "D63", "D63.LOWER", "D63.UPPER", "D80", "D80.LOWER", "D80.UPPER",
    "n_N", "De.MC", "Fit", "Mode",
    "HPDI68_L", "HPDI68_U", "HPDI95_L", "HPDI95_U",
    ".De.plot", ".De.raw",
]  # fmt: skip


def _warn(message: str) -> None:
    warnings.warn(message, LuminescenceWarning, stacklevel=3)


def _se_ratio(num: float, num_err: float, den: float, den_err: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(num / den * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2))


def _status(value: float, threshold: float, *, greater: bool = False) -> str:
    """ "OK" when threshold is NaN or the comparison holds (NaN value fails)."""
    if np.isnan(threshold):
        return "OK"
    ok = value >= threshold if greater else value <= threshold
    return "OK" if ok else "FAILED"


def _info_lookup(info: dict[str, Any], key: str) -> Any:
    for candidate, value in info.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


def _format_range(integral: npt.NDArray[np.int64] | None) -> str:
    if integral is None:
        return "NA:NA"
    return f"{int(integral.min())}:{int(integral.max())}"


def analyse_sar_cwosl(
    obj: Analysis | list[Analysis],
    signal_integral: npt.ArrayLike | None = None,
    background_integral: npt.ArrayLike | None = None,
    *,
    signal_integral_tx: npt.ArrayLike | None = None,
    background_integral_tx: npt.ArrayLike | None = None,
    integral_input: Literal["channel", "measurement"] = "channel",
    rejection_criteria: dict[str, Any] | None = None,
    dose_points: npt.ArrayLike | None = None,
    dose_rate_source: float | None = None,
    only_lxtx_table: bool = False,
    background_count_distribution: Literal["non-poisson", "poisson"] = "non-poisson",
    sigmab: float | tuple[float, float] | None = None,
    sig0: float = 0.0,
    mode: Mode = "interpolation",
    fit_method: str = "SSE",
    fit_weights: str | npt.ArrayLike | None = "inverse_var",
    fit_force_through_origin: bool = False,
    fit_including_repeated_reg_points: bool = True,
    n_mc: int = 100,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
) -> Results | None:
    """Analyse one or more SAR CW-OSL measurement sequences.

    Runs the single-aliquot regenerative-dose (SAR) protocol after Murray and
    Wintle (2000): pairs the OSL/IRSL curves into (Lx, Tx), computes the LxTx
    table via :func:`~luminescence.analysis.lxtx.calc_osl_lxtx_ratio`,
    evaluates the rejection criteria, fits the dose-response curve, and
    derives the equivalent dose. Port of the R function ``analyse_SAR.CWOSL``.

    The records of ``obj`` must alternate Lx, Tx, Lx, Tx, ... after filtering
    to the dominant OSL/IRSL curve type; records whose type starts with an
    underscore and irradiation steps are dropped beforehand. Row 0 of the
    resulting LxTx table is the natural signal.

    Args:
        obj: One measurement sequence, or a list of sequences that are
            analysed one after the other and row-bound in the output (column
            ``ALQ`` numbers the aliquots).
        signal_integral: 1-based, inclusive signal channels (e.g.
            ``range(1, 3)``). ``None`` selects the alternate mode (full-curve
            sums, no errors).
        background_integral: Background channels; must start after the signal
            integral. A single channel is expanded to the 26 channels ending
            there.
        signal_integral_tx: Separate signal channels for the Tx curves;
            defaults to ``signal_integral``.
        background_integral_tx: Separate background channels for the Tx
            curves; defaults to ``background_integral``.
        integral_input: ``"channel"`` (default) or ``"measurement"`` (x-axis
            units, converted using the first CW curve's time axis).
        rejection_criteria: Overrides for the acceptance thresholds. Keys and
            defaults: ``recycling.ratio`` (10, percent), ``recuperation.rate``
            (10, percent), ``palaeodose.error`` (10, percent),
            ``testdose.error`` (10, percent), ``sn.ratio`` (NaN, absolute),
            ``exceed.max.regpoint`` (True), ``consider.uncertainties``
            (False), ``sn_reference`` ("Natural"), ``recuperation_reference``
            ("Natural").
        dose_points: Regeneration doses, one per Lx/Tx pair; overrides the
            ``IRR_TIME`` values from the record metadata.
        dose_rate_source: Source dose rate (e.g. Gy/s); multiplies all dose
            points, turning seconds into absorbed dose.
        only_lxtx_table: Skip the dose-response fit and De determination;
            only the LxTx table and the curve-level rejection criteria are
            returned.
        background_count_distribution: Error model for the count statistics,
            see :func:`~luminescence.analysis.lxtx.calc_osl_lxtx_ratio`.
        sigmab: Overdispersion override, see ``calc_osl_lxtx_ratio``.
        sig0: Extra relative error on LxTx, see ``calc_osl_lxtx_ratio``.
        mode: De determination mode passed to the fit: ``"interpolation"``
            (default), ``"extrapolation"``, or ``"alternate"``.
        fit_method: Dose-response model, see
            :func:`~luminescence.fitting.dose_response.fit_dose_response_curve`.
        fit_weights: Fit weighting scheme, see ``fit_dose_response_curve``.
        fit_force_through_origin: Force the dose-response curve through the
            origin.
        fit_including_repeated_reg_points: Include repeated regeneration
            points in the fit (default True).
        n_mc: Number of Monte-Carlo runs for the De error.
        rng: Seed or generator for the Monte-Carlo error estimation.
        verbose: Print the fit message.

    Returns:
        :class:`~luminescence.core.results.Results` with entries ``"data"``
        (one summary row per aliquot: De, De.Error, D01, fit metadata,
        RC.Status, integral ranges, position/grain), ``"LnLxTnTx.table"``
        (one row per Lx/Tx pair), ``"rejection.criteria"`` (Criteria, Value,
        Threshold, Status), and ``"Formula"``. ``None`` (with a warning) when
        the sequence cannot be analysed.

    Raises:
        TypeError: If ``obj`` is neither an Analysis nor a list of them.
        ValueError: If dose points are missing or of wrong length, an
            integral is invalid, or a rejection-criteria reference names an
            unknown dose point.
    """
    if isinstance(obj, list):
        return _analyse_list(
            obj,
            signal_integral=signal_integral,
            background_integral=background_integral,
            signal_integral_tx=signal_integral_tx,
            background_integral_tx=background_integral_tx,
            integral_input=integral_input,
            rejection_criteria=rejection_criteria,
            dose_points=dose_points,
            dose_rate_source=dose_rate_source,
            only_lxtx_table=only_lxtx_table,
            background_count_distribution=background_count_distribution,
            sigmab=sigmab,
            sig0=sig0,
            mode=mode,
            fit_method=fit_method,
            fit_weights=fit_weights,
            fit_force_through_origin=fit_force_through_origin,
            fit_including_repeated_reg_points=fit_including_repeated_reg_points,
            n_mc=n_mc,
            rng=rng,
            verbose=verbose,
        )
    if not isinstance(obj, Analysis):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("'obj' should be an Analysis or a list of Analysis objects")

    # -- record preparation --------------------------------------------------
    records = [r for r in obj.records if not str(getattr(r, "record_type", "")).startswith("_")]
    records = [r for r in records if getattr(r, "record_type", "") != "irradiation"]

    matches = [
        m.group(0)
        for r in records
        if (m := _CW_TYPE_PATTERN.search(str(getattr(r, "record_type", ""))))
    ]
    if not matches:
        _warn("No record of type 'OSL', 'IRSL', 'POSL' detected, None returned")
        return None
    counts: dict[str, int] = {}
    for m in matches:
        counts[m] = counts.get(m, 0) + 1
    cw_type = max(counts, key=lambda key: counts[key])

    # normalise record types: strip everything from the first space
    records = [
        dataclasses.replace(r, record_type=re.sub(r" .*", "", str(getattr(r, "record_type", ""))))
        if isinstance(r, Curve)
        else r
        for r in records
    ]

    cw_ids = [i for i, r in enumerate(records) if getattr(r, "record_type", "") == cw_type]
    if len(cw_ids) % 2 != 0:
        _warn("Input OSL/IRSL curves are not a multiple of two, None returned")
        return None
    cw_curves = [records[i] for i in cw_ids]
    lengths = {len(c) for c in cw_curves if isinstance(c, Curve)}
    if len(lengths) > 1:
        _warn(f"Input curves have different lengths {sorted(lengths)}, None returned")
        return None
    if not cw_curves or not isinstance(cw_curves[0], Curve):
        _warn("No usable curve records found, None returned")
        return None
    channel_length = len(cw_curves[0])

    # -- integral handling ----------------------------------------------------
    if signal_integral is None or background_integral is None:
        signal_integral = background_integral = None
        signal_integral_tx = background_integral_tx = None
        _warn(
            "No signal or background integral applied as 'signal_integral' or"
            " 'background_integral' is None (alternate mode)"
        )
        sig_v = bg_v = sig_tx_v = bg_tx_v = None
    else:
        sig_arr = np.asarray(signal_integral, dtype=np.float64).ravel()
        bg_arr = np.asarray(background_integral, dtype=np.float64).ravel()
        if integral_input == "measurement":
            x_axis = cw_curves[0].data[:, 0]
            sig_arr = to_channels(sig_arr, x_axis, "signal_integral")
            bg_arr = to_channels(bg_arr, x_axis, "background_integral")
            if signal_integral_tx is not None:
                signal_integral_tx = to_channels(
                    np.asarray(signal_integral_tx, dtype=np.float64).ravel(),
                    x_axis,
                    "signal_integral_tx",
                )
            if background_integral_tx is not None:
                background_integral_tx = to_channels(
                    np.asarray(background_integral_tx, dtype=np.float64).ravel(),
                    x_axis,
                    "background_integral_tx",
                )
        # sig_arr/bg_arr are non-None here, so the validator always returns arrays
        sig_v = cast(
            "npt.NDArray[np.int64]",
            validate_integral(sig_arr, "signal_integral", 1, channel_length),
        )
        bg_v = cast(
            "npt.NDArray[np.int64]",
            validate_integral(bg_arr, "background_integral", int(sig_v.max()) + 1, channel_length),
        )
        if bg_v.size == 1:
            v = int(bg_v[0])
            bg_v = np.arange(v - 25, v + 1, dtype=np.int64)
            _warn(
                "Background integral should contain at least two values, reset to"
                f" {int(bg_v.min())}:{int(bg_v.max())}"
            )

        if signal_integral_tx is None and background_integral_tx is not None:
            signal_integral_tx = sig_v
            _warn(f"'signal_integral_tx' set automatically to {_format_range(sig_v)}")
        sig_tx_v = (
            None
            if signal_integral_tx is None
            else validate_integral(
                np.asarray(signal_integral_tx, dtype=np.float64).ravel(),
                "signal_integral_tx",
                1,
                channel_length,
            )
        )
        if signal_integral_tx is not None and background_integral_tx is None:
            background_integral_tx = bg_v
            _warn(f"'background_integral_tx' set automatically to {_format_range(bg_v)}")
        bg_tx_v = (
            None
            if background_integral_tx is None
            else validate_integral(
                np.asarray(background_integral_tx, dtype=np.float64).ravel(),
                "background_integral_tx",
                1 if sig_tx_v is None else int(sig_tx_v.max()) + 1,
                channel_length,
            )
        )
        if bg_tx_v is not None and bg_tx_v.size == 1:
            v = int(bg_tx_v[0])
            bg_tx_v = np.arange(v - 25, v + 1, dtype=np.int64)
            _warn(
                "Background integral limits for Tx curves cannot be equal, reset to"
                f" {int(bg_tx_v.min())}:{int(bg_tx_v.max())}"
            )

    criteria = dict(_REJECTION_DEFAULTS)
    if rejection_criteria:
        criteria.update(rejection_criteria)

    # -- Lx/Tx pairing ---------------------------------------------------------
    lx_ids = cw_ids[0::2]
    tx_ids = cw_ids[1::2]

    # -- dose points -------------------------------------------------------------
    n_pairs = len(lx_ids)
    doses = np.array(
        [
            float(v) if (v := _info_lookup(records[i].info, "IRR_TIME")) is not None else np.nan
            for i in lx_ids
        ]
    )
    if dose_points is not None:
        dp = np.asarray(dose_points, dtype=np.float64).ravel()
        if dp.size != n_pairs:
            raise ValueError(
                f"Length of 'dose_points' ({dp.size}) differs from number of curves ({n_pairs})"
            )
        doses = dp.copy()
    elif np.any(np.isnan(doses)):
        raise ValueError("'dose_points' contains NA values or was not set")
    if dose_rate_source is not None:
        doses = doses * float(dose_rate_source)
    if doses[0] != 0 and mode != "alternate":
        _warn(
            f"The natural signal has a dose of {doses[0]:g}, which is indicative of a"
            " dose recovery test. The natural dose was set to 0."
        )
        doses[0] = 0.0

    # -- LxTx table --------------------------------------------------------------
    rows = []
    for lx_id, tx_id in zip(lx_ids, tx_ids, strict=True):
        result = calc_osl_lxtx_ratio(
            records[lx_id],  # type: ignore[arg-type]
            records[tx_id],  # type: ignore[arg-type]
            signal_integral=None if sig_v is None else sig_v,
            background_integral=None if bg_v is None else bg_v,
            signal_integral_tx=None if sig_tx_v is None else sig_tx_v,
            background_integral_tx=None if bg_tx_v is None else bg_tx_v,
            background_count_distribution=background_count_distribution,
            sigmab=sigmab,
            sig0=sig0,
        )
        rows.append(result["LxTx.table"])
    table = pd.DataFrame(rows)

    names = [f"R{i}" for i in range(n_pairs)]
    zero_ids = np.flatnonzero(doses == 0)
    for i in zero_ids:
        names[i] = "R0"
    if zero_ids.size:
        names[zero_ids[0]] = "Natural"
    dose_series = pd.Series(doses)
    repeated = dose_series.duplicated().to_numpy().copy()
    repeated[doses == 0] = False

    table.insert(0, "Dose", doses)
    table.insert(0, "Repeated", repeated)
    table.insert(0, "Name", names)
    table["Test_Dose"] = -1.0

    result_uid = str(uuid.uuid4())  # one UID keys all output tables of this run
    table["UID"] = result_uid

    # -- rejection criteria ---------------------------------------------------------
    consider_unc = bool(criteria["consider.uncertainties"])
    crit_rows: list[dict[str, Any]] = []

    lxtx = table["LxTx"].to_numpy()
    lxtx_err = table["LxTx.Error"].to_numpy()

    if np.any(repeated):
        for idx in np.flatnonzero(repeated):
            prev_candidates = np.flatnonzero((doses == doses[idx]) & ~repeated)
            prev = int(prev_candidates[0])
            ratio = float(lxtx[idx] / lxtx[prev])
            if consider_unc:
                u = _se_ratio(lxtx[idx], lxtx_err[idx], lxtx[prev], lxtx_err[prev])
                ratio = ratio - u if ratio > 1 else ratio + u
            ratio = round(ratio, 4)
            thr = float(criteria["recycling.ratio"]) / 100
            crit_rows.append(
                {
                    "Criteria": f"Recycling ratio ({names[idx]}/{names[prev]})",
                    "Value": ratio,
                    "Threshold": 1 + thr if ratio > 1 else 1 - thr,
                    "Status": _status(abs(1 - ratio), thr),
                }
            )

    recup_ref = str(criteria["recuperation_reference"])
    if recup_ref not in names:
        raise ValueError(f"Recuperation reference invalid, valid values are: {sorted(set(names))}")
    r0_ids = [i for i, name in enumerate(names) if name == "R0"]
    ref_ids = [i for i, name in enumerate(names) if name == recup_ref]
    for count, i in enumerate(r0_ids, start=1):
        ref = ref_ids[0]
        value = float(lxtx[i] / lxtx[ref])
        if consider_unc:
            value -= _se_ratio(lxtx[i], lxtx_err[i], lxtx[ref], lxtx_err[ref])
        recup_thr = float(criteria["recuperation.rate"]) / 100
        status = "OK" if np.isnan(value) else _status(value, recup_thr)
        crit_rows.append(
            {
                "Criteria": f"Recuperation rate ({recup_ref}) {count}",
                "Value": value,
                "Threshold": float(criteria["recuperation.rate"]) / 100,
                "Status": status,
            }
        )

    net_tntx = table["Net_TnTx"].to_numpy()
    net_tntx_err = table["Net_TnTx.Error"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        testdose_error = float(np.divide(net_tntx_err[0], net_tntx[0]))
    crit_rows.append(
        {
            "Criteria": "Testdose error",
            "Value": testdose_error,
            "Threshold": float(criteria["testdose.error"]) / 100,
            "Status": _status(testdose_error, float(criteria["testdose.error"]) / 100),
        }
    )

    sn_ref = str(criteria["sn_reference"])
    if sn_ref not in names:
        raise ValueError(
            f"Signal-to-noise reference invalid, valid values are: {sorted(set(names))}"
        )
    sn_value = float(table["SN_RATIO_LnLx"].to_numpy()[names.index(sn_ref)])
    crit_rows.append(
        {
            "Criteria": "Signal-to-noise ratio",
            "Value": sn_value,
            "Threshold": float(criteria["sn.ratio"]),
            "Status": _status(sn_value, float(criteria["sn.ratio"]), greater=True),
        }
    )

    crit_rows = [row for row in crit_rows if not np.isnan(float(row["Value"]))]

    # -- De determination -------------------------------------------------------------
    de_row: dict[str, Any] = dict.fromkeys(_DE_NA_COLUMNS, np.nan)
    formula = ""
    if not only_lxtx_table:
        fit_input = pd.DataFrame(
            {
                "Dose": table["Dose"],
                "LxTx": table["LxTx"],
                "LxTx.Error": table["LxTx.Error"],
                "TnTx": table["Net_TnTx"],
                "Test_Dose": table["Test_Dose"],
            }
        )
        fit = fit_dose_response_curve(
            fit_input,
            mode=mode,
            fit_method=fit_method,
            fit_force_through_origin=fit_force_through_origin,
            fit_weights=fit_weights,
            fit_including_repeated_reg_points=fit_including_repeated_reg_points,
            n_mc=n_mc,
            rng=rng,
            verbose=verbose,
        )
        if fit is not None:
            de_row = dict(fit["De"])
            formula = fit["Formula"]
            de = float(de_row["De"]) if not pd.isna(de_row["De"]) else np.nan
            de_error = float(de_row["De.Error"]) if not pd.isna(de_row["De.Error"]) else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                pal_err = round(float(np.divide(de_error, de)), 5)
            pal_thr = float(criteria["palaeodose.error"]) / 100
            crit_rows.append(
                {
                    "Criteria": "Palaeodose error",
                    "Value": pal_err,
                    "Threshold": pal_thr,
                    "Status": _status(pal_err, pal_thr),
                }
            )
            exceed = criteria["exceed.max.regpoint"]
            if exceed is None or (isinstance(exceed, float) and np.isnan(exceed)):
                exceed_thr = np.nan
            elif exceed is False:
                exceed_thr = np.inf
            else:
                exceed_thr = float(np.max(doses))
            crit_rows.append(
                {
                    "Criteria": "De > max. dose point",
                    "Value": de - (de_error if consider_unc else 0.0),
                    "Threshold": exceed_thr,
                    "Status": _status(de, exceed_thr),
                }
            )

    criteria_table = pd.DataFrame(crit_rows, columns=["Criteria", "Value", "Threshold", "Status"])
    criteria_table.insert(0, "UID", result_uid)
    rc_status = "FAILED" if (criteria_table["Status"] == "FAILED").any() else "OK"

    # -- summary row --------------------------------------------------------------------
    positions = {
        v for r in records if (v := _info_lookup(getattr(r, "info", {}), "position")) is not None
    }
    grains = {
        v for r in records if (v := _info_lookup(getattr(r, "info", {}), "grain")) is not None
    }
    summary = dict(de_row)
    summary.update(
        {
            "RC.Status": rc_status,
            "signal.range": _format_range(sig_v),
            "background.range": _format_range(bg_v),
            "signal.range.Tx": _format_range(sig_tx_v),
            "background.range.Tx": _format_range(bg_tx_v),
            "ALQ": 1,
            "POS": float(next(iter(positions))) if len(positions) == 1 else np.nan,
            "GRAIN": float(next(iter(grains))) if len(grains) == 1 else np.nan,
            "UID": result_uid,
        }
    )

    return Results(
        originator="analyse_sar_cwosl",
        uid=result_uid,
        data={
            "data": pd.DataFrame([summary]),
            "LnLxTnTx.table": table,
            "rejection.criteria": criteria_table,
            "Formula": formula,
        },
    )


def _analyse_list(objects: list[Analysis], **kwargs: Any) -> Results | None:
    """Sequential multi-aliquot analysis; results row-bound, ALQ renumbered."""
    results = []
    for item in objects:
        if not isinstance(item, Analysis):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        result = analyse_sar_cwosl(item, **kwargs)
        if result is not None:
            results.append(result)
    if not results:
        return None

    data = pd.concat([r["data"] for r in results], ignore_index=True)
    data["ALQ"] = np.arange(1, len(data) + 1)
    tables = pd.concat([r["LnLxTnTx.table"] for r in results], ignore_index=True)
    crits = pd.concat([r["rejection.criteria"] for r in results], ignore_index=True)
    return Results(
        originator="analyse_sar_cwosl",
        pids=tuple(r.uid for r in results),
        data={
            "data": data,
            "LnLxTnTx.table": tables,
            "rejection.criteria": crits,
            "Formula": [r["Formula"] for r in results],
        },
    )
