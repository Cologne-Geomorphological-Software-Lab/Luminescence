"""Dose-response curve fitting and equivalent-dose determination.

Port of ``fit_DoseResponseCurve`` (R package v1.7). Implemented fit methods:
``"LIN"``, ``"SSE"`` (the classic saturating exponential, formerly ``"EXP"``),
``"SSE OR LIN"``, and ``"GOK"`` (general-order kinetics). The remaining R
methods (``QDR``, ``SSE+LIN``, ``DSE``, ``OTOR``, ``OTORX``) raise
``NotImplementedError`` until ported.

The Monte-Carlo error model matches R: every LxTx value (including the
natural) is resampled from Normal(value, error), the fit repeated, and
``De.Error`` is the sample standard deviation of the resulting De
distribution (negatives censored in interpolation mode).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from luminescence.core.results import Results
from luminescence.utils.exceptions import LuminescenceWarning

__all__ = ["fit_dose_response_curve"]

_FloatArray = npt.NDArray[np.float64]

_LEGACY_METHODS = {
    "EXP": "SSE",
    "EXP OR LIN": "SSE OR LIN",
    "EXP+LIN": "SSE+LIN",
    "EXP+EXP": "DSE",
    "LambertW": "OTOR",
}
_IMPLEMENTED = {"LIN", "SSE", "SSE OR LIN", "GOK"}
_KNOWN = _IMPLEMENTED | {"QDR", "SSE+LIN", "DSE", "OTOR", "OTORX"}

_DE_COLUMNS = [
    "De", "De.Error", "D01", "D01.ERROR", "D02", "D02.ERROR",
    "R", "R.LOWER", "R.UPPER", "Dc", "Dc.LOWER", "Dc.UPPER",
    "D63", "D63.LOWER", "D63.UPPER", "D80", "D80.LOWER", "D80.UPPER",
    "n_N", "De.MC", "Fit", "Mode",
    "HPDI68_L", "HPDI68_U", "HPDI95_L", "HPDI95_U",
    ".De.plot", ".De.raw",
]  # fmt: skip

Mode = Literal["interpolation", "extrapolation", "alternate"]


def _warn(message: str) -> None:
    warnings.warn(message, LuminescenceWarning, stacklevel=3)


def _sse_model(params: _FloatArray, x: _FloatArray) -> _FloatArray:
    n, d0, di = params
    return n * (1 - np.exp(-(x + di) / d0))


def _gok_model(params: _FloatArray, x: _FloatArray) -> _FloatArray:
    a, d0, c, d = params
    c = max(c, 1e-10)  # GOK order parameter; c -> 0 limit is the exponential
    return a * (d - (1 + (1 / d0) * x * c) ** (-1 / c))


def _extract_columns(data: Any) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Return (dose, lxtx, lxtx_error), by column name if >= 3 names match."""
    names: list[str] | None = None
    if hasattr(data, "columns"):
        names = [str(c).lower() for c in data.columns]
        arr = np.asarray(data, dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("'data' must have at least two columns (dose, LxTx)")
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(arr.shape[0])])

    canonical = ["dose", "lxtx", "lxtx.error"]
    known = [*canonical, "tntx", "test_dose"]
    if names is not None and sum(c in names for c in known) >= 3:
        idx = [names.index(c) if c in names else None for c in canonical]
        cols = [arr[:, i] if i is not None else np.zeros(arr.shape[0]) for i in idx]
        return cols[0], cols[1], cols[2]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _resolve_weights(fit_weights: str | npt.ArrayLike | None, y_error: _FloatArray) -> _FloatArray:
    if fit_weights is not None and (
        np.any(np.isnan(y_error)) or np.any(np.isinf(y_error)) or np.any(y_error == 0)
    ):
        _warn("Error column invalid, infinite, or contains 0, 'fit_weights' reset to None")
        fit_weights = None
    if fit_weights is None:
        return np.ones_like(y_error)
    if isinstance(fit_weights, str):
        if fit_weights == "inverse_std":
            return 1.0 / np.abs(y_error)
        if fit_weights == "norm_inverse_std":
            w = 1.0 / np.abs(y_error)
            return w / np.sum(w)
        if fit_weights == "inverse_var":
            return 1.0 / y_error**2
        raise ValueError(f"unknown fit_weights: {fit_weights!r}")
    w = np.asarray(fit_weights, dtype=np.float64).ravel()
    if w.size == 1:
        return np.full_like(y_error, float(w[0]))
    if w.size != y_error.size:
        raise ValueError("numeric 'fit_weights' must match the number of fitted points")
    return w


def _weighted_linear(
    x: _FloatArray, y: _FloatArray, w: _FloatArray, *, through_origin: bool = False
) -> tuple[float, float]:
    """Weighted least-squares line fit; returns (intercept, slope)."""
    sw = np.sqrt(w)
    if through_origin:
        design = (sw * x)[:, np.newaxis]
        slope = float(np.linalg.lstsq(design, sw * y, rcond=None)[0][0])
        return 0.0, slope
    design = np.column_stack([sw, sw * x])
    coef = np.linalg.lstsq(design, sw * y, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def _nls(
    model: Any,
    x: _FloatArray,
    y: _FloatArray,
    w: _FloatArray,
    start: _FloatArray,
    lower: _FloatArray,
    upper: _FloatArray,
) -> _FloatArray | None:
    """Bounded Levenberg-Marquardt-style fit; None on failure."""
    sw = np.sqrt(w)
    x0 = np.clip(start, lower, upper)

    def residuals(params: _FloatArray) -> _FloatArray:
        with np.errstate(all="ignore"):
            r = sw * (y - model(params, x))
        return np.where(np.isfinite(r), r, 1e150)

    try:
        result = least_squares(residuals, x0, bounds=(lower, upper), method="trf", max_nfev=2000)
    except (ValueError, RuntimeError):
        return None
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    return np.asarray(result.x, dtype=np.float64)


def fit_dose_response_curve(
    data: Any,
    mode: Mode = "interpolation",
    fit_method: str = "SSE",
    *,
    fit_force_through_origin: bool = False,
    fit_weights: str | npt.ArrayLike | None = "inverse_var",
    fit_including_repeated_reg_points: bool = True,
    fit_bounds: bool = True,
    n_mc: int = 100,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
) -> Results | None:
    """Fit a dose-response curve and derive the equivalent dose (De).

    Args:
        data: Table with columns (Dose, LxTx, LxTx.Error, [TnTx, Test_Dose]),
            by name (case-insensitive) or position. **Row 0 is the natural.**
        mode: "interpolation" (De from the natural signal), "extrapolation"
            (De from the x-intercept), or "alternate" (fit only).
        fit_method: "LIN", "SSE" (saturating exponential), "SSE OR LIN", "GOK".
        rng: Seed or generator for the Monte-Carlo error estimation.

    Returns:
        :class:`Results` with entries De (full column schema), De.MC, Fit,
        Formula — or None when the input cannot be fitted at all.
    """
    fit_method = _LEGACY_METHODS.get(fit_method, fit_method)
    if fit_method not in _KNOWN:
        raise ValueError(f"unknown fit_method: {fit_method!r}")
    if fit_method not in _IMPLEMENTED:
        raise NotImplementedError(
            f"fit_method {fit_method!r} is not yet ported; available: {sorted(_IMPLEMENTED)}"
        )
    if mode not in ("interpolation", "extrapolation", "alternate"):
        raise ValueError(f"unknown mode: {mode!r}")

    dose, lxtx, lxtx_error = _extract_columns(data)

    if np.any(np.isinf(dose)) or np.any(np.isinf(lxtx)) or np.any(np.isinf(lxtx_error)):
        _warn("Inf values found, replaced by NA")
        dose = np.where(np.isinf(dose), np.nan, dose)
        lxtx = np.where(np.isinf(lxtx), np.nan, lxtx)
        lxtx_error = np.where(np.isinf(lxtx_error), np.nan, lxtx_error)

    complete = ~(np.isnan(dose) | np.isnan(lxtx) | np.isnan(lxtx_error))
    if not np.all(complete):
        _warn(f"{int(np.sum(~complete))} NA values removed")
        dose, lxtx, lxtx_error = dose[complete], lxtx[complete], lxtx_error[complete]
    if dose.size == 0:
        _warn("Nothing to fit after NA removal, None returned")
        return None
    if float(np.sum(np.abs(np.diff(dose)))) == 0.0:
        _warn("All points have the same dose, None returned")
        return None
    if np.any(lxtx == 0):
        _warn("LxTx values == 0 replaced by machine epsilon")
        lxtx = np.where(lxtx == 0, np.finfo(np.float64).eps, lxtx)

    # -- fit-data selection (row 0 = natural) ------------------------------
    first_idx = 1 if mode == "interpolation" else 0
    x_fit = dose[first_idx:]
    y_fit = lxtx[first_idx:]
    e_fit = lxtx_error[first_idx:]

    if not fit_including_repeated_reg_points:
        _, unique_first = np.unique(x_fit, return_index=True)
        keep = np.zeros(x_fit.size, dtype=bool)
        keep[unique_first] = True
        x_fit, y_fit, e_fit = x_fit[keep], y_fit[keep], e_fit[keep]

    w = _resolve_weights(fit_weights, e_fit)

    # minimum-data fallback to LIN
    num_params = {"SSE": 3, "SSE OR LIN": 3, "GOK": 4}.get(fit_method)
    if fit_method != "LIN" and num_params is not None and x_fit.size < num_params:
        _warn(
            f"Fitting using {fit_method!r} requires at least {num_params} dose points,"
            " fit method changed to 'LIN'"
        )
        fit_method = "LIN"

    generator = np.random.default_rng(rng)

    # -- MC resampling inputs ----------------------------------------------
    data_mc = generator.normal(
        loc=y_fit[:, np.newaxis],
        scale=np.abs(e_fit[:, np.newaxis]),
        size=(y_fit.size, n_mc),
    )
    if mode == "interpolation":
        data_mc_de = generator.normal(lxtx[0], abs(lxtx_error[0]), n_mc)
    else:
        data_mc_de = np.zeros(n_mc)

    # -- start-value heuristics ---------------------------------------------
    a_start = float(np.max(y_fit))
    b_start = 1.0
    positive = y_fit > 0
    if np.any(positive):
        with np.errstate(all="ignore"):
            _, log_slope = _weighted_linear(x_fit[positive], np.log(y_fit[positive]), w[positive])
        if np.isfinite(log_slope) and log_slope != 0:
            b_start = 1.0 / log_slope
    lin_intercept, lin_slope = _weighted_linear(x_fit, y_fit, w)
    c_start = abs(lin_intercept / lin_slope) if lin_slope != 0 else 0.0
    g_start = float(np.max(y_fit / np.max(x_fit)))

    a_mc = generator.normal(a_start, abs(a_start) / 100 or 1e-6, 50)
    b_mc = generator.normal(b_start, abs(b_start) / 100 or 1e-6, 50)
    c_mc = (
        np.zeros(50)
        if fit_force_through_origin
        else generator.normal(c_start, abs(c_start) / 100 or 1e-6, 50)
    )
    del g_start  # needed by SSE+LIN only (not yet ported)

    x_natural = np.full(n_mc, np.nan)
    var_d0 = np.zeros(n_mc)
    de_row: dict[str, Any] = dict.fromkeys(_DE_COLUMNS, np.nan)
    de = np.nan
    d01 = np.nan
    d01_error = np.nan
    formula = ""
    fit_params: dict[str, float] | None = None
    fit_message = ""

    ln_tn = lxtx[0]
    y_target = ln_tn if mode == "interpolation" else 0.0

    # -- SSE ------------------------------------------------------------------
    if fit_method in ("SSE", "SSE OR LIN"):
        pre_lower = np.array([0.0, 1e-6, 0.0])
        pre_upper = np.array([np.inf, np.inf, np.inf])
        pre_fits = [
            fit
            for i in range(50)
            if (
                fit := _nls(
                    _sse_model, x_fit, y_fit, np.ones_like(y_fit),
                    np.array([a_mc[i], b_mc[i], c_mc[i]]), pre_lower, pre_upper,
                )
            )
            is not None
        ]  # fmt: skip

        lower = np.array([0.0, 0.0, 0.0]) if fit_bounds else np.array([-np.inf] * 3)
        upper = (
            np.array([np.inf, np.inf, 0.0]) if fit_force_through_origin else np.array([np.inf] * 3)
        )
        if fit_force_through_origin and fit_bounds:
            upper[2] = 1e-12  # R pins Di to [0, 0]; least_squares needs lb < ub
        start = np.array(
            [
                float(np.median([f[0] for f in pre_fits])) if pre_fits else a_start,
                float(np.mean(b_mc)),  # quirk kept from R (issue 1552)
                0.0,  # quirk kept from R: Di start is hard-coded to 0
            ]
        )
        final = _nls(_sse_model, x_fit, y_fit, w, start, lower, upper)
        if final is None and pre_fits:
            final = pre_fits[-1]

        if final is None:
            if fit_method == "SSE OR LIN":
                fit_method = "LIN"
            else:
                fit_message = f"Fit failed for SSE ({mode})"
        else:
            n_fit, d0_fit, di_fit = (float(v) for v in final)
            fit_params = {"N": n_fit, "D0": d0_fit, "Di": di_fit}
            d01 = d0_fit
            with np.errstate(all="ignore"):
                if mode == "interpolation":
                    de = -di_fit - d0_fit * float(np.log(1 - ln_tn / n_fit))
                elif mode == "extrapolation":
                    de = -di_fit
            formula = f"y ~ {n_fit:.6g} * (1 - exp(-(x + {di_fit:.6g}) / {d0_fit:.6g}))"
            for i in range(n_mc):
                mc_fit = _nls(_sse_model, x_fit, data_mc[:, i], w, start, lower, upper)
                if mc_fit is not None and mode != "alternate":
                    n_i, d0_i, di_i = (float(v) for v in mc_fit)
                    var_d0[i] = d0_i
                    with np.errstate(all="ignore"):
                        x_natural[i] = -di_i - d0_i * float(np.log(1 - data_mc_de[i] / n_i))
            d01_error = float(np.std(var_d0, ddof=1))
            fit_message = f"Fit:    SSE ({mode}) | De = {abs(de):.2f} | D01 = {d01:.2f}"

    # -- GOK ------------------------------------------------------------------
    if fit_method == "GOK":
        lower = np.zeros(4) if fit_bounds else np.array([-np.inf] * 4)
        upper = (
            np.array([np.inf, np.inf, np.inf, 1.0])
            if fit_force_through_origin
            else np.array([np.inf] * 4)
        )
        start = np.array([a_start, b_start, 1.0, 1.0])
        final = _nls(_gok_model, x_fit, y_fit, w, start, lower, upper)
        if final is None:
            fit_message = f"Fit failed for GOK ({mode})"
        else:
            a_fit, d0_fit, c_fit, d_fit = (float(v) for v in final)
            fit_params = {"a": a_fit, "D0": d0_fit, "c": c_fit, "d": d_fit}
            d01 = d0_fit

            def gok_de(a: float, d0: float, c: float, d: float, y: float) -> float:
                with np.errstate(all="ignore"):
                    u = (a * d - y) / a
                    return -d0 * (1 - u ** (-c)) / c

            if mode != "alternate":
                de = gok_de(a_fit, d0_fit, c_fit, d_fit, y_target)
            formula = (
                f"y ~ {a_fit:.6g} * ({d_fit:.6g} - (1 + (1/{d0_fit:.6g}) * x *"
                f" {c_fit:.6g})^(-1/{c_fit:.6g}))"
            )
            for i in range(n_mc):
                mc_fit = _nls(_gok_model, x_fit, data_mc[:, i], w, start, lower, upper)
                if mc_fit is not None and mode != "alternate":
                    var_d0[i] = float(mc_fit[1])
                    x_natural[i] = gok_de(
                        float(mc_fit[0]),
                        float(mc_fit[1]),
                        float(mc_fit[2]),
                        float(mc_fit[3]),
                        float(data_mc_de[i]),
                    )
            d01_error = float(np.std(var_d0, ddof=1))
            fit_message = f"Fit:    GOK ({mode}) | De = {abs(de):.2f} | D01 = {d01:.2f}"

    # -- LIN ------------------------------------------------------------------
    if fit_method == "LIN":
        intercept, slope = _weighted_linear(
            x_fit, y_fit, w, through_origin=fit_force_through_origin
        )
        fit_params = {"intercept": intercept, "slope": slope}
        if mode != "alternate" and slope != 0:
            de = (y_target - intercept) / slope
        formula = f"y ~ {intercept:.6g} + {slope:.6g} * x"
        for i in range(n_mc):
            ic_i, sl_i = _weighted_linear(
                x_fit, data_mc[:, i], w, through_origin=fit_force_through_origin
            )
            if mode != "alternate" and sl_i != 0:
                x_natural[i] = (data_mc_de[i] - ic_i) / sl_i
        fit_message = f"Fit:    LIN ({mode}) | De = {abs(de):.2f}"

    # -- Monte-Carlo post-processing -------------------------------------------
    de_raw = de
    if mode == "interpolation":
        de_mc = np.maximum(x_natural, 0)
        de_mc_na = np.where(x_natural < 0, np.nan, x_natural)
        if not np.isnan(de) and de < 0:
            de = np.nan
    elif mode == "extrapolation":
        de_mc = de_mc_na = np.abs(x_natural)
    else:
        de_mc = de_mc_na = x_natural

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        de_mc_mean = float(np.nanmean(de_mc)) if de_mc.size else np.nan
        finite = de_mc_na[~np.isnan(de_mc_na)]
        de_error = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan

    de_row.update(
        {
            "De": abs(de) if not np.isnan(de) else np.nan,
            "De.Error": de_error,
            "D01": d01,
            "D01.ERROR": d01_error,
            "De.MC": de_mc_mean,
            "Fit": fit_method,
            "Mode": mode,
            ".De.plot": de,
            ".De.raw": de_raw,
        }
    )

    if verbose and fit_message:
        print(f"[fit_dose_response_curve()] {fit_message}")

    return Results(
        originator="fit_dose_response_curve",
        info={"fit_message": fit_message} if fit_message else {},
        data={
            "De": de_row,
            "De.MC": x_natural,
            "Fit": fit_params,
            "Fit.Args": {
                "fit_method": fit_method,
                "mode": mode,
                "fit_force_through_origin": fit_force_through_origin,
                "fit_bounds": fit_bounds,
                "n_mc": n_mc,
            },
            "Formula": formula,
        },
    )
