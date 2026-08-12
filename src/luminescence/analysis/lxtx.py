"""Lx/Tx ratio calculation for CW-OSL curves.

Port of ``calc_OSLLxTxRatio`` (R package v0.9.8), Galbraith (2002/2014) error
model. The Bluszcz et al. (2015) ``od_rates`` error model is not yet ported.

Channel integrals are 1-based and inclusive, matching the R API and the
conventions of the instrument software.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from luminescence.core.curve import Curve
from luminescence.core.results import Results
from luminescence.utils.exceptions import LuminescenceWarning
from luminescence.utils.validation import to_channels, validate_integral

__all__ = ["calc_osl_lxtx_ratio"]

_FloatArray = npt.NDArray[np.float64]


def _as_matrix(data: Curve | npt.ArrayLike, name: str) -> _FloatArray:
    if isinstance(data, Curve):
        return data.data
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.column_stack([np.arange(1, arr.size + 1, dtype=np.float64), arr])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"'{name}' must have two columns (x, counts)")
    return arr[:, :2]


def _sigmab(
    counts: _FloatArray,
    signal: npt.NDArray[np.int64],
    background: npt.NDArray[np.int64] | None,
    k: float,
    what: str,
) -> float:
    """Overdispersion estimate, Galbraith (2002) eq. 4."""
    if background is None:
        return 0.0
    n_sig = signal.size
    min_bg = int(background.min())
    if round(k, 1) >= 2 and min_bg + n_sig * 3 <= counts.size:
        n_blocks = int(k)
        starts = min_bg - 1 + n_sig * np.arange(n_blocks)
        y = np.array([counts[s : s + n_sig].sum() for s in starts])
        scale = 1.0
    else:
        if background.size < 25:
            warnings.warn(
                f"Number of background channels for {what} < 25,"
                " error estimation might not be reliable",
                LuminescenceWarning,
                stacklevel=4,
            )
        y = counts[background - 1]
        scale = float(n_sig)
    return max(float(np.var(y, ddof=1)) - float(np.mean(y)), 0.0) * scale


def _rse(y0: float, y1: float, k: float, sigmab: float) -> float:
    """Relative standard error of a net count (Galbraith 2014 eq. 6)."""
    return float(np.sqrt(y0 + y1 / k**2 + sigmab * (1 + 1 / k)) / (y0 - y1 / k))


def calc_osl_lxtx_ratio(
    lx_data: Curve | npt.ArrayLike,
    tx_data: Curve | npt.ArrayLike | None = None,
    signal_integral: npt.ArrayLike | None = None,
    background_integral: npt.ArrayLike | None = None,
    *,
    signal_integral_tx: npt.ArrayLike | None = None,
    background_integral_tx: npt.ArrayLike | None = None,
    integral_input: Literal["channel", "measurement"] = "channel",
    background_count_distribution: Literal["non-poisson", "poisson"] = "non-poisson",
    use_previous_bg: bool = False,
    sigmab: float | tuple[float, float] | None = None,
    sig0: float = 0.0,
    digits: int | None = None,
) -> Results:
    """Lx/Tx ratio and its error for one pair of CW-OSL curves.

    ``signal_integral=None`` selects the *alternate* mode: full-curve sums
    with zero errors. ``background_integral=None`` disables background
    subtraction. Integrals are 1-based inclusive channel numbers (or
    measurement values with ``integral_input="measurement"``).

    Returns a :class:`Results` with ``"LxTx.table"`` (single-row dict of
    columns) and ``"calc.parameters"``.
    """
    lx = _as_matrix(lx_data, "lx_data")
    if tx_data is not None:
        tx = _as_matrix(tx_data, "tx_data")
        if lx.shape[0] != tx.shape[0]:
            raise ValueError(
                f"Different number of channels for Lx ({lx.shape[0]}) and Tx ({tx.shape[0]})"
            )
    else:
        tx = np.full((1, 2), np.nan)

    # -- alternate mode ----------------------------------------------------
    if signal_integral is None:
        ln_lx = float(np.sum(lx[:, 1]))
        tn_tx = float(np.sum(tx[:, 1])) if tx_data is not None else float("nan")
        table = _build_table(ln_lx, 0.0, tn_tx, 0.0, ln_lx, 0.0, tn_tx, 0.0, 0.0, 0.0, sig0, digits)
        return Results(
            originator="calc_osl_lxtx_ratio",
            data={"LxTx.table": table, "calc.parameters": None},
        )

    n_channels = lx.shape[0]
    sig = np.asarray(signal_integral, dtype=np.float64).ravel()
    bg = (
        None
        if background_integral is None
        else np.asarray(background_integral, dtype=np.float64).ravel()
    )
    sig_tx = (
        None
        if signal_integral_tx is None
        else np.asarray(signal_integral_tx, dtype=np.float64).ravel()
    )
    bg_tx = (
        None
        if background_integral_tx is None
        else np.asarray(background_integral_tx, dtype=np.float64).ravel()
    )

    if tx_data is not None and (sig_tx is None) != (bg_tx is None) and not use_previous_bg:
        raise ValueError(
            "'signal_integral_tx' and 'background_integral_tx' must be provided together"
        )

    if integral_input == "measurement":
        x_axis = lx[:, 0]  # the Lx time axis is used for all four integrals
        sig = to_channels(sig, x_axis, "signal_integral")
        if bg is not None:
            bg = to_channels(bg, x_axis, "background_integral")
        if sig_tx is not None:
            sig_tx = to_channels(sig_tx, x_axis, "signal_integral_tx")
        if bg_tx is not None:
            bg_tx = to_channels(bg_tx, x_axis, "background_integral_tx")

    # sig is non-None here (None selected alternate mode above), so the
    # validator always returns an array
    sig_v = cast(
        "npt.NDArray[np.int64]",
        validate_integral(sig, "signal_integral", 1, n_channels),
    )
    bg_v = validate_integral(bg, "background_integral", int(sig_v.max()) + 1, n_channels)

    if use_previous_bg and (sig_tx is not None or bg_tx is not None):
        warnings.warn(
            "For 'use_previous_bg = True' independent Lx and Tx integral limits are"
            " not allowed, Tx limits reset to Lx limits",
            LuminescenceWarning,
            stacklevel=2,
        )
        sig_tx = bg_tx = None
    sig_tx_v = (
        sig_v
        if sig_tx is None
        else cast(
            "npt.NDArray[np.int64]",
            validate_integral(sig_tx, "signal_integral_tx", 1, n_channels),
        )
    )
    if bg_tx is None:
        bg_tx_v = bg_v
    else:
        bg_tx_v = validate_integral(
            bg_tx, "background_integral_tx", int(sig_tx_v.max()) + 1, n_channels
        )

    # -- signals and backgrounds ------------------------------------------
    n = sig_v.size
    m = 0 if bg_v is None else bg_v.size
    k = m / n
    n_tx = sig_tx_v.size
    m_tx = m if use_previous_bg else (0 if bg_tx_v is None else bg_tx_v.size)
    k_tx = m_tx / n_tx

    lx_counts, tx_counts = lx[:, 1], tx[:, 1]
    y0 = float(np.sum(lx_counts[sig_v - 1]))
    y1 = 0.0 if bg_v is None else float(np.sum(lx_counts[bg_v - 1]))
    lx_background = 0.0 if bg_v is None else y1 / k
    net_lnlx = y0 - lx_background

    if tx_data is not None:
        y0_tx = float(np.sum(tx_counts[sig_tx_v - 1]))
        if use_previous_bg:
            y1_tx = y1
            tx_background = lx_background
        elif bg_tx_v is None:
            y1_tx = 0.0
            tx_background = 0.0
        else:
            y1_tx = float(np.sum(tx_counts[bg_tx_v - 1]))
            tx_background = y1_tx / k_tx
        net_tntx = y0_tx - tx_background
    else:
        y0_tx = y1_tx = tx_background = net_tntx = float("nan")

    # -- overdispersion ----------------------------------------------------
    if sigmab is not None:
        pair = np.atleast_1d(np.asarray(sigmab, dtype=np.float64))
        sigmab_lx, sigmab_tx = float(pair[0]), float(pair[-1])
    else:
        sigmab_lx = _sigmab(lx_counts, sig_v, bg_v, k, "Lx")
        sigmab_tx = (
            _sigmab(tx_counts, sig_tx_v, bg_tx_v, k_tx, "Tx")
            if tx_data is not None
            else float("nan")
        )

    # -- errors (Galbraith) -------------------------------------------------
    poisson = background_count_distribution == "poisson"
    used_sig_lx = 0.0 if (poisson or bg_v is None) else sigmab_lx
    used_sig_tx = 0.0 if (poisson or bg_tx_v is None) else sigmab_tx

    with np.errstate(divide="ignore", invalid="ignore"):
        if bg_v is None:
            net_lnlx_error = abs(net_lnlx * float(np.sqrt(y0) / y0))
        else:
            net_lnlx_error = abs(net_lnlx * _rse(y0, y1, k, used_sig_lx))
        if tx_data is None:
            net_tntx_error = float("nan")
        elif bg_tx_v is None:
            net_tntx_error = abs(net_tntx * float(np.sqrt(y0_tx) / y0_tx))
        else:
            # quirk kept from R: the Tx error uses k, not k_tx
            net_tntx_error = abs(net_tntx * _rse(y0_tx, y1_tx, k, used_sig_tx))

    if np.isnan(net_lnlx_error):
        net_lnlx_error = 0.0
    if np.isnan(net_tntx_error):
        net_tntx_error = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        sn_lx = float("nan") if bg_v is None else float(np.divide(y0, lx_background))
        sn_tx = (
            float("nan")
            if (bg_tx_v is None or tx_data is None)
            else float(np.divide(y0_tx, tx_background))
        )

    table = _build_table(
        y0,
        lx_background,
        y0_tx,
        tx_background,
        net_lnlx,
        net_lnlx_error,
        net_tntx,
        net_tntx_error,
        sn_lx,
        sn_tx,
        sig0,
        digits,
    )
    parameters: dict[str, Any] = {
        "sigmab.LnLx": sigmab_lx,
        "sigmab.TnTx": sigmab_tx,
        "k": k,
    }
    return Results(
        originator="calc_osl_lxtx_ratio",
        data={"LxTx.table": table, "calc.parameters": parameters},
    )


def _build_table(
    lnlx: float,
    lnlx_bg: float,
    tntx: float,
    tntx_bg: float,
    net_lnlx: float,
    net_lnlx_error: float,
    net_tntx: float,
    net_tntx_error: float,
    sn_lx: float,
    sn_tx: float,
    sig0: float,
    digits: int | None,
) -> dict[str, float]:
    if np.isnan(net_tntx):
        # missing Tx curve: the ratio is missing, not zero (R keeps NA here;
        # the NaN->0 rule below is only for the 0/0 case)
        lxtx = float("nan")
        lxtx_error = float("nan")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            lxtx = float(np.divide(net_lnlx, net_tntx))
        if np.isnan(lxtx):
            lxtx = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.sqrt(
                np.divide(net_lnlx_error, net_lnlx) ** 2 + np.divide(net_tntx_error, net_tntx) ** 2
            )
        lxtx_error = abs(lxtx * float(rel))
        if np.isnan(lxtx_error):
            lxtx_error = 0.0
        lxtx_error = float(np.sqrt(lxtx_error**2 + (sig0 * lxtx) ** 2))

    table = {
        "LnLx": lnlx,
        "LnLx.BG": lnlx_bg,
        "TnTx": tntx,
        "TnTx.BG": tntx_bg,
        "Net_LnLx": net_lnlx,
        "Net_LnLx.Error": net_lnlx_error,
        "Net_TnTx": net_tntx,
        "Net_TnTx.Error": net_tntx_error,
        "SN_RATIO_LnLx": sn_lx,
        "SN_RATIO_TnTx": sn_tx,
        "LxTx": lxtx,
        "LxTx.Error": lxtx_error,
    }
    if digits is not None:
        table = {key: round(value, digits) for key, value in table.items()}
    return table
