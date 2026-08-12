"""Dose-response (growth) curve plot.

Idiomatic matplotlib re-design of the R ``plot_GrowthCurve``/
``plot_DoseResponseCurve`` output: measured points with error bars, the
fitted model, and the interpolated De.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from luminescence.core.results import Results

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["plot_growth_curve"]


def plot_growth_curve(
    fit_results: Results,
    data: Any,
    *,
    ax: Axes | None = None,
    xlabel: str = "Dose [s]",
    ylabel: str = "$L_x/T_x$",
) -> Axes:
    """Plot a fitted dose-response curve.

    Args:
        fit_results: Return value of
            :func:`~luminescence.fitting.dose_response.fit_dose_response_curve`.
        data: The table that was fitted (Dose, LxTx, LxTx.Error; row 0 =
            natural signal).
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The matplotlib Axes (no ``plt.show()`` is called).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    arr = np.asarray(data, dtype=np.float64)
    dose, lxtx, lxtx_error = arr[:, 0], arr[:, 1], arr[:, 2]

    de_row = fit_results["De"]
    mode = de_row.get("Mode", "interpolation")
    fit_params = fit_results.get("Fit")

    reg = slice(1, None) if mode == "interpolation" else slice(None)
    ax.errorbar(
        dose[reg], lxtx[reg], yerr=lxtx_error[reg],
        fmt="o", capsize=3, zorder=3, label="Regeneration points",
    )  # fmt: skip

    if fit_params is not None:
        x_grid = np.linspace(0, float(np.max(dose)) * 1.1, 200)
        y_grid = _evaluate_model(de_row.get("Fit", ""), fit_params, x_grid)
        if y_grid is not None:
            ax.plot(x_grid, y_grid, zorder=2, label=f"Fit: {de_row.get('Fit', '')}")

    de = de_row.get("De", np.nan)
    if mode == "interpolation" and not np.isnan(de):
        natural = lxtx[0]
        ax.axhline(natural, linestyle=":", linewidth=0.8, color="grey")
        ax.axvline(de, linestyle=":", linewidth=0.8, color="grey")
        ax.plot([de], [natural], "s", zorder=4, label=f"De = {de:.4g}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return ax


def _evaluate_model(method: str, params: dict[str, float], x: np.ndarray) -> np.ndarray | None:
    if method == "SSE" and {"N", "D0", "Di"} <= params.keys():
        return params["N"] * (1 - np.exp(-(x + params["Di"]) / params["D0"]))
    if method == "GOK" and {"a", "D0", "c", "d"} <= params.keys():
        c = max(params["c"], 1e-10)
        return params["a"] * (params["d"] - (1 + (1 / params["D0"]) * x * c) ** (-1 / c))
    if method == "LIN" and {"intercept", "slope"} <= params.keys():
        return params["intercept"] + params["slope"] * x
    return None
