# Working with the R package

This package is a port of the R package Luminescence. If you know the R API, the Python
API should feel familiar.

## Naming scheme

Function names keep their R verbs and switch to snake_case:

| R | Python |
|---|---|
| `read_BIN2R()` | `read_bin()` |
| `Risoe.BINfileData2RLum.Analysis()` | `RisoeBINFileData.to_analysis()` |
| `analyse_SAR.CWOSL()` | `analyse_sar_cwosl()` |
| `calc_OSLLxTxRatio()` | `calc_osl_lxtx_ratio()` |
| `calc_Statistics()` | `calc_statistics()` |
| `fit_DoseResponseCurve()` | `fit_dose_response_curve()` |
| `plot_GrowthCurve()` | `plot_growth_curve()` |

Argument names follow the same rule (`signal.integral.min`/`signal_integral`,
`fit.method`/`fit_method`, `n.MC`/`n_mc`).

## Classes

The S4 classes map to plain Python classes:

| R (S4) | Python |
|---|---|
| `RLum.Data.Curve` | `Curve` |
| `RLum.Data.Spectrum` | `Spectrum` |
| `RLum.Data.Image` | `ImageData` |
| `RLum.Analysis` | `Analysis` |
| `RLum.Results` | `Results` |
| `Risoe.BINfileData` | `RisoeBINFileData` |

The `set_RLum`/`get_RLum` generics become constructors and methods: records of an
`Analysis` are selected with `analysis.get_records("OSL")` or indexing, `Results`
behaves as a read-only mapping (`results["data"]`), and every class offers
`to_dataframe()` where a tabular view makes sense.

## Behavioural differences

- Stochastic functions (Monte-Carlo error estimation, resampling) take an explicit
  `rng` argument (a seed or a `numpy.random.Generator`). Results are reproducible per
  seed, but random sequences differ from R: Monte-Carlo error estimates agree
  statistically, not digit for digit.
- R's `NULL`/`NA` distinction maps to `None` for "not given" and `NaN` for missing
  numeric values.
- Errors are raised as exceptions (`ValueError`, `luminescence.utils.exceptions.*`)
  instead of R conditions; R warnings become Python warnings of type
  `LuminescenceWarning`.
- Plot functions build on matplotlib and return an `Axes` object instead of drawing to
  the active R graphics device.
