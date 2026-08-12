"""Comprehensive luminescence dating data analysis.

Python port of the R package 'Luminescence'. The public API is re-exported
flat from this module so R users can translate calls one-to-one, e.g.
``Luminescence::analyse_SAR.CWOSL(...)`` becomes
``luminescence.analyse_sar_cwosl(...)``.
"""

__version__ = "0.1.0.dev0"

from luminescence.analysis.lxtx import calc_osl_lxtx_ratio
from luminescence.analysis.sar_cwosl import analyse_sar_cwosl
from luminescence.core import (
    Analysis,
    Curve,
    ImageData,
    Record,
    Results,
    RisoeBINFileData,
    Spectrum,
)
from luminescence.fitting.dose_response import fit_dose_response_curve
from luminescence.io import read_bin
from luminescence.models.statistics import calc_statistics
from luminescence.plot.growth_curve import plot_growth_curve

__all__ = [
    "Analysis",
    "Curve",
    "ImageData",
    "Record",
    "Results",
    "RisoeBINFileData",
    "Spectrum",
    "__version__",
    "analyse_sar_cwosl",
    "calc_osl_lxtx_ratio",
    "calc_statistics",
    "fit_dose_response_curve",
    "plot_growth_curve",
    "read_bin",
]
