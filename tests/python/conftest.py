"""Shared pytest fixtures for the Python port.

Tolerance policy (see the migration plan):

- deterministic arithmetic / linear algebra: ``rtol=1e-9``
- quadrature / iterative results: ``rtol=1e-7``
- optimizer parameters: ``rtol=1e-5``; derived De/ages: ``rtol=1e-6``

Looser tolerances in individual tests carry an inline justification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: R-generated reference values (written by tools/generate_fixtures.R)
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Example instrument files shipped with the R package
EXTDATA_DIR = REPO_ROOT / "inst" / "extdata"

#: Binary-format edge cases used by the R test suite
R_TEST_DATA_DIR = REPO_ROOT / "tests" / "testthat" / "_data"

RTOL_EXACT = 1e-9
RTOL_QUADRATURE = 1e-7
RTOL_DERIVED = 1e-6
RTOL_OPTIMIZER = 1e-5


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def extdata_dir() -> Path:
    return EXTDATA_DIR


@pytest.fixture(scope="session")
def r_test_data_dir() -> Path:
    return R_TEST_DATA_DIR
