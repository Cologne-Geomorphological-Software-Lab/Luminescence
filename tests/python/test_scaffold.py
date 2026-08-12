"""Phase-0 scaffold tests: package imports and the snapshot oracle loader."""

from __future__ import annotations

import importlib

import pytest

import luminescence
from oracle import SNAPS_DIR, load_snapshots, simplify_r_json

SUBPACKAGES = [
    "core",
    "io",
    "models",
    "dosimetry",
    "analysis",
    "fitting",
    "plot",
    "bayes",
    "data",
    "utils",
]


def test_version() -> None:
    assert luminescence.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name: str) -> None:
    importlib.import_module(f"luminescence.{name}")


class TestSnapshotOracle:
    """The testthat _snaps corpus must be readable as our reference oracle."""

    def test_snaps_dir_exists(self) -> None:
        assert SNAPS_DIR.is_dir()
        assert list(SNAPS_DIR.glob("*.md"))

    def test_json_snapshot_roundtrip(self) -> None:
        snaps = load_snapshots("calc_Statistics")
        stats = simplify_r_json(snaps["snapshot tests"][0])
        assert set(stats) == {"weighted", "unweighted", "MCM"}
        assert stats["weighted"]["n"] == 25
        assert stats["weighted"]["mean"] == pytest.approx(2896.03575033)

    def test_s4_snapshot(self) -> None:
        snaps = load_snapshots("calc_CentralDose")
        results = simplify_r_json(snaps["snapshot tests"][0])
        summary = results["data"]["summary"]
        assert summary["de"] == pytest.approx(65.70928553)
        assert summary["OD"] == pytest.approx(22.79495348)

    def test_text_snapshots_and_separator(self) -> None:
        snaps = load_snapshots("RLum.Results-class")
        blocks = snaps["snapshot tests"]
        assert len(blocks) == 2
        assert all(isinstance(block, str) for block in blocks)
        assert "[RLum.Results-class]" in blocks[0]

    def test_all_snapshot_files_parse(self) -> None:
        for path in SNAPS_DIR.glob("*.md"):
            sections = load_snapshots(path.stem)
            assert sections, f"no snapshots parsed from {path.name}"
            for blocks in sections.values():
                for block in blocks:
                    simplify_r_json(block)
