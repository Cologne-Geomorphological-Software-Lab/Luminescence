"""Cross-language test oracle: read testthat snapshots as Python objects.

The R test suite stores value snapshots in ``tests/testthat/_snaps/*.md``
(testthat "json2" serialisation). These files are the reference oracle for
the Python port: :func:`load_snapshots` parses the markdown container and
:func:`simplify_r_json` collapses the type-annotated R serialisation into
plain Python dicts/lists/scalars for comparison with Python results.

The R helpers in ``tests/testthat/setup.R`` already strip non-reproducible
fields (``.uid``, ``.pid``, ``call``, fit objects) before snapshotting, so
whatever appears here is expected to be stable across languages up to
floating-point tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SNAPS_DIR = Path(__file__).resolve().parents[1] / "testthat" / "_snaps"

Snapshot = Any
"""Parsed JSON structure, or the raw text for non-JSON (print-output) snaps."""


def load_snapshots(name: str, snaps_dir: Path = SNAPS_DIR) -> dict[str, list[Snapshot]]:
    """Parse ``<snaps_dir>/<name>.md`` into ``{test_name: [snapshot, ...]}``.

    Each ``# heading`` starts a testthat ``test_that()`` section; within a
    section, consecutive snapshots are separated by ``---`` at column 0.
    Snapshot bodies are indented by four spaces. JSON bodies are parsed;
    anything else is returned as the dedented raw string.
    """
    path = snaps_dir / f"{name}.md"
    sections: dict[str, list[Snapshot]] = {}
    current: list[str] = []
    title = ""

    def close_block() -> None:
        text = "\n".join(current).strip("\n")
        current.clear()
        if not text.strip():
            return
        try:
            parsed: Snapshot = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        sections.setdefault(title, []).append(parsed)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            close_block()
            title = line[2:].strip()
        elif line.rstrip() == "---":
            close_block()
        elif line.startswith("    "):
            current.append(line[4:])
        elif not line.strip():
            if current:
                current.append("")
        else:  # pragma: no cover - unexpected content should fail loudly
            raise ValueError(f"Unexpected line in snapshot file {path}: {line!r}")
    close_block()
    return sections


def simplify_r_json(node: Snapshot) -> Any:
    """Collapse testthat's "json2" R serialisation into plain Python values.

    Mapping rules:

    - atomic vectors (``double``/``integer``/``character``/``logical``)
      become a scalar when length 1, otherwise a list (R ``NA`` -> ``None``);
    - named ``list`` (including ``data.frame``, which becomes a dict of
      columns) -> ``dict``; unnamed ``list`` -> ``list``;
    - ``S4`` objects -> dict of their slots;
    - ``NULL`` -> ``None``.

    Non-dict input (raw-text snapshots) is returned unchanged.
    """
    if not isinstance(node, dict):
        return node
    rtype = node.get("type")
    if not isinstance(rtype, str):
        return node
    attributes: dict[str, Any] = node.get("attributes") or {}
    value = node.get("value")

    if rtype == "NULL":
        return None
    if rtype == "S4":
        return {name: simplify_r_json(slot) for name, slot in attributes.items()}
    if rtype == "list":
        items = [simplify_r_json(item) for item in (value or [])]
        names = _names(attributes)
        if names is not None:
            return dict(zip(names, items, strict=True))
        return items
    if rtype in {"double", "integer", "character", "logical", "complex"}:
        items = list(value or [])
        names = _names(attributes)
        if names is not None:
            return dict(zip(names, items, strict=True))
        if len(items) == 1:
            return items[0]
        return items
    if rtype == "environment":
        return "<environment>"
    return value


def _names(attributes: dict[str, Any]) -> list[str] | None:
    names_node = attributes.get("names")
    if names_node is None:
        return None
    raw = names_node.get("value") or []
    return [str(name) for name in raw]
