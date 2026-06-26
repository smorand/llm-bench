"""Ad-hoc DuckDB analysis over a run's data file (SC-006, FR-052).

This module backs the ``llm-bench analyze <path> --sql "<query>"`` command. It
registers the data file (``.jsonl`` or ``.parquet``) as a DuckDB view named
``data`` and runs the operator's own SQL against it, then renders the result
rows to stdout as compact ``key=value`` lines.

DuckDB reads the file in place (no ETL step): JSONL via ``read_json_auto`` and
Parquet via ``read_parquet``. A 0-record JSONL still resolves the ``data`` view,
so ``SELECT count(*)`` yields 0 rather than raising.

Errors are surfaced as :class:`AnalyzeError` with a clean one-line message (a
missing file, or a DuckDB parser/binder error); the caller maps that to a
non-zero exit without leaking a Python traceback (EXC-006a/EXC-006b).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

_PARQUET_SUFFIX = ".parquet"


class AnalyzeError(Exception):
    """A user-facing analyze failure (missing file or invalid SQL)."""


def _reader_expr(path: Path) -> str:
    """Return the DuckDB table-function call that reads ``path`` in place.

    Parquet files use ``read_parquet``; everything else (``.jsonl`` and friends)
    uses ``read_json_auto``. The path is single-quoted with embedded quotes
    doubled so an operator path containing a quote does not break the SQL.
    """
    quoted = str(path).replace("'", "''")
    if path.suffix.lower() == _PARQUET_SUFFIX:
        return f"read_parquet('{quoted}')"
    return f"read_json_auto('{quoted}')"


def _render(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render result rows as ``col=value`` lines, one ``col=value`` per cell.

    Each row becomes a single line of space-joined ``col=value`` pairs so every
    cell value is surfaced in stdout in a readable form.
    """
    if not rows:
        return "(0 rows)"
    lines: list[str] = []
    for row in rows:
        pairs = [f"{col}={value}" for col, value in zip(columns, row, strict=True)]
        lines.append(" ".join(pairs))
    return "\n".join(lines)


def run_query(data: Path, sql: str) -> str:
    """Run ``sql`` against ``data`` (registered as view ``data``) and render rows.

    Raises :class:`AnalyzeError` if the file is absent (with the exact message
    ``data file not found: <path>``) or if DuckDB rejects the SQL (with the
    DuckDB parser/binder message). On success returns the rendered result.
    """
    if not data.exists():
        raise AnalyzeError(f"data file not found: {data}")

    connection = duckdb.connect()
    try:
        # nosec B608: the only interpolation is the operator's own local file path
        # (single-quotes doubled in _reader_expr); not external untrusted input.
        connection.execute(f"CREATE VIEW data AS SELECT * FROM {_reader_expr(data)}")  # nosec B608
        cursor = connection.execute(sql)
        description = cursor.description or []
        columns = [str(col[0]) for col in description]
        rows = cursor.fetchall()
    except duckdb.Error as exc:
        logger.debug("analyze query failed", extra={"event": "analyze_error"})
        raise AnalyzeError(str(exc).strip()) from exc
    finally:
        connection.close()
    return _render(columns, rows)
