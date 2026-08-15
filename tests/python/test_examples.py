from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("module", "arguments", "expected_output"),
    [
        (
            "examples.python.local.parquet_roundtrip",
            (),
            "Parquet round trip:",
        ),
        ("examples.python.local.quickstart", (), "quickstart hits:"),
        ("examples.python.local.exact_search", (), "exact-search hits:"),
        ("examples.python.local.query_plans", (), "query runtime metrics:"),
        (
            "examples.python.local.datafusion_analysis",
            (),
            "DataFusion analysis:",
        ),
        (
            "examples.python.local.index_lifecycle",
            (),
            "index lifecycle:",
        ),
        (
            "examples.python.spark.query",
            ("--help",),
            "Query a published Relify index",
        ),
        (
            "examples.python.starrocks.query",
            ("--help",),
            "Query a Relify Iceberg index",
        ),
    ],
)
def test_python_example_runs(
    module: str,
    arguments: tuple[str, ...],
    expected_output: str,
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert expected_output in result.stdout
