"""Merge independently executed benchmark results for comparison."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IMPLEMENTATION_ORDER = ("relify", "faiss")
PER_IMPLEMENTATION_PARAMETERS = {"implementation", "encoding", "index_root"}
COMMON_SOFTWARE = ("platform", "machine", "python", "numpy", "pyarrow", "rustc")


def _common_parameters(run: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in run["parameters"].items()
        if key not in PER_IMPLEMENTATION_PARAMETERS
    }


def merge(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("at least two benchmark results are required")
    first = runs[0]
    implementations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for run in runs:
        if run.get("schema_version") != 1 or len(run.get("results", [])) != 1:
            raise ValueError(
                "each input must be a single-implementation schema v1 result"
            )
        result = run["results"][0]
        implementation = run.get("parameters", {}).get("implementation")
        if implementation != result.get("implementation"):
            raise ValueError("result implementation does not match its parameters")
        if implementation in implementations:
            raise ValueError(f"duplicate implementation result: {implementation}")
        implementations[implementation] = (run, result)

        for field in ("benchmark", "benchmark_revision", "dataset", "resources"):
            if run.get(field) != first.get(field):
                raise ValueError(f"benchmark inputs differ in {field}")
        if _common_parameters(run) != _common_parameters(first):
            raise ValueError("benchmark inputs use different common parameters")
        for field in COMMON_SOFTWARE:
            if run.get("software", {}).get(field) != first.get("software", {}).get(
                field
            ):
                raise ValueError(f"benchmark environments differ in software.{field}")

    ordered = sorted(
        implementations,
        key=lambda name: (
            IMPLEMENTATION_ORDER.index(name)
            if name in IMPLEMENTATION_ORDER
            else len(IMPLEMENTATION_ORDER),
            name,
        ),
    )
    merged = copy.deepcopy(first)
    merged["generated_at_utc"] = datetime.now(UTC).isoformat()
    merged["parameters"] = _common_parameters(first)
    merged["parameters"]["encodings"] = {
        name: implementations[name][0]["parameters"]["encoding"] for name in ordered
    }
    merged["parameters"]["index_roots"] = {
        name: implementations[name][0]["parameters"]["index_root"] for name in ordered
    }
    merged["implementation_revisions"] = {
        name: implementations[name][0].get("implementation_revision")
        for name in ordered
    }
    merged["software"] = {
        field: first["software"].get(field) for field in COMMON_SOFTWARE
    }
    merged["software"]["implementations"] = {
        name: implementations[name][0]["software"] for name in ordered
    }
    merged["results"] = [implementations[name][1] for name in ordered]
    merged.pop("implementation_revision", None)
    return merged


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="python -m benchmarks.tools.merge_results",
        description="Merge comparable single-implementation benchmark results.",
    )
    command.add_argument("results", type=Path, nargs="+")
    command.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    result = merge(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
