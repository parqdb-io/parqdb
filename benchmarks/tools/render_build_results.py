"""Render the persisted-index build chart."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

WIDTH = 1280
HEIGHT = 430
IMPLEMENTATIONS = ("relify", "faiss")
COLORS = {"relify": "#0f766e", "faiss": "#f59e0b"}
LABELS = {"relify": "Relify", "faiss": "Faiss"}


def implementation_label(implementation: str, result: dict[str, Any]) -> str:
    encoding = result.get("encoding")
    if not encoding:
        return LABELS[implementation]
    return f"{LABELS[implementation]} ({str(encoding).upper()})"


def results_by_implementation(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        result["implementation"]: result
        for result in run["results"]
        if result["implementation"] in IMPLEMENTATIONS
    }


def validate(run: dict[str, Any]) -> None:
    if run.get("schema_version") != 1:
        raise ValueError("benchmark result schema_version must be 1")
    results = results_by_implementation(run)
    if set(results) != set(IMPLEMENTATIONS):
        raise ValueError("build chart requires Relify and Faiss results")
    if any(result["build_seconds"] <= 0 for result in results.values()):
        raise ValueError("build time must be positive")


def environment_label(run: dict[str, Any]) -> str:
    if "resources" in run and "software" in run:
        return f"{run['software']['machine']} · {run['resources']['cpus']} vCPUs"
    system = run["system"]
    return f"{system['machine']} · {system['logical_cpus']} logical CPUs"


def nice_axis_max(value: float) -> float:
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1, 2, 5, 10):
        if normalized <= candidate:
            return candidate * magnitude
    raise AssertionError("unreachable")


def render(run: dict[str, Any]) -> str:
    validate(run)
    results = results_by_implementation(run)
    dataset = run["dataset"]
    parameters = run["parameters"]
    maximum_seconds = nice_axis_max(
        max(result["build_seconds"] for result in results.values())
    )
    plot_left = 224
    plot_right = 1190
    plot_width = plot_right - plot_left
    row_y = {"relify": 190, "faiss": 280}
    bar_height = 46
    excludes_preparation = all(
        "preparation_seconds" in result for result in results.values()
    )
    legacy_flat = all("encoding" not in result for result in results.values())
    title = (
        "Persisted IVF-Flat Build Time" if legacy_flat else "Persisted IVF Build Time"
    )
    if excludes_preparation:
        description = (
            f"Build time from centroid training through persisted {title.removeprefix('Persisted ').removesuffix(' Build Time')} indexes. "
            "Lower is better."
        )
        boundary = (
            "Timers exclude input loading and training-sample preparation, and stop "
            "after index persistence."
        )
    else:
        description = (
            "Embedded build time from one shared Parquet source through persisted "
            f"{'IVF-Flat' if legacy_flat else 'IVF'} indexes. Lower is better."
        )
        boundary = (
            "Timers start from the shared Parquet source and stop after index "
            "persistence."
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-labelledby="title description">',
        f'<title id="title">{title}</title>',
        f'<desc id="description">{description}</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="24" fill="#f8fafc"/>',
        '<text x="58" y="58" font-size="30" font-weight="700" fill="#0f172a">'
        f"{title}</text>",
        '<text x="58" y="88" font-size="15" fill="#475569">'
        f"Same uncompressed Parquet source · nlist={parameters['nlist']:,} · "
        "lower is better</text>",
    ]

    for tick in range(6):
        seconds = maximum_seconds * tick / 5
        x = plot_left + plot_width * tick / 5
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="132" x2="{x:.1f}" y2="320" stroke="#e2e8f0"/>',
                f'<text x="{x:.1f}" y="342" text-anchor="middle" '
                f'font-size="11" fill="#64748b">{seconds:g}s</text>',
            ]
        )

    for implementation in IMPLEMENTATIONS:
        result = results[implementation]
        seconds = result["build_seconds"]
        y = row_y[implementation]
        width = plot_width * seconds / maximum_seconds
        elements.extend(
            [
                f'<text x="190" y="{y + 6}" text-anchor="end" font-size="17" '
                f'font-weight="700" fill="#334155">'
                f"{implementation_label(implementation, result)}</text>",
                f'<rect x="{plot_left}" y="{y - bar_height / 2:.1f}" '
                f'width="{width:.1f}" height="{bar_height}" rx="8" '
                f'fill="{COLORS[implementation]}"/>',
                f'<text x="{plot_left + width + 12:.1f}" y="{y + 6}" '
                f'font-size="14" font-weight="700" fill="#334155">{seconds:.2f}s</text>',
            ]
        )

    footer = (
        f"{dataset['rows']:,} vectors · d={dataset['dimension']} · "
        f"nlist={parameters['nlist']:,} · {environment_label(run)}"
    )
    elements.extend(
        [
            f'<text x="58" y="385" font-size="12" fill="#475569">{boundary}</text>',
            f'<text x="58" y="409" font-size="11" fill="#64748b">'
            f"{html.escape(footer)}</text>",
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("result", type=Path)
    command.add_argument("--output", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    run = json.loads(args.result.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(run), encoding="utf-8")


if __name__ == "__main__":
    main()
