"""Render query recall-latency curves."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

WIDTH = 1280
HEIGHT = 720
COLORS = {"relify": "#0f766e", "faiss": "#f59e0b"}
LABELS = {"relify": "Relify", "faiss": "Faiss"}


def implementation_label(implementation: str, result: dict[str, Any]) -> str:
    encoding = result.get("encoding")
    if not encoding:
        return LABELS[implementation]
    return f"{LABELS[implementation]} ({str(encoding).upper()})"


def parse_k_values(encoded: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in encoded.split(","))
    except ValueError as error:
        raise ValueError("k-values must be comma-separated integers") from error
    if not 1 <= len(values) <= 3 or any(value <= 0 for value in values):
        raise ValueError("k-values must contain one to three positive values")
    if values != tuple(sorted(set(values))):
        raise ValueError("k-values must be unique and ascending")
    return values


def results_by_implementation(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        result["implementation"]: result
        for result in run["results"]
        if result["implementation"] in COLORS
    }


def environment_label(run: dict[str, Any]) -> str:
    if "resources" in run and "software" in run:
        return f"{run['software']['machine']} · {run['resources']['cpus']} vCPUs"
    system = run["system"]
    return f"{system['machine']} · {system['logical_cpus']} logical CPUs"


def points_for_k(result: dict[str, Any], k: int) -> list[dict[str, Any]]:
    return sorted(
        (point for point in result["search_curve"] if point["k"] == k),
        key=lambda point: point["nprobe"],
    )


def validate(run: dict[str, Any], k_values: tuple[int, ...]) -> None:
    if run.get("schema_version") != 1:
        raise ValueError("benchmark result schema_version must be 1")
    implementations = results_by_implementation(run)
    if not {"relify", "faiss"} <= set(implementations):
        raise ValueError("search chart requires Relify and Faiss results")
    if not set(implementations) <= set(COLORS):
        raise ValueError("search chart contains an unsupported implementation")
    expected_nprobes = None
    for implementation, result in implementations.items():
        for k in k_values:
            points = points_for_k(result, k)
            if not points:
                raise ValueError(f"{implementation} has no search points for k={k}")
            nprobes = [point["nprobe"] for point in points]
            if expected_nprobes is None:
                expected_nprobes = nprobes
            elif nprobes != expected_nprobes:
                raise ValueError(
                    "implementations and k values must use the same nprobes"
                )
            for point in points:
                if not 0 <= point["recall_at_k"] <= 1:
                    raise ValueError("recall_at_k must be in [0, 1]")
                if point["latency_ms_p50"] <= 0 or point["latency_ms_p95"] <= 0:
                    raise ValueError("search latency must be positive")


def compact_k(value: int) -> str:
    if value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:g}K"
    return str(value)


def render(run: dict[str, Any], k_values: tuple[int, ...]) -> str:
    validate(run, k_values)
    implementations = results_by_implementation(run)
    all_points = [
        point
        for result in implementations.values()
        for k in k_values
        for point in points_for_k(result, k)
    ]
    minimum_latency = min(point["latency_ms_p50"] for point in all_points)
    maximum_latency = max(point["latency_ms_p50"] for point in all_points)
    lower_log = math.floor(math.log10(minimum_latency))
    upper_log = math.ceil(math.log10(maximum_latency))
    if lower_log == upper_log:
        upper_log += 1

    margin_x = 58
    gap = 24
    panel_width = (WIDTH - margin_x * 2 - gap * (len(k_values) - 1)) / len(k_values)
    panel_top = 132
    panel_bottom = 540
    plot_top = 208
    plot_bottom = 474

    def latency_x(value: float, panel_x: float) -> float:
        fraction = (math.log10(value) - lower_log) / (upper_log - lower_log)
        return panel_x + 58 + fraction * (panel_width - 82)

    def recall_y(value: float) -> float:
        return plot_bottom - value * (plot_bottom - plot_top)

    if "resources" in run:
        description_mode = "One-query-at-a-time"
        subtitle_mode = "One query at a time"
    else:
        description_mode = "Resident-index one-query-at-a-time"
        subtitle_mode = "Resident-index · one query at a time"

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Large-k IVF Recall-Latency</title>',
        (
            f'<desc id="description">{description_mode} p50 latency '
            "with intra-query parallelism versus exact Recall at increasing "
            "nprobe values.</desc>"
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="24" fill="#f8fafc"/>',
        '<text x="58" y="56" font-size="30" font-weight="700" fill="#0f172a">'
        "Large-k IVF Recall-Latency</text>",
        '<text x="58" y="86" font-size="15" fill="#475569">'
        f"{subtitle_mode} · intra-query parallel · p50 latency"
        "</text>",
    ]

    for panel_index, k in enumerate(k_values):
        panel_x = margin_x + panel_index * (panel_width + gap)
        elements.extend(
            [
                f'<rect x="{panel_x:.1f}" y="{panel_top}" width="{panel_width:.1f}" '
                f'height="{panel_bottom - panel_top}" rx="16" fill="#ffffff" '
                'stroke="#e2e8f0"/>',
                f'<text x="{panel_x + 22:.1f}" y="168" font-size="19" '
                f'font-weight="700" fill="#0f172a">Recall@{compact_k(k)}</text>',
                f'<text x="{panel_x + 22:.1f}" y="190" font-size="12" '
                'fill="#64748b">increasing nprobe →</text>',
            ]
        )
        for recall in (0, 0.25, 0.5, 0.75, 1):
            y = recall_y(recall)
            elements.extend(
                [
                    f'<line x1="{panel_x + 58:.1f}" y1="{y:.1f}" '
                    f'x2="{panel_x + panel_width - 24:.1f}" y2="{y:.1f}" '
                    'stroke="#e2e8f0"/>',
                    f'<text x="{panel_x + 48:.1f}" y="{y + 4:.1f}" '
                    f'text-anchor="end" font-size="10" fill="#64748b">{recall:g}</text>',
                ]
            )
        for power in range(lower_log, upper_log + 1):
            latency = 10**power
            x = latency_x(latency, panel_x)
            elements.extend(
                [
                    f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                    f'y2="{plot_bottom}" stroke="#f1f5f9"/>',
                    f'<text x="{x:.1f}" y="496" text-anchor="middle" '
                    f'font-size="10" fill="#64748b">{latency:g}</text>',
                ]
            )
        elements.append(
            f'<text x="{panel_x + panel_width / 2:.1f}" y="520" '
            'text-anchor="middle" font-size="11" fill="#475569">'
            "p50 latency (ms, log scale)</text>"
        )

        for implementation, result in implementations.items():
            points = points_for_k(result, k)
            coordinates = [
                (
                    latency_x(point["latency_ms_p50"], panel_x),
                    recall_y(point["recall_at_k"]),
                    point["nprobe"],
                )
                for point in points
            ]
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coordinates)
            color = COLORS[implementation]
            elements.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" '
                'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            label_nprobes = {
                coordinates[0][2],
                coordinates[len(coordinates) // 2][2],
                coordinates[-1][2],
            }
            for x, y, nprobe in coordinates:
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                    f'fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
                )
                if nprobe in label_nprobes:
                    elements.append(
                        f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="9" '
                        f'fill="{color}">{nprobe}</text>'
                    )

    legend_x = 58
    legend_step = 130 if len(implementations) == 2 else 150
    for implementation in implementations:
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="584" x2="{legend_x + 24}" y2="584" '
                f'stroke="{COLORS[implementation]}" stroke-width="3"/>',
                f'<circle cx="{legend_x + 12}" cy="584" r="4" '
                f'fill="{COLORS[implementation]}"/>',
                f'<text x="{legend_x + 34}" y="589" font-size="13" '
                f'fill="#334155">'
                f"{html.escape(implementation_label(implementation, implementations[implementation]))}</text>",
            ]
        )
        legend_x += legend_step

    dataset = run["dataset"]
    parameters = run["parameters"]
    footer = (
        f"{dataset['rows']:,} vectors · d={dataset['dimension']} · "
        f"{dataset['queries']} queries x {parameters['search_repetitions']} repetitions · "
        f"nlist={parameters['nlist']} · {environment_label(run)}"
    )
    elements.extend(
        [
            f'<text x="58" y="632" font-size="12" fill="#475569">'
            f"{html.escape(footer)}</text>",
            '<text x="58" y="660" font-size="11" fill="#64748b">'
            "Point labels are nprobe; p95 latency and every per-query sample are "
            "available in the raw JSON.</text>",
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("result", type=Path)
    command.add_argument("--k-values", default="100,1000,10000")
    command.add_argument("--output", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    run = json.loads(args.result.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(run, parse_k_values(args.k_values)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
