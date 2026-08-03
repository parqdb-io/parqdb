"""Shared measurement and host utilities for Relify benchmarks."""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import relify

from benchmarks.tools.datasets import load_float_matrix
from benchmarks.tools.resources import ResourceMonitor

KMEANS_MAX_ITERATIONS = 20
KMEANS_SEED = 42
FAISS_PARALLEL_MODE = 1
IMPLEMENTATIONS = ("relify", "faiss")
SearchFunction = Callable[[np.ndarray, int, int], np.ndarray]


class BuildProgressBar:
    def __init__(self, implementation: str, *, enabled: bool) -> None:
        self._implementation = implementation
        self._enabled = enabled
        self._interactive = enabled and sys.stderr.isatty()
        self._started = time.monotonic()
        self._last_rendered = 0.0
        self._last_phase: str | None = None
        self._rendered = False

    def update(self, status: relify.IndexStatus) -> None:
        if not self._enabled:
            return
        phase = status.phase or status.state
        fraction = status.progress or 0.0
        now = time.monotonic()
        if (
            self._interactive
            and phase == self._last_phase
            and fraction < 1.0
            and now - self._last_rendered < 0.5
        ):
            return
        elapsed = now - self._started
        counters = self._format_counters(phase, status.completed, status.total)
        if self._interactive:
            width = 32
            filled = min(width, int(fraction * width))
            bar = "#" * filled + "-" * (width - filled)
            line = (
                f"\r\033[2K{self._implementation:<8} [{bar}] "
                f"{fraction:6.2%}  {phase}{counters}  {elapsed:,.1f}s"
            )
            sys.stderr.write(line)
            sys.stderr.flush()
            self._last_rendered = now
            self._rendered = True
        elif phase != self._last_phase:
            print(
                f"{self._implementation}: {phase}{counters} ({fraction:.2%})",
                file=sys.stderr,
                flush=True,
            )
        self._last_phase = phase

    def close(self, *, success: bool) -> None:
        if not self._enabled:
            return
        elapsed = time.monotonic() - self._started
        if self._interactive and self._rendered:
            state = "complete" if success else "failed"
            sys.stderr.write(
                f"\r\033[2K{self._implementation:<8} {state} in {elapsed:,.1f}s\n"
            )
            sys.stderr.flush()

    @staticmethod
    def _format_counters(
        phase: str,
        completed: int | None,
        total: int | None,
    ) -> str:
        if completed is None or total in {None, 0}:
            return ""
        if phase == "training_centroids":
            rows_per_iteration = total // KMEANS_MAX_ITERATIONS
            if rows_per_iteration > 0:
                if completed >= total:
                    return f" iteration {KMEANS_MAX_ITERATIONS}/{KMEANS_MAX_ITERATIONS}"
                iteration = completed // rows_per_iteration + 1
                iteration_rows = completed % rows_per_iteration
                iteration_fraction = iteration_rows / rows_per_iteration
                return (
                    f" iteration {iteration}/{KMEANS_MAX_ITERATIONS}"
                    f" ({iteration_fraction:.1%})"
                )
        return f" {completed:,}/{total:,}"


class CounterProgressBar:
    def __init__(self, label: str, *, enabled: bool) -> None:
        self._label = label
        self._enabled = enabled
        self._interactive = enabled and sys.stderr.isatty()
        self._started = time.monotonic()
        self._last_rendered = 0.0
        self._last_detail: str | None = None
        self._rendered = False

    def update(self, completed: int, total: int, detail: str) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if self._interactive and completed < total and now - self._last_rendered < 0.5:
            return
        fraction = completed / total if total else 0.0
        elapsed = now - self._started
        if self._interactive:
            width = 32
            filled = min(width, int(fraction * width))
            bar = "#" * filled + "-" * (width - filled)
            sys.stderr.write(
                f"\r\033[2K{self._label:<14} [{bar}] {fraction:6.2%}  "
                f"{completed:,}/{total:,}  {detail}  {elapsed:,.1f}s"
            )
            sys.stderr.flush()
            self._last_rendered = now
            self._rendered = True
        elif detail != self._last_detail:
            print(
                f"{self._label}: {completed:,}/{total:,} {detail}",
                file=sys.stderr,
                flush=True,
            )
        self._last_detail = detail

    def close(self, *, success: bool) -> None:
        if not self._enabled or not self._interactive or not self._rendered:
            return
        state = "complete" if success else "failed"
        elapsed = time.monotonic() - self._started
        sys.stderr.write(f"\r\033[2K{self._label:<14} {state} in {elapsed:,.1f}s\n")
        sys.stderr.flush()


def load_vectors(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        vectors = np.load(path)
    elif path.suffix == ".fvecs":
        encoded = np.fromfile(path, dtype=np.int32)
        if encoded.size == 0:
            raise ValueError(f"empty fvecs file: {path}")
        dimension = int(encoded[0])
        row_width = dimension + 1
        if dimension <= 0 or encoded.size % row_width != 0:
            raise ValueError(f"invalid fvecs file: {path}")
        rows = encoded.reshape(-1, row_width)
        if not np.all(rows[:, 0] == dimension):
            raise ValueError(f"inconsistent fvecs dimensions: {path}")
        vectors = rows[:, 1:].view(np.float32)
    elif path.suffix in {".bin", ".fbin"}:
        vectors = load_float_matrix(path)
    else:
        raise ValueError("vector input must use .npy, .fvecs, .bin, or .fbin")
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(f"vectors must be a non-empty matrix: {path}")
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError(f"vectors must be finite: {path}")
    return np.ascontiguousarray(vectors)


def parse_positive_ints(encoded: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in encoded.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} values must be unique and ascending")
    return values


def recall_at_k(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise ValueError("actual and expected neighbors must have the same shape")
    matches = sum(
        len(set(actual_row.tolist()) & set(expected_row.tolist()))
        for actual_row, expected_row in zip(actual, expected, strict=True)
    )
    return matches / actual.size


def total_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        return None


def command_version(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_revision() -> str | None:
    override = os.environ.get("RELIFY_BENCHMARK_REVISION")
    if override:
        return override
    return command_version("git", "rev-parse", "HEAD")


def directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def sync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            sync_file(file)


def warm_tree(path: Path) -> int:
    resident_bytes = 0
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        with file.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                resident_bytes += len(chunk)
    return resident_bytes


def evict_file(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def evict_tree(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            evict_file(file)


def measure_search_curve(
    search: SearchFunction,
    queries: np.ndarray,
    expected: np.ndarray,
    *,
    k_values: tuple[int, ...],
    nprobe_values: tuple[int, ...],
    repetitions: int,
    warmup_queries: int,
    progress: CounterProgressBar | None = None,
) -> list[dict[str, Any]]:
    max_k = max(k_values)
    max_nprobe = max(nprobe_values)
    points = [(k, nprobe) for k in k_values for nprobe in nprobe_values]
    np.random.default_rng(KMEANS_SEED).shuffle(points)
    warmups = min(warmup_queries, len(queries))
    total_queries = warmups + len(points) * repetitions * len(queries)
    completed_queries = 0
    try:
        for query in queries[:warmups]:
            search(query, max_nprobe, max_k)
            completed_queries += 1
            if progress is not None:
                progress.update(completed_queries, total_queries, "warmup")

        curve = []
        for execution_order, (k, nprobe) in enumerate(points):
            latency_samples = []
            cpu_samples = []
            recall_samples = []
            with ResourceMonitor() as resource_monitor:
                for _ in range(repetitions):
                    actual = np.full((len(queries), k), -1, dtype=np.int64)
                    for row, query in enumerate(queries):
                        started = time.perf_counter_ns()
                        cpu_started = time.process_time_ns()
                        values = search(query, nprobe, k)
                        cpu_samples.append(
                            (time.process_time_ns() - cpu_started) / 1_000_000
                        )
                        latency_samples.append(
                            (time.perf_counter_ns() - started) / 1_000_000
                        )
                        count = min(len(values), k)
                        actual[row, :count] = values[:count]
                        completed_queries += 1
                        if progress is not None:
                            progress.update(
                                completed_queries,
                                total_queries,
                                f"k={k:,} nprobe={nprobe:,}",
                            )
                    recall_samples.append(recall_at_k(actual, expected[:, :k]))

            mean_latency = statistics.fmean(latency_samples)
            curve.append(
                {
                    "k": k,
                    "nprobe": nprobe,
                    "execution_order": execution_order,
                    "recall_at_k": statistics.median(recall_samples),
                    "recall_samples": recall_samples,
                    "latency_ms_p50": statistics.median(latency_samples),
                    "latency_ms_p95": float(np.percentile(latency_samples, 95)),
                    "latency_ms_p99": float(np.percentile(latency_samples, 99)),
                    "latency_ms_mean": mean_latency,
                    "queries_per_second": 1_000 / mean_latency,
                    "latency_ms_samples": latency_samples,
                    "cpu_ms_per_query": statistics.fmean(cpu_samples),
                    "resource_usage": resource_monitor.result().as_dict(),
                }
            )
    except BaseException:
        if progress is not None:
            progress.close(success=False)
        raise
    if progress is not None:
        progress.close(success=True)
    return curve
