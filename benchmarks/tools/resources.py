"""Process and cgroup resource accounting for benchmark runners."""

from __future__ import annotations

import os
import platform
import resource
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Self


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_seconds: float
    minor_page_faults: int
    major_page_faults: int
    read_bytes: int | None
    write_bytes: int | None
    rss_bytes: int | None
    cgroup: CgroupSnapshot


@dataclass(frozen=True)
class CgroupSnapshot:
    memory_current_bytes: int | None
    memory_peak_bytes: int | None
    memory_anon_bytes: int | None
    memory_file_bytes: int | None
    swap_current_bytes: int | None
    read_bytes: int | None
    write_bytes: int | None
    read_operations: int | None
    write_operations: int | None


@dataclass(frozen=True)
class ResourceMetrics:
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int | None
    minor_page_faults: int
    major_page_faults: int
    read_bytes: int | None
    write_bytes: int | None
    cgroup_peak_memory_current_bytes: int | None
    cgroup_memory_current_bytes: int | None
    cgroup_memory_peak_bytes: int | None
    cgroup_memory_anon_bytes: int | None
    cgroup_memory_file_bytes: int | None
    cgroup_swap_current_bytes: int | None
    cgroup_read_bytes: int | None
    cgroup_write_bytes: int | None
    cgroup_read_operations: int | None
    cgroup_write_operations: int | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def affinity_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        count = os.cpu_count() or 1
        return tuple(range(count))


def effective_cpu_count() -> int:
    return len(affinity_cpu_ids())


def current_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    return int(usage if platform.system() == "Darwin" else usage * 1024)


def current_cgroup_directory() -> Path | None:
    try:
        entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for entry in entries:
        hierarchy, _, relative = entry.partition("::")
        if hierarchy == "0" and relative.startswith("/"):
            return Path("/sys/fs/cgroup") / relative.removeprefix("/")
    return None


def _read_cgroup_integer(directory: Path | None, name: str) -> int | None:
    if directory is None:
        return None
    try:
        encoded = (directory / name).read_text(encoding="utf-8").strip()
        return None if encoded == "max" else int(encoded)
    except (OSError, ValueError):
        return None


def _read_cgroup_values(directory: Path | None, name: str) -> dict[str, int]:
    if directory is None:
        return {}
    try:
        lines = (directory / name).read_text(encoding="utf-8").splitlines()
        return {
            key: int(value) for key, value in (line.split(maxsplit=1) for line in lines)
        }
    except (OSError, ValueError):
        return {}


def _read_cgroup_io(directory: Path | None) -> dict[str, int] | None:
    if directory is None:
        return None
    try:
        lines = (directory / "io.stat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    totals = {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
    try:
        for line in lines:
            for encoded in line.split()[1:]:
                key, value = encoded.split("=", maxsplit=1)
                if key in totals:
                    totals[key] += int(value)
    except ValueError:
        return None
    return totals


def cgroup_snapshot() -> CgroupSnapshot:
    directory = current_cgroup_directory()
    memory = _read_cgroup_values(directory, "memory.stat")
    io = _read_cgroup_io(directory)
    return CgroupSnapshot(
        memory_current_bytes=_read_cgroup_integer(directory, "memory.current"),
        memory_peak_bytes=_read_cgroup_integer(directory, "memory.peak"),
        memory_anon_bytes=memory.get("anon"),
        memory_file_bytes=memory.get("file"),
        swap_current_bytes=_read_cgroup_integer(directory, "memory.swap.current"),
        read_bytes=io["rbytes"] if io is not None else None,
        write_bytes=io["wbytes"] if io is not None else None,
        read_operations=io["rios"] if io is not None else None,
        write_operations=io["wios"] if io is not None else None,
    )


def _effective_cgroup_limit(directory: Path | None, name: str) -> int | None:
    limits = []
    while directory is not None and directory != directory.parent:
        value = _read_cgroup_integer(directory, name)
        if value is not None:
            limits.append(value)
        if directory == Path("/sys/fs/cgroup"):
            break
        directory = directory.parent
    return min(limits) if limits else None


def _inherited_cgroup_text(directory: Path | None, name: str) -> str | None:
    while directory is not None and directory != directory.parent:
        try:
            encoded = (directory / name).read_text(encoding="utf-8").strip()
        except OSError:
            encoded = ""
        if encoded:
            return encoded
        if directory == Path("/sys/fs/cgroup"):
            break
        directory = directory.parent
    return None


def cgroup_environment() -> dict[str, str | int | None]:
    directory = current_cgroup_directory()
    return {
        "path": str(directory) if directory is not None else None,
        "memory_max_bytes": _effective_cgroup_limit(directory, "memory.max"),
        "memory_swap_max_bytes": _effective_cgroup_limit(directory, "memory.swap.max"),
        "cpuset_cpus": _inherited_cgroup_text(directory, "cpuset.cpus.effective"),
        "cpuset_mems": _inherited_cgroup_text(directory, "cpuset.mems.effective"),
    }


def _process_io() -> tuple[int | None, int | None]:
    io_path = Path("/proc/self/io")
    try:
        values = {
            key.rstrip(":"): int(value)
            for key, value in (
                line.split(maxsplit=1)
                for line in io_path.read_text(encoding="utf-8").splitlines()
            )
        }
    except (OSError, ValueError):
        return None, None
    return values.get("read_bytes"), values.get("write_bytes")


def resource_snapshot() -> ResourceSnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    read_bytes, write_bytes = _process_io()
    return ResourceSnapshot(
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        minor_page_faults=usage.ru_minflt,
        major_page_faults=usage.ru_majflt,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        rss_bytes=current_rss_bytes(),
        cgroup=cgroup_snapshot(),
    )


def _optional_delta(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, end - start)


class ResourceMonitor:
    def __init__(self, *, sample_interval_seconds: float = 0.02) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        self._interval = sample_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._start: ResourceSnapshot | None = None
        self._peak_rss_bytes: int | None = None
        self._cgroup_peak_memory_current_bytes: int | None = None
        self.metrics: ResourceMetrics | None = None

    def __enter__(self) -> Self:
        self._start = resource_snapshot()
        self._peak_rss_bytes = self._start.rss_bytes
        self._cgroup_peak_memory_current_bytes = self._start.cgroup.memory_current_bytes
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._record_memory()
        end = resource_snapshot()
        assert self._start is not None
        self.metrics = ResourceMetrics(
            wall_seconds=time.perf_counter() - self._started_at,
            cpu_seconds=max(0.0, end.cpu_seconds - self._start.cpu_seconds),
            peak_rss_bytes=self._peak_rss_bytes,
            minor_page_faults=max(
                0, end.minor_page_faults - self._start.minor_page_faults
            ),
            major_page_faults=max(
                0, end.major_page_faults - self._start.major_page_faults
            ),
            read_bytes=_optional_delta(self._start.read_bytes, end.read_bytes),
            write_bytes=_optional_delta(self._start.write_bytes, end.write_bytes),
            cgroup_peak_memory_current_bytes=(self._cgroup_peak_memory_current_bytes),
            cgroup_memory_current_bytes=end.cgroup.memory_current_bytes,
            cgroup_memory_peak_bytes=end.cgroup.memory_peak_bytes,
            cgroup_memory_anon_bytes=end.cgroup.memory_anon_bytes,
            cgroup_memory_file_bytes=end.cgroup.memory_file_bytes,
            cgroup_swap_current_bytes=end.cgroup.swap_current_bytes,
            cgroup_read_bytes=_optional_delta(
                self._start.cgroup.read_bytes, end.cgroup.read_bytes
            ),
            cgroup_write_bytes=_optional_delta(
                self._start.cgroup.write_bytes, end.cgroup.write_bytes
            ),
            cgroup_read_operations=_optional_delta(
                self._start.cgroup.read_operations,
                end.cgroup.read_operations,
            ),
            cgroup_write_operations=_optional_delta(
                self._start.cgroup.write_operations,
                end.cgroup.write_operations,
            ),
        )

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self._record_memory()

    def _record_memory(self) -> None:
        rss = current_rss_bytes()
        if rss is not None and (
            self._peak_rss_bytes is None or rss > self._peak_rss_bytes
        ):
            self._peak_rss_bytes = rss
        memory_current = cgroup_snapshot().memory_current_bytes
        if memory_current is not None and (
            self._cgroup_peak_memory_current_bytes is None
            or memory_current > self._cgroup_peak_memory_current_bytes
        ):
            self._cgroup_peak_memory_current_bytes = memory_current

    def result(self) -> ResourceMetrics:
        if self.metrics is None:
            raise RuntimeError("resource monitor has not completed")
        return self.metrics
