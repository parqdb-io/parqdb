from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.tools.resources import (
    ResourceMonitor,
    affinity_cpu_ids,
    cgroup_environment,
    cgroup_snapshot,
    current_rss_bytes,
    effective_cpu_count,
)


def test_resource_monitor_records_process_usage() -> None:
    with ResourceMonitor(sample_interval_seconds=0.001) as monitor:
        values = bytearray(1024 * 1024)
        values[0] = 1

    result = monitor.result()
    assert result.wall_seconds > 0
    assert result.cpu_seconds >= 0
    assert result.minor_page_faults >= 0
    assert result.major_page_faults >= 0
    assert result.peak_rss_bytes is None or result.peak_rss_bytes > 0
    assert (
        result.cgroup_peak_memory_current_bytes is None
        or result.cgroup_peak_memory_current_bytes > 0
    )


def test_effective_cpu_count_matches_affinity() -> None:
    assert effective_cpu_count() == len(affinity_cpu_ids())
    assert effective_cpu_count() > 0
    assert current_rss_bytes() is None or current_rss_bytes() > 0


def test_cgroup_metadata_is_internally_consistent() -> None:
    environment = cgroup_environment()
    snapshot = cgroup_snapshot()

    if environment["path"] is not None:
        assert str(environment["path"]).startswith("/sys/fs/cgroup")
    if snapshot.memory_current_bytes is not None:
        assert snapshot.memory_current_bytes > 0
    if environment["memory_max_bytes"] is not None:
        assert int(environment["memory_max_bytes"]) > 0
