"""Query adapters shared by the resident and storage-backed benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.tools.harness import SearchFunction

if TYPE_CHECKING:
    import relify


def _datafusion_memory_size(size_bytes: int) -> str:
    kibibytes = (size_bytes + 1023) // 1024
    return f"{kibibytes}K"


def configure_relify_session(
    session: relify.Session,
    threads: int,
    *,
    max_temp_directory_size_bytes: int | None = None,
) -> None:
    session.datafusion_context().sql(
        f"SET datafusion.execution.target_partitions = '{threads}'"
    ).collect()
    if max_temp_directory_size_bytes is not None:
        session.datafusion_context().sql(
            "SET datafusion.runtime.max_temp_directory_size = "
            f"'{_datafusion_memory_size(max_temp_directory_size_bytes)}'"
        ).collect()


def relify_search(
    session: relify.Session,
    table: relify.SourceTable,
    *,
    id_column: str = "id",
    vector_column: str = "embedding",
) -> SearchFunction:
    def search(query: np.ndarray, nprobe: int, k: int) -> np.ndarray:
        request = (
            table.search(query, column=vector_column)
            .nprobes(nprobe)
            .limit(k)
            .select([id_column])
        )
        result = session.to_arrow(request)
        return result[id_column].to_numpy(zero_copy_only=False)

    return search


def faiss_search(index: Any) -> SearchFunction:
    def search(query: np.ndarray, nprobe: int, k: int) -> np.ndarray:
        index.nprobe = nprobe
        _, neighbors = index.search(query.reshape(1, -1), k)
        return neighbors[0]

    return search
