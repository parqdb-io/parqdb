"""Query adapters shared by the resident and storage-backed benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import relify

from benchmarks.tools.harness import SearchFunction


def configure_relify_session(session: relify.Session, threads: int) -> None:
    session.sql(f"SET datafusion.execution.target_partitions = '{threads}'").collect()


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
