from __future__ import annotations

import os
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from tempfile import TemporaryDirectory

import parqdb
import pyarrow


def main() -> None:
    if find_spec("datafusion") is not None:
        raise RuntimeError(
            "wheel smoke must not rely on an external datafusion package"
        )

    for module in ("pyiceberg",):
        import_module(module)

    with TemporaryDirectory(prefix="parqdb-wheel-smoke-") as directory:
        session = parqdb.connect(os.path.join(directory, "parqdb-data"))
        session.register_parquet("documents", parqdb.datasets.uri("documents"))
        result = session.sql("SELECT document_id FROM documents LIMIT 1")
        if not isinstance(result, pyarrow.Table) or result.to_pydict() != {
            "document_id": [1]
        }:
            raise RuntimeError(f"unexpected installed-wheel SQL result: {result}")
        table = session.table("documents")
        table.create_index(
            "smoke",
            column="embedding",
            key=["document_id"],
            config=parqdb.IVF(nlist=3),
        )
        table.wait_for_index("smoke")
        query = (
            table.search([0.2, 0.0], column="embedding")
            .where("tenant_id = 42 AND status = 'published'")
            .nprobes(3)
            .limit(2)
            .select(["document_id"])
        )
        hits = session.collect(query).to_pylist()
        if [row["document_id"] for row in hits] != [1, 2] or any(
            row["_distance"] < 0 for row in hits
        ):
            raise RuntimeError(f"unexpected installed-wheel search result: {hits}")
    print(f"installed parqdb {version('parqdb')} wheel build/search smoke passed")


if __name__ == "__main__":
    main()
