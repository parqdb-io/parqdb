from __future__ import annotations

import os
from importlib import import_module
from importlib.metadata import version
from tempfile import TemporaryDirectory

import relify


def main() -> None:
    for module in (
        "adbc_driver_flightsql",
        "pyiceberg",
        "pyspark",
        "relify.experimental.spark",
        "relify.experimental.starrocks",
    ):
        import_module(module)

    with TemporaryDirectory(prefix="relify-wheel-smoke-") as directory:
        session = relify.connect(os.path.join(directory, "relify-data"))
        session.register_parquet("documents", relify.datasets.uri("documents"))
        table = session.table("documents")
        table.create_index(
            "smoke",
            column="embedding",
            key=["document_id"],
            config=relify.IVF(nlist=3),
        )
        table.wait_for_index("smoke")
        query = (
            table.search([0.2, 0.0], column="embedding")
            .where("tenant_id = 42 AND status = 'published'")
            .nprobes(3)
            .limit(2)
            .select(["document_id"])
        )
        hits = session.to_arrow(query).to_pylist()
        if [row["document_id"] for row in hits] != [1, 2] or any(
            row["_distance"] < 0 for row in hits
        ):
            raise RuntimeError(f"unexpected installed-wheel search result: {hits}")
    print(f"installed relify {version('relify')} wheel build/search smoke passed")


if __name__ == "__main__":
    main()
