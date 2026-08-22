from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import parqdb
import pyarrow
import pyarrow.parquet as pq


def main() -> None:
    if find_spec("datafusion") is not None:
        raise RuntimeError(
            "wheel smoke must not rely on an external datafusion package"
        )

    for module in ("numpy", "onnxruntime", "pyiceberg", "tokenizers"):
        import_module(module)

    executable = Path(sys.executable).with_name(
        "parqdb.exe" if os.name == "nt" else "parqdb"
    )
    subprocess.run([str(executable), "publish", "--help"], check=True)

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
        publication_source = Path(directory) / "publication.parquet"
        schema = pyarrow.schema(
            [
                pyarrow.field("chunk_id", pyarrow.int64(), nullable=False),
                pyarrow.field(
                    "embedding", pyarrow.list_(pyarrow.float32(), 2), nullable=False
                ),
            ]
        )
        pq.write_table(
            pyarrow.Table.from_arrays(
                [
                    pyarrow.array([0, 1]),
                    pyarrow.array([[1.0, 0.0], [0.0, 1.0]], type=schema.field(1).type),
                ],
                schema=schema,
            ),
            publication_source,
        )
        from parqdb.publish import build_index, publish

        built = build_index(
            source=publication_source,
            source_key="chunk_id",
            work=Path(directory) / "publication-work",
            nlist=1,
            encoding="lvq8",
            metric="cosine",
            threads=1,
            vector_column="embedding",
        )
        published = Path(directory) / "published"
        publish(
            index_manifest=built.manifest,
            source=publication_source,
            source_key="chunk_id",
            destination=str(published),
        )
        if not (published / "manifest.json").is_file():
            raise RuntimeError("installed-wheel artifact publication smoke failed")
    print(f"installed parqdb {version('parqdb')} wheel build/search smoke passed")


if __name__ == "__main__":
    main()
