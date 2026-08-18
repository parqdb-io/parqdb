"""Write a Parquet table, index it, and recover it in a new session."""

import os
from tempfile import TemporaryDirectory

import parqdb
import pyarrow as pa
import pyarrow.parquet as pq

from examples.python._common import DOCUMENT_SCHEMA, DOCUMENTS, build_index


def main() -> None:
    with TemporaryDirectory(prefix="parqdb-parquet-roundtrip-") as workspace:
        root = os.path.join(workspace, "parqdb-data")
        source = os.path.join(workspace, "documents.parquet")

        session = parqdb.connect(root)
        pq.write_table(
            pa.Table.from_pylist(list(DOCUMENTS), schema=DOCUMENT_SCHEMA),
            source,
            compression="zstd",
        )

        session.register_parquet("documents", source)
        documents = session.table("documents")
        assert isinstance(documents, parqdb.SourceTable)
        build_index(documents)

        reopened = parqdb.connect(root)
        recovered = reopened.table("documents")
        assert isinstance(recovered, parqdb.SourceTable)
        query = (
            recovered.search([0.2, 0.0], column="embedding")
            .where("tenant_id = 42 AND status = 'published'")
            .nprobes(3)
            .limit(3)
            .select(["document_id", "title"])
        )
        hits = reopened.collect(query)
        print("Parquet round trip:", hits.to_pylist())


if __name__ == "__main__":
    main()
