"""Write a Parquet table, index it, and recover it in a new session."""

import os
from tempfile import TemporaryDirectory

import pyarrow as pa
import relify

from examples.python._common import DOCUMENT_SCHEMA, DOCUMENTS, build_index


def main() -> None:
    with TemporaryDirectory(prefix="relify-parquet-roundtrip-") as workspace:
        root = os.path.join(workspace, "relify-data")
        source = os.path.join(workspace, "documents")

        session = relify.connect(root)
        source_df = session.from_arrow(
            pa.Table.from_pylist(list(DOCUMENTS), schema=DOCUMENT_SCHEMA)
        )
        source_df.write_parquet(source, compression="zstd")

        session.register_parquet("documents", source)
        documents = session.table("documents")
        assert isinstance(documents, relify.SourceTable)
        build_index(documents)

        reopened = relify.connect(root)
        recovered = reopened.table("documents")
        assert isinstance(recovered, relify.SourceTable)
        query = (
            recovered.search([0.2, 0.0], column="embedding")
            .where("tenant_id = 42 AND status = 'published'")
            .nprobes(3)
            .limit(3)
            .select(["document_id", "title"])
        )
        hits = reopened.to_arrow(query)
        print("Parquet round trip:", hits.to_pylist())


if __name__ == "__main__":
    main()
