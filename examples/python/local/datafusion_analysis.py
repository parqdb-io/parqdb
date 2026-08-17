"""Continue vector search with SQL in the session's native DataFusion context."""

import os
from tempfile import TemporaryDirectory

import parqdb

from examples.python._common import build_index, write_documents


def main() -> None:
    with TemporaryDirectory(prefix="parqdb-datafusion-") as workspace:
        source = os.path.join(workspace, "documents.parquet")
        write_documents(source)

        session = parqdb.connect(os.path.join(workspace, "parqdb-data"))
        session.register_parquet("documents", source)
        documents = session.table("documents")
        assert isinstance(documents, parqdb.SourceTable)
        build_index(documents)

        query = (
            documents.search([0.0, 0.0], column="embedding")
            .nprobes(3)
            .limit(6)
            .select(["document_id", "category", "status"])
        )
        context = session.datafusion_context()
        context.register_record_batches(
            "vector_hits", [session.collect(query).to_batches()]
        )

        try:
            analysis = context.sql(
                """
                SELECT
                    category,
                    COUNT(*) AS matches,
                    MIN(_distance) AS nearest_distance
                FROM vector_hits
                WHERE status = 'published' AND _distance < 100
                GROUP BY category
                ORDER BY matches DESC, category
                """
            )
            print("DataFusion analysis:", analysis.to_pydict())
        finally:
            context.deregister_table("vector_hits")


if __name__ == "__main__":
    main()
