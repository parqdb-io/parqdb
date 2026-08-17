"""Search a Parquet table exactly without creating an index."""

from tempfile import TemporaryDirectory

from examples.python._common import open_documents


def main() -> None:
    with TemporaryDirectory(prefix="parqdb-exact-") as workspace:
        session, documents, _ = open_documents(workspace)

        query = (
            documents.search([8.2, 0.0], column="embedding")
            .bypass_vector_index()
            .limit(3)
            .select(["document_id", "title", "category"])
        )
        hits = session.to_arrow(query)
        print("exact-search hits:", hits.to_pylist())


if __name__ == "__main__":
    main()
