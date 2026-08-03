"""Build an IVF index and run a filtered vector search."""

from tempfile import TemporaryDirectory

from examples.python._common import build_index, open_documents


def main() -> None:
    with TemporaryDirectory(prefix="relify-quickstart-") as workspace:
        session, documents, _ = open_documents(workspace)
        build_index(documents)

        query = (
            documents.search([0.2, 0.0], column="embedding")
            .where("tenant_id = 42 AND status = 'published'")
            .nprobes(3)
            .limit(3)
            .select(["document_id", "title"])
        )
        hits = session.to_arrow(query)
        print("quickstart hits:", hits.to_pylist())


if __name__ == "__main__":
    main()
