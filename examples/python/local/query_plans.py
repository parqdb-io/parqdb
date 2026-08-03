"""Inspect a vector query before and during execution."""

from tempfile import TemporaryDirectory

from examples.python._common import build_index, open_documents


def main() -> None:
    with TemporaryDirectory(prefix="relify-plans-") as workspace:
        session, documents, _source = open_documents(workspace)
        build_index(documents)

        query = (
            documents.search([0.0, 0.0], column="embedding")
            .where("tenant_id = 42 AND status = 'published'")
            .nprobes(2)
            .limit(3)
            .select(["document_id", "title"])
        )

        print("query plan:")
        print(session.explain(query))
        print("query runtime metrics:")
        print(session.analyze(query))


if __name__ == "__main__":
    main()
