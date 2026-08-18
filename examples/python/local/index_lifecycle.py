"""Refresh, inspect, and drop a published index."""

from tempfile import TemporaryDirectory

from examples.python._common import (
    DOCUMENTS,
    build_index,
    open_documents,
    write_documents,
)

NEW_DOCUMENT = {
    "document_id": 7,
    "title": "Incremental data",
    "tenant_id": 42,
    "status": "published",
    "category": "storage",
    "embedding": [-1.0, 0.0],
}


def main() -> None:
    with TemporaryDirectory(prefix="parqdb-lifecycle-") as workspace:
        _session, documents, source = open_documents(workspace)
        build_index(documents)
        before = documents.list_indexes()[0]

        write_documents(source, [*DOCUMENTS, NEW_DOCUMENT])
        documents.refresh_index("documents_embedding")
        documents.wait_for_index("documents_embedding")
        after = documents.list_indexes()[0]

        documents.drop_index("documents_embedding")

        print(
            "index lifecycle:",
            {
                "snapshots": [
                    before.current_snapshot_id,
                    after.current_snapshot_id,
                ],
                "remaining_indexes": [index.name for index in documents.list_indexes()],
            },
        )


if __name__ == "__main__":
    main()
