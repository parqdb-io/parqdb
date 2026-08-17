from __future__ import annotations

import parqdb
import pytest
from _support import build_index, register_source, write_vectors


@pytest.fixture(scope="module")
def indexed_documents(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[parqdb.Session, parqdb.SourceTable]:
    root = tmp_path_factory.mktemp("indexed-documents")
    source = root / "documents.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    session = parqdb.connect(root / "parqdb-data")
    documents = register_source(session, source, "documents")
    build_index(documents)
    return session, documents
