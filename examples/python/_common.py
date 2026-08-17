from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import parqdb
import pyarrow as pa
import pyarrow.parquet as pq

_DOCUMENT_TABLE = pq.read_table(parqdb.datasets.uri("documents"))
DOCUMENT_SCHEMA = _DOCUMENT_TABLE.schema
DOCUMENTS: tuple[dict[str, object], ...] = tuple(_DOCUMENT_TABLE.to_pylist())


def open_documents(
    workspace: str,
    rows: Sequence[Mapping[str, object]] = DOCUMENTS,
) -> tuple[parqdb.Session, parqdb.SourceTable, str]:
    source = os.path.join(workspace, "documents.parquet")
    write_documents(source, rows)
    session = parqdb.connect(os.path.join(workspace, "parqdb-data"))
    session.register_parquet("documents", source)
    documents = session.table("documents")
    assert isinstance(documents, parqdb.SourceTable)
    return session, documents, source


def write_documents(
    path: str,
    rows: Sequence[Mapping[str, object]] = DOCUMENTS,
) -> None:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=DOCUMENT_SCHEMA)
    pq.write_table(table, path)


def build_index(documents: parqdb.SourceTable) -> None:
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=parqdb.IVF(nlist=3),
    )
    documents.wait_for_index("documents_embedding")
