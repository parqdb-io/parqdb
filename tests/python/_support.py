from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import relify
from relify.runtime.catalog import CatalogEntry

WAIT = timedelta(seconds=30)


def vector_type(*, nullable_elements: bool = False) -> pa.ListType:
    return pa.list_(pa.field("element", pa.float32(), nullable=nullable_elements))


def write_vectors(
    path: Path,
    ids: list[int],
    vectors: list[list[float]],
    *,
    vector_column: str = "embedding",
) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
            pa.field(vector_column, vector_type(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(ids, type=pa.int64()),
                pa.array([f"row-{value}" for value in ids], type=pa.string()),
                pa.array(vectors, type=vector_type()),
            ],
            schema=schema,
        ),
        path,
    )


def build_index(
    table: relify.SourceTable,
    name: str = "vectors_embedding",
    *,
    column: str = "embedding",
    key: list[str] | None = None,
    nlist: int = 2,
    encoding: str = "source",
    writer_options: relify.WriteOptions | None = None,
) -> None:
    table.create_index(
        name,
        column=column,
        key=key if key is not None else ["id"],
        config=relify.IVF(nlist=nlist, encoding=encoding),
        writer_options=writer_options,
        wait_timeout=WAIT,
    )


def register_source(
    session: relify.Session,
    source: str | Path,
    name: str = "vectors",
) -> relify.SourceTable:
    session.register_parquet(name, source)
    table = session.table(name)
    assert isinstance(table, relify.SourceTable)
    return table


def load_table_index(
    session: relify.Session,
    table: relify.SourceTable,
    index: str,
) -> CatalogEntry:
    return session._indexes.load(
        index,
        namespace=table.identifier.index_namespace,
    )


def register_table_index(
    session: relify.Session,
    table: relify.SourceTable,
    index: str,
    metadata_location: str,
) -> None:
    session._indexes.register(
        index,
        metadata_location,
        namespace=table.identifier.index_namespace,
    )


def drop_table_index_entry(
    session: relify.Session,
    table: relify.SourceTable,
    index: str,
) -> None:
    session._indexes.drop(
        index,
        namespace=table.identifier.index_namespace,
    )


def relation_root(reference: Mapping[str, Any]) -> Path:
    parsed = urlparse(reference["uri"])
    assert reference["profile"] == "parquet"
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def relation_files(reference: Mapping[str, Any]) -> list[Path]:
    return sorted(relation_root(reference).rglob("*.parquet"))


def relation_path(reference: Mapping[str, Any]) -> Path:
    files = relation_files(reference)
    assert files
    return files[0]


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def load_metadata_file(location: str) -> object:
    parsed = urlparse(location)
    assert parsed.scheme == "file"
    return json.loads(Path(unquote(parsed.path)).read_text())
