from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import unquote, urljoin, urlparse

import parqdb
import pyarrow as pa
import pyarrow.parquet as pq

WAIT = timedelta(seconds=30)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    identifier: str
    metadata_location: str
    metadata: Mapping[str, Any]


def embedded_native(session: parqdb.Session | parqdb.AsyncSession) -> Any:
    """Return the native host for tests that exercise native-only behavior."""
    async_session = session._async if isinstance(session, parqdb.Session) else session
    transport = cast(Any, async_session)._transport
    return transport._service.host._native


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
    table: parqdb.SourceTable,
    name: str = "vectors_embedding",
    *,
    column: str = "embedding",
    key: list[str] | None = None,
    nlist: int = 2,
    encoding: str = "source",
    writer_options: parqdb.WriteOptions | None = None,
) -> None:
    table.create_index(
        name,
        column=column,
        key=key if key is not None else ["id"],
        config=parqdb.IVF(nlist=nlist, encoding=encoding),
        writer_options=writer_options,
        wait_timeout=WAIT,
    )


def register_source(
    session: parqdb.Session,
    source: str | Path,
    name: str = "vectors",
) -> parqdb.SourceTable:
    session.register_parquet(name, source)
    table = session.table(name)
    assert isinstance(table, parqdb.SourceTable)
    return table


def load_table_index(
    session: parqdb.Session,
    table: parqdb.SourceTable,
    index: str,
) -> CatalogEntry:
    with sqlite3.connect(session.root / "catalog.sqlite") as catalog:
        row = catalog.execute(
            "SELECT metadata_location FROM indexes WHERE namespace = ? AND name = ?",
            (_namespace_key(table.identifier.index_namespace), index),
        ).fetchone()
    if row is None:
        raise parqdb.IndexNotFoundError(f"index not found: {index}")
    metadata_location = str(row[0])
    metadata = load_metadata_file(metadata_location)
    assert isinstance(metadata, dict)
    return CatalogEntry(
        identifier=index,
        metadata_location=metadata_location,
        metadata=_freeze_json(metadata),
    )


def register_table_index(
    table: parqdb.SourceTable,
    index: str,
    manifest_location: str,
) -> None:
    table.register_index(
        index,
        manifest_location=manifest_location,
    )


def artifact_manifest_location(entry: CatalogEntry, warehouse: str) -> str:
    snapshot = entry.metadata["snapshots"][0]
    reference = snapshot["index-relations"]["artifact_manifest"]
    assert isinstance(reference, str)
    return urljoin(warehouse.rstrip("/") + "/", reference)


def drop_table_index_entry(
    session: parqdb.Session,
    table: parqdb.SourceTable,
    index: str,
) -> None:
    with sqlite3.connect(session.root / "catalog.sqlite") as catalog:
        catalog.execute("BEGIN IMMEDIATE")
        row = catalog.execute(
            "SELECT metadata_location FROM indexes WHERE namespace = ? AND name = ?",
            (_namespace_key(table.identifier.index_namespace), index),
        ).fetchone()
        if row is None:
            raise parqdb.IndexNotFoundError(f"index not found: {index}")
        catalog.execute(
            """
            INSERT INTO catalog_tombstones(metadata_location, unreachable_since_ms)
            VALUES (?, ?)
            ON CONFLICT(metadata_location) DO UPDATE SET
                unreachable_since_ms = excluded.unreachable_since_ms
            """,
            (str(row[0]), time.time_ns() // 1_000_000),
        )
        catalog.execute(
            "DELETE FROM indexes WHERE namespace = ? AND name = ?",
            (_namespace_key(table.identifier.index_namespace), index),
        )


def relation_root(reference: str, warehouse: str) -> Path:
    assert isinstance(reference, str)
    parsed = urlparse(urljoin(warehouse.rstrip("/") + "/", reference))
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def relation_files(reference: str, warehouse: str) -> list[Path]:
    return sorted(relation_root(reference, warehouse).rglob("*.parquet"))


def relation_path(reference: str, warehouse: str) -> Path:
    files = relation_files(reference, warehouse)
    assert files
    return files[0]


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def _namespace_key(namespace: tuple[str, ...]) -> str:
    return json.dumps(namespace, separators=(",", ":"))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def load_metadata_file(location: str) -> object:
    parsed = urlparse(location)
    assert parsed.scheme == "file"
    return json.loads(Path(unquote(parsed.path)).read_text())
