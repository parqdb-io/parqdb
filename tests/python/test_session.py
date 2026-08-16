from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pyarrow
import pytest
import relify
from _support import build_index, load_table_index, register_source, write_vectors


def test_connect_rejects_invalid_capabilities(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        relify.connect(tmp_path / "iceberg", iceberg=object())
    with pytest.raises(TypeError, match="unexpected keyword argument 'backend'"):
        relify.connect(tmp_path / "backend", backend=cast(Any, object()))


def test_session_has_explicit_idempotent_lifecycle(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")

    assert session.root == (tmp_path / "relify-data").resolve()
    assert session.warehouse == (tmp_path / "relify-data").resolve().as_uri() + "/"
    assert hasattr(session, "close")
    assert hasattr(session, "__enter__")
    assert not hasattr(session, "indexes")
    assert not hasattr(relify, "open_index_catalog")
    assert not hasattr(session, "context")
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.sql("SELECT 1")


def test_native_sql_stream_is_an_async_arrow_iterator(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "async-stream")

    async def consume() -> list[int]:
        stream = await session._native.stream_sql(
            "SELECT value FROM UNNEST([1, 2, 3]) AS values(value)"
        )
        assert stream.schema() == pyarrow.schema([("value", pyarrow.int64())])
        values: list[int] = []
        async for batch in stream:
            assert isinstance(batch, pyarrow.RecordBatch)
            values.extend(batch.column("value").to_pylist())
        await stream.aclose()
        return values

    assert asyncio.run(consume()) == [1, 2, 3]


def test_cancelling_queued_native_stream_releases_admission(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "cancelled-stream")

    async def wait_for_stats(expected: tuple[int, int]) -> None:
        for _ in range(1_000):
            if session._native.query_admission_stats() == expected:
                return
            await asyncio.sleep(0.001)
        raise AssertionError(
            f"query admission did not reach {expected}: "
            f"{session._native.query_admission_stats()}"
        )

    async def cancel_queued() -> None:
        first = await session._native.stream_sql("SELECT 1 AS value")
        await wait_for_stats((1, 0))
        waiting = asyncio.ensure_future(session._native.stream_sql("SELECT 2 AS value"))
        await wait_for_stats((1, 1))

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await wait_for_stats((1, 0))

        await first.aclose()
        await wait_for_stats((0, 0))

    asyncio.run(cancel_queued())


def test_sync_stream_close_releases_runtime_admission(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "sync-stream")

    reader = session.stream("SELECT * FROM range(1000000)")

    assert isinstance(reader, pyarrow.RecordBatchReader)
    assert not session._async._streams
    assert session._native.query_admission_stats() == (1, 0)
    reader.close()
    assert session._native.query_admission_stats() == (0, 0)

    reader = session.stream("SELECT * FROM range(1000000)")
    assert session._native.query_admission_stats() == (1, 0)
    native = session._native
    session.close()
    assert native.query_admission_stats() == (0, 0)


def test_async_facade_matches_portable_embedded_operations(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [1, 2], [[1.0, 0.0], [2.0, 0.0]])

    async def exercise() -> None:
        native: Any
        async with await relify.connect_async(tmp_path / "async-facade") as session:
            await session.register_parquet("vectors", source)
            assert [table.name for table in await session.list_tables()] == ["vectors"]
            vectors = await session.table("vectors")
            assert vectors.identifier.name == "vectors"
            assert vectors.schema.names == ["id", "payload", "embedding"]
            assert (await session.sql("SELECT id FROM vectors")).to_pydict() == {
                "id": [1, 2]
            }
            stream = await session.stream("SELECT * FROM vectors WHERE false")
            assert stream.schema == vectors.schema
            assert [batch async for batch in stream] == []
            native = cast(Any, session)._transport._service.host._native
            assert native.query_admission_stats() == (0, 0)
            await session.stream("SELECT * FROM range(1000000)")
            assert native.query_admission_stats() == (1, 0)
        assert native.query_admission_stats() == (0, 0)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE forbidden (value BIGINT)",
        "SET datafusion.execution.target_partitions = 1",
    ],
)
def test_portable_sql_is_read_only(tmp_path: Path, sql: str) -> None:
    session = relify.connect(tmp_path / "read-only-sql")

    with pytest.raises(relify.InvalidArgumentError, match="SQL execution is read-only"):
        session.sql(sql)


def test_query_runtime_settings_apply_at_session_creation(tmp_path: Path) -> None:
    config = (
        relify.SessionConfig()
        .set("relify.execution.query_dop", "2")
        .set("relify.execution.query_concurrency", "2")
        .set("relify.execution.query_queue_capacity", "0")
        .set("relify.execution.query_queue_timeout", "100ms")
        .with_information_schema()
    )
    session = relify.connect(tmp_path / "query-runtime", config=config)
    assert session.sql("SHOW datafusion.execution.target_partitions").to_pydict()[
        "value"
    ] == ["2"]

    async def occupy_both_slots() -> None:
        first = await session._native.stream_sql("SELECT 1 AS value")
        second = await session._native.stream_sql("SELECT 2 AS value")
        assert session._native.query_admission_stats() == (2, 0)
        with pytest.raises(relify._native.QueryQueueFullError):
            await session._native.stream_sql("SELECT 3 AS value")
        await first.aclose()
        await second.aclose()
        assert session._native.query_admission_stats() == (0, 0)

    asyncio.run(occupy_both_slots())


def test_session_uses_datafusion_config_and_runtime(tmp_path: Path) -> None:
    config = (
        relify.SessionConfig()
        .set("relify.metadata.cache.max_entries", "7")
        .set("relify.metadata.cache.max_bytes", "4096")
        .with_target_partitions(3)
        .with_information_schema()
    )
    assert isinstance(config, relify.datafusion.SessionConfig)
    runtime = relify.datafusion.RuntimeEnvBuilder().with_greedy_memory_pool(1 << 20)
    session = relify.connect(
        tmp_path / "configured",
        config=config,
        runtime=runtime,
    )

    assert not isinstance(session, relify.datafusion.SessionContext)
    context = session.datafusion_context()
    assert context.sql("SHOW relify.metadata.cache.max_entries").to_pydict()[
        "value"
    ] == ["7"]
    assert context.sql("SHOW datafusion.execution.target_partitions").to_pydict()[
        "value"
    ] == ["3"]
    context.sql("SET relify.metadata.cache.max_entries = 8")
    assert context.sql("SHOW relify.metadata.cache.max_entries").to_pydict()[
        "value"
    ] == ["8"]


def test_session_exposes_bounded_parquet_page_cache(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, list(range(1_024)), [[float(i), 0.0] for i in range(1_024)])
    config = relify.SessionConfig().set("relify.parquet.page_cache.capacity", "1048576")
    session = relify.connect(tmp_path / "page-cache", config=config)
    session.register_parquet("vectors", source)

    assert session.sql("SELECT id FROM vectors")["id"].to_pylist() == list(range(1_024))
    cold = session.parquet_page_cache_stats()
    assert isinstance(cold, relify.ParquetPageCacheStats)
    assert cold.capacity == 1_048_576
    assert cold.admissions > 0

    session.sql("SELECT id FROM vectors")
    warm = session.parquet_page_cache_stats()
    assert warm.hits > cold.hits

    session.clear_parquet_page_cache()
    assert session.parquet_page_cache_stats().resident_bytes == 0
    session.datafusion_context().sql("SET relify.parquet.page_cache.capacity = 0")
    session.sql("SELECT id FROM vectors")
    assert session.parquet_page_cache_stats().capacity == 0


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("config", object(), "config must be relify.datafusion.SessionConfig"),
        ("runtime", object(), "runtime must be relify.datafusion.RuntimeEnvBuilder"),
    ],
)
def test_session_rejects_invalid_datafusion_initialization_options(
    tmp_path: Path,
    name: str,
    value: object,
    expected: str,
) -> None:
    with pytest.raises(TypeError, match=expected):
        relify.connect(tmp_path / name, **cast(Any, {name: value}))


def test_session_rejects_zero_build_dop(tmp_path: Path) -> None:
    config = relify.SessionConfig().set("relify.build.dop", "0")

    with pytest.raises(relify.InvalidArgumentError, match=r"relify\.build\.dop"):
        relify.connect(tmp_path / "zero-build-dop", config=config)


def test_table_only_resolves_registered_names(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")

    with pytest.raises(RuntimeError, match="failed to resolve schema"):
        session.table("documents.parquet")


def test_registering_a_missing_source_uses_datafusion_error(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")

    with pytest.raises(Exception, match="No files found"):
        session.register_parquet("documents", tmp_path / "missing.parquet")


def test_persistent_table_identifier_cannot_be_silently_rebound(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    write_vectors(first, [1], [[1.0, 0.0]])
    write_vectors(second, [2], [[2.0, 0.0]])
    root = tmp_path / "relify-data"
    session = relify.connect(root)
    session.register_parquet("vectors", first)

    reopened = relify.connect(root)
    with pytest.raises(Exception, match="already exists"):
        reopened.register_parquet("vectors", second)

    assert reopened.sql("SELECT id FROM vectors").to_pydict() == {"id": [1]}


def test_deregister_releases_the_native_source_binding(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [1], [[1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    session.register_parquet("vectors", source)
    query = (
        session.table("vectors").search([1.0, 0.0]).bypass_vector_index().select(["id"])
    )

    session.deregister_table("vectors")
    write_vectors(
        source,
        [2],
        [[2.0, 0.0]],
        vector_column="replacement_embedding",
    )
    session.register_parquet("vectors", source)

    assert session.sql("SELECT id, payload FROM vectors").to_pydict() == {
        "id": [2],
        "payload": ["row-2"],
    }
    assert session.to_arrow(query)["id"].to_pylist() == [2]


def test_registered_parquet_table_has_relify_index_capabilities(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")

    vectors = register_source(session, source)

    assert isinstance(vectors, relify.SourceTable)
    assert not isinstance(vectors, relify.datafusion.DataFrame)
    assert vectors.schema.names == ["id", "payload", "embedding"]


def test_persistent_table_identity_is_canonical_across_qualified_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    root = tmp_path / "relify-data"
    session = relify.connect(root)
    session.register_parquet("datafusion.public.vectors", source)

    for name in ["vectors", "public.vectors", "datafusion.public.vectors"]:
        assert isinstance(session.table(name), relify.SourceTable)

    reopened = relify.connect(root)
    for name in ["vectors", "public.vectors", "datafusion.public.vectors"]:
        assert isinstance(reopened.table(name), relify.SourceTable)

    reopened.deregister_table("public.vectors")
    after_drop = relify.connect(root)
    with pytest.raises(Exception, match=r"No table|failed to resolve schema"):
        after_drop.table("vectors")


def test_non_parquet_relations_remain_plain_dataframes(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")
    context = session.datafusion_context()
    context.register_view("values", context.from_pydict({"value": [1, 2]}))

    values = context.table("values")

    assert type(values) is relify.datafusion.DataFrame
    assert values.to_pydict() == {"value": [1, 2]}
    with pytest.raises(ValueError, match="not a registered Relify source"):
        session.table("values")


def test_file_warehouse_is_independent_from_local_session_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    catalog_path = root / "catalog.sqlite"
    index_root = tmp_path / "indexes"
    index_root.mkdir()

    session = relify.connect(
        root,
        warehouse=index_root.as_uri(),
    )

    assert session.root == root.resolve()
    assert session.warehouse == index_root.as_uri() + "/"
    assert catalog_path.is_file()
    assert not (index_root / "catalog.sqlite").exists()

    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    entry = load_table_index(session, vectors, "vectors_embedding")

    assert entry.metadata_location.startswith(index_root.as_uri() + "/metadata/")
    assert any((index_root / "indexes").rglob("*.parquet"))
    assert not (catalog_path.parent / "indexes").exists()


def test_connect_requires_a_root_and_validates_warehouse_options(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'catalog'"):
        relify.connect(
            tmp_path / "state",
            catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
            warehouse=tmp_path.as_uri(),
        )
    with pytest.raises(TypeError, match="requires a local root or http"):
        relify.connect()
    with pytest.raises(TypeError, match="keys and values"):
        relify.connect(
            tmp_path / "state",
            warehouse=tmp_path.as_uri(),
            storage_options=cast(Any, {"aws_region": 1}),
        )
