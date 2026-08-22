from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import parqdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import uvicorn
from _support import (
    WAIT,
    artifact_manifest_location,
    build_index,
    load_table_index,
    register_source,
    write_vectors,
)
from parqdb.facade import AsyncSession
from parqdb.runtime.service import SessionService
from parqdb.server.app import create_http_app, create_http_app_for_service
from parqdb.server.openapi import openapi_document
from parqdb.server.source_policy import SourceUriPolicy
from parqdb.transport.base import InProcessTransport
from parqdb.transport.http import HttpTransport
from parqdb.transport.models import (
    decode_identifier_path,
    encode_identifier_path,
    identifier_from_json,
    identifier_to_json,
    index_info_from_json,
    index_info_to_json,
    index_status_from_json,
    index_status_to_json,
    ivf_from_json,
    ivf_to_json,
    registration_from_json,
    registration_to_json,
    vector_query_from_json,
    vector_query_to_json,
    writer_options_from_json,
    writer_options_to_json,
)


class _FailingBatchStream:
    def schema(self) -> pa.Schema:
        return pa.schema([("value", pa.int64())])

    def __aiter__(self) -> _FailingBatchStream:
        return self

    async def __anext__(self) -> pa.RecordBatch:
        raise RuntimeError("execution failed after planning")

    async def aclose(self) -> None:
        pass


class _FailingService(SessionService):
    async def stream(self, query: parqdb.VectorQuery | str) -> _FailingBatchStream:
        return _FailingBatchStream()


def test_http_wire_models_round_trip_portable_values() -> None:
    identifier = parqdb.TableIdentifier("catalog", ("space name", "a/b"), "x$y")
    path, delimiter = encode_identifier_path(identifier, delimiter="~")
    query = parqdb.VectorQuery(
        source=identifier,
        query=(1.0, 2.0),
        column="embedding",
        index="vectors",
        projection=("id", "payload"),
        result_limit=7,
        probe_count=3,
        predicate="id > 0",
        bypass_index=True,
    )

    assert decode_identifier_path(path, delimiter=delimiter) == (
        "space name",
        "a/b",
        "x$y",
    )
    assert identifier_from_json(identifier_to_json(identifier)) == identifier
    assert vector_query_from_json(vector_query_to_json(query)) == query

    options = parqdb.WriteOptions(
        partitions=2,
        compression="zstd(3)",
        target_file_size=1024,
        max_row_group_rows=64,
        write_batch_rows=32,
    )
    config = parqdb.IVF(nlist=8, encoding="lvq4", metric="cosine")
    status = parqdb.IndexStatus("building", 0.5, "posting", 1, 2, None, None)
    info = parqdb.IndexInfo("embedding", "vector", "ivf", "cosine", {"nlist": "8"}, 3)
    registration = registration_to_json(
        "documents",
        "/data/documents/*.parquet",
        table_partition_cols=[("day", pa.date32())],
        parquet_pruning=True,
        file_extension=".parquet",
        skip_metadata=True,
        schema=pa.schema([("id", pa.int64())]),
        file_sort_order=[["day", "id"]],
    )

    identifier_payload = identifier_to_json(identifier)
    query_payload = vector_query_to_json(query)
    config_payload = ivf_to_json(config)
    options_payload = writer_options_to_json(options)
    status_payload = index_status_to_json(status)
    info_payload = index_info_to_json(info)

    assert ivf_from_json(config_payload) == config
    assert writer_options_from_json(options_payload) == options
    assert index_status_from_json(status_payload) == status
    assert index_info_from_json(info_payload) == info
    decoded_registration = registration_from_json(registration)
    assert decoded_registration["table_partition_cols"] == [("day", pa.date32())]
    assert decoded_registration["schema"] == pa.schema([("id", pa.int64())])
    assert decoded_registration["file_sort_order"] == [["day", "id"]]

    schemas = openapi_document()["components"]["schemas"]
    for schema_name, payload in (
        ("TableIdentifier", identifier_payload),
        ("VectorQuery", query_payload),
        ("RegisterParquetRequest", registration),
        ("IvfConfig", config_payload),
        ("WriteOptions", options_payload),
        ("IndexStatus", status_payload),
        ("IndexInfo", info_payload),
    ):
        assert set(schemas[schema_name]["properties"]) == set(payload)


def test_source_uri_policy_uses_canonical_prefix_boundaries(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "allowed-sibling"
    allowed.mkdir()
    outside.mkdir()
    source = allowed / "vectors.parquet"
    source.touch()
    escaped = allowed / "escaped.parquet"
    escaped.symlink_to(outside / "vectors.parquet")
    policy = SourceUriPolicy([allowed, "s3://bucket/indexes"])

    assert policy.authorize(source) == str(source)
    assert policy.authorize("s3://bucket/indexes/vectors/*.parquet") == (
        "s3://bucket/indexes/vectors/*.parquet"
    )
    with pytest.raises(PermissionError):
        SourceUriPolicy().authorize(source)
    with pytest.raises(PermissionError):
        policy.authorize(outside / "vectors.parquet")
    with pytest.raises(PermissionError):
        policy.authorize(escaped)
    with pytest.raises(PermissionError):
        policy.authorize("s3://bucket/indexes-old/vectors.parquet")
    with pytest.raises(PermissionError):
        policy.authorize("s3://bucket/indexes/%2e%2e/private/vectors.parquet")
    with pytest.raises(ValueError, match="credentials"):
        policy.authorize("s3://user:secret@bucket/indexes/vectors.parquet")
    with pytest.raises(ValueError, match="control characters"):
        policy.authorize("s3://bucket/indexes/%00.parquet")


@pytest.mark.parametrize(
    "body",
    [
        {"source": {}, "query": [1.0]},
        {
            "source": {"catalog": "c", "namespace": [], "name": "t"},
            "query": [float("nan")],
        },
        {
            "source": {"catalog": "c", "namespace": [], "name": "t"},
            "query": [1.0],
            "unknown": True,
        },
    ],
)
def test_http_wire_models_reject_invalid_queries(body: object) -> None:
    with pytest.raises(ValueError):
        vector_query_from_json(body)


def test_http_transport_matches_embedded_query_surface(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "parqdb-data"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    embedded = parqdb.connect(root)
    vectors = register_source(embedded, source)
    build_index(vectors)
    expected = embedded.collect(vectors.search([0.0, 0.0]).limit(2))
    embedded.close()

    async def exercise() -> None:
        service = await SessionService.open(
            root,
            warehouse=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        app = create_http_app_for_service(service)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://parqdb.test/",
        )
        transport = await HttpTransport.open("http://parqdb.test", client=client)
        session = AsyncSession(transport)
        try:
            identifiers = await session.list_tables()
            assert [identifier.name for identifier in identifiers] == ["vectors"]
            remote_vectors = await session.table("vectors")
            assert remote_vectors.identifier == vectors.identifier
            assert remote_vectors.schema == vectors.schema

            actual = await session.collect(remote_vectors.search([0.0, 0.0]).limit(2))
            assert actual == expected
            assert (await session.sql("SELECT COUNT(*) AS count FROM vectors"))[
                "count"
            ].to_pylist() == [4]
            empty = await session.stream("SELECT * FROM vectors WHERE false")
            assert empty.schema == vectors.schema
            assert [batch async for batch in empty] == []

            query = remote_vectors.search([0.0, 0.0]).limit(2)
            assert "ORDER BY" in await session.to_sql(query)
            assert "physical_plan" in await session.explain(query)
            with pytest.raises(parqdb.UnsupportedOperationError):
                session.datafusion_context()

            with pytest.raises(parqdb.InvalidArgumentError, match="read-only"):
                await session.sql("CREATE TABLE forbidden (value BIGINT)")
        finally:
            await session.close()
            await client.aclose()
            await service.close()

    asyncio.run(exercise())


def test_http_registers_an_index_already_published_in_the_warehouse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "parqdb-data"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    embedded = parqdb.connect(root)
    vectors = register_source(embedded, source)
    build_index(vectors, nlist=1, encoding="lvq8")
    published = load_table_index(embedded, vectors, "vectors_embedding")
    manifest_location = artifact_manifest_location(published, embedded.warehouse)
    vectors.drop_index("vectors_embedding")
    embedded.close()

    async def exercise() -> None:
        service = await SessionService.open(
            root,
            warehouse=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        app = create_http_app_for_service(service)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://parqdb.test/",
        )
        transport = await HttpTransport.open("http://parqdb.test", client=client)
        session = AsyncSession(transport)
        try:
            remote_vectors = await session.table("vectors")
            await remote_vectors.register_index(
                "vectors_embedding",
                manifest_location=manifest_location,
            )

            assert (
                await remote_vectors.index_status("vectors_embedding")
            ).state == "ready"
            result = await session.collect(remote_vectors.search([0.0, 0.0]).limit(1))
            assert result["id"].to_pylist() == [0]
        finally:
            await session.close()
            await client.aclose()
            await service.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["embedded", "http"])
def test_transport_lifecycle_conformance(tmp_path: Path, mode: str) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "parqdb-data"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )

    async def exercise() -> None:
        service = await SessionService.open(
            root,
            warehouse=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        client: httpx.AsyncClient | None = None
        if mode == "embedded":
            transport = InProcessTransport(service)
        else:
            app = create_http_app_for_service(
                service,
                allowed_source_prefixes=[tmp_path],
            )
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://parqdb.test/",
            )
            transport = await HttpTransport.open("http://parqdb.test", client=client)
        session = AsyncSession(transport)
        try:
            source_schema = pq.read_schema(source)
            await session.register_parquet("vectors", source, schema=source_schema)
            vectors = await session.table("vectors")
            with pytest.raises(parqdb.InvalidSchemaError, match="key column not found"):
                await vectors.create_index(
                    "invalid_embedding",
                    column="embedding",
                    key=["missing"],
                    config=parqdb.IVF(nlist=2),
                    wait_timeout=WAIT,
                )
            failed = await vectors.index_status("invalid_embedding")
            assert failed.state == "failed"
            assert failed.error_code == "invalid_schema"
            await vectors.create_index(
                "vectors_embedding",
                column="embedding",
                key=["id"],
                config=parqdb.IVF(nlist=2, encoding="lvq8"),
                wait_timeout=WAIT,
            )
            status = await vectors.index_status("vectors_embedding")
            assert status.state == "ready"
            indexes = await vectors.list_indexes()
            assert [index.name for index in indexes] == ["vectors_embedding"]
            assert indexes[0].metric == "l2_squared"

            await session.register_parquet("other", source, schema=source_schema)
            other = await session.table("other")
            await other.create_index(
                "vectors_embedding",
                column="embedding",
                key=["id"],
                config=parqdb.IVF(nlist=1),
                wait_timeout=WAIT,
            )
            assert [index.name for index in await other.list_indexes()] == [
                "vectors_embedding"
            ]

            published = status.current_snapshot_id
            with pytest.raises(parqdb.ParqDBError):
                await vectors.refresh_index(
                    "vectors_embedding",
                    config=parqdb.IVF(nlist=8, encoding="lvq8"),
                    wait_timeout=WAIT,
                )
            failed_refresh = await vectors.index_status("vectors_embedding")
            assert failed_refresh.state == "ready"
            assert failed_refresh.current_snapshot_id == published
            assert failed_refresh.error_code is not None

            await vectors.refresh_index(
                "vectors_embedding",
                wait_timeout=WAIT,
            )
            assert (await vectors.index_status("vectors_embedding")).state == "ready"
            assert (
                await session.collect(
                    vectors.search([0.0, 0.0], index="vectors_embedding").limit(2)
                )
            )["id"].to_pylist() == [0, 1]

            await vectors.drop_index("vectors_embedding")
            assert await vectors.list_indexes() == []
            assert [index.name for index in await other.list_indexes()] == [
                "vectors_embedding"
            ]
            await other.drop_index("vectors_embedding")
            await session.deregister_table(other.identifier)
            await session.deregister_table(vectors.identifier)
            assert await session.list_tables() == []
        finally:
            await session.close()
            if client is not None:
                await client.aclose()
            await service.close()

    asyncio.run(exercise())


def test_http_server_denies_unconfigured_source_roots(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])

    async def exercise() -> None:
        service = await SessionService.open(
            tmp_path / "parqdb-data",
            warehouse=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        app = create_http_app_for_service(service)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://parqdb.test/",
        )
        transport = await HttpTransport.open("http://parqdb.test", client=client)
        session = AsyncSession(transport)
        try:
            with pytest.raises(PermissionError, match="allowed file roots"):
                await session.register_parquet("vectors", source)
            assert await session.list_tables() == []
        finally:
            await session.close()
            await client.aclose()
            await service.close()

    asyncio.run(exercise())


def test_http_server_validates_routes_and_publishes_openapi(tmp_path: Path) -> None:
    async def exercise() -> None:
        service = await SessionService.open(
            tmp_path / "parqdb-data",
            warehouse=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        app = create_http_app_for_service(service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://parqdb.test/",
        ) as client:
            openapi = await client.get("openapi.json")
            assert openapi.status_code == 200
            document = openapi.json()
            assert document["openapi"] == "3.1.0"
            for path, path_item in document["paths"].items():
                if "{id}" in path:
                    assert any(
                        parameter["name"] == "id"
                        and parameter["in"] == "path"
                        and parameter["required"]
                        for parameter in path_item["parameters"]
                    )
                for method in {"get", "post"} & path_item.keys():
                    assert path_item[method]["responses"]
            vector_properties = document["components"]["schemas"]["VectorQuery"][
                "properties"
            ]
            assert set(vector_properties) == set(
                vector_query_to_json(
                    parqdb.VectorQuery(
                        parqdb.TableIdentifier("c", (), "t"),
                        (1.0,),
                    )
                )
            )
            assert openapi.headers["x-request-id"]

            mismatch = await client.post(
                "v1/table/wrong/query",
                params={"delimiter": "$"},
                json={
                    "source": {
                        "catalog": "datafusion",
                        "namespace": ["public"],
                        "name": "right",
                    },
                    "query": [1.0],
                    "result_limit": 1,
                },
            )
            assert mismatch.status_code == 400
            assert mismatch.json()["error"]["code"] == "invalid_argument"
            assert mismatch.headers["x-request-id"]
        await service.close()

    asyncio.run(exercise())


def test_http_client_classifies_failure_after_response_headers() -> None:
    async def exercise() -> None:
        service = _FailingService.__new__(_FailingService)
        app = create_http_app_for_service(service)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://parqdb.test/",
        )
        transport = await HttpTransport.open("http://parqdb.test", client=client)
        session = AsyncSession(transport)
        try:
            with pytest.raises(parqdb.StreamExecutionError) as captured:
                await session.stream("SELECT 1")
            assert captured.value.request_id
        finally:
            await session.close()
            await client.aclose()

    asyncio.run(exercise())


def test_public_http_connection_survives_restart_and_releases_cancelled_query(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "parqdb-data"
    write_vectors(source, [1, 2], [[1.0, 0.0], [2.0, 0.0]])
    embedded = parqdb.connect(root)
    register_source(embedded, source)
    embedded.close()
    config = (
        parqdb.SessionConfig()
        .set("parqdb.execution.query_concurrency", "1")
        .set("parqdb.execution.query_queue_capacity", "1")
        .set("parqdb.execution.query_queue_timeout", "100ms")
    )

    async def exercise() -> None:
        for verify_cancellation in (True, False):
            app = create_http_app(root, config=config)
            async with _serve(app) as url:
                session = await parqdb.connect_async(url)
                try:
                    assert (await session.sql("SELECT id FROM vectors ORDER BY id"))[
                        "id"
                    ].to_pylist() == [1, 2]
                    assert await asyncio.to_thread(_sync_count, url) == 2
                    if verify_cancellation:
                        stream = await session.stream("SELECT * FROM range(1000000000)")
                        native = app.state.parqdb_service.host._native
                        await _wait_for_admission(native, (1, 0))
                        queued = asyncio.create_task(session.sql("SELECT 1"))
                        await _wait_for_admission(native, (1, 1))
                        with pytest.raises(parqdb.QueryQueueFullError):
                            await session.sql("SELECT 2")
                        with pytest.raises(parqdb.QueryQueueTimeoutError):
                            await queued
                        await _wait_for_admission(native, (1, 0))
                        await stream.aclose()
                        await _wait_for_admission(native, (0, 0))
                        await asyncio.to_thread(_open_and_close_sync_stream, url)
                        await _wait_for_admission(native, (0, 0))
                finally:
                    await session.close()

    asyncio.run(exercise())


def test_http_lifecycle_survives_server_restart(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "parqdb-data"
    write_vectors(source, [1, 2], [[1.0, 0.0], [2.0, 0.0]])

    async def exercise() -> None:
        for initialize in (True, False):
            app = create_http_app(
                root,
                allowed_source_prefixes=[tmp_path],
            )
            async with _serve(app) as url:
                session = await parqdb.connect_async(url)
                try:
                    if initialize:
                        await session.register_parquet("vectors", source)
                        vectors = await session.table("vectors")
                        await vectors.create_index(
                            "vectors_embedding",
                            column="embedding",
                            key=["id"],
                            config=parqdb.IVF(nlist=1),
                            wait_timeout=WAIT,
                        )
                    vectors = await session.table("vectors")
                    assert (
                        await vectors.index_status("vectors_embedding")
                    ).state == "ready"
                    assert (await session.collect(vectors.search([1.0, 0.0]).limit(1)))[
                        "id"
                    ].to_pylist() == [1]
                finally:
                    await session.close()

    asyncio.run(exercise())


def test_http_transport_reports_unavailable_server() -> None:
    async def exercise() -> None:
        session = await parqdb.connect_async("http://127.0.0.1:1", timeout=0.1)
        try:
            with pytest.raises(parqdb.ServiceUnavailableError):
                await session.sql("SELECT 1")
        finally:
            await session.close()

    asyncio.run(exercise())


@asynccontextmanager
async def _serve(app: Any) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, lifespan="on", log_level="error", access_log=False)
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(500):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("ParqDB test server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)
        listener.close()


async def _wait_for_admission(native: Any, expected: tuple[int, int]) -> None:
    for _ in range(1_000):
        if native.query_admission_stats() == expected:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(
        f"query admission did not reach {expected}: {native.query_admission_stats()}"
    )


def _sync_count(url: str) -> int:
    with parqdb.connect(url) as session:
        return session.sql("SELECT COUNT(*) AS count FROM vectors")["count"][0].as_py()


def _open_and_close_sync_stream(url: str) -> None:
    with parqdb.connect(url) as session:
        reader = session.stream("SELECT * FROM range(1000000000)")
        reader.close()
