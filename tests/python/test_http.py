from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pytest
import relify
import uvicorn
from _support import build_index, register_source, write_vectors
from relify._http_models import (
    decode_identifier_path,
    encode_identifier_path,
    identifier_from_json,
    identifier_to_json,
    vector_query_from_json,
    vector_query_to_json,
)
from relify._http_server import create_http_app, create_http_app_for_service
from relify._http_transport import HttpTransport
from relify._service import SessionService
from relify.facade import AsyncSession


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
    async def stream(self, query: relify.VectorQuery | str) -> _FailingBatchStream:
        return _FailingBatchStream()


def test_http_wire_models_round_trip_portable_values() -> None:
    identifier = relify.TableIdentifier("catalog", ("space name", "a/b"), "x$y")
    path, delimiter = encode_identifier_path(identifier, delimiter="~")
    query = relify.VectorQuery(
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
    root = tmp_path / "relify-data"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    embedded = relify.connect(root)
    vectors = register_source(embedded, source)
    build_index(vectors)
    expected = embedded.collect(vectors.search([0.0, 0.0]).limit(2))
    embedded.close()

    async def exercise() -> None:
        service = await SessionService.open(
            root,
            catalog=None,
            index_root=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        app = create_http_app_for_service(service)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://relify.test/",
        )
        transport = await HttpTransport.open("http://relify.test", client=client)
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
            with pytest.raises(relify.UnsupportedOperationError):
                await remote_vectors.create_index(
                    "other",
                    column="embedding",
                    key=["id"],
                    config=relify.IVF(nlist=2),
                )
            with pytest.raises(relify.UnsupportedOperationError):
                session.datafusion_context()

            with pytest.raises(relify.InvalidArgumentError, match="read-only"):
                await session.sql("CREATE TABLE forbidden (value BIGINT)")
        finally:
            await session.close()
            await client.aclose()
            await service.close()

    asyncio.run(exercise())


def test_http_server_validates_routes_and_publishes_openapi(tmp_path: Path) -> None:
    async def exercise() -> None:
        service = await SessionService.open(
            tmp_path / "relify-data",
            catalog=None,
            index_root=None,
            storage_options=None,
            iceberg=None,
            config=None,
            runtime=None,
        )
        app = create_http_app_for_service(service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://relify.test/",
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
                    relify.VectorQuery(
                        relify.TableIdentifier("c", (), "t"),
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
            base_url="http://relify.test/",
        )
        transport = await HttpTransport.open("http://relify.test", client=client)
        session = AsyncSession(transport)
        try:
            with pytest.raises(relify.StreamExecutionError) as captured:
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
    root = tmp_path / "relify-data"
    write_vectors(source, [1, 2], [[1.0, 0.0], [2.0, 0.0]])
    embedded = relify.connect(root)
    register_source(embedded, source)
    embedded.close()
    config = (
        relify.SessionConfig()
        .set("relify.execution.query_concurrency", "1")
        .set("relify.execution.query_queue_capacity", "1")
        .set("relify.execution.query_queue_timeout", "100ms")
    )

    async def exercise() -> None:
        for verify_cancellation in (True, False):
            app = create_http_app(root, config=config)
            async with _serve(app) as url:
                session = await relify.connect_async(url)
                try:
                    assert (await session.sql("SELECT id FROM vectors ORDER BY id"))[
                        "id"
                    ].to_pylist() == [1, 2]
                    assert await asyncio.to_thread(_sync_count, url) == 2
                    if verify_cancellation:
                        stream = await session.stream("SELECT * FROM range(1000000000)")
                        native = app.state.relify_service.host._native
                        await _wait_for_admission(native, (1, 0))
                        queued = asyncio.create_task(session.sql("SELECT 1"))
                        await _wait_for_admission(native, (1, 1))
                        with pytest.raises(relify.QueryQueueFullError):
                            await session.sql("SELECT 2")
                        with pytest.raises(relify.QueryQueueTimeoutError):
                            await queued
                        await _wait_for_admission(native, (1, 0))
                        await stream.aclose()
                        await _wait_for_admission(native, (0, 0))
                        await asyncio.to_thread(_open_and_close_sync_stream, url)
                        await _wait_for_admission(native, (0, 0))
                finally:
                    await session.close()

    asyncio.run(exercise())


def test_http_transport_reports_unavailable_server() -> None:
    async def exercise() -> None:
        session = await relify.connect_async("http://127.0.0.1:1", timeout=0.1)
        try:
            with pytest.raises(relify.ServiceUnavailableError):
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
            raise AssertionError("Relify test server did not start")
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
    with relify.connect(url) as session:
        return session.sql("SELECT COUNT(*) AS count FROM vectors")["count"][0].as_py()


def _open_and_close_sync_stream(url: str) -> None:
    with relify.connect(url) as session:
        reader = session.stream("SELECT * FROM range(1000000000)")
        reader.close()
