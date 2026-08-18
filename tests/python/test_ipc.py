from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Self

import pyarrow as pa
import pytest
from parqdb._native import _IpcEncoder
from parqdb.transport.ipc import decode_ipc_stream, encode_ipc_stream


class BatchSource:
    def __init__(self, schema: pa.Schema, batches: Sequence[pa.RecordBatch]) -> None:
        self._schema = schema
        self._batches = iter(batches)
        self.reads = 0
        self.closed = False

    def schema(self) -> pa.Schema:
        return self._schema

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> pa.RecordBatch:
        try:
            batch = next(self._batches)
        except StopIteration:
            raise StopAsyncIteration from None
        self.reads += 1
        return batch

    async def aclose(self) -> None:
        self.closed = True


class ByteSource:
    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = iter(chunks)
        self.reads = 0
        self.closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        try:
            chunk = next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None
        self.reads += 1
        return chunk

    async def aclose(self) -> None:
        self.closed = True


def test_ipc_round_trip_accepts_arbitrary_transport_boundaries() -> None:
    dictionary = pa.array(["a", "b", "a"]).dictionary_encode()
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("label", dictionary.type),
        ]
    )
    batches = [
        pa.RecordBatch.from_arrays(
            [pa.array([1, 2, 3]), dictionary],
            schema=schema,
        ),
        pa.RecordBatch.from_arrays(
            [pa.array([4]), pa.array(["c"]).dictionary_encode()],
            schema=schema,
        ),
    ]

    async def exercise() -> None:
        encoded = await encode_ipc_stream(
            BatchSource(schema, batches),
            max_chunk_bytes=37,
        )
        transport_chunks = [chunk async for chunk in encoded]
        assert transport_chunks
        assert max(map(len, transport_chunks)) <= 37

        payload = b"".join(transport_chunks)
        fragments = [
            payload[offset : offset + 3] for offset in range(0, len(payload), 3)
        ]
        decoded = await decode_ipc_stream(ByteSource(fragments))
        actual = [batch async for batch in decoded]

        assert decoded.schema() == schema
        assert pa.Table.from_batches(actual) == pa.Table.from_batches(batches)

    asyncio.run(exercise())


def test_encoder_and_decoder_apply_batch_backpressure() -> None:
    schema = pa.schema([("value", pa.int64())])
    batches = [
        pa.record_batch([[1, 2]], schema=schema),
        pa.record_batch([[3, 4]], schema=schema),
    ]

    async def exercise() -> None:
        batch_source = BatchSource(schema, batches)
        encoded = await encode_ipc_stream(batch_source, max_chunk_bytes=1 << 20)
        assert batch_source.reads == 0
        schema_chunk = await encoded.__anext__()
        assert batch_source.reads == 0
        first_batch_chunk = await encoded.__anext__()
        assert batch_source.reads == 1
        second_batch_chunk = await encoded.__anext__()
        assert batch_source.reads == 2
        eos_chunk = await encoded.__anext__()
        with pytest.raises(StopAsyncIteration):
            await encoded.__anext__()
        assert batch_source.closed

        byte_source = ByteSource(
            [schema_chunk + first_batch_chunk + second_batch_chunk + eos_chunk]
        )
        decoded = await decode_ipc_stream(byte_source)
        assert byte_source.reads == 1
        assert (await decoded.__anext__()).column(0).to_pylist() == [1, 2]
        assert byte_source.reads == 1
        assert (await decoded.__anext__()).column(0).to_pylist() == [3, 4]
        assert byte_source.reads == 1
        with pytest.raises(StopAsyncIteration):
            await decoded.__anext__()
        assert byte_source.closed

    asyncio.run(exercise())


def test_empty_ipc_stream_preserves_its_schema() -> None:
    schema = pa.schema([("value", pa.string())])

    async def exercise() -> None:
        encoded = await encode_ipc_stream(BatchSource(schema, []))
        decoded = await decode_ipc_stream(
            ByteSource([chunk async for chunk in encoded])
        )

        assert decoded.schema() == schema
        assert [batch async for batch in decoded] == []

    asyncio.run(exercise())


def test_decoder_rejects_truncated_and_oversized_frames() -> None:
    schema = pa.schema([("value", pa.string())])
    batch = pa.record_batch([["x" * 4096]], schema=schema)
    encoder = _IpcEncoder(schema, 1 << 20)
    schema_chunk = encoder.start()[0]
    batch_chunk = encoder.write(batch)[0]
    eos = encoder.finish()[0]

    async def truncated() -> None:
        decoded = await decode_ipc_stream(
            ByteSource([schema_chunk, batch_chunk, eos[:-1]])
        )
        with pytest.raises(ValueError, match="Unexpected End of Stream"):
            _ = [batch async for batch in decoded]

    async def oversized() -> None:
        with pytest.raises(ValueError, match="max_frame_bytes"):
            await decode_ipc_stream(
                ByteSource([schema_chunk, batch_chunk, eos]),
                max_frame_bytes=len(schema_chunk) + 64,
            )

    asyncio.run(truncated())
    asyncio.run(oversized())


def test_closing_codec_streams_closes_their_inputs() -> None:
    schema = pa.schema([("value", pa.int64())])
    batch = pa.record_batch([[1]], schema=schema)

    async def exercise() -> None:
        batch_source = BatchSource(schema, [batch])
        encoded = await encode_ipc_stream(batch_source)
        await encoded.aclose()
        assert batch_source.closed

        encoder = _IpcEncoder(schema, 1 << 20)
        byte_source = ByteSource(
            encoder.start() + encoder.write(batch) + encoder.finish()
        )
        decoded = await decode_ipc_stream(byte_source)
        await decoded.aclose()
        assert byte_source.closed

    asyncio.run(exercise())


def test_encoder_rejects_invalid_options_and_batch_schema() -> None:
    schema = pa.schema([("value", pa.int64())])

    with pytest.raises(ValueError, match="max_chunk_bytes must be positive"):
        _IpcEncoder(schema, 0)

    encoder = _IpcEncoder(schema, 1024)
    encoder.start()
    with pytest.raises(ValueError, match="schema does not match"):
        encoder.write(pa.record_batch([["wrong"]], names=["value"]))
