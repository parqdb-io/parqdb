from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import AsyncIterator
from typing import Any, Protocol, Self

import pyarrow

from ._native import _IpcDecoder, _IpcEncoder
from ._service import AsyncBatchStream

DEFAULT_CHUNK_BYTES = 64 * 1024
DEFAULT_MAX_FRAME_BYTES = 256 * 1024 * 1024


class AsyncByteStream(Protocol):
    def __aiter__(self) -> AsyncByteStream: ...

    async def __anext__(self) -> bytes: ...

    async def aclose(self) -> None: ...


async def encode_ipc_stream(
    source: AsyncBatchStream,
    *,
    max_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> AsyncByteStream:
    encoder = await asyncio.to_thread(
        _IpcEncoder,
        source.schema(),
        max_chunk_bytes,
    )
    initial = await asyncio.to_thread(encoder.start)
    return _EncodedIpcStream(source, encoder, initial)


async def decode_ipc_stream(
    source: AsyncIterator[bytes],
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> AsyncBatchStream:
    decoder = _IpcDecoder(max_frame_bytes)
    stream = _DecodedIpcStream(source.__aiter__(), decoder)
    await stream.read_schema()
    return stream


class _EncodedIpcStream:
    def __init__(
        self,
        source: AsyncBatchStream,
        encoder: _IpcEncoder,
        initial: list[bytes],
    ) -> None:
        self._source = source
        self._encoder = encoder
        self._pending = deque(initial)
        self._input_finished = False
        self._closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        while not self._pending:
            if self._input_finished:
                await self.aclose()
                raise StopAsyncIteration
            try:
                batch = await self._source.__anext__()
            except StopAsyncIteration:
                self._input_finished = True
                await self._source.aclose()
                self._pending.extend(await asyncio.to_thread(self._encoder.finish))
                continue
            except BaseException:
                await self.aclose()
                raise
            try:
                chunks = await asyncio.to_thread(self._encoder.write, batch)
            except BaseException:
                await self.aclose()
                raise
            finally:
                del batch
            self._pending.extend(chunks)
        return self._pending.popleft()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        if not self._input_finished:
            await self._source.aclose()


class _DecodedIpcStream:
    def __init__(self, source: AsyncIterator[bytes], decoder: _IpcDecoder) -> None:
        self._source = source
        self._decoder = decoder
        self._schema: pyarrow.Schema | None = None
        self._pending: deque[pyarrow.RecordBatch] = deque()
        self._decoder_buffered = False
        self._input_finished = False
        self._closed = False

    def schema(self) -> pyarrow.Schema:
        if self._schema is None:
            raise RuntimeError("Arrow IPC schema has not been decoded")
        return self._schema

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> pyarrow.RecordBatch:
        if self._closed:
            raise StopAsyncIteration
        while not self._pending:
            if self._input_finished:
                await self.aclose()
                raise StopAsyncIteration
            await self._read_next_chunk()
        return self._pending.popleft()

    async def read_schema(self) -> None:
        while self._schema is None:
            if self._input_finished:
                await self.aclose()
                raise ValueError("Arrow IPC stream ended before its schema")
            await self._read_next_chunk()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        if not self._input_finished:
            await _close_async_iterator(self._source)

    async def _read_next_chunk(self) -> None:
        if self._decoder_buffered:
            chunk = b""
        else:
            try:
                chunk = await self._source.__anext__()
            except StopAsyncIteration:
                try:
                    await asyncio.to_thread(self._decoder.finish)
                except BaseException:
                    await self.aclose()
                    raise
                self._input_finished = True
                await _close_async_iterator(self._source)
                return
            except BaseException:
                await self.aclose()
                raise

        if not isinstance(chunk, bytes):
            await self.aclose()
            raise TypeError("Arrow IPC byte stream must yield bytes")
        try:
            schema, batch, self._decoder_buffered = await asyncio.to_thread(
                self._decoder.push,
                chunk,
            )
        except BaseException:
            await self.aclose()
            raise
        if schema is not None:
            if self._schema is not None and schema != self._schema:
                await self.aclose()
                raise ValueError("Arrow IPC stream contains multiple schemas")
            self._schema = schema
        if batch is not None:
            self._pending.append(batch)


async def _close_async_iterator(source: AsyncIterator[bytes]) -> None:
    close = getattr(source, "aclose", None)
    if close is None:
        return
    result: Any = close()
    if inspect.isawaitable(result):
        await result
