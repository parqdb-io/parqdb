from __future__ import annotations

import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, Protocol
from weakref import finalize

from ._native import (
    AlreadyExistsError,
    BuildAlreadyRunningError,
    IndexNotFoundError,
    InvalidArgumentError,
    _NativeBuildProgress,
)
from .config import IVF, WriteOptions, native_writer_options
from .identifier import TableIdentifier

_INDEX_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class IndexStatus:
    state: Literal["pending", "building", "ready", "failed"]
    builder: str
    progress: float | None = None
    phase: str | None = None
    completed: int | None = None
    total: int | None = None
    current_snapshot_id: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class _BuildRequest:
    source_identifier: TableIdentifier
    source: dict[str, object]
    index: str
    column: str
    key: tuple[str, ...]
    config: IVF
    writer_options: WriteOptions

    @property
    def profile(self) -> str:
        profile = self.source.get("profile")
        if not isinstance(profile, str) or not profile:
            raise ValueError("source relation has no valid profile")
        return profile


@dataclass(frozen=True)
class _BuildProgressSnapshot:
    phase: str | None
    completed: int | None
    total: int | None
    fraction: float | None


class _BuildProgress(Protocol):
    def snapshot(self) -> _BuildProgressSnapshot: ...


@dataclass(frozen=True)
class _LocalBuildProgress:
    tracker: _NativeBuildProgress = field(repr=False)

    def snapshot(self) -> _BuildProgressSnapshot:
        phase, completed, total, fraction = self.tracker.snapshot()
        return _BuildProgressSnapshot(phase, completed, total, fraction)


@dataclass(frozen=True)
class _LocalBuildContext:
    runtime: Any = field(repr=False)
    progress: _LocalBuildProgress = field(repr=False)


@dataclass(frozen=True)
class _RefreshRequest:
    source_identifier: TableIdentifier
    source: dict[str, object]
    index: str
    config: IVF | None
    writer_options: WriteOptions


@dataclass
class _BuildRecord:
    source: TableIdentifier
    builder: str
    state: Literal["pending", "building", "failed"]
    base_metadata_location: str | None
    progress: _BuildProgress | None = field(default=None, repr=False)
    future: Future[str] | None = None
    error: BaseException | None = None


class BuildOperation:
    """Handle for one asynchronous index construction operation."""

    def __init__(
        self,
        coordinator: BuildCoordinator,
        source: TableIdentifier,
        index: str,
        future: Future[str],
    ) -> None:
        self._coordinator = coordinator
        self._source = source
        self._index = index
        self._future = future

    @property
    def index(self) -> str:
        return self._index

    def status(self) -> IndexStatus:
        return self._coordinator.status(self._source, self._index)

    def wait(self, timeout: timedelta = timedelta(minutes=5)) -> None:
        self._coordinator.wait(self._source, self._index, timeout)

    def cancel(self) -> bool:
        return self._future.cancel()

    def result(self, timeout: float | None = None) -> str:
        return self._future.result(timeout=timeout)


class BuildCoordinator:
    """Owns asynchronous lifecycle, publication, and status for one session."""

    def __init__(self, host: Any) -> None:
        self._host = host
        self._records: dict[tuple[TableIdentifier, str], _BuildRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="relify-build",
        )
        self._executor_finalizer = finalize(
            self,
            _shutdown_executor,
            self._executor,
        )

    def create(
        self,
        source_identifier: TableIdentifier,
        *,
        index: str,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None:
        operation = self.submit_create(
            source_identifier,
            index=index,
            column=column,
            key=key,
            config=config,
            writer_options=writer_options,
        )
        if wait_timeout is not None:
            _validate_timeout(wait_timeout, "wait_timeout")
            try:
                operation.result(timeout=wait_timeout.total_seconds())
            except FutureTimeout as error:
                raise TimeoutError(f"timed out waiting for index: {index}") from error

    def submit_create(
        self,
        source_identifier: TableIdentifier,
        *,
        index: str,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None = None,
    ) -> BuildOperation:
        options = writer_options or WriteOptions()
        _validate_create(index, column, key, config, options)
        source = self._host._resolve_build_relation(source_identifier)
        request = _BuildRequest(
            source_identifier=source_identifier,
            source=dict(source),
            index=index,
            column=column,
            key=tuple(key),
            config=config,
            writer_options=options,
        )
        context = self._host._build_context()
        repository = self._host._index_repository()
        with self._lock:
            if repository.index_exists(index):
                raise AlreadyExistsError(f"index already exists: {index}")
            self._reserve(
                source_identifier,
                index,
                "local",
                base_metadata_location=None,
                progress=context.progress,
            )
            record_key = (source_identifier, index)
            record = self._records[record_key]

            def run() -> str:
                self._mark_building(record)
                try:
                    location = _build_local(request, context)
                    self._remove_record(record_key, record)
                    return location
                except BaseException as error:
                    self._mark_failed(record, error)
                    raise

            future = self._executor.submit(run)
            record.future = future
            future.add_done_callback(
                lambda completed: (
                    self._remove_record(record_key, record)
                    if completed.cancelled()
                    else None
                )
            )
        return BuildOperation(self, source_identifier, index, future)

    def refresh(
        self,
        source_identifier: TableIdentifier,
        *,
        index: str,
        config: IVF | None,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None:
        if wait_timeout is not None:
            _validate_timeout(wait_timeout, "wait_timeout")
        options = writer_options or WriteOptions()
        if config is not None and not isinstance(config, IVF):
            raise TypeError("the first implementation supports only relify.IVF")
        if not isinstance(options, WriteOptions):
            raise TypeError("writer_options must be relify.WriteOptions")
        source = self._host._resolve_build_relation(source_identifier)
        source_profile = source.get("profile")
        if not isinstance(source_profile, str):
            raise ValueError("source relation has no valid profile")
        if source_profile != "parquet":
            raise NotImplementedError(
                "the local builder currently supports Parquet source tables"
            )
        context = self._host._build_context()
        _, base_metadata_location = self._published_state(source_identifier, index)
        request = _RefreshRequest(
            source_identifier,
            dict(source),
            index,
            config,
            options,
        )
        with self._lock:
            self._reserve(
                source_identifier,
                index,
                "local",
                base_metadata_location,
                progress=context.progress,
            )
            record_key = (source_identifier, index)
            record = self._records[record_key]

            def run() -> str:
                self._mark_building(record)
                try:
                    location = _refresh_local(request, context)
                    self._remove_record(record_key, record)
                    return location
                except BaseException as error:
                    self._mark_failed(record, error)
                    raise

            future = self._executor.submit(run)
            record.future = future
        if wait_timeout is not None:
            try:
                future.result(timeout=wait_timeout.total_seconds())
            except FutureTimeout as error:
                raise TimeoutError(f"timed out waiting for index: {index}") from error

    def status(
        self,
        source_identifier: TableIdentifier,
        index: str,
    ) -> IndexStatus:
        _validate_index(index)
        with self._lock:
            record_key = (source_identifier, index)
            record = self._records.get(record_key)
            if record is not None:
                state = record.state
                error = record.error
            else:
                record = None
                state = None
                error = None
        published = self._published_state_or_none(source_identifier, index)
        current_snapshot_id = published[0] if published is not None else None
        if record is not None and state in {"pending", "building"}:
            phase, completed, total, progress = _progress_snapshot(record.progress)
            return IndexStatus(
                state=state,
                builder=record.builder,
                progress=progress,
                phase=phase,
                completed=completed,
                total=total,
                current_snapshot_id=current_snapshot_id,
                error=str(error) if error is not None else None,
            )
        if published is not None:
            published_error = (
                error
                if record is not None and record.base_metadata_location == published[1]
                else None
            )
            if record is not None and published_error is None:
                self._remove_record(record_key, record)
            return IndexStatus(
                state="ready",
                builder=self._published_builder(index),
                current_snapshot_id=current_snapshot_id,
                error=str(published_error) if published_error is not None else None,
            )
        if record is not None:
            return IndexStatus(
                state="failed",
                builder=record.builder,
                error=str(error) if error is not None else None,
            )
        raise IndexNotFoundError(f"index not found: {index}")

    def wait(
        self,
        source_identifier: TableIdentifier,
        index: str,
        timeout: timedelta,
    ) -> None:
        _validate_index(index)
        _validate_timeout(timeout, "timeout")
        with self._lock:
            record = self._records.get((source_identifier, index))
            future = record.future if record is not None else None
        if future is None:
            self._published_state(source_identifier, index)
            return
        try:
            future.result(timeout=timeout.total_seconds())
        except FutureTimeout as error:
            raise TimeoutError(f"timed out waiting for index: {index}") from error

    def _reserve(
        self,
        source: TableIdentifier,
        index: str,
        builder: str,
        base_metadata_location: str | None,
        progress: Any,
    ) -> None:
        record_key = (source, index)
        active = self._records.get(record_key)
        if active is not None and active.state in {"pending", "building"}:
            raise BuildAlreadyRunningError(
                f"an index build is already running: {index}"
            )
        self._records[record_key] = _BuildRecord(
            source,
            builder,
            "pending",
            base_metadata_location,
            progress,
        )

    def _published_state(
        self,
        source_identifier: TableIdentifier,
        index: str,
    ) -> tuple[int, str]:
        matching = {
            info.name: info.current_snapshot_id
            for info in self._host._list_table_indexes(source_identifier)
        }
        if index not in matching:
            raise IndexNotFoundError(f"index not found: {index}")
        entry = self._host.indexes.load(index)
        return matching[index], entry.metadata_location

    def _published_state_or_none(
        self,
        source_identifier: TableIdentifier,
        index: str,
    ) -> tuple[int, str] | None:
        try:
            return self._published_state(source_identifier, index)
        except IndexNotFoundError:
            return None

    def _published_builder(self, index: str) -> str:
        metadata = self._host.indexes.load(index).metadata
        snapshot_id = metadata["current-snapshot-id"]
        for snapshot in metadata["snapshots"]:
            if snapshot["snapshot-id"] == snapshot_id:
                summary = snapshot.get("summary", {})
                builder = summary.get("builder")
                if isinstance(builder, str) and builder:
                    return builder
                break
        return "unknown"

    def _mark_building(self, record: _BuildRecord) -> None:
        with self._lock:
            record.state = "building"

    def _remove_record(
        self,
        record_key: tuple[TableIdentifier, str],
        record: _BuildRecord,
    ) -> None:
        with self._lock:
            if self._records.get(record_key) is record:
                del self._records[record_key]

    def _mark_failed(self, record: _BuildRecord, error: BaseException) -> None:
        with self._lock:
            record.state = "failed"
            record.error = error


def _build_local(
    request: _BuildRequest,
    context: _LocalBuildContext,
) -> str:
    if request.profile != "parquet":
        raise NotImplementedError(
            "the local builder currently supports Parquet source tables"
        )
    source = request.source.get("uri")
    if not isinstance(source, str):
        raise ValueError("Parquet source relation has no URI")
    location = context.runtime.create_index(
        source=source,
        index_name=request.index,
        vector_field=request.column,
        source_key_fields=list(request.key),
        nlist=request.config.nlist,
        posting_encoding=request.config.encoding,
        metric=request.config.metric,
        writer_options=native_writer_options(request.writer_options),
        partitions=request.writer_options.partitions,
        threads=None,
        progress=context.progress.tracker,
    )
    return location


def _refresh_local(
    request: _RefreshRequest,
    context: _LocalBuildContext,
) -> str:
    if request.source.get("profile") != "parquet":
        raise NotImplementedError(
            "the local builder currently supports Parquet source tables"
        )
    source = request.source.get("uri")
    if not isinstance(source, str):
        raise ValueError("Parquet source relation has no URI")
    location = context.runtime.refresh_index(
        source=source,
        index_name=request.index,
        nlist=request.config.nlist if request.config is not None else None,
        posting_encoding=(
            request.config.encoding if request.config is not None else None
        ),
        metric=request.config.metric if request.config is not None else None,
        writer_options=native_writer_options(request.writer_options),
        partitions=request.writer_options.partitions,
        threads=None,
        progress=context.progress.tracker,
    )
    return location


def _progress_snapshot(
    tracker: _BuildProgress | None,
) -> tuple[str | None, int | None, int | None, float | None]:
    if tracker is None:
        return None, None, None, None
    snapshot = tracker.snapshot()
    return (
        snapshot.phase,
        snapshot.completed,
        snapshot.total,
        snapshot.fraction,
    )


def _validate_create(
    index: str,
    column: str,
    key: list[str],
    config: IVF,
    writer_options: WriteOptions,
) -> None:
    _validate_index(index)
    if not isinstance(column, str) or not column:
        raise InvalidArgumentError("vector column must not be empty")
    if (
        not isinstance(key, list)
        or not key
        or any(not isinstance(field, str) or not field for field in key)
        or len(set(key)) != len(key)
    ):
        raise InvalidArgumentError("key must contain unique, non-empty column names")
    if not isinstance(config, IVF):
        raise TypeError("the first implementation supports only relify.IVF")
    if not isinstance(writer_options, WriteOptions):
        raise TypeError("writer_options must be relify.WriteOptions")


def _validate_timeout(timeout: timedelta, name: str) -> None:
    if not isinstance(timeout, timedelta):
        raise TypeError(f"{name} must be datetime.timedelta")
    if timeout.total_seconds() <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_index(index: str) -> None:
    if not isinstance(index, str) or _INDEX_NAME.fullmatch(index) is None:
        raise InvalidArgumentError("index name must match [A-Za-z_][A-Za-z0-9_]*")


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    executor.shutdown(wait=False, cancel_futures=False)
