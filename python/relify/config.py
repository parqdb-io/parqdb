from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._native import _new_session_config, _ParquetWriterOptions
from .builders.v1 import (
    BuilderCapabilities,
    BuilderInfo,
    BuildProfile,
)
from .datafusion import SessionConfig as DataFusionSessionConfig

if TYPE_CHECKING:
    from .build import _PublishedBuild, _RefreshRequest
    from .builders.v1 import BuildContext, BuildOutput, BuildRequest


LOCAL_BUILDER_INFO = BuilderInfo("local", "Local Rust", "relify")
LOCAL_BUILDER_CAPABILITIES = BuilderCapabilities(
    frozenset({BuildProfile("ivf", "parquet", "parquet")})
)


class SessionConfig(DataFusionSessionConfig):
    """DataFusion session configuration with Relify options installed."""

    def __init__(self, config_options: dict[str, str] | None = None) -> None:
        self.config_internal = _new_session_config()
        for key, value in (config_options or {}).items():
            self.config_internal = self.config_internal.set(key, value)


@dataclass(frozen=True)
class IVF:
    nlist: int
    encoding: str = "flat"

    def __post_init__(self) -> None:
        if not isinstance(self.nlist, int) or isinstance(self.nlist, bool):
            raise TypeError("nlist must be an integer")
        if self.nlist <= 0:
            raise ValueError("nlist must be positive")
        if not isinstance(self.encoding, str):
            raise TypeError("encoding must be a string")
        if self.encoding not in {"source", "flat", "lvq4", "lvq8"}:
            raise ValueError(f"unsupported encoding: {self.encoding}")


@dataclass(frozen=True)
class WriteOptions:
    """Configure the physical output of one index build."""

    partitions: int | None = None
    compression: str = "uncompressed"
    target_file_size: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.partitions is not None and (
            not isinstance(self.partitions, int)
            or isinstance(self.partitions, bool)
            or self.partitions <= 0
        ):
            raise ValueError("partitions must be a positive integer")
        if not isinstance(self.compression, str):
            raise TypeError("compression must be a string")
        simple_codecs = {"uncompressed", "snappy", "lz4", "lz4_raw"}
        codec_levels = {
            "gzip": (0, 9),
            "brotli": (0, 11),
            "zstd": (1, 22),
        }
        levelled_codec = False
        for codec, (minimum, maximum) in codec_levels.items():
            prefix = f"{codec}("
            if self.compression.startswith(prefix) and self.compression.endswith(")"):
                level = self.compression[len(prefix) : -1]
                levelled_codec = level.isdigit() and minimum <= int(level) <= maximum
                break
        if self.compression not in simple_codecs and not levelled_codec:
            raise ValueError(f"unsupported Parquet compression: {self.compression}")
        if (
            not isinstance(self.target_file_size, int)
            or isinstance(self.target_file_size, bool)
            or self.target_file_size <= 0
        ):
            raise ValueError("target_file_size must be a positive integer")


@dataclass(frozen=True)
class Local:
    """Configure the built-in local Rust builder."""

    threads: int | None = None
    max_row_group_rows: int | None = None
    write_batch_rows: int = 8_192

    def __post_init__(self) -> None:
        if self.threads is not None:
            if not isinstance(self.threads, int) or isinstance(self.threads, bool):
                raise TypeError("threads must be an integer")
            if self.threads <= 0:
                raise ValueError("threads must be positive")
        if self.max_row_group_rows is not None and (
            not isinstance(self.max_row_group_rows, int)
            or isinstance(self.max_row_group_rows, bool)
            or self.max_row_group_rows <= 0
        ):
            raise ValueError("max_row_group_rows must be a positive integer")
        if (
            not isinstance(self.write_batch_rows, int)
            or isinstance(self.write_batch_rows, bool)
            or self.write_batch_rows <= 0
        ):
            raise ValueError("write_batch_rows must be a positive integer")

    @property
    def info(self) -> BuilderInfo:
        return LOCAL_BUILDER_INFO

    @property
    def capabilities(self) -> BuilderCapabilities:
        return LOCAL_BUILDER_CAPABILITIES

    def build(
        self,
        request: BuildRequest,
        context: BuildContext,
    ) -> BuildOutput | _PublishedBuild:
        from .build import _build_local

        return _build_local(self, request, context)

    def refresh(
        self,
        request: _RefreshRequest,
        context: BuildContext,
    ) -> _PublishedBuild:
        from .build import _refresh_local

        return _refresh_local(self, request, context)


def native_writer_options(
    builder: Local,
    options: WriteOptions,
) -> _ParquetWriterOptions:
    return _ParquetWriterOptions(
        compression=options.compression,
        max_row_group_rows=builder.max_row_group_rows,
        target_file_size=options.target_file_size,
        write_batch_rows=builder.write_batch_rows,
    )
