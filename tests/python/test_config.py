from __future__ import annotations

from typing import Any, cast

import pytest
import relify


@pytest.mark.parametrize("nlist", [True, 1.5, "1", None])
def test_ivf_requires_an_integer(nlist: object) -> None:
    with pytest.raises(TypeError, match="nlist must be an integer"):
        relify.IVF(cast(Any, nlist))


@pytest.mark.parametrize("nlist", [0, -1])
def test_ivf_requires_a_positive_value(nlist: int) -> None:
    with pytest.raises(ValueError, match="nlist must be positive"):
        relify.IVF(nlist)


def test_ivf_stores_vectors_by_default_and_requires_a_boolean() -> None:
    assert relify.IVF(1).store_vectors is True
    assert relify.IVF(1).resolved_posting_encoding == "flat"
    assert relify.IVF(1, store_vectors=False).store_vectors is False
    assert relify.IVF(1, store_vectors=False).resolved_posting_encoding == "source"
    assert relify.IVF(1, posting_encoding="lvq4").resolved_posting_encoding == "lvq4"
    assert relify.IVF(1, posting_encoding="lvq8").resolved_posting_encoding == "lvq8"
    with pytest.raises(TypeError, match="store_vectors must be a boolean"):
        relify.IVF(1, store_vectors=cast(Any, 1))
    with pytest.raises(TypeError, match="posting_encoding must be a string"):
        relify.IVF(1, posting_encoding=cast(Any, 1))
    with pytest.raises(ValueError, match="unsupported posting_encoding"):
        relify.IVF(1, posting_encoding="pq")


def test_write_options_are_explicit_and_validated() -> None:
    options = relify.WriteOptions()
    assert options.partitions is None
    assert options.compression == "uncompressed"
    assert options.target_file_size == 512 * 1024 * 1024
    assert relify.Local().threads is None
    assert relify.Local(threads=4).threads == 4
    assert relify.Local().max_row_group_rows is None
    assert relify.Local().write_batch_rows == 8_192
    assert relify.WriteOptions(partitions=4).partitions == 4
    assert relify.WriteOptions(compression="zstd(3)").compression == "zstd(3)"

    with pytest.raises(TypeError, match="compression must be a string"):
        relify.WriteOptions(compression=cast(Any, 1))
    with pytest.raises(ValueError, match="unsupported Parquet compression"):
        relify.WriteOptions(compression=cast(Any, "invalid"))
    with pytest.raises(ValueError, match="unsupported Parquet compression"):
        relify.WriteOptions(compression="zstd")
    for field in ("partitions", "target_file_size"):
        with pytest.raises(ValueError, match=field):
            relify.WriteOptions(**{field: 0})
    for field in ("max_row_group_rows", "write_batch_rows"):
        with pytest.raises(ValueError, match=field):
            relify.Local(**{field: 0})


@pytest.mark.parametrize("threads", [True, 1.5, "1"])
def test_local_builder_requires_an_integer_thread_count(threads: object) -> None:
    with pytest.raises(TypeError, match="threads must be an integer"):
        relify.Local(threads=cast(Any, threads))


@pytest.mark.parametrize("threads", [0, -1])
def test_local_builder_requires_a_positive_thread_count(threads: int) -> None:
    with pytest.raises(ValueError, match="threads must be positive"):
        relify.Local(threads=threads)


@pytest.mark.parametrize(
    "compression",
    [
        "uncompressed",
        "snappy",
        "lz4",
        "lz4_raw",
        "gzip(0)",
        "gzip(9)",
        "brotli(0)",
        "brotli(11)",
        "zstd(1)",
        "zstd(22)",
    ],
)
def test_write_options_accept_supported_compression(
    compression: str,
) -> None:
    assert relify.WriteOptions(compression=compression).compression == compression


@pytest.mark.parametrize(
    "compression",
    [
        "gzip",
        "gzip(10)",
        "brotli",
        "brotli(12)",
        "zstd",
        "zstd(0)",
        "zstd(23)",
        "zstd(-1)",
    ],
)
def test_write_options_reject_invalid_compression_levels(
    compression: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported Parquet compression"):
        relify.WriteOptions(compression=compression)
