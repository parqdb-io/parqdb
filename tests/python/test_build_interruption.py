from __future__ import annotations

import os
import select
import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

import numpy as np
import parqdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process signals")
def test_process_exit_never_publishes_an_interrupted_build(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "parqdb-data"
    _write_large_source(source)

    session = parqdb.connect(root)
    session.register_parquet("vectors", source)
    vectors = session.table("vectors")
    vectors.create_index(
        "stable",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=1),
        wait_timeout=timedelta(minutes=2),
    )
    stable_snapshot = vectors.index_status("stable").current_snapshot_id
    session.close()

    _interrupt_build(root, "create", "interrupted")
    reopened = parqdb.connect(root)
    vectors = reopened.table("vectors")
    with pytest.raises(parqdb.IndexNotFoundError):
        vectors.index_status("interrupted")
    assert vectors.index_status("stable").current_snapshot_id == stable_snapshot
    reopened.close()

    _interrupt_build(root, "refresh", "stable")
    reopened = parqdb.connect(root)
    vectors = reopened.table("vectors")
    status = vectors.index_status("stable")
    assert status.state == "ready"
    assert status.current_snapshot_id == stable_snapshot
    assert status.error is None
    reopened.close()


def _interrupt_build(root: Path, operation: str, index: str) -> None:
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import time

        import parqdb

        root, operation, index = sys.argv[1:]
        config = parqdb.SessionConfig().set("parqdb.build.dop", "1")
        session = parqdb.connect(root, config=config)
        vectors = session.table("vectors")
        build_config = parqdb.IVF(nlist=4096)
        if operation == "create":
            vectors.create_index(
                index,
                column="embedding",
                key=["id"],
                config=build_config,
            )
        else:
            vectors.refresh_index(index, config=build_config)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = vectors.index_status(index)
            if status.state == "building":
                print("BUILDING", flush=True)
                os.kill(os.getpid(), signal.SIGSTOP)
            if status.state in {"ready", "failed"}:
                raise RuntimeError(f"build reached {status.state} before interruption")
            time.sleep(0.001)
        raise TimeoutError("build did not start")
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root), operation, index],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        readable, _, _ = select.select([process.stdout], [], [], 40)
        if not readable:
            raise AssertionError("build process did not report progress")
        assert process.stdout.readline().strip() == "BUILDING"
        _, status = os.waitpid(process.pid, os.WUNTRACED)
        assert os.WIFSTOPPED(status)
    finally:
        if process.poll() is None:
            process.kill()
        _, stderr = process.communicate(timeout=10)
    assert not stderr


def _write_large_source(path: Path) -> None:
    rows = 300_000
    dimensions = 32
    values = np.arange(rows * dimensions, dtype=np.float32)
    values %= 997
    values /= 997.0
    offsets = pa.array(
        np.arange(0, (rows + 1) * dimensions, dimensions, dtype=np.int64)
    )
    vectors = pa.ListArray.from_arrays(
        offsets,
        pa.array(values, type=pa.float32()),
    )
    table = pa.Table.from_arrays(
        [pa.array(np.arange(rows, dtype=np.int64)), vectors],
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field(
                    "embedding",
                    pa.list_(pa.field("element", pa.float32(), nullable=False)),
                    nullable=False,
                ),
            ]
        ),
    )
    pq.write_table(table, path, compression="none", row_group_size=8_192)
