from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from support.config import HdfsConfig

pytestmark = pytest.mark.requires("hdfs")


def test_hdfs_warehouse_contract(hdfs: HdfsConfig) -> None:
    environment = os.environ.copy()
    if hdfs.uri is not None:
        environment["PARQDB_TEST_HDFS_URI"] = hdfs.uri
    subprocess.run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            "parqdb-storage",
            "--features",
            "hdfs-integration",
            "--test",
            "hdfs",
            "--",
            "--nocapture",
        ],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=True,
    )
