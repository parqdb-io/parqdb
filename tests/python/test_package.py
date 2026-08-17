import json
import subprocess
import sys
from importlib.metadata import metadata, version
from pathlib import Path

import parqdb

REPOSITORY = Path(__file__).parents[2]


def test_package_version_matches_distribution_metadata() -> None:
    assert parqdb.__version__ == version("parqdb")


def test_distribution_declares_the_supported_python_range() -> None:
    assert metadata("parqdb")["Requires-Python"] == ">=3.11, <3.15"


def test_distribution_declares_all_included_licenses() -> None:
    distribution = metadata("parqdb")
    assert distribution["License-Expression"] == "MIT AND Apache-2.0"
    assert set(distribution.get_all("License-File", [])) == {
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "vendor/arrow-rs/parquet/LICENSE.txt",
        "vendor/datafusion-python/LICENSE.txt",
        "vendor/datafusion/datasource-parquet/LICENSE.txt",
    }


def test_committed_sbom_is_reproducible_and_path_independent(tmp_path: Path) -> None:
    generated = tmp_path / "parqdb-python.cyclonedx.json"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "tools" / "generate_sbom.py"),
            "--output",
            str(generated),
        ],
        cwd=REPOSITORY,
        check=True,
    )
    committed = REPOSITORY / "sboms" / "parqdb-python.cyclonedx.json"
    assert generated.read_bytes() == committed.read_bytes()
    document = json.loads(generated.read_bytes())
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.5"
    assert document["metadata"]["component"]["name"] == "parqdb"
    assert document["metadata"]["component"]["version"] == version("parqdb")
    assert (
        document["metadata"]["component"]["purl"]
        == f"pkg:pypi/parqdb@{version('parqdb')}"
    )
    assert "file://" not in generated.read_text(encoding="utf-8")
