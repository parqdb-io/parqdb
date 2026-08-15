from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import tomllib
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_record(archive: ZipFile, dist_info: str) -> None:
    files = {info.filename: info for info in archive.infolist() if not info.is_dir()}
    record_path = f"{dist_info}/RECORD"
    rows = {
        row[0]: row[1:]
        for row in csv.reader(
            io.TextIOWrapper(archive.open(record_path), encoding="utf-8")
        )
    }
    require(set(rows) == set(files), "wheel RECORD does not cover every file")
    for path, info in files.items():
        digest, size = rows[path]
        if path == record_path:
            require(not digest and not size, "RECORD must not hash itself")
            continue
        require(digest.startswith("sha256="), f"{path} does not use a SHA-256 hash")
        encoded = digest.removeprefix("sha256=")
        expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = archive.read(path)
        require(
            hashlib.sha256(payload).digest() == expected,
            f"{path} does not match its RECORD hash",
        )
        require(int(size) == info.file_size, f"{path} does not match its RECORD size")


def verify_wheel(repository: Path, wheel: Path) -> None:
    project = tomllib.loads(
        (repository / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    expected_name = project["name"]
    expected_version = project["version"]
    require(
        "-cp310-abi3-" in wheel.name,
        f"wheel is not tagged for the CPython 3.10 stable ABI: {wheel.name}",
    )
    expected_licenses = {
        relative: repository / relative for relative in project["license-files"]
    }
    expected_sbom = repository / "sboms" / "relify-python.cyclonedx.json"
    with ZipFile(wheel) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        for name in names:
            path = PurePosixPath(name)
            require(
                not path.is_absolute() and ".." not in path.parts,
                f"wheel contains unsafe path: {name}",
            )
            require(
                "__pycache__" not in path.parts and path.suffix != ".pyc",
                f"wheel contains Python cache output: {name}",
            )

        metadata_paths = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_paths) == 1, "wheel must contain one METADATA file")
        dist_info = metadata_paths[0].removesuffix("/METADATA")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        wheel_metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/WHEEL"))
        wheel_tags = wheel_metadata.get_all("Tag", [])
        require(
            wheel_tags and all(tag.startswith("cp310-abi3-") for tag in wheel_tags),
            f"unexpected wheel tags: {wheel_tags}",
        )
        require(metadata["Name"] == expected_name, "unexpected distribution name")
        require(
            metadata["Version"] == expected_version,
            "wheel version does not match pyproject.toml",
        )
        require(
            metadata["License-Expression"] == "MIT AND Apache-2.0",
            "unexpected wheel license expression",
        )
        require(
            metadata["Requires-Python"] == ">=3.11, <3.15",
            "unexpected Python compatibility range",
        )
        python_classifiers = {
            classifier
            for classifier in metadata.get_all("Classifier", [])
            if classifier.startswith("Programming Language :: Python :: 3.")
        }
        require(
            python_classifiers
            == {
                "Programming Language :: Python :: 3.11",
                "Programming Language :: Python :: 3.12",
                "Programming Language :: Python :: 3.13",
                "Programming Language :: Python :: 3.14",
            },
            "unexpected Python version classifiers",
        )
        require(
            set(metadata.get_all("License-File", [])) == set(expected_licenses),
            "wheel license file metadata is incomplete",
        )

        required_package_files = {
            "relify/_native.pyi",
            "relify/datasets/document_stats.parquet",
            "relify/datasets/documents.parquet",
            "relify/py.typed",
            "relify/datafusion/py.typed",
            "relify/datafusion/LICENSE.txt",
        }
        require(
            required_package_files <= names,
            "wheel is missing package typing or license files",
        )
        extensions = [
            name
            for name in names
            if name.startswith("relify/_native.") and not name.endswith(".pyi")
        ]
        require(len(extensions) == 1, "wheel must contain one native extension")

        for relative, source in expected_licenses.items():
            packaged = f"{dist_info}/licenses/{relative}"
            require(packaged in names, f"wheel is missing license file: {relative}")
            require(
                archive.read(packaged) == source.read_bytes(),
                f"packaged license differs from source: {relative}",
            )
        require(
            archive.read("relify/datafusion/LICENSE.txt")
            == expected_licenses["vendor/datafusion-python/LICENSE.txt"].read_bytes(),
            "package DataFusion license differs from its source",
        )

        sbom_path = f"{dist_info}/sboms/{expected_sbom.name}"
        require(sbom_path in names, "wheel is missing its CycloneDX SBOM")
        sbom_bytes = archive.read(sbom_path)
        require(
            sbom_bytes == expected_sbom.read_bytes(),
            "packaged SBOM differs from the committed SBOM",
        )
        sbom = json.loads(sbom_bytes)
        require(sbom["bomFormat"] == "CycloneDX", "unexpected SBOM format")
        require(sbom["specVersion"] == "1.5", "unexpected CycloneDX version")
        require(
            sbom["metadata"]["component"]["name"] == expected_name
            and sbom["metadata"]["component"]["version"] == expected_version,
            "SBOM distribution identity does not match pyproject.toml",
        )
        component_names = {component["name"] for component in sbom["components"]}
        require(
            {
                "datafusion",
                "datafusion-python",
                "object_store",
                "parquet",
                "pyo3",
                "relify-core",
            }
            <= component_names,
            "SBOM is missing required native components",
        )
        encoded_sbom = sbom_bytes.decode()
        require(
            "file://" not in encoded_sbom,
            "SBOM exposes a local filesystem location",
        )
        verify_record(archive, dist_info)
        print(
            f"verified {wheel.name}: {len(names)} files, "
            f"{len(sbom['components'])} SBOM components"
        )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("wheels", type=Path, nargs="+")
    command.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).parents[1],
    )
    return command


def main() -> None:
    args = parser().parse_args()
    repository = args.repository.resolve()
    for wheel in args.wheels:
        verify_wheel(repository, wheel.resolve())


if __name__ == "__main__":
    main()
