#!/usr/bin/env python3
"""Synchronize Relify's embedded DataFusion Python bindings.

The generated Rust crate and Python package are committed to the repository so
building Relify never depends on an available network connection. Do not edit
those generated files directly; update this script or the upstream version and
run it again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "vendor" / "datafusion-python"
PYTHON_DIR = ROOT / "python" / "relify" / "datafusion"

CRATE_URL = "https://crates.io/api/v1/crates/datafusion-python/{version}/download"
CRATE_JSON_URL = "https://crates.io/api/v1/crates/datafusion-python/{version}"
PYPI_JSON_URL = "https://pypi.org/pypi/datafusion/{version}/json"

RUST_INITIALIZER = """/// Low-level DataFusion internal package.
///
/// The higher-level public API is defined in pure python files under the
/// datafusion directory.
#[pymodule]
fn _internal(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {"""
EMBEDDED_RUST_INITIALIZER = """/// Initialize the low-level bindings in an embedding Python module.
///
/// Relify calls this function to install the complete DataFusion Python API in
/// the same extension module as its Rust-owned `SessionContext`.
pub fn init_internal_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {"""
RUST_WRAPPER_ANCHOR = """#[cfg(feature = "substrait")]
fn setup_substrait_module"""
RUST_WRAPPER = """#[pymodule]
fn _internal(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    init_internal_module(py, &m)
}

#[cfg(feature = "substrait")]
fn setup_substrait_module"""
RUST_IMPORT_REWRITES = (
    ('py.import("datafusion', 'py.import("relify.datafusion'),
    ('py.import_bound("datafusion', 'py.import_bound("relify.datafusion'),
    ('PyModule::import(py, "datafusion', 'PyModule::import(py, "relify.datafusion'),
    (
        'PyModule::import_bound(py, "datafusion',
        'PyModule::import_bound(py, "relify.datafusion',
    ),
)
RUST_IMPORT_DOC_REWRITES = (
    (
        "the datafusion.dataframe_formatter module",
        "the relify.datafusion.dataframe_formatter module",
    ),
)

VERSION_IMPORT = """try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore[import]

"""
INTERNAL_IMPORT_ANCHOR = """from typing import Any

# Public submodules
"""
EMBEDDED_INTERNAL_IMPORT = """from typing import Any

from . import _internal as _internal

# Public submodules
"""


def download(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "relify-datafusion-sync"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                contents = response.read()
            return contents, hashlib.sha256(contents).hexdigest()
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def extract_archive(contents: bytes, destination: Path) -> Path:
    destination.mkdir(parents=True)
    archive = destination / "archive.tar.gz"
    archive.write_bytes(contents)
    with tarfile.open(archive, mode="r:gz") as source:
        source.extractall(destination, filter="data")
    roots = sorted(
        entry
        for entry in destination.iterdir()
        if entry.is_dir() and entry.name != "__MACOSX"
    )
    if len(roots) != 1:
        raise RuntimeError(f"expected one archive root, found {roots}")
    return roots[0]


def crate_archive(version: str) -> tuple[bytes, str]:
    metadata_bytes, _ = download(CRATE_JSON_URL.format(version=version))
    metadata = json.loads(metadata_bytes)
    expected = metadata["version"]["checksum"]
    contents, digest = download(CRATE_URL.format(version=version))
    if digest != expected:
        raise RuntimeError(
            f"crates.io checksum mismatch: expected {expected}, got {digest}"
        )
    return contents, digest


def pypi_sdist(version: str) -> tuple[bytes, str, str]:
    metadata_bytes, _ = download(PYPI_JSON_URL.format(version=version))
    metadata = json.loads(metadata_bytes)
    sources = [entry for entry in metadata["urls"] if entry["packagetype"] == "sdist"]
    if len(sources) != 1:
        raise RuntimeError(f"expected one PyPI sdist, found {len(sources)}")
    source = sources[0]
    contents, digest = download(source["url"])
    expected = source["digests"]["sha256"]
    if digest != expected:
        raise RuntimeError(f"PyPI checksum mismatch: expected {expected}, got {digest}")
    return contents, source["url"], digest


def patch_rust_initializer(crate_root: Path) -> None:
    lib_rs = crate_root / "src" / "lib.rs"
    source = lib_rs.read_text()
    if source.count(RUST_INITIALIZER) != 1:
        raise RuntimeError("DataFusion module initializer changed upstream")
    source = source.replace(RUST_INITIALIZER, EMBEDDED_RUST_INITIALIZER)
    source = source.replace(
        "setup_substrait_module(py, &m)?;",
        "setup_substrait_module(py, m)?;",
        1,
    )
    if source.count(RUST_WRAPPER_ANCHOR) != 1:
        raise RuntimeError("DataFusion substrait initializer changed upstream")
    source = source.replace(RUST_WRAPPER_ANCHOR, RUST_WRAPPER, 1)
    lib_rs.write_text(source)


def rewrite_rust_imports(crate_root: Path) -> None:
    replacements = 0
    for path in sorted(crate_root.rglob("*.rs")):
        source = path.read_text()
        for original, embedded in RUST_IMPORT_REWRITES:
            replacements += source.count(original)
            source = source.replace(original, embedded)
        for original, embedded in RUST_IMPORT_DOC_REWRITES:
            source = source.replace(original, embedded)
        path.write_text(source)

    if replacements == 0:
        raise RuntimeError("DataFusion has no runtime module imports to relocate")

    remaining = []
    for path in sorted(crate_root.rglob("*.rs")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if any(original in line for original, _ in RUST_IMPORT_REWRITES):
                remaining.append(f"{path}:{line_number}: {line}")
    if remaining:
        raise RuntimeError(
            "unconverted runtime DataFusion imports:\n" + "\n".join(remaining)
        )


def rewrite_python_imports(package_root: Path, version: str) -> None:
    import_patterns = (
        (
            re.compile(r"^(\s*)from datafusion(?=\.|\s)", re.MULTILINE),
            r"\1from relify.datafusion",
        ),
        (
            re.compile(r"^(\s*)import datafusion(?=\.|\s)", re.MULTILINE),
            r"\1import relify.datafusion",
        ),
    )
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text()
        for pattern, replacement in import_patterns:
            source = pattern.sub(replacement, source)
        source = source.replace("from datafusion", "from relify.datafusion")
        source = source.replace("import datafusion", "import relify.datafusion")
        path.write_text(source)

    init_py = package_root / "__init__.py"
    source = init_py.read_text()
    if VERSION_IMPORT not in source:
        raise RuntimeError("DataFusion version import changed upstream")
    source = source.replace(VERSION_IMPORT, "", 1)
    version_assignment = "__version__ = importlib_metadata.version(__name__)"
    if source.count(version_assignment) != 1:
        raise RuntimeError("DataFusion version assignment changed upstream")
    source = source.replace(version_assignment, f'__version__ = "{version}"', 1)
    if source.count(INTERNAL_IMPORT_ANCHOR) != 1:
        raise RuntimeError("DataFusion public module imports changed upstream")
    source = source.replace(
        INTERNAL_IMPORT_ANCHOR,
        EMBEDDED_INTERNAL_IMPORT,
        1,
    )
    init_py.write_text(source)

    remaining = []
    for path in sorted(package_root.rglob("*.py")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith(("from datafusion", "import datafusion")):
                remaining.append(f"{path}:{line_number}: {line}")
    if remaining:
        raise RuntimeError(
            "unconverted absolute DataFusion imports:\n" + "\n".join(remaining)
        )


def replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def synchronize(
    version: str,
    vendor_dir: Path = VENDOR_DIR,
    python_dir: Path = PYTHON_DIR,
) -> None:
    with tempfile.TemporaryDirectory(prefix="relify-datafusion-") as temp_name:
        temp = Path(temp_name)

        crate_contents, crate_digest = crate_archive(version)
        crate_root = extract_archive(crate_contents, temp / "crate")
        for generated_file in (".cargo-ok", ".cargo_vcs_info.json", "Cargo.lock"):
            (crate_root / generated_file).unlink(missing_ok=True)
        patch_rust_initializer(crate_root)
        rewrite_rust_imports(crate_root)

        python_contents, python_url, python_digest = pypi_sdist(version)
        source_root = extract_archive(python_contents, temp / "python")
        python_source = source_root / "python" / "datafusion"
        if not python_source.is_dir():
            raise RuntimeError(f"missing Python package in {source_root}")

        replace_directory(crate_root, vendor_dir)
        shutil.copy2(source_root / "LICENSE.txt", vendor_dir / "LICENSE.txt")
        rewrite_python_imports(python_source, version)
        shutil.copy2(source_root / "LICENSE.txt", python_source / "LICENSE.txt")
        replace_directory(python_source, python_dir)

        manifest = (
            "# Generated by tools/sync_datafusion.py.\n"
            f'version = "{version}"\n'
            f'crate_url = "{CRATE_URL.format(version=version)}"\n'
            f'crate_sha256 = "{crate_digest}"\n'
            f'pypi_url = "{python_url}"\n'
            f'pypi_sha256 = "{python_digest}"\n'
            'patch = "export init_internal_module and relocate runtime imports"\n'
        )
        (vendor_dir / "UPSTREAM.toml").write_text(manifest)
        (vendor_dir / "RELIFY_PATCH.md").write_text(
            "# Relify patch\n\n"
            "The vendored crate exposes its private `_internal` module "
            "initializer as `init_internal_module` and redirects runtime imports "
            "from `datafusion.*` to `relify.datafusion.*`. Relify "
            "uses the initializer to install the complete binding surface inside "
            "`relify._native`, ensuring that Python and Rust share one "
            "`SessionContext` implementation.\n\n"
            "## Upgrade\n\n"
            "1. Update the DataFusion version pins in the workspace manifest.\n"
            "2. Run `python tools/sync_datafusion.py <version>`.\n"
            "3. Run `python tools/sync_datafusion.py <version> --check`.\n"
            "4. Review the generated diff and run the full project checks.\n\n"
            "The synchronizer verifies the upstream checksums and fails if its "
            "Rust embedding patches no longer apply exactly. Generated files must "
            "not be edited by hand. This vendor boundary can be removed when "
            "upstream DataFusion exposes a stable `SessionContext` FFI that "
            "supports the same one-context model.\n"
        )


def directory_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
        ):
            continue
        manifest[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def check_synchronized(version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="relify-datafusion-check-") as temp_name:
        generated = Path(temp_name)
        generated_vendor = generated / "vendor"
        generated_python = generated / "python"
        synchronize(version, generated_vendor, generated_python)

        differences = []
        for label, expected_root, actual_root in (
            ("vendor", generated_vendor, VENDOR_DIR),
            ("python", generated_python, PYTHON_DIR),
        ):
            expected = directory_manifest(expected_root)
            actual = directory_manifest(actual_root)
            for path in sorted(expected.keys() | actual.keys()):
                if expected.get(path) != actual.get(path):
                    differences.append(f"{label}/{path}")
        if differences:
            rendered = "\n".join(f"  {path}" for path in differences)
            raise RuntimeError(
                "vendored DataFusion bindings are not synchronized:\n" + rendered
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="DataFusion Python release to embed")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed bindings without modifying the repository",
    )
    args = parser.parse_args()
    if args.check:
        check_synchronized(args.version)
    else:
        synchronize(args.version)


if __name__ == "__main__":
    main()
