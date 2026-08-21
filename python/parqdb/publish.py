"""Publish a browser-queryable source table and static ParqDB index."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlparse

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

_MINILM_REPOSITORY = "Xenova/all-MiniLM-L6-v2"
_MINILM_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
_MINILM_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "onnx/model_quantized.onnx",
)
_MINILM_DIMENSION = 384
_MINILM_MAX_LENGTH = 256
_PARITY_TEXT = "ParqDB reads immutable Parquet indexes with HTTP Range requests."


@dataclass(frozen=True)
class PublicationResult:
    destination: str
    manifest_url: str | None
    files: int
    bytes: int


@dataclass(frozen=True)
class _Object:
    path: str
    source: Path | None = None
    content: bytes | None = None

    @property
    def size(self) -> int:
        return (
            self.source.stat().st_size
            if self.source is not None
            else len(self.content or b"")
        )


@dataclass(frozen=True)
class BuiltIndex:
    manifest: Path
    model_assets: tuple[tuple[str, Path], ...] = ()


def publish(
    *,
    index_manifest: Path,
    source: Path,
    source_key: str,
    destination: str,
    public_url: str | None = None,
    assets: tuple[tuple[str, Path], ...] = (),
    s3_endpoint: str | None = None,
    s3_region: str | None = None,
    verify_http: bool = True,
    cors_origin: str = "https://example.invalid",
) -> PublicationResult:
    """Publish all objects, making the static index manifest visible last."""
    index_manifest = index_manifest.resolve()
    source = source.resolve()
    manifest = _load_index_manifest(index_manifest)
    source_manifest = _source_manifest(source, source_key, manifest)
    objects = _publication_objects(index_manifest, source, source_manifest, assets)
    _ensure_unique_paths(objects)
    writer = _writer(destination, s3_endpoint=s3_endpoint, s3_region=s3_region)
    writer.ensure_empty(tuple(item.path for item in objects))
    try:
        for item in objects:
            writer.write(item)
    except Exception:
        writer.cleanup_after_failure()
        raise
    manifest_url = None
    if public_url is not None:
        manifest_url = f"{public_url.rstrip('/')}/index/manifest.json"
        if verify_http:
            verify_publication(public_url, objects, cors_origin=cors_origin)
    return PublicationResult(
        destination=destination,
        manifest_url=manifest_url,
        files=len(objects),
        bytes=sum(item.size for item in objects),
    )


def build_index(
    *,
    source: Path,
    source_key: str,
    work: Path,
    nlist: int,
    encoding: str,
    metric: str,
    threads: int,
    vector_column: str | None = None,
    text_columns: tuple[str, ...] = (),
    embedding_batch_size: int = 128,
) -> BuiltIndex:
    """Build a static index, optionally embedding source text with pinned MiniLM."""
    if (vector_column is None) == (not text_columns):
        raise ValueError("choose exactly one of --vector-column or --text-column")
    if not 1 <= threads <= 16:
        raise ValueError("threads must be in [1, 16]")
    source = source.resolve()
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    model_assets: tuple[tuple[str, Path], ...] = ()
    index_source = source
    column = vector_column
    if text_columns:
        model_root = work / "model"
        _download_minilm(model_root)
        index_source = work / "embedding-source.parquet"
        model_metadata = _embed_minilm(
            source,
            index_source,
            source_key=source_key,
            text_columns=text_columns,
            model_root=model_root,
            batch_size=embedding_batch_size,
            threads=threads,
        )
        metadata_path = work / "model.json"
        metadata_path.write_bytes(_json_bytes(model_metadata))
        model_assets = (
            ("model.json", metadata_path),
            *(
                (f"models/all-MiniLM-L6-v2/{path}", model_root / path)
                for path in _MINILM_FILES
            ),
        )
        column = "embedding"
    assert column is not None
    package = _build_parqdb_index(
        index_source,
        source_key=source_key,
        vector_column=column,
        warehouse=work / "parqdb",
        nlist=nlist,
        encoding=encoding,
        metric=metric,
        threads=threads,
    )
    return BuiltIndex(package / "manifest.json", model_assets)


def verify_publication(
    public_url: str,
    objects: list[_Object],
    *,
    cors_origin: str,
) -> None:
    """Verify discovery plus byte-range and browser CORS behavior."""
    root = f"{public_url.rstrip('/')}/"
    paths = {item.path: item for item in objects}
    candidates = [
        item for item in objects if item.path.endswith(".parquet") and item.size > 1
    ]
    if not candidates:
        raise ValueError("publication has no non-empty Parquet object to range-check")
    candidate = max(candidates, key=lambda item: item.size)
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        response = client.get(f"{root}index/manifest.json")
        response.raise_for_status()
        if response.content != paths["index/manifest.json"].content:
            raise RuntimeError(
                "public index manifest bytes differ from the publication"
            )
        ranged = client.get(
            f"{root}{candidate.path}",
            headers={"Range": "bytes=0-0", "Origin": cors_origin},
        )
        if ranged.status_code != 206:
            raise RuntimeError(
                f"public host returned HTTP {ranged.status_code} for a byte range"
            )
        expected = f"bytes 0-0/{candidate.size}"
        if ranged.headers.get("content-range") != expected or len(ranged.content) != 1:
            raise RuntimeError(f"invalid Content-Range; expected {expected}")
        allow_origin = ranged.headers.get("access-control-allow-origin")
        if allow_origin not in {"*", cors_origin}:
            raise RuntimeError("public host does not allow browser CORS reads")


def _load_index_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"index manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid index manifest JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format-version") != 1:
        raise ValueError("index manifest must be a static package format version 1")
    if not isinstance(manifest.get("package-uuid"), str):
        raise ValueError("index manifest is missing package-uuid")
    return manifest


def _source_manifest(
    source: Path, source_key: str, index_manifest: dict[str, object]
) -> dict[str, object]:
    if not source.is_file():
        raise ValueError(f"source Parquet file does not exist: {source}")
    parquet = pq.ParquetFile(source)
    field = parquet.schema_arrow.field(source_key)
    if not pa.types.is_int64(field.type):
        raise ValueError("source key must be a non-null int64 column")
    if field.nullable:
        raise ValueError("source key must be non-nullable")
    rows = parquet.metadata.num_rows
    if rows <= 0:
        raise ValueError("source Parquet file must contain at least one row")
    _validate_source_key(parquet, source_key)
    descriptor = index_manifest.get("index")
    if not isinstance(descriptor, dict) or descriptor.get("ntotal") != rows:
        raise ValueError("source rows do not match index ntotal")
    if descriptor.get("source-key-fields") != [{"name": source_key, "type": "long"}]:
        raise ValueError("source key does not match the index source-key-fields")
    row_group_rows = parquet.metadata.row_group(0).num_rows
    for index in range(max(0, parquet.metadata.num_row_groups - 1)):
        if parquet.metadata.row_group(index).num_rows != row_group_rows:
            raise ValueError(
                "all source row groups except the final group must be uniform"
            )
    return {
        "format-version": 1,
        "rows": rows,
        "row-group-rows": row_group_rows,
        "object": {
            "path": source.name,
            "size": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "key": {"name": source_key, "type": "long"},
        "columns": parquet.schema_arrow.names,
    }


def _validate_source_key(parquet: pq.ParquetFile, source_key: str) -> None:
    expected = 0
    for batch in parquet.iter_batches(batch_size=65_536, columns=[source_key]):
        values = batch.column(0).to_pylist()
        for value in values:
            if value != expected:
                raise ValueError(
                    f"source key must be dense and ordered from zero; expected {expected}, got {value}"
                )
            expected += 1


def _download_minilm(root: Path) -> None:
    for relative in _MINILM_FILES:
        destination = root / relative
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{_MINILM_REPOSITORY}/resolve/"
            f"{_MINILM_REVISION}/{relative}"
        )
        temporary = destination.with_suffix(f"{destination.suffix}.part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)


def _embed_minilm(
    source: Path,
    output: Path,
    *,
    source_key: str,
    text_columns: tuple[str, ...],
    model_root: Path,
    batch_size: int,
    threads: int,
) -> dict[str, object]:
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "text embedding requires the optional dependencies: "
            "python -m pip install 'parqdb[publish]'"
        ) from error
    parquet = pq.ParquetFile(source)
    for column in text_columns:
        if not pa.types.is_string(parquet.schema_arrow.field(column).type):
            raise ValueError(f"text column must be a string: {column}")
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        model_root / "onnx/model_quantized.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    def encode(texts: list[str]) -> Any:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_MINILM_MAX_LENGTH,
            return_tensors="np",
        )
        input_names = {item.name for item in session.get_inputs()}
        feeds = {
            name: np.asarray(encoded[name], dtype=np.int64)
            for name in input_names
            if name in encoded
        }
        hidden = np.asarray(session.run(None, feeds)[0], dtype=np.float32)
        mask = np.asarray(encoded["attention_mask"], dtype=np.float32)[..., None]
        vectors = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return vectors

    schema = pa.schema(
        [
            pa.field(source_key, pa.int64(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(
                    pa.field("element", pa.float32(), nullable=False), _MINILM_DIMENSION
                ),
                nullable=False,
            ),
        ]
    )
    if not output.exists():
        writer = pq.ParquetWriter(
            output, schema, compression="zstd", compression_level=3
        )
        try:
            columns = [source_key, *text_columns]
            for source_batch in parquet.iter_batches(
                batch_size=batch_size, columns=columns
            ):
                rows = source_batch.to_pylist()
                texts = [
                    "\n\n".join(str(row[column]) for column in text_columns)
                    for row in rows
                ]
                vectors = encode(texts)
                if (
                    vectors.shape[1] != _MINILM_DIMENSION
                    or not np.isfinite(vectors).all()
                ):
                    raise RuntimeError("embedding model returned an invalid matrix")
                flat = pa.array(vectors.reshape(-1), type=pa.float32())
                embeddings = pa.FixedSizeListArray.from_arrays(
                    flat, type=schema.field("embedding").type
                )
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        [source_batch.column(0), embeddings], schema=schema
                    ),
                    row_group_size=8_192,
                )
        finally:
            writer.close()
    probe = encode([_PARITY_TEXT])[0]
    return {
        "repository": _MINILM_REPOSITORY,
        "revision": _MINILM_REVISION,
        "runtime": "onnx",
        "onnx-file": "onnx/model_quantized.onnx",
        "onnx-sha256": _sha256(model_root / "onnx/model_quantized.onnx"),
        "dimension": _MINILM_DIMENSION,
        "max-length": _MINILM_MAX_LENGTH,
        "pooling": "attention-mask-mean",
        "normalize": True,
        "input-template": "\n\n".join(f"{{{column}}}" for column in text_columns),
        "parity-probe": {
            "text": _PARITY_TEXT,
            "vector": [round(float(value), 8) for value in probe],
            "max-absolute-error": 0.002,
        },
    }


def _build_parqdb_index(
    source: Path,
    *,
    source_key: str,
    vector_column: str,
    warehouse: Path,
    nlist: int,
    encoding: str,
    metric: str,
    threads: int,
) -> Path:
    from .api import IVF, SessionConfig, WriteOptions, connect

    existing = _find_static_package(warehouse)
    if existing is not None:
        return existing
    config = (
        SessionConfig()
        .with_target_partitions(threads)
        .set("parqdb.build.dop", str(threads))
    )
    session = connect(warehouse, config=config)
    try:
        session.register_parquet("publication_source", source)
        table = session.table("publication_source")
        table.create_index(
            "publication_index",
            column=vector_column,
            key=[source_key],
            config=IVF(nlist=nlist, encoding=encoding, metric=metric),
            writer_options=WriteOptions(
                partitions=threads,
                compression="zstd(3)",
                target_file_size=64 * 1024 * 1024,
                write_batch_rows=8_192,
            ),
            wait_timeout=timedelta(hours=24),
        )
        table.wait_for_index("publication_index", timeout=timedelta(hours=24))
    finally:
        session.close()
    package = _find_static_package(warehouse)
    if package is None:
        raise RuntimeError("index build completed without a static manifest.json")
    return package


def _find_static_package(root: Path) -> Path | None:
    if not root.exists():
        return None
    for path in sorted(root.rglob("manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("format-version") == 1 and "package-uuid" in value:
            return path.parent
    return None


def _publication_objects(
    index_manifest_path: Path,
    source: Path,
    source_manifest: dict[str, object],
    assets: tuple[tuple[str, Path], ...],
) -> list[_Object]:
    manifest = _load_index_manifest(index_manifest_path)
    root = index_manifest_path.parent
    _validate_index_objects(manifest, root)
    referenced = _referenced_index_paths(manifest)
    objects = [_Object(f"index/{path}", source=root / path) for path in referenced]
    native_manifest = root / "ivf_postings" / "manifest.json"
    if native_manifest.is_file():
        objects.append(
            _Object("index/ivf_postings/manifest.json", source=native_manifest)
        )
    objects.append(_Object(source.name, source=source))
    for relative, asset in assets:
        relative = _safe_relative_path(relative)
        asset = asset.resolve()
        if not asset.is_file():
            raise ValueError(f"asset does not exist: {asset}")
        objects.append(_Object(relative, source=asset))
    source_bytes = _json_bytes(source_manifest)
    objects.append(_Object("source-manifest.json", content=source_bytes))
    # The entry-point index manifest is the commit marker and is always written last.
    objects.append(
        _Object("index/manifest.json", content=index_manifest_path.read_bytes())
    )
    for item in objects:
        if item.source is not None and not item.source.is_file():
            raise ValueError(f"index object does not exist: {item.source}")
    return objects


def _referenced_index_paths(manifest: dict[str, object]) -> list[str]:
    hierarchy = manifest.get("hierarchy")
    postings = manifest.get("postings")
    if not isinstance(hierarchy, dict) or not isinstance(postings, dict):
        raise ValueError("index manifest is missing hierarchy or postings")
    values = [hierarchy.get("roots"), hierarchy.get("centroids")]
    files = postings.get("files")
    if not isinstance(files, list):
        raise ValueError("index manifest postings.files must be an array")
    values.extend(files)
    paths: list[str] = []
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError("index object descriptor is missing a path")
        paths.append(_safe_relative_path(value["path"]))
    return paths


def _validate_index_objects(manifest: dict[str, object], root: Path) -> None:
    hierarchy = manifest["hierarchy"]
    postings = manifest["postings"]
    assert isinstance(hierarchy, dict) and isinstance(postings, dict)
    descriptors = [hierarchy["roots"], hierarchy["centroids"], *postings["files"]]
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("index object descriptor must be an object")
        relative = _safe_relative_path(str(descriptor.get("path", "")))
        path = root / relative
        if not path.is_file():
            raise ValueError(f"index object does not exist: {path}")
        if descriptor.get("size") != path.stat().st_size:
            raise ValueError(f"index object size does not match manifest: {relative}")
        if descriptor.get("sha256") != _sha256(path):
            raise ValueError(
                f"index object SHA-256 does not match manifest: {relative}"
            )


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"unsafe publication path: {value!r}")
    return value


def _ensure_unique_paths(objects: list[_Object]) -> None:
    seen: set[str] = set()
    for item in objects:
        if item.path in seen:
            raise ValueError(f"duplicate publication path: {item.path}")
        seen.add(item.path)


class _Writer:
    def ensure_empty(self, paths: tuple[str, ...]) -> None:
        raise NotImplementedError

    def write(self, item: _Object) -> None:
        raise NotImplementedError

    def cleanup_after_failure(self) -> None:
        pass


class _LocalWriter(_Writer):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.created = False

    def ensure_empty(self, paths: tuple[str, ...]) -> None:
        if self.root.exists():
            raise FileExistsError(f"destination already exists: {self.root}")
        self.root.mkdir(parents=True)
        self.created = True

    def write(self, item: _Object) -> None:
        destination = self.root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if item.source is not None:
            shutil.copyfile(item.source, destination)
        else:
            destination.write_bytes(item.content or b"")

    def cleanup_after_failure(self) -> None:
        if self.created:
            shutil.rmtree(self.root, ignore_errors=True)


class _S3Writer(_Writer):
    def __init__(self, filesystem: fs.S3FileSystem, prefix: str) -> None:
        self.filesystem = filesystem
        self.prefix = prefix.rstrip("/")

    def _path(self, relative: str) -> str:
        return f"{self.prefix}/{relative}"

    def ensure_empty(self, paths: tuple[str, ...]) -> None:
        for relative in paths:
            if (
                self.filesystem.get_file_info(self._path(relative)).type
                != fs.FileType.NotFound
            ):
                raise FileExistsError(f"destination object already exists: {relative}")

    def write(self, item: _Object) -> None:
        with self.filesystem.open_output_stream(self._path(item.path)) as output:
            if item.source is None:
                output.write(item.content or b"")
            else:
                with item.source.open("rb") as source:
                    _copy_stream(source, output)


def _writer(
    destination: str, *, s3_endpoint: str | None, s3_region: str | None
) -> _Writer:
    parsed = urlparse(destination)
    if parsed.scheme == "s3":
        if not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(
                "S3 destination must include a bucket and immutable prefix"
            )
        endpoint = urlparse(s3_endpoint) if s3_endpoint is not None else None
        if endpoint is not None and (
            endpoint.scheme not in {"http", "https"} or not endpoint.netloc
        ):
            raise ValueError("--s3-endpoint must be an HTTP(S) URL")
        filesystem = fs.S3FileSystem(
            region=s3_region,
            scheme=endpoint.scheme if endpoint is not None else None,
            endpoint_override=endpoint.netloc if endpoint is not None else None,
            background_writes=False,
        )
        return _S3Writer(filesystem, f"{parsed.netloc}/{parsed.path.strip('/')}")
    if parsed.scheme == "file":
        return _LocalWriter(Path(parsed.path).resolve())
    if parsed.scheme:
        raise ValueError("destination must be a local path, file URI, or s3:// URI")
    return _LocalWriter(Path(destination).resolve())


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while chunk := source.read(8 * 1024 * 1024):
        destination.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def parse_asset(value: str) -> tuple[str, Path]:
    """Parse a CLI ``RELATIVE_PATH=LOCAL_PATH`` asset declaration."""
    relative, separator, local = value.partition("=")
    if not separator or not local:
        raise ValueError("asset must use RELATIVE_PATH=LOCAL_PATH")
    return _safe_relative_path(relative), Path(os.path.expanduser(local))
