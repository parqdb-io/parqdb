from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).parent
SNAPSHOT_ID = 901
FINGERPRINT_NAMESPACE = uuid.UUID("2fb71e63-a27c-4fc5-9d6d-5070698dc398")
INDEX_UUIDS = {
    "lvq4": "ee577329-84db-40da-af50-bb10d86e2d2f",
    "lvq8": "26878cae-d125-4ec9-b42f-7f0b1ed8c64f",
}
IVF_CENTROIDS_UUIDS = {
    "lvq4": "ac413538-8613-4ed5-8411-a9579eda38da",
    "lvq8": "269b3fe5-0fb3-48f9-85cc-78863622bb48",
}
PACKAGE_UUIDS = {
    "lvq4": "849bc22f-7f83-5faf-a2c9-6becd8f6482c",
    "lvq8": "24691ca8-c58e-51d4-bbed-c78ba6fedb62",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fingerprint(descriptor: dict[str, object]) -> str:
    canonical_descriptor = {
        "vector-field": descriptor["vector-field"],
        "dimension": descriptor["dimension"],
        "metric": descriptor["metric"],
        "nlist": descriptor["nlist"],
        "clustering-profile-version": descriptor["clustering-profile-version"],
    }
    canonical = json.dumps(
        canonical_descriptor, ensure_ascii=False, separators=(",", ":")
    )
    return str(uuid.uuid5(FINGERPRINT_NAMESPACE, canonical))


def encode(vector: list[float], bits: int) -> tuple[float, float, bytes]:
    offset = min(vector)
    upper = max(vector)
    levels = (1 << bits) - 1
    scale = (upper - offset) / levels
    if upper == offset:
        codes = [0] * len(vector)
    else:
        codes = [
            min(
                levels,
                max(
                    0,
                    math.floor(levels * (value - offset) / (upper - offset) + 0.5),
                ),
            )
            for value in vector
        ]
    if bits == 8:
        packed = bytes(codes)
    else:
        packed = bytes(
            codes[index] | (codes[index + 1] << 4 if index + 1 < len(codes) else 0)
            for index in range(0, len(codes), 2)
        )
    return offset, scale, packed


def descriptor(encoding: str) -> dict[str, object]:
    _ = encoding
    return {
        "vector-field": "embedding",
        "dimension": 3,
        "metric": "l2_squared",
        "nlist": 2,
        "clustering-profile-version": 1,
    }


def metadata(encoding: str) -> dict[str, object]:
    index_uuid = INDEX_UUIDS[encoding]
    centroid_uuid = IVF_CENTROIDS_UUIDS[encoding]
    centroid_descriptor = descriptor(encoding)
    return {
        "format-version": 1,
        "index-uuid": index_uuid,
        "last-updated-ms": 1_750_000_000_000,
        "last-sequence-number": 1,
        "current-snapshot-id": SNAPSHOT_ID,
        "snapshots": [
            {
                "snapshot-id": SNAPSHOT_ID,
                "sequence-number": 1,
                "timestamp-ms": 1_750_000_000_000,
                "summary": {"operation": "create"},
                "vector-field": "embedding",
                "source-key-fields": ["document_id"],
                "indexed-rows": 3,
                "index-family": "ivf",
                "index-schema-version": 1,
                "metric": "l2_squared",
                "parameters": {
                    "dimension": "3",
                    "nlist": "2",
                    "ntotal": "3",
                    "posting_encoding": encoding,
                    "ivf_centroids_fingerprint": fingerprint(centroid_descriptor),
                    "ivf_centroids_uuid": centroid_uuid,
                    "ivf_centroids_metadata_location": "ivf-centroids.metadata.json",
                },
                "index-relations": {
                    "ivf_centroids": "ivf_centroids.parquet",
                    "ivf_postings": "ivf_postings/",
                },
            }
        ],
        "snapshot-log": [
            {
                "timestamp-ms": 1_750_000_000_000,
                "snapshot-id": SNAPSHOT_ID,
            }
        ],
        "properties": {"fixture": f"ivf-parquet-{encoding}"},
    }


def ivf_centroids_metadata(encoding: str) -> dict[str, object]:
    centroid_descriptor = descriptor(encoding)
    centroid_uuid = IVF_CENTROIDS_UUIDS[encoding]
    return {
        "format-version": 1,
        "artifact-uuid": centroid_uuid,
        "fingerprint": fingerprint(centroid_descriptor),
        "created-at-ms": 1_750_000_000_000,
        "descriptor": centroid_descriptor,
        "centroids": "ivf_centroids.parquet",
        "roots": "ivf_roots.parquet",
    }


def write_fixture(encoding: str) -> None:
    directory = ROOT / "valid" / encoding
    directory.mkdir(parents=True, exist_ok=True)
    postings_root = directory / "ivf_postings"
    shutil.rmtree(postings_root, ignore_errors=True)

    vectors = [[0.0, 0.5, 1.0], [2.0, 2.0, 2.0], [10.0, 11.0, 12.0]]
    vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False))
    source = pa.Table.from_arrays(
        [
            pa.array(["a", "b", "c"], type=pa.string()),
            pa.array(vectors, type=vector_type),
        ],
        schema=pa.schema(
            [
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("embedding", vector_type, nullable=False),
            ]
        ),
    )
    centroid_values = [[1.0, 1.0, 1.0], [10.0, 11.0, 12.0]]
    encoded_centroids = [encode(vector, 8) for vector in centroid_values]
    centroids = pa.Table.from_arrays(
        [
            pa.array([0, 1], type=pa.int32()),
            pa.array([0, 0], type=pa.int32()),
            pa.array([row[0] for row in encoded_centroids], type=pa.float32()),
            pa.array([row[1] for row in encoded_centroids], type=pa.float32()),
            pa.array([row[2] for row in encoded_centroids], type=pa.binary()),
        ],
        schema=pa.schema(
            [
                pa.field("cid", pa.int32(), nullable=False),
                pa.field("cid_bucket", pa.int32(), nullable=False),
                pa.field("offset", pa.float32(), nullable=False),
                pa.field("scale", pa.float32(), nullable=False),
                pa.field("code", pa.binary(), nullable=False),
            ]
        ),
    )
    roots = pa.Table.from_arrays(
        [
            pa.array([0], type=pa.int32()),
            pa.array([0], type=pa.int32()),
            pa.array([2], type=pa.int32()),
            pa.array([[5.5, 6.0, 6.5]], type=vector_type),
        ],
        schema=pa.schema(
            [
                pa.field("cid_bucket", pa.int32(), nullable=False),
                pa.field("cid_begin", pa.int32(), nullable=False),
                pa.field("cid_end", pa.int32(), nullable=False),
                pa.field("centroid", vector_type, nullable=False),
            ]
        ),
    )
    bits = 4 if encoding == "lvq4" else 8
    encoded = [encode(vector, bits) for vector in vectors]
    postings = pa.Table.from_arrays(
        [
            pa.array([0, 0, 1], type=pa.int32()),
            pa.array(["a", "b", "c"], type=pa.string()),
            pa.array([row[0] for row in encoded], type=pa.float32()),
            pa.array([row[1] for row in encoded], type=pa.float32()),
            pa.array([row[2] for row in encoded], type=pa.binary()),
        ],
        schema=pa.schema(
            [
                pa.field("cid", pa.int32(), nullable=False),
                pa.field("key_1", pa.string(), nullable=False),
                pa.field("offset", pa.float32(), nullable=False),
                pa.field("scale", pa.float32(), nullable=False),
                pa.field("code", pa.binary(), nullable=False),
            ]
        ),
    )

    pq.write_table(source, directory / "source.parquet", compression="NONE")
    pq.write_table(centroids, directory / "ivf_centroids.parquet", compression="NONE")
    pq.write_table(roots, directory / "ivf_roots.parquet", compression="NONE")
    pq.write_table(centroids, directory / "centroids.parquet", compression="NONE")
    pq.write_table(roots, directory / "roots.parquet", compression="NONE")
    bucket = postings_root / "cid_bucket=000000"
    bucket.mkdir(parents=True)
    path = bucket / "part-00000.parquet"
    writer = pq.ParquetWriter(path, postings.schema, compression="NONE")
    for cid in range(2):
        rows = postings.filter(pc.equal(postings["cid"], cid))
        if rows.num_rows:
            writer.write_table(rows, row_group_size=rows.num_rows)
    writer.close()
    write_json(
        postings_root / "manifest.json",
        {
            "format-version": 1,
            "nlist": 2,
            "ntotal": postings.num_rows,
            "cid-offsets": [0, 2],
            "files": [
                {
                    "path": "cid_bucket=000000/part-00000.parquet",
                    "cid-bucket": 0,
                    "min-cid": 0,
                    "max-cid": 1,
                    "rows": postings.num_rows,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    static_files = [
        {
            "path": "ivf_postings/cid_bucket=000000/part-00000.parquet",
            "cid-bucket": 0,
            "min-cid": 0,
            "max-cid": 1,
            "rows": postings.num_rows,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]

    def static_object(name: str) -> dict[str, object]:
        object_path = directory / name
        return {
            "path": name,
            "size": object_path.stat().st_size,
            "sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
        }

    write_json(
        directory / "manifest.json",
        {
            "format-version": 1,
            "package-uuid": PACKAGE_UUIDS[encoding],
            "index": {
                "metric": "l2_squared",
                "posting-encoding": encoding,
                "dimension": 3,
                "nlist": 2,
                "ntotal": postings.num_rows,
                "source-key-fields": [{"name": "document_id", "type": "string"}],
            },
            "hierarchy": {
                "root-count": 1,
                "cid-offsets": [0, 2],
                "centroid-encoding": "lvq8",
                "roots": static_object("roots.parquet"),
                "centroids": static_object("centroids.parquet"),
            },
            "postings": {"files": static_files},
        },
    )
    write_json(directory / "metadata.json", metadata(encoding))
    write_json(
        directory / "ivf-centroids.metadata.json", ivf_centroids_metadata(encoding)
    )

    offset, scale, code = encoded[0]
    levels = (1 << bits) - 1
    middle_code = 8 if bits == 4 else 128
    reconstructed = [offset, offset + scale * middle_code, offset + scale * levels]
    distance_a = sum(value * value for value in reconstructed)
    write_json(
        directory / "queries.json",
        [
            {
                "name": "one-cluster",
                "query-vector": [0.0, 0.0, 0.0],
                "nprobe": 1,
                "k": 2,
                "filter": None,
                "projection": ["document_id"],
                "expected-code-hex": code.hex(),
                "expected": [
                    {"document_id": "a", "_distance": distance_a},
                    {"document_id": "b", "_distance": 12.0},
                ],
            }
        ],
    )


def write_lvq_fixtures() -> None:
    for encoding in INDEX_UUIDS:
        write_fixture(encoding)
