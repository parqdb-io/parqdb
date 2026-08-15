from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fingerprint(descriptor: dict[str, object]) -> str:
    source = descriptor["source"]
    assert isinstance(source, dict)
    canonical_source = {"profile": source["profile"]}
    fields = (
        ("uri",) if source["profile"] == "parquet" else ("table-uuid", "snapshot-id")
    )
    canonical_source.update((field, source[field]) for field in fields)
    canonical_descriptor = {
        "source": canonical_source,
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
    root = f"s3://relify-fixtures/v1/valid/{encoding}"
    return {
        "source": {"profile": "parquet", "uri": f"{root}/source/"},
        "vector-field": "embedding",
        "dimension": 3,
        "metric": "l2_squared",
        "nlist": 2,
        "clustering-profile-version": 1,
    }


def metadata(encoding: str) -> dict[str, object]:
    index_uuid = INDEX_UUIDS[encoding]
    centroid_uuid = IVF_CENTROIDS_UUIDS[encoding]
    root = f"s3://relify-fixtures/v1/valid/{encoding}"
    centroid_descriptor = descriptor(encoding)
    return {
        "format-version": 1,
        "index-uuid": index_uuid,
        "location": f"s3://relify-fixtures/v1/metadata/{index_uuid}/",
        "last-updated-ms": 1_750_000_000_000,
        "last-sequence-number": 1,
        "current-snapshot-id": SNAPSHOT_ID,
        "snapshots": [
            {
                "snapshot-id": SNAPSHOT_ID,
                "sequence-number": 1,
                "timestamp-ms": 1_750_000_000_000,
                "summary": {"operation": "create"},
                "source": centroid_descriptor["source"],
                "vector-field": "embedding",
                "source-key-fields": ["document_id"],
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
                    "ivf_centroids_metadata_location": f"{root}/ivf-centroids.metadata.json",
                },
                "index-relations": {
                    "ivf_centroids": {
                        "profile": "parquet",
                        "uri": f"{root}/ivf_centroids/",
                    },
                    "ivf_postings": {
                        "profile": "parquet",
                        "uri": f"{root}/ivf_postings/",
                    },
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
    root = f"s3://relify-fixtures/v1/valid/{encoding}"
    return {
        "format-version": 1,
        "artifact-uuid": centroid_uuid,
        "fingerprint": fingerprint(centroid_descriptor),
        "location": f"{root}/centroid-artifacts/{centroid_uuid}/",
        "created-at-ms": 1_750_000_000_000,
        "descriptor": centroid_descriptor,
        "centroids": {
            "profile": "parquet",
            "uri": f"{root}/ivf_centroids/",
        },
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
    centroids = pa.Table.from_arrays(
        [
            pa.array([0, 1], type=pa.int32()),
            pa.array([[1.0, 1.0, 1.0], [10.0, 11.0, 12.0]], type=vector_type),
        ],
        schema=pa.schema(
            [
                pa.field("cid", pa.int32(), nullable=False),
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
    pq.write_to_dataset(
        postings,
        root_path=postings_root,
        partition_cols=["cid"],
        basename_template="part-{i}.parquet",
        compression="NONE",
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
