from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from generate_lvq import encode, write_lvq_fixtures

ROOT = Path(__file__).parent
VALID = ROOT / "valid"
COMPOSITE = VALID / "composite"
INVALID = ROOT / "invalid"

INDEX_UUID = "2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1"
COMPOSITE_INDEX_UUID = "8e51cf0a-c749-4894-83e7-5e9be2c373b7"
IVF_CENTROIDS_UUID = "fe985f6d-3592-4385-a1ca-71347057a210"
COMPOSITE_IVF_CENTROIDS_UUID = "6f084127-25b1-43f1-a676-b8c055e37fae"
FINGERPRINT_NAMESPACE = uuid.UUID("2fb71e63-a27c-4fc5-9d6d-5070698dc398")
SNAPSHOT_ID = 701
NEXT_SNAPSHOT_ID = 702
COMPOSITE_SNAPSHOT_ID = 801


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def leaf_centroids(values: list[list[float]]) -> pa.Table:
    encoded = [encode(value, 8) for value in values]
    return pa.Table.from_arrays(
        [
            pa.array(range(len(values)), type=pa.int32()),
            pa.array([0] * len(values), type=pa.int32()),
            pa.array([row[0] for row in encoded], type=pa.float32()),
            pa.array([row[1] for row in encoded], type=pa.float32()),
            pa.array([row[2] for row in encoded], type=pa.binary()),
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


def write_postings(directory: Path, table: pa.Table) -> None:
    root = directory / "ivf_postings"
    shutil.rmtree(root, ignore_errors=True)
    (directory / "ivf_postings.parquet").unlink(missing_ok=True)
    bucket = root / "cid_bucket=000000"
    bucket.mkdir(parents=True)
    path = bucket / "part-00000.parquet"
    writer = pq.ParquetWriter(path, table.schema, compression="NONE")
    for cid in range(2):
        rows = table.filter(pc.equal(table["cid"], cid))
        if rows.num_rows:
            writer.write_table(rows, row_group_size=rows.num_rows)
    writer.close()
    write_json(
        root / "manifest.json",
        {
            "format-version": 1,
            "nlist": 2,
            "ntotal": table.num_rows,
            "cid-offsets": [0, 2],
            "files": [
                {
                    "path": "cid_bucket=000000/part-00000.parquet",
                    "cid-bucket": 0,
                    "min-cid": 0,
                    "max-cid": 1,
                    "rows": table.num_rows,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        },
    )


def ivf_centroids_descriptor(
    *,
    dimension: int = 2,
    metric: str = "l2_squared",
    nlist: int = 2,
) -> dict[str, object]:
    return {
        "vector-field": "embedding",
        "dimension": dimension,
        "metric": metric,
        "nlist": nlist,
        "clustering-profile-version": 1,
    }


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


def metadata() -> dict[str, object]:
    descriptor = ivf_centroids_descriptor()
    return {
        "format-version": 1,
        "index-uuid": INDEX_UUID,
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
                    "dimension": "2",
                    "nlist": "2",
                    "ntotal": "3",
                    "posting_encoding": "source",
                    "ivf_centroids_fingerprint": fingerprint(descriptor),
                    "ivf_centroids_uuid": IVF_CENTROIDS_UUID,
                    "ivf_centroids_metadata_location": ("ivf-centroids.metadata.json"),
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
        "properties": {"fixture": "ivf-parquet"},
    }


def ivf_centroids_metadata(
    artifact_uuid: str,
    descriptor: dict[str, object],
) -> dict[str, object]:
    return {
        "format-version": 1,
        "artifact-uuid": artifact_uuid,
        "fingerprint": fingerprint(descriptor),
        "created-at-ms": 1_750_000_000_000,
        "descriptor": descriptor,
        "centroids": "ivf_centroids.parquet",
        "roots": "ivf_roots.parquet",
    }


def next_metadata(base: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(base)
    timestamp_ms = value["last-updated-ms"] + 1
    snapshot = copy.deepcopy(value["snapshots"][0])
    snapshot["snapshot-id"] = NEXT_SNAPSHOT_ID
    snapshot["sequence-number"] = 2
    snapshot["timestamp-ms"] = timestamp_ms
    snapshot["summary"] = {"operation": "refresh"}
    value["last-updated-ms"] = timestamp_ms
    value["last-sequence-number"] = 2
    value["current-snapshot-id"] = NEXT_SNAPSHOT_ID
    value["snapshots"].append(snapshot)
    value["snapshot-log"].append(
        {
            "timestamp-ms": timestamp_ms,
            "snapshot-id": NEXT_SNAPSHOT_ID,
        }
    )
    return value


def catalog_operations() -> dict[str, object]:
    return {
        "identifier": {
            "namespace": ["analytics"],
            "name": "documents",
        },
        "metadata": {
            "base": "valid/metadata.json",
            "next": "valid/metadata-next.json",
        },
        "locations": {
            "base": "s3://parqdb-fixtures/v1/catalog/metadata-v1.json",
            "next": "s3://parqdb-fixtures/v1/catalog/metadata-next.json",
        },
        "operations": [
            {
                "operation": "load",
                "expect": "INDEX_NOT_FOUND",
            },
            {
                "operation": "register",
                "metadata": "base",
                "location": "base",
                "expect": "OK",
            },
            {
                "operation": "register",
                "metadata": "base",
                "location": "base",
                "expect": "ALREADY_EXISTS",
            },
            {
                "operation": "load",
                "expect-location": "base",
                "expect": "OK",
            },
            {
                "operation": "commit",
                "base-metadata": "base",
                "base-location": "base",
                "new-metadata": "next",
                "new-location": "next",
                "expect": "OK",
            },
            {
                "operation": "commit",
                "base-metadata": "base",
                "base-location": "base",
                "new-metadata": "next",
                "new-location": "next",
                "expect": "COMMIT_CONFLICT",
            },
            {
                "operation": "load",
                "expect-location": "next",
                "expect": "OK",
            },
            {
                "operation": "drop",
                "expect": "OK",
            },
            {
                "operation": "load",
                "expect": "INDEX_NOT_FOUND",
            },
        ],
    }


def write_tables() -> None:
    vector = pa.list_(pa.field("element", pa.float32(), nullable=False))
    source_schema = pa.schema(
        [
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("tenant_id", pa.int32(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("embedding", vector, nullable=False),
        ]
    )
    source = pa.Table.from_arrays(
        [
            pa.array(["a", "b", "c"], type=pa.string()),
            pa.array(["Alpha", "Beta", "Gamma"], type=pa.string()),
            pa.array([1, 1, 2], type=pa.int32()),
            pa.array(["published", "draft", "published"], type=pa.string()),
            pa.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], type=vector),
        ],
        schema=source_schema,
    )
    centroids = leaf_centroids([[0.5, 0.0], [10.0, 0.0]])
    roots = pa.Table.from_arrays(
        [
            pa.array([0], type=pa.int32()),
            pa.array([0], type=pa.int32()),
            pa.array([2], type=pa.int32()),
            pa.array([[5.25, 0.0]], type=vector),
        ],
        schema=pa.schema(
            [
                pa.field("cid_bucket", pa.int32(), nullable=False),
                pa.field("cid_begin", pa.int32(), nullable=False),
                pa.field("cid_end", pa.int32(), nullable=False),
                pa.field("centroid", vector, nullable=False),
            ]
        ),
    )
    postings = pa.Table.from_arrays(
        [
            pa.array([0, 0, 1], type=pa.int32()),
            pa.array(["a", "b", "c"], type=pa.string()),
        ],
        schema=pa.schema(
            [
                pa.field("cid", pa.int32(), nullable=False),
                pa.field("key_1", pa.string(), nullable=False),
            ]
        ),
    )
    for name, table in (
        ("source.parquet", source),
        ("ivf_centroids.parquet", centroids),
        ("ivf_roots.parquet", roots),
    ):
        pq.write_table(table, VALID / name, compression="NONE")
    write_postings(VALID, postings)


def composite_metadata() -> dict[str, object]:
    value = metadata()
    value["index-uuid"] = COMPOSITE_INDEX_UUID
    value["current-snapshot-id"] = COMPOSITE_SNAPSHOT_ID
    value["snapshot-log"] = [
        {
            "timestamp-ms": 1_750_000_000_000,
            "snapshot-id": COMPOSITE_SNAPSHOT_ID,
        }
    ]
    value["properties"] = {"fixture": "ivf-parquet-composite"}
    snapshot = value["snapshots"][0]  # type: ignore[index]
    snapshot["snapshot-id"] = COMPOSITE_SNAPSHOT_ID
    snapshot["source-key-fields"] = ["tenant_id", "document_id"]
    descriptor = ivf_centroids_descriptor()
    snapshot["parameters"]["ivf_centroids_fingerprint"] = fingerprint(descriptor)
    snapshot["parameters"]["ivf_centroids_uuid"] = COMPOSITE_IVF_CENTROIDS_UUID
    snapshot["parameters"]["ivf_centroids_metadata_location"] = (
        "ivf-centroids.metadata.json"
    )
    snapshot["index-relations"] = {
        "ivf_centroids": "ivf_centroids.parquet",
        "ivf_postings": "ivf_postings/",
    }
    return value


def write_composite_tables() -> None:
    vector = pa.list_(pa.field("element", pa.float32(), nullable=False))
    source = pa.Table.from_arrays(
        [
            pa.array([1, 1, 2], type=pa.int32()),
            pa.array(["b", "a", "a"], type=pa.string()),
            pa.array(["One B", "One A", "Two A"], type=pa.string()),
            pa.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]], type=vector),
        ],
        schema=pa.schema(
            [
                pa.field("tenant_id", pa.int32(), nullable=False),
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("title", pa.string(), nullable=False),
                pa.field("embedding", vector, nullable=False),
            ]
        ),
    )
    centroids = leaf_centroids([[0.0, 0.0], [10.0, 0.0]])
    roots = pa.Table.from_arrays(
        [
            pa.array([0], type=pa.int32()),
            pa.array([0], type=pa.int32()),
            pa.array([2], type=pa.int32()),
            pa.array([[5.0, 0.0]], type=vector),
        ],
        schema=pa.schema(
            [
                pa.field("cid_bucket", pa.int32(), nullable=False),
                pa.field("cid_begin", pa.int32(), nullable=False),
                pa.field("cid_end", pa.int32(), nullable=False),
                pa.field("centroid", vector, nullable=False),
            ]
        ),
    )
    postings = pa.Table.from_arrays(
        [
            pa.array([0, 0, 1], type=pa.int32()),
            pa.array([1, 1, 2], type=pa.int32()),
            pa.array(["b", "a", "a"], type=pa.string()),
        ],
        schema=pa.schema(
            [
                pa.field("cid", pa.int32(), nullable=False),
                pa.field("key_1", pa.int32(), nullable=False),
                pa.field("key_2", pa.string(), nullable=False),
            ]
        ),
    )
    for name, table in (
        ("source.parquet", source),
        ("ivf_centroids.parquet", centroids),
        ("ivf_roots.parquet", roots),
    ):
        pq.write_table(table, COMPOSITE / name, compression="NONE")
    write_postings(COMPOSITE, postings)


def write_invalid_documents(base: dict[str, object]) -> None:
    cases: list[dict[str, str]] = []

    def write_case(
        filename: str,
        value: object,
        violates: str,
    ) -> None:
        write_json(INVALID / filename, value)
        cases.append({"file": filename, "violates": violates})

    noncanonical = copy.deepcopy(base)
    noncanonical["snapshots"][0]["index-relations"]["ivf_postings"] = (  # type: ignore[index]
        "s3://parqdb-fixtures/v1/valid/ivf_postings/"
    )
    write_case(
        "absolute-index-relation.metadata.json",
        noncanonical,
        "index relation locations must be relative to the warehouse",
    )

    unknown_role = copy.deepcopy(base)
    unknown_role["snapshots"][0]["index-relations"]["unknown"] = "unknown/"  # type: ignore[index]
    write_case(
        "unknown-ivf-role.metadata.json",
        unknown_role,
        "IVF schema version 1 defines exactly two index relation roles",
    )

    unsupported_format = copy.deepcopy(base)
    unsupported_format["format-version"] = 2
    write_case(
        "unsupported-format-version.metadata.json",
        unsupported_format,
        "format-version must be 1",
    )

    missing_current = copy.deepcopy(base)
    missing_current["current-snapshot-id"] = 999
    write_case(
        "current-snapshot-not-retained.metadata.json",
        missing_current,
        "current-snapshot-id must identify one retained snapshot",
    )

    duplicate_source_key = copy.deepcopy(base)
    duplicate_source_key["snapshots"][0]["source-key-fields"] = [  # type: ignore[index]
        "document_id",
        "document_id",
    ]
    write_case(
        "duplicate-source-key-field.metadata.json",
        duplicate_source_key,
        "source-key-fields must not contain duplicates",
    )

    missing_role = copy.deepcopy(base)
    del missing_role["snapshots"][0]["index-relations"]["ivf_postings"]  # type: ignore[index]
    write_case(
        "missing-ivf-role.metadata.json",
        missing_role,
        "every IVF snapshot requires ivf_centroids and ivf_postings",
    )

    leading_zero = copy.deepcopy(base)
    leading_zero["snapshots"][0]["parameters"]["nlist"] = "02"  # type: ignore[index]
    write_case(
        "noncanonical-positive-parameter.metadata.json",
        leading_zero,
        "positive integer parameters must use canonical base-10 representation",
    )

    obsolete_storage = copy.deepcopy(base)
    obsolete_storage["snapshots"][0]["parameters"]["store_vectors"] = "true"  # type: ignore[index]
    write_case(
        "unsupported-store-vectors.metadata.json",
        obsolete_storage,
        "store_vectors is not part of IVF schema version 1",
    )

    obsolete_flat = copy.deepcopy(base)
    obsolete_flat["snapshots"][0]["parameters"]["posting_encoding"] = "flat"  # type: ignore[index]
    write_case(
        "unsupported-flat-encoding.metadata.json",
        obsolete_flat,
        "posting_encoding must be source, lvq4, or lvq8",
    )

    unsupported_metric = copy.deepcopy(base)
    unsupported_metric["snapshots"][0]["metric"] = "l2"  # type: ignore[index]
    write_case(
        "unsupported-metric.metadata.json",
        unsupported_metric,
        "IVF metric must be l2_squared or cosine",
    )

    escaping_relation = copy.deepcopy(base)
    escaping_relation["snapshots"][0]["index-relations"]["ivf_postings"] = (  # type: ignore[index]
        "../ivf_postings/"
    )
    write_case(
        "escaping-index-relation.metadata.json",
        escaping_relation,
        "index relation locations must not escape the warehouse",
    )

    missing_field = copy.deepcopy(base)
    del missing_field["snapshots"][0]["metric"]  # type: ignore[index]
    write_case(
        "missing-required-field.metadata.json",
        missing_field,
        "index snapshot metric is required",
    )

    uppercase_uuid = copy.deepcopy(base)
    uppercase_uuid["index-uuid"] = INDEX_UUID.upper()
    write_case(
        "noncanonical-index-uuid.metadata.json",
        uppercase_uuid,
        "UUIDs use lowercase hexadecimal strings",
    )

    out_of_order_log = copy.deepcopy(base)
    out_of_order_log["snapshot-log"].append(  # type: ignore[union-attr]
        {
            "timestamp-ms": 1_749_999_999_999,
            "snapshot-id": SNAPSHOT_ID,
        }
    )
    write_case(
        "decreasing-snapshot-log.metadata.json",
        out_of_order_log,
        "snapshot-log timestamps must be non-decreasing",
    )

    duplicate_snapshot = copy.deepcopy(base)
    second = copy.deepcopy(duplicate_snapshot["snapshots"][0])  # type: ignore[index]
    second["sequence-number"] = 2
    duplicate_snapshot["snapshots"].append(second)  # type: ignore[union-attr]
    duplicate_snapshot["last-sequence-number"] = 2
    write_case(
        "duplicate-snapshot-id.metadata.json",
        duplicate_snapshot,
        "snapshot IDs must be unique within the index",
    )

    changed_identity = copy.deepcopy(base)
    second = copy.deepcopy(changed_identity["snapshots"][0])  # type: ignore[index]
    second["snapshot-id"] = 702
    second["sequence-number"] = 2
    second["vector-field"] = "other_embedding"
    changed_identity["snapshots"].append(second)  # type: ignore[union-attr]
    changed_identity["last-sequence-number"] = 2
    changed_identity["current-snapshot-id"] = 702
    changed_identity["snapshot-log"].append(  # type: ignore[union-attr]
        {"timestamp-ms": 1_750_000_000_000, "snapshot-id": 702}
    )
    write_case(
        "changed-snapshot-identity.metadata.json",
        changed_identity,
        "logical identity fields remain equal across snapshots",
    )

    valid_json = json.dumps(base, indent=2, sort_keys=True)
    duplicate = valid_json.replace(
        '"dimension": "2",',
        '"dimension": "2",\n          "dimension": "3",',
        1,
    )
    (INVALID / "duplicate-map-key.metadata.json").write_text(
        duplicate + "\n",
        encoding="utf-8",
    )
    cases.append(
        {
            "file": "duplicate-map-key.metadata.json",
            "violates": "JSON maps must contain unique keys",
        }
    )

    unknown_field = valid_json.replace(
        '"format-version": 1,',
        '"format-version": 1,\n  "unexpected": "value",',
        1,
    )
    (INVALID / "unknown-metadata-field.metadata.json").write_text(
        unknown_field + "\n",
        encoding="utf-8",
    )
    cases.append(
        {
            "file": "unknown-metadata-field.metadata.json",
            "violates": "metadata documents reject unknown fields",
        }
    )
    write_json(INVALID / "manifest.json", cases)


def main() -> None:
    VALID.mkdir(parents=True, exist_ok=True)
    COMPOSITE.mkdir(parents=True, exist_ok=True)
    INVALID.mkdir(parents=True, exist_ok=True)
    for path in INVALID.glob("*.metadata.json"):
        path.unlink()
    base = metadata()
    write_json(VALID / "metadata.json", base)
    write_json(VALID / "metadata-next.json", next_metadata(base))
    source_descriptor = ivf_centroids_descriptor()
    write_json(
        VALID / "ivf-centroids.metadata.json",
        ivf_centroids_metadata(
            IVF_CENTROIDS_UUID,
            source_descriptor,
        ),
    )
    write_json(ROOT / "catalog.json", catalog_operations())
    write_json(
        VALID / "queries.json",
        [
            {
                "name": "one-cluster",
                "query-vector": [0.0, 0.0],
                "nprobe": 1,
                "k": 2,
                "filter": None,
                "projection": ["document_id"],
                "expected": [
                    {"document_id": "a", "_distance": 0.0},
                    {"document_id": "b", "_distance": 1.0},
                ],
            },
            {
                "name": "prefilter",
                "query-vector": [0.0, 0.0],
                "nprobe": 2,
                "k": 2,
                "filter": {"status": "published"},
                "projection": ["document_id"],
                "expected": [
                    {"document_id": "a", "_distance": 0.0},
                    {"document_id": "c", "_distance": 100.0},
                ],
            },
            {
                "name": "equal-distance-results",
                "query-vector": [0.5, 0.0],
                "nprobe": 2,
                "k": 2,
                "filter": None,
                "projection": ["document_id"],
                "expected": [
                    {"document_id": "a", "_distance": 0.25},
                    {"document_id": "b", "_distance": 0.25},
                ],
            },
            {
                "name": "centroid-id-tie-break",
                "query-vector": [5.25, 0.0],
                "nprobe": 1,
                "k": 3,
                "filter": None,
                "projection": ["document_id"],
                "expected": [
                    {"document_id": "b", "_distance": 18.0625},
                    {"document_id": "a", "_distance": 27.5625},
                ],
            },
            {
                "name": "k-exceeds-selected-candidates",
                "query-vector": [10.0, 0.0],
                "nprobe": 1,
                "k": 10,
                "filter": None,
                "projection": ["document_id"],
                "expected": [
                    {"document_id": "c", "_distance": 0.0},
                ],
            },
            {
                "name": "filter-excludes-all",
                "query-vector": [0.0, 0.0],
                "nprobe": 2,
                "k": 3,
                "filter": {"status": "missing"},
                "projection": ["document_id"],
                "expected": [],
            },
        ],
    )
    write_json(COMPOSITE / "metadata.json", composite_metadata())
    composite_descriptor = ivf_centroids_descriptor()
    write_json(
        COMPOSITE / "ivf-centroids.metadata.json",
        ivf_centroids_metadata(
            COMPOSITE_IVF_CENTROIDS_UUID,
            composite_descriptor,
        ),
    )
    write_json(
        COMPOSITE / "queries.json",
        [
            {
                "name": "composite-equal-distance-results",
                "query-vector": [0.0, 0.0],
                "nprobe": 2,
                "k": 3,
                "filter": None,
                "projection": ["tenant_id", "document_id"],
                "expected": [
                    {"tenant_id": 1, "document_id": "a", "_distance": 0.0},
                    {"tenant_id": 1, "document_id": "b", "_distance": 0.0},
                    {"tenant_id": 2, "document_id": "a", "_distance": 100.0},
                ],
            },
            {
                "name": "composite-source-encoding",
                "query-vector": [0.0, 0.0],
                "nprobe": 1,
                "k": 10,
                "filter": None,
                "projection": ["tenant_id", "document_id"],
                "expected": [
                    {"tenant_id": 1, "document_id": "a", "_distance": 0.0},
                    {"tenant_id": 1, "document_id": "b", "_distance": 0.0},
                ],
            },
        ],
    )
    write_tables()
    write_composite_tables()
    write_lvq_fixtures()
    write_invalid_documents(base)


if __name__ == "__main__":
    main()
