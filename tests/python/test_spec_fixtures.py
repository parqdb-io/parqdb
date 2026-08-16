from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import relify
from _support import register_source

FIXTURES = Path(__file__).parents[2] / "spec" / "fixtures" / "v1"
VALID = FIXTURES / "valid"
COMPOSITE = VALID / "composite"
INVALID = FIXTURES / "invalid"
INVALID_CASES = json.loads((INVALID / "manifest.json").read_text(encoding="utf-8"))
FINGERPRINT_NAMESPACE = uuid.UUID("2fb71e63-a27c-4fc5-9d6d-5070698dc398")


def squared_l2(left: list[float], right: list[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(left, right, strict=True))


def assert_query_result(
    actual: list[dict[str, object]],
    expected: list[dict[str, object]],
) -> None:
    distances = [row["_distance"] for row in actual]
    assert distances == sorted(distances)

    def canonical(row: dict[str, object]) -> str:
        return json.dumps(row, sort_keys=True)

    assert sorted(actual, key=canonical) == sorted(expected, key=canonical)


def reference_search(
    case: dict[str, Any],
    directory: Path = VALID,
) -> list[dict[str, object]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    snapshot = metadata["snapshots"][0]
    key_fields = snapshot["source-key-fields"]
    vector_field = snapshot["vector-field"]
    source = {
        tuple(row[field] for field in key_fields): row
        for row in pq.read_table(directory / "source.parquet").to_pylist()
    }
    centroids = pq.read_table(directory / "ivf_centroids.parquet").to_pylist()
    postings = pq.read_table(
        directory / "ivf_postings", partitioning="hive"
    ).to_pylist()
    assert all("vector" not in posting for posting in postings)
    query = case["query-vector"]
    selected = {
        row["cid"]
        for row in sorted(
            centroids,
            key=lambda row: (squared_l2(query, row["centroid"]), row["cid"]),
        )[: case["nprobe"]]
    }

    candidates = []
    for posting in postings:
        if posting["cid"] not in selected:
            continue
        key = tuple(
            posting[f"key_{position}"] for position in range(1, len(key_fields) + 1)
        )
        row = source[key]
        predicate = case["filter"]
        if predicate is not None and any(
            row[field] != value for field, value in predicate.items()
        ):
            continue
        candidates.append(
            {
                **{field: row[field] for field in case["projection"]},
                "_distance": squared_l2(
                    query,
                    row[vector_field],
                ),
            }
        )
    return sorted(candidates, key=lambda row: row["_distance"])[: case["k"]]


def reference_lvq_search(
    case: dict[str, Any],
    directory: Path,
) -> list[dict[str, object]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    snapshot = metadata["snapshots"][0]
    encoding = snapshot["parameters"]["posting_encoding"]
    bits = 4 if encoding == "lvq4" else 8
    dimension = int(snapshot["parameters"]["dimension"])
    centroids = pq.read_table(directory / "ivf_centroids.parquet").to_pylist()
    postings = pq.read_table(
        directory / "ivf_postings", partitioning="hive"
    ).to_pylist()
    query = case["query-vector"]
    selected = {
        row["cid"]
        for row in sorted(
            centroids,
            key=lambda row: (squared_l2(query, row["centroid"]), row["cid"]),
        )[: case["nprobe"]]
    }

    candidates = []
    for posting in postings:
        if posting["cid"] not in selected:
            continue
        codes = []
        for index in range(dimension):
            byte = posting["code"][index if bits == 8 else index // 2]
            codes.append(byte if bits == 8 else (byte >> (4 * (index % 2))) & 0x0F)
        vector = [posting["offset"] + posting["scale"] * code for code in codes]
        candidates.append(
            {
                "document_id": posting["key_1"],
                "_distance": squared_l2(query, vector),
            }
        )
    return sorted(candidates, key=lambda row: row["_distance"])[: case["k"]]


def register_fixture(session: relify.Session, fixture: Path, name: str) -> None:
    destination = session.root / fixture.name
    shutil.copyfile(fixture, destination)
    session._indexes.register(name, destination.as_uri())


def ivf_centroids_fingerprint(descriptor: dict[str, object]) -> str:
    source = descriptor["source"]
    assert isinstance(source, dict)
    if source["profile"] == "parquet":
        canonical_source = {"profile": "parquet", "uri": source["uri"]}
    else:
        canonical_source = {
            field: source[field]
            for field in (
                "profile",
                "table-uuid",
                "snapshot-id",
            )
        }
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


def localize_lvq_fixture(warehouse: Path, directory: Path) -> tuple[Path, Path]:
    local = warehouse / "fixture"
    shutil.copytree(directory, local)
    source = (local / "source.parquet").resolve()
    centroids = (local / "ivf_centroids.parquet").resolve()
    postings = (local / "ivf_postings").resolve()
    metadata_root = (warehouse / "metadata").resolve()
    metadata_root.mkdir()

    metadata = json.loads((local / "metadata.json").read_text(encoding="utf-8"))
    metadata["location"] = f"{metadata_root.as_uri()}/"
    snapshot = metadata["snapshots"][0]
    snapshot["source"] = {"profile": "parquet", "uri": source.as_uri()}
    centroid_metadata_path = (local / "ivf-centroids.metadata.json").resolve()
    centroid_metadata = json.loads(centroid_metadata_path.read_text(encoding="utf-8"))
    centroid_metadata["location"] = (
        f"{(local / 'centroid-artifacts').resolve().as_uri()}/"
    )
    centroid_metadata["descriptor"]["source"] = snapshot["source"]
    centroid_metadata["centroids"] = {
        "profile": "parquet",
        "uri": centroids.as_uri(),
    }
    centroid_metadata["fingerprint"] = ivf_centroids_fingerprint(
        centroid_metadata["descriptor"]
    )
    centroid_metadata_path.write_text(
        json.dumps(centroid_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot["parameters"]["ivf_centroids_fingerprint"] = centroid_metadata[
        "fingerprint"
    ]
    snapshot["parameters"]["ivf_centroids_metadata_location"] = (
        centroid_metadata_path.as_uri()
    )
    snapshot["index-relations"] = {
        "ivf_centroids": {
            "profile": "parquet",
            "uri": centroids.as_uri(),
        },
        "ivf_postings": {
            "profile": "parquet",
            "uri": f"{postings.as_uri()}/",
        },
    }
    metadata_path = metadata_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source, metadata_path


def test_valid_metadata_fixture_is_accepted_by_the_native_reader(
    tmp_path: Path,
) -> None:
    session = relify.connect(tmp_path / "relify-data")
    register_fixture(session, VALID / "metadata.json", "fixture")

    entry = session._indexes.load("fixture")
    assert entry.metadata["format-version"] == 1
    assert entry.metadata["current-snapshot-id"] == 701


def test_composite_metadata_fixture_is_accepted_by_the_native_reader(
    tmp_path: Path,
) -> None:
    session = relify.connect(tmp_path / "relify-data")
    register_fixture(session, COMPOSITE / "metadata.json", "composite")

    entry = session._indexes.load("composite")
    snapshot = entry.metadata["snapshots"][0]
    assert snapshot["source-key-fields"] == ("tenant_id", "document_id")
    assert snapshot["parameters"]["posting_encoding"] == "source"


@pytest.mark.parametrize("encoding", ["lvq4", "lvq8"])
def test_lvq_metadata_fixtures_are_accepted_by_the_native_reader(
    tmp_path: Path,
    encoding: str,
) -> None:
    session = relify.connect(tmp_path / "relify-data")
    register_fixture(session, VALID / encoding / "metadata.json", encoding)

    entry = session._indexes.load(encoding)
    snapshot = entry.metadata["snapshots"][0]
    assert snapshot["index-schema-version"] == 1
    assert snapshot["parameters"]["posting_encoding"] == encoding


@pytest.mark.parametrize("encoding", ["lvq4", "lvq8"])
def test_lvq_pyarrow_fixtures_are_queryable_by_the_native_reader(
    tmp_path: Path,
    encoding: str,
) -> None:
    directory = VALID / encoding
    session = relify.connect(tmp_path / "relify-data")
    source, metadata = localize_lvq_fixture(session.root, directory)
    documents = register_source(session, source, "documents")
    session._indexes.register(
        encoding,
        metadata.as_uri(),
        namespace=documents.identifier.index_namespace,
    )
    case = json.loads((directory / "queries.json").read_text(encoding="utf-8"))[0]

    hits = session.to_arrow(
        documents.search(case["query-vector"], index=encoding)
        .nprobes(case["nprobe"])
        .select(case["projection"])
        .limit(case["k"])
    )

    assert hits["document_id"].to_pylist() == [
        row["document_id"] for row in case["expected"]
    ]
    assert hits["_distance"].to_pylist() == pytest.approx(
        [row["_distance"] for row in case["expected"]]
    )


@pytest.mark.parametrize(
    "case",
    INVALID_CASES,
    ids=[case["file"] for case in INVALID_CASES],
)
def test_invalid_metadata_fixtures_are_rejected(
    tmp_path: Path,
    case: dict[str, str],
) -> None:
    session = relify.connect(tmp_path / "relify-data")

    with pytest.raises(relify.InvalidMetadataError):
        register_fixture(session, INVALID / case["file"], "invalid")


def test_invalid_fixture_manifest_covers_exactly_the_invalid_documents() -> None:
    documented = {case["file"] for case in INVALID_CASES}
    actual = {path.name for path in INVALID.glob("*.metadata.json") if path.is_file()}

    assert len(documented) == len(INVALID_CASES)
    assert documented == actual
    assert all(case["violates"] for case in INVALID_CASES)


def test_query_fixtures_are_internally_consistent() -> None:
    for directory in (VALID, COMPOSITE):
        cases = json.loads((directory / "queries.json").read_text(encoding="utf-8"))

        for case in cases:
            assert_query_result(reference_search(case, directory), case["expected"])


@pytest.mark.parametrize(
    ("encoding", "expected_code"),
    [("lvq4", "800f"), ("lvq8", "0080ff")],
)
def test_lvq_query_fixtures_are_internally_consistent(
    encoding: str,
    expected_code: str,
) -> None:
    directory = VALID / encoding
    cases = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    postings = pq.read_table(
        directory / "ivf_postings", partitioning="hive"
    ).to_pylist()
    first_code = next(row["code"] for row in postings if row["key_1"] == "a")
    assert first_code.hex() == expected_code

    for case in cases:
        actual = reference_lvq_search(case, directory)
        assert [row["document_id"] for row in actual] == [
            row["document_id"] for row in case["expected"]
        ]
        assert [row["_distance"] for row in actual] == pytest.approx(
            [row["_distance"] for row in case["expected"]]
        )
