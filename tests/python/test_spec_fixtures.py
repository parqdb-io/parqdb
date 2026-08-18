from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import parqdb
import pyarrow.parquet as pq
import pytest
from _support import register_source

FIXTURES = Path(__file__).parents[2] / "spec" / "fixtures" / "v1"
VALID = FIXTURES / "valid"
COMPOSITE = VALID / "composite"
INVALID = FIXTURES / "invalid"
INVALID_CASES = json.loads((INVALID / "manifest.json").read_text(encoding="utf-8"))


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


def localize_fixture(
    warehouse: Path,
    directory: Path,
    name: str,
) -> tuple[Path, Path, Path]:
    local = warehouse / f"{name}-package"
    shutil.copytree(directory, local)
    prefix = f"{local.name}/"
    metadata_path = local / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for snapshot in metadata["snapshots"]:
        snapshot["index-relations"] = {
            role: f"{prefix}{location}"
            for role, location in snapshot["index-relations"].items()
        }
        parameter = "ivf_centroids_metadata_location"
        snapshot["parameters"][parameter] = (
            f"{prefix}{snapshot['parameters'][parameter]}"
        )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    centroids_metadata_path = local / "ivf-centroids.metadata.json"
    centroids_metadata = json.loads(centroids_metadata_path.read_text(encoding="utf-8"))
    centroids_metadata["centroids"] = f"{prefix}{centroids_metadata['centroids']}"
    centroids_metadata_path.write_text(
        json.dumps(centroids_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return local / "source.parquet", metadata_path, local


def local_warehouse(session: parqdb.Session) -> Path:
    parsed = urlparse(session.warehouse)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def register_fixture(
    session: parqdb.Session,
    directory: Path,
    name: str,
) -> parqdb.SourceTable:
    source, metadata, _package = localize_fixture(
        local_warehouse(session), directory, name
    )
    table = register_source(session, source, f"{name}_source")
    table.register_index(
        name,
        metadata_location=metadata.as_uri(),
    )
    return table


def test_valid_metadata_fixture_is_accepted_by_the_native_reader(
    tmp_path: Path,
) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")
    table = register_fixture(session, VALID, "fixture")

    entry = session._indexes.load("fixture", namespace=table.identifier.index_namespace)
    assert entry.metadata["format-version"] == 1
    assert entry.metadata["current-snapshot-id"] == 701


def test_composite_metadata_fixture_is_accepted_by_the_native_reader(
    tmp_path: Path,
) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")
    table = register_fixture(session, COMPOSITE, "composite")

    entry = session._indexes.load(
        "composite", namespace=table.identifier.index_namespace
    )
    snapshot = entry.metadata["snapshots"][0]
    assert snapshot["source-key-fields"] == ("tenant_id", "document_id")
    assert snapshot["parameters"]["posting_encoding"] == "source"


@pytest.mark.parametrize("encoding", ["lvq4", "lvq8"])
def test_lvq_metadata_fixtures_are_accepted_by_the_native_reader(
    tmp_path: Path,
    encoding: str,
) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")
    table = register_fixture(session, VALID / encoding, encoding)

    entry = session._indexes.load(encoding, namespace=table.identifier.index_namespace)
    snapshot = entry.metadata["snapshots"][0]
    assert snapshot["index-schema-version"] == 1
    assert snapshot["parameters"]["posting_encoding"] == encoding


@pytest.mark.parametrize("encoding", ["lvq4", "lvq8"])
def test_lvq_pyarrow_fixtures_are_queryable_by_the_native_reader(
    tmp_path: Path,
    encoding: str,
) -> None:
    directory = VALID / encoding
    session = parqdb.connect(tmp_path / "parqdb-data")
    source, metadata, _package = localize_fixture(
        local_warehouse(session), directory, encoding
    )
    documents = register_source(session, source, "documents")
    documents.register_index(
        encoding,
        metadata_location=metadata.as_uri(),
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
    session = parqdb.connect(tmp_path / "parqdb-data")
    source, metadata, _package = localize_fixture(
        local_warehouse(session), VALID, "invalid"
    )
    shutil.copyfile(INVALID / case["file"], metadata)
    table = register_source(session, source, "invalid_source")

    with pytest.raises(parqdb.InvalidMetadataError):
        table.register_index(
            "invalid",
            metadata_location=metadata.as_uri(),
        )


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
