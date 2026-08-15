"""StarRocks conformance test over a real shared Iceberg catalog."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from support.config import IcebergConfig, StarRocksConfig
from support.indexes import write_shared_ivf_metadata

pytestmark = pytest.mark.requires("starrocks", "iceberg")

FIXTURES = Path(__file__).parents[2] / "spec" / "fixtures" / "v1" / "valid"


def test_starrocks_reproduces_the_portable_iceberg_fixtures(
    tmp_path: Path,
    iceberg: IcebergConfig,
    starrocks: StarRocksConfig,
) -> None:
    relify = pytest.importorskip("relify")
    flight_sql = pytest.importorskip("adbc_driver_flightsql.dbapi")
    pyiceberg_catalog = pytest.importorskip("pyiceberg.catalog")

    catalog = pyiceberg_catalog.load_catalog(
        iceberg.name,
        **iceberg.properties,
    )
    connection = flight_sql.connect(
        uri=starrocks.flight_uri,
        db_kwargs=starrocks.db_kwargs,
    )
    namespace = (f"relify_it_{uuid.uuid4().hex[:12]}",)
    catalog.create_namespace(namespace)
    created: list[tuple[str, ...]] = []
    try:
        catalog_uri = f"sqlite://{tmp_path / 'relify.sqlite'}"
        metadata_root = tmp_path / "relify-metadata"
        session = relify.experimental.starrocks.connect(
            connection,
            index_catalog=catalog_uri,
            iceberg_catalog=catalog,
            catalog_name=starrocks.catalog_name,
            metadata_root=metadata_root.as_uri(),
        )
        for fixture_name, directory in (
            ("source", FIXTURES),
            ("composite", FIXTURES / "composite"),
        ):
            source_identifier, index_name = _publish_fixture(
                session=session,
                catalog=catalog,
                namespace=namespace,
                fixture_name=fixture_name,
                directory=directory,
                catalog_name=starrocks.catalog_name,
                metadata_root=metadata_root,
                created=created,
            )
            table = session.table(".".join((*namespace, source_identifier[-1])))
            cases = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
            for case in cases:
                query = (
                    table.search(
                        case["query-vector"],
                        column="embedding",
                        index=index_name,
                    )
                    .nprobes(case["nprobe"])
                    .limit(case["k"])
                    .select(case["projection"])
                )
                predicate = _fixture_predicate(case["filter"])
                if predicate is not None:
                    query = query.where(predicate)
                actual = session.collect(query).to_pylist()
                _assert_query_result(actual, case["expected"])
    finally:
        for identifier in reversed(created):
            try:
                catalog.purge_table(identifier)
            except Exception:
                try:
                    catalog.drop_table(identifier)
                except Exception:
                    pass
        try:
            catalog.drop_namespace(namespace)
        except Exception:
            pass
        connection.close()


def _publish_fixture(
    *,
    session: Any,
    catalog: Any,
    namespace: tuple[str, ...],
    fixture_name: str,
    directory: Path,
    catalog_name: str,
    metadata_root: Path,
    created: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], str]:
    tables = {}
    for role, filename in (
        ("source", "source.parquet"),
        ("ivf_centroids", "ivf_centroids.parquet"),
        ("ivf_postings", "ivf_postings"),
    ):
        identifier = (*namespace, f"{fixture_name}_{role}")
        data = _read_fixture_relation(directory / filename, role=role)
        table = catalog.create_table(identifier, schema=data.schema)
        table.append(data)
        table.refresh()
        tables[role] = table
        created.append(identifier)

    fixture_metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    fixture_snapshot = fixture_metadata["snapshots"][0]
    source_identifier = (*namespace, f"{fixture_name}_source")
    index_name = f"{fixture_name}_embedding"
    source = _relation(tables["source"], source_identifier, catalog_name)
    centroids = _relation(
        tables["ivf_centroids"],
        (*namespace, f"{fixture_name}_ivf_centroids"),
        catalog_name,
    )
    shared = write_shared_ivf_metadata(
        metadata_root,
        source=source,
        centroids=centroids,
        vector_field=fixture_snapshot["vector-field"],
        dimension=int(fixture_snapshot["parameters"]["dimension"]),
        metric=fixture_snapshot["metric"],
        nlist=int(fixture_snapshot["parameters"]["nlist"]),
    )
    session._native.publish_initial(
        index_name=index_name,
        source_json=json.dumps(source, separators=(",", ":")),
        vector_field=fixture_snapshot["vector-field"],
        source_key_fields=fixture_snapshot["source-key-fields"],
        builder="fixture",
        metric=fixture_snapshot["metric"],
        parameters={**fixture_snapshot["parameters"], **shared},
        index_relations={
            "ivf_centroids": json.dumps(centroids, separators=(",", ":")),
            "ivf_postings": json.dumps(
                _relation(
                    tables["ivf_postings"],
                    (*namespace, f"{fixture_name}_ivf_postings"),
                    catalog_name,
                ),
                separators=(",", ":"),
            ),
        },
    )
    return source_identifier, index_name


def _read_fixture_relation(path: Path, *, role: str) -> pa.Table:
    data = pq.read_table(
        path,
        partitioning="hive" if role == "ivf_postings" else None,
    )
    if role != "ivf_postings":
        return data

    cid_index = data.schema.get_field_index("cid")
    if cid_index < 0:
        raise ValueError("IVF postings fixture has no cid partition field")
    cid = data.column(cid_index)
    if cid.null_count:
        raise ValueError("IVF postings fixture contains a null cid")
    return data.set_column(
        cid_index,
        pa.field("cid", pa.int32(), nullable=False),
        cid.cast(pa.int32()),
    )


def _relation(
    table: Any,
    identifier: tuple[str, ...],
    catalog_name: str,
) -> dict[str, object]:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return {
        "profile": "iceberg",
        "catalog": catalog_name,
        "namespace": list(identifier[:-1]),
        "name": identifier[-1],
        "table-uuid": str(table.metadata.table_uuid).lower(),
        "snapshot-id": int(snapshot.snapshot_id),
    }


def _fixture_predicate(filters: dict[str, Any] | None) -> str | None:
    if not filters:
        return None
    predicates = []
    for field, value in filters.items():
        quoted_field = "`" + field.replace("`", "``") + "`"
        if isinstance(value, str):
            literal = "'" + value.replace("'", "''") + "'"
        elif isinstance(value, bool):
            literal = "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            literal = repr(value)
        else:
            raise TypeError(f"unsupported fixture filter value: {value!r}")
        predicates.append(f"{quoted_field} = {literal}")
    return " AND ".join(predicates)


def _assert_query_result(
    actual: list[dict[str, object]],
    expected: list[dict[str, object]],
) -> None:
    def key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(
            f"{name}={value!r}"
            for name, value in sorted(row.items())
            if name != "_distance"
        )

    actual_sorted = sorted(actual, key=key)
    expected_sorted = sorted(expected, key=key)
    assert len(actual_sorted) == len(expected_sorted)
    for actual_row, expected_row in zip(actual_sorted, expected_sorted, strict=True):
        assert {
            key: value for key, value in actual_row.items() if key != "_distance"
        } == {key: value for key, value in expected_row.items() if key != "_distance"}
        assert actual_row["_distance"] == pytest.approx(
            expected_row["_distance"],
            rel=1e-5,
            abs=1e-6,
        )
