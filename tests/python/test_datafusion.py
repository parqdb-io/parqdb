from __future__ import annotations

import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any, cast

import pyarrow
import pytest
import relify
import relify.datafusion as datafusion
from _support import build_index, register_source, write_vectors


def test_to_dataframe_returns_a_composable_datafusion_dataframe(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(3).select(["id", "payload"])

    result = session.to_dataframe(query)
    batches = result.filter("id >= 1").select("id", "_distance").collect()

    assert isinstance(result, datafusion.DataFrame)
    assert [value for batch in batches for value in batch["id"].to_pylist()] == [1, 2]


def test_to_sql_returns_executable_sql_over_the_registered_source(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(3).select(["id", "payload"])

    sql = session.to_sql(query)
    sql_result = session.sql(sql).to_pydict()
    dataframe_result = session.to_dataframe(query).to_pydict()

    assert 'FROM "documents"' in sql
    assert '"__relify_postings_' in sql
    assert "__relify_explain_" not in sql
    assert session.to_sql(query) == sql
    assert sql_result == dataframe_result


def test_session_is_its_native_context(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")
    batch = pyarrow.record_batch([[1, 2]], names=["value"])

    assert isinstance(session, datafusion.SessionContext)
    assert "relify_squared_l2" in session.udfs()
    session.register_record_batches("values", [[batch]])
    assert session.sql("SELECT SUM(value) AS total FROM values").to_pydict() == {
        "total": [3]
    }


def test_embedded_dataframe_repr_uses_embedded_formatter(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")
    result = session.sql("SELECT 1 AS value")

    assert "value" in repr(result)
    assert "<table" in result._repr_html_()


def test_context_derivation_does_not_create_an_incomplete_relify_session(
    tmp_path: Path,
) -> None:
    session = relify.connect(tmp_path / "relify-data")

    derived = session.enable_url_table()
    global_context = relify.Session.global_ctx()

    assert type(derived) is datafusion.SessionContext
    assert type(global_context) is datafusion.SessionContext


def test_embedded_dataframe_api_supports_lazy_join_and_aggregation(
    tmp_path: Path,
) -> None:
    from relify.datafusion import col
    from relify.datafusion import functions as functions

    session = relify.connect(tmp_path / "relify-data")
    values = session.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "team": ["a", "a", "b", "b"],
            "score": [5, 15, 20, 30],
        }
    )
    labels = session.from_pydict(
        {
            "id": [2, 3, 4],
            "label": ["two", "three", "four"],
        }
    )

    result = (
        values.filter("score >= 15")
        .with_column("weighted", col("score") * 2)
        .join(labels, on="id")
        .aggregate(
            "team",
            [
                functions.sum(col("weighted")).alias("total"),
                functions.count(col("id")).alias("matches"),
            ],
        )
        .sort("team")
    )

    assert isinstance(result, datafusion.DataFrame)
    assert result.to_pydict() == {
        "team": ["a", "b"],
        "total": [30, 100],
        "matches": [1, 2],
    }


def test_native_types_belong_to_relify_datafusion() -> None:
    expression = datafusion.col("value")
    context = datafusion.SessionContext()

    assert relify.datafusion is datafusion
    assert type(expression.expr).__module__ == "relify.datafusion.expr"
    assert type(context.ctx).__module__ == "relify.datafusion"
    assert datafusion.__version__ == "54.0.0"
    assert hasattr(datafusion.substrait, "Serde")


def test_all_native_type_modules_are_embedded() -> None:
    pending = [datafusion._internal]
    visited: set[int] = set()
    native_types: list[type[Any]] = []

    while pending:
        module = pending.pop()
        if id(module) in visited:
            continue
        visited.add(id(module))
        for value in vars(module).values():
            if isinstance(value, type):
                native_types.append(value)
            elif isinstance(value, types.ModuleType):
                pending.append(value)

    assert native_types
    assert all(
        value.__module__ != "datafusion"
        and not value.__module__.startswith("datafusion.")
        for value in native_types
    )


def test_embedded_datafusion_does_not_replace_top_level_package(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        import types

        external = types.ModuleType("datafusion")
        external.marker = object()
        sys.modules["datafusion"] = external

        import relify
        import relify.datafusion as embedded

        assert sys.modules["datafusion"] is external
        assert embedded.SessionContext().sql("SELECT 1 AS n").to_pydict() == {
            "n": [1]
        }
        assert relify.connect(sys.argv[1]).sql("SELECT 2 AS n").to_pydict() == {
            "n": [2]
        }
        """
    )

    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "coexistence")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_squared_l2_udf_is_stateless_and_accepts_query_as_an_argument(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents

    direct = session.sql(
        """
        SELECT
            relify_squared_l2(
                make_array(CAST(1 AS REAL), CAST(2 AS REAL)),
                make_array(CAST(4 AS REAL), CAST(6 AS REAL))
            ) AS distance
        """
    ).to_pydict()
    near_zero = session.to_dataframe(
        documents.search([0.0, 0.0]).limit(1).select(["id"])
    )
    near_ten = session.to_dataframe(
        documents.search([10.0, 0.0]).limit(1).select(["id"])
    )

    assert direct == {"distance": [25.0]}
    assert near_zero.to_pydict()["id"] == [0]
    assert near_ten.to_pydict()["id"] == [2]


def test_native_planner_restores_the_relify_distance_udf(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents

    fake_distance = datafusion.udf(
        lambda left, _right: pyarrow.array(
            [0.0] * len(left),
            type=pyarrow.float32(),
        ),
        [pyarrow.list_(pyarrow.float32()), pyarrow.list_(pyarrow.float32())],
        pyarrow.float32(),
        "immutable",
        "relify_squared_l2",
    )
    session.register_udf(fake_distance)

    result = session.to_dataframe(documents.search([10.0, 0.0]).limit(1).select(["id"]))

    assert result.to_pydict()["id"] == [2]


def test_native_planner_does_not_collide_with_user_relations(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    sentinel = pyarrow.record_batch([[42]], names=["value"])
    session.register_record_batches("__relify_source_0", [[sentinel]])

    result = session.to_dataframe(documents.search([0.0, 0.0]).limit(1).select(["id"]))

    assert result.to_pydict()["id"] == [0]
    assert session.sql('SELECT value FROM "__relify_source_0"').to_pydict() == {
        "value": [42]
    }


def test_vector_dataframe_uses_native_view_registration(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(2).select(["id"])

    session.register_view("relify_test_hits", session.to_dataframe(query))

    result = session.sql(
        """
        SELECT id, _distance
        FROM relify_test_hits
        WHERE id >= 1
        ORDER BY id
        """
    )
    batches = result.collect()

    assert [value for batch in batches for value in batch["id"].to_pylist()] == [1]
    session.deregister_table("relify_test_hits")
    assert not session.table_exist("relify_test_hits")


def test_explain_plan_reports_search_logical_and_physical_plans(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(2).select(["id"])

    plan = session.explain(query)
    verbose_plan = session.explain(query, verbose=True)

    assert "logical_plan" in plan
    assert "physical_plan" in plan
    assert "relify_squared_l2" in plan
    assert "IvfTopKExec" in plan
    assert "SortExec: TopK" not in plan
    assert "id@0 ASC" not in plan
    assert "schema=[" not in plan
    assert "schema=[" in verbose_plan


def test_explain_plan_does_not_scan_the_source(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    write_vectors(
        source,
        [0, 1, 2],
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    )

    plan = session.explain(vectors.search([0.0, 0.0]))

    assert "physical_plan" in plan
    assert "DataSourceExec" in plan


def test_analyze_plan_executes_search_and_reports_runtime_metrics(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(2).select(["id"])

    plan = session.analyze(query)

    assert "Plan with Metrics" in plan
    assert "IvfTopKExec" in plan
    assert "output_rows=" in plan
    assert "bytes_scanned=" in plan
    assert "distance_compute=" in plan
    assert "distance_evaluations=" in plan
    assert "selection_compute=" in plan
    assert "selection_candidates=" in plan
    assert "selection_discarded=" in plan
    assert "selection_passes=" in plan
    assert "dynamic_filter_pruned=" in plan
    assert "candidate_sort_compute=" in plan
    assert "projection_compute=" in plan
    assert "retained_batches_peak=" in plan
    assert "retained_bytes_peak=" in plan


def test_full_probe_omits_the_cluster_predicate(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).nprobes(2).limit(2).select(["id"])

    plan = session.explain(query)

    assert "cid IN" not in plan
    assert "cid =" not in plan
    assert session.to_arrow(query)["id"].to_pylist() == [0, 1]


def test_large_nprobe_prunes_postings_files_during_planning(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    ids = list(range(256))
    write_vectors(source, ids, [[float(value), 0.0] for value in ids])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=256)
    query = vectors.search([0.0, 0.0]).nprobes(129).limit(3).select(["id"])

    plan = session.explain(query)

    assert "IvfTopKExec" in plan
    assert "join_type=RightSemi" not in plan
    assert "full_filters=" in plan
    assert "file_groups={" in plan
    assert "/cid=129/" not in plan
    assert "FilterExec" not in plan
    sql = session.to_sql(query)
    assert 'relify_selected_clusters("cid") AS (' in sql
    assert "VALUES (" in sql
    assert session.sql(sql).to_pydict()["id"] == [0, 1, 2]
    assert session.to_arrow(query)["id"].to_pylist() == [0, 1, 2]


def test_explain_plan_validates_verbose_argument(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")
    query = relify.VectorQuery(
        source=relify.TableIdentifier("datafusion", ("public",), "documents"),
        query=(1.0, 2.0),
    )

    with pytest.raises(TypeError, match="verbose must be a boolean"):
        session.explain(query, verbose=cast(Any, 1))
