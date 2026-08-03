from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import relify
from relify.datafusion import col, functions

REPOSITORY = Path(__file__).parents[2]
PACKAGED_DATASETS = REPOSITORY / "python" / "relify" / "datasets"


def test_packaged_dataset_names_are_validated() -> None:
    with pytest.raises(TypeError, match="dataset name must be a string"):
        relify.datasets.uri(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="available datasets"):
        relify.datasets.uri("missing")


@pytest.mark.parametrize("name", ["documents", "document_stats"])
def test_packaged_dataset_uri_resolves_to_readable_parquet(name: str) -> None:
    uri = relify.datasets.uri(name)
    parsed = urlsplit(uri)
    assert parsed.scheme == "file"
    path = Path(unquote(parsed.path))
    assert path.is_file()
    assert pq.read_metadata(path).num_rows == 6


def test_packaged_datasets_are_reproducible(tmp_path: Path) -> None:
    generated = tmp_path / "datasets"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "tools" / "generate_example_datasets.py"),
            "--output",
            str(generated),
        ],
        cwd=REPOSITORY,
        check=True,
    )
    for filename in ("documents.parquet", "document_stats.parquet"):
        assert (generated / filename).read_bytes() == (
            PACKAGED_DATASETS / filename
        ).read_bytes()


def test_packaged_datasets_support_the_readme_workflow(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")
    session.register_parquet("documents", relify.datasets.uri("documents"))
    documents = session.table("documents")
    assert isinstance(documents, relify.SourceTable)
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=3),
    )
    documents.wait_for_index("documents_embedding")

    hits = (
        documents.search([0.2, 0.0], column="embedding")
        .where("tenant_id = 42 AND status = 'published'")
        .nprobes(3)
        .limit(3)
        .select(["document_id", "title"])
    )
    collected = session.collect(hits)
    assert collected["document_id"].to_pylist() == [1, 2, 5]

    analysis = (
        session.to_dataframe(hits)
        .join(
            session.read_parquet(relify.datasets.uri("document_stats")),
            on="document_id",
        )
        .aggregate(
            "category",
            [
                functions.count(col("document_id")).alias("matches"),
                functions.avg(col("_distance")).alias("avg_distance"),
                functions.max(col("popularity")).alias("max_popularity"),
            ],
        )
        .sort("category")
    )
    result = pa.Table.from_batches(analysis.collect(), schema=analysis.schema())

    assert result["category"].to_pylist() == ["compute", "search", "storage"]
    assert result["matches"].to_pylist() == [1, 1, 1]
    assert result["max_popularity"].to_pylist() == [95, 91, 74]
