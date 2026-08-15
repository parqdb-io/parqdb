from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path

import pytest
import relify
from _support import register_source, write_vectors
from relify.builders import (
    BuildContext,
    BuilderCapabilities,
    BuilderInfo,
    BuildOutput,
    BuildProfile,
    BuildProgressSnapshot,
)


def test_builtin_builders_publish_independent_capabilities() -> None:
    local = relify.Local(
        threads=4,
        max_row_group_rows=4_096,
        write_batch_rows=1_024,
    )

    assert local.info == BuilderInfo("local", "Local Rust", "relify")
    assert local.capabilities.supports(BuildProfile("ivf", "parquet", "parquet"))
    assert not local.capabilities.supports(BuildProfile("ivf", "iceberg", "iceberg"))
    assert local.capabilities.to_dict() == {
        "profiles": [
            {
                "family": "ivf",
                "source_profile": "parquet",
                "index_profile": "parquet",
            }
        ]
    }


def test_build_output_is_immutable_and_normalizes_parameters() -> None:
    output = BuildOutput(
        parameters={"nlist": 2},  # type: ignore[dict-item]
        index_relations={
            "ivf_centroids": {
                "profile": "parquet",
                "uri": "file:///indexes/centroids",
            }
        },
    )

    assert output.parameters["nlist"] == "2"
    with pytest.raises(TypeError):
        output.parameters["nlist"] = "3"  # type: ignore[index]
    with pytest.raises(TypeError):
        output.index_relations["ivf_centroids"]["uri"] = "other"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        output.discard = None  # type: ignore[misc]


def test_build_context_accepts_a_backend_neutral_progress_source() -> None:
    class _Progress:
        def snapshot(self) -> BuildProgressSnapshot:
            return BuildProgressSnapshot("building", 2, 4, 0.5)

    context = BuildContext(progress=_Progress())

    assert context.progress is not None
    assert context.progress.snapshot() == BuildProgressSnapshot(
        "building",
        2,
        4,
        0.5,
    )
    with pytest.raises(TypeError, match="BuildProgress"):
        BuildContext(progress=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "profile",
    [
        BuildProfile("ivf", "parquet", "parquet"),
        BuildProfile("ivf", "iceberg", "iceberg"),
    ],
)
def test_builder_capabilities_are_typed(profile: BuildProfile) -> None:
    capabilities = BuilderCapabilities(frozenset({profile}))

    assert capabilities.supports(profile)
    assert not capabilities.supports(
        BuildProfile("future", profile.source_profile, profile.index_profile)
    )


def test_coordinator_discards_output_that_violates_the_builder_profile(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    discarded: list[bool] = []

    class _Builder:
        info = BuilderInfo("fixture", "Fixture", "relify-tests")
        capabilities = BuilderCapabilities(
            frozenset({BuildProfile("ivf", "parquet", "parquet")})
        )

        def build(
            self,
            _request: relify.builders.BuildRequest,
            _context: relify.builders.BuildContext,
        ) -> BuildOutput:
            return BuildOutput(
                parameters={
                    "dimension": "2",
                    "nlist": "1",
                    "ntotal": "1",
                    "posting_encoding": "source",
                    "ivf_centroids_fingerprint": (
                        "33333333-3333-3333-3333-333333333333"
                    ),
                    "ivf_centroids_uuid": "44444444-4444-4444-4444-444444444444",
                    "ivf_centroids_metadata_location": (
                        "file:///tmp/ivf-centroids/v1.metadata.json"
                    ),
                },
                index_relations={
                    "ivf_centroids": {
                        "profile": "iceberg",
                        "catalog": "lakehouse",
                    }
                },
                discard=lambda: discarded.append(True),
            )

    with pytest.raises(ValueError, match="declared 'parquet' output"):
        vectors.create_index(
            "vectors_embedding",
            column="embedding",
            key=["id"],
            config=relify.IVF(nlist=1),
            builder=_Builder(),
            wait_timeout=timedelta(seconds=5),
        )

    assert discarded == [True]
    assert vectors.index_status("vectors_embedding").state == "failed"
    assert session.indexes.list() == []


def test_coordinator_rejects_an_incompatible_builder_api(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    class _Builder:
        info = BuilderInfo(
            "future",
            "Future",
            "relify-tests",
            api_version=2,
        )
        capabilities = BuilderCapabilities(
            frozenset({BuildProfile("ivf", "parquet", "parquet")})
        )

        def build(
            self,
            _request: relify.builders.BuildRequest,
            _context: relify.builders.BuildContext,
        ) -> BuildOutput:
            raise AssertionError("an incompatible builder must not run")

    with pytest.raises(ValueError, match="unsupported builder API version"):
        vectors.create_index(
            "vectors_embedding",
            column="embedding",
            key=["id"],
            config=relify.IVF(nlist=1),
            builder=_Builder(),
        )
