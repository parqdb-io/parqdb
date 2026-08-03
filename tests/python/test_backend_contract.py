from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
import relify
from relify.backends.v1 import (
    BackendCapabilities,
    BackendInfo,
    CapabilityReport,
    QueryProfile,
    Terminal,
)
from relify.testing import BackendQueryCase, check_query_backend

PROFILE = QueryProfile("ivf", "parquet", "parquet")
CAPABILITIES = BackendCapabilities(
    query_profiles=frozenset({PROFILE}),
    terminals=frozenset({Terminal.COLLECT, Terminal.EXPLAIN}),
)


def test_backend_contract_checks_portable_collection_and_explain() -> None:
    expected = _result_table([1, 2], [0.0, 1.0])
    session = _Session(expected)
    case = BackendQueryCase(
        "stored vectors",
        PROFILE,
        _query(),
        expected,
    )

    check_query_backend(session, [case])

    assert session.explained


def test_backend_contract_preserves_empty_result_schema() -> None:
    expected = _result_table([], [])
    session = _Session(expected)

    check_query_backend(
        session,
        [BackendQueryCase("empty", PROFILE, _query(), expected)],
    )


def test_backend_contract_requires_cases_for_every_available_profile() -> None:
    second = QueryProfile("ivf", "iceberg", "iceberg")
    capabilities = BackendCapabilities(
        query_profiles=frozenset({PROFILE, second}),
        terminals=CAPABILITIES.terminals,
    )
    session = _Session(
        _result_table([1], [0.0]),
        capabilities=capabilities,
    )

    with pytest.raises(AssertionError, match="have no contract case"):
        check_query_backend(
            session,
            [
                BackendQueryCase(
                    "parquet",
                    PROFILE,
                    _query(),
                    _result_table([1], [0.0]),
                )
            ],
        )


def test_backend_contract_rejects_wrong_distance_type() -> None:
    expected = _result_table([1], [0.0])
    actual = pa.table(
        {
            "document_id": pa.array([1], type=pa.int64()),
            "_distance": pa.array([0.0], type=pa.float64()),
        }
    )

    with pytest.raises(AssertionError, match="returned schema"):
        check_query_backend(
            _Session(actual),
            [BackendQueryCase("wrong type", PROFILE, _query(), expected)],
        )


def _query() -> relify.VectorQuery:
    return relify.VectorQuery(
        relify.TableIdentifier("relify", ("fixtures",), "source"),
        (0.0, 0.0),
        column="embedding",
    )


def _result_table(ids: list[int], distances: list[float]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("document_id", pa.int64(), nullable=False),
            pa.field("_distance", pa.float32(), nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(distances, type=pa.float32()),
        ],
        schema=schema,
    )


class _Session:
    backend = BackendInfo("contract", "Contract", "relify-contract")
    indexes = object()

    def __init__(
        self,
        result: pa.Table,
        *,
        capabilities: BackendCapabilities = CAPABILITIES,
    ) -> None:
        self.capabilities = CapabilityReport.fully_available(capabilities)
        self._result = result
        self.explained = False

    def table(self, identifier: Any) -> object:
        return object()

    def collect(self, query: Any) -> pa.Table:
        return self._result

    def explain(self, query: Any, **options: Any) -> str:
        self.explained = True
        return "physical plan"
