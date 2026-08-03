from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pyarrow

from ..backends.v1 import (
    QueryBackendSession,
    QueryProfile,
    Terminal,
    schema_from_pyarrow,
)
from ..query import VectorQuery


@dataclass(frozen=True, slots=True)
class BackendQueryCase:
    """One prepared portable query case for a backend contract test."""

    name: str
    profile: QueryProfile
    query: VectorQuery
    expected: pyarrow.Table
    distance_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("backend query case name must not be empty")
        if not isinstance(self.query, VectorQuery):
            raise TypeError("backend query case requires a relify.VectorQuery")
        if not isinstance(self.expected, pyarrow.Table):
            raise TypeError("backend query case expected result must be an Arrow table")
        if (
            not isinstance(self.distance_tolerance, float)
            or not math.isfinite(self.distance_tolerance)
            or self.distance_tolerance < 0
        ):
            raise ValueError("distance_tolerance must be a non-negative finite float")


def check_query_backend(
    session: QueryBackendSession,
    cases: list[BackendQueryCase],
) -> None:
    """Check every available query profile against prepared portable cases."""
    if not isinstance(session, QueryBackendSession):
        raise TypeError(
            "session does not implement relify's query backend session contract"
        )
    if not session.capabilities.supports(Terminal.COLLECT):
        raise AssertionError("query backend must expose the collect terminal")
    if not cases:
        raise ValueError("backend query contract requires at least one case")

    covered = {case.profile for case in cases}
    missing = session.capabilities.available.query_profiles - covered
    if missing:
        names = ", ".join(
            f"{profile.family}:{profile.source_profile}->{profile.index_profile}"
            for profile in sorted(missing)
        )
        raise AssertionError(f"query profiles have no contract case: {names}")

    for case in cases:
        if not session.capabilities.supports(case.profile):
            raise AssertionError(
                f"case {case.name!r} targets an unavailable query profile"
            )
        actual = session.collect(case.query)
        _check_query_result(
            actual,
            case.expected,
            tolerance=case.distance_tolerance,
            case=case.name,
        )

    if session.capabilities.supports(Terminal.EXPLAIN):
        explain = getattr(session, "explain", None)
        if not callable(explain):
            raise AssertionError(
                "backend reports the explain terminal but does not implement it"
            )
        plan = explain(cases[0].query)
        if not isinstance(plan, str) or not plan.strip():
            raise AssertionError("backend explain terminal returned an empty plan")


def _check_query_result(
    actual: pyarrow.Table,
    expected: pyarrow.Table,
    *,
    tolerance: float,
    case: str,
) -> None:
    if not isinstance(actual, pyarrow.Table):
        raise AssertionError(f"case {case!r} did not return a pyarrow.Table")
    if schema_from_pyarrow(actual.schema) != schema_from_pyarrow(expected.schema):
        raise AssertionError(
            f"case {case!r} returned schema {actual.schema}, expected {expected.schema}"
        )
    if "_distance" not in actual.column_names:
        raise AssertionError(f"case {case!r} has no _distance result field")
    distance = actual.schema.field("_distance")
    if not pyarrow.types.is_float32(distance.type) or distance.nullable:
        raise AssertionError(f"case {case!r} _distance must be required Arrow float32")

    actual_rows = actual.to_pylist()
    expected_rows = expected.to_pylist()
    actual_distances = [float(row["_distance"]) for row in actual_rows]
    if any(not math.isfinite(value) for value in actual_distances):
        raise AssertionError(f"case {case!r} returned a non-finite distance")
    if actual_distances != sorted(actual_distances):
        raise AssertionError(f"case {case!r} is not ordered by distance")
    if len(actual_rows) != len(expected_rows):
        raise AssertionError(
            f"case {case!r} returned {len(actual_rows)} rows, "
            f"expected {len(expected_rows)}"
        )

    projection = [name for name in actual.column_names if name != "_distance"]
    actual_rows.sort(key=lambda row: _row_key(row, projection))
    expected_rows.sort(key=lambda row: _row_key(row, projection))
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        if any(actual_row[name] != expected_row[name] for name in projection):
            raise AssertionError(f"case {case!r} returned different source values")
        if not math.isclose(
            float(actual_row["_distance"]),
            float(expected_row["_distance"]),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise AssertionError(f"case {case!r} returned a different distance")


def _row_key(row: dict[str, Any], projection: list[str]) -> tuple[str, ...]:
    return (
        *(repr(row[name]) for name in projection),
        repr(row["_distance"]),
    )
