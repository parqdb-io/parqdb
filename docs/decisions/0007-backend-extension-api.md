# Backend Extension API

- Status: Superseded by
  [`20260815-unified-embedded-client-server-api.md`](../rfcs/20260815-unified-embedded-client-server-api.md)
- Date: 2026-07-30

## Context

The local, Spark, and StarRocks implementations share portable metadata and
`VectorQuery`, but their Python modules are imported explicitly by the root
package and their planners repeat index metadata interpretation. Adding
DuckDB, ClickHouse, or a privately maintained engine would require editing and
releasing the ParqDB distribution.

The user API must remain concrete and table-centered. A universal backend
container or connection-type inference would hide connection ownership,
weaken type hints, and force unrelated engines into one constructor. At the
same time, configuration-driven applications need installed integrations and
their capabilities to be discoverable.

## Decision

ParqDB defines a versioned Python extension API under `parqdb.backends.v1`.
One backend distribution registers one `BackendPlugin` through the
`parqdb.backends` package entry-point group. Registration is lazy: importing
ParqDB and listing installed entry points do not import third-party plugin
modules.

Concrete integrations retain their own `connect` functions and session types.
Direct module imports are the recommended user API. The registry exposes
`installed`, `load`, `register`, and `connect` for discovery, private
integrations, tests, and configuration-driven applications.

Every connected session exposes typed `BackendInfo` and a
`CapabilityReport`. Capabilities describe explicit query profiles, native
terminals, and maintenance operations. Static plugin capabilities are
an upper bound; the bound session reports the available subset and reasons for
declared but unavailable capabilities. Normative IVF semantics are not
optional feature flags.

Index discovery and metadata interpretation produce one immutable
`ResolvedSearch` shared by backend compilers. Host relation resolution, schema
mapping and validation, native plan compilation, execution, and plan
formatting remain backend-specific. Index catalog implementations remain a
separate extension boundary.

`parqdb.open_index_catalog` and `IndexCatalog.select` expose the catalog and
metadata-loading path required by an independently packaged backend. Backend
packages do not import private native objects.

`session.collect(query)` returns one `pyarrow.Table` on every backend. This
preserves the result schema for empty searches while native DataFrame,
relation, and SQL terminals remain available on concrete sessions.
Query profiles therefore require `collect`; other declared terminals are
optional and validated independently against the bound session.

Reusable contract checks live under `parqdb.testing`, outside the
specification. A backend must supply prepared engine fixtures for every query
profile it reports as available.

## Consequences

Third-party integrations can be independently packaged and discovered without
modifying `parqdb.__init__`. Ordinary users do not need to understand the
registry or plugin protocol.

Backend packages depend on a public API version rather than private native
symbols. An incompatible extension API is rejected before connection. Query
engines retain their native plan types; construction is a separate extension
boundary.

ParqDB now owns a compatibility surface for `parqdb.backends.v1`. A future
breaking protocol requires a new major extension namespace and loader support.
Capability declarations and contract fixtures must be maintained with each
backend implementation.
