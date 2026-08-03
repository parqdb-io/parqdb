# Backend Integrations

Relify backend integrations bind portable index metadata and `VectorQuery`
semantics to one host engine. The extension API is deliberately separate from
the ordinary user API.

This document is for authors of third-party backend packages. Users of the
built-in integrations should follow the [local](guides/local.md),
[Spark](guides/spark.md), or [StarRocks](guides/starrocks.md) guide instead.

## User Entry Points

A third-party integration should expose a concrete package that accepts a
caller-owned engine connection. For example, a ClickHouse integration could
look like:

```python
import relify_clickhouse

session = relify_clickhouse.connect(
    clickhouse_connection,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=iceberg,
)
```

Bundled experimental integrations follow the same shape through
`relify.experimental.spark.connect` and
`relify.experimental.starrocks.connect`. The stable local embedded
implementation remains `relify.connect`.

The table and query-builder surface is shared:

```python
documents = session.table("lakehouse.analytics.documents")
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42")
    .nprobes(32)
    .limit(10_000)
    .select(["document_id", "title"])
)
hits = session.collect(query)
```

`collect` returns a `pyarrow.Table` for every backend, including empty results.
Concrete sessions may expose additional native lazy terminals such as
`to_dataframe`, `to_relation`, or `to_sql`.

## Discovery

Configuration-driven applications can discover integrations without importing
third-party plugin modules:

```python
import relify

for backend in relify.backends.installed():
    print(backend.name, backend.distribution, backend.version)
```

`installed` reads package entry-point metadata only. Loading and connecting are
explicit:

```python
plugin = relify.backends.load("clickhouse")
session = plugin.connect(
    clickhouse_connection,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
)

# Equivalent convenience for configuration-driven applications.
session = relify.backends.connect(
    "clickhouse",
    clickhouse_connection,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
)
```

Direct integration imports remain the recommended user API because they retain
the concrete connection signature and precise type hints.

Bundled experimental integrations are intentionally absent from this stable
registry and must be imported through `relify.experimental`.

Private integrations and tests may register a plugin in process:

```python
relify.backends.register(plugin)
```

Names are unique. Built-in or installed plugins cannot be replaced unless
`replace=True` is passed explicitly.

## Plugin Package

A third-party distribution registers one plugin object:

```toml
[project.entry-points."relify.backends"]
clickhouse = "relify_clickhouse.backend:plugin"
```

The object implements the versioned public protocol:

```python
from relify.backends.v1 import (
    BackendCapabilities,
    BackendInfo,
    BackendPlugin,
    SimpleBackendPlugin,
)

plugin: BackendPlugin = SimpleBackendPlugin(
    info=BackendInfo(
        name="clickhouse",
        display_name="ClickHouse",
        distribution="relify-clickhouse",
        api_version=1,
    ),
    declared_capabilities=BackendCapabilities(...),
    connector=connect,
)
```

The loader rejects an incompatible API version before opening a connection.
Importing `relify` never imports third-party backend modules.

The connector returns a concrete session satisfying `BackendSession`:

- `backend` identifies the bound integration;
- `capabilities` reports its declared and currently available features;
- `indexes` exposes the Relify index catalog;
- `table` resolves a host table eagerly.

A plugin declaring query profiles also satisfies `QueryBackendSession`, whose
single portable terminal is `collect` returning a `pyarrow.Table`. Other
terminals are independent capabilities: declaring `EXPLAIN`, `SQL`,
`DATAFRAME`, `RELATION`, or `ANALYZE` requires the corresponding concrete
session method.

Concrete sessions may expose additional SQL, DataFrame, or engine-specific
methods. Index construction is provided by the shared table lifecycle and the
independent builder API, not by backend capability stubs.

The public catalog facade is sufficient for a separately distributed query
integration; it does not need `relify._native`:

```python
indexes = relify.open_index_catalog(
    index_catalog,
    metadata_root=metadata_root,
    storage_options=storage_options,
)
selected = indexes.select(
    source_relation,
    index=query.index,
    column=query.column,
)
```

`source_relation` is the exact portable relation mapping produced while
resolving the host table. `indexes.list_for(source_relation)` implements the
same source-centered discovery used by built-in sessions.

## Capabilities

Core capabilities are typed rather than arbitrary key/value flags:

```python
from relify.backends.v1 import QueryProfile, Terminal

profile = QueryProfile(
    family="ivf",
    source_profile="iceberg",
    index_profile="iceberg",
)

session.capabilities.status(profile)
session.capabilities.reason(profile)
session.capabilities.supports(Terminal.SQL)
inventory = session.capabilities.to_dict()
```

`plugin.declared_capabilities` is the implementation's upper bound.
`session.capabilities` is a `CapabilityReport` whose `available` subset reflects
the connected engine version and configured catalogs:

- `supported` means available on the current session;
- `unavailable` means implemented but blocked by runtime configuration; and
- `unsupported` means the integration does not implement it.

Query profiles name explicit source/index-table combinations; they do not imply
an unsupported Cartesian product. Optional terminals and maintenance
operations are also typed. Vendor-specific information belongs under the
namespaced `extensions` mapping.

Filter, projection, source resolution, distance semantics, result ordering,
and exact Iceberg snapshot binding are not optional capability flags. Claiming
an IVF query profile requires the complete semantics in
[`spec/ivf/query.md`](../spec/ivf/query.md).

## Index Builders

Construction is a separate extension boundary:

```python
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=4096),
    builder=relify.experimental.Spark(spark),
)
```

The query session resolves and pins the source relation, but the builder owns
training, assignment, and physical writes. This permits a StarRocks table to
use Spark for construction without making Spark a StarRocks backend feature.
Local and Spark sessions provide `Local()` and `Spark(session.spark)` as
defaults; query sessions without a natural builder require the explicit
argument.

Third-party builders implement `relify.builders.IndexBuilder`. Their
`BuilderCapabilities` contain typed `BuildProfile` values:

```python
from relify.builders import BuildProfile

profile = BuildProfile(
    family="ivf",
    source_profile="iceberg",
    index_profile="iceberg",
)
assert builder.capabilities.supports(profile)
```

`build(request, context)` receives an immutable request whose source relation
has already been pinned. It returns `BuildOutput` containing family parameters,
portable index relation references, and an optional discard callback for
unpublished data. The session coordinator owns asynchronous state and catalog
publication. A builder must not compile or execute vector queries, and a
backend must not advertise builder capabilities.

## Shared Planning

Backend compilers consume the public immutable values in
`relify.backends.v1`:

```text
VectorQuery
    -> IndexCatalog.select(source relation)
    -> resolve_indexed_search(...)
    -> ResolvedSearch
    -> backend-native compiler
```

`ResolvedSearch` contains the selected index, exact portable source and index
relations, dimension, `nlist`, `nprobe`, keys, vector field, projection,
predicate, limit, and the derived source-resolution requirement. A backend
still owns:

- host table and exact-snapshot resolution;
- mapping its schema to Relify's canonical Iceberg logical types;
- schema compatibility checks;
- native SQL, DataFrame, or relation compilation; and
- execution and plan formatting.

This boundary shares Relify semantics without imposing one universal physical
plan on SQL and DataFrame engines. Extension packages must import the public
`relify.backends.v1` API, not private modules such as `relify._native`.

A typical compiler first maps its host schema to `CanonicalSchema`, resolves
the projection, selects metadata through `session.indexes`, and calls:

```python
search = resolve_indexed_search(
    query,
    index=selected.identifier,
    metadata=selected.metadata,
    projection=projection,
)
```

## Contract Tests

`relify.testing` provides reusable checks over a backend-prepared fixture:

```python
from relify.testing import BackendQueryCase, check_query_backend

check_query_backend(
    session,
    [
        BackendQueryCase(
            name="stored vectors",
            profile=profile,
            query=query,
            expected=expected_arrow_table,
        ),
    ],
)
```

The contract requires at least one case for every query profile available on
the bound session. It verifies the common session surface, Arrow schema,
required `float32` distance, finite ordered results, expected source values,
distance tolerance, and explain output when declared.

Integration-specific setup remains in the plugin's tests because connection,
catalog, and table provisioning are engine-specific. Contract checks are
library development tooling; the specification remains the source of portable
behavior.
