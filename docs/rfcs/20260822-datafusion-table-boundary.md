# Table and Index Provider Boundaries

- Status: Proposed
- Date: 2026-08-22

## Problem

ParqDB has three independent storage boundaries. Its catalog persists logical
table definitions and index publication pointers. An ordinary table
needs format-specific scan and optional write plans. A ParqDB index needs
format-specific write plans and CID-aware read plans that perform manifest,
partition, file, row-group, and byte-range pruning.

DataFusion already provides the ordinary table boundary. It does not define
ParqDB's index layout, pruning, file inventory, or publication commit. Treating
both boundaries as one `TableProvider` hides an extension point that ParqDB
must own.

## Goals

- Use the same build and query APIs with any registered table provider.
- Replace the SQLite catalog without changing session, table, or index code.
- Add a table format without changing ParqDB catalog or common execution code.
- Add an index provider format without changing vector training, assignment,
  routing, ranking, or payload lookup.
- Allow table format and index provider format to vary independently.
- Preserve format-specific pruning, partitioning, ordering, parallelism, and
  transaction semantics.
- Support full builds now without making incremental update, delete, and
  compaction require another storage abstraction later.
- Reopen exact table and index definitions after restart and reject stale
  associations.
- Keep credentials and runtime clients outside durable metadata.
- Ship Parquet as the only default table and index provider implementation.

## Design

### Catalog boundary

The session depends on two Rust catalog contracts, not on SQLite:

```rust
trait TableCatalog { /* durable table definitions */ }
trait IndexCatalog { /* index publication and reachability */ }
```

`TableCatalog` persists exact `TableDefinition` values. `IndexCatalog` owns
logical index names, immutable metadata locations, compare-and-swap
publication, discovery, and maintenance reachability. They are independent
extension points: a session may inject different implementations, while
`LocalSession::open` uses one SQLite object for both. `with_catalog` is the
same-implementation shortcut and `with_catalogs` accepts separate trait
objects. A new metadata backend therefore does not change session or execution
code, and replacing one catalog does not require replacing the other.

This catalog is distinct from an external table catalog such as Iceberg REST
or Glue. External discovery resolves a table name to an exact native Rust
`TableDefinition`; the ParqDB catalog persists that definition. Python catalog
objects and credentials are not part of either persistent metadata or the
extension boundary.

### Table boundary

The existing `TableDefinition` persists a logical `TableIdentifier`, provider
name, and versioned provider properties. A registered DataFusion
`TableProviderFactory` normally reconstructs the runtime `TableProvider`.

```text
TableDefinition -- TableProviderFactory --> TableProvider
```

The provider produces its own plans:

```text
read:  TableProvider::scan()                        -> scan plan
write: TableProvider::insert_into(input, InsertOp)  -> write plan
```

ParqDB converts a provider definition into DataFusion's `CreateExternalTable`.
Reserved bridge properties reconstruct the table name, location, schema,
partitions, and ordering. Properties prefixed with `option.` become command
options consumed by the selected factory; all other properties remain part of
exact table identity. The provider validates its complete definition before it
is persisted, and its factory rejects missing, unknown, or unsupported runtime
options.

The built-in Parquet adapter additionally preserves the existing ParqDB glob
and multi-file registration semantics, which `CreateExternalTable` cannot
represent as one location. This is a compatibility adapter, not a closed
format enum: adding another table provider still requires only registering its
factory and properties.

Iceberg resolution is native Rust. Resolving a metadata location produces an
exact definition containing its table UUID and retained snapshot ID; the
registered native factory reconstructs a static Iceberg provider. Runtime I/O
properties stay session-local and do not enter the definition, while reopening
an index uses the pinned definition without consulting Python.

`InsertOp` is only `Append`, `Overwrite`, or `Replace`. Physical partitioning,
sorting, files, transactions, and snapshot commits belong to the returned
plan. Build and payload lookup consume the provider without inspecting its
format.

`definition-version` versions the provider-specific property encoding. It is
not the catalog schema version, index metadata version, or SQLite
`user_version`. Each provider validates the versions it understands.

Each index snapshot embeds its exact source `TableDefinition`. Its fingerprint
is a UUIDv5 over the provider, semantic identity, and ordered properties. A
provider may set `table-identity` so runtime aliases do not change semantic
identity. The full definition reopens a retained snapshot after restart; the
fingerprint supports catalog lookup and stale-index checks. A provider must pin
source state in its properties, such as an Iceberg snapshot ID or an immutable
Parquet manifest. A mutable location alone requires an explicit immutability
contract.

Change discovery is an optional, separate capability because DataFusion's
table interfaces do not expose it:

```rust
#[async_trait]
trait TableChangePlanner: Send + Sync {
    async fn plan_changes(
        &self,
        session: &dyn Session,
        from: &TableDefinition,
        to: &TableDefinition,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>>;
}
```

ParqDB registers this capability under the same provider name as the table
factory. A caller may instead supply the change plan. `None` means that a full
refresh is required.

### Index provider boundary

Index metadata persists an `IndexProviderDefinition`: a provider name plus
versioned provider properties. The name selects a separately registered
`IndexProviderFactory`. The following Rust-like interfaces define the boundary;
request and result structures are omitted here except where their role matters.

Each `IndexSnapshot` therefore persists:

```text
source_table:  exact TableDefinition used by this snapshot
index_provider: provider name + versioned provider properties
index_tables:  index-table role -> immutable provider-defined table definition
```

`index_tables` is the only name used in metadata and APIs. The index provider
interprets each table definition, including its location and layout version;
common metadata does not inspect provider-specific properties.

```rust
struct IndexTableRole(String); // validated by the index family

struct IndexProviderDefinition {
    provider: String,
    properties: BTreeMap<String, String>,
}

struct IndexTableDefinition {
    definition_version: i32,
    properties: BTreeMap<String, String>, // interpreted by IndexProvider
}

struct IndexSelection {
    cids: Option<Arc<[i32]>>, // sorted and deduplicated; None means all rows
}

struct IndexWriteInput {
    tables: BTreeMap<IndexTableRole, Arc<dyn ExecutionPlan>>,
}

struct IndexWriteResult {
    index_tables: BTreeMap<IndexTableRole, IndexTableDefinition>,
}

#[async_trait]
trait IndexProviderFactory: Send + Sync {
    async fn open(
        &self,
        session: &dyn Session,
        definition: &IndexProviderDefinition,
    ) -> Result<Arc<dyn IndexProvider>>;
}

#[async_trait]
trait IndexProvider: Send + Sync {
    async fn validate_snapshot(&self, snapshot: &IndexSnapshot) -> Result<()>;

    async fn open_index_table(
        &self,
        role: &IndexTableRole,
        table: &IndexTableDefinition,
        selection: &IndexSelection,
    ) -> Result<Arc<dyn TableProvider>>;

    async fn plan_create(
        &self,
        request: CreateIndexRequest,
        input: IndexWriteInput,
    ) -> Result<Box<dyn IndexWritePlan>>;

    async fn plan_update(
        &self,
        request: UpdateIndexRequest,
        input: IndexWriteInput,
    ) -> Result<Box<dyn IndexWritePlan>>;

    async fn plan_compact(
        &self,
        request: CompactIndexRequest,
    ) -> Result<Box<dyn IndexWritePlan>>;

    async fn managed_table_locations(
        &self,
        tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
    ) -> Result<Vec<String>>;

    async fn delete_index_tables(
        &self,
        tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
    ) -> Result<()>;
}

#[async_trait]
trait IndexWritePlan: Send + Sync {
    async fn execute(&self, context: Arc<TaskContext>)
        -> Result<IndexWriteResult>;
}
```

`IndexSelection` carries typed CIDs rather than a large SQL `IN` expression.
`open_index_table` returns a DataFusion `TableProvider` already bound to that
selection; its `scan()` implements format-specific partition, file, row-group,
and I/O pruning.

`IndexWriteInput` contains role-keyed DataFusion plans for every table produced
by the index family, including centroids, hierarchy data, postings, and update
tombstones when present. Training and assignment happen before this boundary.
`IndexWriteResult` contains the immutable `index_tables` definitions. For
Parquet, a table definition references a manifest containing the complete
object inventory; another provider may reference its native snapshot. The
index provider neither constructs common index metadata nor commits the
catalog pointer. After the catalog determines that a snapshot is unreachable,
`managed_table_locations` lets warehouse maintenance preserve referenced
objects without inspecting provider properties, and `delete_index_tables`
performs provider-specific garbage collection.

The two factory interfaces are parallel, not an inheritance hierarchy:

```text
TableDefinition        -- TableProviderFactory --> TableProvider (source data)
IndexProviderDefinition -- IndexProviderFactory --> IndexProvider
IndexProvider + selection -----------------------> TableProvider (index table)
```

The source `TableProvider` supplies rows to vector training, assignment, and
payload lookup. `IndexProvider` consumes the resulting postings plan and stores
or reads physical index tables. Returning a `TableProvider` for an index table
lets both paths reuse DataFusion execution without pretending that an
ordinary table factory owns index layout or publication.

For Parquet postings, the current write plan is:

```text
Hash(cid_bucket) -> Sort(cid_bucket, cid) per partition
-> parallel Parquet writes -> file inventory -> manifest commit
```

This cannot use DataFusion's default `DataSinkExec`, which requires one input
partition and returns only a row count. It remains a custom physical plan. Its
shuffle and sort requirements may be explicit plan nodes or declared input
requirements enforced by DataFusion's physical optimizer.

Index snapshots remain immutable. An incremental update writes new postings
segments and key-based tombstones. The common publisher combines the write
result with the source definition and index-family metadata, writes a new
metadata document, then advances the catalog pointer with compare-and-swap.
Failed or conflicting writes leave unreachable objects for later garbage
collection. Updates are delete plus insert. Compaction creates another
snapshot without changing logical results. A centroid retrain starts a new
full-build generation rather than mixing incompatible CID assignments.

The common vector engine converts a source change plan into assigned posting
rows and tombstones before calling `plan_update`. The base and target source
definitions, stable source keys, change sequence, and existing centroid
fingerprint are part of `UpdateIndexRequest`; the index provider rejects an
update across incompatible index generations. When no source change plan is
available, ParqDB performs a full refresh.

Table and index formats are independent. An Iceberg table may use a Parquet
index. Supporting an Iceberg index would require an Iceberg
`IndexProviderFactory`; an Iceberg table provider alone is not sufficient.

## Key Requirements

| Issue | Resolution |
| --- | --- |
| Exact table input | Iceberg pins table UUID and snapshot ID. Parquet pins an immutable manifest; a URI-only table is accepted only under an explicit immutability contract. |
| Table pruning | The table provider owns partition, manifest, file, row-group, and page pruning for ordinary scans. |
| Index pruning | The index reader must accept typed CID selection and guarantee format-specific I/O pruning. A large SQL `IN` list is not the contract. |
| Bucketing | Declare DataFusion hash partitioning only when hash and null semantics are identical. Otherwise buckets remain provider-internal. |
| Index write result | The writer returns immutable table definitions, not a DataFusion row count. A Parquet definition references its complete inventory manifest; other providers may reference a native snapshot. |
| Metadata vocabulary | Persist physical index tables under `index_tables`; use `table` consistently in public and internal APIs. |
| Incremental updates | New segments and tombstones produce a new immutable index snapshot. Readers merge them by source key; compaction controls read amplification. |
| Incremental identity | A stable, unique source key and ordered change sequence make insert, delete, update, and retry deterministic. |
| Change discovery | It is an optional table-format or caller capability. Absence of a change plan requires a full refresh. |
| Garbage collection | The catalog decides reachability; the index provider deletes its own physical tables. |
| Restart | Persist versioned, non-secret definitions sufficient to reconstruct both factories and verify fingerprints. |
| Writes | Ordinary table commits belong to `TableProvider`. `IndexProvider` writes immutable index objects; the common publisher and `IndexCatalog` perform metadata publication and pointer CAS. |

## Validation

- `MemTable`, Parquet, and Iceberg table tests validate the DataFusion table
  boundary, including pruning and exact-version behavior.
- Parquet index tests validate CID pruning, shuffle, local sort, parallel file
  output, inventory, manifest commit, and process reopen.
- Incremental tests cover insert, delete, update, snapshot CAS, stale-index
  rejection, merged reads, and compaction equivalence.
- Change-planner tests cover an available delta, caller-supplied changes, and
  full-refresh fallback.
- Garbage-collection tests prove that only catalog-unreachable index tables are
  passed to the selected index provider.
- A second test `IndexProviderFactory` is required to prove the Parquet index
  implementation is not hard-coded into build and query paths.

The publication manifest v1 remains Parquet-only and does not yet define delta
segments. Incremental publication and a public second index provider format
require manifest-format proposals; the internal boundary is defined here so
those changes do not rewrite the vector engine.
