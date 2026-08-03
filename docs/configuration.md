# Configuration

This reference collects the configuration surface needed to install Relify,
open catalogs and storage, build indexes, and bind compute engines. For method
behavior and return types, use the [Python API](python-api.md).

## Packages

| Install | Adds |
| --- | --- |
| `relify` | Local DataFusion, native builder, Parquet, SQLite catalog |
| `relify[iceberg]` | PyIceberg support for querying Spark-built indexes locally |
| `relify[spark]` | Experimental Spark Classic and Iceberg integration |
| `relify[starrocks]` | Experimental StarRocks Arrow Flight SQL integration |

The 0.1 wheels target CPython 3.11 through 3.14, Linux x86_64 with glibc 2.28
or later, and macOS arm64 11 or later.

## Local Session

### Combined local state

```python
session = relify.connect("./relify-data")
```

| Argument | Meaning |
| --- | --- |
| `root` | Local directory containing `catalog.sqlite` and the default index warehouse |
| `index_root` | Optional independent `file`, `s3`, or `hdfs` warehouse URI |
| `storage_options` | String key/value options passed to object-store or HDFS clients |
| `iceberg` | Optional PyIceberg catalog used to query Iceberg sources and indexes |

### Explicit catalog and warehouse

```python
session = relify.connect(
    catalog="sqlite:///var/lib/relify/catalog.sqlite",
    index_root="s3://lakehouse-indexes/relify",
    storage_options={"aws_region": "us-east-1"},
)
```

`root` and `catalog` are mutually exclusive. An explicit catalog URI must use
`sqlite:///` with an absolute path and requires `index_root`.

## Storage URIs

Relify accepts canonical absolute `file`, `s3`, and `hdfs` URIs. Credentials,
query strings, and fragments do not belong in persisted URIs.

```text
file:///data/relify
s3://lakehouse-indexes/relify
hdfs://namenode:8020/warehouse/relify
```

Parquet source registration also accepts a local absolute path, one concrete
URI, a directory, or `*` wildcards inside path segments. A wildcard does not
cross a `/` boundary.

### S3 and compatible stores

The underlying Apache Arrow Rust object-store client reads standard AWS
environment credentials. Explicit options must be strings:

```python
storage_options = {
    "aws_access_key_id": "access-key",
    "aws_secret_access_key": "secret-key",
    "aws_region": "us-east-1",
}
```

For an S3-compatible endpoint:

```python
storage_options = {
    "aws_access_key_id": "access-key",
    "aws_secret_access_key": "secret-key",
    "aws_region": "us-east-1",
    "aws_endpoint": "http://127.0.0.1:9000",
    "aws_allow_http": "true",
    "aws_virtual_hosted_style_request": "false",
}
```

Prefer environment or workload-identity credentials in deployed systems. Do
not put secrets in a catalog URI or committed source file.

### HDFS

Use an absolute URI containing the NameNode authority:

```python
session = relify.connect(
    "./state",
    index_root="hdfs://namenode:8020/warehouse/relify",
)
```

HDFS options are passed to the native HDFS client. Cluster XML configuration,
authentication, DNS, and network reachability remain deployment concerns.

## Parquet Source Registration

```python
import pyarrow

session.register_parquet(
    "documents",
    "s3://lakehouse/documents/*/part-*.parquet",
    table_partition_cols=[("date", pyarrow.date32())],
    parquet_pruning=True,
    file_extension=".parquet",
    skip_metadata=True,
    schema=None,
    file_sort_order=None,
)
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `table_partition_cols` | `[]` | Hive-style partition columns and Arrow types |
| `parquet_pruning` | `True` | Enable Parquet statistics pruning |
| `file_extension` | `.parquet` | Restrict directory scans to matching files |
| `skip_metadata` | `True` | Ignore file-schema metadata that could cause schema conflicts |
| `schema` | inferred | Explicit Arrow schema when inference is unsuitable |
| `file_sort_order` | none | Declared physical sort metadata for planning |

The registration is persisted in the SQLite table catalog.

## IVF Index

```python
config = relify.IVF(
    nlist=4096,
    store_vectors=True,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `nlist` | required | Positive number of IVF clusters |
| `store_vectors` | `True` | Store exact vectors in postings for index-only distance evaluation |

Relify 0.1 supports IVF-Flat with squared L2 distance. `nprobes` is selected on
each query and must not exceed `nlist`.

## Local Builder

```python
builder = relify.Local(
    threads=None,
    max_row_group_rows=None,
    write_batch_rows=8192,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `threads` | available process parallelism | Isolated centroid-training worker count |
| `max_row_group_rows` | automatic | Explicit maximum rows in a postings row group |
| `write_batch_rows` | `8192` | Maximum Arrow rows passed to each writer batch |

## Physical Output

```python
writer_options = relify.WriteOptions(
    partitions=None,
    compression="uncompressed",
    target_file_size=512 * 1024 * 1024,
)
```

`partitions` controls local writer concurrency or Spark output parallelism. The
supported compression values are `uncompressed`, `snappy`, `lz4`, `lz4_raw`,
and levelled `gzip(n)`, `brotli(n)`, or `zstd(n)` values accepted by the
configuration validator.

## Query Controls

```python
query = (
    documents.search(query_vector, column="embedding", index=None)
    .where("tenant_id = 42")
    .nprobes(64)
    .limit(100)
    .select(["document_id", "title"])
)
```

| Control | Meaning |
| --- | --- |
| `column` | Source vector column; inferred only when selection is unambiguous |
| `index` | Optional explicit index name |
| `where` | SQL predicate applied to source rows before final Top-K |
| `nprobes` | Number of IVF clusters scanned |
| `limit` | Requested result count |
| `select` | Source columns returned before `_distance` |
| `bypass_vector_index()` | Exact scan without a published index |

Batch query vectors are not supported. A query vector must be one-dimensional
and match the indexed vector dimension.

## Spark Session

```python
session = relify.experimental.spark.connect(
    spark,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=iceberg,
    catalog_name=None,
    metadata_root=None,
    storage_options=None,
)
```

`index_catalog` is required. `metadata_root` defaults to a sibling directory
named `<catalog-stem>-metadata`. `storage_options` applies to Relify metadata,
not Spark or PyIceberg data-plane credentials. `catalog_name` defaults to the
PyIceberg catalog's name.

## StarRocks Session

```python
session = relify.experimental.starrocks.connect(
    connection,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=iceberg,
    catalog_name="lakehouse",
    metadata_root=None,
    storage_options=None,
    index_namespace=("relify",),
)
```

The connection must implement the Arrow Flight SQL ADBC DBAPI cursor surface.
`catalog_name` must match the external Iceberg catalog in StarRocks.
`index_namespace` identifies the Iceberg namespace containing index tables.

## PyIceberg Catalogs

Configure PyIceberg through its normal environment, YAML, or
`load_catalog(name, **properties)` mechanism. Relify requires the catalog
object to resolve exact table UUIDs, schemas, metadata locations, and snapshot
IDs. Its logical name must match the host engine's catalog name.
