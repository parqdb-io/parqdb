# StarRocks and Iceberg

The experimental StarRocks backend compiles a Relify query into one StarRocks
SQL statement over exact Iceberg snapshots. It is query-only and accepts a
caller-owned Arrow Flight SQL ADBC connection.

The current planner accepts source-encoded L2 IVF indexes. Cosine and LVQ
postings are currently local-backend capabilities.

## Requirements

- StarRocks 3.5.1 or later with Arrow Flight SQL enabled;
- one Iceberg catalog registered in StarRocks;
- the same logical catalog accessible through PyIceberg;
- a compatible published Relify Iceberg index; and
- a SQLite Relify index catalog and metadata root accessible to the client.

Relify does not deploy StarRocks, create its external catalog, or copy Iceberg
data into StarRocks storage.

## Install

```bash
python -m pip install "relify[starrocks]"
```

## Connect

```python
import os

import adbc_driver_flightsql.dbapi as flight_sql
from adbc_driver_manager import DatabaseOptions
from pyiceberg.catalog import load_catalog
import relify

connection = flight_sql.connect(
    uri="grpc://starrocks.example.com:9408",
    db_kwargs={
        DatabaseOptions.USERNAME.value: "root",
        DatabaseOptions.PASSWORD.value: os.environ["STARROCKS_PASSWORD"],
    },
)
iceberg = load_catalog("lakehouse")

session = relify.experimental.starrocks.connect(
    connection,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=iceberg,
    catalog_name="lakehouse",
)
```

`catalog_name` is the Iceberg catalog name registered in StarRocks. It may be
omitted when the PyIceberg catalog object exposes that same name. The Relify
session does not close the supplied Flight SQL connection.

## Query

Table identifiers omit the already-bound catalog name:

```python
documents = session.table("analytics.documents")
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42")
    .nprobes(64)
    .limit(1_000)
    .select(["document_id", "category"])
)

hits = session.collect(query)
```

`collect` returns a `pyarrow.Table`. Relify validates its schema and requires a
finite, non-null `float32` `_distance` column.

Inspect generated SQL or the StarRocks plan:

```python
print(session.to_sql(query))
print(session.explain(query))
```

The generated statement performs centroid Top-K, posting pruning, source
resolution, filtering, squared L2 evaluation, and final Top-K in StarRocks.
Every Iceberg relation uses the snapshot ID recorded in Relify metadata.

## Run the Maintained Example

From a source checkout, the maintained example reads the password from
`STARROCKS_PASSWORD` by default:

```bash
uv run --extra starrocks python -m examples.python.starrocks.query \
  --flight-uri grpc://starrocks.example.com:9408 \
  --index-catalog sqlite:///data/relify/catalog.sqlite \
  --iceberg-catalog lakehouse \
  --host-catalog lakehouse \
  --table analytics.documents \
  --vector 0.2,0.0 \
  --where "tenant_id = 42"
```

## Current Boundary

The backend does not build with StarRocks compute, query Parquet through
`FILES()`, use StarRocks native vector indexes, or expose a DataFrame API. Its
SQLite index catalog is intended for development and single-coordinator use.
See [current limitations](../limitations.md) before deployment and
[troubleshooting](../troubleshooting.md#starrocks-and-iceberg) for catalog and
Flight SQL checks.
