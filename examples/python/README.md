# Python Examples

Examples are grouped by execution backend. Shared dataset and setup helpers
remain in [`_common.py`](_common.py).

For installation and explanatory workflows, use the
[documentation index](../../docs/README.md). These files are maintained,
executable companions to the [local](../../docs/guides/local.md),
[Spark](../../docs/guides/spark.md), and
[StarRocks](../../docs/guides/starrocks.md) guides.

## Local DataFusion

The local examples are self-contained. They reuse Relify's packaged Parquet
datasets inside temporary workspaces that are removed after each run.

| Example | Demonstrates |
| --- | --- |
| [`quickstart.py`](local/quickstart.py) | Asynchronous IVF construction, automatic index selection, filtering, projection, and collection |
| [`parquet_roundtrip.py`](local/parquet_roundtrip.py) | DataFrame writes, persistent table registration, indexing, and recovery in a new session |
| [`exact_search.py`](local/exact_search.py) | Exact vector search without a published index |
| [`datafusion_analysis.py`](local/datafusion_analysis.py) | SQL analysis over lazy vector-search results in the native DataFusion context |
| [`query_plans.py`](local/query_plans.py) | Query planning and runtime operator metrics |
| [`index_lifecycle.py`](local/index_lifecycle.py) | Snapshot refresh, catalog inspection, removal, and metadata recovery |

Run any local example after `make develop`:

```bash
uv run python -m examples.python.local.quickstart
uv run python -m examples.python.local.parquet_roundtrip
uv run python -m examples.python.local.exact_search
uv run python -m examples.python.local.datafusion_analysis
uv run python -m examples.python.local.query_plans
uv run python -m examples.python.local.index_lifecycle
```

## Spark

[`spark/build_and_query.py`](spark/build_and_query.py) binds a caller-configured
Spark Classic session and matching PyIceberg catalog, creates an IVF index when
needed, and queries it as a native PySpark DataFrame.

The Spark process must already be configured with the named Iceberg catalog.
PyIceberg must resolve the same catalog name from its configuration:

```bash
uv run --extra spark python -m examples.python.spark.build_and_query \
  --index-catalog sqlite:///data/relify/catalog.sqlite \
  --iceberg-catalog lakehouse \
  --table analytics.documents \
  --vector 0.2,0.0 \
  --where "tenant_id = 42"
```

## StarRocks

[`starrocks/query.py`](starrocks/query.py) connects to an existing Arrow Flight
SQL endpoint and queries a Spark-built Iceberg index. StarRocks and PyIceberg
must expose the same logical Iceberg catalog. The Flight SQL password is read
from `STARROCKS_PASSWORD` by default:

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
