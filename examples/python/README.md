# Python Examples

The examples use the embedded runtime and packaged Parquet datasets. Shared
setup helpers live in [`_common.py`](_common.py).

| Example | Demonstrates |
| --- | --- |
| [`quickstart.py`](local/quickstart.py) | IVF construction, automatic index selection, filtering, projection, and collection |
| [`parquet_roundtrip.py`](local/parquet_roundtrip.py) | Parquet writes, persistent registration, indexing, and recovery |
| [`exact_search.py`](local/exact_search.py) | Exact vector search without a published index |
| [`datafusion_analysis.py`](local/datafusion_analysis.py) | Analysis over vector-search results through the explicit DataFusion context |
| [`query_plans.py`](local/query_plans.py) | Query planning and runtime metrics |
| [`index_lifecycle.py`](local/index_lifecycle.py) | Snapshot refresh and removal |

Run examples after `make develop`:

```bash
uv run python -m examples.python.local.quickstart
uv run python -m examples.python.local.parquet_roundtrip
uv run python -m examples.python.local.exact_search
uv run python -m examples.python.local.datafusion_analysis
uv run python -m examples.python.local.query_plans
uv run python -m examples.python.local.index_lifecycle
```

See the [documentation index](../../docs/README.md) and
[embedded guide](../../docs/guides/local.md) for the corresponding workflows.
