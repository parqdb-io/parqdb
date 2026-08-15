# Spark and Iceberg

The experimental Spark backend queries compatible Parquet and Iceberg indexes
with native PySpark DataFrame plans. It accepts a caller-owned
Spark Classic session; Relify does not create or configure the Spark cluster or
Iceberg catalog.

## Requirements

- standard CPython 3.11 through 3.14;
- Spark Classic 4.0 or 4.1;
- the Apache Iceberg runtime JAR matching the Spark and Scala versions;
- one Iceberg catalog configured in Spark and accessible through PyIceberg;
  and
- a SQLite Relify index catalog accessible to the driver.

Spark Connect, index construction, refresh, and remote Relify catalogs are not
implemented in 0.1.

## Install

```bash
python -m pip install "relify[spark]"
```

The extra installs PySpark, PyIceberg, pandas, and NumPy. It does not install
Java or add an Iceberg runtime JAR to Spark.

## Bind Existing Spark and PyIceberg Catalogs

Configure and start Spark using the catalog deployment's normal procedure,
then load the same logical catalog through PyIceberg:

```python
from pyiceberg.catalog import load_catalog
from pyspark.sql import SparkSession
import relify

spark = SparkSession.builder.appName("relify").getOrCreate()
iceberg = load_catalog("lakehouse")

session = relify.experimental.spark.connect(
    spark,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=iceberg,
)
```

The PyIceberg catalog name must match the name configured in Spark. Pass
`catalog_name="lakehouse"` only when the catalog object does not expose its
name. Relify validates table identity and snapshots through PyIceberg while
Spark performs the distributed reads.

## Query with a Native DataFrame

```python
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42")
    .nprobes(64)
    .limit(1_000)
    .select(["document_id", "category"])
)

hits = session.to_dataframe(query)
result = hits.groupBy("category").count()
```

`to_dataframe` keeps centroid selection, posting pruning, source filtering,
distance evaluation, and subsequent analysis in Spark. Use
`session.collect(query)` when a portable `pyarrow.Table` is required.

Inspect the Spark plan without collecting results:

```python
print(session.explain(query))
print(session.explain(query, verbose=True))
```

## Query a Parquet Index

Spark can query a Parquet index built by the local backend without binding an
Iceberg catalog:

```python
session = relify.experimental.spark.connect(
    spark,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
)
documents = session.register_parquet(
    "documents",
    "s3://lakehouse/documents/*.parquet",
)
```

The source URI must exactly match the canonical URI recorded when the local
index was built.

## Query the Same Index with DataFusion

Bind the same SQLite catalog and PyIceberg catalog in a local session:

```python
local = relify.connect(
    catalog="sqlite:///data/relify/catalog.sqlite",
    index_root="file:///data/relify/catalog-metadata",
    iceberg=iceberg,
)
documents = local.table("lakehouse.analytics.documents")
hits = local.to_arrow(
    documents.search(query_vector, column="embedding").nprobes(64).limit(100)
)
```

`index_root` must be the metadata root used by the Spark session. If
`metadata_root` was omitted there, its default is a sibling directory named
`<catalog-stem>-metadata`.

## Run the Maintained Example

From a source checkout, the maintained query example uses an already published
index and a configured Spark and PyIceberg environment:

```bash
uv run --extra spark python -m examples.python.spark.query \
  --index-catalog sqlite:///data/relify/catalog.sqlite \
  --iceberg-catalog lakehouse \
  --table analytics.documents \
  --vector 0.2,0.0 \
  --where "tenant_id = 42"
```

See [configuration](../configuration.md) for every session argument and
[troubleshooting](../troubleshooting.md#spark-and-iceberg) when one catalog can
resolve a table but the other cannot.
