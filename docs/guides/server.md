# Relify Server

The Relify server exposes the same table, index, vector-query, and SQL APIs as
an embedded session over HTTP. It is experimental and currently runs as one
process with one ASGI worker.

## Install and Start

Install the server extra in the process that will own the catalog, index data,
and storage credentials:

```bash
python -m pip install "relify[server]"
mkdir relify-service
cd relify-service
relify config init
relify serve
```

`relify config init` writes `relify.toml`. `relify serve` loads that file from
the current directory by default. Without a file, it starts with the same
safe defaults and prints how to materialize the template.

The default configuration listens only on `127.0.0.1:8000`, stores persistent
state in `./relify`, and permits no remote source registration:

```toml
[server]
root = "./relify"
host = "127.0.0.1"
port = 8000
allowed_source_prefixes = []

[storage]

[session]
```

Set `allowed_source_prefixes` before clients register Parquet sources. Paths
are resolved relative to `relify.toml`; object-store prefixes remain URIs:

```toml
[server]
root = "./relify"
host = "0.0.0.0"
port = 8000
allowed_source_prefixes = [
  "/srv/lakehouse/documents",
  "s3://company-data/documents",
]

[storage]
aws_region = "us-east-1"

[session]
"relify.execution.query_dop" = "8"
"relify.execution.query_concurrency" = "16"
```

Keep cloud credentials in the server process environment or its credential
provider, rather than in `relify.toml`.

To use a different location, pass it explicitly:

```bash
relify serve --config /etc/relify/relify.toml
```

## Connect

The client uses the normal session facade. Registered paths are resolved on the
server and are never uploaded from the client:

```python
import relify

session = relify.connect("http://127.0.0.1:8000")
session.register_parquet("documents", "/srv/lakehouse/documents/*.parquet")
documents = session.table("documents")
```

## Embed the ASGI Application

Applications that already own an ASGI deployment can use the public factory
directly. This is an embedding API; ordinary deployments should use
`relify serve` instead.

```python
from relify.server import create_app

app = create_app(
    "/srv/relify",
    allowed_source_prefixes=["/srv/lakehouse"],
)
```

Do not configure multiple ASGI workers for the first server deployment.
SQLite catalog coordination, accepted index builds, and disposable caches are
process-local. A restart preserves published tables and indexes but abandons
in-progress builds.
