# Contributing to Relify

Relify is an early-stage implementation of the formats and behavior in
[`spec/`](spec/README.md). Contributions should keep the portable contract and
the implementation synchronized.

## Development Setup

Install Python 3.12 for the canonical development environment,
[uv](https://docs.astral.sh/uv/), and
[cargo-deny](https://embarkstudios.github.io/cargo-deny/). The pinned Rust
toolchain is installed automatically by rustup.

```bash
cargo install cargo-deny --version 0.20.2 --locked
make sync
make develop
make check
```

`make develop` installs the native extension into the uv environment. Run it
again after changing Rust code that is exercised by Python.

## Change Scope

Keep each change focused on one observable behavior or component boundary.
The repository sources of truth are, in order:

1. `spec/` for portable formats and behavior;
2. `docs/architecture.md` for implemented component boundaries;
3. `docs/python-api.md` and `docs/roadmap.md` for the supported API and scope.

Schema and query changes must update the affected specification and shared
fixtures. Architectural changes that add a backend, storage format, metric, or
distributed component must update the roadmap and document the architectural
decision before implementation.

Public Python examples must have a smoke or integration test. Avoid
compatibility layers for behavior that has never been released.

## Tests and Quality Gates

Run the complete local gate before requesting review:

```bash
make format
make check
make audit
```

Tests are organized by component under `rust/*/src/tests/` and
`tests/python/`. Add tests at the narrowest layer that observes the behavior,
then add an end-to-end test when the behavior crosses crate or language
boundaries.

Third-party execution integrations use the public extension contract in
[`docs/backends.md`](docs/backends.md). Backend packages must declare typed
capabilities and run `relify.testing.check_query_backend` over prepared cases
for every query profile available in the test session.

Shared format fixtures live in `spec/fixtures/v1/`. Regenerate them with:

```bash
make fixtures
```

Review the resulting diff and run `tests/python/test_spec_fixtures.py`.

Integration tests use one capability configuration. Copy the template and
configure only the environments available on your machine:

```bash
cp tests/test-env.example.toml tests/test-env.toml
```

The local file is ignored by Git. `${NAME}` placeholders are expanded from the
process environment, so credentials do not need to be stored in the file.
Tests declare their requirements with `@pytest.mark.requires(...)`:

```bash
uv run pytest --test-env tests/test-env.toml
uv run pytest --test-env tests/test-env.toml --capabilities
uv run pytest --test-env tests/test-env.toml \
  --require s3,iceberg,starrocks
```

An omitted capability is skipped. A configured capability is probed before its
tests run; connection, authentication, and dependency failures fail the test
instead of becoming skips. `--require` makes missing CI capabilities a startup
error.

Convenience targets use the same capability framework. `make test-s3` starts a
pinned MinIO container and removes it afterward. `make test-hdfs` runs the
storage contract against Hadoop's `MiniDFSCluster`; it requires Java 8 through
17, Maven, and Kerberos client tools including `kdestroy`.
`make test-spark-iceberg` and `make test-starrocks` use `TEST_ENV`, which
defaults to `tests/test-env.toml`:

```bash
make test-s3
make test-hdfs
make test-spark-iceberg
make test-starrocks
```

The StarRocks conformance test requires a running StarRocks 3.5.1 or later
deployment and a shared Iceberg catalog. `[iceberg].name` and
`[starrocks].catalog_name` must identify the same catalog. The test loads both
portable IVF fixtures into a temporary namespace, queries them through
StarRocks, compares every result, and removes the namespace.

## Benchmarks

Performance changes must use a fixed seed and report dataset size, dimension,
index parameters, hardware, build time, throughput, and Recall@K. The
reproducible benchmark entry point is:

```bash
make benchmark-smoke
```

Representative measurements use `python -m benchmarks.build` followed by
`python -m benchmarks.query`; see [`benchmarks/README.md`](benchmarks/README.md).

## Pull Requests

Describe the user-visible effect, specification impact, tests run, and
benchmark impact. Do not mix generated files, refactors, and behavioral changes
without explaining why they must land together.

PR titles and commit subjects use
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>[optional scope][!]: <description>
```

Supported types are:

- `feat` for a new user-visible capability;
- `fix` for a bug fix;
- `perf` for a performance improvement;
- `refactor` for an internal change without new behavior;
- `docs` for documentation only;
- `test` for tests only;
- `build` for build system or dependency changes;
- `ci` for automation changes; and
- `chore` for repository maintenance that fits no type above.

Use a short lowercase scope when it makes the affected component clearer, such
as `catalog`, `meta`, `ivf`, `python`, `storage`, or `spec`. Write the
description in the imperative mood without a trailing period:

```text
feat: persist Parquet table definitions
fix(catalog): reject conflicting table registrations
perf(ivf): reuse assignment buffers
docs: clarify the Parquet relation profile
```

Append `!` before `:` and include a `BREAKING CHANGE:` footer when a change
intentionally breaks a released API or format. These rules apply to new work;
do not rewrite existing repository history solely to change commit subjects.
