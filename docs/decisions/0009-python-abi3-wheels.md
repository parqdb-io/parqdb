# Python Support Through ABI3 Wheels

- Status: Proposed
- Date: 2026-08-02

## Context

Relify currently declares support for CPython 3.12 only and produces a
CPython-minor-specific native wheel. That range is too narrow for a library
intended to integrate with existing lakehouse environments. Building a separate
66 MB wheel for every Python minor also multiplies build, upload, verification,
and retention costs without changing the Rust implementation.

The native module uses PyO3 0.28. The vendored DataFusion Python crate already
defines an `abi3` feature based on `pyo3/abi3-py310`, although Relify currently
disables that crate's default features. The Relify Python sources otherwise
have a Python 3.11 baseline except for two PEP 695 type-alias declarations that
require Python 3.12.

Python 3.10 is not a suitable new minimum for the 0.1 release because its
upstream support ends in October 2026. Python 3.15 is still a prerelease and is
outside the initial compatibility range.

## Proposed Decision

Relify 0.1 targets standard CPython 3.11, 3.12, 3.13, and 3.14 builds:

```toml
requires-python = ">=3.11,<3.15"
```

Relify will enable the PyO3 and vendored DataFusion ABI3 features and publish
one ABI3 wheel per supported platform and architecture. The expected wheel
interpreter and ABI tag is `cp310-abi3`, inherited from DataFusion's minimum
stable ABI. The `cp310` wheel tag does not declare Python 3.10 product support;
the package metadata remains authoritative and requires Python 3.11 or later.

The initial binary platform matrix is Linux x86_64 with a
`manylinux_2_28` baseline and macOS arm64 with a macOS 11 deployment target.
Other platforms may build from source, but they are not part of the 0.1 wheel
compatibility claim.

The two PEP 695 type aliases will use a Python 3.11-compatible type-alias form.
No other compatibility shim for Python 3.10 will be added.

Each built wheel must be installed unchanged into clean CPython 3.11, 3.12,
3.13, and 3.14 environments. The core suite and installed-package build/search
smoke test must pass in every environment. Optional Iceberg, Spark, and
StarRocks dependency stacks must resolve and import on every Python minor. Their
service-backed integration gates run in the canonical Python 3.12 environment.

Free-threaded CPython builds, including the `cp314t` ABI, are not covered by
this decision. They require separate native wheels and concurrency validation.

## Acceptance Criteria

This decision may move to `Accepted` only when:

- Maturin produces one `cp310-abi3` wheel for each advertised platform;
- the wheel passes existing metadata, license, SBOM, `RECORD`, reproducibility,
  and isolated smoke verification;
- the same wheel installs and passes the required test matrix on CPython 3.11,
  3.12, 3.13, and 3.14;
- the dependency resolver finds supported PyArrow and optional-backend
  dependencies for every advertised Python minor;
- no extension code depends on a CPython API excluded from the stable ABI; and
- README, package classifiers, CI, and release documentation agree on the
  supported range.

If Python 3.14 fails only in an optional dependency, the release must either fix
or constrain that dependency before accepting this decision. Core-only success
is insufficient while the corresponding integration remains advertised for
that Python version.

## Consequences

One native artifact per platform replaces one artifact per Python minor. The
release workflow still tests every supported Python minor because ABI
compatibility does not prove Python-source or dependency compatibility.

The package can reach users on several active Python releases without carrying
Python 3.10 compatibility code. New stable Python minors remain unsupported
until their complete matrix is green and the declared upper bound is updated.

ABI3 restricts the native module to CPython's stable C API. A future feature
that requires an unavailable CPython API must be redesigned, isolated in a
separate version-specific module, or recorded as a replacement ADR.
