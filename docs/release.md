# Release Process

Relify releases the Python package as platform-specific wheels. The Rust
workspace crates are implementation components and are not released
independently yet.

This document governs normal package releases from the public repository.

## Versioning

Relify uses PEP 440 for the Python distribution and SemVer for the Cargo
workspace. A release candidate therefore has different canonical spellings in
the two manifests:

| Purpose | First candidate | Final release |
| --- | --- | --- |
| Python package | `0.1.0rc1` | `0.1.0` |
| Cargo workspace | `0.1.0-rc.1` | `0.1.0` |
| Git tag | `v0.1.0rc1` | `v0.1.0` |
| GitHub release | `Relify 0.1.0rc1` (pre-release) | `Relify 0.1.0` |

The first public package is `0.1.0rc1`. At that point the 0.1 feature and API
scope is frozen; only release-blocking fixes are expected before `0.1.0`. A
fix after publication produces `0.1.0rc2`, never a replacement artifact under
the `0.1.0rc1` version.

## Branching Model

Relify uses one development mainline. `main` is the only long-lived development
branch and should remain releasable. Normal changes use short-lived branches,
such as `feat/<topic>`, `fix/<topic>`, or `docs/<topic>`, and enter `main`
through pull requests. Relify does not use a `develop` branch.

Create `release/X.Y` from the latest `main` only when a release series needs a
separate stabilization or maintenance line. The branch is not another
development line: it accepts release metadata and fixes required by that
series. Product fixes land on `main` first and are then backported to the
release branch. Delete the branch when the series is no longer maintained.

All candidates, final releases, and patches in a release series use the same
branch. Signed tags identify exact releases, for example:

```text
release/0.1
  v0.1.0rc1
  v0.1.0rc2
  v0.1.0
  v0.1.1
```

Do not create a branch for each candidate, such as `release/0.1.0rc1`. When no
parallel stabilization or maintenance is needed, a release may be tagged
directly from `main` without creating `release/X.Y`.

## Prepare

1. Start from a clean, current `main` checkout. For the first candidate, create
   `release/X.Y` only if the release needs a separate stabilization line. For
   later candidates, the final release, and patches, reuse that same branch.
2. Confirm that every product fix on the release branch is already present on
   `main` or has an explicit backport plan.
3. Update `pyproject.toml` and the workspace `Cargo.toml` using the matching
   Python and Cargo forms from the table above.
4. Update internal crate dependency versions if the workspace version changes.
5. Move the relevant `CHANGELOG.md` entries from `Unreleased` to a dated
   release heading.
6. Run the complete validation:

   ```bash
   make sync
   make develop
   make check
   make test-remote-storage
   make test-spark-iceberg
   make test-starrocks
   make audit
   make verify-datafusion-vendor
   make benchmark-smoke
   ```

The remote-storage gate requires Docker for the MinIO S3 test and Java plus
Maven for the MiniDFS HDFS test. The experimental Spark gate starts a local
Spark session with the configured Iceberg runtime and verifies that Spark-built
Iceberg index tables are queryable through both Spark and DataFusion. The
experimental StarRocks gate requires a maintained StarRocks 3.5.1 or later
deployment and shared Iceberg catalog configured as described in
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Missing prerequisites fail the release
gate; the tests are never silently skipped.

Any changed portable behavior must already be represented in `spec/` and
`spec/fixtures/v1/`.

## Build and Verify

Build wheels from the exact commit that will be tagged:

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" make verify-package
```

Build and test every advertised Python/platform combination on that platform.
The proposed 0.1 target is standard CPython 3.11 through 3.14 using one ABI3
wheel for Linux x86_64 (`manylinux_2_28`) and one for macOS arm64 (macOS 11 or
later), as defined by
[`0009-python-abi3-wheels.md`](decisions/0009-python-abi3-wheels.md). Install the
same wheel unchanged into clean environments for every supported Python minor
and run the installed-package core suite plus optional-dependency smoke checks.
The service-backed integration gates run separately in the canonical Python
3.12 environment. Do not advertise a Python minor until its complete matrix is
green. Free-threaded CPython requires separate artifacts and is not part of the
0.1 target.

`make verify-package` regenerates the locked CycloneDX SBOM, builds the wheel,
validates package metadata, license files, typing markers, the native extension,
SBOM contents, and every `RECORD` hash. It then rebuilds the wheel and requires
an identical SHA-256 digest before installing the same artifact with all
advertised extras into isolated CPython 3.11, 3.12, 3.13, and 3.14 environments
and running the Python and interoperability suites plus a real local
build-and-search smoke test in each one.

## Publish

The `Release` workflow is the only supported PyPI publishing path. PyPI trusts
`.github/workflows/release.yml` through the `pypi` GitHub environment and OIDC;
the repository does not store a PyPI API token. The workflow rebuilds and
verifies both platform wheels, publishes them together, and retains the exact
wheels and `SHA256SUMS` as one GitHub Actions artifact.

1. Create and push the signed tag using the canonical Python version, for
   example `v0.1.0rc1` or `v0.1.0`, on the verified commit.
2. Require the `Release` workflow to complete successfully. Do not upload
   another build manually when the workflow fails.
3. Download the `release-<tag>` workflow artifact and create the GitHub release
   from the changelog entry. Attach the two wheels and `SHA256SUMS`, and mark
   release candidates as pre-releases.
4. Verify the exact version from PyPI in a new environment, for example with
   `python -m pip install relify==0.1.0rc1`.
5. Restore an empty `Unreleased` section in the changelog.

Do not rebuild artifacts after tagging. If verification fails, fix the issue
and publish a new version rather than replacing an existing artifact.

The 0.1 sequence is `0.1.0rc1`, optionally `0.1.0rc2` and later candidates,
then `0.1.0`. The final release requires newly versioned and verified artifacts
even when no implementation code changed after the last candidate.
