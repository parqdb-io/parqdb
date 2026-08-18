# SQLite Catalog Format Identity

- Status: Accepted
- Date: 2026-08-15

## Context

ParqDB uses SQLite for its embedded catalog. No stable version has been
published, so the current prerelease layout is the format-1 baseline rather
than a migration target. A schema number alone cannot establish that an
arbitrary SQLite database belongs to ParqDB.

The catalog schema generation is an implementation detail. It is independent
of the public ParqDB index schema stored in open index metadata.

## Decision

ParqDB SQLite catalogs use both SQLite header fields:

| Field | Current value | Purpose |
| --- | ---: | --- |
| `application_id` | `0x50514442` (`PQDB`) | Identifies a ParqDB catalog |
| `user_version` | `1` | Identifies the current internal table layout |

A new empty database is initialized with both values and the complete current
schema. An existing database opens only when both values match. Unversioned
existing schemas and unrelated application IDs are rejected before catalog
tables are accessed.

ParqDB does not provide catalog migrations or backward-compatibility guarantees
before a stable release. Users must recreate incompatible prerelease catalogs
and rebuild or republish their indexes. There is no compatibility reader for
earlier development layouts that also happened to use `user_version = 1`.

## Stable-Release Boundary

The first stable release will define the baseline from which catalog
compatibility is supported. Any later layout change must choose explicitly
between:

- retaining a backward-compatible reader;
- providing a transactional migration from a supported baseline; or
- rejecting the old layout with a documented export or rebuild path.

No stable compatibility policy is inferred from prerelease schema generations.
