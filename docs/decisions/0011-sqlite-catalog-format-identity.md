# SQLite Catalog Format Identity

- Status: Accepted
- Date: 2026-08-15

## Context

Relify uses SQLite for its embedded catalog. Earlier prerelease revisions used
internal schema generations 1 through 3. The current layout adds shared-IVF
coordination state and is generation 4.

Resetting the generation to 1 would be unsafe: an old prerelease database could
then be mistaken for the current layout. A schema number also cannot establish
that an arbitrary SQLite database belongs to Relify.

The catalog schema generation is an implementation detail. It is independent
of the public Relify index schema stored in open index metadata.

## Decision

Relify SQLite catalogs use both SQLite header fields:

| Field | Current value | Purpose |
| --- | ---: | --- |
| `application_id` | `0x524c4659` (`RLFY`) | Identifies a Relify catalog |
| `user_version` | `4` | Identifies the current internal table layout |

A new empty database is initialized with both values and the complete current
schema. An existing database opens only when both values match. Catalogs with
prerelease schema generations 1, 2, or 3, unversioned existing schemas, and
unrelated application IDs are rejected before catalog tables are accessed.

Relify does not provide catalog migrations or backward-compatibility guarantees
before a stable release. Users must recreate incompatible prerelease catalogs
and rebuild or republish their indexes. The numeric value 4 prevents format
collisions; it does not imply support for generations 1 through 3.

## Stable-Release Boundary

The first stable release will define the baseline from which catalog
compatibility is supported. Any later layout change must choose explicitly
between:

- retaining a backward-compatible reader;
- providing a transactional migration from a supported baseline; or
- rejecting the old layout with a documented export or rebuild path.

No stable compatibility policy is inferred from prerelease schema generations.
