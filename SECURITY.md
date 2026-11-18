# Security Policy

## Supported Versions

ParqDB 0.3 is the current stable release line. The latest published 0.3.x
release is 0.3.0. Security fixes are applied to the latest 0.3.x release and,
when appropriate, to `main`.

Older pre-release lines, including 0.2.x release candidates, are not supported.
A longer support window for previous stable lines has not been announced.

## Reporting a Vulnerability

Please use
[GitHub private vulnerability reporting](https://github.com/parqdb-io/parqdb/security/advisories/new)
for suspected vulnerabilities. Include affected versions, impact, reproduction
steps, and any known mitigations.

Do not disclose sensitive details in a public issue. If GitHub private
reporting is unavailable, email `petrizhang@outlook.com`.

The maintainers aim to acknowledge a report within five business days. A fix
timeline depends on severity and reproducibility. Reporters will be kept
informed before coordinated disclosure.

## Security-Relevant Scope

Reports concerning metadata validation, path or URI confinement, catalog
publication, native memory safety, SQL injection, object-store credentials, or
dependency vulnerabilities are in scope. General feature requests and
performance issues should use the public issue tracker.
