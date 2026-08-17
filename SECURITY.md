# Security Policy

## Supported Versions

ParqDB has not published a stable release. Security fixes are applied to the
latest code on `main`. After releases begin, only the newest release line will
receive fixes until a longer support policy is announced.

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
