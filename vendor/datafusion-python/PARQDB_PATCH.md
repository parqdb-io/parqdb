# ParqDB patch

The vendored crate exposes its private `_internal` module initializer as `init_internal_module` and redirects runtime imports from `datafusion.*` to `parqdb.datafusion.*`. ParqDB uses the initializer to install the complete binding surface inside `parqdb._native`, ensuring that Python and Rust share one `SessionContext` implementation.

## Upgrade

1. Update the DataFusion version pins in the workspace manifest.
2. Run `python tools/sync_datafusion.py <version>`.
3. Run `python tools/sync_datafusion.py <version> --check`.
4. Review the generated diff and run the full project checks.

The synchronizer verifies the upstream checksums and fails if its Rust embedding patches no longer apply exactly. Generated files must not be edited by hand. This vendor boundary can be removed when upstream DataFusion exposes a stable `SessionContext` FFI that supports the same one-context model.
