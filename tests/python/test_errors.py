from __future__ import annotations

import relify


def test_public_errors_share_the_relify_base_class() -> None:
    for error_type in [
        relify.InvalidArgumentError,
        relify.InvalidSchemaError,
        relify.IndexNotFoundError,
        relify.AlreadyExistsError,
        relify.AmbiguousIndexError,
        relify.InvalidMetadataError,
        relify.CatalogError,
        relify.StorageError,
        relify.BackendError,
        relify.BuildAlreadyRunningError,
    ]:
        assert issubclass(error_type, relify.RelifyError)
