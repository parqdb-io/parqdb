from __future__ import annotations

import parqdb


def test_public_errors_share_the_parqdb_base_class() -> None:
    for error_type in [
        parqdb.InvalidArgumentError,
        parqdb.InvalidSchemaError,
        parqdb.IndexNotFoundError,
        parqdb.AlreadyExistsError,
        parqdb.AmbiguousIndexError,
        parqdb.InvalidMetadataError,
        parqdb.CatalogError,
        parqdb.StorageError,
        parqdb.BackendError,
        parqdb.BuildAlreadyRunningError,
    ]:
        assert issubclass(error_type, parqdb.ParqDBError)
