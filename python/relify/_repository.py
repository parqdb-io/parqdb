from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ._native import _NativeIndexRepository


def open_index_repository(
    index_catalog: str,
    *,
    metadata_root: str | None = None,
    storage_options: Mapping[str, str] | None = None,
) -> _NativeIndexRepository:
    """Open the query-engine-independent index repository used by a session."""
    catalog_path = sqlite_catalog_path(index_catalog)
    resolved_metadata_root = metadata_root or default_metadata_root(catalog_path)
    options = dict(storage_options or {})
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in options.items()
    ):
        raise TypeError("storage_options keys and values must be strings")
    return _NativeIndexRepository(
        catalog_path,
        resolved_metadata_root,
        options or None,
    )


def sqlite_catalog_path(catalog: str) -> Path:
    if not isinstance(catalog, str):
        raise TypeError("index_catalog must be a catalog URI")
    parsed = urlsplit(catalog)
    if (
        parsed.scheme != "sqlite"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise NotImplementedError(
            "the current remote-engine sessions support only sqlite:/// index catalogs"
        )
    path = Path(unquote(parsed.path)).expanduser()
    if not path.is_absolute():
        raise ValueError("SQLite catalog URI must contain an absolute path")
    return path


def default_metadata_root(catalog_path: Path) -> str:
    root = catalog_path.parent / f"{catalog_path.stem}-metadata"
    return root.resolve().as_uri()
