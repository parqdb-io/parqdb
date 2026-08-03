"""Small packaged datasets used by Relify examples."""

from __future__ import annotations

from pathlib import Path

_FILES = {
    "document_stats": "document_stats.parquet",
    "documents": "documents.parquet",
}


def uri(name: str) -> str:
    """Return the absolute file URI for a packaged example dataset."""
    if not isinstance(name, str):
        raise TypeError("dataset name must be a string")
    try:
        filename = _FILES[name]
    except KeyError as error:
        available = ", ".join(sorted(_FILES))
        raise ValueError(
            f"unknown packaged dataset: {name!r}; available datasets: {available}"
        ) from error
    path = Path(__file__).with_name(filename).resolve()
    if not path.is_file():
        raise RuntimeError(f"packaged dataset is missing: {filename}")
    return path.as_uri()


__all__ = ["uri"]
