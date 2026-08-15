from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TableIdentifier:
    """Portable identity of a table mounted in a Relify session."""

    catalog: str
    namespace: tuple[str, ...]
    name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.catalog, str)
            or not self.catalog
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise ValueError("catalog and table name must be non-empty strings")
        if not isinstance(self.namespace, tuple) or any(
            not isinstance(segment, str) or not segment for segment in self.namespace
        ):
            raise ValueError("namespace must contain non-empty string segments")

    @property
    def index_namespace(self) -> tuple[str, ...]:
        """Catalog namespace used by indexes owned by this table."""
        return (self.catalog, *self.namespace, self.name)
