from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from ...identifier import TableIdentifier


@dataclass(frozen=True, slots=True)
class ParquetTableState:
    identifier: TableIdentifier
    uri: str

    def relation_dict(self) -> dict[str, object]:
        return {"profile": "parquet", "uri": self.uri}

    def relation_json(self) -> str:
        return json.dumps(self.relation_dict(), separators=(",", ":"))


def validate_parquet_uri(uri: str) -> str:
    if not isinstance(uri, str):
        raise TypeError("Parquet source must be an absolute URI")
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"file", "s3", "hdfs"}
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Parquet source must be an absolute file://, s3://, or hdfs:// URI"
        )
    return uri
