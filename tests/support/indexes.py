from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_IVF_CENTROIDS_FINGERPRINT_NAMESPACE = uuid.UUID("2fb71e63-a27c-4fc5-9d6d-5070698dc398")


def write_ivf_centroids_metadata(
    metadata_root: Path,
    *,
    source: Mapping[str, Any],
    centroids: Mapping[str, Any],
    vector_field: str = "embedding",
    dimension: int = 2,
    metric: str = "l2_squared",
    nlist: int = 2,
) -> dict[str, str]:
    """Write one valid managed IVF centroid document for backend tests."""
    if source["profile"] == "parquet":
        fingerprint_source: dict[str, object] = {
            "profile": "parquet",
            "uri": source["uri"],
        }
    else:
        fingerprint_source = {
            "profile": "iceberg",
            "table-uuid": source["table-uuid"],
            "snapshot-id": source["snapshot-id"],
        }
    descriptor = {
        "source": dict(source),
        "vector-field": vector_field,
        "dimension": dimension,
        "metric": metric,
        "nlist": nlist,
        "clustering-profile-version": 1,
    }
    fingerprint_descriptor = {
        "source": fingerprint_source,
        "vector-field": vector_field,
        "dimension": dimension,
        "metric": metric,
        "nlist": nlist,
        "clustering-profile-version": 1,
    }
    fingerprint = str(
        uuid.uuid5(
            _IVF_CENTROIDS_FINGERPRINT_NAMESPACE,
            json.dumps(
                fingerprint_descriptor,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    )
    artifact_uuid = uuid.uuid4()
    location = metadata_root / "metadata" / str(artifact_uuid)
    metadata_location = location / "v1.metadata.json"
    metadata_location.parent.mkdir(parents=True, exist_ok=True)
    metadata_location.write_text(
        json.dumps(
            {
                "format-version": 1,
                "artifact-uuid": str(artifact_uuid),
                "fingerprint": fingerprint,
                "location": location.as_uri(),
                "created-at-ms": 1,
                "descriptor": descriptor,
                "centroids": dict(centroids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ivf_centroids_fingerprint": fingerprint,
        "ivf_centroids_uuid": str(artifact_uuid),
        "ivf_centroids_metadata_location": metadata_location.as_uri(),
    }
