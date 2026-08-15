//! Metadata for immutable IVF centroid artifacts shared by logical indexes.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::invalid;
use crate::metadata::validate_metadata_location;
use crate::serde_helpers::lowercase_uuid;
use crate::{RelationReference, Result};

/// Version of the deterministic centroid-training contract.
pub const IVF_CLUSTERING_PROFILE_VERSION: i32 = 1;

const FINGERPRINT_NAMESPACE: Uuid = Uuid::from_bytes([
    0x2f, 0xb7, 0x1e, 0x63, 0xa2, 0x7c, 0x4f, 0xc5, 0x9d, 0x6d, 0x50, 0x70, 0x69, 0x8d, 0xc3, 0x98,
]);

/// Distance metrics supported by the IVF family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Squared Euclidean distance.
    L2Squared,
    /// Cosine distance, represented as half squared L2 over normalized vectors.
    Cosine,
}

impl DistanceMetric {
    /// Returns the canonical metadata value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2Squared => "l2_squared",
            Self::Cosine => "cosine",
        }
    }

    /// Parses a canonical metadata value.
    #[must_use]
    pub fn from_metadata(value: &str) -> Option<Self> {
        match value {
            "l2_squared" => Some(Self::L2Squared),
            "cosine" => Some(Self::Cosine),
            _ => None,
        }
    }
}

/// Semantic identity of one reusable IVF centroid model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SharedIvfDescriptor {
    /// Exact source-table state used for training.
    pub source: RelationReference,
    /// Source column containing vectors.
    pub vector_field: String,
    /// Canonical vector dimension.
    pub dimension: i32,
    /// Distance metric used for training and routing.
    pub metric: DistanceMetric,
    /// Number of coarse clusters.
    pub nlist: i32,
    /// Version of the deterministic clustering contract.
    pub clustering_profile_version: i32,
}

#[derive(Serialize)]
#[serde(tag = "profile", rename_all = "lowercase")]
enum FingerprintSource<'a> {
    Parquet {
        uri: &'a str,
    },
    Iceberg {
        #[serde(rename = "table-uuid")]
        table_uuid: Uuid,
        #[serde(rename = "snapshot-id")]
        snapshot_id: i64,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct FingerprintDescriptor<'a> {
    source: FingerprintSource<'a>,
    vector_field: &'a str,
    dimension: i32,
    metric: DistanceMetric,
    nlist: i32,
    clustering_profile_version: i32,
}

impl SharedIvfDescriptor {
    /// Validates the descriptor independently.
    pub fn validate(&self) -> Result<()> {
        self.source.validate()?;
        if self.vector_field.is_empty() {
            return invalid("shared IVF vector-field must be non-empty");
        }
        if self.dimension <= 0 || self.nlist <= 0 || self.clustering_profile_version <= 0 {
            return invalid("shared IVF dimension, nlist, and profile version must be positive");
        }
        Ok(())
    }

    /// Returns the deterministic lookup fingerprint for this descriptor.
    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        let source = match &self.source {
            RelationReference::Parquet { uri } => FingerprintSource::Parquet { uri },
            RelationReference::Iceberg {
                table_uuid,
                snapshot_id,
                ..
            } => FingerprintSource::Iceberg {
                table_uuid: *table_uuid,
                snapshot_id: *snapshot_id,
            },
        };
        let canonical = serde_json::to_vec(&FingerprintDescriptor {
            source,
            vector_field: &self.vector_field,
            dimension: self.dimension,
            metric: self.metric,
            nlist: self.nlist,
            clustering_profile_version: self.clustering_profile_version,
        })
        .map_err(|error| crate::Error(error.to_string()))?;
        Ok(Uuid::new_v5(&FINGERPRINT_NAMESPACE, &canonical).to_string())
    }

    /// Returns whether two descriptors name the same reusable coarse model.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.source.exact_state_key() == other.source.exact_state_key()
            && self.vector_field == other.vector_field
            && self.dimension == other.dimension
            && self.metric == other.metric
            && self.nlist == other.nlist
            && self.clustering_profile_version == other.clustering_profile_version
    }
}

/// Immutable metadata document for one shared IVF centroid artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SharedIvfMetadata {
    /// Shared-IVF metadata format version.
    pub format_version: i32,
    /// Stable artifact UUID.
    #[serde(with = "lowercase_uuid")]
    pub artifact_uuid: Uuid,
    /// Deterministic descriptor fingerprint.
    pub fingerprint: String,
    /// Base URI for this artifact's metadata files.
    pub location: String,
    /// Artifact creation time as Unix epoch milliseconds.
    pub created_at_ms: i64,
    /// Semantic identity of the centroid model.
    pub descriptor: SharedIvfDescriptor,
    /// Exact relation containing the centroid rows.
    pub centroids: RelationReference,
}

impl SharedIvfMetadata {
    /// Parses and validates one shared-IVF metadata document.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let metadata: Self =
            serde_json::from_slice(bytes).map_err(|error| crate::Error(error.to_string()))?;
        metadata.validate()?;
        Ok(metadata)
    }

    /// Validates this immutable document independently.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return invalid(format!(
                "unsupported shared IVF format-version {}",
                self.format_version
            ));
        }
        validate_metadata_location(&self.location)?;
        if self.artifact_uuid.is_nil() {
            return invalid("shared IVF artifact UUID must not be nil");
        }
        if self.created_at_ms < 0 {
            return invalid("shared IVF creation time must be non-negative");
        }
        self.descriptor.validate()?;
        if self.fingerprint != self.descriptor.fingerprint()? {
            return invalid("shared IVF fingerprint does not match its descriptor");
        }
        self.centroids.validate()?;
        Ok(())
    }
}

/// Reference to a shared IVF artifact stored in logical-index parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedIvfReference {
    /// Deterministic descriptor fingerprint.
    pub fingerprint: String,
    /// Stable artifact UUID.
    pub artifact_uuid: Uuid,
    /// Immutable shared-IVF metadata location.
    pub metadata_location: String,
}

impl SharedIvfReference {
    /// Creates and validates a reference.
    pub fn new(
        fingerprint: impl Into<String>,
        artifact_uuid: Uuid,
        metadata_location: impl Into<String>,
    ) -> Result<Self> {
        let reference = Self {
            fingerprint: fingerprint.into(),
            artifact_uuid,
            metadata_location: metadata_location.into(),
        };
        reference.validate()?;
        Ok(reference)
    }

    /// Validates canonical field encodings.
    pub fn validate(&self) -> Result<()> {
        if self.artifact_uuid.is_nil() {
            return invalid("shared IVF artifact UUID must not be nil");
        }
        let parsed = Uuid::parse_str(&self.fingerprint)
            .map_err(|_| crate::Error("shared IVF fingerprint must be a lowercase UUID".into()))?;
        if parsed.to_string() != self.fingerprint {
            return invalid("shared IVF fingerprint must be a lowercase UUID");
        }
        validate_metadata_location(&self.metadata_location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parquet_descriptor_fingerprint_matches_the_spec_fixture() {
        let descriptor = SharedIvfDescriptor {
            source: RelationReference::Parquet {
                uri: "s3://relify-fixtures/v1/valid/source/".into(),
            },
            vector_field: "embedding".into(),
            dimension: 2,
            metric: DistanceMetric::L2Squared,
            nlist: 2,
            clustering_profile_version: 1,
        };

        assert_eq!(
            descriptor.fingerprint().unwrap(),
            "4c774941-faf4-5e06-9a2a-03ffa111641f"
        );
    }

    #[test]
    fn iceberg_fingerprint_ignores_locator_changes() {
        let table_uuid = Uuid::new_v4();
        let first = SharedIvfDescriptor {
            source: RelationReference::Iceberg {
                catalog: "first".into(),
                namespace: vec!["analytics".into()],
                name: "documents".into(),
                table_uuid,
                snapshot_id: 101,
            },
            vector_field: "embedding".into(),
            dimension: 2,
            metric: DistanceMetric::Cosine,
            nlist: 2,
            clustering_profile_version: 1,
        };
        let mut renamed = first.clone();
        renamed.source = RelationReference::Iceberg {
            catalog: "second".into(),
            namespace: vec!["renamed".into()],
            name: "vectors".into(),
            table_uuid,
            snapshot_id: 101,
        };

        assert!(first.is_compatible_with(&renamed));
        assert_eq!(first.fingerprint().unwrap(), renamed.fingerprint().unwrap());
    }
}
