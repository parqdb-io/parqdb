//! Metadata for immutable IVF centroid artifacts shared by logical indexes.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::invalid;
use crate::serde_helpers::lowercase_uuid;
use crate::{Result, validate_relative_location};

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
pub struct IvfCentroidsDescriptor {
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
#[serde(rename_all = "kebab-case")]
struct FingerprintDescriptor<'a> {
    vector_field: &'a str,
    dimension: i32,
    metric: DistanceMetric,
    nlist: i32,
    clustering_profile_version: i32,
}

impl IvfCentroidsDescriptor {
    /// Validates the descriptor independently.
    pub fn validate(&self) -> Result<()> {
        if self.vector_field.is_empty() {
            return invalid("IVF centroids vector-field must be non-empty");
        }
        if self.dimension <= 0 || self.nlist <= 0 || self.clustering_profile_version <= 0 {
            return invalid("IVF centroids dimension, nlist, and profile version must be positive");
        }
        Ok(())
    }

    /// Returns the deterministic lookup fingerprint for this descriptor.
    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        let canonical = serde_json::to_vec(&FingerprintDescriptor {
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
        self.vector_field == other.vector_field
            && self.dimension == other.dimension
            && self.metric == other.metric
            && self.nlist == other.nlist
            && self.clustering_profile_version == other.clustering_profile_version
    }
}

/// Immutable metadata document for one reusable IVF centroid artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct IvfCentroidsMetadata {
    /// IVF-centroid metadata format version.
    pub format_version: i32,
    /// Stable artifact UUID.
    #[serde(with = "lowercase_uuid")]
    pub artifact_uuid: Uuid,
    /// Deterministic descriptor fingerprint.
    pub fingerprint: String,
    /// Artifact creation time as Unix epoch milliseconds.
    pub created_at_ms: i64,
    /// Semantic identity of the centroid model.
    pub descriptor: IvfCentroidsDescriptor,
    /// Artifact-root-relative path containing the centroid rows.
    pub centroids: String,
    /// Artifact-root-relative path containing hierarchical root centroids and
    /// their contiguous leaf-CID ranges.
    pub roots: String,
}

impl IvfCentroidsMetadata {
    /// Parses and validates one IVF-centroids metadata document.
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
                "unsupported IVF centroids format-version {}",
                self.format_version
            ));
        }
        if self.artifact_uuid.is_nil() {
            return invalid("IVF centroid artifact UUID must not be nil");
        }
        if self.created_at_ms < 0 {
            return invalid("IVF centroids creation time must be non-negative");
        }
        self.descriptor.validate()?;
        if self.fingerprint != self.descriptor.fingerprint()? {
            return invalid("IVF centroid fingerprint does not match its descriptor");
        }
        validate_relative_location(&self.centroids)?;
        validate_relative_location(&self.roots)?;
        if self.centroids == self.roots {
            return invalid("IVF centroid and root locations must be different");
        }
        Ok(())
    }
}

/// Reference to an IVF centroid artifact stored in logical-index parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfCentroidsReference {
    /// Deterministic descriptor fingerprint.
    pub fingerprint: String,
    /// Stable artifact UUID.
    pub artifact_uuid: Uuid,
    /// Artifact-root-relative IVF-centroids metadata location.
    pub metadata_location: String,
}

impl IvfCentroidsReference {
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
            return invalid("IVF centroid artifact UUID must not be nil");
        }
        let parsed = Uuid::parse_str(&self.fingerprint).map_err(|_| {
            crate::Error("IVF centroid fingerprint must be a lowercase UUID".into())
        })?;
        if parsed.to_string() != self.fingerprint {
            return invalid("IVF centroid fingerprint must be a lowercase UUID");
        }
        validate_relative_location(&self.metadata_location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parquet_descriptor() -> IvfCentroidsDescriptor {
        IvfCentroidsDescriptor {
            vector_field: "embedding".into(),
            dimension: 2,
            metric: DistanceMetric::L2Squared,
            nlist: 2,
            clustering_profile_version: 1,
        }
    }

    #[test]
    fn source_free_descriptor_fingerprint_matches_the_spec_fixture() {
        let descriptor = parquet_descriptor();

        assert_eq!(
            descriptor.fingerprint().unwrap(),
            "3ad6988a-389e-53de-aaa6-f210345fd894"
        );
    }

    #[test]
    fn metadata_and_references_reject_mismatched_identity() {
        let descriptor = parquet_descriptor();
        let artifact_uuid = Uuid::new_v4();
        let fingerprint = descriptor.fingerprint().unwrap();
        let mut metadata = IvfCentroidsMetadata {
            format_version: 1,
            artifact_uuid,
            fingerprint: fingerprint.clone(),
            created_at_ms: 1,
            descriptor,
            centroids: "centroids/".into(),
            roots: "roots/".into(),
        };

        metadata.validate().unwrap();
        metadata.fingerprint = Uuid::new_v4().to_string();
        assert!(metadata.validate().is_err());
        assert!(
            IvfCentroidsReference::new(fingerprint.to_uppercase(), artifact_uuid, "metadata.json")
                .is_err()
        );
        assert!(
            IvfCentroidsReference::new(
                fingerprint,
                artifact_uuid,
                "s3://parqdb-fixtures/metadata.json"
            )
            .is_err()
        );
    }

    #[test]
    fn fingerprint_changes_with_the_centroid_descriptor() {
        let first = IvfCentroidsDescriptor {
            vector_field: "embedding".into(),
            dimension: 2,
            metric: DistanceMetric::Cosine,
            nlist: 2,
            clustering_profile_version: 1,
        };
        let mut changed = first.clone();
        changed.vector_field = "other_embedding".into();

        assert!(!first.is_compatible_with(&changed));
        assert_ne!(first.fingerprint().unwrap(), changed.fingerprint().unwrap());
    }
}
