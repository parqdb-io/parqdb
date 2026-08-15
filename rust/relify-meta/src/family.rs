//! Built-in index-family schema validation.

use crate::error::invalid;
use uuid::Uuid;

use crate::metadata::parse_positive_parameter;
use crate::{DistanceMetric, IndexSnapshot, Result, SharedIvfReference};

/// Current IVF index schema version.
pub const IVF_SCHEMA_VERSION: i32 = 1;

/// Vector representation stored in IVF postings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostingEncoding {
    /// Store only source keys and read vectors from the source relation.
    Source,
    /// Store vectors with four-bit locally adaptive scalar quantization.
    Lvq4,
    /// Store vectors with eight-bit locally adaptive scalar quantization.
    Lvq8,
}

impl PostingEncoding {
    /// Returns the canonical metadata value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Lvq4 => "lvq4",
            Self::Lvq8 => "lvq8",
        }
    }

    /// Parses a canonical metadata value.
    #[must_use]
    pub fn from_metadata(value: &str) -> Option<Self> {
        match value {
            "source" => Some(Self::Source),
            "lvq4" => Some(Self::Lvq4),
            "lvq8" => Some(Self::Lvq8),
            _ => None,
        }
    }

    /// Resolves the postings encoding declared by an IVF snapshot.
    pub fn from_snapshot(snapshot: &IndexSnapshot) -> Result<Self> {
        if snapshot.index_family != "ivf" {
            return invalid("posting encoding is defined only for IVF indexes");
        }
        if snapshot.index_schema_version != IVF_SCHEMA_VERSION {
            return invalid("unsupported IVF index schema version; rebuild the index");
        }
        snapshot
            .parameters
            .get("posting_encoding")
            .and_then(|value| Self::from_metadata(value))
            .ok_or_else(|| crate::Error::new("invalid IVF posting_encoding parameter"))
    }
}

/// Resolves the shared-IVF reference declared by an IVF snapshot.
pub fn shared_ivf_reference(snapshot: &IndexSnapshot) -> Result<SharedIvfReference> {
    if snapshot.index_family != "ivf" || snapshot.index_schema_version != IVF_SCHEMA_VERSION {
        return invalid("shared IVF reference requires IVF schema version 1");
    }
    let fingerprint = snapshot
        .parameters
        .get("shared_ivf_fingerprint")
        .ok_or_else(|| crate::Error::new("missing parameter: shared_ivf_fingerprint"))?;
    let artifact_uuid = snapshot
        .parameters
        .get("shared_ivf_uuid")
        .ok_or_else(|| crate::Error::new("missing parameter: shared_ivf_uuid"))?;
    let artifact_uuid = Uuid::parse_str(artifact_uuid)
        .map_err(|_| crate::Error::new("invalid parameter: shared_ivf_uuid"))?;
    let metadata_location = snapshot
        .parameters
        .get("shared_ivf_metadata_location")
        .ok_or_else(|| crate::Error::new("missing parameter: shared_ivf_metadata_location"))?;
    SharedIvfReference::new(fingerprint, artifact_uuid, metadata_location)
}

pub(crate) fn validate_family(snapshot: &IndexSnapshot) -> Result<()> {
    match snapshot.index_family.as_str() {
        "ivf" => validate_ivf(snapshot),
        family => invalid(format!("unsupported index family: {family}")),
    }
}

fn validate_ivf(snapshot: &IndexSnapshot) -> Result<()> {
    if snapshot.index_schema_version != IVF_SCHEMA_VERSION {
        return invalid("unsupported IVF index schema version; rebuild the index");
    }
    if DistanceMetric::from_metadata(&snapshot.metric).is_none() {
        return invalid("IVF metric must be l2_squared or cosine");
    }

    let expected_parameters = [
        "dimension",
        "nlist",
        "ntotal",
        "posting_encoding",
        "shared_ivf_fingerprint",
        "shared_ivf_uuid",
        "shared_ivf_metadata_location",
    ];
    if snapshot.parameters.len() != expected_parameters.len()
        || snapshot
            .parameters
            .keys()
            .any(|key| !expected_parameters.contains(&key.as_str()))
    {
        return invalid("IVF parameters do not match the declared index schema version");
    }
    parse_positive_parameter(&snapshot.parameters, "dimension", i32::MAX as u64)?;
    let nlist = parse_positive_parameter(&snapshot.parameters, "nlist", i32::MAX as u64)?;
    let ntotal = parse_positive_parameter(&snapshot.parameters, "ntotal", i64::MAX as u64)?;
    if snapshot
        .parameters
        .get("posting_encoding")
        .and_then(|value| PostingEncoding::from_metadata(value))
        .is_none()
    {
        return invalid("invalid IVF posting_encoding parameter");
    }
    if nlist > ntotal {
        return invalid("nlist must not exceed ntotal");
    }
    shared_ivf_reference(snapshot)?;

    let expected_relations = ["ivf_centroids", "ivf_postings"];
    if !snapshot.index_relations.contains_key(expected_relations[0])
        || !snapshot.index_relations.contains_key(expected_relations[1])
        || snapshot
            .index_relations
            .keys()
            .any(|role| !expected_relations.contains(&role.as_str()))
    {
        return invalid("invalid IVF index table roles");
    }
    Ok(())
}
