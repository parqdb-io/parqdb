//! Index-family schema validation and registration.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::invalid;
use crate::metadata::{parse_boolean_parameter, parse_positive_parameter};
use crate::{IndexSnapshot, Result};

/// Vector representation stored in IVF postings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostingEncoding {
    /// Store only source keys and read vectors from the source relation.
    Source,
    /// Store exact `f32` vectors in the postings relation.
    Flat,
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
            Self::Flat => "flat",
            Self::Lvq4 => "lvq4",
            Self::Lvq8 => "lvq8",
        }
    }

    /// Parses a canonical metadata value.
    #[must_use]
    pub fn from_metadata(value: &str) -> Option<Self> {
        match value {
            "source" => Some(Self::Source),
            "flat" => Some(Self::Flat),
            "lvq4" => Some(Self::Lvq4),
            "lvq8" => Some(Self::Lvq8),
            _ => None,
        }
    }

    /// Returns the legacy IVF v1 vector-storage flag when representable.
    #[must_use]
    pub const fn v1_store_vectors(self) -> Option<bool> {
        match self {
            Self::Source => Some(false),
            Self::Flat => Some(true),
            Self::Lvq4 | Self::Lvq8 => None,
        }
    }
}

/// Portable schema contract implemented by one index family.
pub trait IndexFamily: Send + Sync {
    /// Canonical family identifier stored in index metadata.
    fn name(&self) -> &str;

    /// Validates the family-defined fields of one structurally valid snapshot.
    fn validate(&self, snapshot: &IndexSnapshot) -> Result<()>;
}

/// Index-family contracts available to a metadata reader.
#[derive(Clone)]
pub struct IndexFamilyRegistry {
    families: BTreeMap<String, Arc<dyn IndexFamily>>,
}

impl IndexFamilyRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: BTreeMap::new(),
        }
    }

    /// Creates the registry used by Relify's built-in metadata APIs.
    #[must_use]
    pub fn builtins() -> Self {
        let mut registry = Self::new();
        registry.families.insert("ivf".into(), Arc::new(IvfFamily));
        registry
    }

    /// Registers one family contract.
    pub fn register(&mut self, family: Arc<dyn IndexFamily>) -> Result<()> {
        let name = family.name();
        if !valid_family_name(name) {
            return invalid("index-family names must use lowercase ASCII letters, digits, or '_'");
        }
        if self.families.contains_key(name) {
            return invalid(format!("index family is already registered: {name}"));
        }
        self.families.insert(name.to_owned(), family);
        Ok(())
    }

    pub(crate) fn validate(&self, snapshot: &IndexSnapshot) -> Result<()> {
        let family = self.families.get(&snapshot.index_family).ok_or_else(|| {
            crate::Error::new(format!(
                "unsupported index family: {}",
                snapshot.index_family
            ))
        })?;
        family.validate(snapshot)
    }
}

impl Default for IndexFamilyRegistry {
    fn default() -> Self {
        Self::builtins()
    }
}

fn valid_family_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

struct IvfFamily;

impl IndexFamily for IvfFamily {
    fn name(&self) -> &'static str {
        "ivf"
    }

    fn validate(&self, snapshot: &IndexSnapshot) -> Result<()> {
        if !matches!(snapshot.index_schema_version, 1 | 2) || snapshot.metric != "l2_squared" {
            return invalid("only IVF schema versions 1 and 2 with l2_squared are supported");
        }

        let encoding_parameter = if snapshot.index_schema_version == 1 {
            "store_vectors"
        } else {
            "posting_encoding"
        };
        let expected_parameters = ["dimension", "nlist", "ntotal", encoding_parameter];
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
        if snapshot.index_schema_version == 1 {
            parse_boolean_parameter(&snapshot.parameters, "store_vectors")?;
        } else if snapshot
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
}
