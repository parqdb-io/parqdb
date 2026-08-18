//! Self-contained immutable IVF package metadata for HTTP Range readers.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::invalid;
use crate::serde_helpers::lowercase_uuid;
use crate::{DistanceMetric, PostingEncoding, Result, validate_relative_location};

/// Largest integer represented exactly by every conforming JavaScript number.
pub const JSON_SAFE_INTEGER_MAX: u64 = (1_u64 << 53) - 1;

/// One source-key field returned by a static package query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticSourceKeyField {
    /// Original source field name.
    pub name: String,
    /// Canonical IVF key type such as `long`, `string`, or `fixed(16)`.
    #[serde(rename = "type")]
    pub data_type: String,
}

/// Logical query contract declared by a static package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexDescriptor {
    /// Distance metric used for routing and result distances.
    pub metric: DistanceMetric,
    /// Posting representation; static package version 1 permits LVQ only.
    pub posting_encoding: PostingEncoding,
    /// Vector dimension.
    pub dimension: i32,
    /// Number of leaf centroids.
    pub nlist: i32,
    /// Number of indexed rows.
    pub ntotal: i64,
    /// Ordered source-key result fields.
    pub source_key_fields: Vec<StaticSourceKeyField>,
}

/// One immutable object referenced by a static package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexObject {
    /// Path relative to the package manifest.
    pub path: String,
    /// Complete object size in bytes.
    pub size: u64,
    /// Lowercase SHA-256 digest of the complete object.
    pub sha256: String,
}

/// Persisted root-to-leaf routing topology and centroid objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexHierarchy {
    /// Number of hierarchical roots.
    pub root_count: i32,
    /// Root-to-leaf prefix sum.
    pub cid_offsets: Vec<i32>,
    /// Leaf-centroid representation used for global routing.
    pub centroid_encoding: PostingEncoding,
    /// Root centroid Parquet object.
    pub roots: StaticIndexObject,
    /// Leaf centroid Parquet object.
    pub centroids: StaticIndexObject,
}

/// One postings Parquet object in a static package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticPostingsFile {
    /// Path relative to the package manifest.
    pub path: String,
    /// Hierarchical root ID owning this file.
    pub cid_bucket: i32,
    /// Smallest CID represented by the file.
    pub min_cid: i32,
    /// Largest CID represented by the file.
    pub max_cid: i32,
    /// Number of postings rows in the file.
    pub rows: u64,
    /// Complete object size in bytes.
    pub size: u64,
    /// Lowercase SHA-256 digest of the complete object.
    pub sha256: String,
}

/// Complete postings inventory for a static package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexPostings {
    /// Canonically ordered postings objects.
    pub files: Vec<StaticPostingsFile>,
}

/// Version 1 immutable, self-contained IVF package manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexPackageManifest {
    /// Manifest format version. Version 1 is the only supported value.
    pub format_version: i32,
    /// Unique identity of this immutable package.
    #[serde(with = "lowercase_uuid")]
    pub package_uuid: Uuid,
    /// Logical query contract.
    pub index: StaticIndexDescriptor,
    /// Hierarchical centroid objects and CID topology.
    pub hierarchy: StaticIndexHierarchy,
    /// Complete postings object inventory.
    pub postings: StaticIndexPostings,
}

impl StaticIndexPackageManifest {
    /// Parses and validates one strict package manifest.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|error| crate::Error(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serializes a validated manifest with deterministic field ordering.
    pub fn to_json_vec(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec_pretty(self).map_err(|error| crate::Error(error.to_string()))
    }

    /// Validates the complete package contract without performing object I/O.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return invalid(format!(
                "unsupported static package format-version {}",
                self.format_version
            ));
        }
        if self.package_uuid.is_nil() {
            return invalid("static package UUID must not be nil");
        }
        self.validate_index()?;
        self.validate_hierarchy()?;
        self.validate_objects()
    }

    fn validate_index(&self) -> Result<()> {
        let index = &self.index;
        if !matches!(
            index.posting_encoding,
            PostingEncoding::Lvq4 | PostingEncoding::Lvq8
        ) {
            return invalid("static package version 1 requires lvq4 or lvq8 postings");
        }
        if index.dimension <= 0
            || index.nlist <= 0
            || index.ntotal <= 0
            || i64::from(index.nlist) > index.ntotal
            || u64::try_from(index.ntotal)
                .ok()
                .is_none_or(|value| value > JSON_SAFE_INTEGER_MAX)
        {
            return invalid("invalid or non-portable static package index cardinality");
        }
        if index.source_key_fields.is_empty() {
            return invalid("static package source-key-fields must not be empty");
        }
        let mut names = HashSet::with_capacity(index.source_key_fields.len());
        for field in &index.source_key_fields {
            if field.name.is_empty()
                || field.name == "_distance"
                || !names.insert(field.name.as_str())
            {
                return invalid(
                    "static package source-key field names must be non-empty and unique",
                );
            }
            validate_source_key_type(&field.data_type)?;
        }
        Ok(())
    }

    fn validate_hierarchy(&self) -> Result<()> {
        let hierarchy = &self.hierarchy;
        if hierarchy.centroid_encoding != PostingEncoding::Lvq8 {
            return invalid("static package version 1 requires lvq8 leaf centroids");
        }
        if hierarchy.root_count <= 0
            || hierarchy.cid_offsets.len()
                != usize::try_from(hierarchy.root_count)
                    .unwrap_or_default()
                    .saturating_add(1)
            || hierarchy.cid_offsets.first() != Some(&0)
            || hierarchy.cid_offsets.last() != Some(&self.index.nlist)
            || hierarchy
                .cid_offsets
                .windows(2)
                .any(|range| range[0] >= range[1])
        {
            return invalid("static package cid-offsets must partition [0, nlist)");
        }
        validate_object(&hierarchy.roots)?;
        validate_object(&hierarchy.centroids)?;
        if hierarchy.roots.path == hierarchy.centroids.path {
            return invalid("static package root and leaf centroid paths must differ");
        }
        Ok(())
    }

    fn validate_objects(&self) -> Result<()> {
        if self.postings.files.is_empty() {
            return invalid("static package must contain postings files");
        }
        let mut paths = HashSet::with_capacity(self.postings.files.len() + 2);
        paths.insert(self.hierarchy.roots.path.as_str());
        paths.insert(self.hierarchy.centroids.path.as_str());
        let mut rows = 0_u64;
        let mut previous: Option<&StaticPostingsFile> = None;
        for file in &self.postings.files {
            validate_relative_location(&file.path)?;
            validate_sha256(&file.sha256)?;
            if !file.path.ends_with(".parquet")
                || !paths.insert(file.path.as_str())
                || file.rows == 0
                || file.size == 0
                || file.rows > JSON_SAFE_INTEGER_MAX
                || file.size > JSON_SAFE_INTEGER_MAX
            {
                return invalid("invalid static package postings object");
            }
            let bucket = usize::try_from(file.cid_bucket)
                .ok()
                .filter(|bucket| *bucket + 1 < self.hierarchy.cid_offsets.len())
                .ok_or_else(|| crate::Error::new("static package cid-bucket is out of range"))?;
            let prefix = format!("ivf_postings/cid_bucket={:06}/", file.cid_bucket);
            if !file.path.starts_with(&prefix)
                || file.min_cid < self.hierarchy.cid_offsets[bucket]
                || file.min_cid > file.max_cid
                || file.max_cid >= self.hierarchy.cid_offsets[bucket + 1]
            {
                return invalid("static package postings range is outside its bucket");
            }
            rows = rows
                .checked_add(file.rows)
                .ok_or_else(|| crate::Error::new("static package row count overflows"))?;

            if let Some(previous) = previous {
                let previous_key = (
                    previous.cid_bucket,
                    previous.min_cid,
                    previous.max_cid,
                    previous.path.as_str(),
                );
                let current_key = (
                    file.cid_bucket,
                    file.min_cid,
                    file.max_cid,
                    file.path.as_str(),
                );
                if previous_key >= current_key
                    || (previous.cid_bucket == file.cid_bucket && file.min_cid < previous.max_cid)
                {
                    return invalid("static package postings files are not canonically ordered");
                }
            }
            previous = Some(file);
        }
        if rows != u64::try_from(self.index.ntotal).expect("ntotal was validated as positive") {
            return invalid("static package postings rows do not sum to ntotal");
        }
        Ok(())
    }
}

fn validate_object(object: &StaticIndexObject) -> Result<()> {
    validate_relative_location(&object.path)?;
    validate_sha256(&object.sha256)?;
    if !object.path.ends_with(".parquet") || object.size == 0 || object.size > JSON_SAFE_INTEGER_MAX
    {
        return invalid("invalid static package Parquet object");
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return invalid("static package sha256 must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn validate_source_key_type(value: &str) -> Result<()> {
    if matches!(
        value,
        "boolean" | "int" | "long" | "binary" | "string" | "date"
    ) {
        return Ok(());
    }
    let Some(length) = value
        .strip_prefix("fixed(")
        .and_then(|value| value.strip_suffix(')'))
    else {
        return invalid("unsupported static package source-key type");
    };
    let parsed = length
        .parse::<u32>()
        .ok()
        .filter(|parsed| *parsed > 0 && parsed.to_string() == length);
    if parsed.is_none() {
        return invalid("invalid static package fixed source-key type");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn valid_manifest() -> StaticIndexPackageManifest {
        StaticIndexPackageManifest {
            format_version: 1,
            package_uuid: Uuid::parse_str("249343c7-9989-48d8-b2ca-d0caa62ba940").unwrap(),
            index: StaticIndexDescriptor {
                metric: DistanceMetric::L2Squared,
                posting_encoding: PostingEncoding::Lvq8,
                dimension: 2,
                nlist: 4,
                ntotal: 10,
                source_key_fields: vec![StaticSourceKeyField {
                    name: "document_id".into(),
                    data_type: "long".into(),
                }],
            },
            hierarchy: StaticIndexHierarchy {
                root_count: 2,
                cid_offsets: vec![0, 2, 4],
                centroid_encoding: PostingEncoding::Lvq8,
                roots: StaticIndexObject {
                    path: "roots.parquet".into(),
                    size: 100,
                    sha256: digest('a'),
                },
                centroids: StaticIndexObject {
                    path: "centroids.parquet".into(),
                    size: 200,
                    sha256: digest('b'),
                },
            },
            postings: StaticIndexPostings {
                files: vec![
                    StaticPostingsFile {
                        path: "ivf_postings/cid_bucket=000000/part-00000.parquet".into(),
                        cid_bucket: 0,
                        min_cid: 0,
                        max_cid: 1,
                        rows: 6,
                        size: 300,
                        sha256: digest('c'),
                    },
                    StaticPostingsFile {
                        path: "ivf_postings/cid_bucket=000001/part-00000.parquet".into(),
                        cid_bucket: 1,
                        min_cid: 2,
                        max_cid: 3,
                        rows: 4,
                        size: 250,
                        sha256: digest('d'),
                    },
                ],
            },
        }
    }

    #[test]
    fn round_trips_a_valid_manifest() {
        let manifest = valid_manifest();
        let json = manifest.to_json_vec().unwrap();
        assert_eq!(
            StaticIndexPackageManifest::from_json_slice(&json).unwrap(),
            manifest
        );
    }

    #[test]
    fn rejects_source_postings_and_noncanonical_key_types() {
        let mut manifest = valid_manifest();
        manifest.index.posting_encoding = PostingEncoding::Source;
        assert!(manifest.validate().is_err());

        let mut manifest = valid_manifest();
        manifest.index.source_key_fields[0].data_type = "fixed(016)".into();
        assert!(manifest.validate().is_err());

        let mut manifest = valid_manifest();
        manifest.index.source_key_fields[0].name = "_distance".into();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn rejects_unsafe_json_integers_and_cross_bucket_ranges() {
        let mut manifest = valid_manifest();
        manifest.postings.files[0].size = JSON_SAFE_INTEGER_MAX + 1;
        assert!(manifest.validate().is_err());

        let mut manifest = valid_manifest();
        manifest.postings.files[0].max_cid = 2;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn rejects_unknown_fields() {
        let mut json = serde_json::to_value(valid_manifest()).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("extra".into(), serde_json::Value::Bool(true));
        assert!(
            StaticIndexPackageManifest::from_json_slice(&serde_json::to_vec(&json).unwrap())
                .is_err()
        );
    }
}
