//! Immutable file manifest for bucketed IVF Parquet postings.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::invalid;
use crate::{Result, validate_relative_location};

/// One Parquet object in a manifested IVF postings relation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct IvfPostingsFile {
    /// Path relative to the postings relation root.
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

/// Version 1 manifest for one immutable IVF postings relation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct IvfPostingsManifest {
    /// Manifest format version. Version 1 is the only supported value.
    pub format_version: i32,
    /// Number of leaf centroids.
    pub nlist: i32,
    /// Number of postings rows across every file.
    pub ntotal: i64,
    /// Root-to-leaf prefix sum. Root `r` owns
    /// `[cid_offsets[r], cid_offsets[r + 1])`.
    pub cid_offsets: Vec<i32>,
    /// Complete ordered Parquet object inventory.
    pub files: Vec<IvfPostingsFile>,
}

impl IvfPostingsManifest {
    /// Parses and validates a strict manifest document.
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

    /// Validates hierarchy, ordering, ranges, paths, and row totals.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return invalid(format!(
                "unsupported IVF postings manifest format-version {}",
                self.format_version
            ));
        }
        if self.nlist <= 0 || self.ntotal <= 0 {
            return invalid("IVF postings manifest nlist and ntotal must be positive");
        }
        if self.cid_offsets.len() < 2
            || self.cid_offsets.first() != Some(&0)
            || self.cid_offsets.last() != Some(&self.nlist)
            || self
                .cid_offsets
                .windows(2)
                .any(|range| range[0] >= range[1])
        {
            return invalid("IVF postings cid-offsets must partition [0, nlist)");
        }
        if self.files.is_empty() {
            return invalid("IVF postings manifest must contain files");
        }

        let mut paths = HashSet::with_capacity(self.files.len());
        let mut previous: Option<&IvfPostingsFile> = None;
        let mut total_rows = 0_u64;
        for file in &self.files {
            validate_relative_location(&file.path)?;
            if !file.path.ends_with(".parquet") || !paths.insert(file.path.as_str()) {
                return invalid("IVF postings file paths must be unique Parquet paths");
            }
            let bucket = usize::try_from(file.cid_bucket)
                .ok()
                .filter(|bucket| *bucket + 1 < self.cid_offsets.len())
                .ok_or_else(|| crate::Error::new("IVF postings cid-bucket is out of range"))?;
            let expected_prefix = format!("cid_bucket={:06}/", file.cid_bucket);
            if !file.path.starts_with(&expected_prefix) {
                return invalid("IVF postings path does not match cid-bucket");
            }
            if file.min_cid < self.cid_offsets[bucket]
                || file.min_cid > file.max_cid
                || file.max_cid >= self.cid_offsets[bucket + 1]
            {
                return invalid("IVF postings file CID range is outside its bucket");
            }
            if file.rows == 0 || file.size == 0 {
                return invalid("IVF postings file rows and size must be positive");
            }
            if file.sha256.len() != 64
                || file
                    .sha256
                    .bytes()
                    .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
            {
                return invalid("IVF postings sha256 must be 64 lowercase hexadecimal characters");
            }
            total_rows = total_rows
                .checked_add(file.rows)
                .ok_or_else(|| crate::Error::new("IVF postings row count overflows"))?;

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
                if previous_key >= current_key {
                    return invalid("IVF postings files are not canonically ordered");
                }
                if previous.cid_bucket == file.cid_bucket && file.min_cid < previous.max_cid {
                    return invalid("IVF postings files overlap by more than one boundary CID");
                }
            }
            previous = Some(file);
        }
        let expected_rows = u64::try_from(self.ntotal)
            .map_err(|_| crate::Error::new("IVF postings ntotal must be positive"))?;
        if total_rows != expected_rows {
            return invalid("IVF postings file rows do not sum to ntotal");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> IvfPostingsManifest {
        IvfPostingsManifest {
            format_version: 1,
            nlist: 4,
            ntotal: 10,
            cid_offsets: vec![0, 2, 4],
            files: vec![
                IvfPostingsFile {
                    path: "cid_bucket=000000/part-00000.parquet".into(),
                    cid_bucket: 0,
                    min_cid: 0,
                    max_cid: 1,
                    rows: 6,
                    size: 100,
                    sha256: "a".repeat(64),
                },
                IvfPostingsFile {
                    path: "cid_bucket=000001/part-00000.parquet".into(),
                    cid_bucket: 1,
                    min_cid: 2,
                    max_cid: 3,
                    rows: 4,
                    size: 80,
                    sha256: "b".repeat(64),
                },
            ],
        }
    }

    #[test]
    fn round_trips_a_valid_manifest() {
        let manifest = valid_manifest();
        let json = manifest.to_json_vec().unwrap();
        assert_eq!(
            IvfPostingsManifest::from_json_slice(&json).unwrap(),
            manifest
        );
    }

    #[test]
    fn permits_one_cid_to_continue_in_the_next_file() {
        let mut manifest = valid_manifest();
        manifest.files[0].max_cid = 0;
        manifest.files.insert(
            1,
            IvfPostingsFile {
                path: "cid_bucket=000000/part-00001.parquet".into(),
                cid_bucket: 0,
                min_cid: 0,
                max_cid: 1,
                rows: 2,
                size: 50,
                sha256: "c".repeat(64),
            },
        );
        manifest.files[0].rows -= 2;
        manifest.validate().unwrap();
    }

    #[test]
    fn rejects_non_contiguous_hierarchy_and_cross_bucket_files() {
        let mut manifest = valid_manifest();
        manifest.cid_offsets = vec![0, 1, 1, 4];
        assert!(manifest.validate().is_err());

        let mut manifest = valid_manifest();
        manifest.files[0].max_cid = 2;
        assert!(manifest.validate().is_err());
    }
}
