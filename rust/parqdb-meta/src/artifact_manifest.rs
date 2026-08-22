//! Self-contained immutable IVF artifact metadata for HTTP Range readers.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::invalid;
use crate::serde_helpers::lowercase_uuid;
use crate::{DistanceMetric, PostingEncoding, Result, validate_relative_location};

/// Largest integer represented exactly by every conforming JavaScript number.
pub const JSON_SAFE_INTEGER_MAX: u64 = (1_u64 << 53) - 1;

/// One source-key field returned by a index artifact query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticSourceKeyField {
    /// Original source field name.
    pub name: String,
    /// Canonical IVF key type such as `long`, `string`, or `fixed(16)`.
    #[serde(rename = "type")]
    pub data_type: String,
}

/// Logical query contract declared by a index artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexDescriptor {
    /// Source field whose vectors were indexed.
    pub vector_field: String,
    /// Distance metric used for routing and result distances.
    pub metric: DistanceMetric,
    /// Posting representation; index artifact version 1 permits LVQ only.
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

/// One immutable object referenced by a index artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexObject {
    /// Path relative to the artifact manifest.
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
    /// Root-to-leaf prefix sum.
    pub cid_offsets: Vec<i32>,
    /// Leaf-centroid representation used for global routing.
    pub centroid_encoding: PostingEncoding,
    /// Leaf centroid Parquet object.
    pub centroids: StaticIndexObject,
}

/// One immutable source Parquet object and its global row interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticSourceFile {
    /// Path relative to the publication manifest.
    pub path: String,
    /// Complete object size in bytes.
    pub size: u64,
    /// Lowercase SHA-256 digest of the complete object.
    pub sha256: String,
    /// First global source row in this object.
    pub row_begin: u64,
    /// Exclusive global source row bound in this object.
    pub row_end: u64,
}

/// Optional immutable source table used for payload lookup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticSourceDescriptor {
    /// Number of source rows.
    pub rows: u64,
    /// Uniform row-group size except for the final row group of each file.
    pub row_group_rows: u64,
    /// Dense ordered lookup key.
    pub key: StaticSourceKeyField,
    /// Ordered source column names available for projection.
    pub columns: Vec<String>,
    /// Complete canonical source object inventory.
    pub files: Vec<StaticSourceFile>,
}

/// Query/embedding parity probe for one published model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticEmbeddingParityProbe {
    /// Probe input text.
    pub text: String,
    /// Expected normalized vector.
    pub vector: Vec<f32>,
    /// Maximum accepted absolute element error.
    pub max_absolute_error: f32,
}

/// Optional immutable embedding-model contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticEmbeddingDescriptor {
    /// Pinned model repository.
    pub repository: String,
    /// Pinned immutable model revision.
    pub revision: String,
    /// Model runtime identifier. Version 1 supports `onnx`.
    pub runtime: String,
    /// Relative ONNX model path present in `assets`.
    pub onnx_file: String,
    /// Embedding dimension.
    pub dimension: i32,
    /// Maximum tokenizer sequence length.
    pub max_length: i32,
    /// Pooling contract identifier.
    pub pooling: String,
    /// Whether vectors are normalized before indexing and querying.
    pub normalize: bool,
    /// Template used to join source text fields.
    pub input_template: String,
    /// Cross-runtime parity probe.
    pub parity_probe: StaticEmbeddingParityProbe,
    /// Complete immutable model asset inventory.
    pub assets: Vec<StaticIndexObject>,
}

/// One postings Parquet object in a index artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticPostingsFile {
    /// Path relative to the artifact manifest.
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

/// Complete postings inventory for a index artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StaticIndexPostings {
    /// Canonically ordered postings objects.
    pub files: Vec<StaticPostingsFile>,
}

/// Version 1 immutable, self-contained IVF artifact manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct IndexArtifactManifest {
    /// Manifest format version. Version 1 is the only supported value.
    pub format_version: i32,
    /// Unique identity of this immutable artifact.
    #[serde(with = "lowercase_uuid")]
    pub artifact_uuid: Uuid,
    /// Logical query contract.
    pub index: StaticIndexDescriptor,
    /// Hierarchical centroid objects and CID topology.
    pub hierarchy: StaticIndexHierarchy,
    /// Complete postings object inventory.
    pub postings: StaticIndexPostings,
    /// Optional immutable source table for payload lookup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<StaticSourceDescriptor>,
    /// Optional immutable embedding-model contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<StaticEmbeddingDescriptor>,
}

impl IndexArtifactManifest {
    /// Parses and validates one strict artifact manifest.
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

    /// Validates the complete artifact contract without performing object I/O.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return invalid(format!(
                "unsupported index artifact format-version {}",
                self.format_version
            ));
        }
        if self.artifact_uuid.is_nil() {
            return invalid("index artifact UUID must not be nil");
        }
        self.validate_index()?;
        self.validate_hierarchy()?;
        self.validate_objects()?;
        self.validate_source()?;
        self.validate_embedding()?;
        self.validate_unique_paths()
    }

    fn validate_index(&self) -> Result<()> {
        let index = &self.index;
        if index.vector_field.is_empty() || index.vector_field == "_distance" {
            return invalid("index artifact vector-field must be non-empty and non-reserved");
        }
        if !matches!(
            index.posting_encoding,
            PostingEncoding::Lvq4 | PostingEncoding::Lvq8
        ) {
            return invalid("index artifact version 1 requires lvq4 or lvq8 postings");
        }
        if index.dimension <= 0
            || index.nlist <= 0
            || index.ntotal <= 0
            || i64::from(index.nlist) > index.ntotal
            || u64::try_from(index.ntotal)
                .ok()
                .is_none_or(|value| value > JSON_SAFE_INTEGER_MAX)
        {
            return invalid("invalid or non-portable index artifact index cardinality");
        }
        if index.source_key_fields.is_empty() {
            return invalid("index artifact source-key-fields must not be empty");
        }
        let mut names = HashSet::with_capacity(index.source_key_fields.len());
        for field in &index.source_key_fields {
            if field.name.is_empty()
                || field.name == "_distance"
                || !names.insert(field.name.as_str())
            {
                return invalid(
                    "index artifact source-key field names must be non-empty and unique",
                );
            }
            validate_source_key_type(&field.data_type)?;
        }
        Ok(())
    }

    fn validate_hierarchy(&self) -> Result<()> {
        let hierarchy = &self.hierarchy;
        if hierarchy.centroid_encoding != PostingEncoding::Lvq8 {
            return invalid("index artifact version 1 requires lvq8 leaf centroids");
        }
        if hierarchy.cid_offsets.len() < 2
            || hierarchy.cid_offsets.first() != Some(&0)
            || hierarchy.cid_offsets.last() != Some(&self.index.nlist)
            || hierarchy
                .cid_offsets
                .windows(2)
                .any(|range| range[0] >= range[1])
        {
            return invalid("index artifact cid-offsets must partition [0, nlist)");
        }
        validate_object(&hierarchy.centroids)?;
        Ok(())
    }

    fn validate_objects(&self) -> Result<()> {
        if self.postings.files.is_empty() {
            return invalid("index artifact must contain postings files");
        }
        let mut paths = HashSet::with_capacity(self.postings.files.len() + 1);
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
                return invalid("invalid index artifact postings object");
            }
            let bucket = usize::try_from(file.cid_bucket)
                .ok()
                .filter(|bucket| *bucket + 1 < self.hierarchy.cid_offsets.len())
                .ok_or_else(|| crate::Error::new("index artifact cid-bucket is out of range"))?;
            let prefix = format!("ivf_postings/cid_bucket={:06}/", file.cid_bucket);
            if !file.path.starts_with(&prefix)
                || file.min_cid < self.hierarchy.cid_offsets[bucket]
                || file.min_cid > file.max_cid
                || file.max_cid >= self.hierarchy.cid_offsets[bucket + 1]
            {
                return invalid("index artifact postings range is outside its bucket");
            }
            rows = rows
                .checked_add(file.rows)
                .ok_or_else(|| crate::Error::new("index artifact row count overflows"))?;

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
                    return invalid("index artifact postings files are not canonically ordered");
                }
            }
            previous = Some(file);
        }
        if rows != u64::try_from(self.index.ntotal).expect("ntotal was validated as positive") {
            return invalid("index artifact postings rows do not sum to ntotal");
        }
        Ok(())
    }

    fn validate_source(&self) -> Result<()> {
        let Some(source) = &self.source else {
            return Ok(());
        };
        if source.rows == 0
            || source.rows > JSON_SAFE_INTEGER_MAX
            || source.rows != u64::try_from(self.index.ntotal).unwrap_or_default()
            || source.row_group_rows == 0
            || source.row_group_rows > JSON_SAFE_INTEGER_MAX
            || source.key.data_type != "long"
            || self.index.source_key_fields.as_slice() != [source.key.clone()]
            || source.columns.is_empty()
            || source.columns.iter().any(String::is_empty)
            || !source
                .columns
                .iter()
                .any(|column| column == &source.key.name)
            || source.columns.iter().collect::<HashSet<_>>().len() != source.columns.len()
            || source.files.is_empty()
        {
            return invalid("invalid static publication source descriptor");
        }
        let mut expected_begin = 0_u64;
        let mut paths = HashSet::with_capacity(source.files.len());
        for file in &source.files {
            validate_relative_location(&file.path)?;
            validate_sha256(&file.sha256)?;
            if !file.path.ends_with(".parquet")
                || file.size == 0
                || file.size > JSON_SAFE_INTEGER_MAX
                || file.row_begin != expected_begin
                || file.row_end <= file.row_begin
                || file.row_end > source.rows
                || !paths.insert(file.path.as_str())
            {
                return invalid("invalid static publication source file");
            }
            expected_begin = file.row_end;
        }
        if expected_begin != source.rows {
            return invalid("static publication source files do not partition its rows");
        }
        Ok(())
    }

    fn validate_embedding(&self) -> Result<()> {
        let Some(embedding) = &self.embedding else {
            return Ok(());
        };
        if embedding.repository.is_empty()
            || embedding.revision.is_empty()
            || embedding.runtime != "onnx"
            || embedding.dimension != self.index.dimension
            || embedding.max_length <= 0
            || embedding.pooling.is_empty()
            || embedding.input_template.is_empty()
            || embedding.parity_probe.text.is_empty()
            || embedding.parity_probe.vector.len()
                != usize::try_from(embedding.dimension).unwrap_or_default()
            || embedding
                .parity_probe
                .vector
                .iter()
                .any(|value| !value.is_finite())
            || !embedding.parity_probe.max_absolute_error.is_finite()
            || embedding.parity_probe.max_absolute_error <= 0.0
            || embedding.assets.is_empty()
        {
            return invalid("invalid static publication embedding descriptor");
        }
        validate_relative_location(&embedding.onnx_file)?;
        let mut paths = HashSet::with_capacity(embedding.assets.len());
        for asset in &embedding.assets {
            validate_asset(asset)?;
            if !paths.insert(asset.path.as_str()) {
                return invalid("embedding asset paths must be unique");
            }
        }
        if !paths.contains(embedding.onnx_file.as_str()) {
            return invalid("embedding ONNX file is missing from the asset inventory");
        }
        Ok(())
    }

    fn validate_unique_paths(&self) -> Result<()> {
        let mut paths = HashSet::<String>::new();
        let mut insert = |path: &str| {
            if paths.insert(path.to_owned()) {
                Ok(())
            } else {
                invalid("publication object paths must be globally unique")
            }
        };
        insert(&self.hierarchy.centroids.path)?;
        for file in &self.postings.files {
            insert(&file.path)?;
        }
        if let Some(source) = &self.source {
            for file in &source.files {
                insert(&file.path)?;
            }
        }
        if let Some(embedding) = &self.embedding {
            for asset in &embedding.assets {
                insert(&asset.path)?;
            }
        }
        Ok(())
    }
}

fn validate_object(object: &StaticIndexObject) -> Result<()> {
    validate_relative_location(&object.path)?;
    validate_sha256(&object.sha256)?;
    if !object.path.ends_with(".parquet") || object.size == 0 || object.size > JSON_SAFE_INTEGER_MAX
    {
        return invalid("invalid index artifact Parquet object");
    }
    Ok(())
}

fn validate_asset(object: &StaticIndexObject) -> Result<()> {
    validate_relative_location(&object.path)?;
    validate_sha256(&object.sha256)?;
    if object.size == 0 || object.size > JSON_SAFE_INTEGER_MAX {
        return invalid("invalid static publication asset");
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return invalid("index artifact sha256 must be 64 lowercase hexadecimal characters");
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
        return invalid("unsupported index artifact source-key type");
    };
    let parsed = length
        .parse::<u32>()
        .ok()
        .filter(|parsed| *parsed > 0 && parsed.to_string() == length);
    if parsed.is_none() {
        return invalid("invalid index artifact fixed source-key type");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn valid_manifest() -> IndexArtifactManifest {
        IndexArtifactManifest {
            format_version: 1,
            artifact_uuid: Uuid::parse_str("249343c7-9989-48d8-b2ca-d0caa62ba940").unwrap(),
            index: StaticIndexDescriptor {
                vector_field: "embedding".into(),
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
                cid_offsets: vec![0, 2, 4],
                centroid_encoding: PostingEncoding::Lvq8,
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
            source: None,
            embedding: None,
        }
    }

    #[test]
    fn round_trips_a_valid_manifest() {
        let manifest = valid_manifest();
        let json = manifest.to_json_vec().unwrap();
        assert_eq!(
            IndexArtifactManifest::from_json_slice(&json).unwrap(),
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
            IndexArtifactManifest::from_json_slice(&serde_json::to_vec(&json).unwrap()).is_err()
        );
    }

    #[test]
    fn validates_optional_source_and_embedding_sections() {
        let mut manifest = valid_manifest();
        manifest.index.source_key_fields[0].data_type = "long".into();
        manifest.source = Some(StaticSourceDescriptor {
            rows: 10,
            row_group_rows: 4,
            key: manifest.index.source_key_fields[0].clone(),
            columns: vec!["document_id".into(), "title".into()],
            files: vec![StaticSourceFile {
                path: "documents.parquet".into(),
                size: 400,
                sha256: digest('e'),
                row_begin: 0,
                row_end: 10,
            }],
        });
        manifest.embedding = Some(StaticEmbeddingDescriptor {
            repository: "example/model".into(),
            revision: "0123456789abcdef".into(),
            runtime: "onnx".into(),
            onnx_file: "models/model.onnx".into(),
            dimension: 2,
            max_length: 256,
            pooling: "attention-mask-mean".into(),
            normalize: true,
            input_template: "{text}".into(),
            parity_probe: StaticEmbeddingParityProbe {
                text: "hello".into(),
                vector: vec![0.0, 1.0],
                max_absolute_error: 0.002,
            },
            assets: vec![StaticIndexObject {
                path: "models/model.onnx".into(),
                size: 500,
                sha256: digest('f'),
            }],
        });
        manifest.validate().unwrap();

        manifest.source.as_mut().unwrap().files[0].row_end = 9;
        assert!(manifest.validate().is_err());
    }
}
