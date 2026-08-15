use std::collections::BTreeMap;

use relify_catalog::IndexIdentifier;
use relify_meta::{
    DistanceMetric, IVF_SCHEMA_VERSION, IndexMetadata, PostingEncoding, RelationReference,
};

/// Portable identity of one index representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexFormat {
    /// Index-family identifier.
    pub family: String,
    /// Family-defined schema version.
    pub schema_version: i32,
    /// Distance metric evaluated by the index.
    pub metric: String,
}

impl IndexFormat {
    /// Returns the current IVF format for `metric`.
    #[must_use]
    pub fn ivf(metric: DistanceMetric) -> Self {
        Self {
            family: "ivf".into(),
            schema_version: IVF_SCHEMA_VERSION,
            metric: metric.as_str().into(),
        }
    }
}

/// Logical IVF construction options shared by backend implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfConfig {
    /// Number of IVF clusters.
    pub nlist: usize,
    /// Vector representation stored in IVF postings.
    pub posting_encoding: PostingEncoding,
    /// Distance metric used for training and search.
    pub metric: DistanceMetric,
}

impl IvfConfig {
    /// Creates an IVF configuration with one canonical postings encoding.
    #[must_use]
    pub const fn new(nlist: usize, posting_encoding: PostingEncoding) -> Self {
        Self {
            nlist,
            posting_encoding,
            metric: DistanceMetric::L2Squared,
        }
    }

    /// Creates an IVF configuration with an explicit metric.
    #[must_use]
    pub const fn with_metric(
        nlist: usize,
        posting_encoding: PostingEncoding,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            nlist,
            posting_encoding,
            metric,
        }
    }
}

/// User-supplied inputs for one vector search.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// Exact portable reference for the source table.
    pub source: RelationReference,
    /// Catalog namespace containing indexes for this table binding.
    pub index_namespace: Vec<String>,
    /// Explicit index name, or `None` for source-based discovery.
    pub index: Option<String>,
    /// Vector column used to disambiguate implicit index discovery.
    pub column: Option<String>,
    /// Query vector.
    pub query: Vec<f32>,
    /// Number of IVF clusters to probe.
    pub nprobe: Option<usize>,
    /// Maximum number of results.
    pub limit: usize,
    /// Optional ordered source-column projection.
    pub projection: Option<Vec<String>>,
    /// Optional backend expression evaluated against source rows before Top-K.
    pub filter: Option<String>,
    /// Whether to scan the source exactly without resolving a vector index.
    pub bypass_index: bool,
}

/// Immutable index relations and family parameters produced by a backend builder.
#[derive(Debug, Clone)]
pub struct IndexArtifacts {
    /// Portable family, schema version, and metric produced by the builder.
    pub format: IndexFormat,
    /// Family-defined canonical parameter values.
    pub parameters: BTreeMap<String, String>,
    /// Relation roles and their exact portable references.
    pub index_relations: BTreeMap<String, RelationReference>,
}

/// Result of publishing one immutable index metadata document.
#[derive(Debug, Clone)]
pub struct PublishedIndex {
    /// Structured catalog identifier that was published.
    pub identifier: IndexIdentifier,
    /// Immutable metadata file made current by the catalog.
    pub metadata_location: String,
    /// Validated metadata stored at `metadata_location`.
    pub metadata: IndexMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posting_encodings_have_canonical_metadata_names() {
        for (encoding, name) in [
            (PostingEncoding::Source, "source"),
            (PostingEncoding::Lvq4, "lvq4"),
            (PostingEncoding::Lvq8, "lvq8"),
        ] {
            assert_eq!(encoding.as_str(), name);
        }
    }
}
