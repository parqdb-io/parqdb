/// `DataFusion` cluster-selection work retained in a resolved IVF search.
#[derive(Debug, Clone)]
pub enum ClusterSelection {
    /// Probe every cluster without adding a postings predicate.
    All,
    /// Cluster IDs selected by Relify's native SIMD router.
    Native(Vec<i32>),
    /// Select clusters inside the backend query plan.
    Relational {
        /// Backend relation key for the IVF centroid relation.
        centroids_relation_key: String,
        /// Number of clusters selected by the backend Top-K.
        nprobe: usize,
    },
}

use relify_meta::{DistanceMetric, PostingEncoding};

/// Fully resolved inputs for one embedded `DataFusion` vector search.
#[derive(Debug, Clone)]
pub struct ResolvedSearch {
    /// Backend relation key for the source table.
    pub source_relation_key: String,
    /// Query vector after conversion to the canonical `float` type.
    pub query: Vec<f32>,
    /// Distance metric applied by this search.
    pub metric: DistanceMetric,
    /// Source column containing vectors.
    pub vector_field: String,
    /// Whether source scoring must cast double elements to canonical float.
    pub source_vector_is_f64: bool,
    /// Ordered source key fields used to resolve postings to source rows.
    pub source_key_fields: Vec<String>,
    /// Backend relation key for IVF postings, absent for exact search.
    pub postings_relation_key: Option<String>,
    /// Vector representation stored in IVF postings.
    pub posting_encoding: PostingEncoding,
    /// IVF cluster selection, absent for exact search.
    pub cluster_selection: Option<ClusterSelection>,
    /// Total number of IVF clusters, absent for exact search.
    pub nlist: Option<usize>,
    /// Indexed source row count, absent for exact search.
    pub ntotal: Option<usize>,
    /// Ordered source columns returned before `_distance`.
    pub projection: Vec<String>,
    /// Optional backend predicate evaluated against source rows before Top-K.
    pub filter: Option<String>,
    /// Maximum result count.
    pub limit: usize,
}
