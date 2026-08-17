//! Embedded `DataFusion` and Parquet implementation of Relify.

mod builder;
mod centroid_navigation;
mod config;
mod coordination;
mod durability;
mod error;
mod ivf;
mod local_uri;
mod maintenance;
mod parquet;
mod progress;
mod query;
mod runtime;
mod search;
mod session;
mod vector;

pub use config::{LocalSessionOptions, RelifyConfig, relify_session_config};
pub use error::{BuildFailureKind, Error, Result};
pub use maintenance::{MaintenanceKind, MaintenanceObject};
pub use parquet::{ParquetPageCacheStats, ParquetWriterOptions};
pub use progress::{LocalBuildProgress, LocalBuildProgressSnapshot};
pub use query::{
    ManagedQueryStream, compile_datafusion_sql, datafusion_centroid_relation_required,
    datafusion_cluster_relation_required, datafusion_source_relation_required, squared_l2_udf,
};
pub use relify_catalog::IndexIdentifier;
pub use relify_core::{
    BatchSearchRequest, DistanceMetric, IndexArtifacts, IndexFormat, IvfConfig, PostingEncoding,
    PublishedIndex, SearchRequest,
};
pub use relify_index::MetadataCacheConfig;
pub use relify_meta::{IndexMetadata, IndexSnapshot, RelationReference};
pub use runtime::{QueryAdmissionOptions, QueryAdmissionStats, RelifyRuntime};
pub use search::{ClusterSelection, ResolvedSearch};
pub use session::{
    IndexBuildState, IndexBuildStatus, IndexInfo, LocalBuildOptions, LocalSession,
    PersistentParquetOptions, SourceDescription, SourceField,
};
