//! Embedded `DataFusion` and Parquet implementation of `ParqDB`.

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

pub use config::{LocalSessionOptions, ParqDBConfig, parqdb_session_config};
pub use error::{BuildFailureKind, Error, Result};
pub use maintenance::{MaintenanceKind, MaintenanceObject};
pub use parqdb_catalog::IndexIdentifier;
pub use parqdb_core::{
    DistanceMetric, IndexArtifacts, IndexFormat, IvfConfig, PostingEncoding, PublishedIndex,
    SearchRequest,
};
pub use parqdb_index::MetadataCacheConfig;
pub use parqdb_meta::{IndexMetadata, IndexSnapshot, RelationReference};
pub use parquet::{ParquetPageCacheStats, ParquetWriterOptions};
pub use progress::{LocalBuildProgress, LocalBuildProgressSnapshot};
pub use query::{
    ManagedQueryStream, compile_datafusion_sql, datafusion_centroid_relation_required,
    datafusion_cluster_relation_required, datafusion_source_relation_required, squared_l2_udf,
};
pub use runtime::{ParqDBRuntime, QueryAdmissionOptions, QueryAdmissionStats};
pub use search::{ClusterSelection, ResolvedSearch};
pub use session::{
    IndexBuildState, IndexBuildStatus, IndexInfo, LocalBuildOptions, LocalSession,
    PersistentParquetOptions, SourceDescription, SourceField,
};
