use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use datafusion::common::config::{
    ConfigEntry, ConfigExtension, ConfigField, ExtensionOptions, Visit,
};
use datafusion::common::{DataFusionError, Result as DataFusionResult, config_namespace};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::SessionConfig;
use relify_index::{
    DEFAULT_METADATA_CACHE_BYTES, DEFAULT_METADATA_CACHE_ENTRIES, MetadataCacheConfig,
};

use crate::Result;
use crate::runtime::{QueryAdmissionOptions, RelifyRuntime};

const DEFAULT_MANIFEST_CACHE_ENTRIES: usize = 128;
const DEFAULT_MANIFEST_CACHE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_CENTROID_CACHE_ENTRIES: usize = 128;
const DEFAULT_CENTROID_CACHE_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_QUERY_CONCURRENCY: usize = 1;
const DEFAULT_QUERY_QUEUE_CAPACITY: usize = 64;

config_namespace! {
    /// Options for runtime query admission.
    #[allow(clippy::struct_field_names)]
    pub struct ExecutionOptions {
        /// Target DataFusion execution parallelism for one query.
        pub query_dop: Option<usize>, default = None

        /// Maximum number of admitted queries.
        pub query_concurrency: usize, default = DEFAULT_QUERY_CONCURRENCY

        /// Maximum number of queries waiting for admission.
        pub query_queue_capacity: usize, default = DEFAULT_QUERY_QUEUE_CAPACITY

        /// Maximum admission wait in a human-readable duration such as `500ms` or `30s`.
        pub query_queue_timeout: String, default = "30s".into()
    }
}

config_namespace! {
    /// Options for index metadata caching.
    pub struct MetadataCacheOptions {
        /// Maximum number of cached metadata documents.
        pub max_entries: usize, default = DEFAULT_METADATA_CACHE_ENTRIES

        /// Maximum serialized bytes represented by cached metadata documents.
        pub max_bytes: usize, default = DEFAULT_METADATA_CACHE_BYTES
    }
}

config_namespace! {
    /// Options for the decompressed Parquet Page cache.
    pub struct ParquetPageCacheOptions {
        /// Cache capacity in bytes. `None` uses 20% of the effective memory limit; zero disables the cache.
        pub capacity: Option<usize>, default = None
    }
}

config_namespace! {
    /// Options for Parquet reads.
    pub struct ParquetOptions {
        /// Decompressed Page-cache options.
        pub page_cache: ParquetPageCacheOptions, default = ParquetPageCacheOptions::default()
    }
}

config_namespace! {
    /// Options for index metadata.
    pub struct MetadataOptions {
        /// Metadata cache options.
        pub cache: MetadataCacheOptions, default = MetadataCacheOptions::default()
    }
}

config_namespace! {
    /// Options for immutable index-relation manifests.
    pub struct ManifestCacheOptions {
        /// Maximum number of cached manifests.
        pub max_entries: usize, default = DEFAULT_MANIFEST_CACHE_ENTRIES

        /// Maximum estimated bytes retained by cached manifests.
        pub max_bytes: usize, default = DEFAULT_MANIFEST_CACHE_BYTES
    }
}

config_namespace! {
    /// Options for index-relation manifest planning.
    pub struct ManifestOptions {
        /// Manifest cache options.
        pub cache: ManifestCacheOptions, default = ManifestCacheOptions::default()
    }
}

config_namespace! {
    /// Options for native centroid routing matrices.
    pub struct CentroidCacheOptions {
        /// Maximum number of cached centroid matrices.
        pub max_entries: usize, default = DEFAULT_CENTROID_CACHE_ENTRIES

        /// Maximum bytes retained by cached centroid values.
        pub max_bytes: usize, default = DEFAULT_CENTROID_CACHE_BYTES
    }
}

config_namespace! {
    /// Options for centroid routing.
    pub struct CentroidOptions {
        /// Centroid matrix cache options.
        pub cache: CentroidCacheOptions, default = CentroidCacheOptions::default()
    }
}

config_namespace! {
    /// Options for query planning and routing.
    pub struct QueryOptions {
        /// Index-relation manifest options.
        pub manifest: ManifestOptions, default = ManifestOptions::default()

        /// Centroid routing options.
        pub centroid: CentroidOptions, default = CentroidOptions::default()
    }
}

/// Relify options carried by `DataFusion`'s session configuration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RelifyConfig {
    /// Runtime execution options.
    pub execution: ExecutionOptions,
    /// Index metadata options.
    pub metadata: MetadataOptions,
    /// Parquet reader options.
    pub parquet: ParquetOptions,
    /// Query planning and routing options.
    pub query: QueryOptions,
}

impl ConfigExtension for RelifyConfig {
    const PREFIX: &'static str = "relify";
}

impl ExtensionOptions for RelifyConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> DataFusionResult<()> {
        ConfigField::set(self, key, value)
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        struct EntryVisitor(Vec<ConfigEntry>);

        impl Visit for EntryVisitor {
            fn some<V: Display>(&mut self, key: &str, value: V, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_owned(),
                    value: Some(value.to_string()),
                    description,
                });
            }

            fn none(&mut self, key: &str, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_owned(),
                    value: None,
                    description,
                });
            }
        }

        let mut visitor = EntryVisitor(Vec::new());
        ConfigField::visit(self, &mut visitor, "relify", "");
        visitor.0
    }
}

impl ConfigField for RelifyConfig {
    fn visit<V: Visit>(&self, visitor: &mut V, key_prefix: &str, _description: &'static str) {
        self.execution.visit(
            visitor,
            &format!("{key_prefix}.execution"),
            "Runtime execution options.",
        );
        self.metadata.visit(
            visitor,
            &format!("{key_prefix}.metadata"),
            "Index metadata options.",
        );
        self.parquet.visit(
            visitor,
            &format!("{key_prefix}.parquet"),
            "Parquet reader options.",
        );
        self.query.visit(
            visitor,
            &format!("{key_prefix}.query"),
            "Query planning and routing options.",
        );
    }

    fn set(&mut self, key: &str, value: &str) -> DataFusionResult<()> {
        let (section, remainder) = key.split_once('.').unwrap_or((key, ""));
        match section {
            "execution" => self.execution.set(remainder, value),
            "metadata" => self.metadata.set(remainder, value),
            "parquet" => self.parquet.set(remainder, value),
            "query" => self.query.set(remainder, value),
            _ => Err(DataFusionError::Configuration(format!(
                "Config value \"{section}\" not found on RelifyConfig"
            ))),
        }
    }
}

/// `DataFusion` initialization options for one local Relify session.
#[derive(Clone)]
pub struct LocalSessionOptions {
    config: SessionConfig,
    runtime: LocalRuntimeOptions,
}

#[derive(Clone)]
enum LocalRuntimeOptions {
    Builder(RuntimeEnvBuilder),
    Shared(Arc<RelifyRuntime>),
}

impl LocalSessionOptions {
    /// Creates options from `DataFusion`'s native configuration types.
    #[must_use]
    pub const fn new(config: SessionConfig, runtime: RuntimeEnvBuilder) -> Self {
        Self {
            config,
            runtime: LocalRuntimeOptions::Builder(runtime),
        }
    }

    /// Creates session options that reuse process-scoped execution resources.
    #[must_use]
    pub fn with_runtime(config: SessionConfig, runtime: Arc<RelifyRuntime>) -> Self {
        Self {
            config,
            runtime: LocalRuntimeOptions::Shared(runtime),
        }
    }

    pub(crate) fn into_parts(self) -> Result<(SessionConfig, Arc<RelifyRuntime>)> {
        let mut config = ensure_relify_config(self.config);
        if let Some(query_dop) = query_dop(&config)? {
            config = config.with_target_partitions(query_dop);
        }
        let runtime = match self.runtime {
            LocalRuntimeOptions::Builder(builder) => Arc::new(RelifyRuntime::with_query_admission(
                builder,
                parquet_page_cache_capacity(&config),
                query_admission_options(&config)?,
            )?),
            LocalRuntimeOptions::Shared(runtime) => runtime,
        };
        Ok((config, runtime))
    }
}

impl Default for LocalSessionOptions {
    fn default() -> Self {
        Self::new(relify_session_config(), RuntimeEnvBuilder::default())
    }
}

/// Creates a `DataFusion` session configuration with Relify options installed.
#[must_use]
pub fn relify_session_config() -> SessionConfig {
    ensure_relify_config(SessionConfig::new())
}

fn ensure_relify_config(mut config: SessionConfig) -> SessionConfig {
    if config.options().extensions.get::<RelifyConfig>().is_none() {
        config
            .options_mut()
            .extensions
            .insert(RelifyConfig::default());
    }
    config
}

pub(crate) fn metadata_cache_config(config: &SessionConfig) -> MetadataCacheConfig {
    let cache = &config
        .options()
        .extensions
        .get::<RelifyConfig>()
        .expect("Relify config extension must be installed")
        .metadata
        .cache;
    MetadataCacheConfig::new(cache.max_entries, cache.max_bytes)
}

pub(crate) fn parquet_page_cache_capacity(config: &SessionConfig) -> Option<usize> {
    config
        .options()
        .extensions
        .get::<RelifyConfig>()
        .expect("Relify config extension must be installed")
        .parquet
        .page_cache
        .capacity
}

pub(crate) fn query_admission_options(config: &SessionConfig) -> Result<QueryAdmissionOptions> {
    let execution = &config
        .options()
        .extensions
        .get::<RelifyConfig>()
        .expect("Relify config extension must be installed")
        .execution;
    let queue_timeout =
        humantime::parse_duration(&execution.query_queue_timeout).map_err(|error| {
            crate::Error::InvalidArgument(format!(
                "invalid relify.execution.query_queue_timeout: {error}"
            ))
        })?;
    Ok(QueryAdmissionOptions {
        max_active: execution.query_concurrency,
        max_queued: execution.query_queue_capacity,
        queue_timeout,
    })
}

fn query_dop(config: &SessionConfig) -> Result<Option<usize>> {
    let query_dop = config
        .options()
        .extensions
        .get::<RelifyConfig>()
        .expect("Relify config extension must be installed")
        .execution
        .query_dop;
    if query_dop == Some(0) {
        return Err(crate::Error::InvalidArgument(
            "relify.execution.query_dop must be positive".into(),
        ));
    }
    Ok(query_dop)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexRelationCacheConfig {
    pub(crate) manifest_max_entries: usize,
    pub(crate) manifest_max_bytes: usize,
    pub(crate) centroid_max_entries: usize,
    pub(crate) centroid_max_bytes: usize,
}

impl Default for IndexRelationCacheConfig {
    fn default() -> Self {
        Self {
            manifest_max_entries: DEFAULT_MANIFEST_CACHE_ENTRIES,
            manifest_max_bytes: DEFAULT_MANIFEST_CACHE_BYTES,
            centroid_max_entries: DEFAULT_CENTROID_CACHE_ENTRIES,
            centroid_max_bytes: DEFAULT_CENTROID_CACHE_BYTES,
        }
    }
}

pub(crate) fn index_relation_cache_config(config: &SessionConfig) -> IndexRelationCacheConfig {
    let query = &config
        .options()
        .extensions
        .get::<RelifyConfig>()
        .expect("Relify config extension must be installed")
        .query;
    IndexRelationCacheConfig {
        manifest_max_entries: query.manifest.cache.max_entries,
        manifest_max_bytes: query.manifest.cache.max_bytes,
        centroid_max_entries: query.centroid.cache.max_entries,
        centroid_max_bytes: query.centroid.cache.max_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relify_config_uses_hierarchical_datafusion_keys() {
        let config = relify_session_config()
            .set_str("relify.execution.query_dop", "2")
            .set_str("relify.execution.query_concurrency", "3")
            .set_str("relify.execution.query_queue_capacity", "5")
            .set_str("relify.execution.query_queue_timeout", "750ms")
            .set_str("relify.metadata.cache.max_entries", "7")
            .set_str("relify.metadata.cache.max_bytes", "4096")
            .set_str("relify.parquet.page_cache.capacity", "8192")
            .set_str("relify.query.manifest.cache.max_entries", "3")
            .set_str("relify.query.manifest.cache.max_bytes", "1024")
            .set_str("relify.query.centroid.cache.max_entries", "5")
            .set_str("relify.query.centroid.cache.max_bytes", "2048");
        let (config, _) = LocalSessionOptions::new(config, RuntimeEnvBuilder::default())
            .into_parts()
            .unwrap();

        assert_eq!(
            query_admission_options(&config).unwrap(),
            QueryAdmissionOptions {
                max_active: 3,
                max_queued: 5,
                queue_timeout: std::time::Duration::from_millis(750),
            }
        );
        assert_eq!(config.target_partitions(), 2);
        assert_eq!(
            metadata_cache_config(&config),
            MetadataCacheConfig::new(7, 4096)
        );
        assert_eq!(parquet_page_cache_capacity(&config), Some(8192));
        assert_eq!(
            index_relation_cache_config(&config),
            IndexRelationCacheConfig {
                manifest_max_entries: 3,
                manifest_max_bytes: 1024,
                centroid_max_entries: 5,
                centroid_max_bytes: 2048,
            }
        );
    }

    #[test]
    fn prepared_relify_config_accepts_runtime_mutation() {
        let (mut config, _) = LocalSessionOptions::default().into_parts().unwrap();
        config
            .options_mut()
            .set("relify.metadata.cache.max_entries", "7")
            .unwrap();

        assert_eq!(metadata_cache_config(&config).max_entries, 7);
    }

    #[test]
    fn invalid_query_admission_config_fails_session_initialization() {
        let zero_concurrency =
            relify_session_config().set_str("relify.execution.query_concurrency", "0");
        assert!(matches!(
            LocalSessionOptions::new(zero_concurrency, RuntimeEnvBuilder::default()).into_parts(),
            Err(crate::Error::InvalidArgument(message))
                if message.contains("at least one active slot")
        ));

        let zero_dop = relify_session_config().set_str("relify.execution.query_dop", "0");
        assert!(matches!(
            LocalSessionOptions::new(zero_dop, RuntimeEnvBuilder::default()).into_parts(),
            Err(crate::Error::InvalidArgument(message)) if message.contains("query_dop")
        ));

        let invalid_timeout =
            relify_session_config().set_str("relify.execution.query_queue_timeout", "soon");
        assert!(matches!(
            LocalSessionOptions::new(invalid_timeout, RuntimeEnvBuilder::default()).into_parts(),
            Err(crate::Error::InvalidArgument(message))
                if message.contains("query_queue_timeout")
        ));
    }
}
