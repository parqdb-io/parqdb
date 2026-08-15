//! Process-scoped execution resources shared by local sessions.

use std::sync::Arc;

use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion_datasource_parquet::ParquetPageCacheFactory;

use crate::Result;
use crate::parquet::{
    DecompressedParquetPageCache, ParquetPageCacheStats, RelifyParquetPageCacheFactory,
    automatic_page_cache_capacity,
};

/// `DataFusion` and cache resources that may be shared by independent sessions.
pub struct RelifyRuntime {
    datafusion: Arc<RuntimeEnv>,
    parquet_page_cache: Arc<DecompressedParquetPageCache>,
    parquet_page_cache_factory: Arc<dyn ParquetPageCacheFactory>,
}

impl RelifyRuntime {
    /// Creates a runtime from a `DataFusion` runtime builder.
    ///
    /// When `parquet_page_cache_capacity` is `None`, the cache uses 20% of the
    /// `DataFusion` memory-pool limit, or 20% of the process memory limit when
    /// the pool is unbounded.
    pub fn new(
        builder: RuntimeEnvBuilder,
        parquet_page_cache_capacity: Option<usize>,
    ) -> Result<Self> {
        let datafusion = Arc::new(builder.build()?);
        let automatic_capacity = automatic_page_cache_capacity(datafusion.memory_pool.as_ref());
        let capacity = parquet_page_cache_capacity.unwrap_or(automatic_capacity);
        let parquet_page_cache = Arc::new(DecompressedParquetPageCache::new(capacity));
        let parquet_page_cache_factory: Arc<dyn ParquetPageCacheFactory> = Arc::new(
            RelifyParquetPageCacheFactory::new(Arc::clone(&parquet_page_cache), capacity),
        );
        Ok(Self {
            datafusion,
            parquet_page_cache,
            parquet_page_cache_factory,
        })
    }

    /// Returns the shared `DataFusion` runtime environment.
    #[must_use]
    pub fn datafusion(&self) -> Arc<RuntimeEnv> {
        Arc::clone(&self.datafusion)
    }

    /// Returns allocation and lookup counters for the shared Parquet Page cache.
    #[must_use]
    pub fn parquet_page_cache_stats(&self) -> ParquetPageCacheStats {
        self.parquet_page_cache.stats()
    }

    /// Removes resident Parquet Pages from future cache lookups.
    ///
    /// Pages referenced by active queries remain alive until those queries
    /// release their Arrow buffers.
    pub fn clear_parquet_page_cache(&self) {
        self.parquet_page_cache.clear();
    }

    pub(crate) fn parquet_page_cache_factory(&self) -> Arc<dyn ParquetPageCacheFactory> {
        Arc::clone(&self.parquet_page_cache_factory)
    }
}
