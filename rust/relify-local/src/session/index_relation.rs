//! Session-scoped providers for immutable index relations.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, RwLock};

use arrow::datatypes::DataType;
use datafusion::catalog::TableProvider;
use hashlink::LinkedHashMap;
use tokio::sync::OnceCell;

use crate::parquet::ParquetStore;
use crate::{Error, Result};

const DEFAULT_PARQUET_PROVIDER_CACHE_ENTRIES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum IndexRelationLayout {
    Plain,
    HiveCid,
}

impl IndexRelationLayout {
    fn partition_columns(self) -> Vec<(String, DataType)> {
        match self {
            Self::Plain => Vec::new(),
            Self::HiveCid => vec![("cid".into(), DataType::Int32)],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParquetProviderKey {
    relation_key: String,
    layout: IndexRelationLayout,
}

type Provider = Arc<dyn TableProvider>;
type ProviderCell = Arc<OnceCell<Provider>>;

/// Resolves explicit providers and reuses lazily constructed Parquet providers.
pub(super) struct IndexRelationProviderRegistry {
    registered: RwLock<HashMap<String, Provider>>,
    parquet: Mutex<LinkedHashMap<ParquetProviderKey, ProviderCell>>,
    parquet_capacity: usize,
}

impl Default for IndexRelationProviderRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_PARQUET_PROVIDER_CACHE_ENTRIES)
    }
}

impl IndexRelationProviderRegistry {
    fn new(parquet_capacity: usize) -> Self {
        Self {
            registered: RwLock::new(HashMap::new()),
            parquet: Mutex::new(LinkedHashMap::new()),
            parquet_capacity,
        }
    }

    pub(super) fn registered(&self, relation_key: &str) -> Result<Option<Provider>> {
        Ok(self
            .registered
            .read()
            .map_err(|_| provider_lock_error())?
            .get(relation_key)
            .cloned())
    }

    pub(super) fn register(&self, relation_key: String, provider: Provider) -> Result<Provider> {
        let mut registered = self.registered.write().map_err(|_| provider_lock_error())?;
        Ok(Arc::clone(
            registered.entry(relation_key).or_insert(provider),
        ))
    }

    pub(super) async fn get_or_create_parquet(
        &self,
        relation_key: &str,
        layout: IndexRelationLayout,
        parquet: &ParquetStore,
    ) -> Result<Provider> {
        if let Some(provider) = self.registered(relation_key)? {
            return Ok(provider);
        }

        let key = ParquetProviderKey {
            relation_key: relation_key.to_owned(),
            layout,
        };
        self.get_or_create(key, || async {
            parquet
                .uniform_dataset_provider(relation_key, layout.partition_columns())
                .await
        })
        .await
    }

    async fn get_or_create<F, Fut>(&self, key: ParquetProviderKey, create: F) -> Result<Provider>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Provider>>,
    {
        let cell = {
            let mut providers = self
                .parquet
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(cell) = providers.to_back(&key) {
                Arc::clone(cell)
            } else {
                let cell = Arc::new(OnceCell::new());
                providers.insert(key, Arc::clone(&cell));
                while providers.len() > self.parquet_capacity {
                    providers.pop_front();
                }
                cell
            }
        };

        let provider = cell.get_or_try_init(create).await?;
        Ok(Arc::clone(provider))
    }

    #[cfg(test)]
    pub(super) fn parquet_entry_count(&self) -> usize {
        self.parquet
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

fn provider_lock_error() -> Error {
    Error::InvalidArgument("relation provider lock is poisoned".into())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;

    use super::*;

    fn empty_provider() -> Provider {
        Arc::new(MemTable::try_new(Arc::new(Schema::empty()), vec![Vec::new()]).unwrap())
    }

    #[tokio::test]
    async fn concurrent_misses_build_one_provider() {
        let registry = IndexRelationProviderRegistry::new(2);
        let builds = Arc::new(AtomicUsize::new(0));
        let key = ParquetProviderKey {
            relation_key: "relation".into(),
            layout: IndexRelationLayout::Plain,
        };

        let first_builds = Arc::clone(&builds);
        let second_builds = Arc::clone(&builds);
        let (first, second) = tokio::join!(
            registry.get_or_create(key.clone(), || async move {
                first_builds.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                Ok(empty_provider())
            }),
            registry.get_or_create(key, || async move {
                second_builds.fetch_add(1, Ordering::Relaxed);
                Ok(empty_provider())
            })
        );
        let first = first.unwrap();
        let second = second.unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn parquet_provider_cache_is_bounded() {
        let registry = IndexRelationProviderRegistry::new(2);
        for relation_key in ["a", "b", "c"] {
            registry
                .get_or_create(
                    ParquetProviderKey {
                        relation_key: relation_key.into(),
                        layout: IndexRelationLayout::Plain,
                    },
                    || async { Ok(empty_provider()) },
                )
                .await
                .unwrap();
        }

        assert_eq!(registry.parquet_entry_count(), 2);
    }
}
