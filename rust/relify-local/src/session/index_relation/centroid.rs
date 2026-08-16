use std::future::Future;
use std::sync::Arc;

use super::bounded_cache::BoundedAsyncCache;
use crate::Result;
pub(in crate::session) use crate::centroid_navigation::CentroidNavigator;

pub(super) struct CentroidCache {
    cache: BoundedAsyncCache<String, Arc<CentroidNavigator>>,
}

impl CentroidCache {
    pub(super) fn new(entry_capacity: usize, byte_capacity: usize) -> Self {
        Self {
            cache: BoundedAsyncCache::new(entry_capacity, byte_capacity),
        }
    }

    pub(super) async fn get_or_load<F, Fut>(
        &self,
        relation_key: &str,
        load: F,
    ) -> Result<Arc<CentroidNavigator>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CentroidNavigator>>,
    {
        self.cache
            .get_or_try_insert(relation_key.to_owned(), || async {
                let navigator = Arc::new(load().await?);
                let charge = navigator.resident_size();
                Ok((navigator, charge))
            })
            .await
    }

    #[cfg(test)]
    fn stats(&self) -> (usize, usize) {
        self.cache.stats()
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[tokio::test]
    async fn caches_one_exact_navigator() {
        let cache = CentroidCache::new(2, 1024);
        let navigator = cache
            .get_or_load("centroids", || async {
                CentroidNavigator::new(1, 2, Arc::from([0.0, 1.0]))
            })
            .await
            .unwrap();

        navigator.validate_shape(1, 2).unwrap();
        assert!(navigator.validate_shape(2, 1).is_err());
        assert_eq!(cache.stats(), (1, 2 * size_of::<f32>()));
    }
}
