use std::future::Future;
use std::mem::size_of_val;
use std::sync::{Arc, Mutex};

use hashlink::LinkedHashMap;
use tokio::sync::OnceCell;

use crate::{Error, Result};

const DEFAULT_ENTRIES: usize = 128;
const DEFAULT_BYTES: usize = 256 * 1024 * 1024;

type CentroidCell = Arc<OnceCell<Arc<CentroidMatrix>>>;

#[derive(Debug)]
pub(in crate::session) struct CentroidMatrix {
    nlist: usize,
    dimension: usize,
    values: Arc<[f32]>,
}

impl CentroidMatrix {
    pub(in crate::session) fn new(
        nlist: usize,
        dimension: usize,
        values: Vec<f32>,
    ) -> Result<Self> {
        if nlist.checked_mul(dimension) != Some(values.len()) {
            return Err(Error::InvalidSchema(
                "invalid cached centroid matrix shape".into(),
            ));
        }
        Ok(Self {
            nlist,
            dimension,
            values: Arc::from(values),
        })
    }

    pub(in crate::session) fn values(&self, nlist: usize, dimension: usize) -> Result<&[f32]> {
        if self.nlist != nlist || self.dimension != dimension {
            return Err(Error::InvalidSchema(
                "cached centroid matrix shape does not match index metadata".into(),
            ));
        }
        Ok(self.values.as_ref())
    }

    fn charge(&self) -> usize {
        size_of_val(self.values.as_ref())
    }
}

struct CacheEntry {
    cell: CentroidCell,
    charge: usize,
}

#[derive(Default)]
struct CacheState {
    entries: LinkedHashMap<String, CacheEntry>,
    resident_bytes: usize,
}

pub(super) struct CentroidCache {
    state: Mutex<CacheState>,
    entry_capacity: usize,
    byte_capacity: usize,
}

impl Default for CentroidCache {
    fn default() -> Self {
        Self::new(DEFAULT_ENTRIES, DEFAULT_BYTES)
    }
}

impl CentroidCache {
    pub(super) fn new(entry_capacity: usize, byte_capacity: usize) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            entry_capacity,
            byte_capacity,
        }
    }

    pub(super) async fn get_or_load<F, Fut>(
        &self,
        relation_key: &str,
        load: F,
    ) -> Result<Arc<CentroidMatrix>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CentroidMatrix>>,
    {
        if self.entry_capacity == 0 || self.byte_capacity == 0 {
            return load().await.map(Arc::new);
        }
        let cell = {
            let mut state = self.state();
            if let Some(entry) = state.entries.to_back(relation_key) {
                Arc::clone(&entry.cell)
            } else {
                let cell = Arc::new(OnceCell::new());
                state.entries.insert(
                    relation_key.to_owned(),
                    CacheEntry {
                        cell: Arc::clone(&cell),
                        charge: 0,
                    },
                );
                self.evict(&mut state);
                cell
            }
        };

        let matrix = Arc::clone(
            cell.get_or_try_init(|| async { load().await.map(Arc::new) })
                .await?,
        );
        let charge = matrix.charge();
        let mut state = self.state();
        if let Some(entry) = state.entries.get_mut(relation_key)
            && Arc::ptr_eq(&entry.cell, &cell)
            && entry.charge == 0
        {
            entry.charge = charge;
            state.resident_bytes = state.resident_bytes.saturating_add(charge);
        }
        self.evict(&mut state);
        Ok(matrix)
    }

    fn evict(&self, state: &mut CacheState) {
        while state.entries.len() > self.entry_capacity || state.resident_bytes > self.byte_capacity
        {
            let Some((_, entry)) = state.entries.pop_front() else {
                break;
            };
            state.resident_bytes = state.resident_bytes.saturating_sub(entry.charge);
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    fn stats(&self) -> (usize, usize) {
        let state = self.state();
        (state.entries.len(), state.resident_bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn concurrent_misses_load_one_matrix() {
        let cache = CentroidCache::new(2, 1024);
        let loads = Arc::new(AtomicUsize::new(0));
        let first_loads = Arc::clone(&loads);
        let second_loads = Arc::clone(&loads);

        let (first, second) = tokio::join!(
            cache.get_or_load("centroids", || async move {
                first_loads.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                CentroidMatrix::new(2, 2, vec![0.0, 1.0, 2.0, 3.0])
            }),
            cache.get_or_load("centroids", || async move {
                second_loads.fetch_add(1, Ordering::Relaxed);
                CentroidMatrix::new(2, 2, vec![0.0, 1.0, 2.0, 3.0])
            })
        );

        assert!(Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(cache.stats(), (1, 4 * size_of::<f32>()));
    }

    #[tokio::test]
    async fn byte_budget_evicts_least_recently_used_matrix() {
        let cache = CentroidCache::new(4, 12);
        for relation_key in ["a", "b"] {
            cache
                .get_or_load(relation_key, || async {
                    CentroidMatrix::new(1, 2, vec![0.0, 1.0])
                })
                .await
                .unwrap();
        }

        assert_eq!(cache.stats(), (1, 2 * size_of::<f32>()));
    }

    #[tokio::test]
    async fn oversized_matrix_is_not_retained() {
        let cache = CentroidCache::new(4, 4);
        let matrix = cache
            .get_or_load("centroids", || async {
                CentroidMatrix::new(1, 2, vec![0.0, 1.0])
            })
            .await
            .unwrap();

        assert_eq!(matrix.values(1, 2).unwrap(), [0.0, 1.0]);
        assert_eq!(cache.stats(), (0, 0));
    }
}
