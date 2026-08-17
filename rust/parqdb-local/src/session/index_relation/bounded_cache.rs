use std::future::Future;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use hashlink::LinkedHashMap;
use tokio::sync::OnceCell;

use crate::Result;

type CacheCell<V> = Arc<OnceCell<LoadedValue<V>>>;

struct LoadedValue<V> {
    value: V,
    charge: usize,
}

struct CacheEntry<V> {
    cell: CacheCell<V>,
    charge: Option<usize>,
}

struct CacheState<K, V> {
    entries: LinkedHashMap<K, CacheEntry<V>>,
    resident_bytes: usize,
}

impl<K, V> Default for CacheState<K, V> {
    fn default() -> Self {
        Self {
            entries: LinkedHashMap::new(),
            resident_bytes: 0,
        }
    }
}

/// Entry- and byte-bounded LRU with single-flight initialization per admitted key.
pub(super) struct BoundedAsyncCache<K, V> {
    state: Mutex<CacheState<K, V>>,
    entry_capacity: usize,
    byte_capacity: usize,
}

impl<K, V> BoundedAsyncCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    pub(super) fn new(entry_capacity: usize, byte_capacity: usize) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            entry_capacity,
            byte_capacity,
        }
    }

    pub(super) async fn get_or_try_insert<F, Fut>(&self, key: K, load: F) -> Result<V>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(V, usize)>>,
    {
        if self.entry_capacity == 0 || self.byte_capacity == 0 {
            return load().await.map(|(value, _)| value);
        }

        let cell = {
            let mut state = self.state();
            if let Some(entry) = state.entries.to_back(&key) {
                Some(Arc::clone(&entry.cell))
            } else if self.reserve_entry(&mut state) {
                let cell = Arc::new(OnceCell::new());
                state.entries.insert(
                    key.clone(),
                    CacheEntry {
                        cell: Arc::clone(&cell),
                        charge: None,
                    },
                );
                Some(cell)
            } else {
                None
            }
        };

        let Some(cell) = cell else {
            return load().await.map(|(value, _)| value);
        };
        let loaded = match cell
            .get_or_try_init(|| async {
                let (value, charge) = load().await?;
                Ok(LoadedValue { value, charge })
            })
            .await
        {
            Ok(loaded) => loaded,
            Err(error) => {
                self.remove_cell(&key, &cell);
                return Err(error);
            }
        };

        let mut state = self.state();
        if loaded.charge > self.byte_capacity {
            remove_matching_cell(&mut state, &key, &cell);
        } else if let Some(entry) = state.entries.get_mut(&key)
            && Arc::ptr_eq(&entry.cell, &cell)
            && entry.charge.is_none()
        {
            entry.charge = Some(loaded.charge);
            state.resident_bytes = state.resident_bytes.saturating_add(loaded.charge);
            state.entries.to_back(&key);
            self.evict_to_byte_capacity(&mut state);
        }
        Ok(loaded.value.clone())
    }

    /// Makes room without evicting an in-flight load and breaking single-flight.
    fn reserve_entry(&self, state: &mut CacheState<K, V>) -> bool {
        while state.entries.len() >= self.entry_capacity {
            if !evict_oldest_loaded(state) {
                return false;
            }
        }
        true
    }

    fn evict_to_byte_capacity(&self, state: &mut CacheState<K, V>) {
        while state.resident_bytes > self.byte_capacity && evict_oldest_loaded(state) {}
    }

    fn remove_cell(&self, key: &K, cell: &CacheCell<V>) {
        remove_matching_cell(&mut self.state(), key, cell);
    }

    fn state(&self) -> std::sync::MutexGuard<'_, CacheState<K, V>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, usize) {
        let state = self.state();
        (state.entries.len(), state.resident_bytes)
    }
}

fn evict_oldest_loaded<K, V>(state: &mut CacheState<K, V>) -> bool
where
    K: Eq + Hash,
{
    let entries = state.entries.len();
    for _ in 0..entries {
        let Some((key, entry)) = state.entries.pop_front() else {
            return false;
        };
        let Some(charge) = entry.charge else {
            state.entries.insert(key, entry);
            continue;
        };
        state.resident_bytes = state.resident_bytes.saturating_sub(charge);
        return true;
    }
    false
}

fn remove_matching_cell<K, V>(state: &mut CacheState<K, V>, key: &K, cell: &CacheCell<V>)
where
    K: Eq + Hash,
{
    let matches = state
        .entries
        .get(key)
        .is_some_and(|entry| Arc::ptr_eq(&entry.cell, cell));
    if !matches {
        return;
    }
    let Some(entry) = state.entries.remove(key) else {
        return;
    };
    if let Some(charge) = entry.charge {
        state.resident_bytes = state.resident_bytes.saturating_sub(charge);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::Notify;

    use super::*;
    use crate::Error;

    #[tokio::test]
    async fn concurrent_misses_load_one_value() {
        let cache = BoundedAsyncCache::new(2, 1024);
        let loads = Arc::new(AtomicUsize::new(0));
        let first_loads = Arc::clone(&loads);
        let second_loads = Arc::clone(&loads);

        let (first, second) = tokio::join!(
            cache.get_or_try_insert("value", || async move {
                first_loads.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                Ok((Arc::new(1), 64))
            }),
            cache.get_or_try_insert("value", || async move {
                second_loads.fetch_add(1, Ordering::Relaxed);
                Ok((Arc::new(2), 64))
            })
        );

        assert!(Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(cache.stats(), (1, 64));
    }

    #[tokio::test]
    async fn cache_is_bounded_by_entries_and_bytes() {
        let cache = BoundedAsyncCache::new(2, 100);
        for (key, charge) in [("a", 40), ("b", 40), ("c", 70)] {
            cache
                .get_or_try_insert(key, || async move { Ok((key, charge)) })
                .await
                .unwrap();
        }

        assert_eq!(cache.stats(), (1, 70));
    }

    #[tokio::test]
    async fn full_pending_cache_bypasses_new_keys_without_growing() {
        let cache = Arc::new(BoundedAsyncCache::new(1, 100));
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let first_cache = Arc::clone(&cache);
        let first_started = Arc::clone(&started);
        let first_release = Arc::clone(&release);
        let first = tokio::spawn(async move {
            first_cache
                .get_or_try_insert("pending", || async move {
                    first_started.notify_one();
                    first_release.notified().await;
                    Ok((1, 10))
                })
                .await
        });
        started.notified().await;

        let bypassed = cache
            .get_or_try_insert("bypassed", || async { Ok((2, 10)) })
            .await
            .unwrap();

        assert_eq!(bypassed, 2);
        assert_eq!(cache.stats(), (1, 0));
        release.notify_one();
        assert_eq!(first.await.unwrap().unwrap(), 1);
        assert_eq!(cache.stats(), (1, 10));
    }

    #[tokio::test]
    async fn oversized_value_does_not_evict_resident_value() {
        let cache = BoundedAsyncCache::new(2, 64);
        cache
            .get_or_try_insert("resident", || async { Ok((1, 32)) })
            .await
            .unwrap();
        cache
            .get_or_try_insert("oversized", || async { Ok((2, 65)) })
            .await
            .unwrap();

        assert_eq!(cache.stats(), (1, 32));
    }

    #[tokio::test]
    async fn failed_load_is_removed_and_retried() {
        let cache = BoundedAsyncCache::new(2, 64);
        let attempts = AtomicUsize::new(0);
        let error = cache
            .get_or_try_insert("value", || async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Err(Error::InvalidArgument("failed".into()))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)));
        assert_eq!(cache.stats(), (0, 0));

        cache
            .get_or_try_insert("value", || async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Ok((1, 32))
            })
            .await
            .unwrap();
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats(), (1, 32));
    }

    #[tokio::test]
    async fn zero_capacity_bypasses_cache() {
        let cache = BoundedAsyncCache::new(0, 64);
        let loads = AtomicUsize::new(0);
        for _ in 0..2 {
            cache
                .get_or_try_insert("value", || async {
                    loads.fetch_add(1, Ordering::Relaxed);
                    Ok((1, 32))
                })
                .await
                .unwrap();
        }

        assert_eq!(loads.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats(), (0, 0));
    }
}
