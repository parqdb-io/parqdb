use std::future::Future;
use std::mem::size_of_val;
use std::sync::Arc;

use super::bounded_cache::BoundedAsyncCache;
use crate::{Error, Result};

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
        let expected = nlist.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidSchema(format!(
                "invalid cached centroid matrix shape: {nlist} x {dimension} overflows usize"
            ))
        })?;
        if expected != values.len() {
            return Err(Error::InvalidSchema(format!(
                "invalid cached centroid matrix shape: {nlist} x {dimension} requires \
                 {expected} values, found {}",
                values.len()
            )));
        }
        Ok(Self {
            nlist,
            dimension,
            values: Arc::from(values),
        })
    }

    pub(in crate::session) fn values(&self, nlist: usize, dimension: usize) -> Result<&[f32]> {
        if self.nlist != nlist || self.dimension != dimension {
            return Err(Error::InvalidSchema(format!(
                "cached centroid matrix is {} x {}, requested {nlist} x {dimension}",
                self.nlist, self.dimension
            )));
        }
        Ok(self.values.as_ref())
    }

    fn charge(&self) -> usize {
        size_of_val(self.values.as_ref())
    }
}

pub(super) struct CentroidCache {
    cache: BoundedAsyncCache<String, Arc<CentroidMatrix>>,
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
    ) -> Result<Arc<CentroidMatrix>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CentroidMatrix>>,
    {
        self.cache
            .get_or_try_insert(relation_key.to_owned(), || async {
                let matrix = Arc::new(load().await?);
                let charge = matrix.charge();
                Ok((matrix, charge))
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

    #[test]
    fn invalid_shape_reports_expected_and_actual_values() {
        let error = CentroidMatrix::new(2, 3, vec![0.0; 5]).unwrap_err();

        assert!(error.to_string().contains("2 x 3"));
        assert!(error.to_string().contains("6 values, found 5"));
    }

    #[tokio::test]
    async fn cached_matrix_validates_requested_shape() {
        let cache = CentroidCache::new(2, 1024);
        let matrix = cache
            .get_or_load("centroids", || async {
                CentroidMatrix::new(1, 2, vec![0.0, 1.0])
            })
            .await
            .unwrap();

        assert_eq!(matrix.values(1, 2).unwrap(), [0.0, 1.0]);
        assert!(matrix.values(2, 1).is_err());
        assert_eq!(cache.stats(), (1, 2 * size_of::<f32>()));
    }
}
