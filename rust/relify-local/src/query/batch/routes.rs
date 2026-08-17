use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::{Error, Result};

/// Immutable routing state shared by all partitions of one batch search.
#[derive(Debug, Clone)]
pub(crate) struct BatchRoutes {
    dimension: usize,
    queries: Arc<[f32]>,
    query_ordinals_by_cid: Arc<BTreeMap<i32, Vec<usize>>>,
}

impl BatchRoutes {
    pub(crate) fn try_new(queries: Vec<Vec<f32>>, cids_by_query: Vec<Vec<i32>>) -> Result<Self> {
        if queries.is_empty() {
            return Err(Error::InvalidArgument(
                "batch search requires at least one query vector".into(),
            ));
        }
        if queries.len() != cids_by_query.len() {
            return Err(Error::InvalidArgument(
                "batch query and cluster-route counts differ".into(),
            ));
        }
        let dimension = queries[0].len();
        if dimension == 0 {
            return Err(Error::InvalidArgument(
                "query vector dimension must be positive".into(),
            ));
        }

        let mut flattened = Vec::with_capacity(queries.len().saturating_mul(dimension));
        let mut query_ordinals_by_cid = BTreeMap::<i32, Vec<usize>>::new();
        for (query_ordinal, (query, cids)) in queries.into_iter().zip(cids_by_query).enumerate() {
            if query.len() != dimension {
                return Err(Error::InvalidArgument(format!(
                    "query vector {query_ordinal} has dimension {}, expected {dimension}",
                    query.len()
                )));
            }
            if query.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidArgument(format!(
                    "query vector {query_ordinal} contains a non-finite value"
                )));
            }
            if cids.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "query vector {query_ordinal} has no selected IVF clusters"
                )));
            }
            let mut seen = HashSet::with_capacity(cids.len());
            for cid in cids {
                if cid < 0 {
                    return Err(Error::InvalidArgument(
                        "selected IVF cluster IDs must be non-negative".into(),
                    ));
                }
                if !seen.insert(cid) {
                    return Err(Error::InvalidArgument(format!(
                        "query vector {query_ordinal} selects IVF cluster {cid} more than once"
                    )));
                }
                query_ordinals_by_cid
                    .entry(cid)
                    .or_default()
                    .push(query_ordinal);
            }
            flattened.extend(query);
        }

        Ok(Self {
            dimension,
            queries: flattened.into(),
            query_ordinals_by_cid: Arc::new(query_ordinals_by_cid),
        })
    }

    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn query_count(&self) -> usize {
        self.queries.len() / self.dimension
    }

    pub(crate) fn query(&self, ordinal: usize) -> Option<&[f32]> {
        let start = ordinal.checked_mul(self.dimension)?;
        let end = start.checked_add(self.dimension)?;
        self.queries.get(start..end)
    }

    pub(crate) fn query_ordinals(&self, cid: i32) -> &[usize] {
        self.query_ordinals_by_cid
            .get(&cid)
            .map_or(&[], Vec::as_slice)
    }

    pub(crate) fn distinct_cids(&self) -> impl Iterator<Item = i32> + '_ {
        self.query_ordinals_by_cid.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_queries_once_and_inverts_cluster_routes() {
        let routes = BatchRoutes::try_new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![7, 3], vec![7, 9]],
        )
        .unwrap();

        assert_eq!(routes.dimension(), 2);
        assert_eq!(routes.query_count(), 2);
        assert_eq!(routes.query(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(routes.query(1), Some([3.0, 4.0].as_slice()));
        assert_eq!(routes.query_ordinals(3), &[0]);
        assert_eq!(routes.query_ordinals(7), &[0, 1]);
        assert_eq!(routes.query_ordinals(9), &[1]);
        assert_eq!(routes.distinct_cids().collect::<Vec<_>>(), [3, 7, 9]);
    }

    #[test]
    fn rejects_ambiguous_or_invalid_routes() {
        assert!(BatchRoutes::try_new(vec![], vec![]).is_err());
        assert!(BatchRoutes::try_new(vec![vec![1.0]], vec![]).is_err());
        assert!(BatchRoutes::try_new(vec![vec![]], vec![vec![0]]).is_err());
        assert!(
            BatchRoutes::try_new(vec![vec![1.0], vec![1.0, 2.0]], vec![vec![0], vec![1]]).is_err()
        );
        assert!(BatchRoutes::try_new(vec![vec![f32::NAN]], vec![vec![0]]).is_err());
        assert!(BatchRoutes::try_new(vec![vec![1.0]], vec![vec![]]).is_err());
        assert!(BatchRoutes::try_new(vec![vec![1.0]], vec![vec![-1]]).is_err());
        assert!(BatchRoutes::try_new(vec![vec![1.0]], vec![vec![1, 1]]).is_err());
    }
}
