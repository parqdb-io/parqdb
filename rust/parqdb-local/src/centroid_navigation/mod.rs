//! Process-local IVF centroid routing.

mod graph;

use std::sync::{Arc, Mutex};

#[cfg(test)]
use parallite::Executor;
use parallite::ParalliteContext;
use parqdb_kernels::{LvqBits, LvqEncodedBatch, detect, encode_lvq_rows};

use graph::{Graph, SearchScratch};

use crate::{Error, Result};

const DEFAULT_M: usize = 16;
const DEFAULT_EF_CONSTRUCTION: usize = 128;
const DEFAULT_EF_SEARCH: usize = 64;
const DEFAULT_SEED: u64 = 0x7265_6c69_6679;
const MIN_HNSW_CENTROIDS: usize = 8_192;
const EXACT_SCAN_NPROBE_DIVISOR: usize = 64;

#[derive(Debug, Clone, Copy)]
struct HnswConfig {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: DEFAULT_M,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            seed: DEFAULT_SEED,
        }
    }
}

#[derive(Debug)]
struct RouteScratch {
    graph: SearchScratch,
}

#[derive(Debug)]
struct HnswIndex {
    centroids: LvqEncodedBatch,
    graph: Graph,
    ef_search: usize,
}

#[derive(Debug)]
struct HnswNavigator {
    index: Arc<HnswIndex>,
    scratches: Mutex<Vec<RouteScratch>>,
    scratch_capacity: usize,
    scratch_resident_limit: usize,
    pooled_result_capacity: usize,
    pooled_candidate_capacity: usize,
}

#[derive(Debug)]
pub(crate) struct CentroidNavigator {
    nlist: usize,
    dimension: usize,
    strategy: NavigatorStrategy,
}

#[derive(Debug)]
enum NavigatorStrategy {
    Exact(LvqEncodedBatch),
    Hnsw(HnswNavigator),
}

impl CentroidNavigator {
    pub(crate) fn new(nlist: usize, dimension: usize, values: &[f32]) -> Result<Self> {
        let parallel = ParalliteContext::default();
        Self::new_parallel(nlist, dimension, values, &parallel)
    }

    pub(crate) fn new_parallel(
        nlist: usize,
        dimension: usize,
        values: &[f32],
        parallel: &ParalliteContext,
    ) -> Result<Self> {
        Self::new_with_policy(nlist, dimension, values, MIN_HNSW_CENTROIDS, parallel)
    }

    pub(crate) fn name(&self) -> &'static str {
        match &self.strategy {
            NavigatorStrategy::Exact(_) => "exact",
            NavigatorStrategy::Hnsw(_) => "hnsw_lvq8",
        }
    }

    fn new_with_policy(
        nlist: usize,
        dimension: usize,
        values: &[f32],
        min_hnsw_centroids: usize,
        parallel: &ParalliteContext,
    ) -> Result<Self> {
        let expected = nlist.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidSchema(format!(
                "invalid centroid matrix shape: {nlist} x {dimension} overflows usize"
            ))
        })?;
        if nlist == 0 || dimension == 0 || expected != values.len() {
            return Err(Error::InvalidSchema(format!(
                "invalid centroid matrix shape: {nlist} x {dimension} requires {expected} values, found {}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidSchema(
                "centroid matrix must contain only finite values".into(),
            ));
        }
        let centroids = encode_lvq_rows(values, dimension, LvqBits::Eight)?;
        let strategy = if nlist >= min_hnsw_centroids {
            NavigatorStrategy::Hnsw(HnswNavigator::build_encoded_parallel(
                centroids,
                HnswConfig::default(),
                parallel,
            )?)
        } else {
            NavigatorStrategy::Exact(centroids)
        };
        Ok(Self {
            nlist,
            dimension,
            strategy,
        })
    }

    pub(crate) fn validate_shape(&self, nlist: usize, dimension: usize) -> Result<()> {
        if self.nlist != nlist || self.dimension != dimension {
            return Err(Error::InvalidSchema(format!(
                "cached centroid matrix is {} x {}, requested {nlist} x {dimension}",
                self.nlist, self.dimension
            )));
        }
        Ok(())
    }

    pub(crate) fn route(&self, query: &[f32], nprobe: usize) -> Result<Vec<usize>> {
        self.validate_query(query, nprobe)?;
        if nprobe == self.nlist {
            return Ok((0..self.nlist).collect());
        }
        match &self.strategy {
            NavigatorStrategy::Hnsw(navigator) => navigator.route(query, nprobe),
            NavigatorStrategy::Exact(centroids) => route_exact(centroids, query, nprobe),
        }
    }

    pub(crate) fn route_batch(&self, queries: &[f32], nprobe: usize) -> Result<Vec<usize>> {
        if nprobe == 0 || nprobe > self.nlist {
            return Err(Error::InvalidArgument(format!(
                "nprobe must be between 1 and {}",
                self.nlist
            )));
        }
        if !queries.len().is_multiple_of(self.dimension) {
            return Err(Error::InvalidArgument(
                "query matrix does not match the centroid dimension".into(),
            ));
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "query matrix must contain only finite values".into(),
            ));
        }
        if nprobe == self.nlist {
            let rows = queries.len() / self.dimension;
            return Ok((0..rows).flat_map(|_| 0..self.nlist).collect());
        }
        match &self.strategy {
            NavigatorStrategy::Hnsw(navigator) => navigator.route_batch(queries, nprobe),
            NavigatorStrategy::Exact(_) => {
                let mut output =
                    Vec::with_capacity(queries.len() / self.dimension * nprobe.min(self.nlist));
                for query in queries.chunks_exact(self.dimension) {
                    output.extend(self.route(query, nprobe)?);
                }
                Ok(output)
            }
        }
    }

    pub(crate) fn resident_size(&self) -> usize {
        match &self.strategy {
            NavigatorStrategy::Exact(centroids) => centroids.resident_size(),
            NavigatorStrategy::Hnsw(navigator) => navigator.resident_size(),
        }
    }

    fn validate_query(&self, query: &[f32], nprobe: usize) -> Result<()> {
        if nprobe == 0 || nprobe > self.nlist {
            return Err(Error::InvalidArgument(format!(
                "nprobe must be between 1 and {}",
                self.nlist
            )));
        }
        if query.len() != self.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "query must match the centroid dimension and contain finite values".into(),
            ));
        }
        Ok(())
    }
}

impl HnswIndex {
    #[cfg(test)]
    pub(crate) fn build(values: &[f32], dimension: usize, config: HnswConfig) -> Result<Self> {
        let parallel = ParalliteContext::with_executor(Executor::serial());
        Self::build_parallel(values, dimension, config, &parallel)
    }

    #[cfg(test)]
    fn build_parallel(
        values: &[f32],
        dimension: usize,
        config: HnswConfig,
        parallel: &ParalliteContext,
    ) -> Result<Self> {
        if dimension == 0 || values.is_empty() || !values.len().is_multiple_of(dimension) {
            return Err(Error::InvalidArgument(
                "centroid matrix must contain complete non-empty rows".into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "centroid matrix must contain only finite values".into(),
            ));
        }
        let centroids = encode_lvq_rows(values, dimension, LvqBits::Eight)?;
        Self::build_encoded_parallel(centroids, config, parallel)
    }

    fn build_encoded_parallel(
        centroids: LvqEncodedBatch,
        config: HnswConfig,
        parallel: &ParalliteContext,
    ) -> Result<Self> {
        if config.m < 2 {
            return Err(Error::InvalidArgument("HNSW m must be at least 2".into()));
        }
        if config.ef_construction < config.m {
            return Err(Error::InvalidArgument(
                "HNSW ef_construction must be at least m".into(),
            ));
        }
        if config.ef_search == 0 {
            return Err(Error::InvalidArgument(
                "HNSW ef_search must be positive".into(),
            ));
        }
        let dimension = centroids.dimension();
        let values = decode_centroids(&centroids)?;
        let graph = Graph::build(
            &values,
            dimension,
            config.m,
            config.ef_construction,
            config.seed,
            parallel,
        )?;
        Ok(Self {
            centroids,
            graph,
            ef_search: config.ef_search,
        })
    }

    fn centroid_count(&self) -> usize {
        self.centroids.row_count()
    }

    fn resident_size(&self) -> usize {
        self.centroids.resident_size() + self.graph.resident_size()
    }

    fn scratch(&self) -> RouteScratch {
        self.scratch_with_capacity(self.ef_search, self.ef_search.saturating_mul(2))
    }

    fn scratch_with_capacity(
        &self,
        result_capacity: usize,
        candidate_capacity: usize,
    ) -> RouteScratch {
        RouteScratch {
            graph: SearchScratch::new_for_query(
                self.centroid_count(),
                result_capacity,
                candidate_capacity,
            ),
        }
    }

    fn route(
        &self,
        query: &[f32],
        nprobe: usize,
        scratch: &mut RouteScratch,
    ) -> Result<Vec<usize>> {
        self.route_with_ef(
            query,
            nprobe,
            self.ef_search.max(nprobe.saturating_mul(2)),
            scratch,
        )
    }

    fn route_with_ef(
        &self,
        query: &[f32],
        nprobe: usize,
        ef: usize,
        scratch: &mut RouteScratch,
    ) -> Result<Vec<usize>> {
        if nprobe == 0 {
            return Err(Error::InvalidArgument("nprobe must be positive".into()));
        }
        if query.len() != self.centroids.dimension() || query.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InvalidArgument(
                "query must match the centroid dimension and contain finite values".into(),
            ));
        }
        let count = nprobe.min(self.centroid_count());
        if exact_scan_preferred(count, self.centroid_count()) {
            return self.route_exact(query, count);
        }
        let mut output = Vec::with_capacity(nprobe.min(self.centroid_count()));
        self.graph.search_append(
            &self.centroids,
            query,
            count,
            ef.max(nprobe),
            &mut scratch.graph,
            &mut output,
        )?;
        Ok(output)
    }

    fn route_batch(
        &self,
        queries: &[f32],
        nprobe: usize,
        scratch: &mut RouteScratch,
    ) -> Result<Vec<usize>> {
        if nprobe == 0 {
            return Err(Error::InvalidArgument("nprobe must be positive".into()));
        }
        if !queries.len().is_multiple_of(self.centroids.dimension()) {
            return Err(Error::InvalidArgument(
                "query matrix does not match the centroid dimension".into(),
            ));
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "query matrix must contain only finite values".into(),
            ));
        }
        let rows = queries.len() / self.centroids.dimension();
        let count = nprobe.min(self.centroid_count());
        let mut output = Vec::with_capacity(rows * count);
        if exact_scan_preferred(count, self.centroid_count()) {
            let mut distances = vec![0.0_f32; self.centroid_count()];
            let mut scored = Vec::with_capacity(self.centroid_count());
            for query in queries.chunks_exact(self.centroids.dimension()) {
                self.route_exact_append(query, count, &mut distances, &mut scored, &mut output)?;
            }
            return Ok(output);
        }
        for query in queries.chunks_exact(self.centroids.dimension()) {
            self.graph.search_append(
                &self.centroids,
                query,
                count,
                self.ef_search.max(nprobe.saturating_mul(2)),
                &mut scratch.graph,
                &mut output,
            )?;
        }
        Ok(output)
    }

    fn route_exact(&self, query: &[f32], count: usize) -> Result<Vec<usize>> {
        let mut distances = vec![0.0_f32; self.centroid_count()];
        let mut scored = Vec::with_capacity(self.centroid_count());
        let mut output = Vec::with_capacity(count);
        self.route_exact_append(query, count, &mut distances, &mut scored, &mut output)?;
        Ok(output)
    }

    fn route_exact_append(
        &self,
        query: &[f32],
        count: usize,
        distances: &mut [f32],
        scored: &mut Vec<(f32, usize)>,
        output: &mut Vec<usize>,
    ) -> Result<()> {
        if count == self.centroid_count() {
            output.extend(0..count);
            return Ok(());
        }
        detect().lvq_squared_l2_rows(&self.centroids.as_view(), query, distances)?;
        scored.clear();
        scored.extend(
            distances
                .iter()
                .copied()
                .enumerate()
                .map(|(cid, distance)| (distance, cid)),
        );
        let compare = |left: &(f32, usize), right: &(f32, usize)| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        };
        scored.select_nth_unstable_by(count, compare);
        output.extend(scored[..count].iter().map(|(_, cid)| *cid));
        Ok(())
    }
}

fn exact_scan_preferred(nprobe: usize, nlist: usize) -> bool {
    nprobe.saturating_mul(EXACT_SCAN_NPROBE_DIVISOR) >= nlist
}

fn decode_centroids(centroids: &LvqEncodedBatch) -> Result<Vec<f32>> {
    let mut values = vec![0.0; centroids.row_count() * centroids.dimension()];
    let view = centroids.as_view();
    for (row, output) in values.chunks_exact_mut(centroids.dimension()).enumerate() {
        view.decode_row(row, output)?;
    }
    Ok(values)
}

fn route_exact(centroids: &LvqEncodedBatch, query: &[f32], nprobe: usize) -> Result<Vec<usize>> {
    let mut distances = vec![0.0_f32; centroids.row_count()];
    detect().lvq_squared_l2_rows(&centroids.as_view(), query, &mut distances)?;
    let mut scored = distances
        .into_iter()
        .enumerate()
        .map(|(cid, distance)| (distance, cid))
        .collect::<Vec<_>>();
    let compare = |left: &(f32, usize), right: &(f32, usize)| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    if nprobe < scored.len() {
        scored.select_nth_unstable_by(nprobe, compare);
        scored.truncate(nprobe);
    }
    scored.sort_unstable_by(compare);
    Ok(scored.into_iter().map(|(_, cid)| cid).collect())
}

impl HnswNavigator {
    #[cfg(test)]
    fn build_parallel(
        values: &[f32],
        dimension: usize,
        config: HnswConfig,
        parallel: &ParalliteContext,
    ) -> Result<Self> {
        let index = Arc::new(HnswIndex::build_parallel(
            values, dimension, config, parallel,
        )?);
        Ok(Self::from_index(index, config, parallel))
    }

    fn from_index(index: Arc<HnswIndex>, config: HnswConfig, parallel: &ParalliteContext) -> Self {
        let scratch_capacity = parallel.thread_count().max(1);
        let pooled_result_capacity = config.ef_search;
        let pooled_candidate_capacity = config.ef_search.saturating_mul(2);
        let scratches = (0..scratch_capacity)
            .map(|_| index.scratch_with_capacity(pooled_result_capacity, pooled_candidate_capacity))
            .collect::<Vec<_>>();
        let scratch_resident_limit = scratches
            .first()
            .map_or(0, |scratch| scratch.graph.resident_size());
        Self {
            index,
            scratches: Mutex::new(scratches),
            scratch_capacity,
            scratch_resident_limit,
            pooled_result_capacity,
            pooled_candidate_capacity,
        }
    }

    fn build_encoded_parallel(
        centroids: LvqEncodedBatch,
        config: HnswConfig,
        parallel: &ParalliteContext,
    ) -> Result<Self> {
        let index = Arc::new(HnswIndex::build_encoded_parallel(
            centroids, config, parallel,
        )?);
        Ok(Self::from_index(index, config, parallel))
    }

    fn route(&self, query: &[f32], nprobe: usize) -> Result<Vec<usize>> {
        let mut scratch = self.take_scratch();
        let result = self.index.route(query, nprobe, &mut scratch);
        self.return_scratch(scratch);
        result
    }

    fn route_batch(&self, queries: &[f32], nprobe: usize) -> Result<Vec<usize>> {
        let mut scratch = self.take_scratch();
        let result = self.index.route_batch(queries, nprobe, &mut scratch);
        self.return_scratch(scratch);
        result
    }

    fn resident_size(&self) -> usize {
        let scratches = mutex_lock(&self.scratches);
        self.index.resident_size()
            + scratches.capacity() * std::mem::size_of::<RouteScratch>()
            + scratches
                .iter()
                .map(|scratch| scratch.graph.resident_size())
                .sum::<usize>()
    }

    #[cfg(test)]
    fn uses_lvq8_centroids(&self) -> bool {
        self.index.centroids.bits() == LvqBits::Eight
    }

    fn take_scratch(&self) -> RouteScratch {
        mutex_lock(&self.scratches)
            .pop()
            .unwrap_or_else(|| self.index.scratch())
    }

    fn return_scratch(&self, scratch: RouteScratch) {
        let mut scratch = scratch;
        scratch
            .graph
            .trim_for_query_pool(self.pooled_result_capacity, self.pooled_candidate_capacity);
        if scratch.graph.resident_size() > self.scratch_resident_limit {
            return;
        }
        let mut scratches = mutex_lock(&self.scratches);
        if scratches.len() < self.scratch_capacity {
            scratches.push(scratch);
        }
    }
}

fn mutex_lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn points() -> Vec<f32> {
        (0_u16..256)
            .flat_map(|value| [f32::from(value), f32::from(value % 17)])
            .collect()
    }

    #[test]
    fn generic_navigator_validates_shape() {
        let error = CentroidNavigator::new(2, 3, &[0.0; 5]).unwrap_err();

        assert!(error.to_string().contains("2 x 3"));
        assert!(error.to_string().contains("6 values, found 5"));
        assert!(CentroidNavigator::new(0, 3, &[]).is_err());
    }

    #[test]
    fn selecting_every_cluster_bypasses_the_strategy() {
        let navigator = CentroidNavigator::new(2, 2, &[0.0, 0.0, 1.0, 1.0]).unwrap();

        assert_eq!(navigator.route(&[0.5, 0.5], 2).unwrap(), [0, 1]);
        assert_eq!(
            navigator.route_batch(&[0.5, 0.5, 0.25, 0.25], 2).unwrap(),
            [0, 1, 0, 1]
        );
    }

    #[test]
    fn generic_navigator_selects_one_exclusive_strategy() {
        let parallel = ParalliteContext::with_executor(Executor::serial());
        let values = (0_u16..256 * 8)
            .map(|value| f32::from(value % 251))
            .collect::<Vec<_>>();
        let navigator = CentroidNavigator::new_with_policy(256, 8, &values, 1, &parallel).unwrap();

        let NavigatorStrategy::Hnsw(hnsw) = &navigator.strategy else {
            panic!("expected HNSW navigation");
        };
        assert!(hnsw.uses_lvq8_centroids());
        assert_eq!(navigator.route(&[0.0; 8], 1).unwrap().len(), 1);
    }

    #[test]
    fn routes_to_the_nearest_centroid() {
        let values = points();
        let navigator = HnswIndex::build(
            &values,
            2,
            HnswConfig {
                ef_search: 128,
                ..HnswConfig::default()
            },
        )
        .unwrap();
        let mut scratch = navigator.scratch();

        assert_eq!(
            navigator.route(&[73.1, 5.0], 1, &mut scratch).unwrap(),
            [73]
        );
        assert_eq!(
            navigator
                .route_batch(&[73.1, 5.0, 201.0, 14.0], 1, &mut scratch)
                .unwrap(),
            [73, 201]
        );
    }

    #[test]
    fn large_nprobe_uses_exact_lvq_routing() {
        let values = points();
        let navigator = HnswIndex::build(&values, 2, HnswConfig::default()).unwrap();
        let mut scratch = navigator.scratch();
        let mut actual = navigator.route(&[73.1, 5.0], 4, &mut scratch).unwrap();
        let mut expected = navigator.route_exact(&[73.1, 5.0], 4).unwrap();
        actual.sort_unstable();
        expected.sort_unstable();

        assert_eq!(actual, expected);
        assert!(!exact_scan_preferred(3, 256));
        assert!(exact_scan_preferred(4, 256));
        assert_eq!(
            navigator
                .route_batch(&[73.1, 5.0, 201.0, 14.0], 4, &mut scratch)
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn validates_navigation_options_and_queries() {
        let values = points();
        assert!(
            HnswIndex::build(
                &values,
                2,
                HnswConfig {
                    m: 1,
                    ..HnswConfig::default()
                },
            )
            .is_err()
        );
        let navigator = HnswIndex::build(&values, 2, HnswConfig::default()).unwrap();
        let mut scratch = navigator.scratch();
        assert!(navigator.route(&[1.0], 1, &mut scratch).is_err());
        assert!(navigator.route(&[1.0, 1.0], 0, &mut scratch).is_err());
    }

    #[test]
    fn concurrent_queries_use_independent_scratch() {
        let values = points();
        let parallel = ParalliteContext::with_executor(Executor::with_threads(4).unwrap());
        let navigator = Arc::new(
            HnswNavigator::build_parallel(&values, 2, HnswConfig::default(), &parallel).unwrap(),
        );
        let workers = (0..8)
            .map(|worker| {
                let navigator = Arc::clone(&navigator);
                std::thread::spawn(move || {
                    let cid = 32 + worker;
                    let value = f32::from(u16::try_from(cid).unwrap());
                    assert_eq!(
                        navigator
                            .route(&[value, f32::from(u16::try_from(cid % 17).unwrap())], 1)
                            .unwrap(),
                        [cid]
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn pooled_scratches_are_accounted_and_bounded() {
        let values = points();
        let parallel = ParalliteContext::with_executor(Executor::with_threads(4).unwrap());
        let navigator =
            HnswNavigator::build_parallel(&values, 2, HnswConfig::default(), &parallel).unwrap();
        let index_size = navigator.index.resident_size();
        let initial_size = navigator.resident_size();

        assert!(initial_size > index_size);
        assert_eq!(mutex_lock(&navigator.scratches).len(), 4);
        navigator.route(&[73.1, 5.0], 200).unwrap();
        assert!(navigator.resident_size() <= initial_size);
    }
}
