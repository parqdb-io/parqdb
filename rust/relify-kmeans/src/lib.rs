#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Parallel Lloyd K-means extracted from `RoverQ` native training.
//!
//! Relify uses only the coarse-centroid path. Product-quantizer and
//! `RoverQ`-specific training remain outside this crate.

use std::collections::HashMap;
use std::sync::Mutex;

use parallite::{CollectPartitions, DatasetExt, ParalliteContext, ParalliteError};
use relify_kernels::{KernelError, detect, row_dot_products};
use thiserror::Error;

const PARTITION_ROWS: usize = 1024;
const EMPTY_SPLIT_EPS: f32 = 1.0 / 1024.0;
const GEMM_MIN_DIMENSION: usize = 8;
const GEMM_MIN_CLUSTERS: usize = 16;
const GEMM_MAX_OUTPUT_VALUES: usize = 4 * 1024 * 1024;
const HIERARCHICAL_AUTO_MIN_CLUSTERS: usize = 8_192;
const ROOT_TRAINING_POINTS_PER_CENTROID: usize = 256;

/// Error returned by K-means training, sampling, or assignment.
#[derive(Debug, Error)]
pub enum Error {
    /// An input matrix or training option is invalid.
    #[error("invalid K-means argument: {0}")]
    InvalidArgument(String),
    /// A shared numerical kernel rejected its matrix buffers.
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

/// Result returned by K-means operations.
pub type Result<T> = std::result::Result<T, Error>;

impl From<ParalliteError<Error>> for Error {
    fn from(error: ParalliteError<Error>) -> Self {
        match error {
            ParalliteError::User(error) => error,
            other => Self::InvalidArgument(other.to_string()),
        }
    }
}

/// Configuration for K-means training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansOptions {
    /// Number of centroids to train.
    pub n_clusters: usize,
    /// Maximum Lloyd iterations.
    pub max_iter: usize,
    /// Deterministic initialization seed.
    pub seed: u64,
    /// Training strategy used to produce the leaf centroids.
    pub mode: KMeansMode,
}

/// Strategy for training a fixed number of K-means centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansMode {
    /// Use hierarchical training for large centroid counts and flat Lloyd K-means otherwise.
    ///
    /// If a hierarchy cannot form every requested leaf centroid from its root
    /// partitions, training falls back to flat Lloyd K-means.
    Auto,
    /// Always run flat Lloyd K-means over every requested centroid.
    Flat,
    /// Train a balanced two-level hierarchy and return its flattened leaf centroids.
    ///
    /// Returns an error when a root partition has too few rows to form its
    /// assigned leaf centroids.
    Hierarchical,
}

impl KMeansOptions {
    /// Returns options with deterministic Relify defaults.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 20,
            seed: 42,
            mode: KMeansMode::Auto,
        }
    }
}

/// Trained dense `f32` centroids in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansModel {
    /// Contiguous centroid values.
    pub centroids: Vec<f32>,
    /// Maximum Lloyd iterations completed by one training stage.
    pub iterations: usize,
}

/// Deterministic bounded reservoir used to collect streaming training rows.
pub struct ReservoirSampler {
    dimension: Option<usize>,
    max_rows: usize,
    seen_rows: usize,
    values: Vec<f32>,
    rng: SmallRng,
}

impl ReservoirSampler {
    /// Creates a sampler retaining at most `max_rows` dense vectors.
    pub fn new(max_rows: usize, seed: u64) -> Result<Self> {
        if max_rows == 0 {
            return Err(Error::InvalidArgument(
                "reservoir row limit must be positive".into(),
            ));
        }
        Ok(Self {
            dimension: None,
            max_rows,
            seen_rows: 0,
            values: Vec::new(),
            rng: SmallRng::new(seed),
        })
    }

    /// Adds one row-major dense vector batch to the reservoir.
    pub fn push(&mut self, vectors: &[f32], dimension: usize) -> Result<()> {
        if dimension == 0 || !vectors.len().is_multiple_of(dimension) {
            return Err(Error::InvalidArgument(
                "reservoir vectors must be a row-aligned dense matrix".into(),
            ));
        }
        if self.dimension.is_some_and(|expected| expected != dimension) {
            return Err(Error::InvalidArgument(
                "vector dimension changes between training batches".into(),
            ));
        }
        if self.dimension.is_none() {
            let values = self.max_rows.checked_mul(dimension).ok_or_else(|| {
                Error::InvalidArgument("reservoir capacity overflows usize".into())
            })?;
            self.values.try_reserve_exact(values).map_err(|error| {
                Error::InvalidArgument(format!("cannot allocate training reservoir: {error}"))
            })?;
            self.dimension = Some(dimension);
        }
        for vector in vectors.chunks_exact(dimension) {
            self.seen_rows = self
                .seen_rows
                .checked_add(1)
                .ok_or_else(|| Error::InvalidArgument("reservoir row count overflows".into()))?;
            if self.values.len() / dimension < self.max_rows {
                self.values.extend_from_slice(vector);
                continue;
            }
            let replacement = self.rng.gen_usize(self.seen_rows);
            if replacement < self.max_rows {
                let start = replacement * dimension;
                self.values[start..start + dimension].copy_from_slice(vector);
            }
        }
        Ok(())
    }

    /// Returns the observed vector dimension after the first non-empty batch.
    #[must_use]
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Returns the number of rows observed across all batches.
    #[must_use]
    pub fn seen_rows(&self) -> usize {
        self.seen_rows
    }

    /// Returns the retained row-major training matrix.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Debug, Clone)]
struct AssignmentBatch {
    cells: Vec<usize>,
    inertia: f64,
}

#[derive(Debug, Clone)]
struct BatchSums {
    sums: Vec<f32>,
    counts: Vec<usize>,
    inertia: f64,
}

#[derive(Debug, Clone, Copy)]
struct RowRange {
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct TrainingPartition {
    range: RowRange,
    assignment_cells: Vec<usize>,
    assignment_distances: Vec<f32>,
    inertia: f64,
}

#[derive(Debug)]
struct TrainingWorkspace {
    partitions: Vec<TrainingPartition>,
    merged: BatchSums,
    dot_scratch: DotScratchPool,
}

impl TrainingWorkspace {
    fn new(rows: usize, n_clusters: usize, dimension: usize, workers: usize) -> Self {
        let partition_count = training_partition_count(rows, workers);
        let gemm = uses_gemm(dimension, n_clusters);
        let partitions = row_ranges(rows, rows.div_ceil(partition_count))
            .into_iter()
            .map(|range| TrainingPartition {
                range,
                assignment_cells: if gemm {
                    Vec::with_capacity(range.end - range.start)
                } else {
                    vec![0; range.end - range.start]
                },
                assignment_distances: if gemm {
                    Vec::new()
                } else {
                    vec![0.0; range.end - range.start]
                },
                inertia: 0.0,
            })
            .collect();
        Self {
            partitions,
            merged: empty_batch_sums(n_clusters, dimension),
            dot_scratch: DotScratchPool::default(),
        }
    }
}

#[derive(Debug, Default)]
struct DotScratchPool {
    buffers: Mutex<Vec<Vec<f32>>>,
}

impl DotScratchPool {
    fn acquire(&self) -> DotScratch<'_> {
        let values = self
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .unwrap_or_default();
        DotScratch {
            pool: self,
            values: Some(values),
        }
    }
}

struct DotScratch<'a> {
    pool: &'a DotScratchPool,
    values: Option<Vec<f32>>,
}

impl Drop for DotScratch<'_> {
    fn drop(&mut self) {
        let values = self.values.take().expect("scratch is present");
        self.pool
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(values);
    }
}

/// Trains row-major dense vectors with deterministic Lloyd K-means.
pub fn fit_lloyd_kmeans(
    vectors: &[f32],
    dimension: usize,
    context: &ParalliteContext,
    options: KMeansOptions,
) -> Result<KMeansModel> {
    fit_lloyd_kmeans_with_progress(vectors, dimension, context, options, |_| {})
}

/// Trains row-major dense vectors and reports assigned rows within each iteration.
pub fn fit_lloyd_kmeans_with_progress(
    vectors: &[f32],
    dimension: usize,
    context: &ParalliteContext,
    options: KMeansOptions,
    report_progress: impl Fn(usize) + Sync,
) -> Result<KMeansModel> {
    let n = validate_inputs(vectors, dimension, options)?;
    if n < options.n_clusters {
        return Err(Error::InvalidArgument(format!(
            "number of training vectors ({n}) must be at least nlist ({})",
            options.n_clusters
        )));
    }
    if n == options.n_clusters {
        return Ok(KMeansModel {
            centroids: vectors.to_vec(),
            iterations: 0,
        });
    }

    match resolved_mode(options) {
        KMeansMode::Flat => {
            fit_flat_kmeans_with_progress(vectors, dimension, n, context, options, &report_progress)
        }
        KMeansMode::Hierarchical => fit_hierarchical_kmeans_with_progress(
            vectors,
            dimension,
            n,
            context,
            options,
            &report_progress,
        ),
        KMeansMode::Auto => unreachable!("resolved mode is never Auto"),
    }
}

fn resolved_mode(options: KMeansOptions) -> KMeansMode {
    match options.mode {
        KMeansMode::Auto if options.n_clusters >= HIERARCHICAL_AUTO_MIN_CLUSTERS => {
            KMeansMode::Hierarchical
        }
        KMeansMode::Auto => KMeansMode::Flat,
        mode => mode,
    }
}

fn fit_flat_kmeans_with_progress(
    vectors: &[f32],
    dimension: usize,
    n: usize,
    context: &ParalliteContext,
    options: KMeansOptions,
    report_progress: &(impl Fn(usize) + Sync),
) -> Result<KMeansModel> {
    let mut rng = SmallRng::new(options.seed);
    let mut centroids = init_centroids_random(vectors, n, dimension, options.n_clusters, &mut rng);
    let row_norms = detect().row_norms(vectors, dimension);
    let mut workspace =
        TrainingWorkspace::new(n, options.n_clusters, dimension, context.thread_count());
    let mut iterations = 0;
    for iteration in 0..options.max_iter {
        let batch = assign_and_sum_all(
            context,
            vectors,
            n,
            dimension,
            &centroids,
            &row_norms,
            &mut workspace,
            &report_progress,
        )?;
        let centroids_changed = replace_centroids_with_means(&mut centroids, dimension, batch);
        let had_empty_centroid = batch.counts.contains(&0);
        if had_empty_centroid {
            split_empty_centroids(&mut centroids, dimension, &batch.counts);
        }
        iterations = iteration + 1;

        if !batch.inertia.is_finite() {
            return Err(Error::InvalidArgument(
                "K-means produced non-finite inertia".into(),
            ));
        }
        if !centroids_changed && !had_empty_centroid {
            break;
        }
    }

    Ok(KMeansModel {
        centroids,
        iterations,
    })
}

fn fit_hierarchical_kmeans_with_progress(
    vectors: &[f32],
    dimension: usize,
    n: usize,
    context: &ParalliteContext,
    options: KMeansOptions,
    report_progress: &(impl Fn(usize) + Sync),
) -> Result<KMeansModel> {
    let (root_count, child_counts) = hierarchical_shape(options.n_clusters);
    let root_sample_rows = n.min(
        root_count
            .checked_mul(ROOT_TRAINING_POINTS_PER_CENTROID)
            .ok_or_else(|| {
                Error::InvalidArgument("root training sample size overflows usize".into())
            })?,
    );
    let root_sample = sample_training_rows(vectors, dimension, root_sample_rows, options.seed)?;
    let root_options = KMeansOptions {
        n_clusters: root_count,
        mode: KMeansMode::Flat,
        ..options
    };
    let root_model = fit_flat_kmeans_with_progress(
        &root_sample,
        dimension,
        root_sample_rows,
        context,
        root_options,
        &|_| {},
    )?;
    let root_labels = assign_to_centroids(vectors, dimension, &root_model.centroids, context)?;
    let (root_offsets, root_rows) = partition_rows(&root_labels, root_count)?;
    if !has_sufficient_root_rows(&root_offsets, &child_counts) {
        if options.mode == KMeansMode::Auto {
            return fit_flat_kmeans_with_progress(
                vectors,
                dimension,
                n,
                context,
                options,
                report_progress,
            );
        }
        return Err(Error::InvalidArgument(
            "hierarchical root partitions cannot form the requested leaf centroids".into(),
        ));
    }

    let mut centroids = Vec::with_capacity(
        options
            .n_clusters
            .checked_mul(dimension)
            .ok_or_else(|| Error::InvalidArgument("centroid shape overflows usize".into()))?,
    );
    let mut child_vectors = Vec::new();
    let mut iterations = 0;
    for root in 0..root_count {
        let rows = &root_rows[root_offsets[root]..root_offsets[root + 1]];
        let child_count = child_counts[root];
        debug_assert!(rows.len() >= child_count);
        child_vectors.clear();
        let child_values = rows
            .len()
            .checked_mul(dimension)
            .ok_or_else(|| Error::InvalidArgument("child training shape overflows usize".into()))?;
        child_vectors.try_reserve(child_values).map_err(|error| {
            Error::InvalidArgument(format!(
                "cannot reserve {child_values} values for hierarchical child training: {error}"
            ))
        })?;
        for &row_id in rows {
            child_vectors.extend_from_slice(row(vectors, dimension, row_id));
        }
        let child_options = KMeansOptions {
            n_clusters: child_count,
            seed: options.seed.wrapping_add(root as u64).wrapping_add(1),
            mode: KMeansMode::Flat,
            ..options
        };
        let model = fit_flat_kmeans_with_progress(
            &child_vectors,
            dimension,
            rows.len(),
            context,
            child_options,
            report_progress,
        )?;
        iterations = iterations.max(model.iterations);
        centroids.extend_from_slice(&model.centroids);
    }
    debug_assert_eq!(centroids.len(), options.n_clusters * dimension);
    Ok(KMeansModel {
        centroids,
        iterations,
    })
}

fn has_sufficient_root_rows(offsets: &[usize], child_counts: &[usize]) -> bool {
    child_counts
        .iter()
        .enumerate()
        .all(|(root, &child_count)| offsets[root + 1] - offsets[root] >= child_count)
}

fn hierarchical_shape(n_clusters: usize) -> (usize, Vec<usize>) {
    let root_count = rounded_sqrt(n_clusters).max(1);
    let children_per_root = n_clusters / root_count;
    let remainder = n_clusters % root_count;
    let child_counts = (0..root_count)
        .map(|root| children_per_root + usize::from(root < remainder))
        .collect();
    (root_count, child_counts)
}

fn rounded_sqrt(value: usize) -> usize {
    if value <= 1 {
        return value;
    }
    let mut lower = 1;
    let mut upper = value;
    while lower < upper {
        let middle = lower + (upper - lower).div_ceil(2);
        if middle <= value / middle {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    let root = lower;
    let next = root.saturating_add(1);
    if let Some(next_square) = next.checked_mul(next)
        && next_square - value < value - root * root
    {
        return next;
    }
    root
}

fn partition_rows(labels: &[usize], cluster_count: usize) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut offsets = vec![0_usize; cluster_count + 1];
    for &label in labels {
        if label >= cluster_count {
            return Err(Error::InvalidArgument(format!(
                "K-means label {label} exceeds cluster count {cluster_count}"
            )));
        }
        offsets[label + 1] += 1;
    }
    for index in 1..offsets.len() {
        offsets[index] += offsets[index - 1];
    }
    let mut next = offsets[..cluster_count].to_vec();
    let mut rows = vec![0_usize; labels.len()];
    for (row, &label) in labels.iter().enumerate() {
        rows[next[label]] = row;
        next[label] += 1;
    }
    Ok((offsets, rows))
}

/// Assigns dense vectors to their nearest centroids using `context`.
pub fn assign_to_centroids(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    context: &ParalliteContext,
) -> Result<Vec<usize>> {
    let n = validate_assignment_inputs(vectors, dimension, centroids)?;
    let row_norms = detect().row_norms(vectors, dimension);
    assign_all_labels(context, vectors, n, dimension, centroids, &row_norms).map(|(cells, _)| cells)
}

/// Assigns one dense batch to its nearest centroids on the calling thread.
pub fn assign_batch_to_centroids(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
) -> Result<Vec<usize>> {
    let n = validate_assignment_inputs(vectors, dimension, centroids)?;
    let row_norms = detect().row_norms(vectors, dimension);
    assign_all_labels_batch(vectors, n, dimension, centroids, &row_norms).map(|(cells, _)| cells)
}

fn validate_assignment_inputs(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
) -> Result<usize> {
    if dimension == 0
        || vectors.is_empty()
        || !vectors.len().is_multiple_of(dimension)
        || centroids.is_empty()
        || !centroids.len().is_multiple_of(dimension)
    {
        return Err(Error::InvalidArgument(
            "vectors and centroids must be non-empty matrices with the same dimension".into(),
        ));
    }
    if vectors
        .iter()
        .chain(centroids)
        .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidArgument(
            "vectors and centroids must contain only finite values".into(),
        ));
    }
    Ok(vectors.len() / dimension)
}

/// Samples dense training rows reproducibly without replacement.
pub fn sample_training_rows(
    vectors: &[f32],
    dimension: usize,
    max_rows: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    if dimension == 0
        || max_rows == 0
        || vectors.is_empty()
        || !vectors.len().is_multiple_of(dimension)
        || vectors.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidArgument(
            "training sample requires a non-empty finite matrix and positive max_rows".into(),
        ));
    }
    let n = vectors.len() / dimension;
    let sample_size = max_rows.min(n);
    if sample_size == n {
        return Ok(vectors.to_vec());
    }
    let mut rng = SmallRng::new(seed);
    let mut sampled = Vec::with_capacity(sample_size * dimension);
    for row_id in sample_without_replacement(n, sample_size, &mut rng) {
        sampled.extend_from_slice(row(vectors, dimension, row_id));
    }
    Ok(sampled)
}

fn validate_inputs(vectors: &[f32], dimension: usize, options: KMeansOptions) -> Result<usize> {
    if dimension == 0 {
        return Err(Error::InvalidArgument(
            "K-means dimension must be positive".into(),
        ));
    }
    if options.n_clusters == 0 {
        return Err(Error::InvalidArgument("nlist must be positive".into()));
    }
    if options.max_iter == 0 {
        return Err(Error::InvalidArgument(
            "K-means max_iter must be positive".into(),
        ));
    }
    if vectors.is_empty() || !vectors.len().is_multiple_of(dimension) {
        return Err(Error::InvalidArgument(
            "K-means vectors must have shape (n, dimension) with n > 0".into(),
        ));
    }
    if vectors.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(
            "K-means vectors must contain only finite values".into(),
        ));
    }
    Ok(vectors.len() / dimension)
}

fn init_centroids_random(
    vectors: &[f32],
    n: usize,
    dimension: usize,
    n_clusters: usize,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(n_clusters * dimension);
    for row_id in sample_without_replacement(n, n_clusters, rng) {
        centroids.extend_from_slice(row(vectors, dimension, row_id));
    }
    centroids
}

#[allow(clippy::too_many_arguments)]
fn assign_and_sum_all<'a>(
    context: &ParalliteContext,
    vectors: &[f32],
    n: usize,
    dimension: usize,
    centroids: &[f32],
    row_norms: &[f32],
    workspace: &'a mut TrainingWorkspace,
    report_progress: &(impl Fn(usize) + Sync),
) -> Result<&'a BatchSums> {
    let n_clusters = centroids.len() / dimension;
    debug_assert_eq!(
        workspace
            .partitions
            .iter()
            .map(|part| part.range.end - part.range.start)
            .sum::<usize>(),
        n
    );
    let partitions = std::mem::take(&mut workspace.partitions);
    let partials = if uses_gemm(dimension, n_clusters) {
        let centroid_norms = detect().row_norms(centroids, dimension);
        context
            .parallelize(partitions)
            .map(|mut partition| {
                partition.inertia = assign_training_range_gemm_into(
                    vectors,
                    dimension,
                    centroids,
                    &centroid_norms,
                    row_norms,
                    n_clusters,
                    partition.range,
                    &mut partition.assignment_cells,
                    &workspace.dot_scratch,
                    report_progress,
                )?;
                Ok(partition)
            })
            .collect_partitions()?
            .collect()
    } else {
        context
            .parallelize(partitions)
            .map(|mut partition| {
                assign_training_range_simd_into(vectors, dimension, centroids, &mut partition)?;
                report_progress(partition.range.end - partition.range.start);
                Ok(partition)
            })
            .collect_partitions()?
            .collect()
    };
    workspace.partitions = partials;
    reduce_training_assignments(
        context,
        vectors,
        dimension,
        n_clusters,
        &workspace.partitions,
        &mut workspace.merged,
    )?;
    Ok(&workspace.merged)
}

fn assign_training_range_simd_into(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    partition: &mut TrainingPartition,
) -> Result<()> {
    let distance = *detect();
    let range = partition.range;
    let values = &vectors[range.start * dimension..range.end * dimension];
    distance.nearest_squared_l2_rows(
        values,
        centroids,
        dimension,
        &mut partition.assignment_cells,
        &mut partition.assignment_distances,
    );
    partition.inertia = 0.0;
    for &squared_distance in &partition.assignment_distances {
        if !squared_distance.is_finite() {
            return Err(Error::InvalidArgument(
                "centroid assignment produced a non-finite distance".into(),
            ));
        }
        partition.inertia += f64::from(squared_distance);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assign_training_range_gemm_into(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    row_norms: &[f32],
    n_clusters: usize,
    range: RowRange,
    cells: &mut Vec<usize>,
    dot_scratch: &DotScratchPool,
    report_progress: &(impl Fn(usize) + Sync),
) -> Result<f64> {
    cells.clear();
    let mut dot_scratch = dot_scratch.acquire();
    let dot_values = dot_scratch.values.as_mut().expect("scratch is present");
    let rows_per_block = gemm_partition_rows(n_clusters);
    let mut block_start = range.start;
    let mut inertia = 0.0_f64;
    while block_start < range.end {
        let block_end = (block_start + rows_per_block).min(range.end);
        inertia += assign_contiguous_range_gemm_into(
            vectors,
            dimension,
            centroids,
            centroid_norms,
            row_norms,
            n_clusters,
            RowRange {
                start: block_start,
                end: block_end,
            },
            cells,
            dot_values,
        )?;
        report_progress(block_end - block_start);
        block_start = block_end;
    }
    Ok(inertia)
}

fn empty_batch_sums(n_clusters: usize, dimension: usize) -> BatchSums {
    BatchSums {
        sums: vec![0.0; n_clusters * dimension],
        counts: vec![0; n_clusters],
        inertia: 0.0,
    }
}

fn reset_batch_sums(batch: &mut BatchSums) {
    batch.sums.fill(0.0);
    batch.counts.fill(0);
    batch.inertia = 0.0;
}

fn reduce_training_assignments(
    context: &ParalliteContext,
    vectors: &[f32],
    dimension: usize,
    n_clusters: usize,
    partitions: &[TrainingPartition],
    merged: &mut BatchSums,
) -> Result<()> {
    // Each worker owns a disjoint centroid stripe, so aggregate memory does not
    // grow with the number of row partitions.
    let cluster_partitions = context.thread_count().min(n_clusters).max(1);
    let cluster_ranges = row_ranges(n_clusters, n_clusters.div_ceil(cluster_partitions));
    let partials = context
        .parallelize(cluster_ranges)
        .map(|range| {
            let clusters = range.end - range.start;
            let mut sums = vec![0.0_f32; clusters * dimension];
            let mut counts = vec![0_usize; clusters];
            for partition in partitions {
                for (local_row, &cluster) in partition.assignment_cells.iter().enumerate() {
                    if cluster < range.start || cluster >= range.end {
                        continue;
                    }
                    let local_cluster = cluster - range.start;
                    counts[local_cluster] += 1;
                    let global_row = partition.range.start + local_row;
                    let base = local_cluster * dimension;
                    for (slot, value) in sums[base..base + dimension]
                        .iter_mut()
                        .zip(row(vectors, dimension, global_row))
                    {
                        *slot += value;
                    }
                }
            }
            Ok(ClusterSums {
                range,
                sums,
                counts,
            })
        })
        .collect_partitions()?
        .collect();

    reset_batch_sums(merged);
    for partial in partials {
        let sum_start = partial.range.start * dimension;
        let sum_end = partial.range.end * dimension;
        merged.sums[sum_start..sum_end].copy_from_slice(&partial.sums);
        merged.counts[partial.range.start..partial.range.end].copy_from_slice(&partial.counts);
    }
    merged.inertia = partitions.iter().map(|partition| partition.inertia).sum();
    Ok(())
}

struct ClusterSums {
    range: RowRange,
    sums: Vec<f32>,
    counts: Vec<usize>,
}

fn assign_all_labels(
    context: &ParalliteContext,
    vectors: &[f32],
    n: usize,
    dimension: usize,
    centroids: &[f32],
    row_norms: &[f32],
) -> Result<(Vec<usize>, f64)> {
    let n_clusters = centroids.len() / dimension;
    let dot_scratch = DotScratchPool::default();
    let partials = if uses_gemm(dimension, n_clusters) {
        let centroid_norms = detect().row_norms(centroids, dimension);
        let ranges = row_ranges(n, gemm_partition_rows(n_clusters));
        context
            .parallelize(ranges)
            .map(|range| {
                assign_contiguous_range_gemm(
                    vectors,
                    dimension,
                    centroids,
                    &centroid_norms,
                    row_norms,
                    n_clusters,
                    range,
                    &dot_scratch,
                )
            })
            .collect_partitions()?
            .collect()
    } else {
        let ranges = row_ranges(n, PARTITION_ROWS);
        context
            .parallelize(ranges)
            .map(|range| assign_contiguous_range_simd(vectors, dimension, centroids, range))
            .collect_partitions()?
            .collect()
    };

    let mut cells = Vec::with_capacity(n);
    let mut inertia = 0.0_f64;
    for partial in partials {
        cells.extend(partial.cells);
        inertia += partial.inertia;
    }
    Ok((cells, inertia))
}

fn assign_all_labels_batch(
    vectors: &[f32],
    n: usize,
    dimension: usize,
    centroids: &[f32],
    row_norms: &[f32],
) -> Result<(Vec<usize>, f64)> {
    let n_clusters = centroids.len() / dimension;
    let mut cells = Vec::with_capacity(n);
    let mut inertia = 0.0_f64;
    if uses_gemm(dimension, n_clusters) {
        let centroid_norms = detect().row_norms(centroids, dimension);
        let mut dot_scratch = Vec::new();
        for range in row_ranges(n, gemm_partition_rows(n_clusters)) {
            inertia += assign_contiguous_range_gemm_into(
                vectors,
                dimension,
                centroids,
                &centroid_norms,
                row_norms,
                n_clusters,
                range,
                &mut cells,
                &mut dot_scratch,
            )?;
        }
    } else {
        for range in row_ranges(n, PARTITION_ROWS) {
            let partial = assign_contiguous_range_simd(vectors, dimension, centroids, range)?;
            cells.extend(partial.cells);
            inertia += partial.inertia;
        }
    }
    Ok((cells, inertia))
}

fn assign_contiguous_range_simd(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    range: RowRange,
) -> Result<AssignmentBatch> {
    let distance = *detect();
    let rows = range.end - range.start;
    let values = &vectors[range.start * dimension..range.end * dimension];
    let mut cells = vec![0; rows];
    let mut distances = vec![0.0; rows];
    distance.nearest_squared_l2_rows(values, centroids, dimension, &mut cells, &mut distances);
    let mut inertia = 0.0_f64;
    for &best_distance in &distances {
        if !best_distance.is_finite() {
            return Err(Error::InvalidArgument(
                "centroid assignment produced a non-finite distance".into(),
            ));
        }
        inertia += f64::from(best_distance);
    }
    Ok(AssignmentBatch { cells, inertia })
}

#[allow(clippy::too_many_arguments)]
fn assign_contiguous_range_gemm(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    row_norms: &[f32],
    n_clusters: usize,
    range: RowRange,
    dot_scratch: &DotScratchPool,
) -> Result<AssignmentBatch> {
    let mut cells = Vec::with_capacity(range.end - range.start);
    let mut dot_scratch = dot_scratch.acquire();
    let dot_values = dot_scratch.values.as_mut().expect("scratch is present");
    let inertia = assign_contiguous_range_gemm_into(
        vectors,
        dimension,
        centroids,
        centroid_norms,
        row_norms,
        n_clusters,
        range,
        &mut cells,
        dot_values,
    )?;
    Ok(AssignmentBatch { cells, inertia })
}

#[allow(clippy::too_many_arguments)]
fn assign_contiguous_range_gemm_into(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    row_norms: &[f32],
    n_clusters: usize,
    range: RowRange,
    cells: &mut Vec<usize>,
    dot_scratch: &mut Vec<f32>,
) -> Result<f64> {
    let rows = range.end - range.start;
    let left = &vectors[range.start * dimension..range.end * dimension];
    let dot_count = rows
        .checked_mul(n_clusters)
        .ok_or_else(|| Error::InvalidArgument("GEMM output size overflows usize".into()))?;
    with_dot_scratch(dot_scratch, dot_count, |dots| {
        row_dot_products(left, centroids, rows, dimension, n_clusters, dots)?;

        let mut inertia = 0.0_f64;
        for local_row in 0..rows {
            let dot_row = &dots[local_row * n_clusters..(local_row + 1) * n_clusters];
            let (cell, distance_without_row_norm) =
                nearest_centroid_from_dots(dot_row, centroid_norms);
            let distance =
                (row_norms[range.start + local_row] + distance_without_row_norm).max(0.0);
            if !distance.is_finite() {
                return Err(Error::InvalidArgument(
                    "GEMM centroid assignment produced a non-finite distance".into(),
                ));
            }
            cells.push(cell);
            inertia += f64::from(distance);
        }
        Ok(inertia)
    })
}

fn nearest_centroid_from_dots(dots: &[f32], centroid_norms: &[f32]) -> (usize, f32) {
    let mut best_cluster = 0;
    let mut best_distance = centroid_norms[0] - 2.0 * dots[0];
    for cluster in 1..dots.len() {
        let candidate = centroid_norms[cluster] - 2.0 * dots[cluster];
        if candidate < best_distance {
            best_cluster = cluster;
            best_distance = candidate;
        }
    }
    (best_cluster, best_distance)
}

const fn uses_gemm(dimension: usize, n_clusters: usize) -> bool {
    dimension >= GEMM_MIN_DIMENSION && n_clusters >= GEMM_MIN_CLUSTERS
}

fn gemm_partition_rows(n_clusters: usize) -> usize {
    (GEMM_MAX_OUTPUT_VALUES / n_clusters).clamp(1, PARTITION_ROWS)
}

fn training_partition_count(rows: usize, workers: usize) -> usize {
    workers.max(1).min(rows).max(1)
}

fn with_dot_scratch<T>(
    scratch: &mut Vec<f32>,
    len: usize,
    visit: impl FnOnce(&mut [f32]) -> Result<T>,
) -> Result<T> {
    if scratch.len() < len {
        scratch.resize(len, 0.0);
    }
    visit(&mut scratch[..len])
}

fn replace_centroids_with_means(centroids: &mut [f32], dimension: usize, sums: &BatchSums) -> bool {
    let mut changed = false;
    for (cluster, &count) in sums.counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let base = cluster * dimension;
        for dim in 0..dimension {
            let mean = sums.sums[base + dim] / count as f32;
            changed |= centroids[base + dim].to_bits() != mean.to_bits();
            centroids[base + dim] = mean;
        }
    }
    changed
}

fn split_empty_centroids(centroids: &mut [f32], dimension: usize, counts: &[usize]) {
    let mut counts = counts.to_vec();
    let total_count = counts.iter().sum::<usize>();
    let mut rng = SmallRng::new(1234);
    for empty in 0..counts.len() {
        if counts[empty] != 0 {
            continue;
        }
        let Some(donor) = splittable_cluster(&counts, total_count, &mut rng) else {
            return;
        };
        let empty_base = empty * dimension;
        let donor_base = donor * dimension;
        let (empty_centroid, donor_centroid) =
            two_mut_rows(centroids, empty_base, donor_base, dimension);
        empty_centroid.copy_from_slice(donor_centroid);
        for dim in 0..dimension {
            if dim % 2 == 0 {
                empty_centroid[dim] *= 1.0 + EMPTY_SPLIT_EPS;
                donor_centroid[dim] *= 1.0 - EMPTY_SPLIT_EPS;
            } else {
                empty_centroid[dim] *= 1.0 - EMPTY_SPLIT_EPS;
                donor_centroid[dim] *= 1.0 + EMPTY_SPLIT_EPS;
            }
        }
        counts[empty] = counts[donor] / 2;
        counts[donor] -= counts[empty];
    }
}

fn splittable_cluster(counts: &[usize], total_count: usize, rng: &mut SmallRng) -> Option<usize> {
    if total_count <= counts.len() {
        return largest_splittable_cluster(counts);
    }
    let denominator = (total_count - counts.len()) as f64;
    let mut candidate = 0;
    for _ in 0..10 * counts.len() {
        let probability = counts
            .get(candidate)
            .and_then(|count| count.checked_sub(1))
            .map_or(0.0, |count| count as f64 / denominator);
        if rng.gen_f64() < probability {
            return Some(candidate);
        }
        candidate = (candidate + 1) % counts.len();
    }
    largest_splittable_cluster(counts)
}

fn largest_splittable_cluster(counts: &[usize]) -> Option<usize> {
    counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 1)
        .max_by_key(|(_, count)| **count)
        .map(|(cluster, _)| cluster)
}

fn two_mut_rows(
    values: &mut [f32],
    first_base: usize,
    second_base: usize,
    dimension: usize,
) -> (&mut [f32], &mut [f32]) {
    debug_assert_ne!(first_base, second_base);
    if first_base < second_base {
        let (left, right) = values.split_at_mut(second_base);
        (
            &mut left[first_base..first_base + dimension],
            &mut right[..dimension],
        )
    } else {
        let (left, right) = values.split_at_mut(first_base);
        (
            &mut right[..dimension],
            &mut left[second_base..second_base + dimension],
        )
    }
}

fn sample_without_replacement(n: usize, sample_size: usize, rng: &mut SmallRng) -> Vec<usize> {
    debug_assert!(sample_size <= n);
    let mut swaps = HashMap::with_capacity(sample_size);
    let mut sample = Vec::with_capacity(sample_size);
    for index in 0..sample_size {
        let swap = index + rng.gen_usize(n - index);
        let selected = swaps.get(&swap).copied().unwrap_or(swap);
        let displaced = swaps.get(&index).copied().unwrap_or(index);
        swaps.insert(swap, displaced);
        swaps.remove(&index);
        sample.push(selected);
    }
    sample
}

fn row_ranges(rows: usize, rows_per_range: usize) -> Vec<RowRange> {
    let rows_per_range = rows_per_range.max(1);
    let mut ranges = Vec::with_capacity(rows.div_ceil(rows_per_range));
    let mut start = 0;
    while start < rows {
        let end = (start + rows_per_range).min(rows);
        ranges.push(RowRange { start, end });
        start = end;
    }
    ranges
}

#[inline]
fn row(values: &[f32], dimension: usize, row_id: usize) -> &[f32] {
    let start = row_id * dimension;
    &values[start..start + dimension]
}

#[derive(Debug, Clone, Copy)]
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn gen_usize(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

    fn gen_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

#[cfg(test)]
mod tests;
