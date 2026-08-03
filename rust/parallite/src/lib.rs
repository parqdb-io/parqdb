#![allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]

//! Spark-like local RDD/dataflow execution for Rust.
//!
//! Parallite is a lightweight local parallel pipeline engine. It borrows the
//! useful parts of Apache Spark's RDD model for in-process workloads: dataset
//! lineage, per-element `map` and `flatten`, partition-level
//! `map_partitions`, shuffle-style `partition_by`, and action-triggered
//! execution with `collect_partitions`.
//!
//! It is intentionally not a distributed engine. The goal is to provide a small
//! local dataflow runtime for parallel pipelines that need Spark-like partition
//! semantics without a Spark cluster.
//!
//! ```rust
//! use parallite::prelude::*;
//!
//! let pc = parallite::ParalliteContext::default();
//! let partitions = pc
//!     .parallelize_n(vec![0, 1, 2, 3], 2)
//!     .map(|value| Ok::<_, ()>((value % 2, value)))
//!     .partition_by(2, |key| Ok::<_, ()>(*key))
//!     .map_partitions(|iter| Ok::<_, ()>(iter.map(|(_key, value)| value)))
//!     .collect_partitions()
//!     .unwrap();
//!
//! assert_eq!(partitions.into_partitions(), vec![vec![0, 2], vec![1, 3]]);
//! ```
//!
//! Use `ParalliteContext` when callers need SparkContext-like control over the
//! executor or source partitioning:
//!
//! ```rust
//! use parallite::prelude::*;
//!
//! let executor = parallite::Executor::with_threads(2).unwrap();
//! let pc = parallite::ParalliteContext::with_executor(executor);
//! let values = pc
//!     .parallelize_n(vec![1, 2, 3, 4], 2)
//!     .flat_map(|value| Ok::<_, ()>(vec![value, value * 10]))
//!     .collect_partitions()
//!     .unwrap()
//!     .collect();
//!
//! assert_eq!(values, vec![1, 10, 2, 20, 3, 30, 4, 40]);
//! ```

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use rayon::prelude::*;

/// Common extension traits for building and collecting pipelines.
pub mod prelude {
    pub use crate::{CollectPartitions, DatasetExt};
}

/// Creates a dataset using a default parallel execution context.
pub fn parallelize<T: Send>(data: Vec<T>) -> VecDataset<T> {
    ParalliteContext::default().parallelize(data)
}

/// Execution policy shared by a pipeline.
#[derive(Clone)]
pub struct Executor {
    kind: ExecutorKind,
}

#[derive(Clone)]
enum ExecutorKind {
    Serial,
    Rayon(Arc<rayon::ThreadPool>),
}

/// Error returned while constructing an executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorError {
    message: String,
}

impl ExecutorError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "executor error: {}", self.message)
    }
}

impl Error for ExecutorError {}

impl Executor {
    /// Creates an executor that runs work on the calling thread.
    pub fn serial() -> Self {
        Self {
            kind: ExecutorKind::Serial,
        }
    }

    /// Creates an executor backed by a Rayon pool with `num_threads` workers.
    pub fn with_threads(num_threads: usize) -> Result<Self, ExecutorError> {
        if num_threads == 0 {
            return Err(ExecutorError::new("thread count must be positive".into()));
        }
        if num_threads == 1 {
            return Ok(Self::serial());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|err| ExecutorError::new(err.to_string()))?;
        Ok(Self {
            kind: ExecutorKind::Rayon(Arc::new(pool)),
        })
    }

    /// Creates an executor backed by an existing Rayon pool.
    pub fn rayon(pool: Arc<rayon::ThreadPool>) -> Self {
        Self {
            kind: ExecutorKind::Rayon(pool),
        }
    }

    /// Creates an executor sized to the process's available parallelism.
    pub fn default_parallel() -> Result<Self, ExecutorError> {
        let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        Self::with_threads(threads)
    }

    fn thread_count(&self) -> usize {
        match &self.kind {
            ExecutorKind::Serial => 1,
            ExecutorKind::Rayon(pool) => pool.current_num_threads(),
        }
    }

    fn map_ordered_fallible<I, O, F, E>(
        &self,
        inputs: Vec<I>,
        f: F,
    ) -> Result<Vec<O>, ParalliteError<E>>
    where
        I: Send,
        O: Send,
        E: Send,
        F: Fn(I) -> Result<O, ParalliteError<E>> + Send + Sync,
    {
        match &self.kind {
            ExecutorKind::Serial => inputs.into_iter().map(f).collect(),
            ExecutorKind::Rayon(pool) => pool.install(|| inputs.into_par_iter().map(f).collect()),
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::default_parallel().unwrap_or_else(|_| Self::serial())
    }
}

/// Error produced while evaluating a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParalliteError<E> {
    /// Error returned by a user-supplied transformation.
    User(E),
    /// Invalid pipeline or collection parameter.
    InvalidParameter(String),
    /// A partition function returned an out-of-range partition.
    InvalidPartition {
        /// Returned partition index.
        partition_id: usize,
        /// Number of available partitions.
        num_partitions: usize,
    },
}

impl<E: fmt::Display> fmt::Display for ParalliteError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(error) => write!(f, "{error}"),
            Self::InvalidParameter(message) => write!(f, "invalid parameter: {message}"),
            Self::InvalidPartition {
                partition_id,
                num_partitions,
            } => write!(
                f,
                "partition function returned {partition_id} for {num_partitions} partitions"
            ),
        }
    }
}

impl<E> Error for ParalliteError<E> where E: Error + 'static {}

/// Partitioning options used when a context creates source datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParalliteOptions {
    default_slices: Option<usize>,
    source_partition_factor: usize,
}

impl ParalliteOptions {
    /// Returns the default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the configured source partition count, if fixed.
    pub fn default_slices(&self) -> Option<usize> {
        self.default_slices
    }

    /// Returns the source partitions created per executor thread.
    pub fn source_partition_factor(&self) -> usize {
        self.source_partition_factor
    }

    fn default_source_slices(&self, executor: &Executor, len: usize) -> usize {
        if len == 0 {
            return 1;
        }
        let slices = self
            .default_slices
            .unwrap_or_else(|| executor.thread_count() * self.source_partition_factor.max(1));
        len.min(slices.max(1))
    }
}

impl Default for ParalliteOptions {
    fn default() -> Self {
        Self {
            default_slices: None,
            source_partition_factor: 4,
        }
    }
}

/// Reusable entry point for constructing parallel dataflow pipelines.
#[derive(Clone, Default)]
pub struct ParalliteContext {
    executor: Executor,
    options: ParalliteOptions,
}

impl ParalliteContext {
    /// Starts building a configured context.
    pub fn builder() -> ParalliteContextBuilder {
        ParalliteContextBuilder::default()
    }

    /// Creates a context that uses `executor` and default options.
    pub fn with_executor(executor: Executor) -> Self {
        Self {
            executor,
            options: ParalliteOptions::default(),
        }
    }

    /// Returns the context's source partitioning options.
    pub fn options(&self) -> ParalliteOptions {
        self.options
    }

    /// Returns the number of executor threads.
    pub fn thread_count(&self) -> usize {
        self.executor.thread_count()
    }

    /// Splits `data` using the context's source partitioning policy.
    pub fn parallelize<T: Send>(&self, data: Vec<T>) -> VecDataset<T> {
        let len = data.len();
        self.parallelize_n(
            data,
            self.options.default_source_slices(&self.executor, len),
        )
    }

    /// Splits `data` into `num_slices` source partitions, normalizing zero to one.
    pub fn parallelize_n<T: Send>(&self, data: Vec<T>, num_slices: usize) -> VecDataset<T> {
        let partitions = split_vec(data, num_slices);
        VecDataset {
            executor: self.executor.clone(),
            partitions,
        }
    }
}

/// Builder for a [`ParalliteContext`].
#[derive(Default)]
pub struct ParalliteContextBuilder {
    executor: Option<Executor>,
    options: ParalliteOptions,
}

impl ParalliteContextBuilder {
    /// Uses an existing executor.
    pub fn executor(mut self, executor: Executor) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Creates and uses an executor with `num_threads` workers.
    pub fn threads(mut self, num_threads: usize) -> Result<Self, ExecutorError> {
        self.executor = Some(Executor::with_threads(num_threads)?);
        Ok(self)
    }

    /// Sets a fixed number of partitions for newly parallelized sources.
    pub fn default_slices(mut self, default_slices: usize) -> Self {
        self.options.default_slices = Some(default_slices.max(1));
        self
    }

    /// Sets the number of source partitions created per executor thread.
    pub fn source_partition_factor(mut self, factor: usize) -> Self {
        self.options.source_partition_factor = factor.max(1);
        self
    }

    /// Builds the configured context.
    pub fn build(self) -> ParalliteContext {
        ParalliteContext {
            executor: self.executor.unwrap_or_default(),
            options: self.options,
        }
    }
}

/// Materialized partitions produced by a pipeline.
pub struct Partitioned<T> {
    executor: Executor,
    partitions: Vec<Vec<T>>,
}

impl<T> Partitioned<T> {
    /// Consumes the result and iterates over whole partitions.
    pub fn into_partition_iter(self) -> impl Iterator<Item = Vec<T>> {
        self.partitions.into_iter()
    }

    /// Consumes the result and returns its partitions.
    pub fn into_partitions(self) -> Vec<Vec<T>> {
        self.partitions
    }

    /// Consumes the result and concatenates partitions in partition order.
    pub fn collect(self) -> Vec<T> {
        self.partitions.into_iter().flatten().collect()
    }

    #[cfg(test)]
    fn partition_count(&self) -> usize {
        self.partitions.len()
    }
}

/// A lazily evaluated partitioned dataset.
pub trait Dataset<E>: Sized {
    /// Element produced by the dataset.
    type Item: Send;
}

/// Lazy transformations available on every [`Dataset`].
pub trait DatasetExt<E>: Dataset<E> {
    /// Applies a fallible function to every element.
    fn map<U, F>(self, f: F) -> MapDataset<Self, F>
    where
        U: Send,
        F: Fn(Self::Item) -> Result<U, E> + Send + Sync,
    {
        MapDataset { input: self, f }
    }

    /// Flattens each iterable element while preserving partition order.
    fn flatten<U>(self) -> FlattenDataset<Self>
    where
        Self::Item: IntoIterator<Item = U>,
        <Self::Item as IntoIterator>::IntoIter: Send,
        U: Send,
    {
        FlattenDataset { input: self }
    }

    /// Maps each element to an iterable and then flattens it.
    fn flat_map<C, U, F>(self, f: F) -> FlattenDataset<MapDataset<Self, F>>
    where
        C: IntoIterator<Item = U> + Send,
        C::IntoIter: Send,
        U: Send,
        F: Fn(Self::Item) -> Result<C, E> + Send + Sync,
    {
        FlattenDataset {
            input: MapDataset { input: self, f },
        }
    }

    /// Shuffles key-value pairs into `num_partitions` output partitions.
    fn partition_by<K, V, F>(
        self,
        num_partitions: usize,
        partition_fn: F,
    ) -> ShuffledDataset<Self, F>
    where
        Self: Dataset<E, Item = (K, V)>,
        K: Send,
        V: Send,
        F: Fn(&K) -> Result<usize, E> + Send + Sync,
    {
        ShuffledDataset {
            input: self,
            num_partitions,
            partition_fn,
        }
    }

    /// Applies a fallible function once to each materialized partition.
    fn map_partitions<U, I, F>(self, f: F) -> MapPartitionsDataset<Self, F>
    where
        U: Send,
        F: Fn(std::vec::IntoIter<Self::Item>) -> Result<I, E> + Send + Sync,
        I: IntoIterator<Item = U>,
    {
        MapPartitionsDataset { input: self, f }
    }
}

impl<N, E> DatasetExt<E> for N where N: Dataset<E> {}

/// Terminal operation that materializes a dataset as ordered partitions.
pub trait CollectPartitions<E>: Dataset<E> {
    /// Executes the pipeline and returns its materialized partitions.
    fn collect_partitions(self) -> Result<Partitioned<Self::Item>, ParalliteError<E>>;
}

trait PartitionLocalDataset<E>: Dataset<E> {
    type PartitionIter: Iterator<Item = Result<Self::Item, ParalliteError<E>>> + Send;

    fn into_partition_iters(self) -> (Executor, Vec<Self::PartitionIter>);
}

/// Source dataset backed by in-memory vectors.
pub struct VecDataset<T> {
    executor: Executor,
    partitions: Vec<Vec<T>>,
}

impl<T: Send, E> Dataset<E> for VecDataset<T> {
    type Item = T;
}

impl<T: Send, E> PartitionLocalDataset<E> for VecDataset<T> {
    type PartitionIter = OkIter<T, E>;

    fn into_partition_iters(self) -> (Executor, Vec<Self::PartitionIter>) {
        let partitions = self
            .partitions
            .into_iter()
            .map(|partition| OkIter {
                inner: partition.into_iter(),
                _error: PhantomData,
            })
            .collect();
        (self.executor, partitions)
    }
}

/// Dataset node produced by [`DatasetExt::map`].
pub struct MapDataset<N, F> {
    input: N,
    f: F,
}

impl<N, F, U, E> Dataset<E> for MapDataset<N, F>
where
    N: Dataset<E>,
    U: Send,
    F: Fn(N::Item) -> Result<U, E> + Send + Sync,
{
    type Item = U;
}

impl<N, F, U, E> PartitionLocalDataset<E> for MapDataset<N, F>
where
    N: PartitionLocalDataset<E>,
    U: Send,
    E: Send,
    F: Fn(N::Item) -> Result<U, E> + Send + Sync,
{
    type PartitionIter = MapIter<N::PartitionIter, F, N::Item, U>;

    fn into_partition_iters(self) -> (Executor, Vec<Self::PartitionIter>) {
        let (executor, partitions) = self.input.into_partition_iters();
        let f = Arc::new(self.f);
        let partitions = partitions
            .into_iter()
            .map(|inner| MapIter {
                inner,
                f: Arc::clone(&f),
                _input: std::marker::PhantomData,
                _output: std::marker::PhantomData,
            })
            .collect();
        (executor, partitions)
    }
}

/// Dataset node produced by [`DatasetExt::flatten`].
pub struct FlattenDataset<N> {
    input: N,
}

impl<N, C, U, E> Dataset<E> for FlattenDataset<N>
where
    N: Dataset<E, Item = C>,
    C: IntoIterator<Item = U>,
    C::IntoIter: Send,
    U: Send,
{
    type Item = U;
}

impl<N, C, U, E> PartitionLocalDataset<E> for FlattenDataset<N>
where
    N: PartitionLocalDataset<E, Item = C>,
    C: IntoIterator<Item = U>,
    C::IntoIter: Send,
    U: Send,
{
    type PartitionIter = FlattenIter<N::PartitionIter, C>;

    fn into_partition_iters(self) -> (Executor, Vec<Self::PartitionIter>) {
        let (executor, partitions) = self.input.into_partition_iters();
        let partitions = partitions
            .into_iter()
            .map(|outer| FlattenIter {
                outer,
                current: None,
            })
            .collect();
        (executor, partitions)
    }
}

/// Dataset node produced by [`DatasetExt::partition_by`].
pub struct ShuffledDataset<N, F> {
    input: N,
    num_partitions: usize,
    partition_fn: F,
}

impl<N, F, E> Dataset<E> for ShuffledDataset<N, F>
where
    N: Dataset<E>,
{
    type Item = N::Item;
}

impl<N, F, K, V, E> CollectPartitions<E> for ShuffledDataset<N, F>
where
    N: PartitionLocalDataset<E, Item = (K, V)>,
    K: Send,
    V: Send,
    E: Send,
    F: Fn(&K) -> Result<usize, E> + Send + Sync,
{
    fn collect_partitions(self) -> Result<Partitioned<Self::Item>, ParalliteError<E>> {
        if self.num_partitions == 0 {
            return Err(ParalliteError::InvalidParameter(
                "partition count must be positive".into(),
            ));
        }

        let (executor, partition_iters) = self.input.into_partition_iters();
        let partition_fn = &self.partition_fn;
        let local_buckets = executor.map_ordered_fallible(partition_iters, |partition| {
            let mut buckets = empty_partitions(self.num_partitions);
            for item in partition {
                let (key, value) = item?;
                let partition_id = partition_fn(&key).map_err(ParalliteError::User)?;
                if partition_id >= self.num_partitions {
                    return Err(ParalliteError::InvalidPartition {
                        partition_id,
                        num_partitions: self.num_partitions,
                    });
                }
                buckets[partition_id].push((key, value));
            }
            Ok(buckets)
        })?;

        let mut partitions = empty_partitions(self.num_partitions);
        for buckets in local_buckets {
            for (partition_id, bucket) in buckets.into_iter().enumerate() {
                partitions[partition_id].extend(bucket);
            }
        }

        Ok(Partitioned {
            executor,
            partitions,
        })
    }
}

/// Dataset node produced by [`DatasetExt::map_partitions`].
pub struct MapPartitionsDataset<N, F> {
    input: N,
    f: F,
}

impl<N, F, U, I, E> Dataset<E> for MapPartitionsDataset<N, F>
where
    N: Dataset<E>,
    U: Send,
    F: Fn(std::vec::IntoIter<N::Item>) -> Result<I, E> + Send + Sync,
    I: IntoIterator<Item = U>,
{
    type Item = U;
}

impl<N, F, U, I, E> CollectPartitions<E> for MapPartitionsDataset<N, F>
where
    N: CollectPartitions<E>,
    U: Send,
    E: Send,
    F: Fn(std::vec::IntoIter<N::Item>) -> Result<I, E> + Send + Sync,
    I: IntoIterator<Item = U>,
{
    fn collect_partitions(self) -> Result<Partitioned<Self::Item>, ParalliteError<E>> {
        let input = self.input.collect_partitions()?;
        let executor = input.executor;
        let output = executor.map_ordered_fallible(input.partitions, |partition| {
            Ok((self.f)(partition.into_iter())
                .map_err(ParalliteError::User)?
                .into_iter()
                .collect::<Vec<_>>())
        })?;
        Ok(Partitioned {
            executor,
            partitions: output,
        })
    }
}

impl<T, E> CollectPartitions<E> for VecDataset<T>
where
    T: Send,
    E: Send,
{
    fn collect_partitions(self) -> Result<Partitioned<Self::Item>, ParalliteError<E>> {
        collect_narrow(self)
    }
}

impl<N, F, U, E> CollectPartitions<E> for MapDataset<N, F>
where
    N: PartitionLocalDataset<E>,
    U: Send,
    E: Send,
    F: Fn(N::Item) -> Result<U, E> + Send + Sync,
{
    fn collect_partitions(self) -> Result<Partitioned<Self::Item>, ParalliteError<E>> {
        collect_narrow(self)
    }
}

impl<N, C, U, E> CollectPartitions<E> for FlattenDataset<N>
where
    N: PartitionLocalDataset<E, Item = C>,
    C: IntoIterator<Item = U>,
    C::IntoIter: Send,
    U: Send,
    E: Send,
{
    fn collect_partitions(self) -> Result<Partitioned<Self::Item>, ParalliteError<E>> {
        collect_narrow(self)
    }
}

fn collect_narrow<N, E>(dataset: N) -> Result<Partitioned<N::Item>, ParalliteError<E>>
where
    N: PartitionLocalDataset<E>,
    E: Send,
{
    let (executor, partition_iters) = dataset.into_partition_iters();
    let partitions = executor.map_ordered_fallible(partition_iters, |partition| {
        partition.collect::<Result<Vec<_>, ParalliteError<E>>>()
    })?;
    Ok(Partitioned {
        executor,
        partitions,
    })
}

/// Iterator that wraps source elements in successful pipeline results.
pub struct OkIter<T, E> {
    inner: std::vec::IntoIter<T>,
    _error: PhantomData<fn() -> E>,
}

impl<T, E> Iterator for OkIter<T, E> {
    type Item = Result<T, ParalliteError<E>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Ok)
    }
}

/// Iterator that applies a fallible map transformation.
pub struct MapIter<I, F, T, U> {
    inner: I,
    f: Arc<F>,
    _input: std::marker::PhantomData<fn(T)>,
    _output: std::marker::PhantomData<fn() -> U>,
}

impl<I, F, T, U, E> Iterator for MapIter<I, F, T, U>
where
    I: Iterator<Item = Result<T, ParalliteError<E>>>,
    F: Fn(T) -> Result<U, E>,
{
    type Item = Result<U, ParalliteError<E>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|item| item.and_then(|value| (self.f)(value).map_err(ParalliteError::User)))
    }
}

/// Iterator that flattens successful iterable values.
pub struct FlattenIter<I, C>
where
    C: IntoIterator,
{
    outer: I,
    current: Option<C::IntoIter>,
}

impl<I, C, U, E> Iterator for FlattenIter<I, C>
where
    I: Iterator<Item = Result<C, ParalliteError<E>>>,
    C: IntoIterator<Item = U>,
{
    type Item = Result<U, ParalliteError<E>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current) = self.current.as_mut()
                && let Some(item) = current.next()
            {
                return Some(Ok(item));
            }

            match self.outer.next()? {
                Ok(items) => self.current = Some(items.into_iter()),
                Err(err) => return Some(Err(err)),
            }
        }
    }
}

fn split_vec<T>(data: Vec<T>, num_slices: usize) -> Vec<Vec<T>> {
    let slices = num_slices.max(1);
    let len = data.len();
    let mut iter = data.into_iter();
    let mut partitions = Vec::with_capacity(slices);
    let mut emitted = 0;
    for slice in 0..slices {
        let remaining = len - emitted;
        let slices_left = slices - slice;
        let size = remaining.div_ceil(slices_left);
        let mut partition = Vec::with_capacity(size);
        for _ in 0..size {
            if let Some(item) = iter.next() {
                partition.push(item);
                emitted += 1;
            }
        }
        partitions.push(partition);
    }
    partitions
}

fn empty_partitions<T>(num_partitions: usize) -> Vec<Vec<T>> {
    (0..num_partitions).map(|_| Vec::new()).collect()
}

#[cfg(test)]
mod tests;
