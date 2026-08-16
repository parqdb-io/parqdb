// HNSW graph construction and search adapted from Faiss (MIT License).
// See THIRD_PARTY_NOTICES.md for attribution.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Mutex, RwLock};

use parallite::ParalliteContext;
use parallite::prelude::{CollectPartitions, DatasetExt};
use relify_kernels::{KernelError, LvqEncodedBatch, LvqRowQuery, detect};

use crate::Error;

const MAX_LEVEL: usize = 16;
const EMPTY_NEIGHBOR: i32 = -1;

#[derive(Debug, Clone, Copy)]
struct Scored {
    cid: usize,
    distance: f32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cid == other.cid && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.cid.cmp(&other.cid))
    }
}

trait GraphAccess: Sync {
    fn neighbor_range(&self, cid: usize, layer: usize) -> (usize, usize);
    fn neighbor(&self, offset: usize) -> Option<usize>;
}

#[derive(Debug)]
pub(super) struct SearchScratch {
    marks: Vec<u32>,
    epoch: u32,
    candidates: BinaryHeap<Reverse<Scored>>,
    nearest: BinaryHeap<Scored>,
    ordered: Vec<Scored>,
    query_candidates: MinimaxHeap,
    query_results: BinaryHeap<Scored>,
    #[cfg(test)]
    pub(super) distance_evaluations: usize,
}

impl SearchScratch {
    pub(super) fn new(size: usize) -> Self {
        Self {
            marks: vec![0; size],
            epoch: 0,
            candidates: BinaryHeap::new(),
            nearest: BinaryHeap::new(),
            ordered: Vec::new(),
            query_candidates: MinimaxHeap::default(),
            query_results: BinaryHeap::new(),
            #[cfg(test)]
            distance_evaluations: 0,
        }
    }

    pub(super) fn new_for_query(
        size: usize,
        result_capacity: usize,
        candidate_capacity: usize,
    ) -> Self {
        let mut scratch = Self::new(size);
        scratch.ordered = Vec::with_capacity(result_capacity);
        scratch.query_candidates.reset(candidate_capacity);
        scratch.query_results = BinaryHeap::with_capacity(result_capacity);
        scratch
    }

    pub(super) fn resident_size(&self) -> usize {
        self.marks.capacity() * std::mem::size_of::<u32>()
            + self.candidates.capacity() * std::mem::size_of::<Reverse<Scored>>()
            + self.nearest.capacity() * std::mem::size_of::<Scored>()
            + self.ordered.capacity() * std::mem::size_of::<Scored>()
            + self.query_candidates.resident_size()
            + self.query_results.capacity() * std::mem::size_of::<Scored>()
    }

    pub(super) fn trim_for_query_pool(
        &mut self,
        result_capacity: usize,
        candidate_capacity: usize,
    ) {
        self.ordered.clear();
        self.ordered.shrink_to(result_capacity);
        self.query_candidates.trim_to(candidate_capacity);
        self.query_results.clear();
        self.query_results.shrink_to(result_capacity);
    }

    fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.candidates.clear();
        self.nearest.clear();
        self.ordered.clear();
        #[cfg(test)]
        {
            self.distance_evaluations = 0;
        }
    }

    fn visit(&mut self, cid: usize) -> bool {
        if self.marks[cid] == self.epoch {
            false
        } else {
            self.marks[cid] = self.epoch;
            true
        }
    }

    fn reset_query(&mut self, ef: usize) {
        self.reset();
        self.query_candidates.reset(ef);
        self.query_results.clear();
    }
}

#[derive(Debug, Default)]
struct MinimaxHeap {
    ids: Vec<i32>,
    distances: Vec<f32>,
    length: usize,
    valid: usize,
    capacity: usize,
}

impl MinimaxHeap {
    fn reset(&mut self, capacity: usize) {
        self.capacity = capacity;
        self.length = 0;
        self.valid = 0;
        if self.ids.len() < capacity {
            self.ids.resize(capacity, EMPTY_NEIGHBOR);
            self.distances.resize(capacity, 0.0);
        }
    }

    fn push(&mut self, cid: usize, distance: f32) {
        if self.length == self.capacity {
            if distance >= self.distances[0] {
                return;
            }
            if self.ids[0] != EMPTY_NEIGHBOR {
                self.valid -= 1;
            }
            self.length -= 1;
            if self.length != 0 {
                self.ids[0] = self.ids[self.length];
                self.distances[0] = self.distances[self.length];
                self.sift_down(0);
            }
        }
        let mut position = self.length;
        self.length += 1;
        while position != 0 {
            let parent = (position - 1) / 2;
            if self.distances[parent] >= distance {
                break;
            }
            self.ids[position] = self.ids[parent];
            self.distances[position] = self.distances[parent];
            position = parent;
        }
        self.ids[position] = match i32::try_from(cid) {
            Ok(cid) => cid,
            Err(_) => unreachable!("HNSW cid was validated before graph construction"),
        };
        self.distances[position] = distance;
        self.valid += 1;
    }

    #[allow(unsafe_code)]
    fn pop_min(&mut self) -> Option<Scored> {
        #[cfg(target_arch = "x86_64")]
        let position = if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F was detected and both slices cover `length` entries.
            unsafe { find_min_position_avx512(&self.ids, &self.distances, self.length) }
        } else {
            find_min_position_scalar(&self.ids, &self.distances, self.length)
        }?;
        #[cfg(not(target_arch = "x86_64"))]
        let position = find_min_position_scalar(&self.ids, &self.distances, self.length)?;
        let cid = usize::try_from(self.ids[position]).ok()?;
        self.ids[position] = EMPTY_NEIGHBOR;
        self.valid -= 1;
        Some(Scored {
            cid,
            distance: self.distances[position],
        })
    }

    #[allow(unsafe_code)]
    fn count_below(&self, distance: f32) -> usize {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F was detected and the slice covers `length` entries.
            return unsafe { count_below_avx512(&self.distances, self.length, distance) };
        }
        count_below_scalar(&self.distances, self.length, distance)
    }

    fn is_empty(&self) -> bool {
        self.valid == 0
    }

    fn resident_size(&self) -> usize {
        self.ids.capacity() * std::mem::size_of::<i32>()
            + self.distances.capacity() * std::mem::size_of::<f32>()
    }

    fn trim_to(&mut self, capacity: usize) {
        self.length = 0;
        self.valid = 0;
        self.capacity = capacity;
        self.ids.truncate(capacity);
        self.ids.shrink_to(capacity);
        self.distances.truncate(capacity);
        self.distances.shrink_to(capacity);
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let left = position * 2 + 1;
            if left >= self.length {
                return;
            }
            let right = left + 1;
            let child = if right < self.length && self.distances[right] > self.distances[left] {
                right
            } else {
                left
            };
            if self.distances[position] >= self.distances[child] {
                return;
            }
            self.ids.swap(position, child);
            self.distances.swap(position, child);
            position = child;
        }
    }
}

fn find_min_position_scalar(ids: &[i32], distances: &[f32], length: usize) -> Option<usize> {
    let mut position = None;
    let mut best = f32::INFINITY;
    for candidate in (0..length).rev() {
        if ids[candidate] != EMPTY_NEIGHBOR && distances[candidate] < best {
            position = Some(candidate);
            best = distances[candidate];
        }
    }
    position
}

fn count_below_scalar(distances: &[f32], length: usize, threshold: f32) -> usize {
    let mut count = 0;
    for &candidate in &distances[..length] {
        count += usize::from(candidate < threshold);
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn find_min_position_avx512(ids: &[i32], distances: &[f32], length: usize) -> Option<usize> {
    use std::arch::x86_64::{
        _mm512_cmpge_epi32_mask, _mm512_loadu_ps, _mm512_loadu_si512, _mm512_mask_blend_ps,
        _mm512_reduce_min_ps, _mm512_set1_ps, _mm512_setzero_si512,
    };

    let mut best = f32::INFINITY;
    let mut position = None;
    let mut offset = 0;
    while offset + 16 <= length {
        let indices = unsafe { _mm512_loadu_si512(ids.as_ptr().add(offset).cast()) };
        let valid = _mm512_cmpge_epi32_mask(indices, _mm512_setzero_si512());
        let values = unsafe { _mm512_loadu_ps(distances.as_ptr().add(offset)) };
        let masked = _mm512_mask_blend_ps(valid, _mm512_set1_ps(f32::INFINITY), values);
        let chunk_best = _mm512_reduce_min_ps(masked);
        if chunk_best < best {
            best = chunk_best;
            for lane in (0..16).rev() {
                if ids[offset + lane] != EMPTY_NEIGHBOR
                    && distances[offset + lane].to_bits() == chunk_best.to_bits()
                {
                    position = Some(offset + lane);
                    break;
                }
            }
        }
        offset += 16;
    }
    for candidate in (offset..length).rev() {
        if ids[candidate] != EMPTY_NEIGHBOR && distances[candidate] < best {
            position = Some(candidate);
            best = distances[candidate];
        }
    }
    position
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn count_below_avx512(distances: &[f32], length: usize, threshold: f32) -> usize {
    use std::arch::x86_64::{_CMP_LT_OQ, _mm512_cmp_ps_mask, _mm512_loadu_ps, _mm512_set1_ps};

    let threshold_vector = _mm512_set1_ps(threshold);
    let mut count = 0;
    let mut offset = 0;
    while offset + 16 <= length {
        let values = unsafe { _mm512_loadu_ps(distances.as_ptr().add(offset)) };
        count += _mm512_cmp_ps_mask(values, threshold_vector, _CMP_LT_OQ).count_ones() as usize;
        offset += 16;
    }
    count + count_below_scalar(&distances[offset..], length - offset, threshold)
}

#[derive(Debug)]
pub(super) struct Graph {
    offsets: Vec<usize>,
    neighbors: Vec<i32>,
    entry: usize,
    max_level: usize,
    m: usize,
}

impl GraphAccess for Graph {
    fn neighbor_range(&self, cid: usize, layer: usize) -> (usize, usize) {
        neighbor_range(&self.offsets, self.m, cid, layer)
    }

    fn neighbor(&self, offset: usize) -> Option<usize> {
        usize::try_from(self.neighbors[offset]).ok()
    }
}

impl Graph {
    pub(super) fn build(
        centroids: &[f32],
        dimension: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
        parallel: &ParalliteContext,
    ) -> crate::Result<Self> {
        ConcurrentGraph::build(centroids, dimension, m, ef_construction, seed, parallel)
    }

    pub(super) fn search_append(
        &self,
        centroids: &LvqEncodedBatch,
        query: &[f32],
        count: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        output: &mut Vec<usize>,
    ) -> Result<(), KernelError> {
        let prepared = centroids.prepare_query(query)?;
        let entry = greedy_search_query(self, self.entry, self.max_level, &prepared)?;
        let results = search_layer_query(self, entry, count, ef, scratch, &prepared)?;
        output.extend(results.iter().take(count).map(|value| value.cid));
        Ok(())
    }

    pub(super) fn resident_size(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.neighbors.capacity() * std::mem::size_of::<i32>()
    }
}

#[derive(Debug, Clone, Copy)]
struct EntryPoint {
    cid: usize,
    level: usize,
}

#[derive(Debug)]
struct ConcurrentGraph {
    offsets: Vec<usize>,
    neighbors: Vec<AtomicI32>,
    locks: Vec<Mutex<()>>,
    entry: RwLock<EntryPoint>,
    m: usize,
}

impl GraphAccess for ConcurrentGraph {
    fn neighbor_range(&self, cid: usize, layer: usize) -> (usize, usize) {
        neighbor_range(&self.offsets, self.m, cid, layer)
    }

    fn neighbor(&self, offset: usize) -> Option<usize> {
        usize::try_from(self.neighbors[offset].load(AtomicOrdering::Acquire)).ok()
    }
}

impl ConcurrentGraph {
    fn build(
        centroids: &[f32],
        dimension: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
        parallel: &ParalliteContext,
    ) -> crate::Result<Graph> {
        let count = centroids.len() / dimension;
        if count > i32::MAX as usize {
            return Err(Error::InvalidArgument(
                "HNSW centroid count exceeds the INT32 graph domain".into(),
            ));
        }
        let mut rng = SmallRng::new(seed);
        let levels = (0..count).map(|_| rng.level(m)).collect::<Vec<_>>();
        let max_level = levels.iter().copied().max().unwrap_or(0);
        let entry_cid = levels
            .iter()
            .position(|level| *level == max_level)
            .unwrap_or(0);
        let offsets = graph_offsets(&levels, m)?;
        let graph = Self {
            neighbors: (0..offsets[count])
                .map(|_| AtomicI32::new(EMPTY_NEIGHBOR))
                .collect(),
            offsets,
            locks: (0..count).map(|_| Mutex::new(())).collect(),
            entry: RwLock::new(EntryPoint {
                cid: entry_cid,
                level: max_level,
            }),
            m,
        };
        let mut order_rng = SmallRng::new(seed ^ 0x3157_9bdf);
        for level in (0..=max_level).rev() {
            let mut bucket = levels
                .iter()
                .enumerate()
                .filter_map(|(cid, candidate_level)| (*candidate_level == level).then_some(cid))
                .filter(|cid| *cid != entry_cid)
                .collect::<Vec<_>>();
            order_rng.shuffle(&mut bucket);
            if bucket.is_empty() {
                continue;
            }
            let next = AtomicUsize::new(0);
            let workers = parallel.thread_count().min(bucket.len()).max(1);
            parallel
                .parallelize_n((0..workers).collect::<Vec<_>>(), workers)
                .map(|_| {
                    let mut scratch = SearchScratch::new(count);
                    loop {
                        let offset = next.fetch_add(1, AtomicOrdering::Relaxed);
                        let Some(&cid) = bucket.get(offset) else {
                            return Ok(());
                        };
                        graph.insert(
                            cid,
                            level,
                            centroids,
                            dimension,
                            ef_construction,
                            &mut scratch,
                        )?;
                    }
                })
                .collect_partitions()
                .map_err(|error| match error {
                    parallite::ParalliteError::User(error) => Error::Kernel(error),
                    other => Error::InvalidArgument(other.to_string()),
                })?;
        }

        let entry = *read_lock(&graph.entry);
        Ok(Graph {
            offsets: graph.offsets,
            neighbors: graph
                .neighbors
                .into_iter()
                .map(AtomicI32::into_inner)
                .collect(),
            entry: entry.cid,
            max_level: entry.level,
            m,
        })
    }

    fn insert(
        &self,
        cid: usize,
        level: usize,
        centroids: &[f32],
        dimension: usize,
        ef: usize,
        scratch: &mut SearchScratch,
    ) -> Result<(), KernelError> {
        let entry = *read_lock(&self.entry);
        let mut current = entry.cid;
        for layer in ((level + 1)..=entry.level).rev() {
            current = greedy_search(self, current, layer, |candidate| {
                Ok(squared_l2_between_rows(
                    centroids, dimension, cid, candidate,
                ))
            })?;
        }
        for layer in (0..=level.min(entry.level)).rev() {
            let candidates = search_layer(self, current, layer, ef, scratch, |candidate| {
                Ok(squared_l2_between_rows(
                    centroids, dimension, cid, candidate,
                ))
            })?;
            let selected =
                diversified_neighbors(candidates, self.layer_capacity(layer), |left, right| {
                    Ok(squared_l2_between_rows(centroids, dimension, left, right))
                })?;
            if let Some(nearest) = selected.first() {
                current = *nearest;
            }
            self.replace_neighbors(cid, layer, &selected);
            for candidate in selected {
                self.add_reciprocal_neighbor(candidate, cid, layer, centroids, dimension)?;
            }
        }
        if level > entry.level {
            let mut shared = write_lock(&self.entry);
            if level > shared.level {
                *shared = EntryPoint { cid, level };
            }
        }
        Ok(())
    }

    fn add_reciprocal_neighbor(
        &self,
        owner: usize,
        neighbor: usize,
        layer: usize,
        centroids: &[f32],
        dimension: usize,
    ) -> Result<(), KernelError> {
        let _guard = mutex_lock(&self.locks[owner]);
        let mut neighbors = self.neighbors_at(owner, layer);
        if neighbors.contains(&neighbor) {
            return Ok(());
        }
        neighbors.push(neighbor);
        let capacity = self.layer_capacity(layer);
        if neighbors.len() > capacity {
            let mut scored = neighbors
                .iter()
                .copied()
                .map(|cid| Scored {
                    cid,
                    distance: squared_l2_between_rows(centroids, dimension, owner, cid),
                })
                .collect::<Vec<_>>();
            scored.sort_unstable();
            neighbors = diversified_neighbors(&scored, capacity, |left, right| {
                Ok(squared_l2_between_rows(centroids, dimension, left, right))
            })?;
        }
        self.write_neighbors(owner, layer, &neighbors);
        Ok(())
    }

    fn replace_neighbors(&self, owner: usize, layer: usize, neighbors: &[usize]) {
        let _guard = mutex_lock(&self.locks[owner]);
        self.write_neighbors(owner, layer, neighbors);
    }

    fn neighbors_at(&self, cid: usize, layer: usize) -> Vec<usize> {
        let (begin, end) = self.neighbor_range(cid, layer);
        (begin..end)
            .map_while(|offset| self.neighbor(offset))
            .collect()
    }

    fn write_neighbors(&self, cid: usize, layer: usize, neighbors: &[usize]) {
        let (begin, end) = self.neighbor_range(cid, layer);
        debug_assert!(neighbors.len() <= end - begin);
        for (offset, slot) in (begin..end).enumerate() {
            let value =
                neighbors
                    .get(offset)
                    .map_or(EMPTY_NEIGHBOR, |cid| match i32::try_from(*cid) {
                        Ok(cid) => cid,
                        Err(_) => unreachable!("HNSW cid was validated before graph construction"),
                    });
            self.neighbors[slot].store(value, AtomicOrdering::Release);
        }
    }

    fn layer_capacity(&self, layer: usize) -> usize {
        if layer == 0 { self.m * 2 } else { self.m }
    }
}

fn graph_offsets(levels: &[usize], m: usize) -> crate::Result<Vec<usize>> {
    let mut offsets = Vec::with_capacity(levels.len() + 1);
    offsets.push(0_usize);
    for &level in levels {
        let slots = m
            .checked_mul(level.saturating_add(2))
            .ok_or_else(|| Error::InvalidArgument("HNSW graph size overflows usize".into()))?;
        offsets.push(
            offsets
                .last()
                .copied()
                .and_then(|offset| offset.checked_add(slots))
                .ok_or_else(|| Error::InvalidArgument("HNSW graph size overflows usize".into()))?,
        );
    }
    Ok(offsets)
}

fn neighbor_range(offsets: &[usize], m: usize, cid: usize, layer: usize) -> (usize, usize) {
    let begin = offsets[cid] + if layer == 0 { 0 } else { m * (layer + 1) };
    let capacity = if layer == 0 { m * 2 } else { m };
    (begin, begin + capacity)
}

fn diversified_neighbors<F>(
    candidates: &[Scored],
    capacity: usize,
    mut symmetric_distance: F,
) -> Result<Vec<usize>, KernelError>
where
    F: FnMut(usize, usize) -> Result<f32, KernelError>,
{
    let mut selected = Vec::with_capacity(capacity);
    for candidate in candidates {
        let mut keep = true;
        for &neighbor in &selected {
            if symmetric_distance(neighbor, candidate.cid)? < candidate.distance {
                keep = false;
                break;
            }
        }
        if keep {
            selected.push(candidate.cid);
            if selected.len() == capacity {
                break;
            }
        }
    }
    Ok(selected)
}

fn greedy_search<G, F>(
    graph: &G,
    mut current: usize,
    layer: usize,
    mut distance: F,
) -> Result<usize, KernelError>
where
    G: GraphAccess,
    F: FnMut(usize) -> Result<f32, KernelError>,
{
    let mut best = distance(current)?;
    loop {
        let mut changed = false;
        let (begin, end) = graph.neighbor_range(current, layer);
        for offset in begin..end {
            let Some(candidate) = graph.neighbor(offset) else {
                break;
            };
            let candidate_distance = distance(candidate)?;
            if candidate_distance < best {
                current = candidate;
                best = candidate_distance;
                changed = true;
            }
        }
        if !changed {
            return Ok(current);
        }
    }
}

fn dense_row(values: &[f32], dimension: usize, row: usize) -> &[f32] {
    &values[row * dimension..(row + 1) * dimension]
}

fn squared_l2_between_rows(values: &[f32], dimension: usize, left: usize, right: usize) -> f32 {
    detect().squared_l2(
        dense_row(values, dimension, left),
        dense_row(values, dimension, right),
    )
}

fn greedy_search_query(
    graph: &Graph,
    mut current: usize,
    layer: usize,
    query: &LvqRowQuery<'_>,
) -> Result<usize, KernelError> {
    let mut best = query.squared_l2(current)?;
    loop {
        let previous = current;
        let (begin, end) = graph.neighbor_range(current, layer);
        let mut buffered = [0_usize; 4];
        let mut count = 0;
        for offset in begin..end {
            let Some(candidate) = graph.neighbor(offset) else {
                break;
            };
            buffered[count] = candidate;
            count += 1;
            if count == 4 {
                for (candidate, distance) in
                    buffered.into_iter().zip(query.squared_l2_four(buffered)?)
                {
                    if distance < best {
                        current = candidate;
                        best = distance;
                    }
                }
                count = 0;
            }
        }
        for &candidate in &buffered[..count] {
            let distance = query.squared_l2(candidate)?;
            if distance < best {
                current = candidate;
                best = distance;
            }
        }
        if current == previous {
            return Ok(current);
        }
    }
}

fn search_layer<'a, G, F>(
    graph: &G,
    entry: usize,
    layer: usize,
    ef: usize,
    scratch: &'a mut SearchScratch,
    mut distance: F,
) -> Result<&'a [Scored], KernelError>
where
    G: GraphAccess,
    F: FnMut(usize) -> Result<f32, KernelError>,
{
    scratch.reset();
    let first = Scored {
        cid: entry,
        distance: distance(entry)?,
    };
    scratch.visit(entry);
    scratch.candidates.push(Reverse(first));
    scratch.nearest.push(first);
    while let Some(Reverse(current)) = scratch.candidates.pop() {
        if scratch.nearest.len() >= ef
            && scratch
                .nearest
                .peek()
                .is_some_and(|worst| current.distance > worst.distance)
        {
            break;
        }
        let (begin, end) = graph.neighbor_range(current.cid, layer);
        for offset in begin..end {
            let Some(candidate) = graph.neighbor(offset) else {
                break;
            };
            if !scratch.visit(candidate) {
                continue;
            }
            let scored = Scored {
                cid: candidate,
                distance: distance(candidate)?,
            };
            if scratch.nearest.len() < ef
                || scratch.nearest.peek().is_some_and(|worst| scored < *worst)
            {
                scratch.candidates.push(Reverse(scored));
                scratch.nearest.push(scored);
                if scratch.nearest.len() > ef {
                    scratch.nearest.pop();
                }
            }
        }
    }
    scratch.ordered.extend(scratch.nearest.iter().copied());
    scratch.ordered.sort_unstable();
    Ok(&scratch.ordered)
}

fn search_layer_query<'a>(
    graph: &Graph,
    entry: usize,
    result_count: usize,
    ef: usize,
    scratch: &'a mut SearchScratch,
    query: &LvqRowQuery<'_>,
) -> Result<&'a [Scored], KernelError> {
    scratch.reset_query(ef);
    let first = Scored {
        cid: entry,
        distance: query.squared_l2(entry)?,
    };
    #[cfg(test)]
    {
        scratch.distance_evaluations += 1;
    }
    scratch.visit(entry);
    scratch.query_candidates.push(first.cid, first.distance);
    scratch.query_results.push(first);
    while !scratch.query_candidates.is_empty() {
        let Some(current) = scratch.query_candidates.pop_min() else {
            break;
        };
        if scratch.query_candidates.count_below(current.distance) >= ef {
            break;
        }
        let (begin, end) = graph.neighbor_range(current.cid, 0);
        let mut buffered = [0_usize; 4];
        let mut count = 0;
        for offset in begin..end {
            let Some(candidate) = graph.neighbor(offset) else {
                break;
            };
            if !scratch.visit(candidate) {
                continue;
            }
            buffered[count] = candidate;
            count += 1;
            if count == 4 {
                let distances = query.squared_l2_four(buffered)?;
                #[cfg(test)]
                {
                    scratch.distance_evaluations += 4;
                }
                for (candidate, distance) in buffered.into_iter().zip(distances) {
                    consider_query_candidate(
                        scratch,
                        Scored {
                            cid: candidate,
                            distance,
                        },
                        result_count,
                    );
                }
                count = 0;
            }
        }
        for &candidate in &buffered[..count] {
            #[cfg(test)]
            {
                scratch.distance_evaluations += 1;
            }
            consider_query_candidate(
                scratch,
                Scored {
                    cid: candidate,
                    distance: query.squared_l2(candidate)?,
                },
                result_count,
            );
        }
    }
    scratch
        .ordered
        .extend(scratch.query_results.iter().copied());
    scratch.ordered.sort_unstable();
    Ok(&scratch.ordered)
}

fn consider_query_candidate(scratch: &mut SearchScratch, scored: Scored, result_count: usize) {
    if scratch.query_results.len() < result_count
        || scratch
            .query_results
            .peek()
            .is_some_and(|worst| scored < *worst)
    {
        scratch.query_results.push(scored);
        if scratch.query_results.len() > result_count {
            scratch.query_results.pop();
        }
    }
    scratch.query_candidates.push(scored.cid, scored.distance);
}

fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn mutex_lock<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    lock.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug)]
struct SmallRng(u64);

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn level(&mut self, m: usize) -> usize {
        let mut level = 0;
        while level < MAX_LEVEL && self.next().is_multiple_of(m as u64) {
            level += 1;
        }
        level
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in 0..values.len().saturating_sub(1) {
            let remaining = values.len() - index;
            let Ok(remaining) = u64::try_from(remaining) else {
                unreachable!("usize is wider than the HNSW random-number domain")
            };
            let Ok(offset) = usize::try_from(self.next() % remaining) else {
                unreachable!("random offset is bounded by usize")
            };
            let selected = index + offset;
            values.swap(index, selected);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_heap_keeps_the_nearest_capacity() {
        let mut heap = MinimaxHeap::default();
        heap.reset(3);
        heap.push(0, 5.0);
        heap.push(1, 1.0);
        heap.push(2, 3.0);
        heap.push(3, 2.0);
        heap.push(4, 8.0);

        assert_eq!(heap.pop_min().unwrap().cid, 1);
        assert_eq!(heap.pop_min().unwrap().cid, 3);
        assert_eq!(heap.pop_min().unwrap().cid, 2);
        assert!(heap.pop_min().is_none());
    }

    #[test]
    fn graph_offsets_cover_every_declared_layer() {
        let offsets = graph_offsets(&[0, 1, 3], 4).unwrap();

        assert_eq!(offsets, [0, 8, 20, 40]);
        assert_eq!(neighbor_range(&offsets, 4, 0, 0), (0, 8));
        assert_eq!(neighbor_range(&offsets, 4, 1, 1), (16, 20));
        assert_eq!(neighbor_range(&offsets, 4, 2, 3), (36, 40));
    }
}
