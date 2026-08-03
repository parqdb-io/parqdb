//! Batch candidate selection and retained-row projection.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, LargeListArray, ListArray};
use arrow::compute::{interleave, interleave_record_batch};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::Result as DataFusionResult;
use datafusion::common::utils::memory::get_record_batch_memory_size;
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionExpr;
use datafusion::physical_plan::metrics::Time;

use super::exec::VectorKind;
use relify_kernels::DistanceKernel;

#[derive(Debug)]
struct Candidate {
    distance: f32,
    sequence: u64,
    batch_id: u64,
    row: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.sequence == other.sequence
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

#[derive(Debug)]
struct RetainedBatch {
    batch: RecordBatch,
    uses: usize,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct CandidateSelector {
    fetch: usize,
    compaction_limit: usize,
    next_batch_id: u64,
    next_sequence: u64,
    threshold: Option<f32>,
    heap: Option<BinaryHeap<Candidate>>,
    candidates: Vec<Candidate>,
    retained_input_columns: Option<Vec<usize>>,
    batches: HashMap<u64, RetainedBatch>,
    retained_bytes: usize,
    reservation: MemoryReservation,
}

impl CandidateSelector {
    pub(super) fn new(
        fetch: usize,
        retained_input_columns: Option<Vec<usize>>,
        reservation: MemoryReservation,
    ) -> Self {
        // A binary heap wins while its random-access working set remains
        // cache-resident. Beyond that, amortized batch selection avoids the
        // per-candidate log(K) and branch cost.
        let use_heap = fetch.saturating_mul(std::mem::size_of::<Candidate>()) <= 32 * 1_024;
        let slack = (fetch / 2).clamp(1_024, 16_384);
        Self {
            fetch,
            compaction_limit: fetch.saturating_add(slack),
            next_batch_id: 0,
            next_sequence: 0,
            threshold: None,
            heap: use_heap.then(|| BinaryHeap::with_capacity(fetch)),
            candidates: Vec::with_capacity(fetch),
            retained_input_columns,
            batches: HashMap::new(),
            retained_bytes: 0,
            reservation,
        }
    }

    #[inline]
    pub(super) fn threshold(&self) -> Option<f32> {
        self.heap
            .as_ref()
            .and_then(|heap| (heap.len() == self.fetch).then(|| heap.peek().unwrap().distance))
            .or(self.threshold)
    }

    fn begin_batch(&mut self) -> u64 {
        let id = self.next_batch_id;
        self.next_batch_id = self.next_batch_id.wrapping_add(1);
        id
    }

    pub(super) fn retained_batch_count(&self) -> usize {
        self.batches.len()
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(super) fn update_batch(
        &mut self,
        batch: RecordBatch,
        distances: &[f32],
        dynamic_limit: f32,
    ) -> DataFusionResult<SelectionStats> {
        if self.heap.is_some() {
            return self.update_heap_batch(batch, distances, dynamic_limit);
        }
        self.update_selection_batch(batch, distances, dynamic_limit)
    }

    fn update_heap_batch(
        &mut self,
        batch: RecordBatch,
        distances: &[f32],
        dynamic_limit: f32,
    ) -> DataFusionResult<SelectionStats> {
        let batch_id = self.begin_batch();
        let mut heap = self.heap.take().expect("heap selection mode");
        let mut current_batch_uses = 0;
        let mut stats = SelectionStats::default();
        for (row, distance) in distances.iter().copied().enumerate() {
            if distance.partial_cmp(&dynamic_limit) != Some(Ordering::Less) {
                stats.pruned += 1;
                continue;
            }
            stats.candidates += 1;
            if heap.len() == self.fetch
                && distance.partial_cmp(&heap.peek().expect("full heap").distance)
                    != Some(Ordering::Less)
            {
                stats.discarded += 1;
                continue;
            }
            if heap.len() == self.fetch {
                let discarded = heap.pop().expect("full heap");
                stats.discarded += 1;
                if discarded.batch_id == batch_id {
                    current_batch_uses -= 1;
                } else {
                    self.release(discarded.batch_id);
                }
            }
            heap.push(Candidate {
                distance,
                sequence: self.next_sequence,
                batch_id,
                row,
            });
            self.next_sequence = self.next_sequence.wrapping_add(1);
            current_batch_uses += 1;
        }
        stats.passes = usize::from(stats.candidates > 0);
        self.heap = Some(heap);
        self.retain_batch(batch_id, batch, current_batch_uses)?;
        Ok(stats)
    }

    fn update_selection_batch(
        &mut self,
        batch: RecordBatch,
        distances: &[f32],
        dynamic_limit: f32,
    ) -> DataFusionResult<SelectionStats> {
        let batch_id = self.begin_batch();
        let mut stats = SelectionStats::default();
        self.candidates.reserve(distances.len());
        for (row, distance) in distances.iter().copied().enumerate() {
            if distance.partial_cmp(&dynamic_limit) != Some(Ordering::Less) {
                stats.pruned += 1;
                continue;
            }
            stats.candidates += 1;
            self.candidates.push(Candidate {
                distance,
                sequence: self.next_sequence,
                batch_id,
                row,
            });
            self.next_sequence = self.next_sequence.wrapping_add(1);
        }

        let mut current_batch_uses = stats.candidates;
        if self.candidates.len() > self.fetch
            && (self.threshold.is_none() || self.candidates.len() >= self.compaction_limit)
        {
            let (discarded, discarded_from_current) = self.compact(Some(batch_id));
            stats.discarded = discarded;
            stats.passes = 1;
            current_batch_uses -= discarded_from_current;
        } else if self.candidates.len() == self.fetch && self.threshold.is_none() {
            self.threshold = self
                .candidates
                .iter()
                .max()
                .map(|candidate| candidate.distance);
        }

        self.retain_batch(batch_id, batch, current_batch_uses)?;
        Ok(stats)
    }

    pub(super) fn compact_retained(&mut self) -> SelectionStats {
        if self.heap.is_some() {
            return SelectionStats::default();
        }
        if self.candidates.len() <= self.fetch {
            return SelectionStats::default();
        }
        let (discarded, discarded_from_unretained) = self.compact(None);
        debug_assert_eq!(discarded_from_unretained, 0);
        SelectionStats {
            discarded,
            passes: 1,
            ..SelectionStats::default()
        }
    }

    fn compact(&mut self, unretained_batch_id: Option<u64>) -> (usize, usize) {
        debug_assert!(self.candidates.len() > self.fetch);
        self.candidates.select_nth_unstable(self.fetch - 1);
        self.threshold = Some(self.candidates[self.fetch - 1].distance);
        let discarded = self.candidates.split_off(self.fetch);
        let discarded_count = discarded.len();
        let mut discarded_from_unretained = 0;
        for candidate in discarded {
            if Some(candidate.batch_id) == unretained_batch_id {
                discarded_from_unretained += 1;
            } else {
                self.release(candidate.batch_id);
            }
        }
        (discarded_count, discarded_from_unretained)
    }

    fn retain_batch(
        &mut self,
        batch_id: u64,
        batch: RecordBatch,
        uses: usize,
    ) -> DataFusionResult<()> {
        if uses == 0 {
            return Ok(());
        }
        let batch = if let Some(columns) = &self.retained_input_columns {
            batch.project(columns)?
        } else {
            batch
        };
        let bytes = get_record_batch_memory_size(&batch);
        self.retained_bytes += bytes;
        self.batches
            .insert(batch_id, RetainedBatch { batch, uses, bytes });
        self.reservation.try_resize(self.retained_bytes)
    }

    fn release(&mut self, batch_id: u64) {
        let entry = self
            .batches
            .get_mut(&batch_id)
            .expect("candidate refers to a retained batch");
        entry.uses -= 1;
        if entry.uses == 0 {
            let entry = self
                .batches
                .remove(&batch_id)
                .expect("retained batch exists");
            self.retained_bytes -= entry.bytes;
        }
    }

    pub(super) fn finish(
        mut self,
        projection: &[ProjectionExpr],
        distance_index: usize,
        output_schema: SchemaRef,
        candidate_sort_compute: &Time,
        projection_compute: &Time,
    ) -> DataFusionResult<RecordBatch> {
        if self.candidates.is_empty() && self.heap.as_ref().is_none_or(BinaryHeap::is_empty) {
            return Ok(RecordBatch::new_empty(output_schema));
        }
        let sort_timer = candidate_sort_compute.timer();
        let candidates = if let Some(heap) = self.heap.take() {
            heap.into_sorted_vec()
        } else {
            let mut candidates = std::mem::take(&mut self.candidates);
            candidates.sort_unstable();
            candidates
        };
        sort_timer.done();
        let projection_timer = projection_compute.timer();
        let mut batch_ids = self.batches.keys().copied().collect::<Vec<_>>();
        batch_ids.sort_unstable();
        let input_batches = batch_ids
            .iter()
            .map(|id| &self.batches[id].batch)
            .collect::<Vec<_>>();
        let positions = batch_ids
            .iter()
            .enumerate()
            .map(|(position, id)| (*id, position))
            .collect::<HashMap<_, _>>();
        let indices = candidates
            .iter()
            .map(|candidate| (positions[&candidate.batch_id], candidate.row))
            .collect::<Vec<_>>();
        let distances: ArrayRef = Arc::new(Float32Array::from_iter_values(
            candidates.iter().map(|candidate| candidate.distance),
        ));
        let mut selected = None;
        let mut columns = Vec::with_capacity(projection.len());
        for (index, expression) in projection.iter().enumerate() {
            if index == distance_index {
                columns.push(Arc::clone(&distances));
                continue;
            }
            if let Some(column) = expression.expr.downcast_ref::<Column>() {
                let column_index = self.retained_input_columns.as_ref().map_or_else(
                    || column.index(),
                    |columns| {
                        columns
                            .iter()
                            .position(|index| *index == column.index())
                            .expect("direct projection column is retained")
                    },
                );
                let arrays = input_batches
                    .iter()
                    .map(|batch| batch.column(column_index).as_ref())
                    .collect::<Vec<_>>();
                columns.push(interleave(&arrays, &indices)?);
                continue;
            }
            if selected.is_none() {
                selected = Some(interleave_record_batch(&input_batches, &indices)?);
            }
            let selected = selected.as_ref().expect("selected batch initialized");
            columns.push(
                expression
                    .expr
                    .evaluate(selected)?
                    .into_array(selected.num_rows())?,
            );
        }
        let output = RecordBatch::try_new(output_schema, columns)?;
        projection_timer.done();
        Ok(output)
    }
}

#[derive(Debug, Default)]
pub(super) struct SelectionStats {
    pub(super) candidates: usize,
    pub(super) pruned: usize,
    pub(super) discarded: usize,
    pub(super) passes: usize,
}

pub(super) fn retained_input_columns(
    projection: &[ProjectionExpr],
    distance_index: usize,
) -> Option<Vec<usize>> {
    let mut retained = Vec::new();
    for (index, expression) in projection.iter().enumerate() {
        if index == distance_index {
            continue;
        }
        let column = expression.expr.downcast_ref::<Column>()?;
        if !retained.contains(&column.index()) {
            retained.push(column.index());
        }
    }
    Some(retained)
}

// Arrow guarantees non-negative in-bounds offsets for a valid array. The
// operator contract also guarantees that every vector has `query.len()`
// values, so the hot loop deliberately avoids checked integer conversions.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn compute_batch_distances(
    distances: &mut Vec<f32>,
    batch: &RecordBatch,
    vector_index: usize,
    vector_kind: VectorKind,
    query: &[f32],
    kernel: DistanceKernel,
) {
    distances.resize(batch.num_rows(), 0.0);
    if batch.num_rows() == 0 {
        return;
    }
    match vector_kind {
        VectorKind::List => {
            let vectors = batch
                .column(vector_index)
                .as_any()
                .downcast_ref::<ListArray>()
                .expect("validated List<Float32> vector column");
            let offsets = vectors.value_offsets();
            let values = vectors
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("validated Float32 vector values");
            let start = offsets[0] as usize;
            let end = start + batch.num_rows() * query.len();
            debug_assert_eq!(offsets[batch.num_rows()] as usize, end);
            kernel.squared_l2_rows(&values.values()[start..end], query, distances);
        }
        VectorKind::LargeList => {
            let vectors = batch
                .column(vector_index)
                .as_any()
                .downcast_ref::<LargeListArray>()
                .expect("validated LargeList<Float32> vector column");
            let offsets = vectors.value_offsets();
            let values = vectors
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("validated Float32 vector values");
            let start = offsets[0] as usize;
            let end = start + batch.num_rows() * query.len();
            debug_assert_eq!(offsets[batch.num_rows()] as usize, end);
            kernel.squared_l2_rows(&values.values()[start..end], query, distances);
        }
        VectorKind::FixedSizeList => {
            let vectors = batch
                .column(vector_index)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("validated FixedSizeList<Float32> vector column");
            let values = vectors
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("validated Float32 vector values");
            let start = vectors.value_offset(0) as usize;
            let end = start + batch.num_rows() * query.len();
            kernel.squared_l2_rows(&values.values()[start..end], query, distances);
        }
    }
}
