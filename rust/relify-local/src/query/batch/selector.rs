use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, UInt32Array};
use arrow::compute::interleave;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::utils::memory::get_record_batch_memory_size;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::memory_pool::MemoryReservation;

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

/// Maintains one bounded Top-K heap per query while sharing retained Arrow batches.
#[derive(Debug)]
pub(crate) struct BatchCandidateSelector {
    fetch: usize,
    heaps: Vec<BinaryHeap<Candidate>>,
    next_batch_id: u64,
    next_sequence: u64,
    retained_columns: Vec<usize>,
    batches: HashMap<u64, RetainedBatch>,
    selection_bytes: usize,
    retained_bytes: usize,
    reservation: MemoryReservation,
}

impl BatchCandidateSelector {
    pub(crate) fn try_new(
        query_count: usize,
        fetch: usize,
        retained_columns: Vec<usize>,
        reservation: MemoryReservation,
    ) -> DataFusionResult<Self> {
        if query_count == 0 || fetch == 0 {
            return Err(DataFusionError::Plan(
                "batch candidate selection requires positive query and fetch counts".into(),
            ));
        }
        let candidate_count = query_count.checked_mul(fetch).ok_or_else(|| {
            DataFusionError::ResourcesExhausted(
                "batch candidate heap capacity overflows usize".into(),
            )
        })?;
        let selection_bytes = candidate_count
            .checked_mul(std::mem::size_of::<Candidate>())
            .and_then(|bytes| {
                query_count
                    .checked_mul(std::mem::size_of::<BinaryHeap<Candidate>>())
                    .and_then(|heaps| bytes.checked_add(heaps))
            })
            .ok_or_else(|| {
                DataFusionError::ResourcesExhausted(
                    "batch candidate memory reservation overflows usize".into(),
                )
            })?;
        reservation.try_resize(selection_bytes)?;
        Ok(Self {
            fetch,
            heaps: (0..query_count)
                .map(|_| BinaryHeap::with_capacity(fetch))
                .collect(),
            next_batch_id: 0,
            next_sequence: 0,
            retained_columns,
            batches: HashMap::new(),
            selection_bytes,
            retained_bytes: 0,
            reservation,
        })
    }

    /// Retains only output columns for one input batch and returns its stable ID.
    pub(crate) fn begin_batch(&mut self, batch: &RecordBatch) -> DataFusionResult<u64> {
        let batch_id = self.next_batch_id;
        self.next_batch_id = self.next_batch_id.wrapping_add(1);
        let batch = batch.project(&self.retained_columns)?;
        let bytes = get_record_batch_memory_size(&batch);
        self.reservation.try_resize(
            self.selection_bytes
                .saturating_add(self.retained_bytes.saturating_add(bytes)),
        )?;
        self.retained_bytes += bytes;
        self.batches.insert(
            batch_id,
            RetainedBatch {
                batch,
                uses: 0,
                bytes,
            },
        );
        Ok(batch_id)
    }

    pub(crate) fn update(
        &mut self,
        batch_id: u64,
        query_ordinal: usize,
        row_offset: usize,
        distances: &[f32],
    ) -> DataFusionResult<()> {
        if self.heaps.get(query_ordinal).is_none() {
            return Err(DataFusionError::Internal(
                "batch candidate has an invalid query ordinal".into(),
            ));
        }
        let batch_rows = self
            .batches
            .get(&batch_id)
            .ok_or_else(|| {
                DataFusionError::Internal("batch candidate input was not retained".into())
            })?
            .batch
            .num_rows();
        if row_offset.saturating_add(distances.len()) > batch_rows {
            return Err(DataFusionError::Internal(
                "batch candidate row range exceeds its input batch".into(),
            ));
        }

        for (row, distance) in distances.iter().copied().enumerate() {
            if !distance.is_finite() {
                return Err(DataFusionError::Execution(
                    "batch search produced a non-finite distance".into(),
                ));
            }
            let discarded = {
                let heap = &mut self.heaps[query_ordinal];
                if heap.len() == self.fetch
                    && distance.total_cmp(&heap.peek().expect("full heap").distance)
                        != Ordering::Less
                {
                    continue;
                }
                (heap.len() == self.fetch).then(|| heap.pop().expect("full heap"))
            };
            if let Some(discarded) = discarded {
                if discarded.batch_id == batch_id {
                    self.batches
                        .get_mut(&batch_id)
                        .expect("current batch is retained")
                        .uses -= 1;
                } else {
                    self.release(discarded.batch_id);
                }
            }
            self.heaps[query_ordinal].push(Candidate {
                distance,
                sequence: self.next_sequence,
                batch_id,
                row: row_offset + row,
            });
            self.next_sequence = self.next_sequence.wrapping_add(1);
            self.batches
                .get_mut(&batch_id)
                .expect("current batch is retained")
                .uses += 1;
        }
        Ok(())
    }

    /// Drops a batch immediately when none of its rows survived any query heap.
    pub(crate) fn end_batch(&mut self, batch_id: u64) {
        if self
            .batches
            .get(&batch_id)
            .is_some_and(|batch| batch.uses == 0)
        {
            self.remove_batch(batch_id);
        }
    }

    pub(crate) fn retained_batch_count(&self) -> usize {
        self.batches.len()
    }

    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    fn release(&mut self, batch_id: u64) {
        let batch = self
            .batches
            .get_mut(&batch_id)
            .expect("candidate refers to a retained batch");
        batch.uses -= 1;
        if batch.uses == 0 {
            self.remove_batch(batch_id);
        }
    }

    fn remove_batch(&mut self, batch_id: u64) {
        let batch = self
            .batches
            .remove(&batch_id)
            .expect("retained batch exists");
        self.retained_bytes -= batch.bytes;
        self.reservation.shrink(batch.bytes);
    }

    pub(crate) fn finish(mut self, output_schema: SchemaRef) -> DataFusionResult<RecordBatch> {
        let mut selected = Vec::new();
        for (query_ordinal, heap) in self.heaps.drain(..).enumerate() {
            selected.extend(
                heap.into_sorted_vec()
                    .into_iter()
                    .map(|candidate| (query_ordinal, candidate)),
            );
        }
        if selected.is_empty() {
            return Ok(RecordBatch::new_empty(output_schema));
        }

        let mut batch_ids = self.batches.keys().copied().collect::<Vec<_>>();
        batch_ids.sort_unstable();
        let batch_positions = batch_ids
            .iter()
            .enumerate()
            .map(|(position, batch_id)| (*batch_id, position))
            .collect::<HashMap<_, _>>();
        let input_batches = batch_ids
            .iter()
            .map(|batch_id| &self.batches[batch_id].batch)
            .collect::<Vec<_>>();
        let indices = selected
            .iter()
            .map(|(_, candidate)| (batch_positions[&candidate.batch_id], candidate.row))
            .collect::<Vec<_>>();

        let mut columns = Vec::<ArrayRef>::with_capacity(output_schema.fields().len());
        let query_ids = selected
            .iter()
            .map(|(query_ordinal, _)| {
                u32::try_from(*query_ordinal).map_err(|_| {
                    DataFusionError::Execution("batch query count exceeds UInt32".into())
                })
            })
            .collect::<DataFusionResult<Vec<_>>>()?;
        columns.push(Arc::new(UInt32Array::from(query_ids)));
        for retained_index in 0..self.retained_columns.len() {
            let arrays = input_batches
                .iter()
                .map(|batch| batch.column(retained_index).as_ref())
                .collect::<Vec<_>>();
            columns.push(interleave(&arrays, &indices)?);
        }
        columns.push(Arc::new(Float32Array::from_iter_values(
            selected.iter().map(|(_, candidate)| candidate.distance),
        )));
        Ok(RecordBatch::try_new(output_schema, columns)?)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

    use super::*;

    fn input(ids: &[&str]) -> RecordBatch {
        let row_count = i32::try_from(ids.len()).unwrap();
        RecordBatch::try_from_iter([
            ("id", Arc::new(StringArray::from(ids.to_vec())) as ArrayRef),
            (
                "ignored",
                Arc::new(Int32Array::from_iter_values(0..row_count)) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("_query_id", DataType::UInt32, false),
            Field::new("id", DataType::Utf8, false),
            Field::new("_distance", DataType::Float32, false),
        ]))
    }

    #[test]
    fn shares_batches_across_query_heaps_and_returns_per_query_top_k() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let reservation = MemoryConsumer::new("batch-selector").register(&pool);
        let mut selector = BatchCandidateSelector::try_new(2, 2, vec![0], reservation).unwrap();
        let first = selector.begin_batch(&input(&["a", "b", "c"])).unwrap();
        selector.update(first, 0, 0, &[3.0, 1.0, 2.0]).unwrap();
        selector.update(first, 1, 0, &[1.0, 3.0, 2.0]).unwrap();
        selector.end_batch(first);
        let second = selector.begin_batch(&input(&["d"])).unwrap();
        selector.update(second, 0, 0, &[0.5]).unwrap();
        selector.update(second, 1, 0, &[4.0]).unwrap();
        selector.end_batch(second);

        let result = selector.finish(output_schema()).unwrap();
        let query_ids = result
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let ids = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let distances = result
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        assert_eq!(query_ids.values(), &[0, 0, 1, 1]);
        assert_eq!(
            (0..ids.len()).map(|row| ids.value(row)).collect::<Vec<_>>(),
            ["d", "b", "a", "c"]
        );
        assert_eq!(distances.values(), &[0.5, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn reserves_candidate_heaps_before_allocating_them() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1));
        let reservation = MemoryConsumer::new("batch-selector").register(&pool);

        assert!(BatchCandidateSelector::try_new(2, 2, vec![0], reservation).is_err());
    }
}
