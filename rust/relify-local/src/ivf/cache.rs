//! Decoded IVF postings cache and cluster directory.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::compute::BatchCoalescer;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::dataframe::DataFrame;
use futures::StreamExt;

use crate::{Error, Result};

pub(crate) struct CachedIvfPostings {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    fragments: BTreeMap<i32, Vec<ClusterFragment>>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ClusterFragment {
    pub(super) batch: usize,
    pub(super) start: usize,
    pub(super) length: usize,
}

impl CachedIvfPostings {
    pub(crate) async fn load(
        dataframe: DataFrame,
        batch_size: usize,
    ) -> Result<(Arc<dyn TableProvider>, usize)> {
        let schema = Arc::clone(dataframe.schema().inner());
        let streams = dataframe.execute_stream_partitioned().await?;
        let partitions =
            coalesce_postings_partitions(streams, Arc::clone(&schema), batch_size).await?;
        let resident_bytes = partitions
            .iter()
            .flatten()
            .map(RecordBatch::get_array_memory_size)
            .sum();
        let postings = Arc::new(Self::from_partitions(&partitions, batch_size)?);
        Ok((postings.provider(), resident_bytes))
    }

    pub(super) fn from_partitions(
        partitions: &[Vec<RecordBatch>],
        batch_size: usize,
    ) -> Result<Self> {
        let Some(first) = partitions.iter().flatten().next() else {
            return Err(Error::InvalidMetadata(
                "cached IVF postings relation is empty".into(),
            ));
        };
        let schema = first.schema();
        let cid_index = schema
            .index_of("cid")
            .map_err(|_| Error::InvalidMetadata("IVF postings is missing cid".into()))?;
        let mut batches = Vec::new();
        let mut fragments = BTreeMap::<i32, Vec<ClusterFragment>>::new();
        for batch in partitions.iter().flatten() {
            if batch.num_rows() > batch_size {
                return Err(Error::InvalidMetadata(format!(
                    "cached IVF postings batch has {} rows, exceeding DataFusion batch size {batch_size}",
                    batch.num_rows()
                )));
            }
            let cids = batch
                .column(cid_index)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::InvalidMetadata("IVF postings cid must be int".into()))?;
            if cids.null_count() != 0 {
                return Err(Error::InvalidMetadata(
                    "IVF postings cid must be required".into(),
                ));
            }
            let values = cids.values();
            if values.windows(2).any(|pair| pair[0] > pair[1]) {
                return Err(Error::InvalidMetadata(
                    "cached IVF postings must be ordered by cid".into(),
                ));
            }
            let batch_index = batches.len();
            let mut start = 0;
            while start < values.len() {
                let cid = values[start];
                let length = values[start..].partition_point(|candidate| *candidate == cid);
                fragments.entry(cid).or_default().push(ClusterFragment {
                    batch: batch_index,
                    start,
                    length,
                });
                start += length;
            }
            batches.push(batch.clone());
        }
        Ok(Self {
            schema,
            batches,
            fragments,
        })
    }

    pub(super) fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub(super) fn fragments(&self, cid: i32) -> Option<&[ClusterFragment]> {
        self.fragments.get(&cid).map(Vec::as_slice)
    }

    pub(super) fn cluster_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.fragments.keys().copied()
    }

    pub(super) fn batch(&self, index: usize) -> &RecordBatch {
        &self.batches[index]
    }
}

async fn coalesce_postings_partition(
    mut input: datafusion::physical_plan::SendableRecordBatchStream,
    schema: SchemaRef,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let cid_index = schema
        .index_of("cid")
        .map_err(|_| Error::InvalidMetadata("IVF postings is missing cid".into()))?;
    let mut coalescer = BatchCoalescer::new(schema, batch_size);
    let mut output = Vec::new();
    let mut last_cid = None;
    while let Some(batch) = input.next().await {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let cids = batch
            .column(cid_index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| Error::InvalidMetadata("IVF postings cid must be int".into()))?;
        if cids.null_count() != 0 {
            return Err(Error::InvalidMetadata(
                "IVF postings cid must be required".into(),
            ));
        }
        let values = cids.values();
        if values.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(Error::InvalidMetadata(
                "cached IVF postings must be ordered by cid".into(),
            ));
        }
        if last_cid.is_some_and(|previous| previous > values[0]) {
            coalescer.finish_buffered_batch()?;
            drain_completed_batches(&mut coalescer, &mut output);
        }
        last_cid = values.last().copied();
        coalescer.push_batch(batch)?;
        drain_completed_batches(&mut coalescer, &mut output);
    }
    coalescer.finish_buffered_batch()?;
    drain_completed_batches(&mut coalescer, &mut output);
    debug_assert!(output.iter().all(|batch| batch.num_rows() <= batch_size));
    Ok(output)
}

async fn coalesce_postings_partitions(
    streams: Vec<datafusion::physical_plan::SendableRecordBatchStream>,
    schema: SchemaRef,
    batch_size: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    let mut tasks = tokio::task::JoinSet::new();
    for (partition, stream) in streams.into_iter().enumerate() {
        let schema = Arc::clone(&schema);
        tasks.spawn(async move {
            (
                partition,
                coalesce_postings_partition(stream, schema, batch_size).await,
            )
        });
    }
    let mut partitions = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok((partition, batches)) => partitions.push((partition, batches?)),
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(error) => {
                return Err(Error::InvalidArgument(format!(
                    "cached IVF postings task failed: {error}"
                )));
            }
        }
    }
    partitions.sort_unstable_by_key(|(partition, _)| *partition);
    Ok(partitions.into_iter().map(|(_, batches)| batches).collect())
}

fn drain_completed_batches(coalescer: &mut BatchCoalescer, output: &mut Vec<RecordBatch>) {
    while let Some(batch) = coalescer.next_completed_batch() {
        output.push(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ArrayRef;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

    fn cid_batch(values: Vec<i32>) -> RecordBatch {
        RecordBatch::try_from_iter([("cid", Arc::new(Int32Array::from(values)) as ArrayRef)])
            .unwrap()
    }

    fn batch_stream(
        batches: Vec<RecordBatch>,
    ) -> datafusion::physical_plan::SendableRecordBatchStream {
        let schema = batches[0].schema();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    #[tokio::test]
    async fn postings_coalescer_enforces_the_datafusion_batch_size() {
        let batches = vec![
            cid_batch(vec![0, 0, 1]),
            cid_batch(vec![1, 2, 2, 3]),
            cid_batch(vec![3, 4, 4, 5, 5]),
        ];
        let schema = batches[0].schema();

        let output = coalesce_postings_partition(batch_stream(batches), schema, 8)
            .await
            .unwrap();

        assert_eq!(
            output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            [8, 4]
        );
    }

    #[tokio::test]
    async fn postings_coalescer_flushes_before_a_cid_order_restart() {
        let batches = vec![cid_batch(vec![4, 5]), cid_batch(vec![0, 1])];
        let schema = batches[0].schema();

        let output = coalesce_postings_partition(batch_stream(batches), schema, 8)
            .await
            .unwrap();

        assert_eq!(
            output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            [2, 2]
        );
    }
}
