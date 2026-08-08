//! Managed Parquet relation I/O for the embedded backend.

use std::io::Cursor;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, Int32Array};
#[cfg(test)]
use arrow::compute::concat_batches;
use arrow::datatypes::DataType;
#[cfg(test)]
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use datafusion::dataframe::DataFrame;
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr_common::sort_expr::PhysicalSortExpr;
use datafusion::physical_plan::execute_stream_partitioned;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{Partitioning as PhysicalPartitioning, SendableRecordBatchStream};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
#[cfg(test)]
use datafusion::prelude::{col, lit};
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectStore, PutMode};
use parquet::arrow::async_writer::ParquetObjectWriter;
use parquet::arrow::{ArrowWriter, AsyncArrowWriter};
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use relify_storage::StorageRegistry;
use url::Url;

use crate::{Error, Result};

/// Physical options used when writing Parquet index tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetWriterOptions {
    /// Parquet compression codec.
    pub compression: String,
    /// Maximum rows in one Parquet row group, or automatic sizing for IVF postings.
    pub max_row_group_rows: Option<usize>,
    /// Approximate target size of one output file in bytes.
    pub target_file_size: usize,
    /// Number of rows passed to the Parquet encoder per write batch.
    pub write_batch_rows: usize,
}

impl Default for ParquetWriterOptions {
    fn default() -> Self {
        Self {
            compression: "uncompressed".into(),
            max_row_group_rows: None,
            target_file_size: 512 * 1024 * 1024,
            write_batch_rows: 8_192,
        }
    }
}

impl ParquetWriterOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if Compression::from_str(&self.compression).is_err() {
            return Err(Error::InvalidArgument(format!(
                "unsupported Parquet compression: {}",
                self.compression
            )));
        }
        if self.max_row_group_rows == Some(0)
            || self.target_file_size == 0
            || self.write_batch_rows == 0
        {
            return Err(Error::InvalidArgument(
                "Parquet writer sizes must be positive".into(),
            ));
        }
        Ok(())
    }

    fn writer_properties(&self) -> Result<WriterProperties> {
        self.writer_properties_for(None)
    }

    fn postings_writer_properties(&self, vector_path: ColumnPath) -> Result<WriterProperties> {
        self.writer_properties_for(Some(vector_path))
    }

    fn writer_properties_for(
        &self,
        dictionary_disabled_column: Option<ColumnPath>,
    ) -> Result<WriterProperties> {
        self.validate()?;
        let compression = Compression::from_str(&self.compression).map_err(|error| {
            Error::InvalidArgument(format!("unsupported Parquet compression: {error}"))
        })?;
        let mut properties = WriterProperties::builder()
            .set_compression(compression)
            .set_write_batch_size(self.write_batch_rows)
            .set_statistics_enabled(EnabledStatistics::Page);
        if let Some(max_rows) = self.max_row_group_rows {
            properties = properties.set_max_row_group_row_count(Some(max_rows));
        }
        if let Some(column) = dictionary_disabled_column {
            properties = properties.set_column_dictionary_enabled(column, false);
        }
        Ok(properties.build())
    }
}

#[derive(Clone)]
pub(crate) struct ParquetStore {
    registry: StorageRegistry,
    context: SessionContext,
}

impl ParquetStore {
    #[cfg(test)]
    pub(crate) fn new(registry: StorageRegistry) -> Self {
        Self::with_context(registry, SessionContext::new())
    }

    pub(crate) fn with_context(registry: StorageRegistry, context: SessionContext) -> Self {
        Self { registry, context }
    }

    #[cfg(test)]
    pub(crate) fn registry(&self) -> StorageRegistry {
        self.registry.clone()
    }

    pub(crate) fn context(&self) -> SessionContext {
        self.context.clone()
    }

    #[cfg(test)]
    pub(crate) async fn schema(&self, location: &str) -> Result<SchemaRef> {
        let dataframe = self.dataframe(location).await?;
        Ok(Arc::clone(dataframe.schema().inner()))
    }

    #[cfg(test)]
    pub(crate) async fn read(
        &self,
        location: &str,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        let dataframe = self.dataframe(location).await?;
        let dataframe = match projection {
            Some(columns) => dataframe.select_columns(columns)?,
            None => dataframe,
        };
        let schema = Arc::clone(dataframe.schema().inner());
        let batches = dataframe.collect().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    #[cfg(test)]
    pub(crate) async fn read_clusters(&self, location: &str, cids: &[i32]) -> Result<RecordBatch> {
        let dataframe = self.clusters_dataframe(location, cids).await?;
        let schema = Arc::clone(dataframe.schema().inner());
        let batches = dataframe.collect().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    #[cfg(test)]
    pub(crate) async fn clusters_dataframe(
        &self,
        location: &str,
        cids: &[i32],
    ) -> Result<DataFrame> {
        if cids.is_empty() {
            return Err(Error::InvalidArgument(
                "at least one cluster must be selected".into(),
            ));
        }
        Ok(self
            .dataframe(location)
            .await?
            .filter(col("cid").in_list(cids.iter().copied().map(lit).collect(), false))?)
    }

    #[cfg(test)]
    pub(crate) async fn write(&self, location: &str, batch: &RecordBatch) -> Result<()> {
        self.write_batch(location, batch, &ParquetWriterOptions::default())
            .await
    }

    pub(crate) async fn write_batch(
        &self,
        location: &str,
        batch: &RecordBatch,
        options: &ParquetWriterOptions,
    ) -> Result<()> {
        options.validate()?;
        self.require_empty(location).await?;
        let file = child_location(location, "part-00000.parquet", false)?;
        let resolved = self.registry.resolve(&file)?;
        let mut bytes = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(
                Cursor::new(&mut bytes),
                batch.schema(),
                Some(options.writer_properties()?),
            )?;
            writer.write(batch)?;
            writer.close()?;
        }
        resolved
            .store()
            .put_opts(
                resolved.path(),
                Bytes::from(bytes).into(),
                PutMode::Create.into(),
            )
            .await
            .map_err(relify_storage::Error::from)?;
        Ok(())
    }

    pub(crate) async fn write_hive_cid_dataframe(
        &self,
        location: &str,
        dataframe: DataFrame,
        partitions: usize,
        options: &ParquetWriterOptions,
    ) -> Result<()> {
        options.validate()?;
        self.require_empty(location).await?;
        self.register(location)?;

        let (state, logical_plan) = dataframe.into_parts();
        let physical_plan = state.create_physical_plan(&logical_plan).await?;
        let cid_index = physical_plan.schema().index_of("cid").map_err(|_| {
            Error::InvalidSchema("Hive-partitioned postings must contain cid".into())
        })?;
        let ordering = LexOrdering::new(vec![PhysicalSortExpr::new_default(Arc::new(
            Column::new("cid", cid_index),
        ))])
        .expect("cid ordering is non-empty");
        let repartitioned = Arc::new(RepartitionExec::try_new(
            physical_plan,
            PhysicalPartitioning::Hash(vec![Arc::new(Column::new("cid", cid_index))], partitions),
        )?);
        let sorted =
            Arc::new(SortExec::new(ordering, repartitioned).with_preserve_partitioning(true));
        let streams = execute_stream_partitioned(sorted, state.task_ctx())?;
        let mut tasks = tokio::task::JoinSet::new();
        for stream in streams {
            let registry = self.registry.clone();
            let location = location.to_owned();
            let options = options.clone();
            tasks.spawn(
                async move { write_hive_cid_stream(registry, location, stream, options).await },
            );
        }
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(result) => result?,
                Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
                Err(error) => {
                    return Err(Error::InvalidArgument(format!(
                        "Hive-partitioned postings writer task failed: {error}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(crate) async fn dataframe(&self, location: &str) -> Result<DataFrame> {
        self.register(location)?;
        Ok(self
            .context
            .read_parquet(location, ParquetReadOptions::default())
            .await?)
    }

    pub(crate) async fn partitioned_dataframe(
        &self,
        location: &str,
        partition_columns: Vec<(String, DataType)>,
    ) -> Result<DataFrame> {
        self.register(location)?;
        Ok(self
            .context
            .read_parquet(
                location,
                ParquetReadOptions::default().table_partition_cols(partition_columns),
            )
            .await?)
    }

    pub(crate) fn register(&self, location: &str) -> Result<()> {
        let resolved = self.registry.resolve(location)?;
        self.context
            .register_object_store(resolved.base_url(), resolved.store());
        Ok(())
    }

    async fn require_empty(&self, location: &str) -> Result<()> {
        let resolved = self.registry.resolve(location)?;
        let mut objects = resolved.store().list(Some(resolved.path()));
        if objects
            .try_next()
            .await
            .map_err(relify_storage::Error::from)?
            .is_some()
        {
            return Err(Error::InvalidArgument(format!(
                "Parquet table already exists: {location}"
            )));
        }
        Ok(())
    }
}

async fn write_hive_cid_stream(
    registry: StorageRegistry,
    location: String,
    mut stream: SendableRecordBatchStream,
    options: ParquetWriterOptions,
) -> Result<()> {
    let schema = stream.schema();
    let cid_index = schema
        .index_of("cid")
        .map_err(|_| Error::InvalidSchema("Hive-partitioned postings must contain cid".into()))?;
    let projection = (0..schema.fields().len())
        .filter(|index| *index != cid_index)
        .collect::<Vec<_>>();
    let output_schema = Arc::new(schema.project(&projection)?);
    let dictionary_disabled_column = match output_schema.field_with_name("vector") {
        Ok(vector_field) => {
            let vector_path = match vector_field.data_type() {
                DataType::List(element)
                | DataType::LargeList(element)
                | DataType::FixedSizeList(element, _) => {
                    ColumnPath::new(vec!["vector".into(), "list".into(), element.name().clone()])
                }
                other => {
                    return Err(Error::InvalidSchema(format!(
                        "IVF postings vector must be a list, got {other}"
                    )));
                }
            };
            Some(vector_path)
        }
        Err(_) if output_schema.field_with_name("code").is_ok() => {
            Some(ColumnPath::new(vec!["code".into()]))
        }
        Err(_) => None,
    };
    let properties = match dictionary_disabled_column {
        Some(column) => options.postings_writer_properties(column)?,
        None => options.writer_properties()?,
    };
    let mut current_cid = None;
    let mut writer: Option<AsyncArrowWriter<ParquetObjectWriter>> = None;

    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let cids = batch
            .column(cid_index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| Error::InvalidSchema("IVF postings cid must be int".into()))?;
        if cids.null_count() != 0 {
            return Err(Error::InvalidSchema(
                "IVF postings cid must be required".into(),
            ));
        }
        let values = cids.values();
        if values.windows(2).any(|pair| pair[0] > pair[1])
            || current_cid
                .is_some_and(|previous| values.first().is_some_and(|first| previous > *first))
        {
            return Err(Error::InvalidArgument(
                "Hive-partitioned postings input must be ordered by cid".into(),
            ));
        }

        let mut start = 0;
        while start < values.len() {
            let cid = values[start];
            let length = values[start..].partition_point(|candidate| *candidate == cid);
            if current_cid != Some(cid) {
                if let Some(writer) = writer.take() {
                    writer.close().await?;
                }
                let file =
                    child_location(&location, &format!("cid={cid}/part-00000.parquet"), false)?;
                let resolved = registry.resolve(&file)?;
                let object_writer =
                    ParquetObjectWriter::new(resolved.store(), resolved.path().clone());
                writer = Some(AsyncArrowWriter::try_new(
                    object_writer,
                    Arc::clone(&output_schema),
                    Some(properties.clone()),
                )?);
                current_cid = Some(cid);
            }
            let projected = batch.slice(start, length).project(&projection)?;
            writer
                .as_mut()
                .expect("current cid has an open writer")
                .write(&projected)
                .await?;
            start += length;
        }
    }
    if let Some(writer) = writer {
        writer.close().await?;
    }
    Ok(())
}

pub(crate) fn child_location(base: &str, child: &str, directory: bool) -> Result<String> {
    let mut base = Url::parse(base).map_err(|error| Error::InvalidArgument(error.to_string()))?;
    if !base.path().ends_with('/') {
        let path = format!("{}/", base.path());
        base.set_path(&path);
    }
    let mut location = base
        .join(child)
        .map_err(|error| Error::InvalidArgument(error.to_string()))?;
    if directory && !location.path().ends_with('/') {
        let path = format!("{}/", location.path());
        location.set_path(&path);
    }
    Ok(location.into())
}

#[cfg(test)]
mod tests;
