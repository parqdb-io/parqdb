//! Managed Parquet relation I/O for the embedded backend.

#[cfg(target_os = "linux")]
mod direct_io;
mod page_cache;

#[cfg(target_os = "linux")]
pub(crate) use direct_io::DirectIoParquetFileReaderFactory;
pub use page_cache::ParquetPageCacheStats;
pub(crate) use page_cache::{
    DecompressedParquetPageCache, ParqDBParquetPageCacheFactory, automatic_page_cache_capacity,
};

use std::io::Cursor;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use datafusion::catalog::Session;
#[cfg(test)]
use datafusion::catalog::TableProvider;
use datafusion::dataframe::DataFrame;
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::options::ReadOptions;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingTable, ListingTableConfig, ListingTableUrl};
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
use object_store::{ObjectStore, ObjectStoreExt, PutMode};
use parqdb_storage::StorageRegistry;
use parquet::arrow::async_writer::ParquetObjectWriter;
use parquet::arrow::{ArrowWriter, AsyncArrowWriter};
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
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

    fn code_writer_properties(&self) -> Result<WriterProperties> {
        self.writer_properties_for(Some(ColumnPath::new(vec!["code".into()])))
    }

    fn writer_properties_for(
        &self,
        plain_encoded_column: Option<ColumnPath>,
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
        if let Some(column) = plain_encoded_column {
            properties = properties
                .set_column_dictionary_enabled(column.clone(), false)
                .set_column_encoding(column, Encoding::PLAIN);
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
            .map_err(parqdb_storage::Error::from)?;
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

    #[cfg(test)]
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

    /// Creates a provider for a dataset whose Parquet files share one schema.
    ///
    /// `ParqDB` index publication guarantees a uniform schema, so the first data
    /// file is sufficient and avoids `DataFusion`'s default all-file schema
    /// merge.
    #[cfg(test)]
    pub(crate) async fn uniform_dataset_provider(
        &self,
        location: &str,
        partition_columns: Vec<(String, DataType)>,
    ) -> Result<Arc<dyn TableProvider>> {
        let (provider, _) = self
            .uniform_dataset_listing_table(location, partition_columns)
            .await?;
        Ok(provider)
    }

    #[cfg(test)]
    pub(crate) async fn uniform_dataset_listing_table(
        &self,
        location: &str,
        partition_columns: Vec<(String, DataType)>,
    ) -> Result<(Arc<ListingTable>, SchemaRef)> {
        uniform_dataset_listing_table(
            &self.registry,
            &self.context.state(),
            location,
            partition_columns,
        )
        .await
    }

    pub(crate) fn register(&self, location: &str) -> Result<()> {
        register_object_store(&self.registry, &self.context.state(), location)
    }

    async fn require_empty(&self, location: &str) -> Result<()> {
        let resolved = self.registry.resolve(location)?;
        let mut objects = resolved.store().list(Some(resolved.path()));
        if objects
            .try_next()
            .await
            .map_err(parqdb_storage::Error::from)?
            .is_some()
        {
            return Err(Error::InvalidArgument(format!(
                "Parquet table already exists: {location}"
            )));
        }
        Ok(())
    }
}

pub(crate) async fn uniform_dataset_listing_table(
    registry: &StorageRegistry,
    state: &dyn Session,
    location: &str,
    partition_columns: Vec<(String, DataType)>,
) -> Result<(Arc<ListingTable>, SchemaRef)> {
    register_object_store(registry, state, location)?;
    let schema = infer_uniform_dataset_schema(registry, state, location).await?;
    let options = ParquetReadOptions::default()
        .schema(schema.as_ref())
        .table_partition_cols(partition_columns)
        .to_listing_options(state.config(), state.default_table_options());
    let table_path = ListingTableUrl::parse(location)?;
    let provider = ListingTable::try_new(
        ListingTableConfig::new(table_path)
            .with_listing_options(options)
            .with_schema(Arc::clone(&schema)),
    )?
    .with_cache(state.runtime_env().cache_manager.get_file_statistic_cache());
    Ok((Arc::new(provider), schema))
}

async fn infer_uniform_dataset_schema(
    registry: &StorageRegistry,
    state: &dyn Session,
    location: &str,
) -> Result<SchemaRef> {
    let resolved = registry.resolve(location)?;
    let store = resolved.store();
    let first = if resolved.uri().path().ends_with('/') {
        first_parquet_object(store.as_ref(), resolved.path()).await?
    } else {
        let object = store
            .head(resolved.path())
            .await
            .map_err(parqdb_storage::Error::from)?;
        valid_parquet_object(object)
    };
    let first = first.ok_or_else(|| {
        Error::InvalidArgument(format!("Parquet table contains no data files: {location}"))
    })?;
    let options = state.default_table_options().parquet;
    Ok(ParquetFormat::new()
        .with_options(options)
        .infer_schema(state, &store, &[first])
        .await?)
}

fn register_object_store(
    registry: &StorageRegistry,
    state: &dyn Session,
    location: &str,
) -> Result<()> {
    let resolved = registry.resolve(location)?;
    state
        .runtime_env()
        .register_object_store(resolved.base_url(), resolved.store());
    Ok(())
}

async fn first_parquet_object(
    store: &dyn ObjectStore,
    prefix: &object_store::path::Path,
) -> Result<Option<object_store::ObjectMeta>> {
    let mut objects = store.list(Some(prefix));
    while let Some(object) = objects
        .try_next()
        .await
        .map_err(parqdb_storage::Error::from)?
    {
        if let Some(object) = valid_parquet_object(object) {
            return Ok(Some(object));
        }
    }
    Ok(None)
}

fn valid_parquet_object(object: object_store::ObjectMeta) -> Option<object_store::ObjectMeta> {
    (object.size > 0 && object.location.as_ref().ends_with(".parquet")).then_some(object)
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
    let properties = if output_schema.field_with_name("code").is_ok() {
        options.code_writer_properties()?
    } else {
        options.writer_properties()?
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
