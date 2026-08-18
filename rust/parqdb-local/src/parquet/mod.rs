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
use parqdb_meta::{IvfPostingsFile, IvfPostingsManifest};
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

    pub(crate) async fn write_manifested_cid_dataframe(
        &self,
        location: &str,
        dataframe: DataFrame,
        partitions: usize,
        cid_offsets: &[usize],
        ntotal: usize,
        options: &ParquetWriterOptions,
    ) -> Result<IvfPostingsManifest> {
        options.validate()?;
        self.require_empty(location).await?;
        self.register(location)?;

        if cid_offsets.len() < 2
            || cid_offsets.first() != Some(&0)
            || cid_offsets.windows(2).any(|range| range[0] >= range[1])
        {
            return Err(Error::InvalidArgument(
                "IVF CID offsets must define non-empty contiguous buckets".into(),
            ));
        }

        let files = write_manifested_cid_files(
            self.registry.clone(),
            location.to_owned(),
            dataframe,
            partitions,
            options.clone(),
        )
        .await?;
        let nlist = i32::try_from(*cid_offsets.last().expect("CID offsets are non-empty"))
            .map_err(|_| Error::InvalidArgument("nlist exceeds the INT32 domain".into()))?;
        let manifest = IvfPostingsManifest {
            format_version: 1,
            nlist,
            ntotal: i64::try_from(ntotal)
                .map_err(|_| Error::InvalidArgument("ntotal exceeds the INT64 domain".into()))?,
            cid_offsets: cid_offsets
                .iter()
                .map(|offset| {
                    i32::try_from(*offset).map_err(|_| {
                        Error::InvalidArgument("CID offset exceeds the INT32 domain".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            files,
        };
        manifest
            .validate()
            .map_err(|error| Error::InvalidSchema(error.to_string()))?;
        let manifest_location = child_location(location, "manifest.json", false)?;
        let resolved = self.registry.resolve(&manifest_location)?;
        resolved
            .store()
            .put_opts(
                resolved.path(),
                Bytes::from(
                    manifest
                        .to_json_vec()
                        .map_err(|error| Error::InvalidSchema(error.to_string()))?,
                )
                .into(),
                PutMode::Create.into(),
            )
            .await
            .map_err(parqdb_storage::Error::from)?;
        Ok(manifest)
    }

    pub(crate) async fn dataframe(&self, location: &str) -> Result<DataFrame> {
        self.register(location)?;
        Ok(self
            .context
            .read_parquet(location, ParquetReadOptions::default())
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

async fn write_manifested_cid_files(
    registry: StorageRegistry,
    location: String,
    dataframe: DataFrame,
    partitions: usize,
    options: ParquetWriterOptions,
) -> Result<Vec<IvfPostingsFile>> {
    let (state, logical_plan) = dataframe.into_parts();
    let physical_plan = state.create_physical_plan(&logical_plan).await?;
    let cid_index = physical_plan
        .schema()
        .index_of("cid")
        .map_err(|_| Error::InvalidSchema("manifested postings must contain cid".into()))?;
    let bucket_index = physical_plan
        .schema()
        .index_of("cid_bucket")
        .map_err(|_| Error::InvalidSchema("manifested postings must contain cid_bucket".into()))?;
    let ordering = LexOrdering::new(vec![
        PhysicalSortExpr::new_default(Arc::new(Column::new("cid_bucket", bucket_index))),
        PhysicalSortExpr::new_default(Arc::new(Column::new("cid", cid_index))),
    ])
    .expect("bucket and cid ordering is non-empty");
    let repartitioned = Arc::new(RepartitionExec::try_new(
        physical_plan,
        PhysicalPartitioning::Hash(
            vec![Arc::new(Column::new("cid_bucket", bucket_index))],
            partitions,
        ),
    )?);
    let sorted = Arc::new(SortExec::new(ordering, repartitioned).with_preserve_partitioning(true));
    let streams = execute_stream_partitioned(sorted, state.task_ctx())?;
    let mut tasks = tokio::task::JoinSet::new();
    for stream in streams {
        let registry = registry.clone();
        let location = location.clone();
        let options = options.clone();
        tasks.spawn(async move {
            write_manifested_cid_stream(registry, location, stream, options).await
        });
    }
    let mut files = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(result) => files.extend(result?),
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(error) => {
                return Err(Error::InvalidArgument(format!(
                    "manifested postings writer task failed: {error}"
                )));
            }
        }
    }
    files.sort_unstable_by(|left, right| {
        (
            left.cid_bucket,
            left.min_cid,
            left.max_cid,
            left.path.as_str(),
        )
            .cmp(&(
                right.cid_bucket,
                right.min_cid,
                right.max_cid,
                right.path.as_str(),
            ))
    });
    Ok(files)
}

struct OpenPostingsFile {
    writer: AsyncArrowWriter<ParquetObjectWriter>,
    store: Arc<dyn ObjectStore>,
    object_path: object_store::path::Path,
    relative_path: String,
    cid_bucket: i32,
    min_cid: i32,
    max_cid: i32,
    rows: u64,
}

impl OpenPostingsFile {
    async fn finish(self) -> Result<IvfPostingsFile> {
        self.writer.close().await?;
        let metadata = self
            .store
            .head(&self.object_path)
            .await
            .map_err(parqdb_storage::Error::from)?;
        Ok(IvfPostingsFile {
            path: self.relative_path,
            cid_bucket: self.cid_bucket,
            min_cid: self.min_cid,
            max_cid: self.max_cid,
            rows: self.rows,
            size: metadata.size,
        })
    }
}

fn open_postings_file(
    registry: &StorageRegistry,
    location: &str,
    bucket: i32,
    part: usize,
    cid: i32,
    schema: SchemaRef,
    properties: WriterProperties,
) -> Result<OpenPostingsFile> {
    let relative_path = format!("cid_bucket={bucket:06}/part-{part:05}.parquet");
    let file = child_location(location, &relative_path, false)?;
    let resolved = registry.resolve(&file)?;
    let store = resolved.store();
    let object_path = resolved.path().clone();
    let object_writer = ParquetObjectWriter::new(Arc::clone(&store), object_path.clone());
    Ok(OpenPostingsFile {
        writer: AsyncArrowWriter::try_new(object_writer, schema, Some(properties))?,
        store,
        object_path,
        relative_path,
        cid_bucket: bucket,
        min_cid: cid,
        max_cid: cid,
        rows: 0,
    })
}

async fn finish_postings_file(
    writer: &mut Option<OpenPostingsFile>,
    files: &mut Vec<IvfPostingsFile>,
) -> Result<()> {
    if let Some(writer) = writer.take() {
        files.push(writer.finish().await?);
    }
    Ok(())
}

struct ManifestedCidStreamWriter {
    registry: StorageRegistry,
    location: String,
    projection: Vec<usize>,
    output_schema: SchemaRef,
    properties: WriterProperties,
    options: ParquetWriterOptions,
    current_bucket: Option<i32>,
    current_cid: Option<i32>,
    next_part: usize,
    writer: Option<OpenPostingsFile>,
    files: Vec<IvfPostingsFile>,
}

impl ManifestedCidStreamWriter {
    fn new(
        registry: StorageRegistry,
        location: String,
        schema: &SchemaRef,
        options: ParquetWriterOptions,
    ) -> Result<(Self, usize, usize)> {
        let cid_index = schema
            .index_of("cid")
            .map_err(|_| Error::InvalidSchema("manifested postings must contain cid".into()))?;
        let bucket_index = schema.index_of("cid_bucket").map_err(|_| {
            Error::InvalidSchema("manifested postings must contain cid_bucket".into())
        })?;
        let projection = (0..schema.fields().len())
            .filter(|index| *index != bucket_index)
            .collect::<Vec<_>>();
        let output_schema = Arc::new(schema.project(&projection)?);
        let properties = if output_schema.field_with_name("code").is_ok() {
            options.code_writer_properties()?
        } else {
            options.writer_properties()?
        };
        Ok((
            Self {
                registry,
                location,
                projection,
                output_schema,
                properties,
                options,
                current_bucket: None,
                current_cid: None,
                next_part: 0,
                writer: None,
                files: Vec::new(),
            },
            cid_index,
            bucket_index,
        ))
    }

    async fn write_batch(
        &mut self,
        batch: &RecordBatch,
        cid_index: usize,
        bucket_index: usize,
    ) -> Result<()> {
        let cids = required_postings_int32(batch, cid_index, "IVF postings cid")?;
        let buckets = required_postings_int32(batch, bucket_index, "cid_bucket")?;
        let cid_values = cids.values();
        let bucket_values = buckets.values();
        if (1..batch.num_rows()).any(|row| {
            (bucket_values[row - 1], cid_values[row - 1]) > (bucket_values[row], cid_values[row])
        }) || self
            .current_bucket
            .zip(self.current_cid)
            .is_some_and(|previous| {
                bucket_values
                    .first()
                    .zip(cid_values.first())
                    .is_some_and(|(&bucket, &cid)| previous > (bucket, cid))
            })
        {
            return Err(Error::InvalidArgument(
                "manifested postings input must be ordered by (cid_bucket, cid)".into(),
            ));
        }

        let mut start = 0;
        while start < batch.num_rows() {
            let bucket = bucket_values[start];
            let cid = cid_values[start];
            let length = (start..batch.num_rows())
                .take_while(|index| bucket_values[*index] == bucket && cid_values[*index] == cid)
                .count();
            self.begin_cid(bucket, cid).await?;
            self.write_cid_rows(batch, start, length, bucket, cid)
                .await?;
            start += length;
        }
        Ok(())
    }

    async fn begin_cid(&mut self, bucket: i32, cid: i32) -> Result<()> {
        if self.current_bucket != Some(bucket) {
            finish_postings_file(&mut self.writer, &mut self.files).await?;
            self.current_bucket = Some(bucket);
            self.next_part = 0;
        } else if self.current_cid != Some(cid)
            && let Some(open) = self.writer.as_mut()
        {
            if open.writer.in_progress_rows() > 0 {
                open.writer.flush().await?;
            }
            if open.writer.bytes_written() >= self.options.target_file_size {
                finish_postings_file(&mut self.writer, &mut self.files).await?;
                self.next_part += 1;
            }
        }
        self.current_cid = Some(cid);
        Ok(())
    }

    async fn write_cid_rows(
        &mut self,
        batch: &RecordBatch,
        start: usize,
        length: usize,
        bucket: i32,
        cid: i32,
    ) -> Result<()> {
        let row_group_rows = self
            .options
            .max_row_group_rows
            .unwrap_or(self.options.write_batch_rows);
        let mut written = 0;
        while written < length {
            if self.writer.is_none() {
                self.writer = Some(open_postings_file(
                    &self.registry,
                    &self.location,
                    bucket,
                    self.next_part,
                    cid,
                    Arc::clone(&self.output_schema),
                    self.properties.clone(),
                )?);
            }
            let open = self.writer.as_mut().expect("postings writer was opened");
            open.max_cid = cid;
            let remaining_in_group = row_group_rows
                .saturating_sub(open.writer.in_progress_rows())
                .max(1);
            let chunk_rows = remaining_in_group.min(length - written);
            let projected = batch
                .slice(start + written, chunk_rows)
                .project(&self.projection)?;
            open.writer.write(&projected).await?;
            open.rows = open
                .rows
                .checked_add(u64::try_from(chunk_rows).expect("usize fits u64"))
                .ok_or_else(|| Error::InvalidArgument("postings row count overflows".into()))?;
            written += chunk_rows;

            if open.writer.in_progress_rows() >= row_group_rows {
                open.writer.flush().await?;
            }
            if open.writer.in_progress_rows() == 0
                && open.writer.bytes_written() >= self.options.target_file_size
            {
                finish_postings_file(&mut self.writer, &mut self.files).await?;
                self.next_part += 1;
            }
        }
        Ok(())
    }

    async fn finish(mut self) -> Result<Vec<IvfPostingsFile>> {
        finish_postings_file(&mut self.writer, &mut self.files).await?;
        Ok(self.files)
    }
}

fn required_postings_int32<'a>(
    batch: &'a RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a Int32Array> {
    let values = batch
        .column(index)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| Error::InvalidSchema(format!("{field} must be int")))?;
    if values.null_count() != 0 {
        return Err(Error::InvalidSchema(format!("{field} must be required")));
    }
    Ok(values)
}

async fn write_manifested_cid_stream(
    registry: StorageRegistry,
    location: String,
    mut stream: SendableRecordBatchStream,
    options: ParquetWriterOptions,
) -> Result<Vec<IvfPostingsFile>> {
    let schema = stream.schema();
    let (mut writer, cid_index, bucket_index) =
        ManifestedCidStreamWriter::new(registry, location, &schema, options)?;

    while let Some(batch) = stream.next().await {
        writer.write_batch(&batch?, cid_index, bucket_index).await?;
    }
    writer.finish().await
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
