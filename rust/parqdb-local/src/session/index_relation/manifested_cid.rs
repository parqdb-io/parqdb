//! Manifest-backed Parquet provider for immutable IVF postings.

use std::collections::BTreeSet;
use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, ScalarValue, Statistics};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingTableUrl, PartitionedFile};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion_datasource_parquet::ParquetAccessPlan;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use parqdb_meta::{IvfPostingsFile, IvfPostingsManifest};
use parqdb_storage::StorageRegistry;
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::file::statistics::Statistics as ParquetStatistics;

use crate::config::IndexIoMode;
#[cfg(target_os = "linux")]
use crate::parquet::DirectIoParquetFileReaderFactory;
use crate::parquet::child_location;
use crate::{Error, Result};

const SCAN_PARTITIONS_PER_WORKER: usize = 4;

#[derive(Clone)]
struct ManifestedFile {
    entry: IvfPostingsFile,
    object_meta: ObjectMeta,
}

#[derive(Clone)]
pub(super) struct ManifestedCidParquetProvider {
    schema: SchemaRef,
    object_store_url: ObjectStoreUrl,
    store: Arc<dyn ObjectStore>,
    format: Arc<dyn FileFormat>,
    manifest: Arc<IvfPostingsManifest>,
    files: Arc<[ManifestedFile]>,
    cid_selection: Option<Arc<BTreeSet<i32>>>,
}

impl fmt::Debug for ManifestedCidParquetProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManifestedCidParquetProvider")
            .field("roots", &self.manifest.cid_offsets.len().saturating_sub(1))
            .field("files", &self.files.len())
            .field(
                "selected_cids",
                &self.cid_selection.as_ref().map(|values| values.len()),
            )
            .finish_non_exhaustive()
    }
}

impl ManifestedCidParquetProvider {
    pub(super) async fn load(
        registry: &StorageRegistry,
        location: &str,
        state: &dyn Session,
        index_io: IndexIoMode,
    ) -> Result<Self> {
        let resolved = registry.resolve(location)?;
        let store = resolved.store();
        state
            .runtime_env()
            .register_object_store(resolved.base_url(), Arc::clone(&store));
        let table_path = ListingTableUrl::parse(location)?;
        let manifest_location = child_location(location, "manifest.json", false)?;
        let resolved_manifest = registry.resolve(&manifest_location)?;
        let manifest_bytes = resolved_manifest
            .store()
            .get(resolved_manifest.path())
            .await
            .map_err(parqdb_storage::Error::from)?
            .bytes()
            .await
            .map_err(parqdb_storage::Error::from)?;
        let manifest = Arc::new(
            IvfPostingsManifest::from_json_slice(&manifest_bytes)
                .map_err(|error| Error::InvalidSchema(error.to_string()))?,
        );
        let mut files = Vec::with_capacity(manifest.files.len());
        for entry in &manifest.files {
            let file_location = child_location(location, &entry.path, false)?;
            let resolved_file = registry.resolve(&file_location)?;
            if resolved_file.base_url() != resolved.base_url() {
                return Err(Error::InvalidSchema(
                    "manifested postings file escaped its object store".into(),
                ));
            }
            let object_meta =
                PartitionedFile::new(resolved_file.path().to_string(), entry.size).object_meta;
            files.push(ManifestedFile {
                entry: entry.clone(),
                object_meta,
            });
        }
        let first = files.first().ok_or_else(|| {
            Error::InvalidArgument(format!("Parquet table contains no data files: {location}"))
        })?;
        let mut format = ParquetFormat::new().with_options(state.default_table_options().parquet);
        if index_io == IndexIoMode::Direct {
            if table_path.object_store() != ObjectStoreUrl::local_filesystem() {
                return Err(Error::InvalidArgument(
                    "parqdb.parquet.index_io='direct' requires a local file index".into(),
                ));
            }
            #[cfg(target_os = "linux")]
            {
                let metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();
                format = format.with_parquet_file_reader_factory(Arc::new(
                    DirectIoParquetFileReaderFactory::new(Arc::clone(&store), metadata_cache),
                ));
            }
            #[cfg(not(target_os = "linux"))]
            return Err(Error::InvalidArgument(
                "parqdb.parquet.index_io='direct' requires Linux".into(),
            ));
        }
        let format = Arc::new(format);
        let schema = format
            .infer_schema(state, &store, std::slice::from_ref(&first.object_meta))
            .await?;
        if schema.field_with_name("cid").is_err() {
            return Err(Error::InvalidSchema(
                "manifested postings Parquet files must contain cid".into(),
            ));
        }
        Ok(Self {
            schema,
            object_store_url: table_path.object_store(),
            store,
            format,
            manifest,
            files: files.into(),
            cid_selection: None,
        })
    }

    pub(super) fn with_cid_selection(&self, cids: &[i32]) -> Result<Self> {
        let mut selected = BTreeSet::new();
        for cid in cids {
            if *cid < 0 || *cid >= self.manifest.nlist {
                return Err(Error::InvalidArgument(format!(
                    "selected CID {cid} is outside [0, {})",
                    self.manifest.nlist
                )));
            }
            selected.insert(*cid);
        }
        let mut provider = self.clone();
        provider.cid_selection = Some(Arc::new(selected));
        Ok(provider)
    }

    pub(super) fn resident_size(&self) -> usize {
        let mut size = size_of::<Self>()
            .saturating_add(
                self.manifest
                    .cid_offsets
                    .capacity()
                    .saturating_mul(size_of::<i32>()),
            )
            .saturating_add(self.files.len().saturating_mul(size_of::<ManifestedFile>()));
        for file in self.files.iter() {
            size = size
                .saturating_add(file.entry.path.capacity())
                .saturating_add(file.object_meta.location.as_ref().len());
        }
        size
    }

    pub(super) fn validate_identity(
        &self,
        nlist: usize,
        ntotal: usize,
        cid_offsets: &[usize],
    ) -> Result<()> {
        if usize::try_from(self.manifest.nlist).ok() != Some(nlist)
            || usize::try_from(self.manifest.ntotal).ok() != Some(ntotal)
            || self.manifest.cid_offsets.len() != cid_offsets.len()
            || self
                .manifest
                .cid_offsets
                .iter()
                .zip(cid_offsets)
                .any(|(manifest, expected)| usize::try_from(*manifest).ok() != Some(*expected))
        {
            return Err(Error::InvalidSchema(
                "postings manifest does not match the index hierarchy".into(),
            ));
        }
        Ok(())
    }

    fn selected_cids(&self, filters: &[Expr]) -> Option<BTreeSet<i32>> {
        let mut selected = self.cid_selection.as_deref().cloned();
        for values in filters.iter().filter_map(cid_filter_values) {
            selected = Some(match selected {
                Some(current) => current.intersection(&values).copied().collect(),
                None => values,
            });
        }
        selected
    }

    fn selected_files<'a>(&'a self, selected: Option<&BTreeSet<i32>>) -> Vec<&'a ManifestedFile> {
        self.files
            .iter()
            .filter(|file| {
                selected.is_none_or(|cids| {
                    cids.range(file.entry.min_cid..=file.entry.max_cid)
                        .next()
                        .is_some()
                })
            })
            .collect()
    }

    async fn partitioned_file(
        &self,
        file: &ManifestedFile,
        selected: Option<&BTreeSet<i32>>,
    ) -> Result<Option<PartitionedFile>> {
        let mut partitioned = PartitionedFile::new_from_meta(file.object_meta.clone());
        let Some(selected) = selected else {
            return Ok(Some(partitioned));
        };
        let mut reader =
            ParquetObjectReader::new(Arc::clone(&self.store), file.object_meta.location.clone())
                .with_file_size(file.entry.size);
        let metadata = reader.get_metadata(None).await?;
        let cid_column = metadata
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .position(|column| column.name() == "cid")
            .ok_or_else(|| {
                Error::InvalidSchema("manifested postings footer is missing cid".into())
            })?;
        let mut access_plan = ParquetAccessPlan::new_none(metadata.num_row_groups());
        let mut footer_rows = 0_u64;
        let mut selected_groups = 0_usize;
        for (index, row_group) in metadata.row_groups().iter().enumerate() {
            footer_rows = footer_rows
                .checked_add(u64::try_from(row_group.num_rows()).map_err(|_| {
                    Error::InvalidSchema("Parquet row-group row count is negative".into())
                })?)
                .ok_or_else(|| Error::InvalidSchema("Parquet row count overflows".into()))?;
            let statistics = row_group.column(cid_column).statistics().ok_or_else(|| {
                Error::InvalidSchema("cid row-group statistics are required".into())
            })?;
            let ParquetStatistics::Int32(statistics) = statistics else {
                return Err(Error::InvalidSchema(
                    "cid row-group statistics must be INT32".into(),
                ));
            };
            let min = statistics.min_opt().copied();
            let max = statistics.max_opt().copied();
            let cid = match (min, max) {
                (Some(min), Some(max))
                    if min == max
                        && statistics.min_is_exact()
                        && statistics.max_is_exact()
                        && statistics.null_count_opt() == Some(0) =>
                {
                    min
                }
                _ => {
                    return Err(Error::InvalidSchema(
                        "each postings row group must contain exactly one non-null CID".into(),
                    ));
                }
            };
            if cid < file.entry.min_cid || cid > file.entry.max_cid {
                return Err(Error::InvalidSchema(
                    "row-group CID is outside the manifest file range".into(),
                ));
            }
            if selected.contains(&cid) {
                access_plan.scan(index);
                selected_groups += 1;
            }
        }
        if footer_rows != file.entry.rows {
            return Err(Error::InvalidSchema(
                "Parquet footer row count does not match the postings manifest".into(),
            ));
        }
        if selected_groups == 0 {
            return Ok(None);
        }
        partitioned = partitioned.with_extension(access_plan);
        Ok(Some(partitioned))
    }

    fn projected_schema(
        &self,
        projection: Option<&Vec<usize>>,
    ) -> datafusion::common::Result<SchemaRef> {
        projection.map_or_else(
            || Ok(Arc::clone(&self.schema)),
            |projection| Ok(Arc::new(self.schema.project(projection)?)),
        )
    }
}

#[async_trait]
impl TableProvider for ManifestedCidParquetProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let selected = self.selected_cids(filters);
        let candidates = self.selected_files(selected.as_ref());
        let mut files = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if let Some(file) = self
                .partitioned_file(candidate, selected.as_ref())
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?
            {
                files.push(file);
            }
        }
        if files.is_empty() {
            return Ok(Arc::new(EmptyExec::new(self.projected_schema(projection)?)));
        }
        let target_partitions = state
            .config()
            .target_partitions()
            .saturating_mul(SCAN_PARTITIONS_PER_WORKER)
            .max(1);
        let file_groups = FileGroup::new(files).split_files(target_partitions);
        let source = self
            .format
            .file_source(TableSchema::new(Arc::clone(&self.schema), Vec::new()));
        let config = FileScanConfigBuilder::new(self.object_store_url.clone(), source)
            .with_file_groups(file_groups)
            .with_statistics(Statistics::new_unknown(self.schema.as_ref()))
            .with_projection_indices(projection.cloned())?
            .with_limit(limit)
            .build();
        self.format.create_physical_plan(state, config).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|filter| {
                if cid_filter_values(filter).is_some() {
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }
}

pub(super) fn cid_filter_values(filter: &Expr) -> Option<BTreeSet<i32>> {
    match filter {
        Expr::InList(list) if !list.negated && is_cid_column(&list.expr) => list
            .list
            .iter()
            .map(cid_literal)
            .collect::<Option<BTreeSet<_>>>(),
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::Eq => {
                if is_cid_column(&binary.left) {
                    cid_literal(&binary.right).map(|value| BTreeSet::from([value]))
                } else if is_cid_column(&binary.right) {
                    cid_literal(&binary.left).map(|value| BTreeSet::from([value]))
                } else {
                    None
                }
            }
            Operator::Or => {
                let mut values = cid_filter_values(&binary.left)?;
                values.extend(cid_filter_values(&binary.right)?);
                Some(values)
            }
            Operator::And => {
                let left = cid_filter_values(&binary.left)?;
                let right = cid_filter_values(&binary.right)?;
                Some(left.intersection(&right).copied().collect())
            }
            _ => None,
        },
        _ => None,
    }
}

fn is_cid_column(expression: &Expr) -> bool {
    matches!(expression, Expr::Column(column) if column.name == "cid")
}

fn cid_literal(expression: &Expr) -> Option<i32> {
    let Expr::Literal(value, _) = expression else {
        return None;
    };
    match value {
        ScalarValue::Int8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int32(value) => *value,
        ScalarValue::Int64(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt64(Some(value)) => i32::try_from(*value).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::{SessionContext, col, lit};
    use tempfile::TempDir;
    use url::Url;

    use super::*;
    use crate::parquet::{ParquetStore, ParquetWriterOptions};

    #[test]
    fn cid_filter_parser_handles_large_and_composed_selections() {
        let cids = col("cid").in_list((0..10_000).map(lit).collect(), false);
        assert_eq!(cid_filter_values(&cids).unwrap().len(), 10_000);

        let reversed = lit(2_i64).eq(col("cid"));
        assert_eq!(cid_filter_values(&reversed).unwrap(), BTreeSet::from([2]));
        let intersection = col("cid")
            .eq(lit(1_i32))
            .or(col("cid").eq(lit(2_i32)))
            .and(col("cid").eq(lit(2_i32)));
        assert_eq!(
            cid_filter_values(&intersection).unwrap(),
            BTreeSet::from([2])
        );
    }

    #[tokio::test]
    async fn manifested_scan_matches_the_physical_cid_filter() {
        let temporary = TempDir::new().unwrap();
        let context = SessionContext::new();
        let store = ParquetStore::with_context(StorageRegistry::default(), context.clone());
        let location = Url::from_directory_path(temporary.path().join("postings"))
            .unwrap()
            .to_string();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cid_bucket", DataType::Int32, false),
                Field::new("cid", DataType::Int32, false),
                Field::new("key_1", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![0, 0, 1, 1])),
                Arc::new(Int32Array::from(vec![0, 0, 1, 1])),
                Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            ],
        )
        .unwrap();
        context.register_batch("input", batch).unwrap();
        let options = ParquetWriterOptions {
            max_row_group_rows: Some(1),
            target_file_size: usize::MAX,
            ..ParquetWriterOptions::default()
        };
        store
            .write_manifested_cid_dataframe(
                &location,
                context.table("input").await.unwrap(),
                2,
                &[0, 1, 2],
                4,
                &options,
            )
            .await
            .unwrap();
        let provider = ManifestedCidParquetProvider::load(
            &store.registry(),
            &location,
            &context.state(),
            IndexIoMode::Buffered,
        )
        .await
        .unwrap()
        .with_cid_selection(&[1])
        .unwrap();
        let selection = provider.selected_cids(&[]).unwrap();
        let candidates = provider.selected_files(Some(&selection));
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            (candidates[0].entry.min_cid, candidates[0].entry.max_cid),
            (1, 1)
        );
        let candidate = candidates[0];
        let partitioned = provider
            .partitioned_file(candidate, Some(&selection))
            .await
            .unwrap()
            .unwrap();
        let access_plan = partitioned.extension::<ParquetAccessPlan>().unwrap();
        assert_eq!(access_plan.row_group_indexes(), [0, 1]);

        context
            .register_table("custom", Arc::new(provider))
            .unwrap();
        let actual = context
            .sql("SELECT cid, key_1 FROM custom WHERE cid IN (1) ORDER BY key_1")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    }

    #[tokio::test]
    async fn typed_selection_above_the_inline_limit_builds_an_exact_access_plan() {
        const NLIST: i32 = 258;

        let temporary = TempDir::new().unwrap();
        let context = SessionContext::new();
        let store = ParquetStore::with_context(StorageRegistry::default(), context.clone());
        let location = Url::from_directory_path(temporary.path().join("postings"))
            .unwrap()
            .to_string();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cid_bucket", DataType::Int32, false),
                Field::new("cid", DataType::Int32, false),
                Field::new("key_1", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![0; NLIST as usize])),
                Arc::new(Int32Array::from_iter_values(0..NLIST)),
                Arc::new(Int64Array::from_iter_values(0..i64::from(NLIST))),
            ],
        )
        .unwrap();
        context.register_batch("large_input", batch).unwrap();
        store
            .write_manifested_cid_dataframe(
                &location,
                context.table("large_input").await.unwrap(),
                1,
                &[0, NLIST as usize],
                NLIST as usize,
                &ParquetWriterOptions {
                    max_row_group_rows: Some(1),
                    target_file_size: usize::MAX,
                    ..ParquetWriterOptions::default()
                },
            )
            .await
            .unwrap();
        let selected = (0..NLIST).step_by(2).collect::<Vec<_>>();
        assert_eq!(selected.len(), 129);
        let provider = ManifestedCidParquetProvider::load(
            &store.registry(),
            &location,
            &context.state(),
            IndexIoMode::Buffered,
        )
        .await
        .unwrap()
        .with_cid_selection(&selected)
        .unwrap();
        let selection = provider.selected_cids(&[]).unwrap();
        let candidate = provider.selected_files(Some(&selection))[0];
        let partitioned = provider
            .partitioned_file(candidate, Some(&selection))
            .await
            .unwrap()
            .unwrap();
        let access_plan = partitioned.extension::<ParquetAccessPlan>().unwrap();
        let expected = selected
            .iter()
            .map(|cid| usize::try_from(*cid).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(access_plan.row_group_indexes(), expected);
    }
}
