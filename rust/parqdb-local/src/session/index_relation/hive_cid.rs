//! Narrow Parquet provider for immutable IVF postings partitioned by `cid`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{ScalarValue, Statistics, TableReference};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::helpers::pruned_partition_list;
use datafusion::datasource::listing::{ListingTableUrl, PartitionedFile};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use futures::TryStreamExt;
use object_store::ObjectMeta;
use parqdb_storage::StorageRegistry;

use crate::config::IndexIoMode;
#[cfg(target_os = "linux")]
use crate::parquet::DirectIoParquetFileReaderFactory;
use crate::{Error, Result};

const SCAN_PARTITIONS_PER_WORKER: usize = 4;

#[derive(Clone)]
struct HiveCidManifestFile {
    object_meta: ObjectMeta,
    table_reference: Option<TableReference>,
}

impl HiveCidManifestFile {
    fn to_partitioned_file(&self, cid: i32) -> PartitionedFile {
        let mut file = PartitionedFile::new_from_meta(self.object_meta.clone());
        file.partition_values = vec![ScalarValue::Int32(Some(cid))];
        file.table_reference.clone_from(&self.table_reference);
        file
    }
}

pub(super) struct HiveCidParquetProvider {
    schema: SchemaRef,
    file_schema: SchemaRef,
    object_store_url: ObjectStoreUrl,
    format: Arc<dyn FileFormat>,
    files: BTreeMap<i32, Vec<HiveCidManifestFile>>,
}

impl fmt::Debug for HiveCidParquetProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HiveCidParquetProvider")
            .field("clusters", &self.files.len())
            .field("files", &self.files.values().map(Vec::len).sum::<usize>())
            .finish_non_exhaustive()
    }
}

impl HiveCidParquetProvider {
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
        let partition_columns = [("cid".to_owned(), DataType::Int32)];
        let listed = pruned_partition_list(
            state,
            store.as_ref(),
            &table_path,
            &[],
            ".parquet",
            &partition_columns,
        )
        .await?
        .try_collect::<Vec<_>>()
        .await?;
        let first = listed.first().ok_or_else(|| {
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
        let file_schema = format
            .infer_schema(state, &store, std::slice::from_ref(&first.object_meta))
            .await?;
        let table_schema = TableSchema::new(
            Arc::clone(&file_schema),
            vec![Arc::new(Field::new("cid", DataType::Int32, false))],
        );
        let schema = Arc::clone(table_schema.table_schema());
        let mut files = BTreeMap::<i32, Vec<HiveCidManifestFile>>::new();
        for file in listed {
            let [ScalarValue::Int32(Some(cid))] = file.partition_values.as_slice() else {
                return Err(Error::InvalidSchema(
                    "Hive CID postings file has an invalid partition value".into(),
                ));
            };
            files.entry(*cid).or_default().push(HiveCidManifestFile {
                object_meta: file.object_meta,
                table_reference: file.table_reference,
            });
        }
        for cluster_files in files.values_mut() {
            cluster_files.sort_unstable_by(|left, right| {
                left.object_meta.location.cmp(&right.object_meta.location)
            });
        }
        Ok(Self {
            schema,
            file_schema,
            object_store_url: table_path.object_store(),
            format,
            files,
        })
    }

    pub(super) fn resident_size(&self) -> usize {
        let mut size = size_of::<Self>();
        for files in self.files.values() {
            size = size
                .saturating_add(size_of::<i32>())
                .saturating_add(size_of::<Vec<HiveCidManifestFile>>())
                .saturating_add(
                    files
                        .capacity()
                        .saturating_mul(size_of::<HiveCidManifestFile>()),
                );
            for file in files {
                size = size
                    .saturating_add(file.object_meta.location.as_ref().len())
                    .saturating_add(file.object_meta.e_tag.as_ref().map_or(0, String::capacity))
                    .saturating_add(
                        file.object_meta
                            .version
                            .as_ref()
                            .map_or(0, String::capacity),
                    )
                    .saturating_add(
                        file.table_reference
                            .as_ref()
                            .map_or(0, |reference| reference.to_string().len()),
                    );
            }
        }
        size
    }

    fn selected_files(&self, filters: &[Expr]) -> Vec<PartitionedFile> {
        let mut selected = None::<BTreeSet<i32>>;
        for values in filters.iter().filter_map(cid_filter_values) {
            selected = Some(match selected {
                Some(current) => current.intersection(&values).copied().collect(),
                None => values,
            });
        }
        match selected {
            Some(cids) => cids
                .into_iter()
                .flat_map(|cid| {
                    self.files
                        .get(&cid)
                        .into_iter()
                        .flatten()
                        .map(move |file| file.to_partitioned_file(cid))
                })
                .collect(),
            None => self
                .files
                .iter()
                .flat_map(|(&cid, files)| {
                    files.iter().map(move |file| file.to_partitioned_file(cid))
                })
                .collect(),
        }
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
impl TableProvider for HiveCidParquetProvider {
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
        let files = self.selected_files(filters);
        if files.is_empty() {
            return Ok(Arc::new(EmptyExec::new(self.projected_schema(projection)?)));
        }
        let target_partitions = state
            .config()
            .target_partitions()
            .saturating_mul(SCAN_PARTITIONS_PER_WORKER)
            .max(1);
        let file_groups = FileGroup::new(files).split_files(target_partitions);
        let table_schema = TableSchema::new(
            Arc::clone(&self.file_schema),
            vec![Arc::new(Field::new("cid", DataType::Int32, false))],
        );
        let source = self.format.file_source(table_schema);
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
                    TableProviderFilterPushDown::Exact
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
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::file_format::parquet::ParquetFormat;
    use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
    use tempfile::TempDir;
    use url::Url;

    use super::*;
    use crate::parquet::{ParquetStore, ParquetWriterOptions};

    fn provider(cluster_count: i32) -> HiveCidParquetProvider {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "key_1",
            DataType::Int64,
            false,
        )]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("key_1", DataType::Int64, false),
            Field::new("cid", DataType::Int32, false),
        ]));
        let files = (0..cluster_count)
            .map(|cid| {
                let file = HiveCidManifestFile {
                    object_meta: PartitionedFile::new(format!("cid={cid}/part.parquet"), 1)
                        .object_meta,
                    table_reference: None,
                };
                (cid, vec![file])
            })
            .collect();
        HiveCidParquetProvider {
            schema,
            file_schema,
            object_store_url: ObjectStoreUrl::local_filesystem(),
            format: Arc::new(ParquetFormat::default()),
            files,
        }
    }

    #[test]
    fn cid_filters_select_manifest_files_without_affecting_other_filters() {
        let provider = provider(4);
        let cids = col("cid").in_list(vec![lit(1_i32), lit(3_i32)], false);
        let selected = provider.selected_files(std::slice::from_ref(&cids));
        let paths = selected
            .iter()
            .map(|file| file.path().to_string())
            .collect::<Vec<_>>();

        assert_eq!(paths, ["cid=1/part.parquet", "cid=3/part.parquet"]);
        assert_eq!(
            selected
                .iter()
                .map(|file| file.partition_values.as_slice())
                .collect::<Vec<_>>(),
            [
                [ScalarValue::Int32(Some(1))].as_slice(),
                [ScalarValue::Int32(Some(3))].as_slice(),
            ]
        );
        assert_eq!(
            provider.supports_filters_pushdown(&[&cids]).unwrap(),
            [TableProviderFilterPushDown::Exact]
        );

        let unrelated = col("key_1").gt(lit(0_i64));
        assert_eq!(
            provider
                .selected_files(std::slice::from_ref(&unrelated))
                .len(),
            4
        );
        assert_eq!(
            provider.supports_filters_pushdown(&[&unrelated]).unwrap(),
            [TableProviderFilterPushDown::Unsupported]
        );

        let reversed = lit(2_i64).eq(col("cid"));
        assert_eq!(
            provider
                .selected_files(std::slice::from_ref(&reversed))
                .len(),
            1
        );
        let union = col("cid").eq(lit(0_i32)).or(col("cid").eq(lit(2_i32)));
        assert_eq!(provider.selected_files(&[union]).len(), 2);
        let empty = col("cid").eq(lit(0_i32)).and(col("cid").eq(lit(2_i32)));
        assert!(provider.selected_files(&[empty]).is_empty());

        let overflow = col("cid").eq(lit(i64::from(i32::MAX) + 1));
        let null = col("cid").eq(Expr::Literal(ScalarValue::Int32(None), None));
        let negated = col("cid").in_list(vec![lit(1_i32)], true);
        for unsupported in [&overflow, &null, &negated] {
            assert_eq!(
                provider.supports_filters_pushdown(&[unsupported]).unwrap(),
                [TableProviderFilterPushDown::Unsupported]
            );
        }
    }

    #[test]
    fn billion_scale_manifest_fits_the_bounded_cache() {
        let provider = provider(65_536);

        assert!(provider.resident_size() < 64 * 1024 * 1024);
    }

    #[tokio::test]
    async fn scan_overpartitions_selected_files() {
        let provider = provider(8);
        let context =
            SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        let plan = provider
            .scan(&context.state(), None, &[], None)
            .await
            .unwrap();

        assert_eq!(plan.properties().output_partitioning().partition_count(), 8);
    }

    #[tokio::test]
    async fn empty_cluster_selection_preserves_projection() {
        let provider = provider(2);
        let context = SessionContext::new();
        let filter = col("cid").eq(lit(99_i32));
        let plan = provider
            .scan(&context.state(), Some(&vec![0]), &[filter], None)
            .await
            .unwrap();

        assert_eq!(plan.name(), "EmptyExec");
        assert_eq!(plan.schema().fields().len(), 1);
        assert_eq!(plan.schema().field(0).name(), "key_1");
    }

    #[tokio::test]
    async fn real_parquet_scan_matches_listing_table_contract() {
        let temporary = TempDir::new().unwrap();
        let context = SessionContext::new();
        let store = ParquetStore::with_context(StorageRegistry::default(), context.clone());
        let location = Url::from_directory_path(temporary.path().join("postings"))
            .unwrap()
            .to_string();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cid", DataType::Int32, false),
                Field::new("key_1", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![0, 0, 1, 1])),
                Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            ],
        )
        .unwrap();
        context.register_batch("input", batch).unwrap();
        store
            .write_hive_cid_dataframe(
                &location,
                context.table("input").await.unwrap(),
                2,
                &ParquetWriterOptions::default(),
            )
            .await
            .unwrap();
        let (listing, _) = store
            .uniform_dataset_listing_table(&location, vec![("cid".into(), DataType::Int32)])
            .await
            .unwrap();
        let custom = Arc::new(
            HiveCidParquetProvider::load(
                &store.registry(),
                &location,
                &context.state(),
                IndexIoMode::Buffered,
            )
            .await
            .unwrap(),
        );
        context.register_table("listing", listing).unwrap();
        context.register_table("custom", custom).unwrap();

        for query in [
            "SELECT key_1 FROM {table} WHERE cid IN (0, 1) ORDER BY key_1 LIMIT 3",
            "SELECT cid FROM {table} WHERE key_1 > 15 ORDER BY cid, key_1 LIMIT 2",
        ] {
            let listing = context
                .sql(&query.replace("{table}", "listing"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let custom = context
                .sql(&query.replace("{table}", "custom"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(custom, listing);
        }
    }
}
