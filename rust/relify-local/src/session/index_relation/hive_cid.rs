use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{ScalarValue, Statistics};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::listing::{ListingTable, PartitionedFile};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;

use crate::{Error, Result};

const SCAN_PARTITIONS_PER_WORKER: usize = 4;

pub(super) struct HiveCidParquetProvider {
    schema: SchemaRef,
    file_schema: SchemaRef,
    object_store_url: ObjectStoreUrl,
    format: Arc<dyn FileFormat>,
    files: BTreeMap<i32, Vec<PartitionedFile>>,
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
        listing: Arc<ListingTable>,
        file_schema: SchemaRef,
        state: &dyn Session,
    ) -> Result<Self> {
        let listed = listing.list_files_for_scan(state, &[], None).await?;
        let mut files = BTreeMap::<i32, Vec<PartitionedFile>>::new();
        for file in listed
            .file_groups
            .into_iter()
            .flat_map(FileGroup::into_inner)
        {
            let [ScalarValue::Int32(Some(cid))] = file.partition_values.as_slice() else {
                return Err(Error::InvalidSchema(
                    "Hive CID postings file has an invalid partition value".into(),
                ));
            };
            files.entry(*cid).or_default().push(file);
        }
        for cluster_files in files.values_mut() {
            cluster_files.sort_unstable_by(|left, right| left.path().cmp(right.path()));
        }
        let object_store_url = listing
            .table_paths()
            .first()
            .ok_or_else(|| Error::InvalidSchema("Parquet relation has no table path".into()))?
            .object_store();
        Ok(Self {
            schema: listing.schema(),
            file_schema,
            object_store_url,
            format: Arc::clone(&listing.options().format),
            files,
        })
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
                .flat_map(|cid| self.files.get(&cid).into_iter().flatten().cloned())
                .collect(),
            None => self.files.values().flatten().cloned().collect(),
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

fn cid_filter_values(filter: &Expr) -> Option<BTreeSet<i32>> {
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
    use arrow::datatypes::{Field, Schema};
    use datafusion::datasource::file_format::parquet::ParquetFormat;
    use datafusion::prelude::{SessionConfig, SessionContext, col, lit};

    use super::*;

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
                let mut file = PartitionedFile::new(format!("cid={cid}/part.parquet"), 1);
                file.partition_values = vec![ScalarValue::Int32(Some(cid))];
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
}
