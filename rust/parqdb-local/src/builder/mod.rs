//! DataFusion-based local IVF construction.

use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
#[cfg(test)]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{
    Array, ArrayRef, BinaryViewArray, BinaryViewBuilder, Float32Array, Float32Builder, Int32Array,
    ListBuilder, StructArray,
};
use arrow::buffer::Buffer;
#[cfg(test)]
use arrow::compute::{SortColumn, SortOptions, lexsort_to_indices, take};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::dataframe::DataFrame;
use datafusion::functions::core::expr_fn::get_field;
use datafusion::logical_expr::{
    ColumnarValue, Partitioning, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature, Volatility, cast,
};
use datafusion::prelude::col;
use futures::StreamExt;
use parallite::ParalliteContext;
use parqdb_meta::{
    IndexArtifactManifest, IvfCentroidsReference, IvfPostingsManifest, PostingEncoding,
    RelationReference, StaticIndexDescriptor, StaticIndexHierarchy, StaticIndexPostings,
    StaticPostingsFile, StaticSourceKeyField,
};
#[cfg(test)]
use parqdb_storage::StorageRegistry;
#[cfg(test)]
use url::Url;

use crate::centroid_navigation::CentroidNavigator;
use crate::ivf::borrow_source_vectors;
#[cfg(test)]
use crate::ivf::source_key_arrays;
use crate::parquet::{ParquetStore, ParquetWriterOptions, child_location};
use crate::progress::BuildPhase;
use crate::{Error, IndexArtifacts, IndexFormat, IvfConfig, LocalBuildProgress, Result};
use parqdb_kernels::{LvqBits, encode_lvq_rows};
use parqdb_kmeans::{KMeansOptions, ReservoirSampler, fit_lloyd_kmeans_with_progress};
#[cfg(test)]
use parqdb_kmeans::{assign_to_centroids, fit_lloyd_kmeans, sample_training_rows};
use uuid::Uuid;

const COARSE_MAX_POINTS_PER_CENTROID: usize = 256;
const DEFAULT_KMEANS_ITERATIONS: usize = 20;
const DEFAULT_SEED: u64 = 42;
const MIN_AUTO_ROW_GROUP_ROWS: usize = 8_192;
const MAX_AUTO_ROW_GROUP_ROWS: usize = 131_072;

#[cfg(test)]
pub(crate) struct IvfTables {
    pub dimension: usize,
    pub nlist: usize,
    pub ntotal: usize,
    pub centroids: RecordBatch,
    pub roots: RecordBatch,
    pub postings: RecordBatch,
}

pub(crate) struct IvfBuildContext<'a> {
    pub parquet: &'a ParquetStore,
    pub output_root: &'a str,
    pub writer_options: &'a ParquetWriterOptions,
    pub partitions: Option<usize>,
    pub parallel: &'a ParalliteContext,
    pub progress: &'a LocalBuildProgress,
}

pub(crate) struct IvfPostingsSpec<'a> {
    pub artifact_uuid: Uuid,
    pub vector_field: &'a str,
    pub source_key_fields: &'a [String],
    pub config: IvfConfig,
    pub trained: &'a TrainedIvf,
    pub ivf_centroids: &'a IvfCentroidsReference,
    pub centroids: RelationReference,
}

pub(crate) struct PreparedIvf {
    source: DataFrame,
    sample: ReservoirSampler,
    pub(crate) dimension: usize,
    pub(crate) nlist: usize,
    pub(crate) ntotal: usize,
}

pub(crate) struct TrainedIvf {
    dimension: usize,
    nlist: usize,
    ntotal: usize,
    centroids: Arc<[f32]>,
    root_centroids: Arc<[f32]>,
    cid_offsets: Arc<[usize]>,
}

struct AssignIvf {
    id: u64,
    signature: Signature,
    dimension: usize,
    navigator: Arc<CentroidNavigator>,
}

struct CidBucket {
    id: u64,
    signature: Signature,
    cid_offsets: Arc<[usize]>,
}

struct RequireNonNull {
    id: u64,
    signature: Signature,
    output_type: DataType,
    vector: bool,
}

struct EncodeLvq {
    id: u64,
    signature: Signature,
    bits: LvqBits,
    dimension: usize,
    output_type: DataType,
}

impl std::fmt::Debug for AssignIvf {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AssignIvf")
            .field("id", &self.id)
            .field("signature", &self.signature)
            .field("dimension", &self.dimension)
            .field("routing", &self.navigator.name())
            .finish()
    }
}

impl PartialEq for AssignIvf {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for AssignIvf {}

impl Hash for AssignIvf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::fmt::Debug for CidBucket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CidBucket")
            .field("id", &self.id)
            .field("roots", &self.cid_offsets.len().saturating_sub(1))
            .finish_non_exhaustive()
    }
}

impl PartialEq for CidBucket {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CidBucket {}

impl Hash for CidBucket {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl ScalarUDFImpl for CidBucket {
    fn name(&self) -> &'static str {
        "parqdb_cid_bucket"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(DataType::Int32)
    }

    fn return_field_from_args(
        &self,
        _arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, false)))
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&arguments.args)?;
        let cids = arrays
            .first()
            .and_then(|array| array.as_any().downcast_ref::<Int32Array>())
            .ok_or_else(|| DataFusionError::Execution("cid must be required int32".into()))?;
        if cids.null_count() != 0 {
            return Err(DataFusionError::Execution(
                "cid must be required int32".into(),
            ));
        }
        let values = cids
            .values()
            .iter()
            .map(|cid| {
                let cid = usize::try_from(*cid)
                    .map_err(|_| DataFusionError::Execution("cid must be non-negative".into()))?;
                let boundary = self.cid_offsets.partition_point(|offset| *offset <= cid);
                if boundary == 0 || boundary == self.cid_offsets.len() {
                    return Err(DataFusionError::Execution(
                        "cid is outside the hierarchical leaf range".into(),
                    ));
                }
                i32::try_from(boundary - 1)
                    .map_err(|_| DataFusionError::Execution("cid_bucket exceeds int32".into()))
            })
            .collect::<datafusion::common::Result<Vec<_>>>()?;
        Ok(ColumnarValue::Array(Arc::new(Int32Array::from(values))))
    }
}

impl ScalarUDFImpl for AssignIvf {
    fn name(&self) -> &'static str {
        "parqdb_assign_ivf"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(DataType::Int32)
    }

    fn return_field_from_args(
        &self,
        _arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, false)))
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&arguments.args)?;
        let vector_array = arrays.first().ok_or_else(|| {
            DataFusionError::Execution("parqdb_assign_ivf requires one argument".into())
        })?;
        let (vectors, actual_dimension) =
            crate::ivf::borrow_vectors_allow_nullable_elements(vector_array)
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        if actual_dimension != self.dimension {
            return Err(DataFusionError::Execution(format!(
                "source vector dimension {actual_dimension} does not match trained dimension {}",
                self.dimension
            )));
        }
        let cells = self
            .navigator
            .route_batch(vectors, 1)
            .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        let cids = Int32Array::from_iter_values(
            cells
                .into_iter()
                .map(|cid| i32::try_from(cid).expect("nlist was validated as int32")),
        );
        Ok(ColumnarValue::Array(Arc::new(cids)))
    }
}

impl std::fmt::Debug for RequireNonNull {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequireNonNull")
            .field("id", &self.id)
            .field("signature", &self.signature)
            .field("output_type", &self.output_type)
            .field("vector", &self.vector)
            .finish()
    }
}

impl PartialEq for RequireNonNull {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RequireNonNull {}

impl Hash for RequireNonNull {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl ScalarUDFImpl for RequireNonNull {
    fn name(&self) -> &'static str {
        "parqdb_require_non_null"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(self.output_type.clone())
    }

    fn return_field_from_args(
        &self,
        _arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        Ok(Arc::new(Field::new(
            self.name(),
            self.output_type.clone(),
            false,
        )))
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&arguments.args)?;
        let array = arrays.first().ok_or_else(|| {
            DataFusionError::Execution("parqdb_require_non_null requires one argument".into())
        })?;
        if array.null_count() != 0 {
            return Err(DataFusionError::Execution(
                "indexed source columns must not contain nulls".into(),
            ));
        }
        if self.vector {
            crate::ivf::borrow_vectors_allow_nullable_elements(array)
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        }
        Ok(ColumnarValue::Array(Arc::clone(array)))
    }
}

impl std::fmt::Debug for EncodeLvq {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EncodeLvq")
            .field("id", &self.id)
            .field("signature", &self.signature)
            .field("bits", &self.bits)
            .field("dimension", &self.dimension)
            .field("output_type", &self.output_type)
            .finish()
    }
}

impl PartialEq for EncodeLvq {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for EncodeLvq {}

impl Hash for EncodeLvq {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl ScalarUDFImpl for EncodeLvq {
    fn name(&self) -> &'static str {
        match self.bits {
            LvqBits::Four => "parqdb_encode_lvq4",
            LvqBits::Eight => "parqdb_encode_lvq8",
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(self.output_type.clone())
    }

    fn return_field_from_args(
        &self,
        _arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        Ok(Arc::new(Field::new(
            self.name(),
            self.output_type.clone(),
            false,
        )))
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&arguments.args)?;
        let vector = arrays.first().ok_or_else(|| {
            DataFusionError::Execution(format!("{} requires one argument", self.name()))
        })?;
        let (vectors, actual_dimension) =
            crate::ivf::borrow_vectors_allow_nullable_elements(vector)
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        if actual_dimension != self.dimension {
            return Err(DataFusionError::Execution(format!(
                "source vector dimension {actual_dimension} does not match trained dimension {}",
                self.dimension
            )));
        }
        let encoded = encode_lvq_rows(vectors, self.dimension, self.bits)
            .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        let (codes, offsets, scales) = encoded.into_parts();
        let code_size = self.bits.code_size(self.dimension);
        let fields = match &self.output_type {
            DataType::Struct(fields) => fields.clone(),
            _ => unreachable!("LVQ encoder output is a struct"),
        };
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float32Array::from(offsets)),
            Arc::new(Float32Array::from(scales)),
            Arc::new(binary_view_codes(codes, code_size)?),
        ];
        Ok(ColumnarValue::Array(Arc::new(StructArray::new(
            fields, columns, None,
        ))))
    }
}

fn binary_view_codes(
    codes: Vec<u8>,
    code_size: usize,
) -> datafusion::common::Result<BinaryViewArray> {
    debug_assert!(code_size > 0);
    debug_assert!(codes.len().is_multiple_of(code_size));
    let row_count = codes.len() / code_size;
    if code_size >= u32::MAX as usize {
        return Err(DataFusionError::Execution(
            "LVQ code rows exceed the Arrow BinaryView limit".into(),
        ));
    }
    let code_size_u32 = u32::try_from(code_size).map_err(|_| {
        DataFusionError::Execution("LVQ code rows exceed the Arrow BinaryView limit".into())
    })?;
    let rows_per_block = (u32::MAX as usize - 1) / code_size;
    let values = Buffer::from(codes);
    let mut builder = BinaryViewBuilder::with_capacity(row_count);

    for first_row in (0..row_count).step_by(rows_per_block) {
        let block_rows = rows_per_block.min(row_count - first_row);
        let block_start = first_row * code_size;
        let block =
            builder.append_block(values.slice_with_length(block_start, block_rows * code_size));
        for row in 0..block_rows {
            builder.try_append_view(
                block,
                u32::try_from(row * code_size).expect("BinaryView block is bounded by u32"),
                code_size_u32,
            )?;
        }
    }
    Ok(builder.finish())
}

pub(crate) async fn prepare_ivf_datafusion(
    source: DataFrame,
    vector_field: &str,
    source_key_fields: &[String],
    config: IvfConfig,
    progress: &LocalBuildProgress,
) -> Result<PreparedIvf> {
    validate_request(vector_field, source_key_fields, config.nlist)?;
    validate_source_schema(source.schema().inner(), vector_field, source_key_fields)?;
    let transformed = crate::vector::transform_vector_udf(config.metric)
        .call(vec![col(vector_field)])
        .alias(vector_field);
    let source = source.with_column(vector_field, transformed)?;
    prepare_training(source, vector_field, config.nlist, progress).await
}

pub(crate) fn train_prepared_ivf(
    prepared: &PreparedIvf,
    parallel: &ParalliteContext,
    progress: &LocalBuildProgress,
) -> Result<TrainedIvf> {
    let mut options = KMeansOptions::new(prepared.nlist);
    options.max_iter = DEFAULT_KMEANS_ITERATIONS;
    options.seed = DEFAULT_SEED;
    let training_rows = prepared.sample.values().len() / prepared.dimension;
    progress.begin(
        BuildPhase::TrainingCentroids,
        training_rows.saturating_mul(options.max_iter),
    );
    let model = fit_lloyd_kmeans_with_progress(
        prepared.sample.values(),
        prepared.dimension,
        parallel,
        options,
        |assigned_rows| {
            let _ = progress.advance(assigned_rows);
        },
    )?;
    let hierarchy = model.hierarchy.ok_or_else(|| {
        Error::InvalidSchema("default IVF training did not produce a hierarchy".into())
    })?;
    if let Some(fallback) = hierarchy.fallback {
        eprintln!(
            "ParqDB warning: hierarchical root {} has {} training rows but requires at least {}; falling back to flat leaf training with a synthetic hierarchy",
            fallback.root, fallback.available_rows, fallback.required_rows
        );
    }
    Ok(TrainedIvf {
        dimension: prepared.dimension,
        nlist: prepared.nlist,
        ntotal: prepared.ntotal,
        centroids: model.centroids.into(),
        root_centroids: hierarchy.root_centroids.into(),
        cid_offsets: hierarchy.cid_offsets.into(),
    })
}

pub(crate) fn reused_ivf(
    prepared: &PreparedIvf,
    centroids: Vec<f32>,
    root_centroids: Vec<f32>,
    cid_offsets: Vec<usize>,
) -> Result<TrainedIvf> {
    let expected = prepared
        .dimension
        .checked_mul(prepared.nlist)
        .ok_or_else(|| Error::InvalidSchema("IVF centroid shape overflows usize".into()))?;
    if centroids.len() != expected || centroids.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidSchema(
            "IVF centroid relation does not match the build descriptor".into(),
        ));
    }
    if cid_offsets.len() < 2
        || cid_offsets.first() != Some(&0)
        || cid_offsets.last() != Some(&prepared.nlist)
        || cid_offsets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::InvalidSchema(
            "IVF root CID offsets do not form a complete hierarchy".into(),
        ));
    }
    let root_count = cid_offsets.len() - 1;
    let expected_roots = prepared
        .dimension
        .checked_mul(root_count)
        .ok_or_else(|| Error::InvalidSchema("IVF root centroid shape overflows usize".into()))?;
    if root_centroids.len() != expected_roots
        || root_centroids.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidSchema(
            "IVF root centroid relation does not match the build descriptor".into(),
        ));
    }
    Ok(TrainedIvf {
        dimension: prepared.dimension,
        nlist: prepared.nlist,
        ntotal: prepared.ntotal,
        centroids: centroids.into(),
        root_centroids: root_centroids.into(),
        cid_offsets: cid_offsets.into(),
    })
}

pub(crate) async fn write_ivf_centroids(
    parquet: &ParquetStore,
    centroids_location: &str,
    roots_location: &str,
    trained: &TrainedIvf,
    writer_options: &ParquetWriterOptions,
    progress: &LocalBuildProgress,
) -> Result<()> {
    let centroid_table =
        centroids_batch(&trained.centroids, trained.dimension, &trained.cid_offsets)?;
    let roots_table = roots_batch(
        &trained.root_centroids,
        trained.dimension,
        &trained.cid_offsets,
    )?;
    progress.begin(BuildPhase::WritingCentroids, 2);
    parquet
        .write_batch(centroids_location, &centroid_table, writer_options)
        .await?;
    progress.set_completed(1);
    parquet
        .write_batch(roots_location, &roots_table, writer_options)
        .await?;
    progress.set_completed(2);
    Ok(())
}

pub(crate) async fn build_ivf_postings(
    prepared: PreparedIvf,
    spec: IvfPostingsSpec<'_>,
    context: IvfBuildContext<'_>,
) -> Result<IndexArtifacts> {
    let IvfPostingsSpec {
        artifact_uuid,
        vector_field,
        source_key_fields,
        config,
        trained,
        ivf_centroids,
        centroids,
    } = spec;
    let IvfBuildContext {
        parquet,
        output_root,
        writer_options,
        partitions,
        parallel,
        progress,
    } = context;
    writer_options.validate()?;
    let source_schema = Arc::clone(prepared.source.schema().inner());
    let source = prepared.source;

    let postings_location = child_location(output_root, "ivf_postings", true)?;
    progress.begin(BuildPhase::BuildingPostings, 0);
    let postings_manifest = write_postings(
        parquet,
        PostingsBuild {
            source,
            vector_field,
            source_key_fields,
            source_schema: source_schema.as_ref(),
            trained,
            posting_encoding: config.posting_encoding,
            output_location: &postings_location,
            parallel,
        },
        writer_options,
        partitions,
    )
    .await?;
    let artifact_manifest = if matches!(
        config.posting_encoding,
        PostingEncoding::Lvq4 | PostingEncoding::Lvq8
    ) {
        Some(
            write_artifact_manifest(
                parquet,
                output_root,
                artifact_uuid,
                source_schema.as_ref(),
                vector_field,
                source_key_fields,
                config,
                trained,
                &postings_manifest,
                writer_options,
            )
            .await?,
        )
    } else {
        None
    };
    Ok(ivf_index_artifacts(
        artifact_uuid,
        &config,
        trained,
        ivf_centroids,
        centroids,
        postings_location,
        artifact_manifest,
    ))
}

fn ivf_index_artifacts(
    artifact_uuid: Uuid,
    config: &IvfConfig,
    trained: &TrainedIvf,
    ivf_centroids: &IvfCentroidsReference,
    centroids: RelationReference,
    postings_location: String,
    artifact_manifest: Option<String>,
) -> IndexArtifacts {
    let mut parameters = BTreeMap::from([
        ("dimension".into(), trained.dimension.to_string()),
        ("nlist".into(), trained.nlist.to_string()),
        ("ntotal".into(), trained.ntotal.to_string()),
        (
            "posting_encoding".into(),
            config.posting_encoding.as_str().into(),
        ),
    ]);
    let index_relations = if let Some(manifest_location) = artifact_manifest {
        parameters.insert("artifact_uuid".into(), artifact_uuid.to_string());
        BTreeMap::from([(
            "artifact_manifest".into(),
            RelationReference::Parquet {
                uri: manifest_location,
            },
        )])
    } else {
        parameters.extend([
            (
                "ivf_centroids_fingerprint".into(),
                ivf_centroids.fingerprint.clone(),
            ),
            (
                "ivf_centroids_uuid".into(),
                ivf_centroids.artifact_uuid.to_string(),
            ),
            (
                "ivf_centroids_metadata_location".into(),
                ivf_centroids.metadata_location.clone(),
            ),
        ]);
        BTreeMap::from([
            ("ivf_centroids".into(), centroids),
            (
                "ivf_postings".into(),
                RelationReference::Parquet {
                    uri: postings_location,
                },
            ),
        ])
    };
    IndexArtifacts {
        format: IndexFormat::ivf(config.metric),
        parameters,
        index_relations,
    }
}

struct PostingsBuild<'a> {
    source: DataFrame,
    vector_field: &'a str,
    source_key_fields: &'a [String],
    source_schema: &'a Schema,
    trained: &'a TrainedIvf,
    posting_encoding: PostingEncoding,
    output_location: &'a str,
    parallel: &'a ParalliteContext,
}

async fn write_postings(
    parquet: &ParquetStore,
    build: PostingsBuild<'_>,
    writer_options: &ParquetWriterOptions,
    partitions: Option<usize>,
) -> Result<IvfPostingsManifest> {
    let PostingsBuild {
        source,
        vector_field,
        source_key_fields,
        source_schema,
        trained,
        posting_encoding,
        output_location,
        parallel,
    } = build;
    let writer_options =
        resolved_postings_writer_options(writer_options, trained.ntotal, trained.nlist);
    let row_width = estimate_posting_row_width(
        source_schema,
        source_key_fields,
        trained.dimension,
        posting_encoding,
    );
    let writers = writer_count(
        &writer_options,
        partitions,
        trained.ntotal,
        row_width,
        parallel.thread_count(),
    );
    let postings = project_postings(
        source,
        vector_field,
        source_key_fields,
        trained,
        posting_encoding,
        parallel,
    )?;
    let (mut state, plan) = postings.into_parts();
    state.config_mut().options_mut().execution.target_partitions = parallel.thread_count();
    state.config_mut().options_mut().execution.batch_size = writer_options.write_batch_rows;
    let postings = DataFrame::new(state, plan);
    parquet
        .write_manifested_cid_dataframe(
            output_location,
            postings,
            writers,
            &trained.cid_offsets,
            trained.ntotal,
            &writer_options,
        )
        .await
}

#[allow(clippy::too_many_arguments)]
async fn write_artifact_manifest(
    parquet: &ParquetStore,
    output_root: &str,
    artifact_uuid: Uuid,
    source_schema: &Schema,
    vector_field: &str,
    source_key_fields: &[String],
    config: IvfConfig,
    trained: &TrainedIvf,
    postings: &IvfPostingsManifest,
    writer_options: &ParquetWriterOptions,
) -> Result<String> {
    let centroids = centroids_batch(&trained.centroids, trained.dimension, &trained.cid_offsets)?;
    let centroid_ranges = trained
        .cid_offsets
        .windows(2)
        .map(|range| range[0]..range[1])
        .collect::<Vec<_>>();
    let centroids_path = "centroids.parquet";
    let centroids_location = child_location(output_root, centroids_path, false)?;
    let centroids = parquet
        .write_static_parquet_object(
            &centroids_location,
            centroids_path,
            &centroids,
            &centroid_ranges,
            writer_options,
        )
        .await?;
    let source_key_fields = static_source_key_fields(source_schema, source_key_fields)?;
    let manifest = IndexArtifactManifest {
        format_version: 1,
        artifact_uuid,
        index: StaticIndexDescriptor {
            vector_field: vector_field.to_owned(),
            metric: config.metric,
            posting_encoding: config.posting_encoding,
            dimension: i32::try_from(trained.dimension)
                .map_err(|_| Error::InvalidSchema("vector dimension exceeds int32".into()))?,
            nlist: i32::try_from(trained.nlist)
                .map_err(|_| Error::InvalidSchema("nlist exceeds int32".into()))?,
            ntotal: i64::try_from(trained.ntotal)
                .map_err(|_| Error::InvalidSchema("ntotal exceeds int64".into()))?,
            source_key_fields,
        },
        hierarchy: StaticIndexHierarchy {
            cid_offsets: trained
                .cid_offsets
                .iter()
                .map(|offset| {
                    i32::try_from(*offset)
                        .map_err(|_| Error::InvalidSchema("CID offset exceeds int32".into()))
                })
                .collect::<Result<Vec<_>>>()?,
            centroid_encoding: PostingEncoding::Lvq8,
            centroids,
        },
        postings: StaticIndexPostings {
            files: postings
                .files
                .iter()
                .map(|file| StaticPostingsFile {
                    path: format!("ivf_postings/{}", file.path),
                    cid_bucket: file.cid_bucket,
                    min_cid: file.min_cid,
                    max_cid: file.max_cid,
                    rows: file.rows,
                    size: file.size,
                    sha256: file.sha256.clone(),
                })
                .collect(),
        },
        source: None,
        embedding: None,
    };
    let bytes = manifest
        .to_json_vec()
        .map_err(|error| Error::InvalidSchema(error.to_string()))?;
    let location = child_location(output_root, "manifest.json", false)?;
    parquet.write_new_object(&location, bytes.into()).await?;
    Ok(location)
}

fn static_source_key_fields(
    schema: &Schema,
    source_key_fields: &[String],
) -> Result<Vec<StaticSourceKeyField>> {
    source_key_fields
        .iter()
        .map(|name| {
            let field = schema
                .field_with_name(name)
                .map_err(|_| Error::InvalidSchema(format!("key column not found: {name}")))?;
            let data_type = match field.data_type() {
                DataType::Boolean => "boolean".into(),
                DataType::Int32 => "int".into(),
                DataType::Int64 => "long".into(),
                DataType::Binary | DataType::BinaryView | DataType::LargeBinary => "binary".into(),
                DataType::FixedSizeBinary(length) => format!("fixed({length})"),
                DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => "string".into(),
                DataType::Date32 => "date".into(),
                data_type => {
                    return Err(Error::InvalidSchema(format!(
                        "unsupported source key type: {data_type}"
                    )));
                }
            };
            Ok(StaticSourceKeyField {
                name: name.clone(),
                data_type,
            })
        })
        .collect()
}

fn project_postings(
    source: DataFrame,
    vector_field: &str,
    source_key_fields: &[String],
    trained: &TrainedIvf,
    posting_encoding: PostingEncoding,
    parallel: &ParalliteContext,
) -> Result<DataFrame> {
    let vector_type = source
        .schema()
        .inner()
        .field_with_name(vector_field)
        .map_err(|_| Error::InvalidSchema(format!("vector column not found: {vector_field}")))?
        .data_type()
        .clone();
    let assignment = assignment_udf(vector_type.clone(), trained, parallel)?;
    let mut expressions = vec![assignment.call(vec![col(vector_field)]).alias("cid")];
    for (index, key) in source_key_fields.iter().enumerate() {
        let output_name = format!("key_{}", index + 1);
        let key_type = source
            .schema()
            .inner()
            .field_with_name(key)
            .expect("source key schema was validated")
            .data_type();
        let output_type = match key_type {
            DataType::Utf8View | DataType::LargeUtf8 => DataType::Utf8,
            DataType::BinaryView | DataType::LargeBinary => DataType::Binary,
            other => other.clone(),
        };
        let expression = if key_type == &output_type {
            col(key)
        } else {
            cast(col(key), output_type.clone())
        };
        expressions.push(
            require_non_null_udf(output_type, false)
                .call(vec![expression])
                .alias(&output_name),
        );
    }
    match posting_encoding {
        PostingEncoding::Source => {}
        PostingEncoding::Lvq4 | PostingEncoding::Lvq8 => {
            let bits = match posting_encoding {
                PostingEncoding::Lvq4 => LvqBits::Four,
                PostingEncoding::Lvq8 => LvqBits::Eight,
                PostingEncoding::Source => unreachable!(),
            };
            expressions.push(
                encode_lvq_udf(vector_type.clone(), trained.dimension, bits)
                    .call(vec![col(vector_field)])
                    .alias("__parqdb_lvq"),
            );
        }
    }
    let postings = source
        .repartition(Partitioning::RoundRobinBatch(parallel.thread_count()))?
        .select(expressions)?;
    let postings = postings.with_column(
        "cid_bucket",
        cid_bucket_udf(Arc::clone(&trained.cid_offsets)).call(vec![col("cid")]),
    )?;
    let postings = if matches!(
        posting_encoding,
        PostingEncoding::Lvq4 | PostingEncoding::Lvq8
    ) {
        let mut output = vec![col("cid_bucket"), col("cid")];
        output.extend((1..=source_key_fields.len()).map(|index| col(format!("key_{index}"))));
        output.extend([
            get_field(col("__parqdb_lvq"), "offset").alias("offset"),
            get_field(col("__parqdb_lvq"), "scale").alias("scale"),
            get_field(col("__parqdb_lvq"), "code").alias("code"),
        ]);
        postings.select(output)?
    } else {
        postings
    };
    Ok(postings)
}

fn resolved_postings_writer_options(
    options: &ParquetWriterOptions,
    ntotal: usize,
    nlist: usize,
) -> ParquetWriterOptions {
    let mut resolved = options.clone();
    resolved.max_row_group_rows = Some(options.max_row_group_rows.unwrap_or_else(|| {
        ntotal
            .div_ceil(nlist)
            .clamp(MIN_AUTO_ROW_GROUP_ROWS, MAX_AUTO_ROW_GROUP_ROWS)
    }));
    resolved
}

async fn prepare_training(
    source: datafusion::dataframe::DataFrame,
    vector_field: &str,
    nlist: usize,
    progress: &LocalBuildProgress,
) -> Result<PreparedIvf> {
    let sample_rows = nlist.saturating_mul(COARSE_MAX_POINTS_PER_CENTROID);
    progress.begin(BuildPhase::ReadingTraining, 0);
    let mut sample = ReservoirSampler::new(sample_rows, DEFAULT_SEED)?;
    let mut batches = source
        .clone()
        .select_columns(&[vector_field])?
        .execute_stream()
        .await?;
    while let Some(batch) = batches.next().await {
        let batch = batch?;
        let (vectors, dimension) = borrow_source_vectors(&batch, vector_field)?;
        sample.push(vectors, dimension)?;
        let _ = progress.advance(batch.num_rows());
    }
    let ntotal = sample.seen_rows();
    if ntotal == 0 {
        return Err(Error::InvalidSchema(
            "source table must contain at least one row".into(),
        ));
    }
    if nlist > ntotal {
        return Err(Error::InvalidArgument(format!(
            "nlist ({nlist}) must not exceed ntotal ({ntotal})"
        )));
    }
    let dimension = sample
        .dimension()
        .ok_or_else(|| Error::InvalidSchema("source table must contain at least one row".into()))?;
    Ok(PreparedIvf {
        source,
        sample,
        dimension,
        nlist,
        ntotal,
    })
}

fn assignment_udf(
    vector_type: DataType,
    trained: &TrainedIvf,
    parallel: &ParalliteContext,
) -> Result<ScalarUDF> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let navigator = Arc::new(CentroidNavigator::new_parallel(
        trained.nlist,
        trained.dimension,
        &trained.centroids,
        parallel,
    )?);
    Ok(ScalarUDF::new_from_impl(AssignIvf {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        signature: Signature::exact(vec![vector_type], Volatility::Immutable),
        dimension: trained.dimension,
        navigator,
    }))
}

fn cid_bucket_udf(cid_offsets: Arc<[usize]>) -> ScalarUDF {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    ScalarUDF::new_from_impl(CidBucket {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        signature: Signature::exact(vec![DataType::Int32], Volatility::Immutable),
        cid_offsets,
    })
}

fn require_non_null_udf(data_type: DataType, vector: bool) -> ScalarUDF {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    ScalarUDF::new_from_impl(RequireNonNull {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        signature: Signature::exact(vec![data_type.clone()], Volatility::Immutable),
        output_type: data_type,
        vector,
    })
}

fn encode_lvq_udf(vector_type: DataType, dimension: usize, bits: LvqBits) -> ScalarUDF {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let output_type = DataType::Struct(
        vec![
            Arc::new(Field::new("offset", DataType::Float32, false)),
            Arc::new(Field::new("scale", DataType::Float32, false)),
            Arc::new(Field::new("code", DataType::BinaryView, false)),
        ]
        .into(),
    );
    ScalarUDF::new_from_impl(EncodeLvq {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        signature: Signature::exact(vec![vector_type], Volatility::Immutable),
        bits,
        dimension,
        output_type,
    })
}

fn validate_source_schema(
    schema: &Schema,
    vector_field: &str,
    source_key_fields: &[String],
) -> Result<()> {
    let vector = schema
        .field_with_name(vector_field)
        .map_err(|_| Error::InvalidSchema(format!("vector column not found: {vector_field}")))?;
    if crate::vector::canonical_vector_type(vector.data_type()).is_none() {
        return Err(Error::InvalidSchema(
            "source vector column must be list<float> or list<double>".into(),
        ));
    }
    for key in source_key_fields {
        let field = schema
            .field_with_name(key)
            .map_err(|_| Error::InvalidSchema(format!("key column not found: {key}")))?;
        if !supported_key_type(field.data_type()) {
            return Err(Error::InvalidSchema(format!(
                "unsupported source key type: {}",
                field.data_type()
            )));
        }
    }
    Ok(())
}

fn supported_key_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int32
            | DataType::Int64
            | DataType::Binary
            | DataType::BinaryView
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_)
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::LargeUtf8
            | DataType::Date32
    )
}

fn estimate_posting_row_width(
    schema: &Schema,
    source_key_fields: &[String],
    dimension: usize,
    posting_encoding: PostingEncoding,
) -> usize {
    let key_width = source_key_fields
        .iter()
        .filter_map(|name| schema.field_with_name(name).ok())
        .map(|field| match field.data_type() {
            DataType::Boolean => 1,
            DataType::Int32 | DataType::Date32 => 4,
            DataType::Int64 => 8,
            DataType::FixedSizeBinary(length) => usize::try_from(*length).unwrap_or(32),
            _ => 32,
        })
        .sum::<usize>();
    let vector_width = match posting_encoding {
        PostingEncoding::Source => 0,
        PostingEncoding::Lvq4 => LvqBits::Four
            .code_size(dimension)
            .saturating_add(2 * std::mem::size_of::<f32>()),
        PostingEncoding::Lvq8 => LvqBits::Eight
            .code_size(dimension)
            .saturating_add(2 * std::mem::size_of::<f32>()),
    };
    4_usize
        .saturating_add(key_width)
        .saturating_add(vector_width)
}

fn writer_count(
    options: &ParquetWriterOptions,
    partitions: Option<usize>,
    ntotal: usize,
    row_width: usize,
    maximum: usize,
) -> usize {
    if let Some(partitions) = partitions {
        return partitions.max(1);
    }
    let estimated_bytes = ntotal.saturating_mul(row_width);
    let files = estimated_bytes.div_ceil(options.target_file_size).max(1);
    files.min(maximum).max(1)
}

#[cfg(test)]
pub(crate) async fn build_ivf(
    source: &RecordBatch,
    vector_field: &str,
    source_key_fields: &[String],
    nlist: usize,
    output_root: &Path,
) -> Result<IndexArtifacts> {
    build_ivf_with_options(
        source,
        vector_field,
        source_key_fields,
        nlist,
        output_root,
        &ParquetWriterOptions::default(),
    )
    .await
}

#[cfg(test)]
pub(crate) async fn build_ivf_with_options(
    source: &RecordBatch,
    vector_field: &str,
    source_key_fields: &[String],
    nlist: usize,
    output_root: &Path,
    writer_options: &ParquetWriterOptions,
) -> Result<IndexArtifacts> {
    let tables = build_ivf_tables(source, vector_field, source_key_fields, nlist)?;
    let output_root = Url::from_directory_path(output_root)
        .map_err(|()| Error::InvalidArgument("output path is not absolute".into()))?;
    let centroids_location = child_location(output_root.as_str(), "ivf_centroids", true)?;
    let roots_location = child_location(output_root.as_str(), "ivf_roots", true)?;
    let postings_location = child_location(output_root.as_str(), "ivf_postings", true)?;
    let parquet = ParquetStore::new(StorageRegistry::default());
    parquet
        .write_batch(&centroids_location, &tables.centroids, writer_options)
        .await?;
    parquet
        .write_batch(&roots_location, &tables.roots, writer_options)
        .await?;
    let postings_writer_options =
        resolved_postings_writer_options(writer_options, tables.ntotal, tables.nlist);
    parquet
        .write_batch(
            &postings_location,
            &tables.postings,
            &postings_writer_options,
        )
        .await?;

    let index_relations = BTreeMap::from([
        (
            "ivf_centroids".into(),
            RelationReference::Parquet {
                uri: centroids_location,
            },
        ),
        (
            "ivf_postings".into(),
            RelationReference::Parquet {
                uri: postings_location,
            },
        ),
    ]);
    Ok(IndexArtifacts {
        format: IndexFormat::ivf(crate::DistanceMetric::L2Squared),
        parameters: BTreeMap::from([
            ("dimension".into(), tables.dimension.to_string()),
            ("nlist".into(), tables.nlist.to_string()),
            ("ntotal".into(), tables.ntotal.to_string()),
            ("posting_encoding".into(), "source".into()),
            (
                "ivf_centroids_fingerprint".into(),
                "73a6be1d-5c50-4f9f-a70b-035ca68b105d".into(),
            ),
            (
                "ivf_centroids_uuid".into(),
                "fe985f6d-3592-4385-a1ca-71347057a210".into(),
            ),
            (
                "ivf_centroids_metadata_location".into(),
                "file:///metadata/fe985f6d-3592-4385-a1ca-71347057a210/v1.metadata.json".into(),
            ),
        ]),
        index_relations,
    })
}

#[cfg(test)]
pub(crate) fn build_ivf_tables(
    source: &RecordBatch,
    vector_field: &str,
    source_key_fields: &[String],
    nlist: usize,
) -> Result<IvfTables> {
    validate_request(vector_field, source_key_fields, nlist)?;

    let (vectors, dimension) = borrow_source_vectors(source, vector_field)?;
    let ntotal = source.num_rows();
    if ntotal == 0 {
        return Err(Error::InvalidSchema(
            "source table must contain at least one row".into(),
        ));
    }
    if nlist > ntotal {
        return Err(Error::InvalidArgument(format!(
            "nlist ({nlist}) must not exceed ntotal ({ntotal})"
        )));
    }

    let key_arrays = source_key_arrays(source, source_key_fields)?;

    let training_rows = nlist.saturating_mul(COARSE_MAX_POINTS_PER_CENTROID);
    let training = sample_training_rows(vectors, dimension, training_rows, DEFAULT_SEED)?;
    let mut options = KMeansOptions::new(nlist);
    options.max_iter = DEFAULT_KMEANS_ITERATIONS;
    options.seed = DEFAULT_SEED;
    let parallel = ParalliteContext::default();
    let model = fit_lloyd_kmeans(&training, dimension, &parallel, options)?;
    let hierarchy = model.hierarchy.as_ref().ok_or_else(|| {
        Error::InvalidSchema("default IVF training did not produce a hierarchy".into())
    })?;
    let cells = assign_to_centroids(vectors, dimension, &model.centroids, &parallel)?;
    Ok(IvfTables {
        dimension,
        nlist,
        ntotal,
        centroids: centroids_batch(&model.centroids, dimension, &hierarchy.cid_offsets)?,
        roots: roots_batch(&hierarchy.root_centroids, dimension, &hierarchy.cid_offsets)?,
        postings: postings_batch(&cells, &key_arrays, None)?,
    })
}

fn validate_request(vector_field: &str, source_key_fields: &[String], nlist: usize) -> Result<()> {
    if vector_field.is_empty() {
        return Err(Error::InvalidArgument(
            "vector column must not be empty".into(),
        ));
    }
    if source_key_fields.is_empty()
        || source_key_fields.iter().any(String::is_empty)
        || source_key_fields.iter().collect::<HashSet<_>>().len() != source_key_fields.len()
    {
        return Err(Error::InvalidArgument(
            "key must contain unique, non-empty column names".into(),
        ));
    }
    if nlist == 0 || nlist > i32::MAX as usize {
        return Err(Error::InvalidArgument(
            "nlist must be in 1..=2147483647".into(),
        ));
    }
    Ok(())
}

fn centroids_batch(
    centroids: &[f32],
    dimension: usize,
    cid_offsets: &[usize],
) -> Result<RecordBatch> {
    let nlist = centroids.len() / dimension;
    let cids = Int32Array::from_iter_values(
        (0..nlist).map(|cid| i32::try_from(cid).expect("nlist was validated as int32")),
    );
    let cid_buckets = Int32Array::from_iter_values(cid_offsets.windows(2).enumerate().flat_map(
        |(bucket, range)| {
            std::iter::repeat_n(
                i32::try_from(bucket).expect("root count is bounded by nlist"),
                range[1] - range[0],
            )
        },
    ));
    if cid_buckets.len() != nlist {
        return Err(Error::InvalidSchema(
            "IVF CID offsets do not cover every leaf centroid".into(),
        ));
    }
    let encoded = encode_lvq_rows(centroids, dimension, LvqBits::Eight)
        .map_err(|error| Error::InvalidSchema(error.to_string()))?;
    let (codes, offsets, scales) = encoded.into_parts();
    let codes = binary_view_codes(codes, dimension)
        .map_err(|error| Error::InvalidSchema(error.to_string()))?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("cid", DataType::Int32, false),
        Field::new("cid_bucket", DataType::Int32, false),
        Field::new("offset", DataType::Float32, false),
        Field::new("scale", DataType::Float32, false),
        Field::new("code", codes.data_type().clone(), false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(cids),
            Arc::new(cid_buckets),
            Arc::new(Float32Array::from(offsets)),
            Arc::new(Float32Array::from(scales)),
            Arc::new(codes),
        ],
    )?)
}

fn roots_batch(
    root_centroids: &[f32],
    dimension: usize,
    cid_offsets: &[usize],
) -> Result<RecordBatch> {
    let root_count = cid_offsets.len().checked_sub(1).ok_or_else(|| {
        Error::InvalidSchema("IVF CID offsets must contain at least two entries".into())
    })?;
    if root_centroids.len() != root_count.saturating_mul(dimension) {
        return Err(Error::InvalidSchema(
            "IVF root centroid shape does not match CID offsets".into(),
        ));
    }
    let buckets = Int32Array::from_iter_values(
        (0..root_count)
            .map(|bucket| i32::try_from(bucket).expect("root count is bounded by nlist")),
    );
    let cid_begin = Int32Array::from_iter_values(
        cid_offsets[..root_count]
            .iter()
            .map(|offset| i32::try_from(*offset).expect("nlist was validated as int32")),
    );
    let cid_end = Int32Array::from_iter_values(
        cid_offsets[1..]
            .iter()
            .map(|offset| i32::try_from(*offset).expect("nlist was validated as int32")),
    );
    let mut builder = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for centroid in root_centroids.chunks_exact(dimension) {
        builder.values().append_slice(centroid);
        builder.append(true);
    }
    let centroid_array = builder.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("cid_bucket", DataType::Int32, false),
        Field::new("cid_begin", DataType::Int32, false),
        Field::new("cid_end", DataType::Int32, false),
        Field::new("centroid", centroid_array.data_type().clone(), false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(buckets),
            Arc::new(cid_begin),
            Arc::new(cid_end),
            Arc::new(centroid_array),
        ],
    )?)
}

#[cfg(test)]
fn postings_batch(
    cells: &[usize],
    key_arrays: &[ArrayRef],
    vectors: Option<ArrayRef>,
) -> Result<RecordBatch> {
    let cids = Int32Array::from_iter_values(
        cells
            .iter()
            .map(|cid| i32::try_from(*cid).expect("nlist was validated as int32")),
    );
    let mut fields = vec![Field::new("cid", DataType::Int32, false)];
    fields.extend(key_arrays.iter().enumerate().map(|(index, array)| {
        Field::new(
            format!("key_{}", index + 1),
            array.data_type().clone(),
            false,
        )
    }));
    let mut arrays = vec![Arc::new(cids) as ArrayRef];
    arrays.extend(key_arrays.iter().cloned());
    if let Some(vectors) = vectors {
        fields.push(Field::new("vector", vectors.data_type().clone(), false));
        arrays.push(vectors);
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    let columns = batch
        .columns()
        .iter()
        .take(key_arrays.len() + 1)
        .map(|array| SortColumn {
            values: Arc::clone(array),
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        })
        .collect::<Vec<_>>();
    let indices = lexsort_to_indices(&columns, None)?;
    let sorted = batch
        .columns()
        .iter()
        .map(|array| take(array, &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), sorted)?)
}

#[cfg(test)]
mod tests;
