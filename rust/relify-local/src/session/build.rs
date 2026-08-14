//! Session-scoped index construction and refresh.

use std::collections::HashSet;

use parallite::ParalliteContext;
use relify_index::{
    InitialIndex, RefreshedIndex, new_snapshot_id, publish_initial, publish_refresh,
};
use relify_meta::PostingEncoding;
use uuid::Uuid;

use super::LocalSession;
use super::catalog::local_index_identifier;
use crate::builder::{IvfBuildContext, build_ivf_datafusion};
use crate::parquet::ParquetWriterOptions;
use crate::progress::BuildPhase;
use crate::{Error, IvfConfig, LocalBuildProgress, PublishedIndex, Result};

/// Resource options for one local index build.
#[derive(Debug, Clone, Default)]
pub struct LocalBuildOptions {
    /// Physical Parquet output settings.
    pub writer_options: ParquetWriterOptions,
    /// Number of postings output partitions, or automatic sizing when unset.
    pub partitions: Option<usize>,
    /// K-means worker count, or available parallelism when unset.
    pub threads: Option<usize>,
    /// Optional thread-safe progress state for this build.
    pub progress: Option<LocalBuildProgress>,
}

impl LocalSession {
    /// Builds and publishes an IVF index over a Parquet source table.
    pub async fn create_index(
        &self,
        source: &str,
        index_name: &str,
        vector_field: &str,
        source_key_fields: &[String],
        nlist: usize,
    ) -> Result<PublishedIndex> {
        self.create_index_with_options(
            source,
            index_name,
            vector_field,
            source_key_fields,
            IvfConfig::new(nlist, PostingEncoding::Flat),
            &LocalBuildOptions::default(),
        )
        .await
    }

    /// Builds and publishes an IVF index with explicit Parquet options.
    pub async fn create_index_with_options(
        &self,
        source: &str,
        index_name: &str,
        vector_field: &str,
        source_key_fields: &[String],
        config: IvfConfig,
        options: &LocalBuildOptions,
    ) -> Result<PublishedIndex> {
        options.writer_options.validate()?;
        validate_partitions(options.partitions)?;
        let progress = options.progress.clone().unwrap_or_default();
        let parallel = build_context(options.threads)?;
        let identifier = local_index_identifier(index_name)?;
        let mut build_lease = self.coordination.reserve_build(&identifier)?;
        if self.index_exists(index_name)? {
            return Err(Error::AlreadyExists(index_name.to_owned()));
        }
        validate_build_request(vector_field, source_key_fields, config.nlist)?;

        let source = self.bind_source(source).await?;
        let source_reference = source.reference.clone();
        let source_schema = source.schema;
        if source_schema.field_with_name("_distance").is_ok() {
            return Err(Error::InvalidSchema(
                "source table must not contain reserved column _distance".into(),
            ));
        }
        let index_uuid = Uuid::new_v4();
        let snapshot_id = new_snapshot_id();
        let snapshot_root = self.warehouse.location(
            &format!("indexes/{}/{snapshot_id}", index_uuid.simple()),
            true,
        )?;
        build_lease.set_snapshot_root(&snapshot_root)?;
        let build = build_ivf_datafusion(
            self.context.read_table(source.provider)?,
            vector_field,
            source_key_fields,
            config,
            IvfBuildContext {
                parquet: &self.parquet,
                output_root: &snapshot_root,
                writer_options: &options.writer_options,
                partitions: options.partitions,
                parallel: &parallel,
                progress: &progress,
            },
        )
        .await?;
        progress.begin(BuildPhase::Publishing, 1);
        let _guard = self.coordination.write()?;
        let result = publish_initial(
            self.indexes.catalog(),
            self.indexes.metadata_store(),
            InitialIndex {
                identifier,
                index_uuid,
                snapshot_id,
                source: source_reference,
                vector_field,
                source_key_fields,
                builder: "local",
                build,
            },
        )
        .await?;
        progress.finish();
        Ok(result)
    }

    /// Rebuilds and atomically publishes a new snapshot of an IVF index.
    pub async fn refresh_index_with_options(
        &self,
        source: &str,
        index_name: &str,
        config: Option<IvfConfig>,
        options: &LocalBuildOptions,
    ) -> Result<PublishedIndex> {
        options.writer_options.validate()?;
        validate_partitions(options.partitions)?;
        let progress = options.progress.clone().unwrap_or_default();
        let parallel = build_context(options.threads)?;
        let identifier = local_index_identifier(index_name)?;
        let mut build_lease = self.coordination.reserve_build(&identifier)?;
        let loaded = self.load_entry(&identifier).await?;
        let current = loaded.metadata.current_snapshot()?;
        let source = self.bind_source(source).await?;
        let source_reference = source.reference.clone();
        if current.source.identity_key() != source_reference.identity_key() {
            return Err(Error::IndexNotFound(index_name.to_owned()));
        }
        let config = config.unwrap_or(IvfConfig::new(
            current.parameter_usize("nlist")?,
            PostingEncoding::from_snapshot(current)?,
        ));
        if config.nlist == 0 || config.nlist > i32::MAX as usize {
            return Err(Error::InvalidArgument(
                "nlist must be in 1..=2147483647".into(),
            ));
        }
        let snapshot_id = new_snapshot_id();
        let snapshot_root = self.warehouse.location(
            &format!(
                "indexes/{}/{snapshot_id}",
                loaded.metadata.index_uuid.simple()
            ),
            true,
        )?;
        build_lease.set_snapshot_root(&snapshot_root)?;
        let build = build_ivf_datafusion(
            self.context.read_table(source.provider)?,
            &current.vector_field,
            &current.source_key_fields,
            config,
            IvfBuildContext {
                parquet: &self.parquet,
                output_root: &snapshot_root,
                writer_options: &options.writer_options,
                partitions: options.partitions,
                parallel: &parallel,
                progress: &progress,
            },
        )
        .await?;
        progress.begin(BuildPhase::Publishing, 1);
        let _guard = self.coordination.write()?;
        let result = publish_refresh(
            self.indexes.catalog(),
            self.indexes.metadata_store(),
            RefreshedIndex {
                identifier,
                base_metadata_location: &loaded.entry.metadata_location,
                base_metadata: &loaded.metadata,
                snapshot_id,
                source: source_reference,
                builder: "local",
                build,
            },
        )
        .await?;
        progress.finish();
        Ok(result)
    }
}

fn validate_partitions(partitions: Option<usize>) -> Result<()> {
    if partitions == Some(0) {
        return Err(Error::InvalidArgument(
            "partitions must be a positive integer".into(),
        ));
    }
    Ok(())
}

fn build_context(threads: Option<usize>) -> Result<ParalliteContext> {
    match threads {
        Some(threads) => ParalliteContext::builder()
            .threads(threads)
            .map_err(|error| Error::InvalidArgument(error.to_string()))
            .map(parallite::ParalliteContextBuilder::build),
        None => Ok(ParalliteContext::default()),
    }
}

fn validate_build_request(
    vector_field: &str,
    source_key_fields: &[String],
    nlist: usize,
) -> Result<()> {
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
