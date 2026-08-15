//! Session-scoped index construction and refresh.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parallite::ParalliteContext;
use relify_catalog::{IndexCatalog, IndexIdentifier, IvfCentroidsClaim, IvfCentroidsClaimResult};
use relify_index::{
    InitialIndex, RefreshedIndex, new_snapshot_id, publish_initial, publish_refresh,
};
use relify_meta::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, IvfCentroidsDescriptor, IvfCentroidsMetadata,
    IvfCentroidsReference, PostingEncoding, RelationReference,
};
use uuid::Uuid;

use super::LocalSession;
use super::build_coordinator::{ActiveBuildState, BuildKey};
use super::catalog::{local_index_identifier, namespaced_index_identifier};
use crate::builder::{
    IvfBuildContext, IvfPostingsSpec, PreparedIvf, TrainedIvf, build_ivf_postings,
    prepare_ivf_datafusion, reused_ivf, train_prepared_ivf, write_ivf_centroids,
};
use crate::coordination::BuildLease;
use crate::parquet::{ParquetWriterOptions, child_location};
use crate::progress::BuildPhase;
use crate::{BuildFailureKind, Error, IvfConfig, LocalBuildProgress, PublishedIndex, Result};

const IVF_CENTROIDS_LEASE_MS: i64 = 30_000;
const IVF_CENTROIDS_RENEW_INTERVAL: Duration = Duration::from_secs(10);
const IVF_CENTROIDS_WAIT_INTERVAL: Duration = Duration::from_millis(250);

/// Observable state of one process-scoped index build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBuildState {
    /// Accepted and waiting for the session build slot.
    Pending,
    /// Actively constructing or publishing index data.
    Building,
    /// A published index is available.
    Ready,
    /// The initial build failed and no index is published.
    Failed,
}

/// Current build and publication state for one source-scoped index.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexBuildStatus {
    /// Current lifecycle state.
    pub state: IndexBuildState,
    /// Estimated overall completion.
    pub progress: Option<f64>,
    /// Current build phase.
    pub phase: Option<&'static str>,
    /// Work units completed in the current phase.
    pub completed: Option<u64>,
    /// Total work units in the current phase.
    pub total: Option<u64>,
    /// Published snapshot available to queries, if any.
    pub current_snapshot_id: Option<i64>,
    /// Most recent process-local build failure.
    pub error: Option<String>,
    /// Stable category of the most recent process-local build failure.
    pub error_code: Option<&'static str>,
}

struct ResolvedIvfCentroids {
    reference: IvfCentroidsReference,
    centroids: RelationReference,
    trained: TrainedIvf,
}

struct IvfCentroidsBuildContext<'a> {
    prepared: &'a PreparedIvf,
    parallel: &'a ParalliteContext,
    writer_options: &'a ParquetWriterOptions,
    progress: &'a LocalBuildProgress,
    build_lease: &'a mut BuildLease,
}

struct ClaimHeartbeat {
    stop: Sender<()>,
    task: JoinHandle<relify_catalog::Result<()>>,
}

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
    /// Accepts an IVF build for background execution by this session.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_create_index(
        &self,
        source: &RelationReference,
        index_namespace: &[String],
        index_name: String,
        vector_field: String,
        source_key_fields: Vec<String>,
        config: IvfConfig,
        writer_options: ParquetWriterOptions,
        partitions: Option<usize>,
    ) -> Result<()> {
        source.validate()?;
        writer_options.validate()?;
        validate_partitions(partitions)?;
        validate_build_request(&vector_field, &source_key_fields, config.nlist)?;
        let identifier = namespaced_index_identifier(index_namespace, &index_name)?;
        if self.index_exists_identifier(&identifier)? {
            return Err(Error::AlreadyExists(index_name));
        }
        let source_uri = parquet_source_uri(source)?.to_owned();
        let key = build_key(source, &identifier);
        let progress = LocalBuildProgress::default();
        let options = LocalBuildOptions {
            writer_options,
            partitions,
            threads: self.builds.dop(),
            progress: Some(progress.clone()),
        };
        let session = self.clone();
        self.builds.submit(key, None, progress, async move {
            session
                .create_index_with_identifier(
                    &source_uri,
                    identifier,
                    &vector_field,
                    &source_key_fields,
                    config,
                    &options,
                )
                .await
                .map(|_| ())
        })
    }

    /// Accepts an IVF refresh for background execution by this session.
    pub async fn submit_refresh_index(
        &self,
        source: &RelationReference,
        index_namespace: Vec<String>,
        index_name: String,
        config: Option<IvfConfig>,
        writer_options: ParquetWriterOptions,
        partitions: Option<usize>,
    ) -> Result<()> {
        source.validate()?;
        writer_options.validate()?;
        validate_partitions(partitions)?;
        let published = self
            .list_relation_indexes_in(&index_namespace, source)
            .await?
            .into_iter()
            .find(|index| index.name == index_name)
            .ok_or_else(|| Error::IndexNotFound(index_name.clone()))?;
        let source_uri = parquet_source_uri(source)?.to_owned();
        let identifier = namespaced_index_identifier(&index_namespace, &index_name)?;
        let key = build_key(source, &identifier);
        let progress = LocalBuildProgress::default();
        let options = LocalBuildOptions {
            writer_options,
            partitions,
            threads: self.builds.dop(),
            progress: Some(progress.clone()),
        };
        let session = self.clone();
        self.builds.submit(
            key,
            Some(published.current_snapshot_id),
            progress,
            async move {
                session
                    .refresh_index_with_identifier(&source_uri, identifier, config, &options)
                    .await
                    .map(|_| ())
            },
        )
    }

    /// Returns process-local build state combined with durable publication state.
    pub async fn index_build_status(
        &self,
        source: &RelationReference,
        index_namespace: &[String],
        index_name: &str,
    ) -> Result<IndexBuildStatus> {
        source.validate()?;
        let identifier = namespaced_index_identifier(index_namespace, index_name)?;
        let key = build_key(source, &identifier);
        let active = self.builds.snapshot(&key);
        let published = self
            .list_relation_indexes_in(index_namespace, source)
            .await?
            .into_iter()
            .find(|index| index.name == index_name);

        if let Some(active) = &active
            && matches!(
                active.state,
                ActiveBuildState::Pending | ActiveBuildState::Building
            )
        {
            return Ok(active_status(active, published.as_ref()));
        }
        if let Some(published) = published {
            let error = active.as_ref().and_then(|active| {
                if active.state == ActiveBuildState::Failed
                    && active.base_snapshot_id == Some(published.current_snapshot_id)
                {
                    active.error.clone()
                } else {
                    None
                }
            });
            let error_code = active.as_ref().and_then(|active| {
                if active.state == ActiveBuildState::Failed
                    && active.base_snapshot_id == Some(published.current_snapshot_id)
                {
                    active.failure_kind.map(BuildFailureKind::code)
                } else {
                    None
                }
            });
            if active.as_ref().is_some_and(|active| {
                active.state == ActiveBuildState::Failed
                    && active.base_snapshot_id != Some(published.current_snapshot_id)
            }) {
                self.builds.forget_failure(&key);
            }
            return Ok(IndexBuildStatus {
                state: IndexBuildState::Ready,
                progress: None,
                phase: None,
                completed: None,
                total: None,
                current_snapshot_id: Some(published.current_snapshot_id),
                error,
                error_code,
            });
        }
        if let Some(active) = active
            && active.state == ActiveBuildState::Failed
        {
            return Ok(IndexBuildStatus {
                state: IndexBuildState::Failed,
                progress: Some(active.progress.fraction),
                phase: Some(active.progress.phase),
                completed: Some(active.progress.completed),
                total: Some(active.progress.total),
                current_snapshot_id: None,
                error: active.error,
                error_code: active.failure_kind.map(BuildFailureKind::code),
            });
        }
        Err(Error::IndexNotFound(index_name.to_owned()))
    }

    /// Waits for an accepted build or an existing publication.
    pub async fn wait_for_index_build(
        &self,
        source: &RelationReference,
        index_namespace: &[String],
        index_name: &str,
    ) -> Result<()> {
        source.validate()?;
        let identifier = namespaced_index_identifier(index_namespace, index_name)?;
        let key = build_key(source, &identifier);
        if let Some(result) = self.builds.wait(&key).await {
            result?;
            return Ok(());
        }
        self.index_build_status(source, index_namespace, index_name)
            .await
            .map(|_| ())
    }

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
            IvfConfig::new(nlist, PostingEncoding::Source),
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
        self.create_index_with_identifier(
            source,
            local_index_identifier(index_name)?,
            vector_field,
            source_key_fields,
            config,
            options,
        )
        .await
    }

    async fn create_index_with_identifier(
        &self,
        source: &str,
        identifier: IndexIdentifier,
        vector_field: &str,
        source_key_fields: &[String],
        config: IvfConfig,
        options: &LocalBuildOptions,
    ) -> Result<PublishedIndex> {
        options.writer_options.validate()?;
        validate_partitions(options.partitions)?;
        let progress = options.progress.clone().unwrap_or_default();
        let parallel = build_context(options.threads)?;
        let mut build_lease = self.coordination.reserve_build(&identifier)?;
        if self.index_exists_identifier(&identifier)? {
            return Err(Error::AlreadyExists(identifier.name().to_owned()));
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
        let prepared = prepare_ivf_datafusion(
            self.context.read_table(source.provider)?,
            vector_field,
            source_key_fields,
            config,
            &progress,
        )
        .await?;
        let centroid_artifact = {
            let mut context = IvfCentroidsBuildContext {
                prepared: &prepared,
                parallel: &parallel,
                writer_options: &options.writer_options,
                progress: &progress,
                build_lease: &mut build_lease,
            };
            self.resolve_ivf_centroids(&source_reference, vector_field, config, &mut context)
                .await?
        };
        let build = build_ivf_postings(
            prepared,
            IvfPostingsSpec {
                vector_field,
                source_key_fields,
                config,
                trained: &centroid_artifact.trained,
                ivf_centroids: &centroid_artifact.reference,
                centroids: centroid_artifact.centroids,
            },
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
        self.refresh_index_with_identifier(
            source,
            local_index_identifier(index_name)?,
            config,
            options,
        )
        .await
    }

    async fn refresh_index_with_identifier(
        &self,
        source: &str,
        identifier: IndexIdentifier,
        config: Option<IvfConfig>,
        options: &LocalBuildOptions,
    ) -> Result<PublishedIndex> {
        options.writer_options.validate()?;
        validate_partitions(options.partitions)?;
        let progress = options.progress.clone().unwrap_or_default();
        let parallel = build_context(options.threads)?;
        let mut build_lease = self.coordination.reserve_build(&identifier)?;
        let loaded = self.load_entry(&identifier).await?;
        let current = loaded.metadata.current_snapshot()?;
        let source = self.bind_source(source).await?;
        let source_reference = source.reference.clone();
        if current.source.identity_key() != source_reference.identity_key() {
            return Err(Error::IndexNotFound(identifier.name().to_owned()));
        }
        let config = resolve_refresh_config(current, config)?;
        let snapshot_id = new_snapshot_id();
        let snapshot_root = self.warehouse.location(
            &format!(
                "indexes/{}/{snapshot_id}",
                loaded.metadata.index_uuid.simple()
            ),
            true,
        )?;
        build_lease.set_snapshot_root(&snapshot_root)?;
        let prepared = prepare_ivf_datafusion(
            self.context.read_table(source.provider)?,
            &current.vector_field,
            &current.source_key_fields,
            config,
            &progress,
        )
        .await?;
        let centroid_artifact = {
            let mut context = IvfCentroidsBuildContext {
                prepared: &prepared,
                parallel: &parallel,
                writer_options: &options.writer_options,
                progress: &progress,
                build_lease: &mut build_lease,
            };
            self.resolve_ivf_centroids(
                &source_reference,
                &current.vector_field,
                config,
                &mut context,
            )
            .await?
        };
        let build = build_ivf_postings(
            prepared,
            IvfPostingsSpec {
                vector_field: &current.vector_field,
                source_key_fields: &current.source_key_fields,
                config,
                trained: &centroid_artifact.trained,
                ivf_centroids: &centroid_artifact.reference,
                centroids: centroid_artifact.centroids,
            },
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

    async fn resolve_ivf_centroids(
        &self,
        source: &RelationReference,
        vector_field: &str,
        config: IvfConfig,
        context: &mut IvfCentroidsBuildContext<'_>,
    ) -> Result<ResolvedIvfCentroids> {
        let descriptor = IvfCentroidsDescriptor {
            source: source.clone(),
            vector_field: vector_field.to_owned(),
            dimension: i32::try_from(context.prepared.dimension)
                .map_err(|_| Error::InvalidSchema("vector dimension exceeds int32".into()))?,
            metric: config.metric,
            nlist: i32::try_from(context.prepared.nlist)
                .map_err(|_| Error::InvalidArgument("nlist exceeds int32".into()))?,
            clustering_profile_version: IVF_CLUSTERING_PROFILE_VERSION,
        };
        let fingerprint = descriptor.fingerprint()?;
        let owner = Uuid::new_v4();
        loop {
            match self
                .catalog
                .claim_ivf_centroids(&descriptor, owner, IVF_CENTROIDS_LEASE_MS)?
            {
                IvfCentroidsClaimResult::Ready(_) => {
                    return self
                        .load_ivf_centroids(&fingerprint, &descriptor, context.prepared)
                        .await;
                }
                IvfCentroidsClaimResult::Busy { .. } => {
                    tokio::time::sleep(IVF_CENTROIDS_WAIT_INTERVAL).await;
                }
                IvfCentroidsClaimResult::Claimed(claim) => {
                    return self.build_ivf_centroids(descriptor, claim, context).await;
                }
            }
        }
    }

    async fn load_ivf_centroids(
        &self,
        fingerprint: &str,
        descriptor: &IvfCentroidsDescriptor,
        prepared: &PreparedIvf,
    ) -> Result<ResolvedIvfCentroids> {
        let loaded = self.indexes.load_ivf_centroids(fingerprint).await?;
        if !loaded.metadata.descriptor.is_compatible_with(descriptor) {
            return Err(Error::InvalidMetadata(
                "IVF centroids descriptor does not match the requested build".into(),
            ));
        }
        let RelationReference::Parquet { uri } = &loaded.metadata.centroids else {
            return Err(Error::InvalidMetadata(
                "the local builder requires Parquet IVF centroids".into(),
            ));
        };
        let batch = self.parquet.read(uri, None).await?;
        let centroids = crate::ivf::read_centroids(&batch, prepared.nlist, prepared.dimension)?;
        let reference = IvfCentroidsReference::new(
            loaded.entry.fingerprint,
            loaded.entry.artifact_uuid,
            loaded.entry.metadata_location,
        )?;
        Ok(ResolvedIvfCentroids {
            reference,
            centroids: loaded.metadata.centroids,
            trained: reused_ivf(prepared, centroids)?,
        })
    }

    async fn build_ivf_centroids(
        &self,
        descriptor: IvfCentroidsDescriptor,
        claim: IvfCentroidsClaim,
        context: &mut IvfCentroidsBuildContext<'_>,
    ) -> Result<ResolvedIvfCentroids> {
        let heartbeat = match ClaimHeartbeat::start(Arc::clone(&self.catalog), claim.clone()) {
            Ok(heartbeat) => heartbeat,
            Err(error) => {
                let _ = self
                    .catalog
                    .abandon_ivf_centroids(&claim, &error.to_string());
                return Err(error);
            }
        };
        let result: Result<ResolvedIvfCentroids> = async {
            let artifact_uuid = Uuid::new_v4();
            let artifact_root = self
                .warehouse
                .location(&format!("indexes/{}/1", artifact_uuid.simple()), true)?;
            context.build_lease.add_snapshot_root(&artifact_root)?;
            let centroids_location = child_location(&artifact_root, "ivf_centroids", true)?;
            let trained = train_prepared_ivf(context.prepared, context.parallel, context.progress)?;
            write_ivf_centroids(
                &self.parquet,
                &centroids_location,
                &trained,
                context.writer_options,
                context.progress,
            )
            .await?;
            let metadata = IvfCentroidsMetadata {
                format_version: 1,
                artifact_uuid,
                fingerprint: claim.fingerprint.clone(),
                location: self
                    .indexes
                    .metadata_store()
                    .ivf_centroids_location(artifact_uuid)?,
                created_at_ms: now_ms()?,
                descriptor: descriptor.clone(),
                centroids: RelationReference::Parquet {
                    uri: centroids_location.clone(),
                },
            };
            let metadata_location = self
                .indexes
                .metadata_store()
                .write_ivf_centroids(&metadata)
                .await?;
            let entry =
                self.catalog
                    .publish_ivf_centroids(&claim, &metadata_location, &metadata)?;
            let reference = IvfCentroidsReference::new(
                entry.fingerprint,
                entry.artifact_uuid,
                entry.metadata_location,
            )?;
            Ok(ResolvedIvfCentroids {
                reference,
                centroids: metadata.centroids,
                trained,
            })
        }
        .await;
        let heartbeat_result = heartbeat.stop();
        match result {
            Ok(centroid_artifact) => {
                // Publication clears the owner. A simultaneous heartbeat may
                // observe that ready state after this owner has committed it.
                if let Err(error) = heartbeat_result
                    && !matches!(
                        &error,
                        Error::Catalog(relify_catalog::Error::IvfCentroidsClaimLost(fingerprint))
                            if fingerprint == &centroid_artifact.reference.fingerprint
                    )
                {
                    return Err(error);
                }
                Ok(centroid_artifact)
            }
            Err(error) => {
                let _ = self
                    .catalog
                    .abandon_ivf_centroids(&claim, &error.to_string());
                if let Ok(centroid_artifact) = self
                    .load_ivf_centroids(&claim.fingerprint, &descriptor, context.prepared)
                    .await
                {
                    return Ok(centroid_artifact);
                }
                Err(error)
            }
        }
    }
}

fn build_key(source: &RelationReference, identifier: &IndexIdentifier) -> BuildKey {
    BuildKey::new(source.identity_key(), identifier.clone())
}

fn parquet_source_uri(source: &RelationReference) -> Result<&str> {
    match source {
        RelationReference::Parquet { uri } => Ok(uri),
        RelationReference::Iceberg { .. } => Err(Error::InvalidArgument(
            "the installed builder currently supports Parquet source tables".into(),
        )),
    }
}

fn active_status(
    active: &super::build_coordinator::ActiveBuildSnapshot,
    published: Option<&super::IndexInfo>,
) -> IndexBuildStatus {
    IndexBuildStatus {
        state: match active.state {
            ActiveBuildState::Pending => IndexBuildState::Pending,
            ActiveBuildState::Building => IndexBuildState::Building,
            ActiveBuildState::Complete => IndexBuildState::Ready,
            ActiveBuildState::Failed => IndexBuildState::Failed,
        },
        progress: Some(active.progress.fraction),
        phase: Some(active.progress.phase),
        completed: Some(active.progress.completed),
        total: Some(active.progress.total),
        current_snapshot_id: published.map(|index| index.current_snapshot_id),
        error: active.error.clone(),
        error_code: active.failure_kind.map(BuildFailureKind::code),
    }
}

impl ClaimHeartbeat {
    fn start(catalog: Arc<dyn IndexCatalog>, claim: IvfCentroidsClaim) -> Result<Self> {
        let (stop, stopped) = mpsc::channel();
        let task = thread::Builder::new()
            .name("relify-ivf-centroids-heartbeat".into())
            .spawn(move || {
                loop {
                    match stopped.recv_timeout(IVF_CENTROIDS_RENEW_INTERVAL) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
                        Err(RecvTimeoutError::Timeout) => {
                            catalog.renew_ivf_centroids_claim(&claim, IVF_CENTROIDS_LEASE_MS)?;
                        }
                    }
                }
            })
            .map_err(|error| {
                Error::InvalidArgument(format!("failed to start IVF centroids heartbeat: {error}"))
            })?;
        Ok(Self { stop, task })
    }

    fn stop(self) -> Result<()> {
        let _ = self.stop.send(());
        self.task.join().map_err(|_| {
            Error::InvalidArgument("IVF centroids heartbeat thread panicked".into())
        })??;
        Ok(())
    }
}

fn now_ms() -> Result<i64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::InvalidArgument(error.to_string()))?
        .as_millis();
    i64::try_from(millis)
        .map_err(|_| Error::InvalidArgument("current timestamp is out of range".into()))
}

fn validate_partitions(partitions: Option<usize>) -> Result<()> {
    if partitions == Some(0) {
        return Err(Error::InvalidArgument(
            "partitions must be a positive integer".into(),
        ));
    }
    Ok(())
}

fn resolve_refresh_config(
    current: &relify_meta::IndexSnapshot,
    requested: Option<IvfConfig>,
) -> Result<IvfConfig> {
    let current_metric = DistanceMetric::from_metadata(&current.metric).ok_or_else(|| {
        Error::InvalidMetadata(format!("unsupported IVF metric: {}", current.metric))
    })?;
    let config = requested.unwrap_or(IvfConfig::with_metric(
        current.parameter_usize("nlist")?,
        PostingEncoding::from_snapshot(current)?,
        current_metric,
    ));
    if config.nlist == 0 || config.nlist > i32::MAX as usize {
        return Err(Error::InvalidArgument(
            "nlist must be in 1..=2147483647".into(),
        ));
    }
    if config.metric != current_metric {
        return Err(Error::InvalidArgument(
            "refresh cannot change the distance metric; create a new index instead".into(),
        ));
    }
    Ok(config)
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
