//! Session-scoped index construction and refresh.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parallite::ParalliteContext;
use relify_catalog::{IndexCatalog, SharedIvfClaim, SharedIvfClaimResult};
use relify_index::{
    InitialIndex, RefreshedIndex, new_snapshot_id, publish_initial, publish_refresh,
};
use relify_meta::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, PostingEncoding, RelationReference,
    SharedIvfDescriptor, SharedIvfMetadata, SharedIvfReference,
};
use tokio::sync::oneshot;
use uuid::Uuid;

use super::LocalSession;
use super::catalog::local_index_identifier;
use crate::builder::{
    IvfBuildContext, IvfPostingsSpec, PreparedIvf, TrainedIvf, build_ivf_postings,
    prepare_ivf_datafusion, reused_ivf, train_prepared_ivf, write_ivf_centroids,
};
use crate::coordination::BuildLease;
use crate::parquet::{ParquetWriterOptions, child_location};
use crate::progress::BuildPhase;
use crate::{Error, IvfConfig, LocalBuildProgress, PublishedIndex, Result};

const SHARED_IVF_LEASE_MS: i64 = 30_000;
const SHARED_IVF_RENEW_INTERVAL: Duration = Duration::from_secs(10);
const SHARED_IVF_WAIT_INTERVAL: Duration = Duration::from_millis(250);

struct ResolvedSharedIvf {
    reference: SharedIvfReference,
    centroids: RelationReference,
    trained: TrainedIvf,
}

struct SharedIvfBuildContext<'a> {
    prepared: &'a PreparedIvf,
    parallel: &'a ParalliteContext,
    writer_options: &'a ParquetWriterOptions,
    progress: &'a LocalBuildProgress,
    build_lease: &'a mut BuildLease,
}

struct ClaimHeartbeat {
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<relify_catalog::Result<()>>,
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
        let prepared = prepare_ivf_datafusion(
            self.context.read_table(source.provider)?,
            vector_field,
            source_key_fields,
            config,
            &progress,
        )
        .await?;
        let shared = {
            let mut context = SharedIvfBuildContext {
                prepared: &prepared,
                parallel: &parallel,
                writer_options: &options.writer_options,
                progress: &progress,
                build_lease: &mut build_lease,
            };
            self.resolve_shared_ivf(&source_reference, vector_field, config, &mut context)
                .await?
        };
        let build = build_ivf_postings(
            prepared,
            IvfPostingsSpec {
                vector_field,
                source_key_fields,
                config,
                trained: &shared.trained,
                shared_ivf: &shared.reference,
                centroids: shared.centroids,
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
        let shared = {
            let mut context = SharedIvfBuildContext {
                prepared: &prepared,
                parallel: &parallel,
                writer_options: &options.writer_options,
                progress: &progress,
                build_lease: &mut build_lease,
            };
            self.resolve_shared_ivf(
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
                trained: &shared.trained,
                shared_ivf: &shared.reference,
                centroids: shared.centroids,
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

    async fn resolve_shared_ivf(
        &self,
        source: &RelationReference,
        vector_field: &str,
        config: IvfConfig,
        context: &mut SharedIvfBuildContext<'_>,
    ) -> Result<ResolvedSharedIvf> {
        let descriptor = SharedIvfDescriptor {
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
                .claim_shared_ivf(&descriptor, owner, SHARED_IVF_LEASE_MS)?
            {
                SharedIvfClaimResult::Ready(_) => {
                    return self
                        .load_shared_ivf(&fingerprint, &descriptor, context.prepared)
                        .await;
                }
                SharedIvfClaimResult::Busy { .. } => {
                    tokio::time::sleep(SHARED_IVF_WAIT_INTERVAL).await;
                }
                SharedIvfClaimResult::Claimed(claim) => {
                    return self.build_shared_ivf(descriptor, claim, context).await;
                }
            }
        }
    }

    async fn load_shared_ivf(
        &self,
        fingerprint: &str,
        descriptor: &SharedIvfDescriptor,
        prepared: &PreparedIvf,
    ) -> Result<ResolvedSharedIvf> {
        let loaded = self.indexes.load_shared_ivf(fingerprint).await?;
        if !loaded.metadata.descriptor.is_compatible_with(descriptor) {
            return Err(Error::InvalidMetadata(
                "shared IVF descriptor does not match the requested build".into(),
            ));
        }
        let RelationReference::Parquet { uri } = &loaded.metadata.centroids else {
            return Err(Error::InvalidMetadata(
                "the local builder requires Parquet shared centroids".into(),
            ));
        };
        let batch = self.parquet.read(uri, None).await?;
        let centroids = crate::ivf::read_centroids(&batch, prepared.nlist, prepared.dimension)?;
        let reference = SharedIvfReference::new(
            loaded.entry.fingerprint,
            loaded.entry.artifact_uuid,
            loaded.entry.metadata_location,
        )?;
        Ok(ResolvedSharedIvf {
            reference,
            centroids: loaded.metadata.centroids,
            trained: reused_ivf(prepared, centroids)?,
        })
    }

    async fn build_shared_ivf(
        &self,
        descriptor: SharedIvfDescriptor,
        claim: SharedIvfClaim,
        context: &mut SharedIvfBuildContext<'_>,
    ) -> Result<ResolvedSharedIvf> {
        let heartbeat = ClaimHeartbeat::start(Arc::clone(&self.catalog), claim.clone());
        let artifact_uuid = Uuid::new_v4();
        let artifact_root = self
            .warehouse
            .location(&format!("indexes/{}/1", artifact_uuid.simple()), true)?;
        context.build_lease.add_snapshot_root(&artifact_root)?;
        let centroids_location = child_location(&artifact_root, "ivf_centroids", true)?;
        let result: Result<ResolvedSharedIvf> = async {
            let trained = train_prepared_ivf(context.prepared, context.parallel, context.progress)?;
            write_ivf_centroids(
                &self.parquet,
                &centroids_location,
                &trained,
                context.writer_options,
                context.progress,
            )
            .await?;
            let metadata = SharedIvfMetadata {
                format_version: 1,
                artifact_uuid,
                fingerprint: claim.fingerprint.clone(),
                location: self
                    .indexes
                    .metadata_store()
                    .shared_ivf_location(artifact_uuid)?,
                created_at_ms: now_ms()?,
                descriptor: descriptor.clone(),
                centroids: RelationReference::Parquet {
                    uri: centroids_location.clone(),
                },
            };
            let metadata_location = self
                .indexes
                .metadata_store()
                .write_shared_ivf(&metadata)
                .await?;
            let entry = self
                .catalog
                .publish_shared_ivf(&claim, &metadata_location, &metadata)?;
            let reference = SharedIvfReference::new(
                entry.fingerprint,
                entry.artifact_uuid,
                entry.metadata_location,
            )?;
            Ok(ResolvedSharedIvf {
                reference,
                centroids: metadata.centroids,
                trained,
            })
        }
        .await;
        let heartbeat_result = heartbeat.stop().await;
        match result {
            Ok(shared) => {
                // Publication clears the owner. A simultaneous heartbeat may
                // observe that ready state after this owner has committed it.
                if let Err(error) = heartbeat_result
                    && !matches!(
                        &error,
                        Error::Catalog(relify_catalog::Error::SharedIvfClaimLost(fingerprint))
                            if fingerprint == &shared.reference.fingerprint
                    )
                {
                    return Err(error);
                }
                Ok(shared)
            }
            Err(error) => {
                let _ = self.catalog.abandon_shared_ivf(&claim, &error.to_string());
                if let Ok(shared) = self
                    .load_shared_ivf(&claim.fingerprint, &descriptor, context.prepared)
                    .await
                {
                    return Ok(shared);
                }
                Err(error)
            }
        }
    }
}

impl ClaimHeartbeat {
    fn start(catalog: Arc<dyn IndexCatalog>, claim: SharedIvfClaim) -> Self {
        let (stop, mut stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stopped => return Ok(()),
                    () = tokio::time::sleep(SHARED_IVF_RENEW_INTERVAL) => {
                        catalog.renew_shared_ivf_claim(&claim, SHARED_IVF_LEASE_MS)?;
                    }
                }
            }
        });
        Self { stop, task }
    }

    async fn stop(self) -> Result<()> {
        let _ = self.stop.send(());
        self.task.await.map_err(|error| {
            Error::InvalidArgument(format!("shared IVF heartbeat failed: {error}"))
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
