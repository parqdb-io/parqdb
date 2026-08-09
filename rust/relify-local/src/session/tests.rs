//! Embedded session integration tests.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arrow::array::{Array, Float32Builder, Int32Array, Int64Array, ListBuilder, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use relify_meta::{IndexMetadata, PostingEncoding, SnapshotLogEntry};
use rusqlite::Connection;
use tempfile::TempDir;
use uuid::Uuid;

use super::*;
use crate::local_uri::directory_to_file_uri;
use crate::{IvfConfig, MaintenanceKind, relify_session_config};

struct MemoryEntry {
    entry: CatalogEntry,
    source_identity: String,
}

fn age_catalog_tombstones(session: &LocalSession) {
    let connection = Connection::open(session.state_root.join("catalog.sqlite")).unwrap();
    connection
        .execute("UPDATE catalog_tombstones SET unreachable_since_ms = 0", [])
        .unwrap();
}

#[test]
fn metadata_cache_config_is_shared_with_repository_handles() {
    let temporary = TempDir::new().unwrap();
    let config = MetadataCacheConfig::new(7, 4096);
    let datafusion_config = relify_session_config()
        .set_str("relify.metadata.cache.max_entries", "7")
        .set_str("relify.metadata.cache.max_bytes", "4096");
    let session = LocalSession::open_with_options(
        temporary.path(),
        LocalSessionOptions::new(datafusion_config, RuntimeEnvBuilder::default()),
    )
    .unwrap();
    let repository = session.index_repository();

    assert_eq!(session.metadata_cache_config(), config);
    assert_eq!(repository.metadata_store().cache_config(), config);
}

#[tokio::test]
async fn metadata_cache_config_tracks_datafusion_session_set() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path()).unwrap();

    session
        .context()
        .sql("SET relify.metadata.cache.max_entries = 7")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    session
        .context()
        .sql("SET relify.metadata.cache.max_bytes = 4096")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(
        session.metadata_cache_config(),
        MetadataCacheConfig::new(7, 4096)
    );
    assert_eq!(
        session.index_repository().metadata_store().cache_config(),
        MetadataCacheConfig::new(7, 4096)
    );
}

#[test]
fn session_preserves_datafusion_config_and_runtime() {
    let temporary = TempDir::new().unwrap();
    let config = relify_session_config().with_target_partitions(3);
    let memory_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
    let runtime = RuntimeEnvBuilder::default().with_memory_pool(Arc::clone(&memory_pool));
    let session = LocalSession::open_with_options(
        temporary.path(),
        LocalSessionOptions::new(config, runtime),
    )
    .unwrap();
    let context = session.context();

    assert_eq!(context.copied_config().target_partitions(), 3);
    assert!(Arc::ptr_eq(
        &context.runtime_env().memory_pool,
        &memory_pool
    ));
}

#[derive(Default)]
struct MemoryCatalog {
    entries: Mutex<BTreeMap<IndexIdentifier, MemoryEntry>>,
}

impl IndexCatalog for MemoryCatalog {
    fn load(&self, identifier: &IndexIdentifier) -> relify_catalog::Result<CatalogEntry> {
        self.entries
            .lock()
            .unwrap()
            .get(identifier)
            .map(|entry| entry.entry.clone())
            .ok_or_else(|| CatalogError::IndexNotFound(identifier.clone()))
    }

    fn register(
        &self,
        identifier: &IndexIdentifier,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> relify_catalog::Result<()> {
        metadata.validate()?;
        let mut entries = self.entries.lock().unwrap();
        if entries.contains_key(identifier) {
            return Err(CatalogError::AlreadyExists(identifier.clone()));
        }
        entries.insert(
            identifier.clone(),
            MemoryEntry {
                entry: CatalogEntry {
                    identifier: identifier.clone(),
                    metadata_location: metadata_location.to_owned(),
                },
                source_identity: metadata.current_snapshot()?.source.exact_state_key(),
            },
        );
        Ok(())
    }

    fn list(&self, namespace: &[String]) -> relify_catalog::Result<Vec<IndexIdentifier>> {
        Ok(self
            .entries
            .lock()
            .unwrap()
            .keys()
            .filter(|identifier| identifier.namespace() == namespace)
            .cloned()
            .collect())
    }

    fn find_by_source(
        &self,
        namespace: &[String],
        source: &RelationReference,
    ) -> relify_catalog::Result<Vec<CatalogEntry>> {
        let source_identity = source.exact_state_key();
        Ok(self
            .entries
            .lock()
            .unwrap()
            .values()
            .filter(|entry| entry.entry.identifier.namespace() == namespace)
            .filter(|entry| entry.source_identity == source_identity)
            .map(|entry| entry.entry.clone())
            .collect())
    }
}

#[test]
fn index_name_is_deliberately_narrow() {
    assert!(validate_index_name("documents_embedding").is_ok());
    assert!(validate_index_name("analytics.documents").is_err());
    assert!(validate_index_name("1index").is_err());
}

#[test]
fn session_and_parquet_store_share_one_datafusion_context() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path()).unwrap();

    assert_eq!(
        session.context().session_id(),
        session.parquet.context().session_id()
    );
}

#[test]
fn explicit_sqlite_coordination_is_scoped_to_the_catalog_database() {
    let temporary = TempDir::new().unwrap();
    let warehouse_path = temporary.path().join("warehouse");
    std::fs::create_dir_all(&warehouse_path).unwrap();
    let warehouse = directory_to_file_uri(&warehouse_path).unwrap();
    let first_database = temporary.path().join("state").join("first.sqlite");
    let second_database = temporary.path().join("state").join("second.sqlite");
    let first = LocalSession::open_sqlite(&first_database, &warehouse, HashMap::new()).unwrap();
    let same_catalog =
        LocalSession::open_sqlite(&first_database, &warehouse, HashMap::new()).unwrap();
    let other_catalog =
        LocalSession::open_sqlite(&second_database, &warehouse, HashMap::new()).unwrap();
    let identifier = IndexIdentifier::root("documents_embedding").unwrap();

    let lease = first.coordination.reserve_build(&identifier).unwrap();
    assert!(matches!(
        same_catalog.coordination.reserve_build(&identifier),
        Err(Error::BuildAlreadyRunning(running)) if running == identifier
    ));
    assert!(
        other_catalog
            .coordination
            .reserve_build(&identifier)
            .is_ok()
    );
    drop(lease);
    assert!(same_catalog.coordination.reserve_build(&identifier).is_ok());
}

async fn write_direct_pid_source(store: &ParquetStore, path: &Path) {
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for vector in [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]] {
        vectors.values().append_slice(&vector);
        vectors.append(true);
    }
    let vectors = vectors.finish();
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("source_pid", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("embedding", vectors.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![20, 10, 30])),
            Arc::new(StringArray::from(vec!["twenty", "ten", "thirty"])),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let location = directory_to_file_uri(path).unwrap();
    store.write(&location, &source).await.unwrap();
}

async fn write_direct_pid_relations(store: &ParquetStore, root: &Path) -> (PathBuf, PathBuf) {
    let centroids_path = root.join("centroids");
    let postings_path = root.join("postings");
    let mut centroids = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    centroids.values().append_slice(&[0.0, 0.0]);
    centroids.append(true);
    centroids.values().append_slice(&[10.0, 0.0]);
    centroids.append(true);
    let centroids = centroids.finish();
    let centroid_table = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int32, false),
            Field::new("centroid", centroids.data_type().clone(), false),
        ])),
        vec![Arc::new(Int32Array::from(vec![0, 1])), Arc::new(centroids)],
    )
    .unwrap();
    store
        .write(
            &directory_to_file_uri(&centroids_path).unwrap(),
            &centroid_table,
        )
        .await
        .unwrap();
    for (cid, keys) in [(0, vec![20, 10]), (1, vec![30])] {
        let postings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "key_1",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(keys))],
        )
        .unwrap();
        let partition = postings_path.join(format!("cid={cid}"));
        store
            .write(&directory_to_file_uri(&partition).unwrap(), &postings)
            .await
            .unwrap();
    }
    (centroids_path, postings_path)
}

fn direct_pid_metadata(
    session: &LocalSession,
    source_path: &Path,
    centroids_path: &Path,
    postings_path: &Path,
) -> IndexMetadata {
    let index_uuid = Uuid::new_v4();
    let snapshot_id = 701;
    let timestamp_ms = 1_750_000_000_000;
    let snapshot = IndexSnapshot {
        snapshot_id,
        sequence_number: 1,
        timestamp_ms,
        summary: BTreeMap::new(),
        source: RelationReference::Parquet {
            uri: directory_to_file_uri(&source_path.canonicalize().unwrap()).unwrap(),
        },
        vector_field: "embedding".into(),
        source_key_fields: vec!["source_pid".into()],
        index_family: "ivf".into(),
        index_schema_version: 1,
        metric: "l2_squared".into(),
        parameters: BTreeMap::from([
            ("dimension".into(), "2".into()),
            ("nlist".into(), "2".into()),
            ("ntotal".into(), "3".into()),
            ("store_vectors".into(), "false".into()),
        ]),
        index_relations: BTreeMap::from([
            (
                "ivf_centroids".into(),
                RelationReference::Parquet {
                    uri: directory_to_file_uri(centroids_path).unwrap(),
                },
            ),
            (
                "ivf_postings".into(),
                RelationReference::Parquet {
                    uri: directory_to_file_uri(postings_path).unwrap(),
                },
            ),
        ]),
    };
    IndexMetadata {
        format_version: 1,
        index_uuid,
        location: session
            .indexes
            .metadata_store()
            .index_location(index_uuid)
            .unwrap(),
        last_updated_ms: timestamp_ms,
        last_sequence_number: 1,
        current_snapshot_id: snapshot_id,
        snapshots: vec![snapshot],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms,
            snapshot_id,
        }],
        properties: BTreeMap::new(),
    }
}

async fn direct_pid_fixture() -> (TempDir, LocalSession, PathBuf) {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    write_direct_pid_source(&session.parquet, &source_path).await;
    let (centroids_path, postings_path) =
        write_direct_pid_relations(&session.parquet, temporary.path()).await;
    let metadata = direct_pid_metadata(&session, &source_path, &centroids_path, &postings_path);
    let metadata_location = session
        .indexes
        .metadata_store()
        .write_initial(&metadata)
        .await
        .unwrap();
    let identifier = IndexIdentifier::root("direct_index").unwrap();
    session
        .catalog
        .register(&identifier, &metadata_location, &metadata)
        .unwrap();
    (temporary, session, source_path)
}

#[tokio::test]
async fn source_bindings_are_session_scoped_and_reused_by_queries() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;
    assert_eq!(session.source_binding_count().unwrap(), 0);

    let first = session
        .describe(source_path.to_str().unwrap())
        .await
        .unwrap();
    let second = session.describe(&first.uri).await.unwrap();
    assert_eq!(second, first);
    assert_eq!(session.source_binding_count().unwrap(), 1);

    let binding = session.source_binding(&first.uri).unwrap().unwrap();
    assert!(
        session
            .context()
            .table_exist(binding.table_name.clone())
            .unwrap()
    );
    let request = SearchRequest {
        source: RelationReference::Parquet {
            uri: first.uri.clone(),
        },
        index: Some("direct_index".into()),
        column: None,
        query: vec![0.0, 0.0],
        nprobe: Some(1),
        limit: 2,
        projection: Some(vec!["source_pid".into(), "label".into()]),
        filter: None,
        bypass_index: false,
    };

    session.search(&request).await.unwrap();
    session.search(&request).await.unwrap();

    assert_eq!(session.source_binding_count().unwrap(), 1);
    assert!(session.context().table_exist(binding.table_name).unwrap());
}

#[tokio::test]
async fn registered_source_binding_reuses_the_datafusion_table_provider() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;
    let source_uri = directory_to_file_uri(&source_path).unwrap();
    let provider = session
        .parquet
        .dataframe(&source_uri)
        .await
        .unwrap()
        .into_view();
    session
        .context()
        .register_table("documents", Arc::clone(&provider))
        .unwrap();

    let description = session
        .bind_registered_source("documents", &source_uri)
        .await
        .unwrap();
    let binding = session.source_binding(&description.uri).unwrap().unwrap();

    assert_eq!(binding.table_name, "documents");
    assert!(Arc::ptr_eq(&binding.provider, &provider));
    assert_eq!(binding.schema, provider.schema());
}

#[tokio::test]
async fn backend_cache_materializes_and_releases_complete_index_snapshots() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;

    let cached = session.cache_index("direct_index").await.unwrap();

    assert_eq!(
        cached,
        IndexCacheInfo {
            name: "direct_index".into(),
            snapshot_id: 701,
            relation_count: 2,
            resident_bytes: cached.resident_bytes,
        }
    );
    assert!(cached.resident_bytes > 0);
    assert!(session.is_index_cached("direct_index").unwrap());
    assert_eq!(session.cache_index("direct_index").await.unwrap(), cached);
    let loaded = session
        .select_index(
            &RelationReference::Parquet {
                uri: directory_to_file_uri(&source_path).unwrap(),
            },
            Some("direct_index"),
            None,
        )
        .await
        .unwrap();
    let centroid_key = source::relation_key(
        loaded
            .metadata
            .current_snapshot()
            .unwrap()
            .index_relations
            .get("ivf_centroids")
            .unwrap(),
    );
    assert_eq!(
        session
            .cached_centroid_values(&centroid_key)
            .unwrap()
            .unwrap()
            .as_ref(),
        [0.0, 0.0, 10.0, 0.0]
    );
    let request = SearchRequest {
        source: RelationReference::Parquet {
            uri: directory_to_file_uri(&source_path).unwrap(),
        },
        index: Some("direct_index".into()),
        column: None,
        query: vec![0.0, 0.0],
        nprobe: Some(1),
        limit: 2,
        projection: Some(vec!["source_pid".into()]),
        filter: None,
        bypass_index: false,
    };
    let plan = session.explain_search(&request, false).await.unwrap();
    assert!(plan.contains("IvfTopKExec"), "{plan}");
    assert!(plan.contains("CachedIvfScanExec"), "{plan}");
    assert!(!plan.contains("FilterExec"), "{plan}");
    assert!(!plan.contains("cid IN"), "{plan}");
    let sql = session.search_sql(&request).await.unwrap();
    let cached_sql_plan = session
        .context()
        .sql(&sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let cached_sql_plan = datafusion::physical_plan::displayable(cached_sql_plan.as_ref())
        .indent(false)
        .to_string();
    assert!(
        cached_sql_plan.contains("CachedIvfScanExec"),
        "{cached_sql_plan}"
    );
    let cached_parquet_scans = cached_sql_plan.matches("file_type=parquet").count();
    assert!(session.uncache_index("direct_index").unwrap());
    assert!(!session.uncache_index("direct_index").unwrap());
    assert!(!session.is_index_cached("direct_index").unwrap());
    let uncached_sql_plan = session
        .context()
        .sql(&sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let uncached_sql_plan = datafusion::physical_plan::displayable(uncached_sql_plan.as_ref())
        .indent(false)
        .to_string();
    assert!(
        uncached_sql_plan.matches("file_type=parquet").count() > cached_parquet_scans,
        "{uncached_sql_plan}"
    );
}

#[tokio::test]
async fn cached_source_join_uses_runtime_filter_and_matches_uncached_results() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;
    let request = SearchRequest {
        source: RelationReference::Parquet {
            uri: directory_to_file_uri(&source_path).unwrap(),
        },
        index: Some("direct_index".into()),
        column: None,
        query: vec![0.0, 0.0],
        nprobe: Some(1),
        limit: 2,
        projection: Some(vec!["source_pid".into(), "label".into()]),
        filter: None,
        bypass_index: false,
    };
    let (uncached, uncached_schema) = session.search(&request).await.unwrap();
    let sql = session.search_sql(&request).await.unwrap();

    session.cache_index("direct_index").await.unwrap();
    let plan = session.explain_search(&request, false).await.unwrap();
    assert!(plan.contains("CachedIvfScanExec"), "{plan}");
    assert!(plan.contains("runtime_filters=1"), "{plan}");
    let (cached, cached_schema) = session.search(&request).await.unwrap();
    assert_eq!(cached_schema, uncached_schema);
    assert_eq!(cached, uncached);
    let cached_sql = session
        .context()
        .sql(&sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert!(session.uncache_index("direct_index").unwrap());
    let uncached_sql = session
        .context()
        .sql(&sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(cached_sql, uncached_sql);
}

#[tokio::test]
async fn accepts_a_caller_supplied_index_catalog() {
    let temporary = TempDir::new().unwrap();
    let root = temporary.path().join("relify");
    let source_path = temporary.path().join("source");
    let catalog = Arc::new(MemoryCatalog::default());
    let session = LocalSession::with_catalog(&root, catalog.clone()).unwrap();
    write_direct_pid_source(&session.parquet, &source_path).await;

    session
        .create_index(
            source_path.to_str().unwrap(),
            "documents_embedding",
            "embedding",
            &["source_pid".into()],
            2,
        )
        .await
        .unwrap();

    let identifier = IndexIdentifier::root("documents_embedding").unwrap();
    assert_eq!(catalog.load(&identifier).unwrap().identifier, identifier);
    assert_eq!(session.list_indexes().unwrap(), ["documents_embedding"]);
    assert!(!root.join("catalog.sqlite").exists());
}

#[tokio::test]
async fn manages_published_indexes_through_catalog_and_source_scoped_operations() {
    let (temporary, session, source_path) = direct_pid_fixture().await;
    let indexes = session
        .list_source_indexes(source_path.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(
        indexes,
        [IndexInfo {
            name: "direct_index".into(),
            column: "embedding".into(),
            family: "ivf".into(),
            metric: "l2_squared".into(),
            parameters: BTreeMap::from([
                ("dimension".into(), "2".into()),
                ("nlist".into(), "2".into()),
                ("ntotal".into(), "3".into()),
                ("store_vectors".into(), "false".into()),
            ]),
            current_snapshot_id: 701,
        }]
    );

    let other_source = temporary.path().join("other-source");
    write_direct_pid_source(&session.parquet, &other_source).await;
    assert!(matches!(
        session
            .drop_source_index(other_source.to_str().unwrap(), "direct_index")
            .await,
        Err(Error::IndexNotFound(name)) if name == "direct_index"
    ));

    let metadata_location = session.load_index_entry("direct_index").await.unwrap().0;
    session
        .drop_source_index(source_path.to_str().unwrap(), "direct_index")
        .await
        .unwrap();
    assert_eq!(session.list_indexes().unwrap(), Vec::<String>::new());

    session
        .register_index("direct_index", &metadata_location)
        .await
        .unwrap();
    assert_eq!(session.list_indexes().unwrap(), ["direct_index"]);
    session.drop_index("direct_index").unwrap();
    assert_eq!(session.list_indexes().unwrap(), Vec::<String>::new());
}

#[tokio::test]
async fn searches_an_index_by_its_persisted_source_key() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;
    let (batches, schema) = session
        .search(&SearchRequest {
            source: RelationReference::Parquet {
                uri: directory_to_file_uri(&source_path).unwrap(),
            },
            index: Some("direct_index".into()),
            column: None,
            query: vec![0.0, 0.0],
            nprobe: Some(1),
            limit: 2,
            projection: Some(vec!["source_pid".into(), "label".into()]),
            filter: None,
            bypass_index: false,
        })
        .await
        .unwrap();

    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["source_pid", "label", "_distance"]
    );
    let mut pids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    pids.sort_unstable();
    assert_eq!(pids, [10, 20]);
}

#[tokio::test]
async fn large_nprobe_pushes_a_static_filter_into_uncached_postings() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for value in 0_u16..256 {
        vectors.values().append_slice(&[f32::from(value), 0.0]);
        vectors.append(true);
    }
    let vectors = vectors.finish();
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("embedding", vectors.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0_i64..256)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    session
        .parquet
        .write(&directory_to_file_uri(&source_path).unwrap(), &source)
        .await
        .unwrap();
    session
        .create_index(
            source_path.to_str().unwrap(),
            "large_index",
            "embedding",
            &["id".into()],
            256,
        )
        .await
        .unwrap();
    let request = SearchRequest {
        source: RelationReference::Parquet {
            uri: directory_to_file_uri(&source_path).unwrap(),
        },
        index: Some("large_index".into()),
        column: None,
        query: vec![0.0, 0.0],
        nprobe: Some(129),
        limit: 3,
        projection: Some(vec!["id".into()]),
        filter: None,
        bypass_index: false,
    };

    let (batches, _) = session.search(&request).await.unwrap();
    let ids = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let plan = session.explain_search(&request, false).await.unwrap();

    assert_eq!(ids, [0, 1, 2]);
    assert!(plan.contains("IvfTopKExec"), "{plan}");
    assert!(!plan.contains("join_type=RightSemi"), "{plan}");
    assert!(plan.contains("full_filters="), "{plan}");
    assert!(plan.contains("file_groups={"), "{plan}");
    assert!(!plan.contains("/cid=129/"), "{plan}");
    assert!(!plan.contains("FilterExec"), "{plan}");
}

#[tokio::test]
async fn lvq_indexes_build_and_query_through_uncached_and_cached_paths() {
    for (encoding, name) in [
        (PostingEncoding::Lvq4, "lvq4_index"),
        (PostingEncoding::Lvq8, "lvq8_index"),
    ] {
        assert_lvq_index(encoding, name).await;
    }
}

async fn assert_lvq_index(encoding: PostingEncoding, name: &str) {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    write_direct_pid_source(&session.parquet, &source_path).await;
    let published = session
        .create_index_with_options(
            source_path.to_str().unwrap(),
            name,
            "embedding",
            &["source_pid".into()],
            IvfConfig::new(2, encoding),
            &LocalBuildOptions::default(),
        )
        .await
        .unwrap();
    let snapshot = published.metadata.current_snapshot().unwrap();
    assert_eq!(snapshot.index_schema_version, 2);
    assert_eq!(PostingEncoding::from_snapshot(snapshot).unwrap(), encoding);
    let RelationReference::Parquet { uri } = &snapshot.index_relations["ivf_postings"] else {
        panic!("local postings must use Parquet");
    };
    let postings = session
        .parquet
        .partitioned_dataframe(uri, vec![("cid".into(), DataType::Int32)])
        .await
        .unwrap();
    assert_lvq_postings_schema(postings.schema().inner());

    let request = SearchRequest {
        source: RelationReference::Parquet {
            uri: directory_to_file_uri(&source_path).unwrap(),
        },
        index: Some(name.into()),
        column: None,
        query: vec![10.0, 0.0],
        nprobe: Some(2),
        limit: 1,
        projection: Some(vec!["source_pid".into()]),
        filter: None,
        bypass_index: false,
    };
    let (uncached, _) = session.search(&request).await.unwrap();
    assert_nearest_pid(&uncached);
    let plan = session.explain_search(&request, false).await.unwrap();
    assert!(plan.contains("IvfTopKExec"), "{plan}");
    assert!(
        plan.contains(&format!("encoding={}", encoding.as_str())),
        "{plan}"
    );

    let mut source_projection = request.clone();
    source_projection.projection = Some(vec!["source_pid".into(), "label".into()]);
    let (joined, _) = session.search(&source_projection).await.unwrap();
    assert_nearest_pid(&joined);
    assert_eq!(
        arrow::util::display::array_value_to_string(joined[0].column(1), 0).unwrap(),
        "thirty"
    );

    session.cache_index(name).await.unwrap();
    let (cached, _) = session.search(&request).await.unwrap();
    assert_nearest_pid(&cached);
}

fn assert_lvq_postings_schema(schema: &Schema) {
    assert_eq!(
        schema.field_with_name("offset").unwrap().data_type(),
        &DataType::Float32
    );
    assert_eq!(
        schema.field_with_name("scale").unwrap().data_type(),
        &DataType::Float32
    );
    assert_eq!(
        schema.field_with_name("code").unwrap().data_type(),
        &DataType::BinaryView
    );
}

fn assert_nearest_pid(batches: &[RecordBatch]) {
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[30]);
}

#[tokio::test]
async fn filters_source_rows_before_top_k() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;
    let (batches, _) = session
        .search(&SearchRequest {
            source: RelationReference::Parquet {
                uri: directory_to_file_uri(&source_path).unwrap(),
            },
            index: Some("direct_index".into()),
            column: None,
            query: vec![0.0, 0.0],
            nprobe: Some(1),
            limit: 1,
            projection: Some(vec!["source_pid".into()]),
            filter: Some("source_pid >= 20".into()),
            bypass_index: false,
        })
        .await
        .unwrap();
    let pids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(pids.values(), &[20]);
}

#[tokio::test]
async fn exact_search_does_not_require_a_published_index() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    write_direct_pid_source(&session.parquet, &source_path).await;

    let (batches, _) = session
        .search(&SearchRequest {
            source: RelationReference::Parquet {
                uri: directory_to_file_uri(&source_path).unwrap(),
            },
            index: None,
            column: None,
            query: vec![10.0, 0.0],
            nprobe: None,
            limit: 1,
            projection: Some(vec!["source_pid".into()]),
            filter: None,
            bypass_index: true,
        })
        .await
        .unwrap();
    let pids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(pids.values(), &[30]);
    assert_eq!(session.list_indexes().unwrap(), Vec::<String>::new());
}

#[tokio::test]
async fn refresh_reuses_identity_and_publishes_a_new_snapshot() {
    let (_temporary, session, source_path) = direct_pid_fixture().await;
    let before = session.load_index_entry("direct_index").await.unwrap();
    let before_metadata = IndexMetadata::from_json_slice(before.1.as_bytes()).unwrap();

    let refreshed = session
        .refresh_index_with_options(
            source_path.to_str().unwrap(),
            "direct_index",
            None,
            &LocalBuildOptions::default(),
        )
        .await
        .unwrap();

    assert_ne!(refreshed.metadata_location, before.0);
    assert_eq!(refreshed.metadata.snapshots.len(), 2);
    assert_eq!(refreshed.metadata.index_uuid, before_metadata.index_uuid);
    assert_eq!(refreshed.metadata.location, before_metadata.location);
    assert_eq!(
        refreshed.metadata.snapshots[0],
        before_metadata.snapshots[0]
    );
    assert_eq!(
        session.load_index_entry("direct_index").await.unwrap().0,
        refreshed.metadata_location
    );
}

#[tokio::test]
async fn orphan_removal_preserves_reachable_data_and_removes_dropped_data() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    write_direct_pid_source(&session.parquet, &source_path).await;
    session
        .create_index(
            source_path.to_str().unwrap(),
            "managed_index",
            "embedding",
            &["source_pid".into()],
            2,
        )
        .await
        .unwrap();

    assert!(
        session
            .remove_orphans(i64::MAX, true)
            .await
            .unwrap()
            .is_empty()
    );
    session.drop_index("managed_index").unwrap();
    assert!(
        session
            .remove_orphans(i64::MAX, true)
            .await
            .unwrap()
            .is_empty()
    );
    age_catalog_tombstones(&session);
    let candidates = session.remove_orphans(i64::MAX, true).await.unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.kind.as_str())
            .collect::<Vec<_>>(),
        ["index_data", "metadata"]
    );
    assert!(source_path.is_dir());

    assert_eq!(
        session.remove_orphans(i64::MAX, false).await.unwrap(),
        candidates
    );
    assert!(source_path.is_dir());
    assert!(
        session
            .remove_orphans(i64::MAX, true)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn orphan_removal_invalidates_deleted_metadata_before_registration() {
    let (_temporary, session, _source_path) = direct_pid_fixture().await;
    let metadata_location = session.load_index_entry("direct_index").await.unwrap().0;
    session.drop_index("direct_index").unwrap();
    age_catalog_tombstones(&session);

    let removed = session.remove_orphans(i64::MAX, false).await.unwrap();
    assert!(
        removed
            .iter()
            .any(|object| object.kind == MaintenanceKind::Metadata)
    );
    assert!(!file_uri_to_path(&metadata_location).unwrap().exists());

    assert!(
        session
            .register_index("resurrected", &metadata_location)
            .await
            .is_err()
    );
    assert!(session.list_indexes().unwrap().is_empty());
}

#[tokio::test]
async fn orphan_removal_preserves_an_unpublished_active_build_root() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let index_uuid = Uuid::new_v4();
    let snapshot_root = session
        .warehouse
        .location(&format!("indexes/{}/1", index_uuid.simple()), true)
        .unwrap();
    let object = session
        .warehouse
        .location(
            &format!("indexes/{}/1/postings/part-0.parquet", index_uuid.simple()),
            false,
        )
        .unwrap();
    let identifier = IndexIdentifier::root("managed_index").unwrap();
    let mut lease = session.coordination.reserve_build(&identifier).unwrap();
    lease.set_snapshot_root(&snapshot_root).unwrap();
    session
        .warehouse
        .put_new(&object, Bytes::from_static(b"active"))
        .await
        .unwrap();
    let object_path = file_uri_to_path(&object).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(object_path)
        .unwrap();
    file.set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH))
        .unwrap();

    assert!(
        session
            .remove_orphans(i64::MAX, false)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(session.warehouse.head(&object).await.is_ok());

    drop(lease);
    let removed = session.remove_orphans(i64::MAX, false).await.unwrap();
    assert_eq!(removed.len(), 1);
    assert_eq!(removed[0].kind, MaintenanceKind::IndexData);
    assert_eq!(removed[0].reference, snapshot_root);
}

#[tokio::test]
async fn native_build_path_honors_the_cross_session_reservation() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    write_direct_pid_source(&session.parquet, &source_path).await;
    let identifier = IndexIdentifier::root("managed_index").unwrap();
    let lease = session.coordination.reserve_build(&identifier).unwrap();

    let error = session
        .create_index(
            source_path.to_str().unwrap(),
            "managed_index",
            "embedding",
            &["source_pid".into()],
            2,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        Error::BuildAlreadyRunning(running) if running == identifier
    ));

    drop(lease);
    session
        .create_index(
            source_path.to_str().unwrap(),
            "managed_index",
            "embedding",
            &["source_pid".into()],
            2,
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn published_builds_use_final_immutable_snapshot_paths() {
    let temporary = TempDir::new().unwrap();
    let session = LocalSession::open(temporary.path().join("relify")).unwrap();
    let source_path = temporary.path().join("source");
    write_direct_pid_source(&session.parquet, &source_path).await;

    let result = session
        .create_index(
            source_path.to_str().unwrap(),
            "managed_index",
            "embedding",
            &["source_pid".into()],
            2,
        )
        .await
        .unwrap();

    for reference in result
        .metadata
        .current_snapshot()
        .unwrap()
        .index_relations
        .values()
    {
        let RelationReference::Parquet { uri } = reference else {
            panic!("native build must publish Parquet relations");
        };
        assert!(!uri.contains(".tmp-"));
        assert!(file_uri_to_path(uri).unwrap().is_dir());
        assert!(uri.starts_with(session.warehouse_root()));
    }
    assert!(
        session
            .remove_orphans(i64::MAX, true)
            .await
            .unwrap()
            .is_empty()
    );
}
