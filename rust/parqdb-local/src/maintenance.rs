//! Local warehouse orphan discovery and collection.

use std::collections::{BTreeMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use object_store::ObjectMeta;
use parqdb_catalog::{CatalogTombstone, IndexCatalog};
use parqdb_meta::{RelationReference, ivf_centroids_reference};
use parqdb_storage::Warehouse;
use uuid::Uuid;

use crate::Result;

const MINIMUM_RETENTION_MS: i64 = 7 * 24 * 60 * 60 * 1_000;
use parqdb_index::MetadataStore;

/// One unreachable object managed by a `ParqDB` warehouse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceObject {
    /// Managed object category.
    pub kind: MaintenanceKind,
    /// Canonical URI of the object or snapshot prefix.
    pub reference: String,
    /// Latest modification time in Unix epoch milliseconds.
    pub modified_ms: i64,
}

/// Category of a maintenance object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceKind {
    /// Immutable `ParqDB` metadata file.
    Metadata,
    /// One snapshot prefix containing index tables.
    IndexData,
}

impl MaintenanceKind {
    /// Returns the stable Python-facing spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::IndexData => "index_data",
        }
    }
}

#[derive(Debug)]
struct Candidate {
    object: MaintenanceObject,
    members: Vec<String>,
    verified: bool,
}

struct TombstoneReferences {
    tombstone: CatalogTombstone,
    references: HashSet<String>,
}

pub(crate) async fn remove_orphans(
    warehouse: &Warehouse,
    metadata_store: &MetadataStore,
    catalog: &dyn IndexCatalog,
    active_roots: &HashSet<String>,
    older_than_ms: i64,
    dry_run: bool,
) -> Result<Vec<MaintenanceObject>> {
    let older_than_ms = older_than_ms.min(now_ms()?.saturating_sub(MINIMUM_RETENTION_MS));
    let tombstones = tombstone_references(warehouse, metadata_store, catalog).await?;
    let mut reachable = reachable_locations(warehouse, metadata_store, catalog).await?;
    reachable.extend(active_roots.iter().cloned());
    let mut verified = HashSet::new();
    for state in &tombstones {
        if state.tombstone.unreachable_since_ms < older_than_ms {
            verified.extend(state.references.iter().cloned());
        } else {
            reachable.extend(state.references.iter().cloned());
        }
    }
    let mut candidates = candidates(warehouse, &reachable, &verified, older_than_ms).await?;
    candidates.sort_by(|left, right| left.object.reference.cmp(&right.object.reference));
    if dry_run {
        return Ok(candidates
            .into_iter()
            .map(|candidate| candidate.object)
            .collect());
    }

    let mut reachable = reachable_locations(warehouse, metadata_store, catalog).await?;
    reachable.extend(active_roots.iter().cloned());
    for state in &tombstones {
        if state.tombstone.unreachable_since_ms >= older_than_ms {
            reachable.extend(state.references.iter().cloned());
        }
    }
    let mut removed = Vec::new();
    for candidate in candidates {
        if reachable.contains(&candidate.object.reference) {
            continue;
        }
        if !candidate.verified {
            let mut still_old = true;
            for location in &candidate.members {
                match warehouse.head(location).await {
                    Ok(meta) if modified_ms(&meta) >= older_than_ms => {
                        still_old = false;
                        break;
                    }
                    Ok(_)
                    | Err(parqdb_storage::Error::ObjectStore(object_store::Error::NotFound {
                        ..
                    })) => {}
                    Err(error) => return Err(error.into()),
                }
            }
            if !still_old {
                continue;
            }
        }
        for member in &candidate.members {
            match warehouse.delete(member).await {
                Ok(())
                | Err(parqdb_storage::Error::ObjectStore(object_store::Error::NotFound {
                    ..
                })) => {}
                Err(error) => return Err(error.into()),
            }
            if candidate.object.kind == MaintenanceKind::Metadata {
                metadata_store.invalidate(member);
            }
        }
        removed.push(candidate.object);
    }
    for state in tombstones {
        if state.tombstone.unreachable_since_ms >= older_than_ms
            || reachable.contains(&state.tombstone.metadata_location)
        {
            continue;
        }
        match warehouse.head(&state.tombstone.metadata_location).await {
            Err(parqdb_storage::Error::ObjectStore(object_store::Error::NotFound { .. })) => {
                catalog.purge_tombstone(&state.tombstone)?;
            }
            Ok(_) => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(removed)
}

async fn tombstone_references(
    warehouse: &Warehouse,
    metadata_store: &MetadataStore,
    catalog: &dyn IndexCatalog,
) -> Result<Vec<TombstoneReferences>> {
    let mut states = Vec::new();
    for tombstone in catalog.list_tombstones()? {
        let mut references = HashSet::new();
        if warehouse.managed(&tombstone.metadata_location).is_ok() {
            references.insert(tombstone.metadata_location.clone());
        }
        match metadata_store.load(&tombstone.metadata_location).await {
            Ok(metadata) => {
                references.extend(index_metadata_references(warehouse, &metadata)?);
            }
            Err(parqdb_index::Error::InvalidMetadata(_)) => {
                let metadata = metadata_store
                    .load_ivf_centroids(&tombstone.metadata_location)
                    .await?;
                if let RelationReference::Parquet { uri } = &metadata.centroids
                    && let Some(root) = snapshot_root(warehouse, uri)?
                {
                    references.insert(root);
                }
            }
            Err(parqdb_index::Error::Storage(parqdb_storage::Error::ObjectStore(
                object_store::Error::NotFound { .. },
            ))) => {}
            Err(error) => return Err(error.into()),
        }
        states.push(TombstoneReferences {
            tombstone,
            references,
        });
    }
    Ok(states)
}

async fn reachable_locations(
    warehouse: &Warehouse,
    metadata_store: &MetadataStore,
    catalog: &dyn IndexCatalog,
) -> Result<HashSet<String>> {
    let mut reachable = HashSet::new();
    for identifier in catalog.list_all()? {
        let entry = catalog.load(&identifier)?;
        let metadata = metadata_store.load(&entry.metadata_location).await?;
        if warehouse.managed(&entry.metadata_location).is_ok() {
            reachable.insert(entry.metadata_location);
        }
        reachable.extend(index_metadata_references(warehouse, &metadata)?);
    }
    for entry in catalog.list_ivf_centroids()? {
        let metadata = metadata_store
            .load_ivf_centroids(&entry.metadata_location)
            .await?;
        if warehouse.managed(&entry.metadata_location).is_ok() {
            reachable.insert(entry.metadata_location);
        }
        if let RelationReference::Parquet { uri } = &metadata.centroids
            && let Some(root) = snapshot_root(warehouse, uri)?
        {
            reachable.insert(root);
        }
    }
    Ok(reachable)
}

fn index_metadata_references(
    warehouse: &Warehouse,
    metadata: &parqdb_meta::IndexMetadata,
) -> Result<HashSet<String>> {
    let mut references = HashSet::new();
    for snapshot in &metadata.snapshots {
        if let Ok(centroids) = ivf_centroids_reference(snapshot)
            && warehouse.managed(&centroids.metadata_location).is_ok()
        {
            references.insert(centroids.metadata_location);
        }
        for reference in snapshot.index_relations.values() {
            let RelationReference::Parquet { uri } = reference else {
                continue;
            };
            if let Some(root) = snapshot_root(warehouse, uri)? {
                references.insert(root);
            }
        }
    }
    Ok(references)
}

async fn candidates(
    warehouse: &Warehouse,
    reachable: &HashSet<String>,
    verified: &HashSet<String>,
    older_than_ms: i64,
) -> Result<Vec<Candidate>> {
    let mut candidates = metadata_candidates(warehouse, reachable, verified, older_than_ms).await?;
    candidates.extend(index_candidates(warehouse, reachable, verified, older_than_ms).await?);
    Ok(candidates)
}

async fn metadata_candidates(
    warehouse: &Warehouse,
    reachable: &HashSet<String>,
    verified: &HashSet<String>,
    older_than_ms: i64,
) -> Result<Vec<Candidate>> {
    let mut candidates = Vec::new();
    for meta in list_if_present(warehouse, "metadata").await? {
        let relative = warehouse.object_location(&meta.location, false)?;
        let path = warehouse.relative_location(&relative)?;
        let mut parts = path.split('/');
        let valid = parts.next() == Some("metadata")
            && parts
                .next()
                .is_some_and(|value| Uuid::parse_str(value).is_ok())
            && parts.next().is_some_and(is_metadata_filename)
            && parts.next().is_none();
        let modified_ms = modified_ms(&meta);
        if valid
            && !reachable.contains(&relative)
            && (verified.contains(&relative) || modified_ms < older_than_ms)
        {
            let is_verified = verified.contains(&relative);
            candidates.push(Candidate {
                object: MaintenanceObject {
                    kind: MaintenanceKind::Metadata,
                    reference: relative.clone(),
                    modified_ms,
                },
                members: vec![relative],
                verified: is_verified,
            });
        }
    }
    Ok(candidates)
}

async fn index_candidates(
    warehouse: &Warehouse,
    reachable: &HashSet<String>,
    verified: &HashSet<String>,
    older_than_ms: i64,
) -> Result<Vec<Candidate>> {
    let mut groups: BTreeMap<String, (i64, Vec<String>)> = BTreeMap::new();
    for meta in list_if_present(warehouse, "indexes").await? {
        let location = warehouse.object_location(&meta.location, false)?;
        let Some(root) = snapshot_root(warehouse, &location)? else {
            continue;
        };
        let entry = groups.entry(root).or_insert((i64::MIN, Vec::new()));
        entry.0 = entry.0.max(modified_ms(&meta));
        entry.1.push(location);
    }
    Ok(groups
        .into_iter()
        .filter_map(|(reference, (modified_ms, members))| {
            let is_verified = verified.contains(&reference);
            (!reachable.contains(&reference) && (is_verified || modified_ms < older_than_ms))
                .then_some(Candidate {
                    object: MaintenanceObject {
                        kind: MaintenanceKind::IndexData,
                        reference,
                        modified_ms,
                    },
                    members,
                    verified: is_verified,
                })
        })
        .collect())
}

async fn list_if_present(warehouse: &Warehouse, prefix: &str) -> Result<Vec<ObjectMeta>> {
    match warehouse.list(prefix).await {
        Ok(objects) => Ok(objects),
        Err(parqdb_storage::Error::ObjectStore(object_store::Error::NotFound { .. })) => {
            Ok(Vec::new())
        }
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn snapshot_root(warehouse: &Warehouse, location: &str) -> Result<Option<String>> {
    if warehouse.managed(location).is_err() {
        return Ok(None);
    }
    let relative = warehouse.relative_location(location)?;
    let parts = relative.split('/').collect::<Vec<_>>();
    if parts.len() < 3
        || parts[0] != "indexes"
        || parts[1].len() != 32
        || Uuid::parse_str(parts[1]).is_err()
        || parts[2]
            .parse::<i64>()
            .map_or(true, |snapshot_id| snapshot_id <= 0)
    {
        return Ok(None);
    }
    Ok(Some(warehouse.location(&parts[..3].join("/"), true)?))
}

fn is_metadata_filename(value: &str) -> bool {
    let Some(version) = value
        .strip_prefix('v')
        .and_then(|value| value.strip_suffix(".metadata.json"))
    else {
        return false;
    };
    let version = version
        .split_once('-')
        .map_or(version, |(version, _)| version);
    version.parse::<u64>().is_ok_and(|version| version > 0)
}

fn modified_ms(meta: &ObjectMeta) -> i64 {
    meta.last_modified.timestamp_millis()
}

fn now_ms() -> Result<i64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| crate::Error::InvalidArgument(error.to_string()))?
        .as_millis();
    i64::try_from(millis)
        .map_err(|_| crate::Error::InvalidArgument("current timestamp is out of range".into()))
}
