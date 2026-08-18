use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parqdb_meta::{IndexMetadata, IvfCentroidsDescriptor, IvfCentroidsMetadata, RelationReference};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use uuid::Uuid;

use crate::identifier::namespace_key;
use crate::{
    CatalogEntry, CatalogTombstone, Error, IndexCatalog, IndexIdentifier, IvfCentroidsCatalogEntry,
    IvfCentroidsClaim, IvfCentroidsClaimResult, Result, TableCatalog, TableDefinition,
    TableIdentifier,
};

const SCHEMA_VERSION: i64 = 1;
const APPLICATION_ID: i64 = 0x5051_4442; // ASCII "PQDB"
const ROOT_NAMESPACE_KEY: &str = "[]";
const IVF_CENTROIDS_STATE_BUILDING: &str = "building";
const IVF_CENTROIDS_STATE_READY: &str = "ready";
const IVF_CENTROIDS_STATE_FAILED: &str = "failed";

/// A namespace-aware `ParqDB` catalog stored in `SQLite`.
#[derive(Debug, Clone)]
pub struct SqliteCatalog {
    database: PathBuf,
}

impl SqliteCatalog {
    /// Opens or creates a `SQLite` catalog at `database`.
    pub fn open(database: impl AsRef<Path>) -> Result<Self> {
        let database = database.as_ref();
        if let Some(parent) = database.parent() {
            fs::create_dir_all(parent)?;
        }
        let catalog = Self {
            database: database.to_owned(),
        };
        catalog.initialize()?;
        Ok(catalog)
    }

    /// Returns the path of the `SQLite` database.
    #[must_use]
    pub fn database_path(&self) -> &Path {
        &self.database
    }

    fn exists(&self, identifier: &IndexIdentifier) -> Result<bool> {
        let namespace = identifier.namespace_key()?;
        Ok(self
            .connection()?
            .query_row(
                "SELECT 1 FROM indexes WHERE namespace = ?1 AND name = ?2",
                params![namespace, identifier.name()],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    fn require_namespace(&self, namespace: &[String]) -> Result<String> {
        let key = namespace_key(namespace)?;
        let exists = self
            .connection()?
            .query_row(
                "SELECT 1 FROM namespaces WHERE namespace = ?1",
                [&key],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if exists {
            Ok(key)
        } else {
            Err(Error::NamespaceNotFound(namespace.to_vec()))
        }
    }

    fn connection(&self) -> Result<Connection> {
        let connection = Connection::open(&self.database)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.execute_batch(
            "PRAGMA foreign_keys=ON;
             PRAGMA synchronous=FULL;",
        )?;
        Ok(connection)
    }

    fn initialize(&self) -> Result<()> {
        let mut connection = self.connection()?;
        let application_id =
            connection.query_row("PRAGMA application_id", [], |row| row.get::<_, i64>(0))?;
        let version =
            connection.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?;
        if application_id == 0 && version == 0 {
            if database_is_empty(&connection)? {
                connection.execute_batch("PRAGMA journal_mode=WAL;")?;
                return create_schema(&mut connection);
            }
            if !table_exists(&connection, "indexes")? {
                return Err(Error::UnsupportedApplicationId(application_id));
            }
        }
        if version != SCHEMA_VERSION {
            return Err(Error::UnsupportedSchemaVersion(version));
        }
        if application_id != APPLICATION_ID {
            return Err(Error::UnsupportedApplicationId(application_id));
        }
        connection.execute_batch("PRAGMA journal_mode=WAL;")?;
        Ok(())
    }
}

impl IndexCatalog for SqliteCatalog {
    fn register(
        &self,
        identifier: &IndexIdentifier,
        source: &RelationReference,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> Result<()> {
        metadata.validate()?;
        source.validate()?;
        parqdb_meta::validate_absolute_location(metadata_location)?;
        let namespace = identifier.namespace_key()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT OR IGNORE INTO namespaces(namespace) VALUES (?1)",
            [&namespace],
        )?;
        let result = transaction.execute(
            "INSERT INTO indexes(
                namespace, name, metadata_location, index_uuid,
                source_identity, source_reference
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                namespace,
                identifier.name(),
                metadata_location,
                metadata.index_uuid.to_string(),
                source.exact_state_key(),
                serde_json::to_string(source)?,
            ],
        );
        match result {
            Ok(_) => {
                transaction.execute(
                    "DELETE FROM catalog_tombstones WHERE metadata_location = ?1",
                    [metadata_location],
                )?;
                transaction.commit()?;
                Ok(())
            }
            Err(error) if is_unique_constraint(&error) => {
                Err(Error::AlreadyExists(identifier.clone()))
            }
            Err(error) => Err(error.into()),
        }
    }

    fn commit(
        &self,
        identifier: &IndexIdentifier,
        source: &RelationReference,
        base_metadata_location: &str,
        new_metadata_location: &str,
        base_metadata: &IndexMetadata,
        new_metadata: &IndexMetadata,
    ) -> Result<()> {
        if base_metadata_location == new_metadata_location {
            return Err(Error::InvalidMetadata(
                "new metadata location must differ from base".into(),
            ));
        }
        let namespace = identifier.namespace_key()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = transaction
            .query_row(
                "SELECT metadata_location, index_uuid
                 FROM indexes
                 WHERE namespace = ?1 AND name = ?2",
                params![namespace, identifier.name()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?
            .ok_or_else(|| Error::IndexNotFound(identifier.clone()))?;
        if current.0 != base_metadata_location {
            return Err(Error::CommitConflict(identifier.clone()));
        }

        if new_metadata.index_uuid.to_string() != current.1
            || new_metadata.index_uuid != base_metadata.index_uuid
        {
            return Err(Error::IndexUuidMismatch(identifier.clone()));
        }
        new_metadata.validate_update_from(base_metadata)?;
        source.validate()?;
        parqdb_meta::validate_absolute_location(new_metadata_location)?;
        let source_identity = source.exact_state_key();
        let updated = transaction.execute(
            "UPDATE indexes
             SET metadata_location = ?1, source_identity = ?2,
                 source_reference = ?3
             WHERE namespace = ?4
               AND name = ?5
               AND metadata_location = ?6
               AND index_uuid = ?7",
            params![
                new_metadata_location,
                source_identity,
                serde_json::to_string(source)?,
                namespace,
                identifier.name(),
                base_metadata_location,
                new_metadata.index_uuid.to_string(),
            ],
        )?;
        if updated == 1 {
            transaction.execute(
                "INSERT INTO catalog_tombstones(metadata_location, unreachable_since_ms)
                 VALUES (?1, ?2)
                 ON CONFLICT(metadata_location) DO UPDATE SET
                     unreachable_since_ms = excluded.unreachable_since_ms",
                params![base_metadata_location, now_ms()?],
            )?;
            transaction.commit()?;
            return Ok(());
        }
        drop(transaction);
        if !self.exists(identifier)? {
            return Err(Error::IndexNotFound(identifier.clone()));
        }
        Err(Error::CommitConflict(identifier.clone()))
    }

    fn drop(&self, identifier: &IndexIdentifier) -> Result<()> {
        let namespace = identifier.namespace_key()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let metadata_location = transaction
            .query_row(
                "SELECT metadata_location
                 FROM indexes
                 WHERE namespace = ?1 AND name = ?2",
                params![namespace, identifier.name()],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| Error::IndexNotFound(identifier.clone()))?;
        transaction.execute(
            "INSERT INTO catalog_tombstones(metadata_location, unreachable_since_ms)
             VALUES (?1, ?2)
             ON CONFLICT(metadata_location) DO UPDATE SET
                 unreachable_since_ms = excluded.unreachable_since_ms",
            params![metadata_location, now_ms()?],
        )?;
        let removed = transaction.execute(
            "DELETE FROM indexes WHERE namespace = ?1 AND name = ?2",
            params![namespace, identifier.name()],
        )?;
        if removed != 1 {
            return Err(Error::Implementation(
                "catalog entry disappeared during drop".into(),
            ));
        }
        transaction.commit()?;
        Ok(())
    }

    fn load(&self, identifier: &IndexIdentifier) -> Result<CatalogEntry> {
        let namespace = identifier.namespace_key()?;
        let entry = self
            .connection()?
            .query_row(
                "SELECT metadata_location, source_reference
                 FROM indexes
                 WHERE namespace = ?1 AND name = ?2",
                params![namespace, identifier.name()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?
            .ok_or_else(|| Error::IndexNotFound(identifier.clone()))?;
        Ok(CatalogEntry {
            identifier: identifier.clone(),
            metadata_location: entry.0,
            source: serde_json::from_str(&entry.1)?,
        })
    }

    fn find_by_source(
        &self,
        namespace: &[String],
        source: &RelationReference,
    ) -> Result<Vec<CatalogEntry>> {
        let namespace_key = self.require_namespace(namespace)?;
        let source_identity = source.exact_state_key();
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT name, metadata_location, source_reference
             FROM indexes
             WHERE namespace = ?1 AND source_identity = ?2
             ORDER BY name",
        )?;
        let rows = statement.query_map(params![namespace_key, source_identity], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut entries = Vec::new();
        for row in rows {
            let (name, metadata_location, source_reference) = row?;
            entries.push(CatalogEntry {
                identifier: IndexIdentifier::new(namespace.to_vec(), name)?,
                metadata_location,
                source: serde_json::from_str(&source_reference)?,
            });
        }
        Ok(entries)
    }

    fn list(&self, namespace: &[String]) -> Result<Vec<IndexIdentifier>> {
        let namespace_key = self.require_namespace(namespace)?;
        let connection = self.connection()?;
        let mut statement =
            connection.prepare("SELECT name FROM indexes WHERE namespace = ?1 ORDER BY name")?;
        let names = statement
            .query_map([namespace_key], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        names
            .into_iter()
            .map(|name| IndexIdentifier::new(namespace.to_vec(), name))
            .collect()
    }

    fn list_all(&self) -> Result<Vec<IndexIdentifier>> {
        let connection = self.connection()?;
        let mut statement =
            connection.prepare("SELECT namespace, name FROM indexes ORDER BY namespace, name")?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows.into_iter()
            .map(|(namespace, name)| IndexIdentifier::new(serde_json::from_str(&namespace)?, name))
            .collect()
    }

    fn list_tombstones(&self) -> Result<Vec<CatalogTombstone>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT metadata_location, unreachable_since_ms
             FROM catalog_tombstones
             ORDER BY metadata_location",
        )?;
        Ok(statement
            .query_map([], |row| {
                Ok(CatalogTombstone {
                    metadata_location: row.get(0)?,
                    unreachable_since_ms: row.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn purge_tombstone(&self, tombstone: &CatalogTombstone) -> Result<bool> {
        Ok(self.connection()?.execute(
            "DELETE FROM catalog_tombstones
             WHERE metadata_location = ?1 AND unreachable_since_ms = ?2",
            params![tombstone.metadata_location, tombstone.unreachable_since_ms],
        )? == 1)
    }

    fn load_ivf_centroids(
        &self,
        source: &RelationReference,
        fingerprint: &str,
    ) -> Result<IvfCentroidsCatalogEntry> {
        source.validate()?;
        let source_identity = source.exact_state_key();
        let row = self
            .connection()?
            .query_row(
                "SELECT artifact_uuid, metadata_location
                 FROM ivf_centroid_artifacts
                 WHERE source_identity = ?1 AND fingerprint = ?2 AND state = ?3",
                params![source_identity, fingerprint, IVF_CENTROIDS_STATE_READY],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?
            .ok_or_else(|| Error::IvfCentroidsNotFound(fingerprint.to_owned()))?;
        ivf_centroids_entry(&source.exact_state_key(), fingerprint, &row.0, &row.1)
    }

    fn claim_ivf_centroids(
        &self,
        source: &RelationReference,
        descriptor: &IvfCentroidsDescriptor,
        owner: Uuid,
        lease_duration_ms: i64,
    ) -> Result<IvfCentroidsClaimResult> {
        source.validate()?;
        descriptor.validate()?;
        validate_lease(owner, lease_duration_ms)?;
        let source_identity = source.exact_state_key();
        let fingerprint = descriptor.fingerprint()?;
        let descriptor_json = serde_json::to_string(descriptor)?;
        let owner_string = owner.to_string();
        let now = now_ms()?;
        let lease_expires_ms = now.saturating_add(lease_duration_ms);
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing = transaction
            .query_row(
                "SELECT descriptor, state, owner, lease_expires_ms,
                        artifact_uuid, metadata_location
                 FROM ivf_centroid_artifacts
                 WHERE source_identity = ?1 AND fingerprint = ?2",
                params![source_identity, fingerprint],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<i64>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                    ))
                },
            )
            .optional()?;
        if let Some((stored, state, current_owner, current_lease, artifact_uuid, location)) =
            existing
        {
            let stored_descriptor: IvfCentroidsDescriptor = serde_json::from_str(&stored)?;
            if !stored_descriptor.is_compatible_with(descriptor) {
                return Err(Error::InvalidMetadata(
                    "IVF centroid fingerprint collision".into(),
                ));
            }
            if state == IVF_CENTROIDS_STATE_READY {
                let entry = ivf_centroids_entry(
                    &source_identity,
                    &fingerprint,
                    artifact_uuid.as_deref().ok_or_else(|| {
                        Error::Implementation("ready IVF centroid artifact has no UUID".into())
                    })?,
                    location.as_deref().ok_or_else(|| {
                        Error::Implementation(
                            "ready IVF centroid artifact has no metadata location".into(),
                        )
                    })?,
                )?;
                transaction.commit()?;
                return Ok(IvfCentroidsClaimResult::Ready(entry));
            }
            if state == IVF_CENTROIDS_STATE_BUILDING
                && current_owner.as_deref() != Some(owner_string.as_str())
                && let Some(lease_expires_ms) = current_lease
                && lease_expires_ms > now
            {
                transaction.commit()?;
                return Ok(IvfCentroidsClaimResult::Busy { lease_expires_ms });
            }
            transaction.execute(
                "UPDATE ivf_centroid_artifacts
                 SET descriptor = ?1, state = ?2, owner = ?3, lease_expires_ms = ?4,
                     artifact_uuid = NULL, metadata_location = NULL,
                     error = NULL
                 WHERE source_identity = ?5 AND fingerprint = ?6",
                params![
                    descriptor_json,
                    IVF_CENTROIDS_STATE_BUILDING,
                    owner_string,
                    lease_expires_ms,
                    source_identity,
                    fingerprint
                ],
            )?;
        } else {
            transaction.execute(
                "INSERT INTO ivf_centroid_artifacts(
                    source_identity, fingerprint, descriptor, state, owner, lease_expires_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    source_identity,
                    fingerprint,
                    descriptor_json,
                    IVF_CENTROIDS_STATE_BUILDING,
                    owner_string,
                    lease_expires_ms
                ],
            )?;
        }
        transaction.commit()?;
        Ok(IvfCentroidsClaimResult::Claimed(IvfCentroidsClaim {
            source_identity,
            fingerprint,
            owner,
        }))
    }

    fn renew_ivf_centroids_claim(
        &self,
        claim: &IvfCentroidsClaim,
        lease_duration_ms: i64,
    ) -> Result<()> {
        validate_lease(claim.owner, lease_duration_ms)?;
        let now = now_ms()?;
        let lease_expires_ms = now.saturating_add(lease_duration_ms);
        let updated = self.connection()?.execute(
            "UPDATE ivf_centroid_artifacts SET lease_expires_ms = ?1
             WHERE source_identity = ?2 AND fingerprint = ?3
               AND state = ?4 AND owner = ?5 AND lease_expires_ms > ?6",
            params![
                lease_expires_ms,
                claim.source_identity,
                claim.fingerprint,
                IVF_CENTROIDS_STATE_BUILDING,
                claim.owner.to_string(),
                now
            ],
        )?;
        if updated == 1 {
            Ok(())
        } else {
            Err(Error::IvfCentroidsClaimLost(claim.fingerprint.clone()))
        }
    }

    fn publish_ivf_centroids(
        &self,
        claim: &IvfCentroidsClaim,
        metadata_location: &str,
        metadata: &IvfCentroidsMetadata,
    ) -> Result<IvfCentroidsCatalogEntry> {
        metadata.validate()?;
        if metadata.fingerprint != claim.fingerprint {
            return Err(Error::InvalidMetadata(
                "published IVF centroid fingerprint does not match claim".into(),
            ));
        }
        parqdb_meta::validate_absolute_location(metadata_location)?;
        let descriptor_json = serde_json::to_string(&metadata.descriptor)?;
        let now = now_ms()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let updated = transaction.execute(
            "UPDATE ivf_centroid_artifacts
             SET state = ?1, owner = NULL, lease_expires_ms = NULL,
                 artifact_uuid = ?2, metadata_location = ?3,
                 error = NULL
             WHERE source_identity = ?4 AND fingerprint = ?5 AND descriptor = ?6
               AND state = ?7 AND owner = ?8 AND lease_expires_ms > ?9",
            params![
                IVF_CENTROIDS_STATE_READY,
                metadata.artifact_uuid.to_string(),
                metadata_location,
                claim.source_identity,
                claim.fingerprint,
                descriptor_json,
                IVF_CENTROIDS_STATE_BUILDING,
                claim.owner.to_string(),
                now,
            ],
        )?;
        if updated != 1 {
            return Err(Error::IvfCentroidsClaimLost(claim.fingerprint.clone()));
        }
        transaction.execute(
            "DELETE FROM catalog_tombstones WHERE metadata_location = ?1",
            [metadata_location],
        )?;
        transaction.commit()?;
        Ok(IvfCentroidsCatalogEntry {
            source_identity: claim.source_identity.clone(),
            fingerprint: claim.fingerprint.clone(),
            artifact_uuid: metadata.artifact_uuid,
            metadata_location: metadata_location.to_owned(),
        })
    }

    fn abandon_ivf_centroids(&self, claim: &IvfCentroidsClaim, error: &str) -> Result<()> {
        let updated = self.connection()?.execute(
            "UPDATE ivf_centroid_artifacts
             SET state = ?1, owner = NULL, lease_expires_ms = NULL, error = ?2
             WHERE source_identity = ?3 AND fingerprint = ?4
               AND state = ?5 AND owner = ?6",
            params![
                IVF_CENTROIDS_STATE_FAILED,
                error,
                claim.source_identity,
                claim.fingerprint,
                IVF_CENTROIDS_STATE_BUILDING,
                claim.owner.to_string()
            ],
        )?;
        if updated == 1 {
            Ok(())
        } else {
            Err(Error::IvfCentroidsClaimLost(claim.fingerprint.clone()))
        }
    }

    fn list_ivf_centroids(&self) -> Result<Vec<IvfCentroidsCatalogEntry>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT source_identity, fingerprint, artifact_uuid, metadata_location
             FROM ivf_centroid_artifacts
             WHERE state = ?1 ORDER BY source_identity, fingerprint",
        )?;
        let rows = statement.query_map([IVF_CENTROIDS_STATE_READY], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        rows.map(|row| {
            let (source_identity, fingerprint, artifact_uuid, location) = row?;
            ivf_centroids_entry(&source_identity, &fingerprint, &artifact_uuid, &location)
        })
        .collect()
    }

    fn purge_ivf_centroids(&self, entry: &IvfCentroidsCatalogEntry) -> Result<bool> {
        Ok(self.connection()?.execute(
            "DELETE FROM ivf_centroid_artifacts
             WHERE source_identity = ?1 AND fingerprint = ?2 AND state = ?3
               AND artifact_uuid = ?4 AND metadata_location = ?5",
            params![
                entry.source_identity,
                entry.fingerprint,
                IVF_CENTROIDS_STATE_READY,
                entry.artifact_uuid.to_string(),
                entry.metadata_location
            ],
        )? == 1)
    }
}

impl TableCatalog for SqliteCatalog {
    fn create_table(&self, definition: &TableDefinition) -> Result<()> {
        let definition = TableDefinition::new(
            definition.identifier.clone(),
            definition.provider.clone(),
            definition.properties.clone(),
        )?;
        let namespace = definition.identifier.namespace_key()?;
        let properties = serde_json::to_string(&definition.properties)?;
        let result = self.connection()?.execute(
            "INSERT INTO datafusion_tables(
                catalog, namespace, name, provider, properties
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                definition.identifier.catalog(),
                namespace,
                definition.identifier.name(),
                definition.provider,
                properties,
            ],
        );
        match result {
            Ok(_) => Ok(()),
            Err(error) if is_unique_constraint(&error) => {
                Err(Error::TableAlreadyExists(definition.identifier))
            }
            Err(error) => Err(error.into()),
        }
    }

    fn load_table(&self, identifier: &TableIdentifier) -> Result<TableDefinition> {
        let namespace = identifier.namespace_key()?;
        let definition = self
            .connection()?
            .query_row(
                "SELECT provider, properties
                 FROM datafusion_tables
                 WHERE catalog = ?1 AND namespace = ?2 AND name = ?3",
                params![identifier.catalog(), namespace, identifier.name()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?
            .ok_or_else(|| Error::TableNotFound(identifier.clone()))?;
        TableDefinition::new(
            identifier.clone(),
            definition.0,
            serde_json::from_str(&definition.1)?,
        )
    }

    fn list_tables(&self, catalog: &str, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        if catalog.is_empty() {
            return Err(Error::InvalidIdentifier("catalog must not be empty".into()));
        }
        let namespace_key = namespace_key(namespace)?;
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT name
             FROM datafusion_tables
             WHERE catalog = ?1 AND namespace = ?2
             ORDER BY name",
        )?;
        let names = statement
            .query_map(params![catalog, namespace_key], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        names
            .into_iter()
            .map(|name| TableIdentifier::new(catalog, namespace.to_vec(), name))
            .collect()
    }

    fn drop_table(&self, identifier: &TableIdentifier) -> Result<()> {
        let namespace = identifier.namespace_key()?;
        let removed = self.connection()?.execute(
            "DELETE FROM datafusion_tables
             WHERE catalog = ?1 AND namespace = ?2 AND name = ?3",
            params![identifier.catalog(), namespace, identifier.name()],
        )?;
        if removed == 1 {
            Ok(())
        } else {
            Err(Error::TableNotFound(identifier.clone()))
        }
    }
}

fn create_schema(connection: &mut Connection) -> Result<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TABLE namespaces (
             namespace TEXT PRIMARY KEY
         );
         CREATE TABLE indexes (
             namespace TEXT NOT NULL,
             name TEXT NOT NULL,
             metadata_location TEXT NOT NULL,
             index_uuid TEXT NOT NULL UNIQUE,
             source_identity TEXT NOT NULL,
             source_reference TEXT NOT NULL,
             PRIMARY KEY(namespace, name),
             FOREIGN KEY(namespace) REFERENCES namespaces(namespace)
         );
         CREATE INDEX indexes_by_source
             ON indexes(namespace, source_identity);
         CREATE TABLE catalog_tombstones (
             metadata_location TEXT PRIMARY KEY,
             unreachable_since_ms INTEGER NOT NULL
         );
         CREATE TABLE datafusion_tables (
             catalog TEXT NOT NULL,
             namespace TEXT NOT NULL,
             name TEXT NOT NULL,
             provider TEXT NOT NULL,
             properties TEXT NOT NULL,
             PRIMARY KEY(catalog, namespace, name)
         );
         CREATE TABLE ivf_centroid_artifacts (
             source_identity TEXT NOT NULL,
             fingerprint TEXT NOT NULL,
             descriptor TEXT NOT NULL,
             state TEXT NOT NULL CHECK(state IN ('building', 'ready', 'failed')),
             owner TEXT,
             lease_expires_ms INTEGER,
             artifact_uuid TEXT,
             metadata_location TEXT,
             error TEXT,
             PRIMARY KEY(source_identity, fingerprint)
         );",
    )?;
    transaction.pragma_update(None, "application_id", APPLICATION_ID)?;
    transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    transaction.execute(
        "INSERT INTO namespaces(namespace) VALUES (?1)",
        [ROOT_NAMESPACE_KEY],
    )?;
    transaction.commit()?;
    Ok(())
}

fn validate_lease(owner: Uuid, lease_duration_ms: i64) -> Result<()> {
    if owner.is_nil() || lease_duration_ms <= 0 {
        return Err(Error::Implementation(
            "IVF centroid owner and lease duration must be valid".into(),
        ));
    }
    Ok(())
}

fn ivf_centroids_entry(
    source_identity: &str,
    fingerprint: &str,
    artifact_uuid: &str,
    metadata_location: &str,
) -> Result<IvfCentroidsCatalogEntry> {
    let artifact_uuid =
        Uuid::parse_str(artifact_uuid).map_err(|error| Error::Implementation(error.to_string()))?;
    Ok(IvfCentroidsCatalogEntry {
        source_identity: source_identity.to_owned(),
        fingerprint: fingerprint.to_owned(),
        artifact_uuid,
        metadata_location: metadata_location.to_owned(),
    })
}

fn table_exists(connection: &Connection, name: &str) -> Result<bool> {
    Ok(connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn database_is_empty(connection: &Connection) -> Result<bool> {
    // ParqDB only claims a truly empty database and never adopts existing user objects.
    Ok(!connection.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM sqlite_schema
             WHERE name NOT LIKE 'sqlite_%'
         )",
        [],
        |row| row.get(0),
    )?)
}

fn is_unique_constraint(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(inner, _)
            if inner.code == rusqlite::ErrorCode::ConstraintViolation
    )
}

fn now_ms() -> Result<i64> {
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::Implementation(error.to_string()))?
        .as_millis();
    i64::try_from(milliseconds)
        .map_err(|_| Error::Implementation("current timestamp is out of range".into()))
}
