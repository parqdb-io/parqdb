use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use relify_meta::{IndexMetadata, RelationReference};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use crate::identifier::namespace_key;
use crate::{
    CatalogEntry, CatalogTombstone, Error, IndexCatalog, IndexIdentifier, Result, TableCatalog,
    TableDefinition, TableIdentifier,
};

const SCHEMA_VERSION: i64 = 3;
const ROOT_NAMESPACE_KEY: &str = "[]";

/// A namespace-aware Relify catalog stored in `SQLite`.
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
        connection.execute_batch("PRAGMA journal_mode=WAL;")?;
        let version =
            connection.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?;
        match version {
            SCHEMA_VERSION => Ok(()),
            0 if table_exists(&connection, "indexes")? => {
                Err(Error::UnsupportedSchemaVersion(version))
            }
            0 => create_schema(&mut connection),
            other => Err(Error::UnsupportedSchemaVersion(other)),
        }
    }
}

impl IndexCatalog for SqliteCatalog {
    fn register(
        &self,
        identifier: &IndexIdentifier,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> Result<()> {
        metadata.validate()?;
        let snapshot = metadata.current_snapshot()?;
        let namespace = identifier.namespace_key()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT OR IGNORE INTO namespaces(namespace) VALUES (?1)",
            [&namespace],
        )?;
        let result = transaction.execute(
            "INSERT INTO indexes(
                namespace, name, metadata_location, index_uuid, source_identity
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                namespace,
                identifier.name(),
                metadata_location,
                metadata.index_uuid.to_string(),
                snapshot.source.exact_state_key(),
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
        if new_metadata.location != base_metadata.location {
            return Err(Error::InvalidMetadata(
                "index location must remain unchanged".into(),
            ));
        }
        new_metadata.validate_update_from(base_metadata)?;
        let source_identity = new_metadata.current_snapshot()?.source.exact_state_key();
        let updated = transaction.execute(
            "UPDATE indexes
             SET metadata_location = ?1, source_identity = ?2
             WHERE namespace = ?3
               AND name = ?4
               AND metadata_location = ?5
               AND index_uuid = ?6",
            params![
                new_metadata_location,
                source_identity,
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
        let metadata_location = self
            .connection()?
            .query_row(
                "SELECT metadata_location
                 FROM indexes
                 WHERE namespace = ?1 AND name = ?2",
                params![namespace, identifier.name()],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| Error::IndexNotFound(identifier.clone()))?;
        Ok(CatalogEntry {
            identifier: identifier.clone(),
            metadata_location,
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
            "SELECT name, metadata_location
             FROM indexes
             WHERE namespace = ?1 AND source_identity = ?2
             ORDER BY name",
        )?;
        let rows = statement.query_map(params![namespace_key, source_identity], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut entries = Vec::new();
        for row in rows {
            let (name, metadata_location) = row?;
            entries.push(CatalogEntry {
                identifier: IndexIdentifier::new(namespace.to_vec(), name)?,
                metadata_location,
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
         PRAGMA user_version=3;",
    )?;
    transaction.execute(
        "INSERT INTO namespaces(namespace) VALUES (?1)",
        [ROOT_NAMESPACE_KEY],
    )?;
    transaction.commit()?;
    Ok(())
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
