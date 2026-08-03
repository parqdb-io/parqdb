use relify_meta::{IndexMetadata, RelationReference};

use crate::{Error, IndexIdentifier, Result};

/// A catalog entry that identifies the current immutable metadata document.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    /// Structured identifier of the index.
    pub identifier: IndexIdentifier,
    /// URI of the current immutable metadata document.
    pub metadata_location: String,
}

/// A metadata document that lost catalog reachability at a known time.
///
/// Catalog implementations may retain tombstones as private maintenance state.
/// They are not part of portable Relify index metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogTombstone {
    /// URI of the metadata document that was made unreachable.
    pub metadata_location: String,
    /// Unix epoch milliseconds when the catalog mapping stopped referencing it.
    pub unreachable_since_ms: i64,
}

/// Runtime catalog operations used by Relify.
///
/// Implementations own identifier mappings and the atomic publication of new
/// metadata locations. Metadata files are read through the session's storage
/// layer rather than by the catalog implementation.
///
/// Mutation and discovery methods have unsupported defaults so that a
/// read-only implementation needs to provide only [`IndexCatalog::load`].
pub trait IndexCatalog: Send + Sync {
    /// Loads the current metadata location for an index.
    fn load(&self, identifier: &IndexIdentifier) -> Result<CatalogEntry>;

    /// Registers a new index if its identifier is absent.
    ///
    /// `metadata` is the already loaded and validated document stored at
    /// `metadata_location`. Implementations may persist fields needed for
    /// discovery but must make `metadata_location` the publication pointer.
    fn register(
        &self,
        _identifier: &IndexIdentifier,
        _metadata_location: &str,
        _metadata: &IndexMetadata,
    ) -> Result<()> {
        Err(Error::UnsupportedOperation("register"))
    }

    /// Atomically advances an index if `base_metadata_location` remains current.
    fn commit(
        &self,
        _identifier: &IndexIdentifier,
        _base_metadata_location: &str,
        _new_metadata_location: &str,
        _base_metadata: &IndexMetadata,
        _new_metadata: &IndexMetadata,
    ) -> Result<()> {
        Err(Error::UnsupportedOperation("commit"))
    }

    /// Removes an index mapping without deleting its metadata or index tables.
    fn drop(&self, _identifier: &IndexIdentifier) -> Result<()> {
        Err(Error::UnsupportedOperation("drop"))
    }

    /// Lists identifiers registered directly in a namespace.
    fn list(&self, _namespace: &[String]) -> Result<Vec<IndexIdentifier>> {
        Err(Error::UnsupportedOperation("list"))
    }

    /// Finds indexes in a namespace that refer to an exact source-table state.
    ///
    /// Source discovery is a Relify runtime extension to the portable catalog
    /// operations defined by the specification.
    fn find_by_source(
        &self,
        _namespace: &[String],
        _source: &RelationReference,
    ) -> Result<Vec<CatalogEntry>> {
        Err(Error::UnsupportedOperation("find_by_source"))
    }

    /// Lists metadata documents made unreachable by catalog mutations.
    ///
    /// This is an implementation extension used by safe garbage collection.
    fn list_tombstones(&self) -> Result<Vec<CatalogTombstone>> {
        Err(Error::UnsupportedOperation("list_tombstones"))
    }

    /// Removes an unchanged catalog tombstone after its objects are reclaimed.
    ///
    /// Returns `false` when the tombstone no longer exists or has changed.
    fn purge_tombstone(&self, _tombstone: &CatalogTombstone) -> Result<bool> {
        Err(Error::UnsupportedOperation("purge_tombstone"))
    }
}
