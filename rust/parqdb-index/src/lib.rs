#![warn(missing_docs)]
//! Backend-neutral index loading, selection, storage, and publication.
//!
//! Execution backends resolve source and index relations themselves. This
//! crate owns the portable state shared by those backends: immutable metadata
//! documents, catalog discovery, index selection, and catalog publication.

mod error;
mod publication;
mod repository;
mod store;

pub use error::{Error, Result};
pub use publication::{
    InitialIndex, RefreshedIndex, new_snapshot_id, publish_initial, publish_refresh,
};
pub use repository::{
    IndexRepository, LoadedIndex, LoadedIvfCentroids, resolve_warehouse_location,
};
pub use store::{
    DEFAULT_METADATA_CACHE_BYTES, DEFAULT_METADATA_CACHE_ENTRIES, MetadataCacheConfig,
    MetadataStore,
};

#[cfg(test)]
mod tests;
