//! URI-based access to Relify-managed files.
//!
//! A [`StorageRegistry`] resolves absolute `file`, `s3`, and `hdfs` URIs to
//! implementations of the Apache Arrow [`object_store::ObjectStore`] contract.
//! A [`Warehouse`] adds a managed root without changing the underlying storage
//! semantics.

mod error;
mod registry;
mod warehouse;

pub use error::{Error, Result};
pub use registry::{ResolvedLocation, StorageRegistry};
pub use warehouse::Warehouse;

#[cfg(test)]
mod tests;
