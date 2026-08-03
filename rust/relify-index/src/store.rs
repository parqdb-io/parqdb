use bytes::Bytes;
use relify_meta::IndexMetadata;
use relify_storage::Warehouse;
use uuid::Uuid;

use crate::{Error, Result};

/// Immutable index metadata storage under one managed warehouse.
#[derive(Debug, Clone)]
pub struct MetadataStore {
    warehouse: Warehouse,
}

impl MetadataStore {
    /// Creates a metadata store rooted in `warehouse`.
    #[must_use]
    pub const fn open(warehouse: Warehouse) -> Self {
        Self { warehouse }
    }

    /// Returns the stable root recorded in metadata for an index UUID.
    pub fn index_location(&self, index_uuid: Uuid) -> Result<String> {
        Ok(self
            .warehouse
            .location(&format!("metadata/{index_uuid}"), true)?)
    }

    /// Validates and writes the first immutable metadata document.
    pub async fn write_initial(&self, metadata: &IndexMetadata) -> Result<String> {
        metadata.validate()?;
        self.write(metadata, "v1.metadata.json").await
    }

    /// Validates and writes an immutable successor metadata document.
    pub async fn write_update(
        &self,
        base: &IndexMetadata,
        metadata: &IndexMetadata,
    ) -> Result<String> {
        metadata.validate_update_from(base)?;
        let version = u64::try_from(metadata.last_sequence_number).map_err(|_| {
            Error::InvalidMetadata("metadata sequence number is out of range".into())
        })?;
        self.write(
            metadata,
            &format!("v{version}-{}.metadata.json", metadata.current_snapshot_id),
        )
        .await
    }

    /// Loads and decodes one immutable metadata document.
    pub async fn load(&self, location: &str) -> Result<IndexMetadata> {
        let bytes = self.warehouse.read(location).await?;
        Ok(IndexMetadata::from_json_slice(&bytes)?)
    }

    async fn write(&self, metadata: &IndexMetadata, filename: &str) -> Result<String> {
        let expected_location = self.index_location(metadata.index_uuid)?;
        if metadata.location != expected_location {
            return Err(Error::InvalidMetadata(
                "metadata location does not match the configured warehouse".into(),
            ));
        }
        let destination = self.warehouse.location(
            &format!("metadata/{}/{filename}", metadata.index_uuid),
            false,
        )?;
        let mut bytes = serde_json::to_vec_pretty(metadata)?;
        bytes.push(b'\n');
        self.warehouse
            .put_new(&destination, Bytes::from(bytes))
            .await?;
        Ok(destination)
    }
}
