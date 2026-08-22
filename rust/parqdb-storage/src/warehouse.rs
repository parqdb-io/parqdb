use bytes::Bytes;
use futures::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectMeta, ObjectStoreExt, PutMode};
use url::Url;

use crate::registry::parse_location;
use crate::{Error, ResolvedLocation, Result, StorageRegistry};

/// A managed URI prefix containing `ParqDB` metadata and index data.
#[derive(Debug, Clone)]
pub struct Warehouse {
    registry: StorageRegistry,
    root: Url,
}

impl Warehouse {
    /// Opens a warehouse rooted at an absolute `file`, `s3`, or `hdfs` URI.
    pub fn open(root: &str, registry: StorageRegistry) -> Result<Self> {
        let mut root = parse_location(root)?;
        if !root.path().ends_with('/') {
            let path = format!("{}/", root.path());
            root.set_path(&path);
        }
        let _ = registry.resolve(root.as_str())?;
        Ok(Self { registry, root })
    }

    /// Returns the canonical warehouse root URI.
    #[must_use]
    pub fn root(&self) -> &str {
        self.root.as_str()
    }

    /// Returns a clone of the URI registry used by this warehouse.
    #[must_use]
    pub fn registry(&self) -> StorageRegistry {
        self.registry.clone()
    }

    /// Returns an absolute URI below the warehouse root.
    pub fn location(&self, relative: &str, directory: bool) -> Result<String> {
        validate_relative(relative)?;
        let relative = relative.strip_suffix('/').unwrap_or(relative);
        let mut location = self
            .root
            .join(relative)
            .map_err(|error| Error::InvalidLocation(error.to_string()))?;
        if directory && !location.path().ends_with('/') {
            let path = format!("{}/", location.path());
            location.set_path(&path);
        }
        Ok(location.into())
    }

    /// Writes a new immutable object and fails if it already exists.
    pub async fn put_new(&self, location: &str, bytes: Bytes) -> Result<()> {
        let resolved = self.managed(location)?;
        resolved
            .store()
            .put_opts(resolved.path(), bytes.into(), PutMode::Create.into())
            .await?;
        Ok(())
    }

    /// Reads an object completely.
    pub async fn read(&self, location: &str) -> Result<Bytes> {
        let resolved = self.managed(location)?;
        Ok(resolved.store().get(resolved.path()).await?.bytes().await?)
    }

    /// Reads an absolute location through this warehouse's storage registry.
    ///
    /// Unlike [`Self::read`], the location need not be below the managed
    /// warehouse root. This is used to inspect immutable artifacts before they
    /// are registered in a catalog.
    pub async fn read_external(&self, location: &str) -> Result<Bytes> {
        let resolved = self.registry.resolve(location)?;
        Ok(resolved.store().get(resolved.path()).await?.bytes().await?)
    }

    /// Returns metadata for an object below the warehouse.
    pub async fn head(&self, location: &str) -> Result<ObjectMeta> {
        let resolved = self.managed(location)?;
        Ok(resolved.store().head(resolved.path()).await?)
    }

    /// Lists all objects below a warehouse-relative prefix.
    pub async fn list(&self, relative_prefix: &str) -> Result<Vec<ObjectMeta>> {
        validate_relative(relative_prefix)?;
        let location = self.location(relative_prefix, true)?;
        let resolved = self.managed(&location)?;
        Ok(resolved
            .store()
            .list(Some(resolved.path()))
            .try_collect()
            .await?)
    }

    /// Deletes one object below the warehouse.
    pub async fn delete(&self, location: &str) -> Result<()> {
        let resolved = self.managed(location)?;
        resolved.store().delete(resolved.path()).await?;
        self.remove_empty_local_parents(resolved.uri())?;
        self.remove_empty_hdfs_parents(&resolved).await?;
        Ok(())
    }

    /// Converts an object-store path below this warehouse to an absolute URI.
    pub fn object_location(&self, path: &ObjectPath, directory: bool) -> Result<String> {
        let root = self.registry.resolve(self.root.as_str())?;
        let relative = strip_object_prefix(path, root.path()).ok_or_else(|| {
            Error::InvalidLocation(format!("object is outside warehouse {}: {path}", self.root))
        })?;
        self.location(&relative, directory)
    }

    /// Returns a warehouse-relative object path for a managed URI.
    pub fn relative_location(&self, location: &str) -> Result<String> {
        let resolved = self.managed(location)?;
        let root = self.registry.resolve(self.root.as_str())?;
        let mut relative = strip_object_prefix(resolved.path(), root.path()).ok_or_else(|| {
            Error::InvalidLocation(format!("location is outside warehouse: {location}"))
        })?;
        if resolved.uri().path().ends_with('/') && !relative.ends_with('/') {
            relative.push('/');
        }
        Ok(relative)
    }

    /// Resolves a warehouse-owned location.
    pub fn managed(&self, location: &str) -> Result<ResolvedLocation> {
        let resolved = self.registry.resolve(location)?;
        if !same_authority(&self.root, resolved.uri())
            || !resolved.uri().path().starts_with(self.root.path())
        {
            return Err(Error::InvalidLocation(format!(
                "location is outside warehouse {}: {location}",
                self.root
            )));
        }
        Ok(resolved)
    }

    fn remove_empty_local_parents(&self, location: &Url) -> Result<()> {
        if self.root.scheme() != "file" {
            return Ok(());
        }
        let root = self
            .root
            .to_file_path()
            .map_err(|()| Error::InvalidLocation(self.root.to_string()))?;
        let path = location
            .to_file_path()
            .map_err(|()| Error::InvalidLocation(location.to_string()))?;
        let mut parent = path.parent();
        while let Some(directory) = parent {
            if directory == root || !directory.starts_with(&root) {
                break;
            }
            match std::fs::remove_dir(directory) {
                Ok(()) => parent = directory.parent(),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::DirectoryNotEmpty | std::io::ErrorKind::NotFound
                    ) =>
                {
                    break;
                }
                Err(error) => {
                    return Err(Error::ObjectStore(object_store::Error::Generic {
                        store: "LocalFileSystem",
                        source: Box::new(error),
                    }));
                }
            }
        }
        Ok(())
    }

    async fn remove_empty_hdfs_parents(&self, location: &ResolvedLocation) -> Result<()> {
        if self.root.scheme() != "hdfs" {
            return Ok(());
        }
        let root = self.registry.resolve(self.root.as_str())?;
        let store = location.store();
        for parent in object_parents_below(root.path(), location.path()) {
            let children = store.list_with_delimiter(Some(&parent)).await?;
            if !children.objects.is_empty() || !children.common_prefixes.is_empty() {
                break;
            }
            store.delete(&parent).await?;
        }
        Ok(())
    }
}

fn validate_relative(relative: &str) -> Result<()> {
    let path = relative.strip_suffix('/').unwrap_or(relative);
    if relative.is_empty()
        || relative.starts_with('/')
        || path.is_empty()
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(Error::InvalidLocation(format!(
            "invalid warehouse-relative path: {relative}"
        )));
    }
    Ok(())
}

fn same_authority(left: &Url, right: &Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn strip_object_prefix(path: &ObjectPath, prefix: &ObjectPath) -> Option<String> {
    let path = path.as_ref();
    let prefix = prefix.as_ref().trim_end_matches('/');
    if prefix.is_empty() {
        return Some(path.to_owned());
    }
    path.strip_prefix(prefix)
        .and_then(|relative| relative.strip_prefix('/'))
        .map(str::to_owned)
}

pub(crate) fn object_parents_below(root: &ObjectPath, location: &ObjectPath) -> Vec<ObjectPath> {
    let mut parts = location.parts().collect::<Vec<_>>();
    let mut parents = Vec::new();
    while parts.pop().is_some() {
        let parent = parts.iter().cloned().collect::<ObjectPath>();
        if &parent == root || strip_object_prefix(&parent, root).is_none() {
            break;
        }
        parents.push(parent);
    }
    parents
}
