//! URI resolution and object-store client registry.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use futures::TryStreamExt;
use hdfs_native_object_store::HdfsObjectStoreBuilder;
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use url::Url;

use crate::{Error, Result};

/// One URI resolved to an object store and store-relative path.
#[derive(Clone)]
pub struct ResolvedLocation {
    uri: Url,
    base_url: Url,
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
}

impl std::fmt::Debug for ResolvedLocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedLocation")
            .field("uri", &self.uri)
            .field("base_url", &self.base_url)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl ResolvedLocation {
    /// Returns the canonical absolute URI.
    #[must_use]
    pub fn uri(&self) -> &Url {
        &self.uri
    }

    /// Returns the scheme-and-authority URL used to register the store.
    #[must_use]
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Returns the resolved object store.
    #[must_use]
    pub fn store(&self) -> Arc<dyn ObjectStore> {
        Arc::clone(&self.store)
    }

    /// Returns the path relative to the object store.
    #[must_use]
    pub fn path(&self) -> &ObjectPath {
        &self.path
    }
}

/// Resolves storage URIs and reuses one object-store client per authority.
#[derive(Clone, Default)]
pub struct StorageRegistry {
    options: Arc<HashMap<String, String>>,
    stores: Arc<RwLock<HashMap<String, Arc<dyn ObjectStore>>>>,
}

impl std::fmt::Debug for StorageRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageRegistry")
            .field("option_keys", &self.options.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl StorageRegistry {
    /// Creates a registry using environment credentials and the supplied
    /// object-store or Hadoop configuration.
    #[must_use]
    pub fn new(options: HashMap<String, String>) -> Self {
        Self {
            options: Arc::new(options),
            stores: Arc::default(),
        }
    }

    /// Registers a caller-supplied object store for one URI authority.
    ///
    /// A later registration replaces the previous store for the same
    /// scheme-and-authority pair.
    pub fn register_store(
        &self,
        location: &str,
        store: Arc<dyn ObjectStore>,
    ) -> Result<Option<Arc<dyn ObjectStore>>> {
        let uri = parse_location(location)?;
        let key = base_url(&uri)?.to_string();
        let mut stores = self.stores.write().map_err(|_| Error::Poisoned)?;
        Ok(stores.insert(key, store))
    }

    /// Resolves an absolute `file`, `s3`, or `hdfs` URI.
    pub fn resolve(&self, location: &str) -> Result<ResolvedLocation> {
        let uri = parse_location(location)?;
        let base_url = base_url(&uri)?;
        let key = base_url.as_str().to_owned();
        let store = {
            let stores = self.stores.read().map_err(|_| Error::Poisoned)?;
            stores.get(&key).cloned()
        };
        let store = if let Some(store) = store {
            store
        } else {
            let store = self.build_store(&uri)?;
            let mut stores = self.stores.write().map_err(|_| Error::Poisoned)?;
            Arc::clone(stores.entry(key).or_insert(store))
        };
        let path = ObjectPath::from_url_path(uri.path())
            .map_err(|error| Error::InvalidLocation(error.to_string()))?;
        Ok(ResolvedLocation {
            uri,
            base_url,
            store,
            path,
        })
    }

    /// Resolves a concrete URI or expands a URI containing `*` path wildcards.
    ///
    /// Expansion is evaluated against the object store on every call. The
    /// pattern remains the durable table identity; returned object URIs are
    /// execution inputs only.
    pub async fn expand(&self, location: &str) -> Result<Vec<String>> {
        let resolved = self.resolve(location)?;
        let pattern = resolved.path().as_ref();
        let Some(wildcard) = pattern.find('*') else {
            return Ok(vec![resolved.uri().to_string()]);
        };
        let prefix = &pattern[..wildcard];
        let prefix = (!prefix.is_empty()).then(|| ObjectPath::from(prefix));
        let mut objects = resolved
            .store()
            .list(prefix.as_ref())
            .try_collect::<Vec<_>>()
            .await?;
        objects.retain(|object| wildcard_path_matches(pattern, object.location.as_ref()));
        objects.sort_unstable_by(|left, right| left.location.cmp(&right.location));
        if objects.is_empty() {
            return Err(Error::InvalidLocation(format!(
                "pattern matched no objects: {location}"
            )));
        }
        Ok(objects
            .into_iter()
            .map(|object| {
                let mut uri = resolved.base_url().clone();
                uri.set_path(&format!("/{}", object.location.as_ref()));
                uri.to_string()
            })
            .collect())
    }

    fn build_store(&self, uri: &Url) -> Result<Arc<dyn ObjectStore>> {
        if uri.scheme() == "hdfs" {
            let origin = base_url(uri)?;
            let store = HdfsObjectStoreBuilder::new()
                .with_url(origin.as_str())
                .with_config(
                    self.options
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone())),
                )
                .build()?;
            return Ok(Arc::new(store));
        }
        let (store, _) = object_store::parse_url_opts(
            uri,
            self.options
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone())),
        )?;
        Ok(Arc::from(store))
    }
}

fn wildcard_path_matches(pattern: &str, candidate: &str) -> bool {
    let pattern = pattern.split('/').collect::<Vec<_>>();
    let candidate = candidate.split('/').collect::<Vec<_>>();
    pattern.len() == candidate.len()
        && pattern
            .into_iter()
            .zip(candidate)
            .all(|(pattern, candidate)| wildcard_segment_matches(pattern, candidate))
}

fn wildcard_segment_matches(pattern: &str, candidate: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == candidate;
    }
    let mut remainder = candidate;
    let mut parts = pattern.split('*').peekable();
    if let Some(prefix) = parts.next() {
        let Some(after_prefix) = remainder.strip_prefix(prefix) else {
            return false;
        };
        remainder = after_prefix;
    }
    while let Some(part) = parts.next() {
        if part.is_empty() {
            continue;
        }
        if parts.peek().is_none() && !pattern.ends_with('*') {
            return remainder.ends_with(part);
        }
        let Some(position) = remainder.find(part) else {
            return false;
        };
        remainder = &remainder[position + part.len()..];
    }
    pattern.ends_with('*') || remainder.is_empty()
}

pub(crate) fn parse_location(location: &str) -> Result<Url> {
    let uri = Url::parse(location).map_err(|error| Error::InvalidLocation(error.to_string()))?;
    if !matches!(uri.scheme(), "file" | "s3" | "hdfs") {
        return Err(Error::InvalidLocation(format!(
            "unsupported URI scheme: {}",
            uri.scheme()
        )));
    }
    if uri.cannot_be_a_base()
        || !uri.username().is_empty()
        || uri.password().is_some()
        || uri.query().is_some()
        || uri.fragment().is_some()
    {
        return Err(Error::InvalidLocation(location.to_owned()));
    }
    if uri.scheme() != "file" && uri.host_str().is_none() {
        return Err(Error::InvalidLocation(format!(
            "{} URI requires an authority",
            uri.scheme()
        )));
    }
    if uri.scheme() == "file" && uri.host_str().is_some() {
        return Err(Error::InvalidLocation(
            "file URI must not contain an authority".into(),
        ));
    }
    Ok(uri)
}

fn base_url(uri: &Url) -> Result<Url> {
    let mut base = uri.clone();
    base.set_path("/");
    base.set_query(None);
    base.set_fragment(None);
    if base.scheme() == "file" {
        base.set_host(None)
            .map_err(|error| Error::InvalidLocation(error.to_string()))?;
    }
    Ok(base)
}
