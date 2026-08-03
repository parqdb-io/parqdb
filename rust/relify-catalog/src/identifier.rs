use std::fmt;

use crate::{Error, Result};

/// A structured index identifier consisting of a namespace and a name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexIdentifier {
    namespace: Vec<String>,
    name: String,
}

/// A fully qualified table identifier consisting of a catalog, namespace, and name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableIdentifier {
    catalog: String,
    namespace: Vec<String>,
    name: String,
}

impl TableIdentifier {
    /// Creates a fully qualified table identifier.
    pub fn new(
        catalog: impl Into<String>,
        namespace: Vec<String>,
        name: impl Into<String>,
    ) -> Result<Self> {
        let catalog = catalog.into();
        let name = name.into();
        if catalog.is_empty() || name.is_empty() || namespace.iter().any(String::is_empty) {
            return Err(Error::InvalidIdentifier(
                "catalog, namespace segments, and name must be non-empty".into(),
            ));
        }
        Ok(Self {
            catalog,
            namespace,
            name,
        })
    }

    /// Returns the catalog name.
    #[must_use]
    pub fn catalog(&self) -> &str {
        &self.catalog
    }

    /// Returns the namespace segments.
    #[must_use]
    pub fn namespace(&self) -> &[String] {
        &self.namespace
    }

    /// Returns the table name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn namespace_key(&self) -> Result<String> {
        namespace_key(&self.namespace)
    }
}

impl fmt::Display for TableIdentifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "catalog={:?}, namespace={:?}, name={:?}",
            self.catalog, self.namespace, self.name
        )
    }
}

impl IndexIdentifier {
    /// Creates an identifier after validating every namespace segment and the name.
    pub fn new(namespace: Vec<String>, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.is_empty() || namespace.iter().any(String::is_empty) {
            return Err(Error::InvalidIdentifier(
                "namespace segments and name must be non-empty".into(),
            ));
        }
        Ok(Self { namespace, name })
    }

    /// Creates an identifier in the root namespace.
    pub fn root(name: impl Into<String>) -> Result<Self> {
        Self::new(Vec::new(), name)
    }

    /// Returns the namespace segments.
    #[must_use]
    pub fn namespace(&self) -> &[String] {
        &self.namespace
    }

    /// Returns the index name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn namespace_key(&self) -> Result<String> {
        namespace_key(&self.namespace)
    }
}

impl fmt::Display for IndexIdentifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "namespace={:?}, name={:?}",
            self.namespace, self.name
        )
    }
}

#[cfg(feature = "sqlite")]
pub(crate) fn namespace_key(namespace: &[String]) -> Result<String> {
    if namespace.iter().any(String::is_empty) {
        return Err(Error::InvalidIdentifier(
            "namespace segments must be non-empty".into(),
        ));
    }
    Ok(serde_json::to_string(namespace)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_preserves_structured_namespace_and_unicode() {
        let identifier =
            IndexIdentifier::new(vec!["analytics".into(), "中文".into()], "documents").unwrap();
        assert_eq!(identifier.namespace(), ["analytics", "中文"]);
        assert_eq!(identifier.name(), "documents");
        assert_ne!(
            identifier,
            IndexIdentifier::new(vec!["analytics.中文".into()], "documents").unwrap()
        );
    }

    #[test]
    fn identifier_rejects_empty_segments_and_names() {
        assert!(IndexIdentifier::root("").is_err());
        assert!(IndexIdentifier::new(vec![String::new()], "documents").is_err());
        assert!(TableIdentifier::new("", vec!["public".into()], "documents").is_err());
        assert!(TableIdentifier::new("datafusion", vec![String::new()], "documents").is_err());
    }
}
