//! Portable table identifiers and definitions.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::invalid;
use crate::serde_helpers::deserialize_unique_map;
use crate::{Error, Result};

const DEFINITION_FINGERPRINT_NAMESPACE: Uuid = Uuid::from_bytes([
    0x8f, 0xdd, 0xf2, 0xdf, 0x5b, 0x30, 0x46, 0x75, 0x91, 0x7c, 0xe9, 0x9f, 0x43, 0x32, 0xa5, 0x91,
]);

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct ExactTableState<'a> {
    provider: &'a str,
    identity: String,
    properties: &'a BTreeMap<String, String>,
}

/// A fully qualified logical table identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TableIdentifier {
    catalog: String,
    namespace: Vec<String>,
    name: String,
}

impl TableIdentifier {
    /// Creates and validates a fully qualified table identifier.
    pub fn new(
        catalog: impl Into<String>,
        namespace: Vec<String>,
        name: impl Into<String>,
    ) -> Result<Self> {
        let identifier = Self {
            catalog: catalog.into(),
            namespace,
            name: name.into(),
        };
        identifier.validate()?;
        Ok(identifier)
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

    /// Validates this identifier.
    pub fn validate(&self) -> Result<()> {
        if self.catalog.is_empty()
            || self.name.is_empty()
            || self.namespace.iter().any(String::is_empty)
        {
            return invalid("catalog, namespace segments, and name must be non-empty");
        }
        Ok(())
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

/// An exact, provider-neutral definition of one table state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TableDefinition {
    /// Fully qualified logical identifier.
    pub identifier: TableIdentifier,
    /// Registered `DataFusion` table-provider name.
    pub provider: String,
    /// Provider-defined, versioned, non-secret properties.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub properties: BTreeMap<String, String>,
}

impl TableDefinition {
    /// Creates and validates a table definition.
    pub fn new(
        identifier: TableIdentifier,
        provider: impl Into<String>,
        properties: BTreeMap<String, String>,
    ) -> Result<Self> {
        let definition = Self {
            identifier,
            provider: provider.into(),
            properties,
        };
        definition.validate()?;
        Ok(definition)
    }

    /// Validates provider-neutral invariants.
    pub fn validate(&self) -> Result<()> {
        self.identifier.validate()?;
        if self.provider.is_empty() {
            return invalid("table provider must not be empty");
        }
        if self.properties.keys().any(String::is_empty) {
            return invalid("table property names must not be empty");
        }
        Ok(())
    }

    /// Returns the deterministic exact-state fingerprint.
    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        let canonical = serde_json::to_vec(&ExactTableState {
            provider: &self.provider,
            identity: self.semantic_identity(),
            properties: &self.properties,
        })
        .map_err(|error| Error(error.to_string()))?;
        Ok(Uuid::new_v5(&DEFINITION_FINGERPRINT_NAMESPACE, &canonical).to_string())
    }

    /// Returns the stable logical identity shared by provider-state revisions.
    ///
    /// # Panics
    ///
    /// Panics only if serializing two strings to JSON fails.
    #[must_use]
    pub fn identity_key(&self) -> String {
        let canonical = serde_json::to_vec(&(self.provider.as_str(), self.semantic_identity()))
            .expect("table identity serialization is infallible");
        Uuid::new_v5(&DEFINITION_FINGERPRINT_NAMESPACE, &canonical).to_string()
    }

    /// Returns the exact-state key used for catalog matching.
    ///
    /// # Panics
    ///
    /// Panics if this definition is invalid or cannot be serialized.
    #[must_use]
    pub fn exact_state_key(&self) -> String {
        self.fingerprint()
            .expect("validated table definition has a fingerprint")
    }

    fn semantic_identity(&self) -> String {
        self.properties
            .get("table-identity")
            .cloned()
            .unwrap_or_else(|| {
                serde_json::to_string(&self.identifier)
                    .expect("table identifier serialization is infallible")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> TableDefinition {
        TableDefinition::new(
            TableIdentifier::new("datafusion", vec!["public".into()], "items").unwrap(),
            "parquet",
            BTreeMap::from([
                ("definition-version".into(), "1".into()),
                ("location".into(), "file:///tmp/items.parquet".into()),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn fingerprint_is_deterministic_and_state_sensitive() {
        let first = definition();
        let mut second = definition();
        assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());
        assert_eq!(first.identity_key(), second.identity_key());
        second
            .properties
            .insert("location".into(), "file:///tmp/next.parquet".into());
        assert_ne!(first.fingerprint().unwrap(), second.fingerprint().unwrap());
        assert_eq!(first.identity_key(), second.identity_key());
    }

    #[test]
    fn serde_rejects_unknown_fields() {
        let mut value = serde_json::to_value(definition()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<TableDefinition>(value).is_err());
    }
}
