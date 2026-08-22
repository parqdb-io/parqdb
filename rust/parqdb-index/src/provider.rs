//! Provider boundary for physical index tables.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use parqdb_meta::{IndexProviderDefinition, IndexSnapshot, IndexTableDefinition};

use crate::Result;

/// A family-defined role such as `ivf_centroids` or `ivf_postings`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexTableRole(String);

impl IndexTableRole {
    /// Creates a non-empty role.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() {
            return Err(crate::Error::InvalidProvider(
                "index table role must not be empty".into(),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the role name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IndexTableRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Typed pruning requested while opening an index table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexSelection {
    /// Sorted, deduplicated cluster IDs, or `None` for every row.
    pub cids: Option<Arc<[i32]>>,
}

impl IndexSelection {
    /// Creates and validates a CID selection.
    pub fn cids(cids: impl Into<Arc<[i32]>>) -> Result<Self> {
        let cids = cids.into();
        if cids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(crate::Error::InvalidProvider(
                "selected CIDs must be sorted and deduplicated".into(),
            ));
        }
        Ok(Self { cids: Some(cids) })
    }
}

/// Role-keyed physical plans produced by an index family.
#[derive(Clone, Default)]
pub struct IndexWriteInput {
    /// Complete table inputs for one write operation.
    pub tables: BTreeMap<IndexTableRole, Arc<dyn ExecutionPlan>>,
}

/// Immutable definitions created by an index write.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexWriteResult {
    /// Complete role-to-table mapping for the new snapshot.
    pub index_tables: BTreeMap<IndexTableRole, IndexTableDefinition>,
}

/// Common request data for a full index creation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CreateIndexRequest {
    /// Provider-defined, versioned, non-secret write properties.
    pub properties: BTreeMap<String, String>,
}

/// Common request data for an incremental index update.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpdateIndexRequest {
    /// Tables from the parent snapshot.
    pub previous_tables: BTreeMap<IndexTableRole, IndexTableDefinition>,
    /// Provider-defined, versioned, non-secret write properties.
    pub properties: BTreeMap<String, String>,
}

/// Common request data for index compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactIndexRequest {
    /// Tables to compact.
    pub current_tables: BTreeMap<IndexTableRole, IndexTableDefinition>,
    /// Provider-defined, versioned, non-secret write properties.
    pub properties: BTreeMap<String, String>,
}

/// Executable provider-specific index write.
#[async_trait]
pub trait IndexWritePlan: Send + Sync {
    /// Executes the write and returns immutable table definitions.
    async fn execute(&self, context: Arc<TaskContext>) -> Result<IndexWriteResult>;
}

/// Opens and writes the physical tables of one index provider.
#[async_trait]
pub trait IndexProvider: Send + Sync {
    /// Validates provider-specific snapshot metadata before planning reads.
    async fn validate_snapshot(&self, snapshot: &IndexSnapshot) -> Result<()>;

    /// Opens one immutable index table with typed pruning already bound.
    async fn open_index_table(
        &self,
        role: &IndexTableRole,
        table: &IndexTableDefinition,
        selection: &IndexSelection,
    ) -> Result<Arc<dyn TableProvider>>;

    /// Plans a full index creation.
    async fn plan_create(
        &self,
        request: CreateIndexRequest,
        input: IndexWriteInput,
    ) -> Result<Box<dyn IndexWritePlan>>;

    /// Plans an incremental update.
    async fn plan_update(
        &self,
        request: UpdateIndexRequest,
        input: IndexWriteInput,
    ) -> Result<Box<dyn IndexWritePlan>>;

    /// Plans compaction of existing immutable tables.
    async fn plan_compact(&self, request: CompactIndexRequest) -> Result<Box<dyn IndexWritePlan>>;

    /// Resolves warehouse-managed locations referenced by immutable tables.
    async fn managed_table_locations(
        &self,
        tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
    ) -> Result<Vec<String>>;

    /// Deletes physical tables after common metadata establishes they are unreachable.
    async fn delete_index_tables(
        &self,
        tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
    ) -> Result<()>;
}

/// Creates an index provider from persisted provider metadata.
#[async_trait]
pub trait IndexProviderFactory: Send + Sync {
    /// Validates and opens one provider instance.
    async fn open(
        &self,
        session: &dyn Session,
        definition: &IndexProviderDefinition,
    ) -> Result<Arc<dyn IndexProvider>>;
}

/// Session-independent registry of named index-provider factories.
#[derive(Default)]
pub struct IndexProviderRegistry {
    factories: RwLock<BTreeMap<String, Arc<dyn IndexProviderFactory>>>,
}

impl IndexProviderRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one provider name. Existing names are not replaced.
    pub fn register(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn IndexProviderFactory>,
    ) -> Result<()> {
        let name = name.into();
        let name = normalized_provider_name(&name)?;
        let mut factories = self.factories.write().map_err(|_| {
            crate::Error::InvalidProvider("provider registry lock is poisoned".into())
        })?;
        if factories.contains_key(&name) {
            return Err(crate::Error::InvalidProvider(format!(
                "index provider factory is already registered: {name}"
            )));
        }
        factories.insert(name, factory);
        Ok(())
    }

    /// Opens the provider selected by a persisted definition.
    pub async fn open(
        &self,
        session: &dyn Session,
        definition: &IndexProviderDefinition,
    ) -> Result<Arc<dyn IndexProvider>> {
        definition.validate()?;
        let name = normalized_provider_name(&definition.provider)?;
        let factory = self
            .factories
            .read()
            .map_err(|_| {
                crate::Error::InvalidProvider("provider registry lock is poisoned".into())
            })?
            .get(&name)
            .cloned()
            .ok_or_else(|| {
                crate::Error::InvalidProvider(format!(
                    "index provider factory is not registered: {name}"
                ))
            })?;
        factory.open(session, definition).await
    }
}

fn normalized_provider_name(name: &str) -> Result<String> {
    let name = name.trim().to_ascii_lowercase();
    if name.is_empty() {
        return Err(crate::Error::InvalidProvider(
            "index provider name must not be empty".into(),
        ));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;

    use super::*;

    struct TestFactory;

    #[async_trait]
    impl IndexProviderFactory for TestFactory {
        async fn open(
            &self,
            _session: &dyn Session,
            definition: &IndexProviderDefinition,
        ) -> Result<Arc<dyn IndexProvider>> {
            if definition.properties.get("version").map(String::as_str) != Some("1") {
                return Err(crate::Error::InvalidProvider(
                    "test provider requires version=1".into(),
                ));
            }
            Ok(Arc::new(TestProvider))
        }
    }

    struct TestProvider;

    #[async_trait]
    impl IndexProvider for TestProvider {
        async fn validate_snapshot(&self, _snapshot: &IndexSnapshot) -> Result<()> {
            Ok(())
        }

        async fn open_index_table(
            &self,
            _role: &IndexTableRole,
            _table: &IndexTableDefinition,
            _selection: &IndexSelection,
        ) -> Result<Arc<dyn TableProvider>> {
            Ok(Arc::new(MemTable::try_new(
                Arc::new(Schema::empty()),
                vec![Vec::new()],
            )?))
        }

        async fn plan_create(
            &self,
            _request: CreateIndexRequest,
            _input: IndexWriteInput,
        ) -> Result<Box<dyn IndexWritePlan>> {
            Err(crate::Error::UnsupportedProviderOperation("create".into()))
        }

        async fn plan_update(
            &self,
            _request: UpdateIndexRequest,
            _input: IndexWriteInput,
        ) -> Result<Box<dyn IndexWritePlan>> {
            Err(crate::Error::UnsupportedProviderOperation("update".into()))
        }

        async fn plan_compact(
            &self,
            _request: CompactIndexRequest,
        ) -> Result<Box<dyn IndexWritePlan>> {
            Err(crate::Error::UnsupportedProviderOperation("compact".into()))
        }

        async fn delete_index_tables(
            &self,
            _tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
        ) -> Result<()> {
            Ok(())
        }

        async fn managed_table_locations(
            &self,
            _tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
        ) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn registry_dispatches_a_second_provider_without_common_code_changes() {
        let registry = IndexProviderRegistry::new();
        registry.register("test", Arc::new(TestFactory)).unwrap();
        let definition =
            IndexProviderDefinition::new("test", BTreeMap::from([("version".into(), "1".into())]))
                .unwrap();
        let state = SessionContext::new().state();
        let provider = registry.open(&state, &definition).await.unwrap();
        let table = IndexTableDefinition::new(1, BTreeMap::new()).unwrap();
        provider
            .open_index_table(
                &IndexTableRole::new("test_table").unwrap(),
                &table,
                &IndexSelection::default(),
            )
            .await
            .unwrap();
    }

    #[test]
    fn cid_selection_requires_sorted_unique_values() {
        assert!(IndexSelection::cids(Arc::<[i32]>::from([1, 3, 7])).is_ok());
        assert!(IndexSelection::cids(Arc::<[i32]>::from([1, 1])).is_err());
        assert!(IndexSelection::cids(Arc::<[i32]>::from([2, 1])).is_err());
    }
}
