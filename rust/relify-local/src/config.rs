use std::any::Any;
use std::fmt::Display;

use datafusion::common::config::{
    ConfigEntry, ConfigExtension, ConfigField, ExtensionOptions, Visit,
};
use datafusion::common::{DataFusionError, Result as DataFusionResult, config_namespace};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::SessionConfig;
use relify_index::{
    DEFAULT_METADATA_CACHE_BYTES, DEFAULT_METADATA_CACHE_ENTRIES, MetadataCacheConfig,
};

config_namespace! {
    /// Options for index metadata caching.
    pub struct MetadataCacheOptions {
        /// Maximum number of cached metadata documents.
        pub max_entries: usize, default = DEFAULT_METADATA_CACHE_ENTRIES

        /// Maximum serialized bytes represented by cached metadata documents.
        pub max_bytes: usize, default = DEFAULT_METADATA_CACHE_BYTES
    }
}

config_namespace! {
    /// Options for index metadata.
    pub struct MetadataOptions {
        /// Metadata cache options.
        pub cache: MetadataCacheOptions, default = MetadataCacheOptions::default()
    }
}

/// Relify options carried by `DataFusion`'s session configuration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RelifyConfig {
    /// Index metadata options.
    pub metadata: MetadataOptions,
}

impl ConfigExtension for RelifyConfig {
    const PREFIX: &'static str = "relify";
}

impl ExtensionOptions for RelifyConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> DataFusionResult<()> {
        ConfigField::set(self, key, value)
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        struct EntryVisitor(Vec<ConfigEntry>);

        impl Visit for EntryVisitor {
            fn some<V: Display>(&mut self, key: &str, value: V, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_owned(),
                    value: Some(value.to_string()),
                    description,
                });
            }

            fn none(&mut self, key: &str, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_owned(),
                    value: None,
                    description,
                });
            }
        }

        let mut visitor = EntryVisitor(Vec::new());
        ConfigField::visit(self, &mut visitor, "relify", "");
        visitor.0
    }
}

impl ConfigField for RelifyConfig {
    fn visit<V: Visit>(&self, visitor: &mut V, key_prefix: &str, _description: &'static str) {
        self.metadata.visit(
            visitor,
            &format!("{key_prefix}.metadata"),
            "Index metadata options.",
        );
    }

    fn set(&mut self, key: &str, value: &str) -> DataFusionResult<()> {
        let (section, remainder) = key.split_once('.').unwrap_or((key, ""));
        match section {
            "metadata" => self.metadata.set(remainder, value),
            _ => Err(DataFusionError::Configuration(format!(
                "Config value \"{section}\" not found on RelifyConfig"
            ))),
        }
    }
}

/// `DataFusion` initialization options for one local Relify session.
#[derive(Clone)]
pub struct LocalSessionOptions {
    config: SessionConfig,
    runtime: RuntimeEnvBuilder,
}

impl LocalSessionOptions {
    /// Creates options from `DataFusion`'s native configuration types.
    #[must_use]
    pub const fn new(config: SessionConfig, runtime: RuntimeEnvBuilder) -> Self {
        Self { config, runtime }
    }

    pub(crate) fn into_parts(self) -> (SessionConfig, RuntimeEnvBuilder) {
        (ensure_relify_config(self.config), self.runtime)
    }
}

impl Default for LocalSessionOptions {
    fn default() -> Self {
        Self::new(relify_session_config(), RuntimeEnvBuilder::default())
    }
}

/// Creates a `DataFusion` session configuration with Relify options installed.
#[must_use]
pub fn relify_session_config() -> SessionConfig {
    ensure_relify_config(SessionConfig::new())
}

fn ensure_relify_config(mut config: SessionConfig) -> SessionConfig {
    if config.options().extensions.get::<RelifyConfig>().is_none() {
        config
            .options_mut()
            .extensions
            .insert(RelifyConfig::default());
    }
    config
}

pub(crate) fn metadata_cache_config(config: &SessionConfig) -> MetadataCacheConfig {
    let cache = &config
        .options()
        .extensions
        .get::<RelifyConfig>()
        .expect("Relify config extension must be installed")
        .metadata
        .cache;
    MetadataCacheConfig::new(cache.max_entries, cache.max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relify_config_uses_hierarchical_datafusion_keys() {
        let config = relify_session_config()
            .set_str("relify.metadata.cache.max_entries", "7")
            .set_str("relify.metadata.cache.max_bytes", "4096");
        let (config, _) =
            LocalSessionOptions::new(config, RuntimeEnvBuilder::default()).into_parts();

        assert_eq!(
            metadata_cache_config(&config),
            MetadataCacheConfig::new(7, 4096)
        );
    }

    #[test]
    fn prepared_relify_config_accepts_runtime_mutation() {
        let (mut config, _) = LocalSessionOptions::default().into_parts();
        config
            .options_mut()
            .set("relify.metadata.cache.max_entries", "7")
            .unwrap();

        assert_eq!(metadata_cache_config(&config).max_entries, 7);
    }
}
