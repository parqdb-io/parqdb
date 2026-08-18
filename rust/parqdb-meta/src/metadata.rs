//! Metadata document and snapshot types.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;

use crate::error::invalid;
use crate::family::validate_family;
use crate::serde_helpers::{deserialize_unique_map, lowercase_uuid};
use crate::{Error, Result};

/// Entry recording when an index snapshot became current.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SnapshotLogEntry {
    /// Unix epoch time in milliseconds.
    pub timestamp_ms: i64,
    /// Snapshot that became current.
    pub snapshot_id: i64,
}

/// Immutable logical state of one `ParqDB` index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct IndexSnapshot {
    /// Positive snapshot ID unique within the index.
    pub snapshot_id: i64,
    /// Positive commit sequence number.
    pub sequence_number: i64,
    /// Snapshot creation time as Unix epoch milliseconds.
    pub timestamp_ms: i64,
    /// Non-semantic provenance values.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub summary: BTreeMap<String, String>,
    /// Source column containing vectors.
    pub vector_field: String,
    /// Ordered source unique-key columns.
    pub source_key_fields: Vec<String>,
    /// Number of source rows represented by this snapshot.
    pub indexed_rows: i64,
    /// Index-family identifier.
    pub index_family: String,
    /// Family schema version.
    pub index_schema_version: i32,
    /// Distance metric identifier.
    pub metric: String,
    /// Family-defined canonical parameters.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub parameters: BTreeMap<String, String>,
    /// Family role to warehouse-relative index-table path.
    #[serde(deserialize_with = "deserialize_unique_map")]
    pub index_relations: BTreeMap<String, String>,
}

impl IndexSnapshot {
    /// Validates the snapshot and its family-defined fields.
    pub fn validate(&self) -> Result<()> {
        if self.snapshot_id <= 0 || self.sequence_number <= 0 || self.timestamp_ms < 0 {
            return invalid(
                "snapshot IDs and sequence numbers must be positive and timestamps must be non-negative",
            );
        }
        if self.vector_field.is_empty()
            || self.source_key_fields.is_empty()
            || self.source_key_fields.iter().any(String::is_empty)
        {
            return invalid("vector-field and source-key-fields must be non-empty");
        }
        let unique_keys = self.source_key_fields.iter().collect::<HashSet<_>>();
        if unique_keys.len() != self.source_key_fields.len() {
            return invalid("source-key-fields must not contain duplicates");
        }
        if self.indexed_rows <= 0 {
            return invalid("indexed-rows must be positive");
        }
        if self.summary.keys().any(String::is_empty)
            || self.parameters.keys().any(String::is_empty)
            || self.index_relations.keys().any(String::is_empty)
        {
            return invalid("map keys must be non-empty");
        }
        for location in self.index_relations.values() {
            validate_relative_location(location)?;
        }
        validate_family(self)
    }

    /// Reads a positive family parameter as `usize`.
    pub fn parameter_usize(&self, name: &str) -> Result<usize> {
        let value = self
            .parameters
            .get(name)
            .ok_or_else(|| Error(format!("missing parameter: {name}")))?;
        value
            .parse()
            .map_err(|_| Error(format!("invalid parameter: {name}")))
    }
}

/// Complete immutable metadata document for a logical `ParqDB` index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct IndexMetadata {
    /// Metadata format version.
    pub format_version: i32,
    /// Stable UUID of the logical index.
    #[serde(with = "lowercase_uuid")]
    pub index_uuid: Uuid,
    /// Metadata creation time as Unix epoch milliseconds.
    pub last_updated_ms: i64,
    /// Greatest snapshot sequence number ever allocated.
    pub last_sequence_number: i64,
    /// ID of the current retained snapshot.
    pub current_snapshot_id: i64,
    /// Current and retained immutable snapshots.
    pub snapshots: Vec<IndexSnapshot>,
    /// History of snapshots becoming current.
    pub snapshot_log: Vec<SnapshotLogEntry>,
    /// Non-semantic index properties.
    #[serde(default, deserialize_with = "deserialize_unique_map")]
    pub properties: BTreeMap<String, String>,
}

impl IndexMetadata {
    /// Parses and validates one JSON metadata document.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let metadata: Self =
            serde_json::from_slice(bytes).map_err(|error| Error(error.to_string()))?;
        metadata.validate()?;
        Ok(metadata)
    }

    /// Validates this metadata document independently.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return invalid(format!(
                "unsupported format-version {}",
                self.format_version
            ));
        }
        if self.last_updated_ms < 0
            || self.last_sequence_number <= 0
            || self.current_snapshot_id <= 0
            || self.snapshots.is_empty()
            || self.snapshot_log.is_empty()
        {
            return invalid("invalid metadata timestamps, sequence, or snapshot state");
        }
        if self.properties.keys().any(String::is_empty) {
            return invalid("map keys must be non-empty");
        }

        let mut snapshot_ids = HashSet::new();
        let mut sequence_numbers = HashSet::new();
        let identity = SnapshotIdentity::from_snapshot(&self.snapshots[0]);
        for snapshot in &self.snapshots {
            snapshot.validate()?;
            if SnapshotIdentity::from_snapshot(snapshot) != identity {
                return invalid("logical identity fields must remain equal across snapshots");
            }
            if !snapshot_ids.insert(snapshot.snapshot_id)
                || !sequence_numbers.insert(snapshot.sequence_number)
                || snapshot.sequence_number > self.last_sequence_number
                || snapshot.timestamp_ms > self.last_updated_ms
            {
                return invalid("invalid or duplicate snapshot identity");
            }
        }
        if !snapshot_ids.contains(&self.current_snapshot_id) {
            return invalid("current-snapshot-id is not retained");
        }

        let mut previous_timestamp = None;
        for entry in &self.snapshot_log {
            if entry.timestamp_ms < 0
                || entry.timestamp_ms > self.last_updated_ms
                || !snapshot_ids.contains(&entry.snapshot_id)
                || previous_timestamp.is_some_and(|previous| entry.timestamp_ms < previous)
            {
                return invalid("invalid snapshot-log");
            }
            previous_timestamp = Some(entry.timestamp_ms);
        }
        if self.snapshot_log.last().map(|entry| entry.snapshot_id) != Some(self.current_snapshot_id)
        {
            return invalid("snapshot-log must end at current-snapshot-id");
        }
        Ok(())
    }

    /// Validates this document as a legal update from `base`.
    pub fn validate_update_from(&self, base: &Self) -> Result<()> {
        base.validate()?;
        self.validate()?;
        if self.index_uuid != base.index_uuid {
            return invalid("index-uuid must remain unchanged");
        }
        if self.last_updated_ms < base.last_updated_ms {
            return invalid("last-updated-ms must not decrease");
        }
        if self.last_sequence_number < base.last_sequence_number {
            return invalid("last-sequence-number must not decrease");
        }

        let base_identity = SnapshotIdentity::from_snapshot(base.current_snapshot()?);
        if self
            .snapshots
            .iter()
            .any(|snapshot| SnapshotIdentity::from_snapshot(snapshot) != base_identity)
        {
            return invalid("logical identity fields must remain equal to the base metadata");
        }

        let base_snapshots = base
            .snapshots
            .iter()
            .map(|snapshot| (snapshot.snapshot_id, snapshot))
            .collect::<HashMap<_, _>>();
        let mut added = Vec::new();
        for snapshot in &self.snapshots {
            match base_snapshots.get(&snapshot.snapshot_id) {
                Some(base_snapshot) if *base_snapshot != snapshot => {
                    return invalid("retained snapshots are immutable");
                }
                Some(_) => {}
                None => added.push(snapshot),
            }
        }
        if added.len() > 1 {
            return invalid("one metadata update may add at most one snapshot");
        }
        match added.as_slice() {
            [] if self.last_sequence_number != base.last_sequence_number => {
                return invalid("last-sequence-number changed without a new snapshot");
            }
            [snapshot] => {
                let expected_sequence = base
                    .last_sequence_number
                    .checked_add(1)
                    .ok_or_else(|| Error("last-sequence-number is exhausted".into()))?;
                if snapshot.sequence_number != expected_sequence
                    || self.last_sequence_number != expected_sequence
                    || self.current_snapshot_id != snapshot.snapshot_id
                {
                    return invalid(
                        "a new snapshot must use the next sequence number and become current",
                    );
                }
            }
            _ => {}
        }

        let retained_snapshot_ids = self
            .snapshots
            .iter()
            .map(|snapshot| snapshot.snapshot_id)
            .collect::<HashSet<_>>();
        let retained_base_log = base
            .snapshot_log
            .iter()
            .filter(|entry| retained_snapshot_ids.contains(&entry.snapshot_id))
            .collect::<Vec<_>>();
        if self.snapshot_log.len() < retained_base_log.len()
            || self
                .snapshot_log
                .iter()
                .zip(&retained_base_log)
                .any(|(entry, base_entry)| entry != *base_entry)
        {
            return invalid("snapshot-log history for retained snapshots is immutable");
        }
        let appended_log = &self.snapshot_log[retained_base_log.len()..];
        if self.current_snapshot_id == base.current_snapshot_id {
            if !appended_log.is_empty() {
                return invalid("snapshot-log changed without changing the current snapshot");
            }
        } else if appended_log.len() != 1 || appended_log[0].snapshot_id != self.current_snapshot_id
        {
            return invalid("changing the current snapshot must append one snapshot-log entry");
        }
        Ok(())
    }

    /// Returns the current retained snapshot.
    pub fn current_snapshot(&self) -> Result<&IndexSnapshot> {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.snapshot_id == self.current_snapshot_id)
            .ok_or_else(|| Error("current-snapshot-id is not retained".into()))
    }
}

#[derive(PartialEq, Eq)]
struct SnapshotIdentity {
    vector_field: String,
    source_key_fields: Vec<String>,
    index_family: String,
    metric: String,
}

impl SnapshotIdentity {
    fn from_snapshot(snapshot: &IndexSnapshot) -> Self {
        Self {
            vector_field: snapshot.vector_field.clone(),
            source_key_fields: snapshot.source_key_fields.clone(),
            index_family: snapshot.index_family.clone(),
            metric: snapshot.metric.clone(),
        }
    }
}

/// Validates one canonical path below the warehouse root.
pub fn validate_relative_location(location: &str) -> Result<()> {
    let path = location.strip_suffix('/').unwrap_or(location);
    if location.is_empty()
        || location.starts_with('/')
        || location.contains('\\')
        || location.contains('?')
        || location.contains('#')
        || path.is_empty()
        || path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return invalid(format!("invalid warehouse-relative location: {location}"));
    }
    Ok(())
}

/// Validates one deployment-owned absolute storage URI.
pub fn validate_absolute_location(location: &str) -> Result<()> {
    let parsed = Url::parse(location).map_err(|error| Error(error.to_string()))?;
    if (!parsed.has_host() && parsed.scheme() != "file")
        || parsed.username() != ""
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return invalid(
            "location must be an absolute URI without user information, query, or fragment",
        );
    }
    Ok(())
}

pub(crate) fn parse_positive_parameter(
    parameters: &BTreeMap<String, String>,
    name: &str,
    maximum: u64,
) -> Result<u64> {
    let value = parameters
        .get(name)
        .ok_or_else(|| Error(format!("missing parameter: {name}")))?;
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return invalid(format!("{name} is not a canonical positive integer"));
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| Error(format!("{name} is out of range")))?;
    if parsed == 0 || parsed > maximum {
        return invalid(format!("{name} is out of range"));
    }
    Ok(parsed)
}
