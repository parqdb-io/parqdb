use serde::{Deserialize, Serialize};
use url::{Host, Url};
use uuid::Uuid;

use crate::Result;
use crate::error::invalid;
use crate::serde_helpers::lowercase_uuid;

/// Portable reference to a source or index table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "lowercase", deny_unknown_fields)]
pub enum RelationReference {
    /// Parquet table identified by a canonical absolute URI.
    Parquet {
        /// Canonical absolute URI of the Parquet table root.
        uri: String,
    },
    /// Iceberg table identified by catalog, name, UUID, and snapshot.
    Iceberg {
        /// Logical runtime name of the registered Iceberg catalog.
        catalog: String,
        /// Ordered Iceberg namespace segments.
        namespace: Vec<String>,
        /// Iceberg table name.
        name: String,
        /// Stable Iceberg table UUID.
        #[serde(rename = "table-uuid", with = "lowercase_uuid")]
        table_uuid: Uuid,
        /// Exact Iceberg snapshot ID.
        #[serde(rename = "snapshot-id")]
        snapshot_id: i64,
    },
}

impl RelationReference {
    #[must_use]
    /// Returns the profile-defined stable identity key.
    pub fn identity_key(&self) -> String {
        match self {
            Self::Parquet { uri } => format!("parquet\0{uri}"),
            Self::Iceberg { table_uuid, .. } => format!("iceberg\0{table_uuid}"),
        }
    }

    #[must_use]
    /// Returns the profile-defined exact-state key.
    pub fn exact_state_key(&self) -> String {
        match self {
            Self::Parquet { uri } => format!("parquet\0{uri}"),
            Self::Iceberg {
                table_uuid,
                snapshot_id,
                ..
            } => format!("iceberg\0{table_uuid}\0{snapshot_id}"),
        }
    }

    /// Validates the reference against its relation profile.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Parquet { uri } => validate_parquet_uri(uri),
            Self::Iceberg {
                catalog,
                namespace,
                name,
                snapshot_id,
                ..
            } => {
                if catalog.is_empty()
                    || name.is_empty()
                    || namespace.iter().any(String::is_empty)
                    || *snapshot_id <= 0
                {
                    return invalid("invalid Iceberg relation reference");
                }
                Ok(())
            }
        }
    }
}

fn validate_parquet_uri(uri: &str) -> Result<()> {
    let parsed = Url::parse(uri).map_err(|error| crate::Error(error.to_string()))?;
    let scheme_end = uri
        .find(':')
        .ok_or_else(|| crate::Error(format!("invalid Parquet table URI: {uri}")))?;
    let raw_scheme = &uri[..scheme_end];
    if raw_scheme != raw_scheme.to_ascii_lowercase()
        || parsed.username() != ""
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return invalid(format!("invalid Parquet table URI: {uri}"));
    }

    let (authority, path) = raw_authority_and_path(&uri[scheme_end + 1..]);
    if authority.contains('@')
        || path.contains("//")
        || path
            .split('/')
            .any(|segment| segment == "." || segment == "..")
    {
        return invalid(format!("non-canonical Parquet table URI: {uri}"));
    }
    if matches!(parsed.host(), Some(Host::Domain(_))) {
        let raw_host = authority
            .rsplit_once(':')
            .map_or(authority, |(host, _port)| host);
        if raw_host.chars().any(char::is_uppercase) {
            return invalid(format!("non-canonical Parquet table URI: {uri}"));
        }
    }
    validate_percent_encodings(uri)?;
    Ok(())
}

fn raw_authority_and_path(after_scheme: &str) -> (&str, &str) {
    let Some(without_prefix) = after_scheme.strip_prefix("//") else {
        return ("", after_scheme);
    };
    match without_prefix.find('/') {
        Some(path_start) => (&without_prefix[..path_start], &without_prefix[path_start..]),
        None => (without_prefix, ""),
    }
}

fn validate_percent_encodings(uri: &str) -> Result<()> {
    let bytes = uri.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        if index + 2 >= bytes.len()
            || !bytes[index + 1].is_ascii_hexdigit()
            || !bytes[index + 2].is_ascii_hexdigit()
            || bytes[index + 1].is_ascii_lowercase()
            || bytes[index + 2].is_ascii_lowercase()
        {
            return invalid(format!("non-canonical percent encoding in URI: {uri}"));
        }
        let decoded = (hex_value(bytes[index + 1]) << 4) | hex_value(bytes[index + 2]);
        if decoded.is_ascii_alphanumeric()
            || matches!(decoded, b'-' | b'.' | b'_' | b'~' | b'/' | b'*')
        {
            return invalid(format!("non-canonical percent encoding in URI: {uri}"));
        }
        index += 3;
    }
    Ok(())
}

fn hex_value(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'A'..=b'F' => byte - b'A' + 10,
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("hex_value is called only for ASCII hexadecimal digits"),
    }
}
