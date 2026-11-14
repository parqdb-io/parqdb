//! Replacement for the archived `serde_yaml` 0.9 crate.
//!
//! `substrait` 0.63 parses its packaged JSON Schemas with `serde_yaml` while
//! generating the `text` module. DataFusion still pins that Substrait release,
//! and `unsafe-libyaml` 0.2.11 has no patched release. This crate provides the
//! `from_reader` and `from_str` entry points that build script calls, using
//! the safe `serde-saphyr` parser.

use serde::de::DeserializeOwned;
use std::io::Read;

/// Deserializes one YAML document from `reader`.
///
/// # Errors
///
/// Returns an error when the document is not valid YAML or does not match `T`.
pub fn from_reader<R, T>(reader: R) -> Result<T, serde_saphyr::Error>
where
    R: Read,
    T: DeserializeOwned,
{
    serde_saphyr::from_reader(reader)
}

/// Deserializes one YAML document from `input`.
///
/// # Errors
///
/// Returns an error when the document is not valid YAML or does not match `T`.
pub fn from_str<T>(input: &str) -> Result<T, serde_saphyr::Error>
where
    T: DeserializeOwned,
{
    serde_saphyr::from_str(input)
}
