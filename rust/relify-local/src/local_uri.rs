//! Canonical local `file` URI conversion.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

pub(crate) fn path_to_file_uri(path: &Path) -> Result<String> {
    url::Url::from_file_path(path)
        .map(String::from)
        .map_err(|()| {
            Error::InvalidArgument(format!("cannot convert path to URI: {}", path.display()))
        })
}

pub(crate) fn directory_to_file_uri(path: &Path) -> Result<String> {
    url::Url::from_directory_path(path)
        .map(String::from)
        .map_err(|()| {
            Error::InvalidArgument(format!("cannot convert path to URI: {}", path.display()))
        })
}

pub(crate) fn file_uri_to_path(uri: &str) -> Result<PathBuf> {
    let parsed = url::Url::parse(uri).map_err(|error| Error::InvalidMetadata(error.to_string()))?;
    if parsed.scheme() != "file" {
        return Err(Error::InvalidArgument(format!(
            "local session supports only file URIs, received: {uri}"
        )));
    }
    parsed
        .to_file_path()
        .map_err(|()| Error::InvalidMetadata(format!("invalid file URI: {uri}")))
}
