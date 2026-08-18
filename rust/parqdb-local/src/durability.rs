//! Local filesystem durability helpers.

use std::fs;
#[cfg(unix)]
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::Result;

pub(crate) fn create_dir_all(path: &Path) -> Result<()> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut missing = Vec::<PathBuf>::new();
    let mut current = absolute.as_path();
    while !current.exists() {
        missing.push(current.to_owned());
        current = current.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("directory has no existing ancestor: {}", absolute.display()),
            )
        })?;
    }

    fs::create_dir_all(&absolute)?;
    for directory in missing.iter().rev() {
        sync_directory(directory)?;
        if let Some(parent) = directory.parent() {
            sync_directory(parent)?;
        }
    }
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn creates_and_syncs_nested_directories() {
        let temporary = TempDir::new().unwrap();
        let nested = temporary.path().join("a").join("b").join("c");

        create_dir_all(&nested).unwrap();
        sync_directory(&nested).unwrap();

        assert!(nested.is_dir());
    }
}
