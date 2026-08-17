//! Cross-process coordination for the embedded backend.

use std::collections::HashSet;
use std::fs::{File, OpenOptions, read_dir, remove_file};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use crate::Result;
use crate::durability::{create_dir_all, sync_directory};
use parqdb_catalog::IndexIdentifier;
use uuid::Uuid;

const LOCK_DIRECTORY: &str = ".locks";
const SESSION_LOCK: &str = "session.lock";
const BUILD_LEASE_DIRECTORY: &str = "build-leases";

/// Cross-process coordination for one local catalog state directory.
#[derive(Debug, Clone)]
pub(crate) struct SessionCoordination {
    lock_path: PathBuf,
    build_lease_root: PathBuf,
}

/// A held session lock.
#[derive(Debug)]
pub(crate) struct SessionGuard {
    _file: File,
}

/// An exclusive build reservation that can pin its unpublished snapshot root.
#[derive(Debug)]
pub(crate) struct BuildLease {
    file: File,
    setup_guard: Option<SessionGuard>,
    session_lock_path: PathBuf,
    roots: Vec<String>,
}

impl SessionCoordination {
    pub(crate) fn open(root: &Path) -> Result<Self> {
        let lock_root = root.join(LOCK_DIRECTORY);
        create_dir_all(&lock_root)?;
        let lock_path = lock_root.join(SESSION_LOCK);
        let _ = open_lock_file(&lock_path)?;
        let build_lease_root = lock_root.join(BUILD_LEASE_DIRECTORY);
        create_dir_all(&build_lease_root)?;
        sync_directory(&lock_root)?;
        Ok(Self {
            lock_path,
            build_lease_root,
        })
    }

    pub(crate) fn read(&self) -> Result<SessionGuard> {
        shared_guard(&self.lock_path)
    }

    pub(crate) fn write(&self) -> Result<SessionGuard> {
        let file = open_lock_file(&self.lock_path)?;
        file.lock()?;
        Ok(SessionGuard { _file: file })
    }

    pub(crate) fn reserve_build(&self, identifier: &IndexIdentifier) -> Result<BuildLease> {
        let setup_guard = self.read()?;
        let path = self
            .build_lease_root
            .join(format!("{}.lease", identifier_lease_key(identifier)));
        let mut file = open_lock_file(&path)?;
        match file.try_lock() {
            Ok(()) => {
                file.set_len(0)?;
                file.rewind()?;
                file.sync_all()?;
                sync_directory(&self.build_lease_root)?;
                Ok(BuildLease {
                    file,
                    setup_guard: Some(setup_guard),
                    session_lock_path: self.lock_path.clone(),
                    roots: Vec::new(),
                })
            }
            Err(std::fs::TryLockError::WouldBlock) => {
                Err(crate::Error::BuildAlreadyRunning(identifier.clone()))
            }
            Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }

    pub(crate) fn active_build_roots(&self) -> Result<HashSet<String>> {
        active_lease_roots(&self.build_lease_root, true)
    }
}

impl BuildLease {
    pub(crate) fn set_snapshot_root(&mut self, root: &str) -> Result<()> {
        if self.setup_guard.is_none() {
            return Err(crate::Error::InvalidArgument(
                "build snapshot root is already set".into(),
            ));
        }
        self.roots.push(validate_lease_root(root)?);
        self.write_roots()?;
        drop(self.setup_guard.take());
        Ok(())
    }

    pub(crate) fn add_snapshot_root(&mut self, root: &str) -> Result<()> {
        let _guard = shared_guard(&self.session_lock_path)?;
        let root = validate_lease_root(root)?;
        if !self.roots.contains(&root) {
            self.roots.push(root);
            self.write_roots()?;
        }
        Ok(())
    }

    fn write_roots(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.rewind()?;
        self.file.write_all(self.roots.join("\n").as_bytes())?;
        self.file.sync_all()?;
        Ok(())
    }
}

fn active_lease_roots(root: &Path, allow_empty: bool) -> Result<HashSet<String>> {
    let mut roots = HashSet::new();
    let mut removed_stale_lease = false;
    for entry in read_dir(root)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("lease") {
            continue;
        }
        let mut file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        match file.try_lock() {
            Ok(()) => {
                drop(file);
                match remove_file(path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
                removed_stale_lease = true;
            }
            Err(std::fs::TryLockError::WouldBlock) => {
                let mut lease_roots = String::new();
                file.read_to_string(&mut lease_roots)?;
                if lease_roots.is_empty() && allow_empty {
                    continue;
                }
                for root in lease_roots.lines() {
                    roots.insert(validate_lease_root(root)?);
                }
            }
            Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
        }
    }
    if removed_stale_lease {
        sync_directory(root)?;
    }
    Ok(roots)
}

fn validate_lease_root(root: &str) -> Result<String> {
    if root.is_empty() || root.contains('\n') || root.contains('\r') {
        return Err(crate::Error::InvalidArgument(
            "lease root must be a non-empty single-line URI".into(),
        ));
    }
    Ok(root.to_owned())
}

fn identifier_lease_key(identifier: &IndexIdentifier) -> Uuid {
    let mut bytes = Vec::new();
    for segment in identifier
        .namespace()
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(identifier.name()))
    {
        bytes.extend_from_slice(&(segment.len() as u64).to_be_bytes());
        bytes.extend_from_slice(segment.as_bytes());
    }
    Uuid::new_v5(&Uuid::NAMESPACE_OID, &bytes)
}

fn open_lock_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?)
}

fn shared_guard(path: &Path) -> Result<SessionGuard> {
    let file = open_lock_file(path)?;
    file.lock_shared()?;
    Ok(SessionGuard { _file: file })
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn shared_reader_blocks_an_independent_writer() {
        let temporary = TempDir::new().unwrap();
        let first = SessionCoordination::open(temporary.path()).unwrap();
        let second = SessionCoordination::open(temporary.path()).unwrap();
        let reader = first.read().unwrap();
        let (sender, receiver) = mpsc::channel();

        let writer = std::thread::spawn(move || {
            let _guard = second.write().unwrap();
            sender.send(()).unwrap();
        });

        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        drop(reader);
        receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        writer.join().unwrap();
    }

    #[test]
    fn build_reservation_is_exclusive_and_pins_its_snapshot_root() {
        let temporary = TempDir::new().unwrap();
        let first = SessionCoordination::open(temporary.path()).unwrap();
        let second = SessionCoordination::open(temporary.path()).unwrap();
        let root = "file:///tmp/parqdb/indexes/abc/1/";

        let identifier = IndexIdentifier::root("documents_embedding").unwrap();
        let other = IndexIdentifier::root("other_embedding").unwrap();
        let namespaced =
            IndexIdentifier::new(vec!["analytics".into()], "documents_embedding").unwrap();
        let mut lease = first.reserve_build(&identifier).unwrap();
        let writer_coordination = second.clone();
        let (sender, receiver) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let _guard = writer_coordination.write().unwrap();
            sender.send(()).unwrap();
        });
        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        assert!(second.active_build_roots().unwrap().is_empty());
        lease.set_snapshot_root(root).unwrap();
        receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        writer.join().unwrap();

        let extra_root = "file:///tmp/parqdb/indexes/shared/1/";
        let maintenance = second.write().unwrap();
        let (sender, receiver) = mpsc::channel();
        let updater = std::thread::spawn(move || {
            lease.add_snapshot_root(extra_root).unwrap();
            sender.send(()).unwrap();
            lease
        });
        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        drop(maintenance);
        receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        let lease = updater.join().unwrap();

        assert!(matches!(
            second.reserve_build(&identifier),
            Err(crate::Error::BuildAlreadyRunning(running))
                if running == identifier
        ));
        assert!(second.reserve_build(&other).is_ok());
        assert!(second.reserve_build(&namespaced).is_ok());
        assert_eq!(
            second.active_build_roots().unwrap(),
            HashSet::from([root.into(), extra_root.into()])
        );

        drop(lease);
        assert!(second.active_build_roots().unwrap().is_empty());
        assert!(second.reserve_build(&identifier).is_ok());
    }
}
