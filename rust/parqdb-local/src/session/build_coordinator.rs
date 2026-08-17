use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use tokio::sync::{Semaphore, watch};

use parqdb_catalog::IndexIdentifier;

use crate::progress::LocalBuildProgressSnapshot;
use crate::{BuildFailureKind, Error, LocalBuildProgress, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BuildKey {
    source_identity: String,
    index: IndexIdentifier,
}

impl BuildKey {
    pub(super) fn new(source_identity: String, index: IndexIdentifier) -> Self {
        Self {
            source_identity,
            index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActiveBuildState {
    Pending,
    Building,
    Complete,
    Failed,
}

#[derive(Debug, Clone)]
pub(super) struct ActiveBuildSnapshot {
    pub(super) state: ActiveBuildState,
    pub(super) base_snapshot_id: Option<i64>,
    pub(super) progress: LocalBuildProgressSnapshot,
    pub(super) error: Option<String>,
    pub(super) failure_kind: Option<BuildFailureKind>,
}

#[derive(Debug, Clone)]
struct RetainedFailure {
    kind: BuildFailureKind,
    message: String,
}

impl RetainedFailure {
    fn from_error(error: &Error) -> Self {
        let Error::BuildFailed { kind, message } = error.retained_build_failure() else {
            unreachable!("retained_build_failure always returns BuildFailed")
        };
        Self { kind, message }
    }

    fn into_error(self) -> Error {
        Error::BuildFailed {
            kind: self.kind,
            message: self.message,
        }
    }
}

#[derive(Debug, Clone)]
enum RecordState {
    Pending,
    Building,
    Complete,
    Failed(RetainedFailure),
}

#[derive(Debug)]
struct BuildRecord {
    state: watch::Sender<RecordState>,
    base_snapshot_id: Option<i64>,
    progress: LocalBuildProgress,
}

impl BuildRecord {
    fn new(base_snapshot_id: Option<i64>, progress: LocalBuildProgress) -> Self {
        let (state, _) = watch::channel(RecordState::Pending);
        Self {
            state,
            base_snapshot_id,
            progress,
        }
    }

    fn snapshot(&self) -> ActiveBuildSnapshot {
        let state = self.state.borrow();
        let (state, error, failure_kind) = match &*state {
            RecordState::Pending => (ActiveBuildState::Pending, None, None),
            RecordState::Building => (ActiveBuildState::Building, None, None),
            RecordState::Complete => (ActiveBuildState::Complete, None, None),
            RecordState::Failed(error) => (
                ActiveBuildState::Failed,
                Some(error.message.clone()),
                Some(error.kind),
            ),
        };
        ActiveBuildSnapshot {
            state,
            base_snapshot_id: self.base_snapshot_id,
            progress: self.progress.snapshot(),
            error,
            failure_kind,
        }
    }

    fn mark_building(&self) {
        self.state.send_replace(RecordState::Building);
    }

    fn finish(&self, result: &Result<()>) {
        self.state.send_replace(match result {
            Ok(()) => RecordState::Complete,
            Err(error) => RecordState::Failed(RetainedFailure::from_error(error)),
        });
    }

    async fn wait(&self) -> Result<()> {
        let mut state = self.state.subscribe();
        loop {
            let outcome = {
                let state = state.borrow_and_update();
                match &*state {
                    RecordState::Pending | RecordState::Building => None,
                    RecordState::Complete => Some(Ok(())),
                    RecordState::Failed(error) => Some(Err(error.clone().into_error())),
                }
            };
            if let Some(outcome) = outcome {
                return outcome;
            }
            state
                .changed()
                .await
                .expect("build record owns the watch sender");
        }
    }
}

#[derive(Clone)]
pub(super) struct BuildCoordinator {
    records: Arc<Mutex<HashMap<BuildKey, Arc<BuildRecord>>>>,
    slots: Arc<Semaphore>,
    dop: Option<usize>,
}

impl BuildCoordinator {
    pub(super) fn new(dop: Option<usize>) -> Self {
        Self {
            records: Arc::new(Mutex::new(HashMap::new())),
            slots: Arc::new(Semaphore::new(1)),
            dop,
        }
    }

    pub(super) fn dop(&self) -> Option<usize> {
        self.dop
    }

    pub(super) fn submit<F>(
        &self,
        key: BuildKey,
        base_snapshot_id: Option<i64>,
        progress: LocalBuildProgress,
        task: F,
    ) -> Result<()>
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
            Error::InvalidArgument(format!("index builds require a Tokio runtime: {error}"))
        })?;
        let record = Arc::new(BuildRecord::new(base_snapshot_id, progress));
        {
            let mut records = self.records.lock().expect("build records mutex poisoned");
            let already_running = records.get(&key).is_some_and(|active| {
                matches!(
                    active.snapshot().state,
                    ActiveBuildState::Pending | ActiveBuildState::Building
                )
            });
            if already_running {
                return Err(Error::BuildAlreadyRunning(key.index.clone()));
            }
            records.insert(key.clone(), Arc::clone(&record));
        }

        let coordinator = self.clone();
        runtime.spawn(async move {
            let permit = coordinator
                .slots
                .clone()
                .acquire_owned()
                .await
                .expect("build semaphore is never closed");
            record.mark_building();
            let result = AssertUnwindSafe(task)
                .catch_unwind()
                .await
                .unwrap_or_else(|panic| {
                    Err(Error::BuildFailed {
                        kind: BuildFailureKind::Backend,
                        message: format!("index build task panicked: {}", panic_message(&panic)),
                    })
                });
            drop(permit);
            record.finish(&result);
            if result.is_ok() {
                coordinator.remove_if_same(&key, &record);
            }
        });
        Ok(())
    }

    pub(super) fn snapshot(&self, key: &BuildKey) -> Option<ActiveBuildSnapshot> {
        self.records
            .lock()
            .expect("build records mutex poisoned")
            .get(key)
            .map(|record| record.snapshot())
    }

    pub(super) async fn wait(&self, key: &BuildKey) -> Option<Result<()>> {
        let record = self
            .records
            .lock()
            .expect("build records mutex poisoned")
            .get(key)
            .cloned();
        match record {
            Some(record) => Some(record.wait().await),
            None => None,
        }
    }

    pub(super) fn forget_failure(&self, key: &BuildKey) {
        let mut records = self.records.lock().expect("build records mutex poisoned");
        if records
            .get(key)
            .is_some_and(|record| record.snapshot().state == ActiveBuildState::Failed)
        {
            records.remove(key);
        }
    }

    fn remove_if_same(&self, key: &BuildKey, record: &Arc<BuildRecord>) {
        let mut records = self.records.lock().expect("build records mutex poisoned");
        if records
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, record))
        {
            records.remove(key);
        }
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{Notify, oneshot};
    use tokio::time::{Duration, timeout};

    use super::*;

    fn key(source: &str, index: &str) -> BuildKey {
        BuildKey::new(
            source.to_owned(),
            IndexIdentifier::root(index).expect("valid test index"),
        )
    }

    async fn wait_for_state(
        coordinator: &BuildCoordinator,
        key: &BuildKey,
        expected: ActiveBuildState,
    ) {
        timeout(Duration::from_secs(1), async {
            loop {
                if coordinator.snapshot(key).map(|state| state.state) == Some(expected) {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("build did not reach expected state");
    }

    #[tokio::test]
    async fn accepted_builds_share_one_execution_slot() {
        let coordinator = BuildCoordinator::new(Some(3));
        assert_eq!(coordinator.dop(), Some(3));

        let first_key = key("source-a", "first");
        let first_release = Arc::new(Notify::new());
        let (first_started_tx, first_started_rx) = oneshot::channel();
        let task_release = Arc::clone(&first_release);
        coordinator
            .submit(
                first_key.clone(),
                None,
                LocalBuildProgress::default(),
                async move {
                    first_started_tx.send(()).expect("receiver remains alive");
                    task_release.notified().await;
                    Ok(())
                },
            )
            .unwrap();
        first_started_rx.await.unwrap();

        let second_key = key("source-b", "second");
        let second_release = Arc::new(Notify::new());
        let (second_started_tx, second_started_rx) = oneshot::channel();
        let task_release = Arc::clone(&second_release);
        coordinator
            .submit(
                second_key.clone(),
                None,
                LocalBuildProgress::default(),
                async move {
                    second_started_tx.send(()).expect("receiver remains alive");
                    task_release.notified().await;
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            coordinator.snapshot(&first_key).unwrap().state,
            ActiveBuildState::Building
        );
        assert_eq!(
            coordinator.snapshot(&second_key).unwrap().state,
            ActiveBuildState::Pending
        );

        first_release.notify_one();
        timeout(Duration::from_secs(1), second_started_rx)
            .await
            .expect("second build did not start")
            .unwrap();
        assert!(coordinator.snapshot(&first_key).is_none());
        assert_eq!(
            coordinator.snapshot(&second_key).unwrap().state,
            ActiveBuildState::Building
        );

        second_release.notify_one();
        timeout(Duration::from_secs(1), async {
            while coordinator.snapshot(&second_key).is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("successful build record was not released");
    }

    #[tokio::test]
    async fn duplicate_active_key_is_rejected() {
        let coordinator = BuildCoordinator::new(None);
        let build_key = key("source", "embedding");
        let release = Arc::new(Notify::new());
        let task_release = Arc::clone(&release);
        coordinator
            .submit(
                build_key.clone(),
                None,
                LocalBuildProgress::default(),
                async move {
                    task_release.notified().await;
                    Ok(())
                },
            )
            .unwrap();
        wait_for_state(&coordinator, &build_key, ActiveBuildState::Building).await;

        let error = coordinator
            .submit(
                build_key.clone(),
                None,
                LocalBuildProgress::default(),
                async { Ok(()) },
            )
            .unwrap_err();
        assert!(matches!(error, Error::BuildAlreadyRunning(_)));

        release.notify_one();
    }

    #[tokio::test]
    async fn cancelling_a_waiter_does_not_cancel_the_build() {
        let coordinator = BuildCoordinator::new(None);
        let build_key = key("source", "embedding");
        let release = Arc::new(Notify::new());
        let task_release = Arc::clone(&release);
        coordinator
            .submit(
                build_key.clone(),
                None,
                LocalBuildProgress::default(),
                async move {
                    task_release.notified().await;
                    Ok(())
                },
            )
            .unwrap();
        wait_for_state(&coordinator, &build_key, ActiveBuildState::Building).await;

        let waiting_coordinator = coordinator.clone();
        let waiting_key = build_key.clone();
        let waiter = tokio::spawn(async move { waiting_coordinator.wait(&waiting_key).await });
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());

        release.notify_one();
        timeout(Duration::from_secs(1), async {
            while coordinator.snapshot(&build_key).is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("build was cancelled with its waiter");
    }

    #[tokio::test]
    async fn failed_build_is_retained_and_can_be_retried() {
        let coordinator = BuildCoordinator::new(None);
        let build_key = key("source", "embedding");
        coordinator
            .submit(
                build_key.clone(),
                None,
                LocalBuildProgress::default(),
                async { Err(Error::InvalidSchema("missing vector column".into())) },
            )
            .unwrap();

        let error = coordinator
            .wait(&build_key)
            .await
            .expect("failed build remains observable")
            .unwrap_err();
        assert!(matches!(
            error,
            Error::BuildFailed {
                kind: BuildFailureKind::InvalidSchema,
                ..
            }
        ));
        let snapshot = coordinator.snapshot(&build_key).unwrap();
        assert_eq!(snapshot.state, ActiveBuildState::Failed);
        assert_eq!(snapshot.failure_kind, Some(BuildFailureKind::InvalidSchema));
        assert_eq!(
            snapshot.error.as_deref(),
            Some("invalid source schema: missing vector column")
        );

        coordinator
            .submit(
                build_key.clone(),
                None,
                LocalBuildProgress::default(),
                async { Ok(()) },
            )
            .unwrap();
        coordinator
            .wait(&build_key)
            .await
            .expect("retry is registered")
            .unwrap();
    }

    #[tokio::test]
    async fn panicking_build_wakes_waiters_with_a_backend_failure() {
        let coordinator = BuildCoordinator::new(None);
        let build_key = key("source", "embedding");
        coordinator
            .submit(
                build_key.clone(),
                None,
                LocalBuildProgress::default(),
                async { panic!("builder panic") },
            )
            .unwrap();

        let error = coordinator
            .wait(&build_key)
            .await
            .expect("panicking build remains observable")
            .unwrap_err();
        assert!(matches!(
            error,
            Error::BuildFailed {
                kind: BuildFailureKind::Backend,
                ref message,
            } if message.contains("builder panic")
        ));
    }
}
