use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::timeout;

use crate::{Error, Result};

/// Bounds for active and waiting queries in one runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryAdmissionOptions {
    /// Maximum number of queries admitted for planning or execution.
    pub max_active: usize,
    /// Maximum number of queries waiting for an active slot.
    pub max_queued: usize,
    /// Maximum time a query may wait for an active slot.
    pub queue_timeout: Duration,
}

impl Default for QueryAdmissionOptions {
    fn default() -> Self {
        Self {
            max_active: 1,
            max_queued: 64,
            queue_timeout: Duration::from_secs(30),
        }
    }
}

impl QueryAdmissionOptions {
    pub(crate) fn validate(self) -> Result<Self> {
        if self.max_active == 0 {
            return Err(Error::InvalidArgument(
                "query admission requires at least one active slot".into(),
            ));
        }
        Ok(self)
    }
}

/// Current query-admission occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryAdmissionStats {
    /// Queries that hold an active slot.
    pub active: usize,
    /// Queries waiting for an active slot.
    pub queued: usize,
}

#[derive(Debug)]
pub(super) struct QueryAdmission {
    slots: Arc<Semaphore>,
    max_active: usize,
    queued: AtomicUsize,
    max_queued: usize,
    queue_timeout: Duration,
}

impl QueryAdmission {
    pub(super) fn new(options: QueryAdmissionOptions) -> Result<Self> {
        let options = options.validate()?;
        Ok(Self {
            slots: Arc::new(Semaphore::new(options.max_active)),
            max_active: options.max_active,
            queued: AtomicUsize::new(0),
            max_queued: options.max_queued,
            queue_timeout: options.queue_timeout,
        })
    }

    pub(super) async fn acquire(&self) -> Result<QueryPermit> {
        if let Ok(permit) = Arc::clone(&self.slots).try_acquire_owned() {
            return Ok(QueryPermit { _permit: permit });
        }

        let queue_guard = self.enter_queue()?;
        let permit = timeout(self.queue_timeout, Arc::clone(&self.slots).acquire_owned())
            .await
            .map_err(|_| Error::QueryQueueTimeout(self.queue_timeout))?
            .expect("query-admission semaphore is never closed");
        drop(queue_guard);
        Ok(QueryPermit { _permit: permit })
    }

    pub(super) fn stats(&self) -> QueryAdmissionStats {
        QueryAdmissionStats {
            active: self
                .max_active
                .saturating_sub(self.slots.available_permits()),
            queued: self.queued.load(Ordering::Acquire),
        }
    }

    fn enter_queue(&self) -> Result<QueueGuard<'_>> {
        let mut queued = self.queued.load(Ordering::Acquire);
        loop {
            if queued >= self.max_queued {
                return Err(Error::QueryQueueFull(self.max_queued));
            }
            match self.queued.compare_exchange_weak(
                queued,
                queued + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(QueueGuard(&self.queued)),
                Err(current) => queued = current,
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct QueryPermit {
    _permit: OwnedSemaphorePermit,
}

struct QueueGuard<'a>(&'a AtomicUsize);

impl Drop for QueueGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::mpsc;

    use super::*;

    fn admission(max_active: usize, max_queued: usize) -> Arc<QueryAdmission> {
        Arc::new(
            QueryAdmission::new(QueryAdmissionOptions {
                max_active,
                max_queued,
                queue_timeout: Duration::from_millis(100),
            })
            .unwrap(),
        )
    }

    async fn wait_for_queued(admission: &QueryAdmission, expected: usize) {
        timeout(Duration::from_secs(1), async {
            while admission.stats().queued != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn rejects_queries_beyond_the_bounded_queue() {
        let admission = admission(1, 1);
        let active = admission.acquire().await.unwrap();
        let waiting_admission = Arc::clone(&admission);
        let waiting = tokio::spawn(async move { waiting_admission.acquire().await });
        wait_for_queued(&admission, 1).await;

        assert_eq!(
            admission.acquire().await.unwrap_err().to_string(),
            "query queue is full (capacity 1)"
        );
        assert_eq!(
            admission.stats(),
            QueryAdmissionStats {
                active: 1,
                queued: 1,
            }
        );

        drop(active);
        drop(waiting.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn waiting_queries_are_admitted_in_fifo_order() {
        let admission = admission(1, 2);
        let active = admission.acquire().await.unwrap();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut tasks = Vec::new();
        for id in 0..2 {
            let task_admission = Arc::clone(&admission);
            let sender = sender.clone();
            tasks.push(tokio::spawn(async move {
                let permit = task_admission.acquire().await.unwrap();
                sender.send(id).unwrap();
                permit
            }));
            wait_for_queued(&admission, id + 1).await;
        }

        drop(active);
        assert_eq!(receiver.recv().await, Some(0));
        drop(tasks.remove(0).await.unwrap());
        assert_eq!(receiver.recv().await, Some(1));
        drop(tasks.remove(0).await.unwrap());
    }

    #[tokio::test]
    async fn timeout_and_cancellation_remove_queue_entries() {
        let timeout_admission = Arc::new(
            QueryAdmission::new(QueryAdmissionOptions {
                max_active: 1,
                max_queued: 2,
                queue_timeout: Duration::from_millis(10),
            })
            .unwrap(),
        );
        let active = timeout_admission.acquire().await.unwrap();
        assert!(matches!(
            timeout_admission.acquire().await,
            Err(Error::QueryQueueTimeout(_))
        ));
        assert_eq!(timeout_admission.stats().queued, 0);

        drop(active);
        let admission = admission(1, 1);
        let active = admission.acquire().await.unwrap();

        let waiting_admission = Arc::clone(&admission);
        let waiting = tokio::spawn(async move { waiting_admission.acquire().await });
        wait_for_queued(&admission, 1).await;
        assert_eq!(admission.stats().queued, 1);
        waiting.abort();
        assert!(waiting.await.unwrap_err().is_cancelled());
        assert_eq!(admission.stats().queued, 0);
        drop(active);
    }
}
