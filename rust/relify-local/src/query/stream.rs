use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::error::Result as DataFusionResult;
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};
use futures::Stream;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

use crate::runtime::QueryPermit;

/// A query result stream that owns cancellation and runtime admission state.
pub struct ManagedQueryStream {
    stream: Option<SendableRecordBatchStream>,
    schema: SchemaRef,
    cancellation: CancellationToken,
    cancelled: Pin<Box<WaitForCancellationFutureOwned>>,
    permit: Option<QueryPermit>,
}

impl ManagedQueryStream {
    pub(crate) fn new(stream: SendableRecordBatchStream, permit: QueryPermit) -> Self {
        let schema = stream.schema();
        let cancellation = CancellationToken::new();
        let cancelled = Box::pin(cancellation.clone().cancelled_owned());
        Self {
            stream: Some(stream),
            schema,
            cancellation,
            cancelled,
            permit: Some(permit),
        }
    }

    /// Cancels execution and releases the active-query slot.
    pub fn cancel(&mut self) {
        self.cancellation.cancel();
        self.finish();
    }

    /// Returns a token that may be used to cancel this query externally.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Returns the result schema before the first batch is consumed.
    #[must_use]
    pub fn schema_ref(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn finish(&mut self) {
        self.stream.take();
        self.permit.take();
    }
}

impl Stream for ManagedQueryStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.cancelled.as_mut().poll(context).is_ready() {
            this.finish();
            return Poll::Ready(None);
        }
        let Some(stream) = this.stream.as_mut() else {
            return Poll::Ready(None);
        };
        let result = stream.as_mut().poll_next(context);
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            this.finish();
        }
        result
    }
}

impl RecordBatchStream for ManagedQueryStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Drop for ManagedQueryStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
