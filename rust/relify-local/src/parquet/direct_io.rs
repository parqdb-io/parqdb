//! Linux Direct I/O reader for immutable local index Parquet files.

use std::fs::{File, OpenOptions};
use std::io;
use std::ops::Range;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::execution::cache::cache_manager::FileMetadataCache;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_datasource_parquet::metadata::DFParquetMetadata;
use datafusion_datasource_parquet::{ParquetFileMetrics, ParquetFileReaderFactory};
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;

const DIRECT_IO_ALIGNMENT: usize = 4 * 1024;

fn metric_range_len(range: &Range<u64>) -> usize {
    usize::try_from(range.end.saturating_sub(range.start)).unwrap_or(usize::MAX)
}

/// Creates readers that bypass the OS Page Cache for Parquet data ranges.
#[derive(Debug)]
pub(crate) struct DirectIoParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    metadata_cache: Arc<dyn FileMetadataCache>,
    filesystem: LocalFileSystem,
}

impl DirectIoParquetFileReaderFactory {
    pub(crate) fn new(
        store: Arc<dyn ObjectStore>,
        metadata_cache: Arc<dyn FileMetadataCache>,
    ) -> Self {
        Self {
            store,
            metadata_cache,
            filesystem: LocalFileSystem::new(),
        }
    }
}

impl ParquetFileReaderFactory for DirectIoParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion::common::Result<Box<dyn AsyncFileReader + Send>> {
        let path = self
            .filesystem
            .path_to_filesystem(&partitioned_file.object_meta.location)
            .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))?;
        let file_metrics = ParquetFileMetrics::new(
            partition_index,
            partitioned_file.object_meta.location.as_ref(),
            metrics,
        );
        Ok(Box::new(DirectIoParquetFileReader {
            file_metrics,
            store: Arc::clone(&self.store),
            partitioned_file,
            metadata_cache: Arc::clone(&self.metadata_cache),
            metadata_size_hint,
            file: Arc::new(DirectFile::new(path)),
        }))
    }
}

struct DirectIoParquetFileReader {
    file_metrics: ParquetFileMetrics,
    store: Arc<dyn ObjectStore>,
    partitioned_file: PartitionedFile,
    metadata_cache: Arc<dyn FileMetadataCache>,
    metadata_size_hint: Option<usize>,
    file: Arc<DirectFile>,
}

impl AsyncFileReader for DirectIoParquetFileReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.file_metrics
            .bytes_scanned
            .add(metric_range_len(&range));
        let file = Arc::clone(&self.file);
        async move {
            tokio::task::spawn_blocking(move || file.read_range(range))
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))?
                .map_err(ParquetError::from)
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let total = ranges.iter().fold(0_usize, |total, range| {
            total.saturating_add(metric_range_len(range))
        });
        self.file_metrics.bytes_scanned.add(total);
        let file = Arc::clone(&self.file);
        async move {
            tokio::task::spawn_blocking(move || {
                ranges
                    .into_iter()
                    .map(|range| file.read_range(range))
                    .collect::<io::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| ParquetError::External(Box::new(error)))?
            .map_err(ParquetError::from)
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let object_meta = self.partitioned_file.object_meta.clone();
        let metadata_cache = Arc::clone(&self.metadata_cache);
        async move {
            DFParquetMetadata::new(self.store.as_ref(), &object_meta)
                .with_file_metadata_cache(Some(metadata_cache))
                .with_metadata_size_hint(self.metadata_size_hint)
                .fetch_metadata()
                .await
                .map_err(|error| {
                    ParquetError::General(format!(
                        "failed to fetch metadata for file {}: {error}",
                        object_meta.location
                    ))
                })
        }
        .boxed()
    }
}

impl Drop for DirectIoParquetFileReader {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        self.file_metrics.scan_efficiency_ratio.set_total(
            usize::try_from(self.partitioned_file.object_meta.size).unwrap_or(usize::MAX),
        );
    }
}

#[derive(Debug)]
struct DirectFile {
    path: PathBuf,
    file: Mutex<Option<Arc<File>>>,
}

impl DirectFile {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            file: Mutex::new(None),
        }
    }

    fn open(&self) -> io::Result<Arc<File>> {
        let mut current = self
            .file
            .lock()
            .map_err(|_| io::Error::other("Direct I/O file lock is poisoned"))?;
        if let Some(file) = current.as_ref() {
            return Ok(Arc::clone(file));
        }
        let file = Arc::new(
            OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&self.path)?,
        );
        current.replace(Arc::clone(&file));
        Ok(file)
    }

    fn read_range(&self, range: Range<u64>) -> io::Result<Bytes> {
        if range.start > range.end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Direct I/O range start exceeds end",
            ));
        }
        if range.is_empty() {
            return Ok(Bytes::new());
        }

        let alignment = DIRECT_IO_ALIGNMENT as u64;
        let aligned_start = range.start / alignment * alignment;
        let aligned_end = range
            .end
            .checked_add(alignment - 1)
            .ok_or_else(|| io::Error::other("Direct I/O range overflow"))?
            / alignment
            * alignment;
        let aligned_len = usize::try_from(aligned_end - aligned_start)
            .map_err(|_| io::Error::other("Direct I/O range is too large"))?;
        let requested_start = usize::try_from(range.start - aligned_start)
            .map_err(|_| io::Error::other("Direct I/O range is too large"))?;
        let requested_len = usize::try_from(range.end - range.start)
            .map_err(|_| io::Error::other("Direct I/O range is too large"))?;

        let mut buffer = AlignedBuffer::new(aligned_len, DIRECT_IO_ALIGNMENT)?;
        let read = self.open()?.read_at(buffer.as_mut_slice(), aligned_start)?;
        let requested_end = requested_start
            .checked_add(requested_len)
            .ok_or_else(|| io::Error::other("Direct I/O range is too large"))?;
        if read < requested_end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Direct I/O read returned {read} bytes, but {requested_end} were required"),
            ));
        }
        buffer.set_len(read);
        Ok(Bytes::from_owner(buffer).slice(requested_start..requested_end))
    }
}

#[derive(Debug)]
struct AlignedBuffer {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize, alignment: usize) -> io::Result<Self> {
        let allocation = len
            .checked_add(alignment - 1)
            .ok_or_else(|| io::Error::other("Direct I/O buffer size overflow"))?;
        let storage = vec![0; allocation];
        let address = storage.as_ptr() as usize;
        let start = (alignment - address % alignment) % alignment;
        Ok(Self {
            storage,
            start,
            len,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.len]
    }

    fn set_len(&mut self, len: usize) {
        self.len = len;
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn direct_reader_returns_exact_unaligned_ranges() {
        let mut temporary = NamedTempFile::new().unwrap();
        let contents = (0_u8..=u8::MAX).cycle().take(10_000).collect::<Vec<_>>();
        temporary.write_all(&contents).unwrap();
        temporary.as_file().sync_all().unwrap();
        let file = DirectFile::new(temporary.path().to_owned());

        assert_eq!(file.read_range(13..7_777).unwrap(), contents[13..7_777]);
        assert_eq!(file.read_range(9_900..10_000).unwrap(), contents[9_900..]);
    }
}
