// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::RowGroups;
use crate::arrow::arrow_reader::RowSelection;
use crate::column::page::{CachedPage, PageCache, PageIterator, PageReader};
use crate::errors::ParquetError;
use crate::file::metadata::{ParquetMetaData, RowGroupMetaData};
use crate::file::page_index::offset_index::OffsetIndexMetaData;
use crate::file::reader::{ChunkReader, Length, SerializedPageReader};
use bytes::{Buf, Bytes};
use std::ops::Range;
use std::sync::Arc;

/// Source data for one requested column-chunk range.
#[derive(Debug)]
pub(crate) enum ColumnChunkSource {
    /// Compressed bytes fetched from storage.
    Bytes(Bytes),
    /// Decompressed Pages that completely cover the requested range.
    Cached(Vec<Arc<CachedPage>>),
}

/// An in-memory collection of column chunks
#[derive(Debug)]
pub(crate) struct InMemoryRowGroup<'a> {
    pub(crate) offset_index: Option<&'a [OffsetIndexMetaData]>,
    /// Column chunks for this row group
    pub(crate) column_chunks: Vec<Option<Arc<ColumnChunkData>>>,
    pub(crate) row_count: usize,
    pub(crate) row_group_idx: usize,
    pub(crate) metadata: &'a ParquetMetaData,
    pub(crate) page_cache: Option<Arc<dyn PageCache>>,
}

/// What ranges to fetch for the columns in this row group
#[derive(Debug)]
pub(crate) struct FetchRanges {
    /// The byte ranges to fetch
    pub(crate) ranges: Vec<Range<u64>>,
    /// If `Some`, the start offsets of each page for each column chunk
    pub(crate) page_start_offsets: Option<Vec<Vec<u64>>>,
}

impl InMemoryRowGroup<'_> {
    /// Returns the byte ranges to fetch for the columns specified in
    /// `projection` and `selection`.
    ///
    /// `cache_mask` indicates which columns, if any, are being cached by
    /// [`RowGroupCache`](crate::arrow::array_reader::RowGroupCache).
    /// The `selection` for Cached columns is expanded to batch boundaries to simplify
    /// accounting for what data is cached.
    pub(crate) fn fetch_ranges(
        &self,
        projection: &ProjectionMask,
        selection: Option<&RowSelection>,
        batch_size: usize,
        cache_mask: Option<&ProjectionMask>,
    ) -> FetchRanges {
        let metadata = self.metadata.row_group(self.row_group_idx);
        if let Some((selection, offset_index)) = selection.zip(self.offset_index) {
            let expanded_selection =
                selection.expand_to_batch_boundaries(batch_size, self.row_count);

            // If we have a `RowSelection` and an `OffsetIndex` then only fetch
            // pages required for the `RowSelection`
            // Consider preallocating outer vec: https://github.com/apache/arrow-rs/issues/8667
            let mut page_start_offsets: Vec<Vec<u64>> = vec![];

            let ranges = self
                .column_chunks
                .iter()
                .zip(metadata.columns())
                .enumerate()
                .filter(|&(idx, (chunk, _chunk_meta))| {
                    chunk.is_none() && projection.leaf_included(idx)
                })
                .flat_map(|(idx, (_chunk, chunk_meta))| {
                    // If the first page does not start at the beginning of the column,
                    // then we need to also fetch a dictionary page.
                    let mut ranges: Vec<Range<u64>> = vec![];
                    let (start, _len) = chunk_meta.byte_range();
                    match offset_index[idx].page_locations.first() {
                        Some(first) if first.offset as u64 != start => {
                            ranges.push(start..first.offset as u64);
                        }
                        _ => (),
                    }

                    // Expand selection to batch boundaries if needed for caching
                    // (see doc comment for this function for details on `cache_mask`)
                    let use_expanded = cache_mask.map(|m| m.leaf_included(idx)).unwrap_or(false);
                    if use_expanded {
                        ranges.extend(
                            expanded_selection.scan_ranges(&offset_index[idx].page_locations),
                        );
                    } else {
                        ranges.extend(selection.scan_ranges(&offset_index[idx].page_locations));
                    }
                    page_start_offsets.push(ranges.iter().map(|range| range.start).collect());

                    ranges
                })
                .collect();
            FetchRanges {
                ranges,
                page_start_offsets: Some(page_start_offsets),
            }
        } else {
            let ranges = self
                .column_chunks
                .iter()
                .enumerate()
                .filter(|&(idx, chunk)| chunk.is_none() && projection.leaf_included(idx))
                .map(|(idx, _chunk)| {
                    let column = metadata.column(idx);
                    let (start, length) = column.byte_range();
                    start..(start + length)
                })
                .collect();
            FetchRanges {
                ranges,
                page_start_offsets: None,
            }
        }
    }

    /// Fills in `self.column_chunks` with the data fetched from `chunk_data`.
    ///
    /// This function **must** be called with the data from the ranges returned by
    /// `fetch_ranges` and the corresponding page_start_offsets, with the exact same and `selection`.
    pub(crate) fn fill_column_chunks<I>(
        &mut self,
        projection: &ProjectionMask,
        page_start_offsets: Option<Vec<Vec<u64>>>,
        chunk_data: I,
    ) where
        I: IntoIterator<Item = ColumnChunkSource>,
    {
        let mut chunk_data = chunk_data.into_iter();
        let metadata = self.metadata.row_group(self.row_group_idx);
        if let Some(page_start_offsets) = page_start_offsets {
            // If we have a `RowSelection` and an `OffsetIndex` then only fetch pages required for the
            // `RowSelection`
            let mut page_start_offsets = page_start_offsets.into_iter();

            for (idx, chunk) in self.column_chunks.iter_mut().enumerate() {
                if chunk.is_some() || !projection.leaf_included(idx) {
                    continue;
                }

                if let Some(offsets) = page_start_offsets.next() {
                    let mut chunks = Vec::with_capacity(offsets.len());
                    for _ in 0..offsets.len() {
                        chunks.push(chunk_data.next().unwrap());
                    }

                    let mut data = Vec::with_capacity(chunks.len());
                    let mut pinned_pages = Vec::new();
                    for (offset, source) in offsets.into_iter().zip(chunks) {
                        match source {
                            ColumnChunkSource::Bytes(bytes) => {
                                data.push((offset as usize, bytes));
                            }
                            ColumnChunkSource::Cached(pages) => pinned_pages.extend(pages),
                        }
                    }

                    *chunk = Some(Arc::new(ColumnChunkData::Sparse {
                        length: metadata.column(idx).byte_range().1 as usize,
                        data,
                        pinned_pages,
                    }))
                }
            }
        } else {
            for (idx, chunk) in self.column_chunks.iter_mut().enumerate() {
                if chunk.is_some() || !projection.leaf_included(idx) {
                    continue;
                }

                if let Some(source) = chunk_data.next() {
                    let (data, pinned_pages) = match source {
                        ColumnChunkSource::Bytes(data) => (data, Vec::new()),
                        ColumnChunkSource::Cached(pages) => (Bytes::new(), pages),
                    };
                    let (offset, length) = metadata.column(idx).byte_range();
                    *chunk = Some(Arc::new(ColumnChunkData::Dense {
                        offset: offset as usize,
                        length: length as usize,
                        data,
                        pinned_pages,
                    }));
                }
            }
        }
    }
}

impl RowGroups for InMemoryRowGroup<'_> {
    fn num_rows(&self) -> usize {
        self.row_count
    }

    /// Return chunks for column i
    fn column_chunks(&self, i: usize) -> crate::errors::Result<Box<dyn PageIterator>> {
        match &self.column_chunks[i] {
            None => Err(ParquetError::General(format!(
                "Invalid column index {i}, column was not fetched"
            ))),
            Some(data) => {
                let page_locations = self
                    .offset_index
                    // filter out empty offset indexes (old versions specified Some(vec![]) when no present)
                    .filter(|index| !index.is_empty())
                    .map(|index| index[i].page_locations.clone());
                let column_chunk_metadata = self.metadata.row_group(self.row_group_idx).column(i);
                let page_reader = SerializedPageReader::new(
                    data.clone(),
                    column_chunk_metadata,
                    self.row_count,
                    page_locations,
                )?;
                let page_reader = page_reader.add_crypto_context(
                    self.row_group_idx,
                    i,
                    self.metadata,
                    column_chunk_metadata,
                )?;

                let page_cache = PinnedPageCache::new(
                    data.pinned_pages(),
                    self.page_cache.as_ref().map(Arc::clone),
                );
                let page_reader = match page_cache {
                    Some(cache) => page_reader.with_page_cache(Arc::new(cache)),
                    None => page_reader,
                };

                let page_reader: Box<dyn PageReader> = Box::new(page_reader);

                Ok(Box::new(ColumnChunkIterator {
                    reader: Some(Ok(page_reader)),
                }))
            }
        }
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(std::iter::once(self.metadata.row_group(self.row_group_idx)))
    }

    fn metadata(&self) -> &ParquetMetaData {
        self.metadata
    }
}

/// An in-memory column chunk.
/// This allows us to hold either dense column chunks or sparse column chunks and easily
/// access them by offset.
#[derive(Clone, Debug)]
pub(crate) enum ColumnChunkData {
    /// Column chunk data representing only a subset of data pages.
    /// For example if a row selection (possibly caused by a filter in a query) causes us to read only
    /// a subset of the rows in the column.
    Sparse {
        /// Length of the full column chunk
        length: usize,
        /// Subset of data pages included in this sparse chunk.
        ///
        /// Each element is a tuple of (page offset within file, page data).
        /// Each entry is a complete page and the list is ordered by offset.
        data: Vec<(usize, Bytes)>,
        /// Cached Pages pinned for the lifetime of this column chunk.
        pinned_pages: Vec<Arc<CachedPage>>,
    },
    /// Full column chunk and the offset within the original file
    Dense {
        offset: usize,
        length: usize,
        data: Bytes,
        /// Cached Pages pinned for the lifetime of this column chunk.
        pinned_pages: Vec<Arc<CachedPage>>,
    },
}

impl ColumnChunkData {
    fn pinned_pages(&self) -> &[Arc<CachedPage>] {
        match self {
            Self::Sparse { pinned_pages, .. } | Self::Dense { pinned_pages, .. } => pinned_pages,
        }
    }

    /// Return the data for this column chunk at the given offset
    fn get(&self, start: u64) -> crate::errors::Result<Bytes> {
        match &self {
            ColumnChunkData::Sparse { data, .. } => data
                .binary_search_by_key(&start, |(offset, _)| *offset as u64)
                .map(|idx| data[idx].1.clone())
                .map_err(|_| {
                    ParquetError::General(format!(
                        "Invalid offset in sparse column chunk data: {start}, no matching page found.\
                         If you are using a `SelectionStrategyPolicy::Mask`, ensure that the OffsetIndex is provided when \
                         creating the InMemoryRowGroup."
                    ))
                }),
            ColumnChunkData::Dense { offset, data, .. } => {
                let start = start as usize - *offset;
                Ok(data.slice(start..))
            }
        }
    }
}

impl Length for ColumnChunkData {
    /// Return the total length of the full column chunk
    fn len(&self) -> u64 {
        match &self {
            ColumnChunkData::Sparse { length, .. } => *length as u64,
            ColumnChunkData::Dense { length, .. } => *length as u64,
        }
    }
}

#[derive(Debug)]
struct PinnedPageCache {
    pages: Vec<Arc<CachedPage>>,
    fallback: Option<Arc<dyn PageCache>>,
}

impl PinnedPageCache {
    fn new(pages: &[Arc<CachedPage>], fallback: Option<Arc<dyn PageCache>>) -> Option<Self> {
        (!pages.is_empty() || fallback.is_some()).then(|| Self {
            pages: pages.to_vec(),
            fallback,
        })
    }
}

impl PageCache for PinnedPageCache {
    fn get(&self, page_offset: u64) -> Option<Arc<CachedPage>> {
        self.pages
            .binary_search_by_key(&page_offset, |page| page.compressed_range().start)
            .ok()
            .map(|index| Arc::clone(&self.pages[index]))
            .or_else(|| {
                self.fallback
                    .as_ref()
                    .and_then(|cache| cache.get(page_offset))
            })
    }

    fn insert(&self, page: CachedPage) -> Arc<CachedPage> {
        match &self.fallback {
            Some(cache) => cache.insert(page),
            None => Arc::new(page),
        }
    }
}

impl ChunkReader for ColumnChunkData {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> crate::errors::Result<Self::T> {
        Ok(self.get(start)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> crate::errors::Result<Bytes> {
        Ok(self.get(start)?.slice(..length))
    }
}

/// Implements [`PageIterator`] for a single column chunk, yielding a single [`PageReader`]
struct ColumnChunkIterator {
    reader: Option<crate::errors::Result<Box<dyn PageReader>>>,
}

impl Iterator for ColumnChunkIterator {
    type Item = crate::errors::Result<Box<dyn PageReader>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.take()
    }
}

impl PageIterator for ColumnChunkIterator {}
