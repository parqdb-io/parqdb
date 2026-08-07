use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, RowSelection};
use parquet::arrow::{ArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use serde::Serialize;

const PAGE_SIZES_KIB: [usize; 4] = [16, 64, 256, 1024];

#[derive(Debug)]
struct Config {
    rows: usize,
    text_bytes: usize,
    queries: usize,
    warmups: usize,
    top_k: usize,
    output_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rows: 50_000,
            text_bytes: 2_048,
            queries: 100,
            warmups: 10,
            top_k: 10,
            output_dir: PathBuf::from("/tmp/relify-parquet-page-bench"),
        }
    }
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--rows" => config.rows = parse_usize(&flag, &value)?,
                "--text-bytes" => config.text_bytes = parse_usize(&flag, &value)?,
                "--queries" => config.queries = parse_usize(&flag, &value)?,
                "--warmups" => config.warmups = parse_usize(&flag, &value)?,
                "--top-k" => config.top_k = parse_usize(&flag, &value)?,
                "--output-dir" => config.output_dir = PathBuf::from(value),
                _ => return Err(format!("unknown argument: {flag}")),
            }
        }
        if config.rows == 0
            || config.text_bytes < 128
            || config.queries == 0
            || config.top_k == 0
            || config.top_k > config.rows
        {
            return Err("rows, queries, and top-k must be positive; text-bytes must be at least 128; top-k must not exceed rows".into());
        }
        Ok(config)
    }
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|error| format!("invalid value for {flag}: {error}"))
}

#[derive(Default)]
struct IoCounters {
    api_calls: u64,
    ranges: u64,
    bytes: u64,
}

#[derive(Clone)]
struct CountingReader {
    path: Arc<PathBuf>,
    metadata: Arc<ParquetMetaData>,
    counters: Arc<Mutex<IoCounters>>,
}

impl CountingReader {
    fn new(path: &Path, metadata: Arc<ParquetMetaData>) -> Self {
        Self {
            path: Arc::new(path.to_owned()),
            metadata,
            counters: Arc::new(Mutex::new(IoCounters::default())),
        }
    }

    fn snapshot(&self) -> IoSample {
        let counters = self.counters.lock().expect("I/O counters poisoned");
        IoSample {
            api_calls: counters.api_calls,
            ranges: counters.ranges,
            bytes: counters.bytes,
        }
    }
}

fn read_range(path: &Path, range: &Range<u64>) -> ParquetResult<Bytes> {
    let length = usize::try_from(range.end - range.start)
        .map_err(|_| ParquetError::General("range is too large".into()))?;
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(range.start))?;
    let mut buffer = vec![0_u8; length];
    file.read_exact(&mut buffer)?;
    Ok(Bytes::from(buffer))
}

impl parquet::arrow::async_reader::AsyncFileReader for CountingReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        let path = Arc::clone(&self.path);
        let counters = Arc::clone(&self.counters);
        async move {
            {
                let mut counters = counters.lock().expect("I/O counters poisoned");
                counters.api_calls += 1;
                counters.ranges += 1;
                counters.bytes += range.end - range.start;
            }
            read_range(&path, &range)
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        let path = Arc::clone(&self.path);
        let counters = Arc::clone(&self.counters);
        async move {
            {
                let mut counters = counters.lock().expect("I/O counters poisoned");
                counters.api_calls += 1;
                counters.ranges += ranges.len() as u64;
                counters.bytes += ranges
                    .iter()
                    .map(|range| range.end - range.start)
                    .sum::<u64>();
            }
            ranges
                .iter()
                .map(|range| read_range(&path, range))
                .collect()
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        let metadata = Arc::clone(&self.metadata);
        async move { Ok(metadata) }.boxed()
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct IoSample {
    api_calls: u64,
    ranges: u64,
    bytes: u64,
}

#[derive(Debug, Serialize)]
struct FileResult {
    page_size_kib: usize,
    file_bytes: u64,
    file_growth_percent: f64,
    text_pages: usize,
    mean_compressed_text_page_bytes: f64,
    build_seconds: f64,
    latency_ms_p50: f64,
    latency_ms_p95: f64,
    latency_ms_mean: f64,
    range_requests_mean: f64,
    io_api_calls_mean: f64,
    bytes_read_mean: f64,
    bytes_read_p95: u64,
}

#[derive(Debug, Serialize)]
struct Output {
    rows: usize,
    text_bytes_per_row: usize,
    top_k: usize,
    measured_queries: usize,
    warmup_queries: usize,
    metadata_cached: bool,
    row_groups_per_file: usize,
    compression: &'static str,
    baseline_page_size_kib: usize,
    results: Vec<FileResult>,
}

fn generate_text(doc_id: usize, target_bytes: usize) -> String {
    let prefix = format!(
        "{{\"trace_id\":\"{doc_id:016x}\",\"level\":\"INFO\",\"service\":\"retrieval\",\"message\":\""
    );
    let suffix = "\",\"status\":\"ok\"}";
    let payload_len = target_bytes.saturating_sub(prefix.len() + suffix.len());
    let mut state = (doc_id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut text = String::with_capacity(target_bytes);
    text.push_str(&prefix);
    for index in 0..payload_len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let byte = b'a' + ((state.wrapping_add(index as u64) % 26) as u8);
        text.push(char::from(byte));
    }
    text.push_str(suffix);
    text
}

fn make_batch(config: &Config) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let ids = Int64Array::from_iter_values(0..config.rows as i64);
    let texts = StringArray::from_iter_values(
        (0..config.rows).map(|doc_id| generate_text(doc_id, config.text_bytes)),
    );
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(ids) as ArrayRef, Arc::new(texts) as ArrayRef],
    )?)
}

fn write_file(
    path: &Path,
    batch: &RecordBatch,
    page_size_kib: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let text_path = ColumnPath::from("text");
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_column_statistics_enabled(text_path.clone(), EnabledStatistics::Page)
        .set_column_dictionary_enabled(text_path.clone(), false)
        .set_column_data_page_size_limit(text_path, page_size_kib * 1024)
        .set_write_batch_size(8)
        .set_max_row_group_row_count(Some(batch.num_rows()))
        .build();
    let started = Instant::now();
    let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), Some(properties))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(started.elapsed().as_secs_f64())
}

fn make_queries(config: &Config) -> Vec<Vec<usize>> {
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    (0..config.queries + config.warmups)
        .map(|_| {
            let mut ids = BTreeSet::new();
            while ids.len() < config.top_k {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ids.insert((state as usize) % config.rows);
            }
            ids.into_iter().collect()
        })
        .collect()
}

async fn run_query(
    path: &Path,
    metadata: &ArrowReaderMetadata,
    row_ids: &[usize],
    total_rows: usize,
) -> Result<(f64, IoSample), Box<dyn std::error::Error>> {
    let selection = RowSelection::from_consecutive_ranges(
        row_ids.iter().copied().map(|row| row..row + 1),
        total_rows,
    );
    let projection = ProjectionMask::leaves(metadata.parquet_schema(), [1]);
    let reader = CountingReader::new(path, Arc::clone(metadata.metadata()));
    let counters = reader.clone();
    let started = Instant::now();
    let stream = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata.clone())
        .with_projection(projection)
        .with_row_selection(selection)
        .with_batch_size(1024)
        .build()?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let actual_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if actual_rows != row_ids.len() {
        return Err(format!("expected {} rows, read {actual_rows}", row_ids.len()).into());
    }
    let mut actual = Vec::with_capacity(actual_rows);
    for batch in &batches {
        let texts = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        actual.extend((0..texts.len()).map(|index| texts.value(index).to_owned()));
    }
    let expected = row_ids
        .iter()
        .map(|row| generate_text(*row, actual[0].len()))
        .collect::<Vec<_>>();
    if actual != expected {
        return Err("RowSelection returned unexpected text values".into());
    }

    Ok((elapsed_ms, counters.snapshot()))
}

fn percentile_f64(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

fn percentile_u64(values: &mut [u64], percentile: f64) -> u64 {
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

fn mean_u64(values: &[u64]) -> f64 {
    values.iter().sum::<u64>() as f64 / values.len() as f64
}

fn page_layout(metadata: &ArrowReaderMetadata) -> Result<(usize, f64), String> {
    let indexes = metadata
        .metadata()
        .offset_index()
        .ok_or_else(|| "offset index was not loaded".to_owned())?;
    let text_index = &indexes[0][1];
    let pages = text_index.page_locations();
    let bytes = pages
        .iter()
        .map(|page| page.compressed_page_size as u64)
        .sum::<u64>();
    Ok((pages.len(), bytes as f64 / pages.len() as f64))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse().map_err(|error| format!("argument error: {error}"))?;
    std::fs::create_dir_all(&config.output_dir)?;

    println!(
        "Generating {} rows x {} bytes of deterministic text...",
        config.rows, config.text_bytes
    );
    let batch = make_batch(&config)?;
    let queries = make_queries(&config);
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);

    let mut pending = Vec::new();
    for page_size_kib in PAGE_SIZES_KIB {
        let path = config
            .output_dir
            .join(format!("text-{page_size_kib}k.parquet"));
        println!("Writing {}...", path.display());
        let build_seconds = write_file(&path, &batch, page_size_kib)?;
        let metadata = ArrowReaderMetadata::load(&File::open(&path)?, options.clone())?;
        let file_bytes = std::fs::metadata(&path)?.len();
        let (text_pages, mean_page_bytes) = page_layout(&metadata)?;
        pending.push((
            page_size_kib,
            path,
            build_seconds,
            metadata,
            file_bytes,
            text_pages,
            mean_page_bytes,
        ));
    }
    drop(batch);

    let baseline_bytes = pending
        .iter()
        .find(|entry| entry.0 == 1024)
        .expect("1 MiB baseline missing")
        .4;
    let mut results = Vec::new();
    for (page_size_kib, path, build_seconds, metadata, file_bytes, text_pages, mean_page_bytes) in
        pending
    {
        println!(
            "Reading random Top-{} from {} KiB pages...",
            config.top_k, page_size_kib
        );
        let mut latencies = Vec::with_capacity(config.queries);
        let mut api_calls = Vec::with_capacity(config.queries);
        let mut ranges = Vec::with_capacity(config.queries);
        let mut bytes = Vec::with_capacity(config.queries);
        for (query_index, row_ids) in queries.iter().enumerate() {
            let (latency, io) = run_query(&path, &metadata, row_ids, config.rows).await?;
            if query_index >= config.warmups {
                latencies.push(latency);
                api_calls.push(io.api_calls);
                ranges.push(io.ranges);
                bytes.push(io.bytes);
            }
        }
        let latency_mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let latency_p50 = percentile_f64(&mut latencies.clone(), 0.50);
        let latency_p95 = percentile_f64(&mut latencies, 0.95);
        let bytes_p95 = percentile_u64(&mut bytes.clone(), 0.95);
        results.push(FileResult {
            page_size_kib,
            file_bytes,
            file_growth_percent: (file_bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0,
            text_pages,
            mean_compressed_text_page_bytes: mean_page_bytes,
            build_seconds,
            latency_ms_p50: latency_p50,
            latency_ms_p95: latency_p95,
            latency_ms_mean: latency_mean,
            range_requests_mean: mean_u64(&ranges),
            io_api_calls_mean: mean_u64(&api_calls),
            bytes_read_mean: mean_u64(&bytes),
            bytes_read_p95: bytes_p95,
        });
    }

    let output = Output {
        rows: config.rows,
        text_bytes_per_row: config.text_bytes,
        top_k: config.top_k,
        measured_queries: config.queries,
        warmup_queries: config.warmups,
        metadata_cached: true,
        row_groups_per_file: 1,
        compression: "ZSTD(level=1), text dictionary disabled",
        baseline_page_size_kib: 1024,
        results,
    };
    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(config.output_dir.join("result.json"), format!("{json}\n"))?;
    println!("\n{json}");
    Ok(())
}
