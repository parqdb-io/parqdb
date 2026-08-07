use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow::array::{Array, ArrayRef, BinaryArray, Int64Array, StringArray};
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

const PAGE_SIZE: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum StorageMode {
    PageZstdBatch8,
    PageZstdBatch1,
    RowZstd,
}

impl StorageMode {
    fn name(self) -> &'static str {
        match self {
            Self::PageZstdBatch8 => "page-zstd-batch-8",
            Self::PageZstdBatch1 => "page-zstd-batch-1",
            Self::RowZstd => "row-zstd",
        }
    }

    fn write_batch_size(self) -> usize {
        match self {
            Self::PageZstdBatch8 => 8,
            Self::PageZstdBatch1 | Self::RowZstd => 1,
        }
    }
}

#[derive(Debug)]
struct Config {
    text_mib: usize,
    rows: usize,
    queries: usize,
    warmups: usize,
    top_k: usize,
    output_dir: PathBuf,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut text_mib = 1;
        let mut rows = 128;
        let mut queries = 20;
        let mut warmups = 3;
        let mut top_k = 10;
        let mut output_dir = PathBuf::from("/tmp/relify-large-text");
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--text-mib" => text_mib = parse_usize(&flag, &value)?,
                "--rows" => rows = parse_usize(&flag, &value)?,
                "--queries" => queries = parse_usize(&flag, &value)?,
                "--warmups" => warmups = parse_usize(&flag, &value)?,
                "--top-k" => top_k = parse_usize(&flag, &value)?,
                "--output-dir" => output_dir = PathBuf::from(value),
                _ => return Err(format!("unknown argument: {flag}")),
            }
        }
        if text_mib == 0 || rows == 0 || queries == 0 || top_k == 0 || top_k > rows {
            return Err(
                "text-mib, rows, queries, and top-k must be positive; top-k must not exceed rows"
                    .into(),
            );
        }
        Ok(Self {
            text_mib,
            rows,
            queries,
            warmups,
            top_k,
            output_dir,
        })
    }

    fn text_bytes(&self) -> usize {
        self.text_mib * 1024 * 1024
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

#[derive(Clone, Copy, Debug)]
struct IoSample {
    api_calls: u64,
    ranges: u64,
    bytes: u64,
}

#[derive(Debug, Serialize)]
struct ModeResult {
    mode: StorageMode,
    file_bytes: u64,
    file_growth_percent: f64,
    compression_ratio: f64,
    payload_pages: usize,
    rows_per_page: f64,
    mean_page_bytes_on_disk: f64,
    build_seconds: f64,
    latency_ms_p50: f64,
    latency_ms_p95: f64,
    latency_ms_mean: f64,
    range_requests_mean: f64,
    io_api_calls_mean: f64,
    bytes_read_mean: f64,
    bytes_read_p95: u64,
    io_bytes_per_selected_raw_byte: f64,
}

#[derive(Debug, Serialize)]
struct Output {
    rows: usize,
    text_mib_per_row: usize,
    raw_payload_bytes: u64,
    top_k: usize,
    measured_queries: usize,
    warmup_queries: usize,
    page_size_kib: usize,
    metadata_cached: bool,
    corpus: &'static str,
    results: Vec<ModeResult>,
}

const WORDS: [&str; 32] = [
    "agent",
    "assistant",
    "context",
    "database",
    "embedding",
    "error",
    "evaluation",
    "function",
    "generation",
    "input",
    "knowledge",
    "language",
    "message",
    "metadata",
    "model",
    "observation",
    "output",
    "prompt",
    "request",
    "response",
    "retrieval",
    "score",
    "session",
    "span",
    "system",
    "token",
    "tool",
    "trace",
    "trajectory",
    "user",
    "vector",
    "workflow",
];

fn generate_text(doc_id: usize, target_bytes: usize) -> String {
    let prefix = format!("{{\"trace_id\":\"{doc_id:016x}\",\"type\":\"generation\",\"input\":\"");
    let suffix = "\",\"status\":\"ok\",\"model\":\"example-model\"}";
    let payload_end = target_bytes.saturating_sub(suffix.len());
    let mut state = (doc_id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut text = String::with_capacity(target_bytes);
    text.push_str(&prefix);
    while text.len() < payload_end {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let word = WORDS[(state as usize) % WORDS.len()];
        let remaining = payload_end - text.len();
        if word.len() + 1 > remaining {
            text.extend(std::iter::repeat_n('x', remaining));
            break;
        }
        text.push_str(word);
        text.push(' ');
    }
    text.push_str(suffix);
    text
}

fn make_page_batch(texts: &[String]) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let ids = Int64Array::from_iter_values(0..texts.len() as i64);
    let payloads = StringArray::from_iter_values(texts);
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(ids) as ArrayRef, Arc::new(payloads) as ArrayRef],
    )?)
}

fn make_row_batch(texts: &[String]) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Int64, false),
        Field::new("payload", DataType::Binary, false),
    ]));
    let ids = Int64Array::from_iter_values(0..texts.len() as i64);
    let compressed = texts
        .iter()
        .map(|text| zstd::stream::encode_all(text.as_bytes(), 1))
        .collect::<Result<Vec<_>, _>>()?;
    let payloads = BinaryArray::from_iter_values(compressed.iter());
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(ids) as ArrayRef, Arc::new(payloads) as ArrayRef],
    )?)
}

fn write_file(
    path: &Path,
    texts: &[String],
    mode: StorageMode,
) -> Result<f64, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let batch = match mode {
        StorageMode::PageZstdBatch8 | StorageMode::PageZstdBatch1 => make_page_batch(texts)?,
        StorageMode::RowZstd => make_row_batch(texts)?,
    };
    let payload = ColumnPath::from("payload");
    let payload_compression = match mode {
        StorageMode::PageZstdBatch8 | StorageMode::PageZstdBatch1 => {
            Compression::ZSTD(ZstdLevel::default())
        }
        StorageMode::RowZstd => Compression::UNCOMPRESSED,
    };
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_column_compression(payload.clone(), payload_compression)
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_column_statistics_enabled(payload.clone(), EnabledStatistics::Page)
        .set_column_dictionary_enabled(payload.clone(), false)
        .set_column_data_page_size_limit(payload, PAGE_SIZE)
        .set_write_batch_size(mode.write_batch_size())
        .set_max_row_group_row_count(Some(texts.len()))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), Some(properties))?;
    writer.write(&batch)?;
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
    expected: &[String],
    mode: StorageMode,
) -> Result<(f64, IoSample), Box<dyn std::error::Error>> {
    let selection = RowSelection::from_consecutive_ranges(
        row_ids.iter().copied().map(|row| row..row + 1),
        expected.len(),
    );
    let projection = ProjectionMask::leaves(metadata.parquet_schema(), [1]);
    let reader = CountingReader::new(path, Arc::clone(metadata.metadata()));
    let counters = reader.clone();
    let started = Instant::now();
    let stream = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata.clone())
        .with_projection(projection)
        .with_row_selection(selection)
        .with_batch_size(64)
        .build()?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut result_index = 0;
    for batch in &batches {
        match mode {
            StorageMode::PageZstdBatch8 | StorageMode::PageZstdBatch1 => {
                let payloads = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                for value_index in 0..payloads.len() {
                    let row_id = row_ids[result_index];
                    if payloads.value(value_index) != expected[row_id] {
                        return Err("page-compressed payload mismatch".into());
                    }
                    result_index += 1;
                }
            }
            StorageMode::RowZstd => {
                let payloads = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .unwrap();
                for value_index in 0..payloads.len() {
                    let decoded = zstd::stream::decode_all(payloads.value(value_index))?;
                    let row_id = row_ids[result_index];
                    if decoded != expected[row_id].as_bytes() {
                        return Err("row-compressed payload mismatch".into());
                    }
                    result_index += 1;
                }
            }
        }
    }
    if result_index != row_ids.len() {
        return Err(format!("expected {} rows, read {result_index}", row_ids.len()).into());
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok((elapsed_ms, counters.snapshot()))
}

fn page_layout(metadata: &ArrowReaderMetadata) -> Result<(usize, f64), String> {
    let indexes = metadata
        .metadata()
        .offset_index()
        .ok_or_else(|| "offset index was not loaded".to_owned())?;
    let pages = indexes[0][1].page_locations();
    let bytes = pages
        .iter()
        .map(|page| page.compressed_page_size as u64)
        .sum::<u64>();
    Ok((pages.len(), bytes as f64 / pages.len() as f64))
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse().map_err(|error| format!("argument error: {error}"))?;
    std::fs::create_dir_all(&config.output_dir)?;
    println!(
        "Generating {} rows x {} MiB of Langfuse-like text...",
        config.rows, config.text_mib
    );
    let texts = (0..config.rows)
        .map(|doc_id| generate_text(doc_id, config.text_bytes()))
        .collect::<Vec<_>>();
    let raw_payload_bytes = texts.iter().map(String::len).sum::<usize>() as u64;
    let queries = make_queries(&config);
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    let modes = [
        StorageMode::PageZstdBatch8,
        StorageMode::PageZstdBatch1,
        StorageMode::RowZstd,
    ];

    let mut pending = Vec::new();
    for mode in modes {
        let path = config.output_dir.join(format!("{}.parquet", mode.name()));
        println!("Writing {}...", path.display());
        let build_seconds = write_file(&path, &texts, mode)?;
        let metadata = ArrowReaderMetadata::load(&File::open(&path)?, options.clone())?;
        let file_bytes = std::fs::metadata(&path)?.len();
        let (payload_pages, mean_page_bytes) = page_layout(&metadata)?;
        pending.push((
            mode,
            path,
            build_seconds,
            metadata,
            file_bytes,
            payload_pages,
            mean_page_bytes,
        ));
    }

    let baseline_bytes = pending
        .iter()
        .find(|entry| matches!(entry.0, StorageMode::PageZstdBatch1))
        .expect("page-zstd-batch-1 baseline missing")
        .4;
    let selected_raw_bytes = (config.top_k * config.text_bytes()) as f64;
    let mut results = Vec::new();
    for (mode, path, build_seconds, metadata, file_bytes, payload_pages, mean_page_bytes) in pending
    {
        println!(
            "Reading random Top-{} from {}...",
            config.top_k,
            mode.name()
        );
        let mut latencies = Vec::with_capacity(config.queries);
        let mut api_calls = Vec::with_capacity(config.queries);
        let mut ranges = Vec::with_capacity(config.queries);
        let mut bytes = Vec::with_capacity(config.queries);
        for (query_index, row_ids) in queries.iter().enumerate() {
            let (latency, io) = run_query(&path, &metadata, row_ids, &texts, mode).await?;
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
        let bytes_mean = mean_u64(&bytes);
        let bytes_p95 = percentile_u64(&mut bytes, 0.95);
        results.push(ModeResult {
            mode,
            file_bytes,
            file_growth_percent: (file_bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0,
            compression_ratio: raw_payload_bytes as f64 / file_bytes as f64,
            payload_pages,
            rows_per_page: config.rows as f64 / payload_pages as f64,
            mean_page_bytes_on_disk: mean_page_bytes,
            build_seconds,
            latency_ms_p50: latency_p50,
            latency_ms_p95: latency_p95,
            latency_ms_mean: latency_mean,
            range_requests_mean: mean_u64(&ranges),
            io_api_calls_mean: mean_u64(&api_calls),
            bytes_read_mean: bytes_mean,
            bytes_read_p95: bytes_p95,
            io_bytes_per_selected_raw_byte: bytes_mean / selected_raw_bytes,
        });
    }

    let output = Output {
        rows: config.rows,
        text_mib_per_row: config.text_mib,
        raw_payload_bytes,
        top_k: config.top_k,
        measured_queries: config.queries,
        warmup_queries: config.warmups,
        page_size_kib: PAGE_SIZE / 1024,
        metadata_cached: true,
        corpus: "deterministic Langfuse-like JSON with natural-language vocabulary",
        results,
    };
    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(config.output_dir.join("result.json"), format!("{json}\n"))?;
    println!("\n{json}");
    Ok(())
}
