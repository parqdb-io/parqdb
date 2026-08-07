use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrow::array::{Array, ArrayRef, BinaryArray, StringArray, StringBuilder};
use arrow::datatypes::DataType;
use datafusion::common::DataFusionError;
use datafusion::datasource::file_format::options::ParquetReadOptions;
use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{SessionConfig, SessionContext};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    PageZstd,
    RowZstd,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "page" => Ok(Self::PageZstd),
            "row" => Ok(Self::RowZstd),
            _ => Err(format!("mode must be 'page' or 'row', got {value}")),
        }
    }
}

#[derive(Debug)]
struct Config {
    mode: Mode,
    path: PathBuf,
    queries: usize,
    warmups: usize,
    batch_size: usize,
    output: PathBuf,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut mode = None;
        let mut path = None;
        let mut queries = 10;
        let mut warmups = 2;
        let mut batch_size = 8;
        let mut output = None;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--mode" => mode = Some(Mode::parse(&value)?),
                "--path" => path = Some(PathBuf::from(value)),
                "--queries" => queries = parse_usize(&flag, &value)?,
                "--warmups" => warmups = parse_usize(&flag, &value)?,
                "--batch-size" => batch_size = parse_usize(&flag, &value)?,
                "--output" => output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown argument: {flag}")),
            }
        }
        let mode = mode.ok_or_else(|| "--mode is required".to_owned())?;
        let path = path.ok_or_else(|| "--path is required".to_owned())?;
        let output = output.ok_or_else(|| "--output is required".to_owned())?;
        if queries == 0 || batch_size == 0 {
            return Err("queries and batch-size must be positive".into());
        }
        Ok(Self {
            mode,
            path,
            queries,
            warmups,
            batch_size,
            output,
        })
    }
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|error| format!("invalid value for {flag}: {error}"))
}

#[derive(Default)]
struct DecompressionMetrics {
    rows: AtomicU64,
    compressed_bytes: AtomicU64,
    uncompressed_bytes: AtomicU64,
}

fn register_decompress_udf(ctx: &SessionContext, metrics: Arc<DecompressionMetrics>) {
    let implementation = Arc::new(move |args: &[ColumnarValue]| {
        let ColumnarValue::Array(array) = &args[0] else {
            return Err(DataFusionError::Execution(
                "zstd_decompress expects an array".into(),
            ));
        };
        let input = array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| DataFusionError::Execution("expected BinaryArray".into()))?;
        let mut output = StringBuilder::with_capacity(input.len(), 0);
        for index in 0..input.len() {
            if input.is_null(index) {
                output.append_null();
                continue;
            }
            let compressed = input.value(index);
            let decoded = zstd::stream::decode_all(compressed).map_err(|error| {
                DataFusionError::Execution(format!("zstd decompression failed: {error}"))
            })?;
            let text = std::str::from_utf8(&decoded).map_err(|error| {
                DataFusionError::Execution(format!("payload is not UTF-8: {error}"))
            })?;
            metrics.rows.fetch_add(1, Ordering::Relaxed);
            metrics
                .compressed_bytes
                .fetch_add(compressed.len() as u64, Ordering::Relaxed);
            metrics
                .uncompressed_bytes
                .fetch_add(decoded.len() as u64, Ordering::Relaxed);
            output.append_value(text);
        }
        Ok(ColumnarValue::Array(Arc::new(output.finish()) as ArrayRef))
    });
    ctx.register_udf(create_udf(
        "zstd_decompress",
        vec![DataType::Binary],
        DataType::Utf8,
        Volatility::Immutable,
        implementation,
    ));
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * fraction).ceil() as usize;
    values[index]
}

fn proc_status_kib(name: &str) -> Result<u64, Box<dyn std::error::Error>> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let line = status
        .lines()
        .find(|line| line.starts_with(name))
        .ok_or_else(|| format!("{name} missing from /proc/self/status"))?;
    let value = line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| format!("invalid {name} line"))?
        .parse()?;
    Ok(value)
}

fn result_payload_bytes(batches: &[arrow::record_batch::RecordBatch]) -> usize {
    batches
        .iter()
        .map(|batch| {
            let payload = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("query payload should be Utf8");
            payload.iter().flatten().map(str::len).sum::<usize>()
        })
        .sum()
}

#[derive(Debug, Serialize)]
struct Output {
    mode: Mode,
    path: String,
    file_bytes: u64,
    queries: usize,
    warmups: usize,
    batch_size: usize,
    result_rows: usize,
    result_payload_bytes: usize,
    latency_ms_p50: f64,
    latency_ms_p95: f64,
    latency_ms_mean: f64,
    rss_before_query_kib: u64,
    peak_rss_kib: u64,
    peak_growth_kib: u64,
    decompressed_rows: u64,
    decompressed_compressed_bytes: u64,
    decompressed_uncompressed_bytes: u64,
    physical_plan: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse().map_err(|error| format!("argument error: {error}"))?;
    let mut session_config = SessionConfig::new()
        .with_batch_size(config.batch_size)
        .with_target_partitions(8);
    session_config
        .options_mut()
        .execution
        .parquet
        .schema_force_view_types = false;
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = false;
    let ctx = SessionContext::new_with_config(session_config);
    let metrics = Arc::new(DecompressionMetrics::default());
    if matches!(config.mode, Mode::RowZstd) {
        register_decompress_udf(&ctx, Arc::clone(&metrics));
    }
    ctx.register_parquet(
        "documents",
        config.path.to_string_lossy(),
        ParquetReadOptions::default(),
    )
    .await?;

    let ranked =
        "SELECT doc_id, payload FROM documents ORDER BY ((doc_id * 7919) % 104729) DESC LIMIT 10";
    let sql = match config.mode {
        Mode::PageZstd => ranked.to_owned(),
        Mode::RowZstd => {
            format!("SELECT doc_id, zstd_decompress(payload) AS payload FROM ({ranked}) ranked")
        }
    };
    let dataframe = ctx.sql(&sql).await?;
    let plan = dataframe.create_physical_plan().await?;
    let physical_plan = displayable(plan.as_ref()).indent(true).to_string();
    println!("Physical plan:\n{physical_plan}");

    let rss_before_query_kib = proc_status_kib("VmRSS:")?;
    let mut latencies = Vec::with_capacity(config.queries);
    let mut result_rows = 0;
    let mut payload_bytes = 0;
    for query_index in 0..config.warmups + config.queries {
        // Top-K dynamic filters retain run-specific state, so each execution needs
        // a fresh physical plan. Planning remains outside the measured interval.
        let run_dataframe = ctx.sql(&sql).await?;
        let run_plan = run_dataframe.create_physical_plan().await?;
        let started = Instant::now();
        let batches = collect(run_plan, ctx.task_ctx()).await?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        result_rows = batches.iter().map(|batch| batch.num_rows()).sum();
        payload_bytes = result_payload_bytes(&batches);
        if result_rows != 10 {
            return Err(format!("expected 10 result rows, got {result_rows}").into());
        }
        if query_index >= config.warmups {
            latencies.push(elapsed_ms);
        }
    }

    let latency_mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let latency_p50 = percentile(&mut latencies.clone(), 0.50);
    let latency_p95 = percentile(&mut latencies, 0.95);
    let peak_rss_kib = proc_status_kib("VmHWM:")?;
    let output = Output {
        mode: config.mode,
        path: config.path.display().to_string(),
        file_bytes: std::fs::metadata(&config.path)?.len(),
        queries: config.queries,
        warmups: config.warmups,
        batch_size: config.batch_size,
        result_rows,
        result_payload_bytes: payload_bytes,
        latency_ms_p50: latency_p50,
        latency_ms_p95: latency_p95,
        latency_ms_mean: latency_mean,
        rss_before_query_kib,
        peak_rss_kib,
        peak_growth_kib: peak_rss_kib.saturating_sub(rss_before_query_kib),
        decompressed_rows: metrics.rows.load(Ordering::Relaxed),
        decompressed_compressed_bytes: metrics.compressed_bytes.load(Ordering::Relaxed),
        decompressed_uncompressed_bytes: metrics.uncompressed_bytes.load(Ordering::Relaxed),
        physical_plan,
    };
    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&config.output, format!("{json}\n"))?;
    println!("\n{json}");
    Ok(())
}
