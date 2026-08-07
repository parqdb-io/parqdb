use std::fs::File;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow::array::{Array, Float32Array, RecordBatch, UInt8Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

use crate::layout::{
    ColumnLayout, LayoutKind, PackedLayout, ParquetLayout, fixed_binary_array, list_array,
    write_column_parquet, write_fixed_binary_parquet, write_list_parquet,
};
use crate::quantization::{EncodedVectors, Quantizer, QuantizerKind};
use crate::simd;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchmarkConfig {
    pub(crate) rows: usize,
    pub(crate) dimension: usize,
    pub(crate) batch_rows: usize,
    pub(crate) candidate_rows: Vec<usize>,
    pub(crate) warmups: usize,
    pub(crate) repetitions: usize,
}

#[derive(Serialize)]
pub(crate) struct BenchmarkReport {
    pub(crate) input: PathBuf,
    pub(crate) distance_kernel: &'static str,
    pub(crate) config: BenchmarkConfig,
    pub(crate) cases: Vec<QuantizerReport>,
}

#[derive(Serialize)]
pub(crate) struct QuantizerReport {
    pub(crate) quantizer: QuantizerKind,
    pub(crate) layouts: Vec<LayoutReport>,
    pub(crate) max_absolute_distance_difference: f32,
    pub(crate) max_relative_distance_difference: f32,
}

#[derive(Serialize)]
pub(crate) struct LayoutReport {
    pub(crate) layout: LayoutKind,
    pub(crate) code_bits: u8,
    pub(crate) code_columns: usize,
    pub(crate) parquet_physical_code_type: &'static str,
    pub(crate) parquet_file_bytes: u64,
    pub(crate) resident_code_bytes: usize,
    pub(crate) distance_buffer_bytes: usize,
    pub(crate) memory: Vec<Measurement>,
    pub(crate) parquet_scan_decode: Measurement,
    pub(crate) parquet_scan_decode_distance: Measurement,
    pub(crate) decoded_batch_bytes_max: usize,
}

#[derive(Serialize)]
pub(crate) struct Measurement {
    pub(crate) rows: usize,
    pub(crate) samples_ms: Vec<f64>,
    pub(crate) minimum_ms: f64,
    pub(crate) median_ms: f64,
    pub(crate) vectors_per_second: f64,
}

pub(crate) fn benchmark_case(
    output_dir: &Path,
    config: &BenchmarkConfig,
    quantizer: &Quantizer,
    encoded: &EncodedVectors,
    query: &[f32],
) -> Result<QuantizerReport, Box<dyn std::error::Error>> {
    let columns = ColumnLayout::from_quantized(quantizer, encoded, config.rows, config.dimension);
    let packed = PackedLayout::from_encoded(quantizer, encoded, config.rows, config.dimension);

    let column_path = output_dir.join(format!("{}-columns.parquet", quantizer.kind.name()));
    let list_path = output_dir.join(format!("{}-list.parquet", quantizer.kind.name()));
    let fixed_path = output_dir.join(format!("{}-fixed-binary.parquet", quantizer.kind.name()));
    let column_file = write_column_parquet(&column_path, &columns, config.rows, config.batch_rows)?;
    let list_file = write_list_parquet(&list_path, &packed, config.rows, config.batch_rows)?;
    let fixed_file =
        write_fixed_binary_parquet(&fixed_path, &packed, config.rows, config.batch_rows)?;

    let (max_absolute_difference, max_relative_difference) =
        validate_layouts(config, quantizer, &columns, &packed, query);
    if max_relative_difference > 1e-5 {
        return Err(format!(
            "{} layouts disagree: max relative distance difference {max_relative_difference}",
            quantizer.kind.name()
        )
        .into());
    }

    eprintln!("benchmarking {} columns", quantizer.kind.name());
    let column_report = benchmark_columns(config, quantizer, &columns, query, &column_file)?;
    eprintln!("benchmarking {} list", quantizer.kind.name());
    let list_report = benchmark_packed(config, quantizer, &packed, query, &list_file)?;
    eprintln!("benchmarking {} fixed binary", quantizer.kind.name());
    let fixed_report = benchmark_packed(config, quantizer, &packed, query, &fixed_file)?;

    Ok(QuantizerReport {
        quantizer: quantizer.kind,
        layouts: vec![column_report, list_report, fixed_report],
        max_absolute_distance_difference: max_absolute_difference,
        max_relative_distance_difference: max_relative_difference,
    })
}

fn benchmark_columns(
    config: &BenchmarkConfig,
    quantizer: &Quantizer,
    layout: &ColumnLayout,
    query: &[f32],
    parquet: &ParquetLayout,
) -> Result<LayoutReport, Box<dyn std::error::Error>> {
    let mut memory = Vec::new();
    for &rows in &config.candidate_rows {
        let mut distances = vec![0.0_f32; rows.min(config.batch_rows)];
        memory.push(measure(config, rows, || {
            let mut checksum = 0.0_f64;
            for start in (0..rows).step_by(config.batch_rows) {
                let batch_rows = config.batch_rows.min(rows - start);
                distance_columns(quantizer, layout, query, start, batch_rows, &mut distances);
                checksum += distances[..batch_rows]
                    .iter()
                    .map(|&value| f64::from(value))
                    .sum::<f64>();
            }
            black_box(checksum);
        }));
    }
    let (parquet_scan_decode, decoded_batch_bytes_max) = measure_scan(config, parquet)?;
    let parquet_scan_decode_distance = measure_scan_distance(config, quantizer, parquet, query)?;
    Ok(LayoutReport {
        layout: parquet.kind,
        code_bits: quantizer.kind.bits(),
        code_columns: layout.codes.len(),
        parquet_physical_code_type: "INT32",
        parquet_file_bytes: parquet.file_bytes,
        resident_code_bytes: layout.resident_bytes(),
        distance_buffer_bytes: config.rows.min(config.batch_rows) * size_of::<f32>(),
        memory,
        parquet_scan_decode,
        parquet_scan_decode_distance,
        decoded_batch_bytes_max,
    })
}

fn benchmark_packed(
    config: &BenchmarkConfig,
    quantizer: &Quantizer,
    layout: &PackedLayout,
    query: &[f32],
    parquet: &ParquetLayout,
) -> Result<LayoutReport, Box<dyn std::error::Error>> {
    let mut memory = Vec::new();
    for &rows in &config.candidate_rows {
        let mut distances = vec![0.0_f32; rows.min(config.batch_rows)];
        memory.push(measure(config, rows, || {
            let mut checksum = 0.0_f64;
            for start in (0..rows).step_by(config.batch_rows) {
                let batch_rows = config.batch_rows.min(rows - start);
                distance_packed(quantizer, layout, query, start, batch_rows, &mut distances);
                checksum += distances[..batch_rows]
                    .iter()
                    .map(|&value| f64::from(value))
                    .sum::<f64>();
            }
            black_box(checksum);
        }));
    }
    let (parquet_scan_decode, decoded_batch_bytes_max) = measure_scan(config, parquet)?;
    let parquet_scan_decode_distance = measure_scan_distance(config, quantizer, parquet, query)?;
    Ok(LayoutReport {
        layout: parquet.kind,
        code_bits: quantizer.kind.bits(),
        code_columns: 1,
        parquet_physical_code_type: match parquet.kind {
            LayoutKind::List => "INT32",
            LayoutKind::FixedBinary => "FIXED_LEN_BYTE_ARRAY",
            LayoutKind::Columns => unreachable!("row layout cannot be columns"),
        },
        parquet_file_bytes: parquet.file_bytes,
        resident_code_bytes: layout.resident_bytes(),
        distance_buffer_bytes: config.rows.min(config.batch_rows) * size_of::<f32>(),
        memory,
        parquet_scan_decode,
        parquet_scan_decode_distance,
        decoded_batch_bytes_max,
    })
}

fn measure(config: &BenchmarkConfig, rows: usize, mut operation: impl FnMut()) -> Measurement {
    for _ in 0..config.warmups {
        operation();
    }
    let mut samples_ms = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        let start = Instant::now();
        operation();
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    summarize(rows, samples_ms)
}

fn summarize(rows: usize, mut samples_ms: Vec<f64>) -> Measurement {
    samples_ms.sort_by(f64::total_cmp);
    let minimum_ms = samples_ms[0];
    let median_ms = if samples_ms.len() & 1 == 0 {
        let middle = samples_ms.len() / 2;
        (samples_ms[middle - 1] + samples_ms[middle]) * 0.5
    } else {
        samples_ms[samples_ms.len() / 2]
    };
    Measurement {
        rows,
        minimum_ms,
        median_ms,
        vectors_per_second: rows as f64 / (median_ms / 1_000.0),
        samples_ms,
    }
}

fn distance_columns(
    quantizer: &Quantizer,
    layout: &ColumnLayout,
    query: &[f32],
    start: usize,
    rows: usize,
    distances: &mut [f32],
) {
    let offsets = layout
        .offsets
        .as_ref()
        .map(|values| &values[start..start + rows]);
    let scales = layout
        .scales
        .as_ref()
        .map(|values| &values[start..start + rows]);
    simd::distance_columns(
        quantizer.kind.bits(),
        layout.codes.len(),
        |column| &layout.codes[column][start..start + rows],
        query,
        &quantizer.lower,
        &quantizer.scale,
        offsets,
        scales,
        &mut distances[..rows],
    );
}

fn distance_packed(
    quantizer: &Quantizer,
    layout: &PackedLayout,
    query: &[f32],
    start: usize,
    rows: usize,
    distances: &mut [f32],
) {
    let offsets = layout
        .offsets
        .as_ref()
        .map(|values| &values[start..start + rows]);
    let scales = layout
        .scales
        .as_ref()
        .map(|values| &values[start..start + rows]);
    simd::distance_rows(
        quantizer.kind.bits(),
        &layout.codes[start * layout.stride..(start + rows) * layout.stride],
        layout.stride,
        query,
        &quantizer.lower,
        &quantizer.scale,
        offsets,
        scales,
        &mut distances[..rows],
    );
}

fn validate_layouts(
    config: &BenchmarkConfig,
    quantizer: &Quantizer,
    columns: &ColumnLayout,
    packed: &PackedLayout,
    query: &[f32],
) -> (f32, f32) {
    let mut column_distances = vec![0.0_f32; config.batch_rows.min(config.rows)];
    let mut packed_distances = vec![0.0_f32; config.batch_rows.min(config.rows)];
    let mut max_difference = 0.0_f32;
    let mut max_relative_difference = 0.0_f32;
    for start in (0..config.rows).step_by(config.batch_rows) {
        let rows = config.batch_rows.min(config.rows - start);
        distance_columns(
            quantizer,
            columns,
            query,
            start,
            rows,
            &mut column_distances,
        );
        distance_packed(quantizer, packed, query, start, rows, &mut packed_distances);
        for (&left, &right) in column_distances[..rows]
            .iter()
            .zip(&packed_distances[..rows])
        {
            max_difference = max_difference.max((left - right).abs());
            let denominator = left.abs().max(right.abs()).max(1.0);
            max_relative_difference =
                max_relative_difference.max((left - right).abs() / denominator);
        }
    }
    (max_difference, max_relative_difference)
}

fn measure_scan(
    config: &BenchmarkConfig,
    parquet: &ParquetLayout,
) -> Result<(Measurement, usize), Box<dyn std::error::Error>> {
    let (_, decoded_batch_bytes_max) = scan_file(parquet, config.batch_rows)?;
    for _ in 0..config.warmups {
        black_box(scan_file(parquet, config.batch_rows)?.0);
    }
    let mut samples = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        let start = Instant::now();
        black_box(scan_file(parquet, config.batch_rows)?.0);
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok((summarize(config.rows, samples), decoded_batch_bytes_max))
}

fn scan_file(
    parquet: &ParquetLayout,
    batch_rows: usize,
) -> Result<(u64, usize), Box<dyn std::error::Error>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&parquet.path)?)?
        .with_batch_size(batch_rows)
        .build()?;
    let mut checksum = 0_u64;
    let mut decoded_batch_bytes_max = 0;
    for batch in reader {
        let batch = batch?;
        checksum = checksum.wrapping_add(consume_batch(&batch));
        decoded_batch_bytes_max = decoded_batch_bytes_max.max(batch.get_array_memory_size());
    }
    Ok((checksum, decoded_batch_bytes_max))
}

fn consume_batch(batch: &RecordBatch) -> u64 {
    let mut checksum = batch.num_rows() as u64;
    for array in batch.columns() {
        checksum = checksum
            .wrapping_mul(31)
            .wrapping_add(array.get_buffer_memory_size() as u64)
            .wrapping_add(array.null_count() as u64);
    }
    checksum
}

fn measure_scan_distance(
    config: &BenchmarkConfig,
    quantizer: &Quantizer,
    parquet: &ParquetLayout,
    query: &[f32],
) -> Result<Measurement, Box<dyn std::error::Error>> {
    for _ in 0..config.warmups {
        black_box(scan_distance(parquet, config.batch_rows, quantizer, query)?);
    }
    let mut samples = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        let start = Instant::now();
        black_box(scan_distance(parquet, config.batch_rows, quantizer, query)?);
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok(summarize(config.rows, samples))
}

fn scan_distance(
    parquet: &ParquetLayout,
    batch_rows: usize,
    quantizer: &Quantizer,
    query: &[f32],
) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&parquet.path)?)?
        .with_batch_size(batch_rows)
        .build()?;
    let mut checksum = 0.0_f64;
    for batch in reader {
        let batch = batch?;
        let mut distances = vec![0.0_f32; batch.num_rows()];
        match parquet.kind {
            LayoutKind::Columns => {
                distance_column_batch(quantizer, &batch, query, &mut distances);
            }
            LayoutKind::List => {
                distance_packed_batch(quantizer, &batch, query, &mut distances);
            }
            LayoutKind::FixedBinary => {
                distance_packed_batch(quantizer, &batch, query, &mut distances);
            }
        }
        checksum += distances.iter().map(|&value| f64::from(value)).sum::<f64>();
    }
    Ok(checksum)
}

fn distance_column_batch(
    quantizer: &Quantizer,
    batch: &RecordBatch,
    query: &[f32],
    distances: &mut [f32],
) {
    let code_columns = if quantizer.kind.bits() == 4 {
        query.len().div_ceil(2)
    } else {
        query.len()
    };
    let offsets = quantizer
        .kind
        .is_lvq()
        .then(|| float_values(batch, code_columns));
    let scales = quantizer
        .kind
        .is_lvq()
        .then(|| float_values(batch, code_columns + 1));
    simd::distance_columns(
        quantizer.kind.bits(),
        code_columns,
        |column| uint8_values(batch, column),
        query,
        &quantizer.lower,
        &quantizer.scale,
        offsets,
        scales,
        distances,
    );
}

fn distance_packed_batch(
    quantizer: &Quantizer,
    batch: &RecordBatch,
    query: &[f32],
    distances: &mut [f32],
) {
    let expected_stride = if quantizer.kind.bits() == 4 {
        query.len().div_ceil(2)
    } else {
        query.len()
    };
    let codes = packed_batch_values(batch, expected_stride);
    let offsets = quantizer.kind.is_lvq().then(|| float_values(batch, 1));
    let scales = quantizer.kind.is_lvq().then(|| float_values(batch, 2));
    simd::distance_rows(
        quantizer.kind.bits(),
        codes,
        expected_stride,
        query,
        &quantizer.lower,
        &quantizer.scale,
        offsets,
        scales,
        distances,
    );
}

fn packed_batch_values(batch: &RecordBatch, stride: usize) -> &[u8] {
    match batch.column(0).data_type() {
        arrow::datatypes::DataType::List(_) => {
            let codes = list_array(batch);
            let offsets = codes.value_offsets();
            let values = codes
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .expect("LIST<UINT8> values");
            let start = usize::try_from(offsets[0]).expect("non-negative LIST offset");
            let end = usize::try_from(offsets[batch.num_rows()]).expect("non-negative LIST offset");
            assert_eq!(end - start, batch.num_rows() * stride);
            &values.values()[start..end]
        }
        arrow::datatypes::DataType::FixedSizeBinary(_) => {
            let codes = fixed_binary_array(batch);
            let start = usize::try_from(codes.value_offset(0)).expect("binary value offset");
            let end = start + batch.num_rows() * stride;
            &codes.value_data()[start..end]
        }
        other => panic!("unexpected packed code type: {other}"),
    }
}

fn uint8_values(batch: &RecordBatch, column: usize) -> &[u8] {
    batch
        .column(column)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .expect("UINT8 code column")
        .values()
}

fn float_values(batch: &RecordBatch, column: usize) -> &[f32] {
    batch
        .column(column)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("FLOAT32 parameter column")
        .values()
}
