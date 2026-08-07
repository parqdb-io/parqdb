use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeBinaryBuilder, Float32Array, ListArray,
    ListBuilder, RecordBatch, UInt8Array, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use serde::Serialize;

use crate::quantization::{EncodedVectors, Quantizer, pack_codes};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LayoutKind {
    Columns,
    List,
    FixedBinary,
}

pub(crate) struct ColumnLayout {
    pub(crate) codes: Vec<Vec<u8>>,
    pub(crate) offsets: Option<Vec<f32>>,
    pub(crate) scales: Option<Vec<f32>>,
}

pub(crate) struct PackedLayout {
    pub(crate) codes: Vec<u8>,
    pub(crate) stride: usize,
    pub(crate) offsets: Option<Vec<f32>>,
    pub(crate) scales: Option<Vec<f32>>,
}

pub(crate) struct ParquetLayout {
    pub(crate) kind: LayoutKind,
    pub(crate) path: PathBuf,
    pub(crate) file_bytes: u64,
}

impl ColumnLayout {
    pub(crate) fn from_encoded(encoded: &EncodedVectors, rows: usize, dimension: usize) -> Self {
        let mut codes = (0..dimension)
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for row in encoded.codes.chunks_exact(dimension).take(rows) {
            for (dim, &code) in row.iter().enumerate() {
                codes[dim].push(code);
            }
        }
        Self {
            codes,
            offsets: encoded.offsets.clone(),
            scales: encoded.scales.clone(),
        }
    }

    pub(crate) fn from_quantized(
        quantizer: &Quantizer,
        encoded: &EncodedVectors,
        rows: usize,
        dimension: usize,
    ) -> Self {
        if quantizer.kind.bits() == 8 {
            return Self::from_encoded(encoded, rows, dimension);
        }

        let code_columns = dimension.div_ceil(2);
        let mut codes = (0..code_columns)
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for row in encoded.codes.chunks_exact(dimension).take(rows) {
            for (column, values) in codes.iter_mut().enumerate() {
                let low = row[column * 2] & 0x0f;
                let high = row.get(column * 2 + 1).copied().unwrap_or(0) & 0x0f;
                values.push(low | (high << 4));
            }
        }
        Self {
            codes,
            offsets: encoded.offsets.clone(),
            scales: encoded.scales.clone(),
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.codes.iter().map(Vec::len).sum::<usize>()
            + self.offsets.as_ref().map_or(0, |values| values.len() * 4)
            + self.scales.as_ref().map_or(0, |values| values.len() * 4)
    }
}

impl PackedLayout {
    pub(crate) fn from_encoded(
        quantizer: &Quantizer,
        encoded: &EncodedVectors,
        rows: usize,
        dimension: usize,
    ) -> Self {
        let codes = pack_codes(&encoded.codes, rows, dimension, quantizer.kind.bits());
        let stride = if quantizer.kind.bits() == 4 {
            dimension.div_ceil(2)
        } else {
            dimension
        };
        Self {
            codes,
            stride,
            offsets: encoded.offsets.clone(),
            scales: encoded.scales.clone(),
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.codes.len()
            + self.offsets.as_ref().map_or(0, |values| values.len() * 4)
            + self.scales.as_ref().map_or(0, |values| values.len() * 4)
    }
}

pub(crate) fn write_column_parquet(
    path: &Path,
    layout: &ColumnLayout,
    rows: usize,
    batch_rows: usize,
) -> Result<ParquetLayout, Box<dyn std::error::Error>> {
    let mut fields = Vec::with_capacity(layout.codes.len() + 2);
    let mut arrays = Vec::<ArrayRef>::with_capacity(layout.codes.len() + 2);
    for (dim, values) in layout.codes.iter().enumerate() {
        fields.push(Field::new(format!("code_{dim:04}"), DataType::UInt8, false));
        arrays.push(Arc::new(UInt8Array::from(values.clone())));
    }
    append_lvq_columns(
        &mut fields,
        &mut arrays,
        layout.offsets.as_deref(),
        layout.scales.as_deref(),
    );
    write_batch(path, fields, arrays, rows, batch_rows)?;
    Ok(ParquetLayout {
        kind: LayoutKind::Columns,
        path: path.to_path_buf(),
        file_bytes: path.metadata()?.len(),
    })
}

pub(crate) fn write_list_parquet(
    path: &Path,
    layout: &PackedLayout,
    rows: usize,
    batch_rows: usize,
) -> Result<ParquetLayout, Box<dyn std::error::Error>> {
    let mut builder = ListBuilder::new(UInt8Builder::with_capacity(rows * layout.stride))
        .with_field(Arc::new(Field::new("element", DataType::UInt8, false)));
    for row in layout.codes.chunks_exact(layout.stride).take(rows) {
        builder.values().append_slice(row);
        builder.append(true);
    }
    let codes = builder.finish();
    let mut fields = vec![Field::new("code", codes.data_type().clone(), false)];
    let mut arrays = vec![Arc::new(codes) as ArrayRef];
    append_lvq_columns(
        &mut fields,
        &mut arrays,
        layout.offsets.as_deref(),
        layout.scales.as_deref(),
    );
    write_batch(path, fields, arrays, rows, batch_rows)?;
    Ok(ParquetLayout {
        kind: LayoutKind::List,
        path: path.to_path_buf(),
        file_bytes: path.metadata()?.len(),
    })
}

pub(crate) fn write_fixed_binary_parquet(
    path: &Path,
    layout: &PackedLayout,
    rows: usize,
    batch_rows: usize,
) -> Result<ParquetLayout, Box<dyn std::error::Error>> {
    let mut builder = FixedSizeBinaryBuilder::with_capacity(rows, layout.stride as i32);
    for row in layout.codes.chunks_exact(layout.stride).take(rows) {
        builder.append_value(row)?;
    }
    let mut fields = vec![Field::new(
        "code",
        DataType::FixedSizeBinary(layout.stride as i32),
        false,
    )];
    let mut arrays = vec![Arc::new(builder.finish()) as ArrayRef];
    append_lvq_columns(
        &mut fields,
        &mut arrays,
        layout.offsets.as_deref(),
        layout.scales.as_deref(),
    );
    write_batch(path, fields, arrays, rows, batch_rows)?;
    Ok(ParquetLayout {
        kind: LayoutKind::FixedBinary,
        path: path.to_path_buf(),
        file_bytes: path.metadata()?.len(),
    })
}

fn append_lvq_columns(
    fields: &mut Vec<Field>,
    arrays: &mut Vec<ArrayRef>,
    offsets: Option<&[f32]>,
    scales: Option<&[f32]>,
) {
    if let (Some(offsets), Some(scales)) = (offsets, scales) {
        fields.push(Field::new("offset", DataType::Float32, false));
        fields.push(Field::new("scale", DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from(offsets.to_vec())));
        arrays.push(Arc::new(Float32Array::from(scales.to_vec())));
    }
}

fn write_batch(
    path: &Path,
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
    rows: usize,
    batch_rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    if batch.num_rows() != rows {
        return Err("layout row count mismatch".into());
    }
    let properties = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_encoding(Encoding::PLAIN)
        .set_write_batch_size(batch_rows)
        .set_max_row_group_row_count(Some(batch_rows))
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(crate) fn list_array(batch: &RecordBatch) -> &ListArray {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("LIST<UINT8> code column")
}

pub(crate) fn fixed_binary_array(batch: &RecordBatch) -> &FixedSizeBinaryArray {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("FIXED_LEN_BYTE_ARRAY code column")
}
