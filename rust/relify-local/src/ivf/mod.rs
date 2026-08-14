//! Arrow-level IVF relation validation and access.

#[cfg(test)]
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, LargeListArray, ListArray,
};
#[cfg(test)]
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
#[cfg(test)]
use arrow::row::{RowConverter, SortField};

use crate::{Error, Result};
use relify_kernels::detect;

pub(crate) fn borrow_vectors(array: &ArrayRef) -> Result<(&[f32], usize)> {
    borrow_vectors_with_element_nullability(array, false)
}

pub(crate) fn borrow_vectors_allow_nullable_elements(array: &ArrayRef) -> Result<(&[f32], usize)> {
    borrow_vectors_with_element_nullability(array, true)
}

fn borrow_vectors_with_element_nullability(
    array: &ArrayRef,
    allow_nullable_elements: bool,
) -> Result<(&[f32], usize)> {
    match array.data_type() {
        DataType::List(field) if field.data_type() == &DataType::Float32 => {
            if field.is_nullable() && !allow_nullable_elements {
                return Err(Error::InvalidSchema(
                    "vector list elements must be required".into(),
                ));
            }
            borrow_list_vectors(
                array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .expect("List datatype must downcast"),
            )
        }
        DataType::LargeList(field) if field.data_type() == &DataType::Float32 => {
            if field.is_nullable() && !allow_nullable_elements {
                return Err(Error::InvalidSchema(
                    "vector list elements must be required".into(),
                ));
            }
            borrow_large_list_vectors(
                array
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .expect("LargeList datatype must downcast"),
            )
        }
        DataType::FixedSizeList(field, _) if field.data_type() == &DataType::Float32 => {
            if field.is_nullable() && !allow_nullable_elements {
                return Err(Error::InvalidSchema(
                    "vector list elements must be required".into(),
                ));
            }
            borrow_fixed_list_vectors(
                array
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .expect("FixedSizeList datatype must downcast"),
            )
        }
        other => Err(Error::InvalidSchema(format!(
            "vector column must have type list<float>, received {other}"
        ))),
    }
}

pub(crate) fn borrow_source_vectors<'a>(
    table: &'a RecordBatch,
    field_name: &str,
) -> Result<(&'a [f32], usize)> {
    let index = table
        .schema()
        .index_of(field_name)
        .map_err(|_| Error::InvalidSchema(format!("vector column not found: {field_name}")))?;
    borrow_vectors_allow_nullable_elements(table.column(index))
}

#[cfg(test)]
pub(crate) fn normalize_key_array(array: &ArrayRef) -> Result<ArrayRef> {
    if array.null_count() != 0 {
        return Err(Error::InvalidSchema(
            "source key columns must not contain nulls".into(),
        ));
    }
    match array.data_type() {
        DataType::Boolean
        | DataType::Int32
        | DataType::Int64
        | DataType::Binary
        | DataType::FixedSizeBinary(_)
        | DataType::Utf8
        | DataType::Date32 => Ok(Arc::clone(array)),
        DataType::Utf8View | DataType::LargeUtf8 => Ok(cast(array, &DataType::Utf8)?),
        DataType::BinaryView | DataType::LargeBinary => Ok(cast(array, &DataType::Binary)?),
        other => Err(Error::InvalidSchema(format!(
            "unsupported source key type: {other}"
        ))),
    }
}

#[cfg(test)]
pub(crate) fn source_key_arrays(table: &RecordBatch, fields: &[String]) -> Result<Vec<ArrayRef>> {
    fields
        .iter()
        .map(|name| {
            let index = table
                .schema()
                .index_of(name)
                .map_err(|_| Error::InvalidSchema(format!("key column not found: {name}")))?;
            normalize_key_array(table.column(index))
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn validate_unique_keys(arrays: &[ArrayRef]) -> Result<()> {
    let keys = row_keys(arrays)?;
    let mut unique = HashSet::with_capacity(keys.len());
    for key in keys {
        if !unique.insert(key) {
            return Err(Error::InvalidSchema(
                "source key tuple must be unique".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn row_keys(arrays: &[ArrayRef]) -> Result<Vec<Vec<u8>>> {
    let fields = arrays
        .iter()
        .map(|array| SortField::new(array.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(arrays)?;
    Ok(rows.iter().map(|row| row.as_ref().to_vec()).collect())
}

pub(crate) fn read_centroids(
    table: &RecordBatch,
    nlist: usize,
    dimension: usize,
) -> Result<Vec<f32>> {
    let centroid_index = table
        .schema()
        .index_of("centroid")
        .map_err(|_| Error::InvalidSchema("ivf_centroids is missing centroid".into()))?;
    let cids = required_int32(table, "cid", "ivf_centroids")?;
    if table.schema().field(centroid_index).is_nullable() {
        return Err(Error::InvalidSchema(
            "ivf_centroids.centroid must be required".into(),
        ));
    }
    let (values, actual_dimension) = borrow_vectors(table.column(centroid_index))?;
    if table.num_rows() != nlist || actual_dimension != dimension {
        return Err(Error::InvalidSchema(
            "ivf_centroids shape does not match metadata".into(),
        ));
    }
    let mut reordered = vec![0.0_f32; nlist * dimension];
    let mut seen = vec![false; nlist];
    for row in 0..table.num_rows() {
        let cid = usize::try_from(cids.value(row))
            .map_err(|_| Error::InvalidSchema("centroid cid must be non-negative".into()))?;
        if cid >= nlist || seen[cid] {
            return Err(Error::InvalidSchema(
                "centroid cid must be unique and in range".into(),
            ));
        }
        seen[cid] = true;
        reordered[cid * dimension..(cid + 1) * dimension]
            .copy_from_slice(&values[row * dimension..(row + 1) * dimension]);
    }
    Ok(reordered)
}

pub(crate) fn select_clusters(
    query: &[f32],
    centroids: &[f32],
    dimension: usize,
    nprobe: usize,
) -> Vec<bool> {
    let nlist = centroids.len() / dimension;
    if nprobe == nlist {
        return vec![true; nlist];
    }
    let distance_kernel = detect();
    let mut distance_values = vec![0.0; nlist];
    distance_kernel.squared_l2_rows(centroids, query, &mut distance_values);
    let mut distances = distance_values
        .into_iter()
        .enumerate()
        .map(|(cid, distance)| (distance, cid))
        .collect::<Vec<_>>();
    let compare = |left: &(f32, usize), right: &(f32, usize)| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    distances.select_nth_unstable_by(nprobe, compare);
    let mut selected = vec![false; distances.len()];
    for &(_, cid) in &distances[..nprobe] {
        selected[cid] = true;
    }
    selected
}

#[cfg(test)]
pub(crate) fn source_rows_by_key(source_keys: &[ArrayRef]) -> Result<HashMap<Vec<u8>, usize>> {
    let keys = row_keys(source_keys)?;
    let mut rows = HashMap::with_capacity(keys.len());
    for (row, key) in keys.into_iter().enumerate() {
        if rows.insert(key, row).is_some() {
            return Err(Error::InvalidSchema(
                "source key tuple must be unique".into(),
            ));
        }
    }
    Ok(rows)
}

#[cfg(test)]
pub(crate) fn candidate_source_rows(
    table: &RecordBatch,
    selected_clusters: &[bool],
    nlist: usize,
    source_keys: &[ArrayRef],
    source_by_key: &HashMap<Vec<u8>, usize>,
) -> Result<Vec<(Vec<u8>, usize, usize)>> {
    let cids = required_int32(table, "cid", "ivf_postings")?;
    let posting_keys = source_keys
        .iter()
        .enumerate()
        .map(|(index, source_key)| {
            let name = format!("key_{}", index + 1);
            let column = table
                .schema()
                .index_of(&name)
                .map_err(|_| Error::InvalidSchema(format!("ivf_postings is missing {name}")))?;
            if table.schema().field(column).is_nullable() {
                return Err(Error::InvalidSchema(format!(
                    "ivf_postings.{name} must be required"
                )));
            }
            let posting_key = normalize_key_array(table.column(column))?;
            if posting_key.data_type() != source_key.data_type() {
                return Err(Error::InvalidSchema(format!(
                    "ivf_postings.{name} type does not match source key type"
                )));
            }
            Ok(posting_key)
        })
        .collect::<Result<Vec<_>>>()?;
    let keys = row_keys(&posting_keys)?;
    let mut seen = HashSet::with_capacity(keys.len());
    let mut candidates = Vec::new();
    for (row, key) in keys.into_iter().enumerate() {
        let cid = usize::try_from(cids.value(row))
            .map_err(|_| Error::InvalidSchema("posting cid must be non-negative".into()))?;
        let source_row = source_by_key.get(&key).copied();
        if cid >= nlist
            || !selected_clusters.get(cid).copied().unwrap_or(false)
            || !seen.insert(key.clone())
            || source_row.is_none()
        {
            return Err(Error::InvalidSchema(
                "posting cid/key values violate IVF source-resolution constraints".into(),
            ));
        }
        candidates.push((key, source_row.expect("source row was checked"), row));
    }
    Ok(candidates)
}

fn borrow_list_vectors(array: &ListArray) -> Result<(&[f32], usize)> {
    let offsets = array.value_offsets();
    borrow_contiguous_list_vectors(
        array.len(),
        array.null_count(),
        offsets.len(),
        |index| {
            usize::try_from(offsets[index])
                .map_err(|_| Error::InvalidSchema("vector list offset must be non-negative".into()))
        },
        array.values(),
    )
}

fn borrow_large_list_vectors(array: &LargeListArray) -> Result<(&[f32], usize)> {
    let offsets = array.value_offsets();
    borrow_contiguous_list_vectors(
        array.len(),
        array.null_count(),
        offsets.len(),
        |index| {
            usize::try_from(offsets[index])
                .map_err(|_| Error::InvalidSchema("vector list offset must fit usize".into()))
        },
        array.values(),
    )
}

fn borrow_fixed_list_vectors(array: &FixedSizeListArray) -> Result<(&[f32], usize)> {
    let dimension = usize::try_from(array.value_length())
        .map_err(|_| Error::InvalidSchema("invalid fixed-size vector dimension".into()))?;
    if array.is_empty() || dimension == 0 {
        return Err(Error::InvalidSchema(
            "vector dimension must be positive".into(),
        ));
    }
    if array.null_count() != 0 {
        return Err(Error::InvalidSchema(
            "vector column must not contain nulls".into(),
        ));
    }
    let value_count = array
        .len()
        .checked_mul(dimension)
        .ok_or_else(|| Error::InvalidSchema("vector value count overflows usize".into()))?;
    let value_offset = array
        .offset()
        .checked_mul(dimension)
        .ok_or_else(|| Error::InvalidSchema("vector value offset overflows usize".into()))?;
    let vectors = borrow_float_values(array.values(), value_offset, value_count)?;
    Ok((vectors, dimension))
}

fn borrow_contiguous_list_vectors(
    rows: usize,
    null_count: usize,
    offset_count: usize,
    offset_at: impl Fn(usize) -> Result<usize>,
    values: &ArrayRef,
) -> Result<(&[f32], usize)> {
    if rows == 0 || offset_count != rows + 1 {
        return Err(Error::InvalidSchema(
            "source table must contain at least one vector".into(),
        ));
    }
    if null_count != 0 {
        return Err(Error::InvalidSchema(
            "vector column must not contain nulls".into(),
        ));
    }
    let first_offset = offset_at(0)?;
    let dimension = offset_at(1)?
        .checked_sub(first_offset)
        .ok_or_else(|| Error::InvalidSchema("vector list offsets must be ordered".into()))?;
    if dimension == 0 {
        return Err(Error::InvalidSchema(
            "all vectors must have the same positive dimension".into(),
        ));
    }
    let mut previous = first_offset;
    for index in 1..=rows {
        let current = offset_at(index)?;
        if current
            .checked_sub(previous)
            .is_none_or(|length| length != dimension)
        {
            return Err(Error::InvalidSchema(
                "all vectors must have the same positive dimension".into(),
            ));
        }
        previous = current;
    }
    let value_count = rows
        .checked_mul(dimension)
        .ok_or_else(|| Error::InvalidSchema("vector value count overflows usize".into()))?;
    let vectors = borrow_float_values(values, first_offset, value_count)?;
    Ok((vectors, dimension))
}

fn borrow_float_values(array: &ArrayRef, offset: usize, len: usize) -> Result<&[f32]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidSchema("vector value range overflows usize".into()))?;
    if end > array.len() {
        return Err(Error::InvalidSchema(
            "vector list offsets exceed the child array".into(),
        ));
    }
    let values = array
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| Error::InvalidSchema("vector elements must be float32".into()))?;
    let has_nulls = values
        .nulls()
        .is_some_and(|nulls| nulls.slice(offset, len).null_count() != 0);
    let values = &values.values()[offset..end];
    if has_nulls || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidSchema(
            "every vector must contain finite, non-null float values".into(),
        ));
    }
    Ok(values)
}

fn required_int32<'a>(
    table: &'a RecordBatch,
    name: &str,
    relation: &str,
) -> Result<&'a Int32Array> {
    let index = table
        .schema()
        .index_of(name)
        .map_err(|_| Error::InvalidSchema(format!("{relation} is missing {name}")))?;
    if table.schema().field(index).is_nullable() {
        return Err(Error::InvalidSchema(format!(
            "{relation}.{name} must be required"
        )));
    }
    let array: &Int32Array = table
        .column(index)
        .as_any()
        .downcast_ref()
        .ok_or_else(|| Error::InvalidSchema(format!("{relation}.{name} must be int")))?;
    if array.null_count() != 0 {
        return Err(Error::InvalidSchema(format!(
            "{relation}.{name} must not contain nulls"
        )));
    }
    Ok(array)
}

#[cfg(test)]
mod tests;
