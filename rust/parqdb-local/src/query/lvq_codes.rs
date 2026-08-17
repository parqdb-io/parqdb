use arrow::array::{Array, BinaryArray, BinaryViewArray};
use arrow::datatypes::DataType;
use arrow_data::{ByteView, MAX_INLINE_VIEW_LEN};
use datafusion::common::{DataFusionError, Result};
use parqdb_kernels::{KernelError, LvqCodeRows};

pub(crate) fn lvq_code_rows(codes: &dyn Array, code_size: usize) -> Result<LvqCodeRows<'_>> {
    if codes.null_count() != 0 {
        return Err(DataFusionError::Execution(
            "LVQ posting columns must not contain nulls".into(),
        ));
    }
    match codes.data_type() {
        DataType::Binary => binary_lvq_codes(codes, code_size),
        DataType::BinaryView => binary_view_lvq_codes(codes, code_size),
        _ => Err(DataFusionError::Execution("invalid LVQ code column".into())),
    }
}

fn binary_lvq_codes(codes: &dyn Array, code_size: usize) -> Result<LvqCodeRows<'_>> {
    let array = codes
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("Binary arrays have the matching concrete type");
    if array
        .value_offsets()
        .windows(2)
        .any(|offsets| usize::try_from(offsets[1] - offsets[0]) != Ok(code_size))
    {
        return Err(DataFusionError::Execution(
            "LVQ code row has the wrong length".into(),
        ));
    }
    let offsets = array.value_offsets();
    let start = usize::try_from(offsets[0]).expect("validated Binary offset");
    let end = usize::try_from(offsets[array.len()]).expect("validated Binary offset");
    LvqCodeRows::try_from_packed(&array.value_data()[start..end], code_size)
        .map_err(|error| lvq_codes_error(&error))
}

fn binary_view_lvq_codes(codes: &dyn Array, code_size: usize) -> Result<LvqCodeRows<'_>> {
    let array = codes
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .expect("BinaryView arrays have the matching concrete type");
    binary_view_lvq_spans(array, code_size)
}

fn binary_view_lvq_spans(array: &BinaryViewArray, code_size: usize) -> Result<LvqCodeRows<'_>> {
    let mut codes = LvqCodeRows::try_new(code_size).map_err(|error| lvq_codes_error(&error))?;
    let mut row = 0;
    while row < array.len() {
        let raw = array.views()[row];
        let first = ByteView::from(raw);
        require_code_size(first, code_size)?;
        if first.length <= MAX_INLINE_VIEW_LEN {
            codes
                .push_span(array.value(row), 1, code_size)
                .map_err(|error| lvq_codes_error(&error))?;
            row += 1;
            continue;
        }

        let mut rows = 1;
        let mut stride = code_size;
        if row + 1 < array.len() {
            let candidate = ByteView::from(array.views()[row + 1]);
            require_code_size(candidate, code_size)?;
            if candidate.buffer_index == first.buffer_index {
                let first_offset = usize::try_from(first.offset).expect("u32 offsets fit usize");
                let candidate_offset =
                    usize::try_from(candidate.offset).expect("u32 offsets fit usize");
                if let Some(candidate_stride) = candidate_offset.checked_sub(first_offset)
                    && candidate_stride >= code_size
                {
                    stride = candidate_stride;
                }
            }
        }
        let first_offset = usize::try_from(first.offset).expect("u32 offsets fit usize");
        while row + rows < array.len() {
            let candidate = ByteView::from(array.views()[row + rows]);
            require_code_size(candidate, code_size)?;
            let expected_offset = first_offset
                .checked_add(rows.checked_mul(stride).ok_or_else(span_overflow)?)
                .ok_or_else(span_overflow)?;
            if candidate.buffer_index != first.buffer_index
                || usize::try_from(candidate.offset).expect("u32 offsets fit usize")
                    != expected_offset
            {
                break;
            }
            rows += 1;
        }
        let buffer = &array.data_buffers()
            [usize::try_from(first.buffer_index).expect("u32 buffer indexes fit usize")];
        let start = usize::try_from(first.offset).expect("u32 offsets fit usize");
        let span_bytes = (rows - 1)
            .checked_mul(stride)
            .and_then(|bytes| bytes.checked_add(code_size))
            .ok_or_else(span_overflow)?;
        let end = start.checked_add(span_bytes).ok_or_else(span_overflow)?;
        codes
            .push_span(&buffer[start..end], rows, stride)
            .map_err(|error| lvq_codes_error(&error))?;
        row += rows;
    }
    Ok(codes)
}

fn require_code_size(view: ByteView, code_size: usize) -> Result<()> {
    if usize::try_from(view.length).expect("u32 BinaryView lengths fit usize") != code_size {
        return Err(DataFusionError::Execution(
            "LVQ code row has the wrong length".into(),
        ));
    }
    Ok(())
}

fn span_overflow() -> DataFusionError {
    DataFusionError::Execution("LVQ code span overflows usize".into())
}

fn lvq_codes_error(error: &KernelError) -> DataFusionError {
    DataFusionError::Execution(error.to_string())
}
