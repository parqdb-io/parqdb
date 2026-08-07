#![allow(clippy::too_many_arguments)]

pub(crate) fn backend_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        return "avx512f+avx512bw";
    }
    "scalar"
}

pub(crate) fn distance_columns<'a, F>(
    bits: u8,
    column_count: usize,
    column: F,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    row_offsets: Option<&[f32]>,
    row_scales: Option<&[f32]>,
    distances: &mut [f32],
) where
    F: Fn(usize) -> &'a [u8] + Copy,
{
    distances.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        // SAFETY: runtime detection covers every target feature used by the kernel.
        unsafe {
            columns_avx512(
                bits,
                column_count,
                column,
                query,
                global_lower,
                global_scale,
                row_offsets,
                row_scales,
                distances,
            );
        }
        return;
    }
    columns_scalar(
        bits,
        column_count,
        column,
        query,
        global_lower,
        global_scale,
        row_offsets,
        row_scales,
        distances,
    );
}

pub(crate) fn distance_rows(
    bits: u8,
    codes: &[u8],
    stride: usize,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    row_offsets: Option<&[f32]>,
    row_scales: Option<&[f32]>,
    distances: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        // SAFETY: runtime detection covers every target feature used by the kernel.
        unsafe {
            rows_avx512(
                bits,
                codes,
                stride,
                query,
                global_lower,
                global_scale,
                row_offsets,
                row_scales,
                distances,
            );
        }
        return;
    }
    rows_scalar(
        bits,
        codes,
        stride,
        query,
        global_lower,
        global_scale,
        row_offsets,
        row_scales,
        distances,
    );
}

fn columns_scalar<'a, F>(
    bits: u8,
    column_count: usize,
    column: F,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    row_offsets: Option<&[f32]>,
    row_scales: Option<&[f32]>,
    distances: &mut [f32],
) where
    F: Fn(usize) -> &'a [u8] + Copy,
{
    let lvq = row_offsets.zip(row_scales);
    for code_column in 0..column_count {
        let codes = column(code_column);
        let first_dim = if bits == 4 {
            code_column * 2
        } else {
            code_column
        };
        for row in 0..distances.len() {
            let code = if bits == 4 {
                codes[row] & 0x0f
            } else {
                codes[row]
            };
            let decoded = decode_scalar(code, row, first_dim, global_lower, global_scale, lvq);
            let delta = query[first_dim] - decoded;
            distances[row] += delta * delta;
        }
        if bits == 4 && first_dim + 1 < query.len() {
            for row in 0..distances.len() {
                let code = codes[row] >> 4;
                let decoded =
                    decode_scalar(code, row, first_dim + 1, global_lower, global_scale, lvq);
                let delta = query[first_dim + 1] - decoded;
                distances[row] += delta * delta;
            }
        }
    }
}

fn rows_scalar(
    bits: u8,
    codes: &[u8],
    stride: usize,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    row_offsets: Option<&[f32]>,
    row_scales: Option<&[f32]>,
    distances: &mut [f32],
) {
    let lvq = row_offsets.zip(row_scales);
    for (row, distance) in distances.iter_mut().enumerate() {
        let row_codes = &codes[row * stride..(row + 1) * stride];
        let mut sum = 0.0;
        for dim in 0..query.len() {
            let code = if bits == 4 {
                let byte = row_codes[dim / 2];
                if dim & 1 == 0 { byte & 0x0f } else { byte >> 4 }
            } else {
                row_codes[dim]
            };
            let decoded = decode_scalar(code, row, dim, global_lower, global_scale, lvq);
            let delta = query[dim] - decoded;
            sum += delta * delta;
        }
        *distance = sum;
    }
}

#[inline]
fn decode_scalar(
    code: u8,
    row: usize,
    dim: usize,
    global_lower: &[f32],
    global_scale: &[f32],
    lvq: Option<(&[f32], &[f32])>,
) -> f32 {
    if let Some((offsets, scales)) = lvq {
        offsets[row] + scales[row] * f32::from(code)
    } else {
        global_lower[dim] + global_scale[dim] * f32::from(code)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn columns_avx512<'a, F>(
    bits: u8,
    column_count: usize,
    column: F,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    row_offsets: Option<&[f32]>,
    row_scales: Option<&[f32]>,
    distances: &mut [f32],
) where
    F: Fn(usize) -> &'a [u8] + Copy,
{
    use std::arch::x86_64::*;

    let lvq = row_offsets.zip(row_scales);
    let lanes = 16;
    let vectorized_rows = distances.len() / lanes * lanes;
    for code_column in 0..column_count {
        let codes = column(code_column);
        let first_dim = if bits == 4 {
            code_column * 2
        } else {
            code_column
        };
        let mut row = 0;
        while row < vectorized_rows {
            let bytes = unsafe { _mm_loadu_si128(codes.as_ptr().add(row).cast()) };
            let unpacked = _mm512_cvtepu8_epi32(bytes);
            let first_codes = if bits == 4 {
                _mm512_and_si512(unpacked, _mm512_set1_epi32(0x0f))
            } else {
                unpacked
            };
            let mut accumulated = unsafe { _mm512_loadu_ps(distances.as_ptr().add(row)) };
            accumulated = unsafe {
                accumulate_column_avx512(
                    first_codes,
                    first_dim,
                    row,
                    query,
                    global_lower,
                    global_scale,
                    lvq,
                    accumulated,
                )
            };
            if bits == 4 && first_dim + 1 < query.len() {
                let second_codes = _mm512_srli_epi32(unpacked, 4);
                accumulated = unsafe {
                    accumulate_column_avx512(
                        second_codes,
                        first_dim + 1,
                        row,
                        query,
                        global_lower,
                        global_scale,
                        lvq,
                        accumulated,
                    )
                };
            }
            unsafe { _mm512_storeu_ps(distances.as_mut_ptr().add(row), accumulated) };
            row += lanes;
        }
        for row in vectorized_rows..distances.len() {
            let first_code = if bits == 4 {
                codes[row] & 0x0f
            } else {
                codes[row]
            };
            let decoded =
                decode_scalar(first_code, row, first_dim, global_lower, global_scale, lvq);
            let delta = query[first_dim] - decoded;
            distances[row] += delta * delta;
            if bits == 4 && first_dim + 1 < query.len() {
                let decoded = decode_scalar(
                    codes[row] >> 4,
                    row,
                    first_dim + 1,
                    global_lower,
                    global_scale,
                    lvq,
                );
                let delta = query[first_dim + 1] - decoded;
                distances[row] += delta * delta;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_column_avx512(
    codes: std::arch::x86_64::__m512i,
    dim: usize,
    row: usize,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    lvq: Option<(&[f32], &[f32])>,
    accumulated: std::arch::x86_64::__m512,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;

    let codes = _mm512_cvtepi32_ps(codes);
    let decoded = if let Some((offsets, scales)) = lvq {
        let offsets = unsafe { _mm512_loadu_ps(offsets.as_ptr().add(row)) };
        let scales = unsafe { _mm512_loadu_ps(scales.as_ptr().add(row)) };
        _mm512_fmadd_ps(scales, codes, offsets)
    } else {
        _mm512_fmadd_ps(
            _mm512_set1_ps(global_scale[dim]),
            codes,
            _mm512_set1_ps(global_lower[dim]),
        )
    };
    let delta = _mm512_sub_ps(_mm512_set1_ps(query[dim]), decoded);
    _mm512_fmadd_ps(delta, delta, accumulated)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn rows_avx512(
    bits: u8,
    codes: &[u8],
    stride: usize,
    query: &[f32],
    global_lower: &[f32],
    global_scale: &[f32],
    row_offsets: Option<&[f32]>,
    row_scales: Option<&[f32]>,
    distances: &mut [f32],
) {
    use std::arch::x86_64::*;

    let lvq = row_offsets.zip(row_scales);
    for (row, distance) in distances.iter_mut().enumerate() {
        let row_codes = &codes[row * stride..(row + 1) * stride];
        let mut accumulated = _mm512_setzero_ps();
        let mut dim = 0;
        while dim + 16 <= query.len() {
            let unpacked = if bits == 8 {
                let bytes = unsafe { _mm_loadu_si128(row_codes.as_ptr().add(dim).cast()) };
                _mm512_cvtepu8_epi32(bytes)
            } else {
                let bytes = unsafe { _mm_loadl_epi64(row_codes.as_ptr().add(dim / 2).cast()) };
                let mask = _mm_set1_epi8(0x0f);
                let low = _mm_and_si128(bytes, mask);
                let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
                _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low, high))
            };
            let code_values = _mm512_cvtepi32_ps(unpacked);
            let decoded = if let Some((offsets, scales)) = lvq {
                _mm512_fmadd_ps(
                    _mm512_set1_ps(scales[row]),
                    code_values,
                    _mm512_set1_ps(offsets[row]),
                )
            } else {
                let lower = unsafe { _mm512_loadu_ps(global_lower.as_ptr().add(dim)) };
                let scale = unsafe { _mm512_loadu_ps(global_scale.as_ptr().add(dim)) };
                _mm512_fmadd_ps(scale, code_values, lower)
            };
            let query_values = unsafe { _mm512_loadu_ps(query.as_ptr().add(dim)) };
            let delta = _mm512_sub_ps(query_values, decoded);
            accumulated = _mm512_fmadd_ps(delta, delta, accumulated);
            dim += 16;
        }

        let mut lanes = [0.0_f32; 16];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), accumulated) };
        let mut sum = lanes.into_iter().sum::<f32>();
        for dim in dim..query.len() {
            let code = if bits == 4 {
                let byte = row_codes[dim / 2];
                if dim & 1 == 0 { byte & 0x0f } else { byte >> 4 }
            } else {
                row_codes[dim]
            };
            let decoded = decode_scalar(code, row, dim, global_lower, global_scale, lvq);
            let delta = query[dim] - decoded;
            sum += delta * delta;
        }
        *distance = sum;
    }
}
