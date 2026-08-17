//! Locally adaptive scalar quantization and distance kernels.

use crate::{CpuBackend, DistanceKernel, KernelError, Result, detect};

type RowScorer = fn(&LvqEncodedBatch, &[f32], usize) -> f32;

/// Number of bits used by one LVQ dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LvqBits {
    /// Two dimensions packed into each byte.
    Four,
    /// One dimension stored in each byte.
    Eight,
}

impl LvqBits {
    const fn levels(self) -> u8 {
        match self {
            Self::Four => 15,
            Self::Eight => u8::MAX,
        }
    }

    /// Returns the number of code bytes required for one vector.
    #[must_use]
    pub const fn code_size(self, dimension: usize) -> usize {
        match self {
            Self::Four => dimension.div_ceil(2),
            Self::Eight => dimension,
        }
    }
}

/// Owned row-major LVQ buffers produced by the encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct LvqEncodedBatch {
    bits: LvqBits,
    dimension: usize,
    codes: Vec<u8>,
    offsets: Vec<f32>,
    scales: Vec<f32>,
}

impl LvqEncodedBatch {
    /// Returns the quantization width.
    #[must_use]
    pub const fn bits(&self) -> LvqBits {
        self.bits
    }

    /// Returns the vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of encoded vectors.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.offsets.len()
    }

    /// Returns packed row-major codes.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Returns the per-row lower bounds.
    #[must_use]
    pub fn offsets(&self) -> &[f32] {
        &self.offsets
    }

    /// Returns the per-row quantization scales.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Consumes the batch and returns codes, offsets, and scales.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        (self.codes, self.offsets, self.scales)
    }

    /// Borrows the encoded buffers for distance evaluation.
    #[must_use]
    pub fn as_view(&self) -> LvqBatchView<'_> {
        let code_size = self.bits.code_size(self.dimension);
        LvqBatchView {
            bits: self.bits,
            dimension: self.dimension,
            codes: LvqCodeRows::from_encoded(&self.codes, code_size),
            offsets: &self.offsets,
            scales: &self.scales,
        }
    }

    /// Validates a query once for repeated random-access row scoring.
    pub fn prepare_query<'a>(&'a self, query: &'a [f32]) -> Result<LvqRowQuery<'a>> {
        if query.len() != self.dimension {
            return invalid("query vector has the wrong dimension");
        }
        if query.iter().any(|value| !value.is_finite()) {
            return invalid("query vector must contain only finite values");
        }
        let scorer = match detect().backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => lvq_squared_l2_row_scalar_dispatch,
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => lvq_squared_l2_row_scalar_dispatch,
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                {
                    lvq_squared_l2_row_avx512_dispatch
                } else {
                    lvq_squared_l2_row_avx2_dispatch
                }
            }
        };
        Ok(LvqRowQuery {
            batch: self,
            query,
            scorer,
        })
    }

    /// Returns the retained heap size of the encoded buffers.
    #[must_use]
    pub fn resident_size(&self) -> usize {
        self.codes.capacity()
            + (self.offsets.capacity() + self.scales.capacity()) * std::mem::size_of::<f32>()
    }
}

/// A validated exact query over an owned LVQ batch.
pub struct LvqRowQuery<'a> {
    batch: &'a LvqEncodedBatch,
    query: &'a [f32],
    scorer: RowScorer,
}

impl LvqRowQuery<'_> {
    /// Computes the squared-L2 distance to one encoded row.
    pub fn squared_l2(&self, row: usize) -> Result<f32> {
        if row >= self.batch.row_count() {
            return invalid("LVQ row is out of bounds");
        }
        Ok((self.scorer)(self.batch, self.query, row))
    }

    /// Computes squared-L2 distances to four encoded rows.
    pub fn squared_l2_four(&self, rows: [usize; 4]) -> Result<[f32; 4]> {
        if rows.iter().any(|row| *row >= self.batch.row_count()) {
            return invalid("LVQ row is out of bounds");
        }
        Ok(rows.map(|row| (self.scorer)(self.batch, self.query, row)))
    }
}

#[derive(Debug, Clone, Copy)]
struct LvqCodeSpan<'a> {
    values: &'a [u8],
    rows: usize,
    stride: usize,
}

#[derive(Debug, Default)]
enum LvqCodeSpans<'a> {
    #[default]
    Empty,
    One(LvqCodeSpan<'a>),
    Many(Vec<LvqCodeSpan<'a>>),
}

/// Checked borrowed LVQ code rows, optionally split across strided spans.
///
/// Storage adapters append spans through [`Self::push_span`]. The checked
/// representation lets SIMD kernels consume non-contiguous storage without
/// trusting an external behavioral contract.
#[derive(Debug)]
pub struct LvqCodeRows<'a> {
    spans: LvqCodeSpans<'a>,
    row_count: usize,
    code_size: usize,
}

impl<'a> LvqCodeRows<'a> {
    fn from_encoded(values: &'a [u8], code_size: usize) -> Self {
        debug_assert!(code_size > 0);
        debug_assert!(values.len().is_multiple_of(code_size));
        let row_count = values.len() / code_size;
        let spans = if row_count == 0 {
            LvqCodeSpans::Empty
        } else {
            LvqCodeSpans::One(LvqCodeSpan {
                values,
                rows: row_count,
                stride: code_size,
            })
        };
        Self {
            spans,
            row_count,
            code_size,
        }
    }

    /// Creates an empty collection for rows of `code_size` bytes.
    pub fn try_new(code_size: usize) -> Result<Self> {
        if code_size == 0 {
            return invalid("LVQ code size must be positive");
        }
        Ok(Self {
            spans: LvqCodeSpans::Empty,
            row_count: 0,
            code_size,
        })
    }

    /// Validates contiguous row-major code bytes.
    pub fn try_from_packed(values: &'a [u8], code_size: usize) -> Result<Self> {
        let mut codes = Self::try_new(code_size)?;
        if !values.len().is_multiple_of(code_size) {
            return invalid("LVQ code buffer does not contain complete rows");
        }
        let rows = values.len() / code_size;
        if rows != 0 {
            codes.push_span(values, rows, code_size)?;
        }
        Ok(codes)
    }

    /// Appends `rows` codes separated by `stride` bytes.
    ///
    /// `values` must contain exactly the bytes from the first code through the
    /// final code, including any bytes between adjacent codes.
    pub fn push_span(&mut self, values: &'a [u8], rows: usize, stride: usize) -> Result<()> {
        if rows == 0 {
            return invalid("LVQ code spans must contain at least one row");
        }
        if stride < self.code_size {
            return invalid("LVQ code span stride is smaller than one code row");
        }
        let expected_bytes = (rows - 1)
            .checked_mul(stride)
            .and_then(|bytes| bytes.checked_add(self.code_size))
            .ok_or_else(|| KernelError("LVQ code span shape overflows usize".into()))?;
        if values.len() != expected_bytes {
            return invalid("LVQ code span does not match its declared shape");
        }
        self.row_count = self
            .row_count
            .checked_add(rows)
            .ok_or_else(|| KernelError("LVQ code row count overflows usize".into()))?;
        let span = LvqCodeSpan {
            values,
            rows,
            stride,
        };
        self.spans = match std::mem::take(&mut self.spans) {
            LvqCodeSpans::Empty => LvqCodeSpans::One(span),
            LvqCodeSpans::One(first) => LvqCodeSpans::Many(vec![first, span]),
            LvqCodeSpans::Many(mut spans) => {
                spans.push(span);
                LvqCodeSpans::Many(spans)
            }
        };
        Ok(())
    }

    /// Returns the number of code rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    fn spans(&self) -> impl Iterator<Item = &LvqCodeSpan<'a>> {
        let (one, many) = match &self.spans {
            LvqCodeSpans::Empty => (None, &[][..]),
            LvqCodeSpans::One(span) => (Some(span), &[][..]),
            LvqCodeSpans::Many(spans) => (None, spans.as_slice()),
        };
        one.into_iter().chain(many)
    }

    fn row(&self, mut row: usize) -> &[u8] {
        for span in self.spans() {
            if row < span.rows {
                let offset = row * span.stride;
                return &span.values[offset..offset + self.code_size];
            }
            row -= span.rows;
        }
        panic!("LVQ code row is out of bounds");
    }

    fn for_each_row(&self, mut visitor: impl FnMut(usize, &[u8])) {
        let mut row = 0;
        for span in self.spans() {
            for offset in (0..span.rows).map(|index| index * span.stride) {
                visitor(row, &span.values[offset..offset + self.code_size]);
                row += 1;
            }
        }
        debug_assert_eq!(row, self.row_count);
    }
}

/// Validated borrowed LVQ buffers.
#[derive(Debug)]
pub struct LvqBatchView<'a> {
    bits: LvqBits,
    dimension: usize,
    codes: LvqCodeRows<'a>,
    offsets: &'a [f32],
    scales: &'a [f32],
}

impl<'a> LvqBatchView<'a> {
    /// Validates and borrows externally stored LVQ columns.
    pub fn try_new(
        bits: LvqBits,
        dimension: usize,
        codes: &'a [u8],
        offsets: &'a [f32],
        scales: &'a [f32],
    ) -> Result<Self> {
        if dimension == 0 {
            return invalid("vector dimension must be positive");
        }
        if offsets.len() != scales.len() {
            return invalid("LVQ offset and scale row counts differ");
        }
        let code_size = bits.code_size(dimension);
        let expected_codes = offsets
            .len()
            .checked_mul(code_size)
            .ok_or_else(|| KernelError("LVQ code buffer shape overflows usize".into()))?;
        if codes.len() != expected_codes {
            return invalid("LVQ code buffer does not match its declared shape");
        }

        Self::try_new_rows(
            bits,
            dimension,
            LvqCodeRows::try_from_packed(codes, code_size)?,
            offsets,
            scales,
        )
    }

    /// Validates and borrows a row-oriented LVQ code source.
    pub fn try_new_rows(
        bits: LvqBits,
        dimension: usize,
        codes: LvqCodeRows<'a>,
        offsets: &'a [f32],
        scales: &'a [f32],
    ) -> Result<Self> {
        if dimension == 0 {
            return invalid("vector dimension must be positive");
        }
        if offsets.len() != scales.len() {
            return invalid("LVQ offset and scale row counts differ");
        }
        if codes.row_count() != offsets.len() {
            return invalid("LVQ code, offset, and scale row counts differ");
        }
        if codes.code_size != bits.code_size(dimension) {
            return invalid("LVQ code rows do not match the declared dimension");
        }

        let levels = f32::from(bits.levels());
        let mut validation_error = None;
        codes.for_each_row(|row, row_codes| {
            if validation_error.is_some() {
                return;
            }
            let offset = offsets[row];
            let scale = scales[row];
            if !offset.is_finite()
                || !scale.is_finite()
                || scale < 0.0
                || !(offset + scale * levels).is_finite()
            {
                validation_error = Some("LVQ offsets and scales must define finite values");
                return;
            }
            if scale == 0.0 && row_codes.iter().any(|code| *code != 0) {
                validation_error = Some("constant LVQ rows must contain only zero codes");
                return;
            }
            if bits == LvqBits::Four
                && !dimension.is_multiple_of(2)
                && row_codes.last().is_some_and(|code| code & 0xf0 != 0)
            {
                validation_error = Some("the unused LVQ4 high nibble must be zero");
            }
        });
        if let Some(message) = validation_error {
            return invalid(message);
        }

        Ok(Self {
            bits,
            dimension,
            codes,
            offsets,
            scales,
        })
    }

    /// Returns the quantization width.
    #[must_use]
    pub const fn bits(&self) -> LvqBits {
        self.bits
    }

    /// Returns the vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of encoded vectors.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.offsets.len()
    }

    /// Decodes one row into `output`.
    pub fn decode_row(&self, row: usize, output: &mut [f32]) -> Result<()> {
        if row >= self.row_count() {
            return invalid("LVQ row is out of bounds");
        }
        if output.len() != self.dimension {
            return invalid("LVQ decode output has the wrong dimension");
        }
        let row_codes = self.codes.row(row);
        for (dimension, value) in output.iter_mut().enumerate() {
            *value = self.offsets[row]
                + self.scales[row] * f32::from(code_at(self.bits, row_codes, dimension));
        }
        Ok(())
    }
}

/// Encodes a row-major matrix with per-row locally adaptive quantization.
pub fn encode_lvq_rows(values: &[f32], dimension: usize, bits: LvqBits) -> Result<LvqEncodedBatch> {
    if dimension == 0 {
        return invalid("vector dimension must be positive");
    }
    if !values.len().is_multiple_of(dimension) {
        return invalid("source vector matrix does not match its declared dimension");
    }
    if values.iter().any(|value| !value.is_finite()) {
        return invalid("source vectors must contain only finite values");
    }

    let rows = values.len() / dimension;
    let code_size = bits.code_size(dimension);
    let mut codes = vec![0_u8; rows * code_size];
    let mut offsets = Vec::with_capacity(rows);
    let mut scales = Vec::with_capacity(rows);
    let levels = bits.levels();

    for (row, vector) in values.chunks_exact(dimension).enumerate() {
        let offset = vector.iter().copied().fold(f32::INFINITY, f32::min);
        let upper = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = upper - offset;
        if !range.is_finite() {
            return invalid("source vector range is not finite");
        }
        let scale = range / f32::from(levels);
        offsets.push(offset);
        scales.push(scale);
        let row_codes = &mut codes[row * code_size..(row + 1) * code_size];
        if scale == 0.0 {
            continue;
        }
        for (dimension, value) in vector.iter().copied().enumerate() {
            let code = quantize_value(value, offset, range, levels);
            match bits {
                LvqBits::Four if dimension.is_multiple_of(2) => {
                    row_codes[dimension / 2] = code;
                }
                LvqBits::Four => row_codes[dimension / 2] |= code << 4,
                LvqBits::Eight => row_codes[dimension] = code,
            }
        }
    }

    Ok(LvqEncodedBatch {
        bits,
        dimension,
        codes,
        offsets,
        scales,
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantize_value(value: f32, offset: f32, range: f32, levels: u8) -> u8 {
    let code = ((f64::from(value) - f64::from(offset)) * f64::from(levels) / f64::from(range))
        .round()
        .clamp(0.0, f64::from(levels));
    // The clamp proves that the rounded value is in the complete u8 range.
    code as u8
}

impl DistanceKernel {
    #[allow(unsafe_code)]
    /// Computes squared-L2 distances from LVQ rows to one exact query vector.
    pub fn lvq_squared_l2_rows(
        self,
        batch: &LvqBatchView<'_>,
        query: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if query.len() != batch.dimension {
            return invalid("query vector has the wrong dimension");
        }
        if query.iter().any(|value| !value.is_finite()) {
            return invalid("query vector must contain only finite values");
        }
        if output.len() != batch.row_count() {
            return invalid("distance output does not match the LVQ row count");
        }

        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => lvq_squared_l2_rows_scalar(batch, query, output),
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => lvq_squared_l2_rows_scalar(batch, query, output),
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                {
                    // SAFETY: both AVX-512 features are checked at runtime and
                    // the view validates every buffer before construction.
                    unsafe { lvq_squared_l2_rows_avx512(batch, query, output) };
                } else {
                    // SAFETY: this backend is selected only after AVX2 runtime
                    // detection and the view validates all buffer shapes.
                    unsafe { lvq_squared_l2_rows_avx2(batch, query, output) };
                }
            }
        }
        Ok(())
    }

    #[allow(unsafe_code)]
    /// Computes pairwise squared-L2 distances between LVQ rows and exact query rows.
    pub fn lvq_squared_l2_pairs(
        self,
        batch: &LvqBatchView<'_>,
        queries: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let expected_values = batch
            .row_count()
            .checked_mul(batch.dimension)
            .ok_or_else(|| KernelError("query vector matrix shape overflows usize".into()))?;
        if queries.len() != expected_values {
            return invalid("query vector matrix does not match the LVQ row count");
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return invalid("query vectors must contain only finite values");
        }
        if output.len() != batch.row_count() {
            return invalid("distance output does not match the LVQ row count");
        }

        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => lvq_squared_l2_pairs_scalar(batch, queries, output),
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => lvq_squared_l2_pairs_scalar(batch, queries, output),
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                {
                    // SAFETY: both AVX-512 features are checked at runtime and
                    // the view validates every buffer before construction.
                    unsafe { lvq_squared_l2_pairs_avx512(batch, queries, output) };
                } else {
                    // SAFETY: this backend is selected only after AVX2 runtime
                    // detection and the view validates all buffer shapes.
                    unsafe { lvq_squared_l2_pairs_avx2(batch, queries, output) };
                }
            }
        }
        Ok(())
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(KernelError(message.into()))
}

fn code_at(bits: LvqBits, codes: &[u8], dimension: usize) -> u8 {
    match bits {
        LvqBits::Four => {
            let packed = codes[dimension / 2];
            if dimension.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            }
        }
        LvqBits::Eight => codes[dimension],
    }
}

fn lvq_squared_l2_rows_scalar(batch: &LvqBatchView<'_>, query: &[f32], output: &mut [f32]) {
    batch.codes.for_each_row(|row, row_codes| {
        let mut sum = 0.0_f32;
        for (dimension, query_value) in query.iter().copied().enumerate() {
            let decoded = batch.offsets[row]
                + batch.scales[row] * f32::from(code_at(batch.bits, row_codes, dimension));
            let delta = query_value - decoded;
            sum += delta * delta;
        }
        output[row] = sum;
    });
}

fn lvq_squared_l2_pairs_scalar(batch: &LvqBatchView<'_>, queries: &[f32], output: &mut [f32]) {
    batch.codes.for_each_row(|row, row_codes| {
        let query = &queries[row * batch.dimension..(row + 1) * batch.dimension];
        let mut sum = 0.0_f32;
        for (dimension, query_value) in query.iter().copied().enumerate() {
            let decoded = batch.offsets[row]
                + batch.scales[row] * f32::from(code_at(batch.bits, row_codes, dimension));
            let delta = query_value - decoded;
            sum += delta * delta;
        }
        output[row] = sum;
    });
}

fn lvq_squared_l2_row_scalar_dispatch(batch: &LvqEncodedBatch, query: &[f32], row: usize) -> f32 {
    let codes = encoded_row_codes(batch, row);
    let mut sum = 0.0_f32;
    for (dimension, query_value) in query.iter().copied().enumerate() {
        let decoded = batch.offsets[row]
            + batch.scales[row] * f32::from(code_at(batch.bits, codes, dimension));
        let delta = query_value - decoded;
        sum += delta * delta;
    }
    sum
}

fn encoded_row_codes(batch: &LvqEncodedBatch, row: usize) -> &[u8] {
    let code_size = batch.bits.code_size(batch.dimension);
    &batch.codes[row * code_size..(row + 1) * code_size]
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn lvq_squared_l2_row_avx2_dispatch(batch: &LvqEncodedBatch, query: &[f32], row: usize) -> f32 {
    // SAFETY: this scorer is selected only after AVX2 runtime detection.
    unsafe {
        lvq_squared_l2_row_avx2(
            batch.bits,
            batch.dimension,
            encoded_row_codes(batch, row),
            batch.offsets[row],
            batch.scales[row],
            query,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn lvq_squared_l2_row_avx2(
    bits: LvqBits,
    vector_dimension: usize,
    row_codes: &[u8],
    row_offset: f32,
    row_scale: f32,
    query: &[f32],
) -> f32 {
    use std::arch::x86_64::{
        _mm_and_si128, _mm_loadl_epi64, _mm_set1_epi8, _mm_srli_epi16, _mm_srli_si128,
        _mm_unpacklo_epi8, _mm256_add_ps, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32,
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    let codes = row_codes.as_ptr();
    let offset = _mm256_set1_ps(row_offset);
    let scale = _mm256_set1_ps(row_scale);
    let mut accumulated = _mm256_setzero_ps();
    let mut dimension = 0;
    match bits {
        LvqBits::Eight => {
            while dimension + 8 <= vector_dimension {
                let bytes = unsafe { _mm_loadl_epi64(codes.add(dimension).cast()) };
                let code_values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                let decoded = _mm256_add_ps(_mm256_mul_ps(scale, code_values), offset);
                let query_values = unsafe { _mm256_loadu_ps(query.as_ptr().add(dimension)) };
                let delta = _mm256_sub_ps(query_values, decoded);
                accumulated = _mm256_add_ps(accumulated, _mm256_mul_ps(delta, delta));
                dimension += 8;
            }
        }
        LvqBits::Four => {
            while dimension + 16 <= vector_dimension {
                let bytes = unsafe { _mm_loadl_epi64(codes.add(dimension / 2).cast()) };
                let mask = _mm_set1_epi8(0x0f);
                let low = _mm_and_si128(bytes, mask);
                let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
                let interleaved = _mm_unpacklo_epi8(low, high);
                for (offset_in_chunk, selected) in
                    [(0, interleaved), (8, _mm_srli_si128::<8>(interleaved))]
                {
                    let code_values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(selected));
                    let decoded = _mm256_add_ps(_mm256_mul_ps(scale, code_values), offset);
                    let query_values =
                        unsafe { _mm256_loadu_ps(query.as_ptr().add(dimension + offset_in_chunk)) };
                    let delta = _mm256_sub_ps(query_values, decoded);
                    accumulated = _mm256_add_ps(accumulated, _mm256_mul_ps(delta, delta));
                }
                dimension += 16;
            }
        }
    }
    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulated) };
    let mut sum = lanes.into_iter().sum::<f32>();
    for (index, query_value) in query.iter().copied().enumerate().skip(dimension) {
        let code = unsafe { code_at_ptr(bits, codes, index) };
        let decoded = row_offset + row_scale * f32::from(code);
        let delta = query_value - decoded;
        sum += delta * delta;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn lvq_squared_l2_row_avx512_dispatch(batch: &LvqEncodedBatch, query: &[f32], row: usize) -> f32 {
    // SAFETY: this scorer is selected only after AVX-512F and AVX-512BW detection.
    unsafe {
        lvq_squared_l2_row_avx512(
            batch.bits,
            batch.dimension,
            encoded_row_codes(batch, row),
            batch.offsets[row],
            batch.scales[row],
            query,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_code)]
unsafe fn lvq_squared_l2_row_avx512(
    bits: LvqBits,
    vector_dimension: usize,
    row_codes: &[u8],
    row_offset: f32,
    row_scale: f32,
    query: &[f32],
) -> f32 {
    use std::arch::x86_64::{
        _mm_and_si128, _mm_loadl_epi64, _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16,
        _mm_unpacklo_epi8, _mm512_add_ps, _mm512_cvtepi32_ps, _mm512_cvtepu8_epi32,
        _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
        _mm512_sub_ps,
    };

    let codes = row_codes.as_ptr();
    let offset = _mm512_set1_ps(row_offset);
    let scale = _mm512_set1_ps(row_scale);
    let mut accumulated = _mm512_setzero_ps();
    let mut dimension = 0;
    while dimension + 16 <= vector_dimension {
        let unpacked = match bits {
            LvqBits::Eight => {
                let bytes = unsafe { _mm_loadu_si128(codes.add(dimension).cast()) };
                _mm512_cvtepu8_epi32(bytes)
            }
            LvqBits::Four => {
                let bytes = unsafe { _mm_loadl_epi64(codes.add(dimension / 2).cast()) };
                let mask = _mm_set1_epi8(0x0f);
                let low = _mm_and_si128(bytes, mask);
                let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
                _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low, high))
            }
        };
        let code_values = _mm512_cvtepi32_ps(unpacked);
        let decoded = _mm512_add_ps(_mm512_mul_ps(scale, code_values), offset);
        let query_values = unsafe { _mm512_loadu_ps(query.as_ptr().add(dimension)) };
        let delta = _mm512_sub_ps(query_values, decoded);
        accumulated = _mm512_add_ps(accumulated, _mm512_mul_ps(delta, delta));
        dimension += 16;
    }
    let mut lanes = [0.0_f32; 16];
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), accumulated) };
    let mut sum = lanes.into_iter().sum::<f32>();
    for (index, query_value) in query.iter().copied().enumerate().skip(dimension) {
        let code = unsafe { code_at_ptr(bits, codes, index) };
        let decoded = row_offset + row_scale * f32::from(code);
        let delta = query_value - decoded;
        sum += delta * delta;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn lvq_squared_l2_rows_avx2(batch: &LvqBatchView<'_>, query: &[f32], output: &mut [f32]) {
    batch.codes.for_each_row(|row, row_codes| {
        output[row] = unsafe {
            lvq_squared_l2_row_avx2(
                batch.bits,
                batch.dimension,
                row_codes,
                batch.offsets[row],
                batch.scales[row],
                query,
            )
        };
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn lvq_squared_l2_pairs_avx2(batch: &LvqBatchView<'_>, queries: &[f32], output: &mut [f32]) {
    batch.codes.for_each_row(|row, row_codes| {
        let query = &queries[row * batch.dimension..(row + 1) * batch.dimension];
        output[row] = unsafe {
            lvq_squared_l2_row_avx2(
                batch.bits,
                batch.dimension,
                row_codes,
                batch.offsets[row],
                batch.scales[row],
                query,
            )
        };
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_code)]
unsafe fn lvq_squared_l2_rows_avx512(batch: &LvqBatchView<'_>, query: &[f32], output: &mut [f32]) {
    batch.codes.for_each_row(|row, row_codes| {
        output[row] = unsafe {
            lvq_squared_l2_row_avx512(
                batch.bits,
                batch.dimension,
                row_codes,
                batch.offsets[row],
                batch.scales[row],
                query,
            )
        };
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_code)]
unsafe fn lvq_squared_l2_pairs_avx512(
    batch: &LvqBatchView<'_>,
    queries: &[f32],
    output: &mut [f32],
) {
    batch.codes.for_each_row(|row, row_codes| {
        let query = &queries[row * batch.dimension..(row + 1) * batch.dimension];
        output[row] = unsafe {
            lvq_squared_l2_row_avx512(
                batch.bits,
                batch.dimension,
                row_codes,
                batch.offsets[row],
                batch.scales[row],
                query,
            )
        };
    });
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
unsafe fn code_at_ptr(bits: LvqBits, codes: *const u8, dimension: usize) -> u8 {
    match bits {
        LvqBits::Four => {
            let packed = unsafe { *codes.add(dimension / 2) };
            if dimension.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            }
        }
        LvqBits::Eight => unsafe { *codes.add(dimension) },
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::detect;

    #[test]
    fn lvq4_uses_the_canonical_nibble_layout() {
        let encoded = encode_lvq_rows(&[0.0, 0.5, 1.0], 3, LvqBits::Four).unwrap();
        assert_eq!(encoded.codes(), &[0x80, 0x0f]);
        assert_eq!(encoded.offsets(), &[0.0]);
        assert_eq!(encoded.scales(), &[1.0 / 15.0]);

        let mut decoded = [0.0; 3];
        encoded.as_view().decode_row(0, &mut decoded).unwrap();
        assert_eq!(decoded, [0.0, 8.0 / 15.0, 1.0]);
    }

    #[test]
    fn constant_rows_have_zero_scales_and_codes() {
        for bits in [LvqBits::Four, LvqBits::Eight] {
            let encoded = encode_lvq_rows(&[3.5; 10], 5, bits).unwrap();
            assert_eq!(encoded.row_count(), 2);
            assert!(encoded.codes().iter().all(|code| *code == 0));
            assert_eq!(encoded.offsets(), &[3.5, 3.5]);
            assert_eq!(encoded.scales(), &[0.0, 0.0]);
        }
    }

    #[test]
    fn prepared_query_scores_random_rows() {
        let values = (0..8 * 31)
            .map(|index| (index % 97) as f32 * 0.013 - 0.4)
            .collect::<Vec<_>>();
        let query = (0..31)
            .map(|index| 0.7 - (index % 29) as f32 * 0.009)
            .collect::<Vec<_>>();
        let encoded = encode_lvq_rows(&values, 31, LvqBits::Eight).unwrap();
        let prepared = encoded.prepare_query(&query).unwrap();
        let mut expected = vec![0.0; encoded.row_count()];
        detect()
            .lvq_squared_l2_rows(&encoded.as_view(), &query, &mut expected)
            .unwrap();

        for (row, expected) in expected.iter().copied().enumerate() {
            assert_eq!(prepared.squared_l2(row).unwrap(), expected);
        }
        assert_eq!(
            prepared.squared_l2_four([7, 1, 5, 3]).unwrap(),
            [expected[7], expected[1], expected[5], expected[3]]
        );
        assert!(prepared.squared_l2(8).is_err());
        assert!(prepared.squared_l2_four([0, 1, 2, 8]).is_err());
    }

    #[test]
    fn validates_external_lvq_buffers() {
        assert!(LvqBatchView::try_new(LvqBits::Eight, 0, &[], &[], &[]).is_err());
        assert!(LvqBatchView::try_new(LvqBits::Eight, 2, &[0], &[0.0], &[1.0]).is_err());
        assert!(LvqBatchView::try_new(LvqBits::Four, 3, &[0, 0xf0], &[0.0], &[1.0]).is_err());
        assert!(LvqBatchView::try_new(LvqBits::Eight, 1, &[1], &[2.0], &[0.0]).is_err());
    }

    #[test]
    fn distance_kernel_accepts_non_contiguous_code_rows() {
        let first = [1_u8, 2];
        let second = [3_u8, 4];
        let mut rows = LvqCodeRows::try_new(2).unwrap();
        rows.push_span(&first, 1, 2).unwrap();
        rows.push_span(&second, 1, 2).unwrap();
        let view =
            LvqBatchView::try_new_rows(LvqBits::Eight, 2, rows, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        let mut distances = [0.0; 2];

        detect()
            .lvq_squared_l2_rows(&view, &[0.0, 0.0], &mut distances)
            .unwrap();

        assert_eq!(distances, [5.0, 25.0]);
    }

    #[test]
    fn distance_kernel_scores_row_aligned_query_vectors() {
        let encoded = encode_lvq_rows(&[1.0, 2.0, 3.0, 4.0], 2, LvqBits::Eight).unwrap();
        let mut distances = [0.0; 2];

        detect()
            .lvq_squared_l2_pairs(&encoded.as_view(), &[1.0, 2.0, 2.0, 2.0], &mut distances)
            .unwrap();

        assert_eq!(distances, [0.0, 5.0]);
        assert!(
            detect()
                .lvq_squared_l2_pairs(&encoded.as_view(), &[1.0, 2.0], &mut distances)
                .is_err()
        );
        assert!(
            detect()
                .lvq_squared_l2_pairs(
                    &encoded.as_view(),
                    &[1.0, 2.0, f32::NAN, 4.0],
                    &mut distances,
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_code_spans_with_the_wrong_shape() {
        let mut rows = LvqCodeRows::try_new(2).unwrap();

        assert!(rows.push_span(&[1_u8], 1, 2).is_err());
        assert!(rows.push_span(&[1_u8, 2, 3], 2, 2).is_err());
        assert!(rows.push_span(&[1_u8, 2], 1, 1).is_err());
    }

    #[test]
    fn rejects_invalid_source_matrices() {
        assert!(encode_lvq_rows(&[], 0, LvqBits::Eight).is_err());
        assert!(encode_lvq_rows(&[1.0], 2, LvqBits::Eight).is_err());
        assert!(encode_lvq_rows(&[f32::NAN], 1, LvqBits::Eight).is_err());
        assert!(encode_lvq_rows(&[-f32::MAX, f32::MAX], 2, LvqBits::Eight).is_err());
    }

    #[test]
    fn selected_kernel_matches_decoded_dense_rows() {
        for bits in [LvqBits::Four, LvqBits::Eight] {
            for dimension in [1, 3, 8, 15, 16, 31, 32, 127] {
                let values = (0..4 * dimension)
                    .map(|index| (index % 97) as f32 * 0.013 - 0.4)
                    .collect::<Vec<_>>();
                let query = (0..dimension)
                    .map(|index| 0.7 - (index % 89) as f32 * 0.009)
                    .collect::<Vec<_>>();
                let encoded = encode_lvq_rows(&values, dimension, bits).unwrap();
                let view = encoded.as_view();
                let mut actual = vec![0.0; view.row_count()];
                detect()
                    .lvq_squared_l2_rows(&view, &query, &mut actual)
                    .unwrap();

                let mut decoded = vec![0.0; dimension];
                for (row, actual) in actual.into_iter().enumerate() {
                    view.decode_row(row, &mut decoded).unwrap();
                    let expected = decoded
                        .iter()
                        .zip(&query)
                        .map(|(value, query)| {
                            let delta = query - value;
                            delta * delta
                        })
                        .sum::<f32>();
                    let tolerance = 1e-4_f32.max(expected.abs() * 1e-5);
                    assert!((actual - expected).abs() <= tolerance);
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(unsafe_code)]
    fn every_available_x86_kernel_matches_scalar() {
        for bits in [LvqBits::Four, LvqBits::Eight] {
            for dimension in [7, 16, 31, 64, 127] {
                let values = (0..3 * dimension)
                    .map(|index| (index % 101) as f32 * 0.017 - 0.8)
                    .collect::<Vec<_>>();
                let query = (0..dimension)
                    .map(|index| 0.3 - (index % 83) as f32 * 0.011)
                    .collect::<Vec<_>>();
                let encoded = encode_lvq_rows(&values, dimension, bits).unwrap();
                let view = encoded.as_view();
                let mut expected = vec![0.0; view.row_count()];
                lvq_squared_l2_rows_scalar(&view, &query, &mut expected);

                if std::is_x86_feature_detected!("avx2") {
                    let mut actual = vec![0.0; view.row_count()];
                    unsafe { lvq_squared_l2_rows_avx2(&view, &query, &mut actual) };
                    assert_distances_close(&actual, &expected);
                }
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                {
                    let mut actual = vec![0.0; view.row_count()];
                    unsafe { lvq_squared_l2_rows_avx512(&view, &query, &mut actual) };
                    assert_distances_close(&actual, &expected);
                }
            }
        }
    }

    #[test]
    fn distance_kernel_validates_query_and_output_shapes() {
        let encoded = encode_lvq_rows(&[1.0, 2.0], 2, LvqBits::Eight).unwrap();
        let view = encoded.as_view();
        assert!(
            detect()
                .lvq_squared_l2_rows(&view, &[1.0], &mut [0.0])
                .is_err()
        );
        assert!(
            detect()
                .lvq_squared_l2_rows(&view, &[1.0, f32::NAN], &mut [0.0])
                .is_err()
        );
        assert!(
            detect()
                .lvq_squared_l2_rows(&view, &[1.0, 2.0], &mut [])
                .is_err()
        );
    }

    fn assert_distances_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-4_f32.max(expected.abs() * 1e-5);
            assert!((actual - expected).abs() <= tolerance);
        }
    }
}
