//! Runtime-selected vector kernels and the platform GEMM boundary.
//!
//! This is adapted from `RoverQ`'s native distance layer. Unsafe code is kept in
//! this crate so its consumers remain safe Rust. macOS uses
//! Accelerate; other platforms use the portable `matrixmultiply` fallback.

use std::sync::OnceLock;

use thiserror::Error;

mod lvq;

pub use lvq::{LvqBatchView, LvqBits, LvqCodeRows, LvqEncodedBatch, LvqRowQuery, encode_lvq_rows};

/// Error returned when matrix buffers do not match their declared shapes.
#[derive(Debug, Error)]
#[error("invalid kernel argument: {0}")]
pub struct KernelError(String);

/// Result returned by fallible vector kernels.
pub type Result<T> = std::result::Result<T, KernelError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuBackend {
    #[cfg(not(target_arch = "aarch64"))]
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

#[derive(Debug, Clone, Copy)]
/// Runtime-selected implementation of Relify's dense `f32` vector kernels.
pub struct DistanceKernel {
    backend: CpuBackend,
}

static DETECTED: OnceLock<DistanceKernel> = OnceLock::new();

/// Returns the process-wide kernel selected for the current CPU.
pub fn detect() -> &'static DistanceKernel {
    DETECTED.get_or_init(|| DistanceKernel {
        backend: detect_backend(),
    })
}

impl DistanceKernel {
    #[cfg(test)]
    pub(crate) fn name(self) -> &'static str {
        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => "avx2",
        }
    }

    #[allow(unsafe_code)]
    /// Computes the squared Euclidean distance between equal-length vectors.
    ///
    /// # Panics
    ///
    /// Panics when the vectors have different dimensions.
    #[must_use]
    pub fn squared_l2(self, left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(
            left.len(),
            right.len(),
            "squared-L2 vectors must have equal dimensions"
        );
        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => squared_l2_scalar(left, right),
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => {
                // SAFETY: AArch64 guarantees NEON support. The implementation
                // reads only in-bounds vector chunks and a bounded scalar tail.
                unsafe { squared_l2_neon(left.as_ptr(), right.as_ptr(), left.len()) }
            }
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                // SAFETY: this variant is selected only after AVX2 runtime
                // detection and the implementation handles its scalar tail.
                unsafe { squared_l2_avx2(left.as_ptr(), right.as_ptr(), left.len()) }
            }
        }
    }

    #[allow(unsafe_code)]
    /// Computes squared Euclidean distance from every dense row to `query`.
    ///
    /// # Panics
    ///
    /// Panics when the query is empty or the matrix shape does not match the
    /// output row count.
    pub fn squared_l2_rows(self, values: &[f32], query: &[f32], output: &mut [f32]) {
        assert!(!query.is_empty(), "query vector must not be empty");
        let expected_values = output
            .len()
            .checked_mul(query.len())
            .expect("dense vector matrix shape must not overflow");
        assert_eq!(
            values.len(),
            expected_values,
            "dense vector matrix does not match the output row count"
        );
        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => squared_l2_rows_scalar(values, query, output),
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => {
                // SAFETY: AArch64 guarantees NEON support. The declared matrix
                // shape is validated by the caller and debug assertions above.
                unsafe {
                    squared_l2_rows_neon(
                        values.as_ptr(),
                        query.as_ptr(),
                        output.as_mut_ptr(),
                        output.len(),
                        query.len(),
                    );
                }
            }
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                // SAFETY: this variant is selected only after AVX2 runtime
                // detection and the buffers match the declared matrix shape.
                unsafe {
                    squared_l2_rows_avx2(
                        values.as_ptr(),
                        query.as_ptr(),
                        output.as_mut_ptr(),
                        output.len(),
                        query.len(),
                    );
                }
            }
        }
    }

    #[allow(unsafe_code)]
    /// Computes squared Euclidean distance between corresponding dense rows.
    ///
    /// # Panics
    ///
    /// Panics when the dimension is zero, either matrix is not row-aligned, or
    /// the matrix and output row counts differ.
    pub fn squared_l2_pairs(
        self,
        left: &[f32],
        right: &[f32],
        dimension: usize,
        output: &mut [f32],
    ) {
        assert!(dimension > 0, "vector dimension must be positive");
        let expected_values = output
            .len()
            .checked_mul(dimension)
            .expect("dense vector matrix shape must not overflow");
        assert_eq!(
            left.len(),
            expected_values,
            "left dense vector matrix does not match the output row count"
        );
        assert_eq!(
            right.len(),
            expected_values,
            "right dense vector matrix does not match the output row count"
        );
        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => squared_l2_pairs_scalar(left, right, dimension, output),
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => {
                // SAFETY: AArch64 guarantees NEON support and the validated
                // matrices contain `output.len()` rows of `dimension` values.
                unsafe {
                    squared_l2_pairs_neon(
                        left.as_ptr(),
                        right.as_ptr(),
                        output.as_mut_ptr(),
                        output.len(),
                        dimension,
                    );
                }
            }
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                // SAFETY: this variant is selected only after AVX2 runtime
                // detection and the matrix shapes were validated above.
                unsafe {
                    squared_l2_pairs_avx2(
                        left.as_ptr(),
                        right.as_ptr(),
                        output.as_mut_ptr(),
                        output.len(),
                        dimension,
                    );
                }
            }
        }
    }

    #[allow(unsafe_code)]
    /// Finds the nearest reference row for every dense input row.
    ///
    /// `references` is a non-empty row-major matrix with the same dimension as
    /// `values`. The selected reference row and squared distance are written to
    /// the corresponding output slots.
    ///
    /// # Panics
    ///
    /// Panics when the dimension is zero, either matrix is not row-aligned,
    /// `references` is empty, or the output row counts differ.
    pub fn nearest_squared_l2_rows(
        self,
        values: &[f32],
        references: &[f32],
        dimension: usize,
        indices: &mut [usize],
        distances: &mut [f32],
    ) {
        assert!(dimension > 0, "vector dimension must be positive");
        assert!(
            values.len().is_multiple_of(dimension),
            "dense vector matrix does not match its dimension"
        );
        assert!(
            !references.is_empty() && references.len().is_multiple_of(dimension),
            "reference matrix must contain complete rows"
        );
        let rows = values.len() / dimension;
        assert_eq!(
            indices.len(),
            rows,
            "nearest-reference indices do not match the input row count"
        );
        assert_eq!(
            distances.len(),
            rows,
            "nearest-reference distances do not match the input row count"
        );
        let reference_rows = references.len() / dimension;
        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => {
                nearest_squared_l2_rows_scalar(values, references, dimension, indices, distances);
            }
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => {
                // SAFETY: AArch64 guarantees NEON support and all matrix and
                // output shapes were validated above.
                unsafe {
                    nearest_squared_l2_rows_neon(
                        values.as_ptr(),
                        references.as_ptr(),
                        indices.as_mut_ptr(),
                        distances.as_mut_ptr(),
                        rows,
                        reference_rows,
                        dimension,
                    );
                }
            }
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                // SAFETY: this variant is selected only after AVX2 runtime
                // detection and all matrix and output shapes were validated.
                unsafe {
                    nearest_squared_l2_rows_avx2(
                        values.as_ptr(),
                        references.as_ptr(),
                        indices.as_mut_ptr(),
                        distances.as_mut_ptr(),
                        rows,
                        reference_rows,
                        dimension,
                    );
                }
            }
        }
    }

    #[allow(unsafe_code)]
    /// Computes one squared L2 norm per dense row.
    ///
    /// # Panics
    ///
    /// Panics when the dimension is zero or the matrix is not row-aligned.
    #[must_use]
    pub fn row_norms(self, values: &[f32], dimension: usize) -> Vec<f32> {
        assert!(dimension > 0, "vector dimension must be positive");
        assert!(
            values.len().is_multiple_of(dimension),
            "dense vector matrix does not match its dimension"
        );
        let mut output = vec![0.0; values.len() / dimension];
        match self.backend {
            #[cfg(not(target_arch = "aarch64"))]
            CpuBackend::Scalar => norm_squared_rows_scalar(values, dimension, &mut output),
            #[cfg(target_arch = "aarch64")]
            CpuBackend::Neon => {
                // SAFETY: AArch64 guarantees NEON support and the matrix shape
                // was validated above.
                unsafe {
                    norm_squared_rows_neon(
                        values.as_ptr(),
                        output.as_mut_ptr(),
                        output.len(),
                        dimension,
                    );
                }
            }
            #[cfg(target_arch = "x86_64")]
            CpuBackend::Avx2 => {
                // SAFETY: this variant is selected only after AVX2 runtime
                // detection and the matrix shape was validated above.
                unsafe {
                    norm_squared_rows_avx2(
                        values.as_ptr(),
                        output.as_mut_ptr(),
                        output.len(),
                        dimension,
                    );
                }
            }
        }
        output
    }
}

#[allow(unsafe_code)]
/// Multiplies a row-major matrix by the transpose of another row-major matrix.
pub fn row_dot_products(
    left: &[f32],
    right: &[f32],
    rows: usize,
    dimension: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<()> {
    let left_len = rows
        .checked_mul(dimension)
        .ok_or_else(|| KernelError("row-dot left shape overflows".into()))?;
    let right_len = columns
        .checked_mul(dimension)
        .ok_or_else(|| KernelError("row-dot right shape overflows".into()))?;
    let output_len = rows
        .checked_mul(columns)
        .ok_or_else(|| KernelError("row-dot output shape overflows".into()))?;
    if dimension == 0 || columns == 0 {
        return Err(KernelError(
            "row-dot dimension and columns must be positive".into(),
        ));
    }
    if left.len() != left_len || right.len() != right_len || output.len() != output_len {
        return Err(KernelError(
            "row-dot buffers do not match the declared matrix shapes".into(),
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    accelerate_row_dot_products(left, right, rows, dimension, columns, output)?;
    #[cfg(not(target_os = "macos"))]
    portable_row_dot_products(left, right, rows, dimension, columns, output)?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn accelerate_row_dot_products(
    left: &[f32],
    right: &[f32],
    rows: usize,
    dimension: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<()> {
    let rows = i32::try_from(rows).map_err(|_| KernelError("row-dot rows exceed i32".into()))?;
    let dimension = i32::try_from(dimension)
        .map_err(|_| KernelError("row-dot dimension exceeds i32".into()))?;
    let columns =
        i32::try_from(columns).map_err(|_| KernelError("row-dot columns exceed i32".into()))?;

    // SAFETY: the caller validates that all matrix buffers cover the row-major
    // dimensions and leading dimensions passed to CBLAS.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows,
            columns,
            dimension,
            1.0,
            left.as_ptr(),
            dimension,
            right.as_ptr(),
            dimension,
            0.0,
            output.as_mut_ptr(),
            columns,
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
#[allow(unsafe_code)]
fn portable_row_dot_products(
    left: &[f32],
    right: &[f32],
    rows: usize,
    dimension: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<()> {
    let dimension_stride = isize::try_from(dimension)
        .map_err(|_| KernelError("row-dot dimension exceeds isize".into()))?;
    let column_stride =
        isize::try_from(columns).map_err(|_| KernelError("row-dot columns exceed isize".into()))?;

    // SAFETY: lengths above validate row-major `left` as rows x dimension,
    // `right.T` as dimension x columns, and `output` as rows x columns. The
    // output buffer is uniquely borrowed for the duration of the call.
    unsafe {
        matrixmultiply::sgemm(
            rows,
            dimension,
            columns,
            1.0,
            left.as_ptr(),
            dimension_stride,
            1,
            right.as_ptr(),
            1,
            dimension_stride,
            0.0,
            output.as_mut_ptr(),
            column_stride,
            1,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transpose_left: i32,
        transpose_right: i32,
        rows: i32,
        columns: i32,
        dimension: i32,
        alpha: f32,
        left: *const f32,
        left_stride: i32,
        right: *const f32,
        right_stride: i32,
        beta: f32,
        output: *mut f32,
        output_stride: i32,
    );
}

#[cfg(target_arch = "aarch64")]
fn detect_backend() -> CpuBackend {
    CpuBackend::Neon
}

#[cfg(target_arch = "x86_64")]
fn detect_backend() -> CpuBackend {
    if std::is_x86_feature_detected!("avx2") {
        CpuBackend::Avx2
    } else {
        CpuBackend::Scalar
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn detect_backend() -> CpuBackend {
    CpuBackend::Scalar
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn squared_l2_scalar(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    while index + 4 <= left.len() {
        let delta0 = left[index] - right[index];
        let delta1 = left[index + 1] - right[index + 1];
        let delta2 = left[index + 2] - right[index + 2];
        let delta3 = left[index + 3] - right[index + 3];
        sum0 += delta0 * delta0;
        sum1 += delta1 * delta1;
        sum2 += delta2 * delta2;
        sum3 += delta3 * delta3;
        index += 4;
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    while index < left.len() {
        let delta = left[index] - right[index];
        sum += delta * delta;
        index += 1;
    }
    sum
}

#[cfg(not(target_arch = "aarch64"))]
fn squared_l2_rows_scalar(values: &[f32], query: &[f32], output: &mut [f32]) {
    for (row, distance) in values.chunks_exact(query.len()).zip(output) {
        *distance = squared_l2_scalar(row, query);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn squared_l2_pairs_scalar(left: &[f32], right: &[f32], dimension: usize, output: &mut [f32]) {
    for ((left, right), distance) in left
        .chunks_exact(dimension)
        .zip(right.chunks_exact(dimension))
        .zip(output)
    {
        *distance = squared_l2_scalar(left, right);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn nearest_squared_l2_rows_scalar(
    values: &[f32],
    references: &[f32],
    dimension: usize,
    indices: &mut [usize],
    distances: &mut [f32],
) {
    for ((value, best_index), best_distance) in
        values.chunks_exact(dimension).zip(indices).zip(distances)
    {
        let mut index = 0;
        let mut distance = squared_l2_scalar(value, &references[..dimension]);
        for (candidate_index, reference) in references.chunks_exact(dimension).enumerate().skip(1) {
            let candidate = squared_l2_scalar(value, reference);
            if candidate < distance {
                index = candidate_index;
                distance = candidate;
            }
        }
        *best_index = index;
        *best_distance = distance;
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn norm_squared_scalar(values: &[f32]) -> f32 {
    let mut index = 0;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    while index + 4 <= values.len() {
        sum0 += values[index] * values[index];
        sum1 += values[index + 1] * values[index + 1];
        sum2 += values[index + 2] * values[index + 2];
        sum3 += values[index + 3] * values[index + 3];
        index += 4;
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    while index < values.len() {
        sum += values[index] * values[index];
        index += 1;
    }
    sum
}

#[cfg(not(target_arch = "aarch64"))]
fn norm_squared_rows_scalar(values: &[f32], dimension: usize, output: &mut [f32]) {
    for (row, norm) in values.chunks_exact(dimension).zip(output) {
        *norm = norm_squared_scalar(row);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn squared_l2_neon(left: *const f32, right: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};

    let mut index = 0;
    let mut accumulator = vdupq_n_f32(0.0);
    while index + 4 <= len {
        // SAFETY: the loop condition guarantees both four-lane loads are in bounds.
        let left_values = unsafe { vld1q_f32(left.add(index)) };
        let right_values = unsafe { vld1q_f32(right.add(index)) };
        let delta = vsubq_f32(left_values, right_values);
        accumulator = vfmaq_f32(accumulator, delta, delta);
        index += 4;
    }
    let mut sum = vaddvq_f32(accumulator);
    while index < len {
        // SAFETY: the scalar tail is bounded by `index < len`.
        let delta = unsafe { *left.add(index) - *right.add(index) };
        sum += delta * delta;
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn squared_l2_rows_neon(
    values: *const f32,
    query: *const f32,
    output: *mut f32,
    rows: usize,
    dimension: usize,
) {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};

    let mut row = 0;
    while row + 4 <= rows {
        let mut accumulator0 = vdupq_n_f32(0.0);
        let mut accumulator1 = vdupq_n_f32(0.0);
        let mut accumulator2 = vdupq_n_f32(0.0);
        let mut accumulator3 = vdupq_n_f32(0.0);
        let mut index = 0;
        while index + 4 <= dimension {
            // SAFETY: the row and dimension loop bounds keep all vector loads
            // inside the declared dense row-major matrix and query.
            let query_values = unsafe { vld1q_f32(query.add(index)) };
            let values0 = unsafe { vld1q_f32(values.add(row * dimension + index)) };
            let values1 = unsafe { vld1q_f32(values.add((row + 1) * dimension + index)) };
            let values2 = unsafe { vld1q_f32(values.add((row + 2) * dimension + index)) };
            let values3 = unsafe { vld1q_f32(values.add((row + 3) * dimension + index)) };
            let delta0 = vsubq_f32(values0, query_values);
            let delta1 = vsubq_f32(values1, query_values);
            let delta2 = vsubq_f32(values2, query_values);
            let delta3 = vsubq_f32(values3, query_values);
            accumulator0 = vfmaq_f32(accumulator0, delta0, delta0);
            accumulator1 = vfmaq_f32(accumulator1, delta1, delta1);
            accumulator2 = vfmaq_f32(accumulator2, delta2, delta2);
            accumulator3 = vfmaq_f32(accumulator3, delta3, delta3);
            index += 4;
        }
        let mut sums = [
            vaddvq_f32(accumulator0),
            vaddvq_f32(accumulator1),
            vaddvq_f32(accumulator2),
            vaddvq_f32(accumulator3),
        ];
        while index < dimension {
            // SAFETY: the scalar tail is bounded by `index < dimension`.
            let query_value = unsafe { *query.add(index) };
            for (offset, sum) in sums.iter_mut().enumerate() {
                let delta =
                    unsafe { *values.add((row + offset) * dimension + index) } - query_value;
                *sum += delta * delta;
            }
            index += 1;
        }
        // SAFETY: `row..row + 4` is in bounds by the outer loop condition.
        unsafe {
            output.add(row).copy_from_nonoverlapping(sums.as_ptr(), 4);
        }
        row += 4;
    }
    while row < rows {
        // SAFETY: the remaining row and query both contain `dimension` values.
        unsafe {
            *output.add(row) = squared_l2_neon(values.add(row * dimension), query, dimension);
        }
        row += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn squared_l2_pairs_neon(
    left: *const f32,
    right: *const f32,
    output: *mut f32,
    rows: usize,
    dimension: usize,
) {
    for row in 0..rows {
        // SAFETY: each matrix contains `rows * dimension` values and the output
        // contains `rows` slots.
        unsafe {
            *output.add(row) = squared_l2_neon(
                left.add(row * dimension),
                right.add(row * dimension),
                dimension,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn nearest_squared_l2_rows_neon(
    values: *const f32,
    references: *const f32,
    indices: *mut usize,
    distances: *mut f32,
    rows: usize,
    reference_rows: usize,
    dimension: usize,
) {
    for row in 0..rows {
        // SAFETY: the caller validates both dense matrices and output buffers.
        let value = unsafe { values.add(row * dimension) };
        let mut best_index = 0;
        let mut best_distance = unsafe { squared_l2_neon(value, references, dimension) };
        for reference_row in 1..reference_rows {
            let candidate = unsafe {
                squared_l2_neon(value, references.add(reference_row * dimension), dimension)
            };
            if candidate < best_distance {
                best_index = reference_row;
                best_distance = candidate;
            }
        }
        unsafe {
            *indices.add(row) = best_index;
            *distances.add(row) = best_distance;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn norm_squared_neon(values: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let mut index = 0;
    let mut accumulator = vdupq_n_f32(0.0);
    while index + 4 <= len {
        // SAFETY: the loop condition guarantees the four-lane load is in bounds.
        let value = unsafe { vld1q_f32(values.add(index)) };
        accumulator = vfmaq_f32(accumulator, value, value);
        index += 4;
    }
    let mut sum = vaddvq_f32(accumulator);
    while index < len {
        // SAFETY: the scalar tail is bounded by `index < len`.
        let value = unsafe { *values.add(index) };
        sum += value * value;
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn norm_squared_rows_neon(
    values: *const f32,
    output: *mut f32,
    rows: usize,
    dimension: usize,
) {
    for row in 0..rows {
        // SAFETY: the caller validates the dense matrix and output buffer.
        unsafe {
            *output.add(row) = norm_squared_neon(values.add(row * dimension), dimension);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn squared_l2_avx2(left: *const f32, right: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    // SAFETY: the caller runtime-checks AVX2 and provides valid slices.
    unsafe {
        let mut index = 0;
        let mut accumulator0 = _mm256_setzero_ps();
        let mut accumulator1 = _mm256_setzero_ps();
        while index + 16 <= len {
            let left0 = _mm256_loadu_ps(left.add(index));
            let right0 = _mm256_loadu_ps(right.add(index));
            let delta0 = _mm256_sub_ps(left0, right0);
            accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(delta0, delta0));

            let left1 = _mm256_loadu_ps(left.add(index + 8));
            let right1 = _mm256_loadu_ps(right.add(index + 8));
            let delta1 = _mm256_sub_ps(left1, right1);
            accumulator1 = _mm256_add_ps(accumulator1, _mm256_mul_ps(delta1, delta1));
            index += 16;
        }
        accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
        while index + 8 <= len {
            let left_values = _mm256_loadu_ps(left.add(index));
            let right_values = _mm256_loadu_ps(right.add(index));
            let delta = _mm256_sub_ps(left_values, right_values);
            accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(delta, delta));
            index += 8;
        }
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator0);
        let mut sum = lanes.iter().sum();
        while index < len {
            let delta = *left.add(index) - *right.add(index);
            sum += delta * delta;
            index += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn squared_l2_rows_avx2(
    values: *const f32,
    query: *const f32,
    output: *mut f32,
    rows: usize,
    dimension: usize,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    // SAFETY: the caller runtime-checks AVX2 and supplies a dense row-major
    // matrix, a query with `dimension` values, and `rows` output slots.
    unsafe {
        let mut row = 0;
        while row + 4 <= rows {
            let mut accumulator0 = _mm256_setzero_ps();
            let mut accumulator1 = _mm256_setzero_ps();
            let mut accumulator2 = _mm256_setzero_ps();
            let mut accumulator3 = _mm256_setzero_ps();
            let mut index = 0;
            while index + 8 <= dimension {
                let query_values = _mm256_loadu_ps(query.add(index));
                let delta0 = _mm256_sub_ps(
                    _mm256_loadu_ps(values.add(row * dimension + index)),
                    query_values,
                );
                let delta1 = _mm256_sub_ps(
                    _mm256_loadu_ps(values.add((row + 1) * dimension + index)),
                    query_values,
                );
                let delta2 = _mm256_sub_ps(
                    _mm256_loadu_ps(values.add((row + 2) * dimension + index)),
                    query_values,
                );
                let delta3 = _mm256_sub_ps(
                    _mm256_loadu_ps(values.add((row + 3) * dimension + index)),
                    query_values,
                );
                accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(delta0, delta0));
                accumulator1 = _mm256_add_ps(accumulator1, _mm256_mul_ps(delta1, delta1));
                accumulator2 = _mm256_add_ps(accumulator2, _mm256_mul_ps(delta2, delta2));
                accumulator3 = _mm256_add_ps(accumulator3, _mm256_mul_ps(delta3, delta3));
                index += 8;
            }
            let mut lanes = [[0.0_f32; 8]; 4];
            _mm256_storeu_ps(lanes[0].as_mut_ptr(), accumulator0);
            _mm256_storeu_ps(lanes[1].as_mut_ptr(), accumulator1);
            _mm256_storeu_ps(lanes[2].as_mut_ptr(), accumulator2);
            _mm256_storeu_ps(lanes[3].as_mut_ptr(), accumulator3);
            let mut sums = [
                lanes[0].iter().sum::<f32>(),
                lanes[1].iter().sum::<f32>(),
                lanes[2].iter().sum::<f32>(),
                lanes[3].iter().sum::<f32>(),
            ];
            while index < dimension {
                let query_value = *query.add(index);
                for (offset, sum) in sums.iter_mut().enumerate() {
                    let delta = *values.add((row + offset) * dimension + index) - query_value;
                    *sum += delta * delta;
                }
                index += 1;
            }
            output.add(row).copy_from_nonoverlapping(sums.as_ptr(), 4);
            row += 4;
        }
        while row < rows {
            *output.add(row) = squared_l2_avx2(values.add(row * dimension), query, dimension);
            row += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn squared_l2_pairs_avx2(
    left: *const f32,
    right: *const f32,
    output: *mut f32,
    rows: usize,
    dimension: usize,
) {
    // SAFETY: the caller validates both dense matrices and the output buffer.
    unsafe {
        for row in 0..rows {
            *output.add(row) = squared_l2_avx2(
                left.add(row * dimension),
                right.add(row * dimension),
                dimension,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn nearest_squared_l2_rows_avx2(
    values: *const f32,
    references: *const f32,
    indices: *mut usize,
    distances: *mut f32,
    rows: usize,
    reference_rows: usize,
    dimension: usize,
) {
    // SAFETY: the caller validates both dense matrices and output buffers.
    unsafe {
        for row in 0..rows {
            let value = values.add(row * dimension);
            let mut best_index = 0;
            let mut best_distance = squared_l2_avx2(value, references, dimension);
            for reference_row in 1..reference_rows {
                let candidate =
                    squared_l2_avx2(value, references.add(reference_row * dimension), dimension);
                if candidate < best_distance {
                    best_index = reference_row;
                    best_distance = candidate;
                }
            }
            *indices.add(row) = best_index;
            *distances.add(row) = best_distance;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn norm_squared_avx2(values: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // SAFETY: the caller runtime-checks AVX2 and provides a valid slice.
    unsafe {
        let mut index = 0;
        let mut accumulator0 = _mm256_setzero_ps();
        let mut accumulator1 = _mm256_setzero_ps();
        while index + 16 <= len {
            let values0 = _mm256_loadu_ps(values.add(index));
            accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(values0, values0));
            let values1 = _mm256_loadu_ps(values.add(index + 8));
            accumulator1 = _mm256_add_ps(accumulator1, _mm256_mul_ps(values1, values1));
            index += 16;
        }
        accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
        while index + 8 <= len {
            let chunk = _mm256_loadu_ps(values.add(index));
            accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(chunk, chunk));
            index += 8;
        }
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator0);
        let mut sum = lanes.iter().sum();
        while index < len {
            let value = *values.add(index);
            sum += value * value;
            index += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn norm_squared_rows_avx2(
    values: *const f32,
    output: *mut f32,
    rows: usize,
    dimension: usize,
) {
    // SAFETY: the caller validates the dense matrix and output buffer.
    unsafe {
        for row in 0..rows {
            *output.add(row) = norm_squared_avx2(values.add(row * dimension), dimension);
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests;
