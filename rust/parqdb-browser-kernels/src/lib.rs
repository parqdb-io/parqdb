//! Small WebAssembly ABI for static-package distance and bounded top-k search.

#[cfg(any(target_arch = "wasm32", test))]
use std::cmp::Ordering;
#[cfg(any(target_arch = "wasm32", test))]
use std::collections::BinaryHeap;

#[cfg(any(target_arch = "wasm32", test))]
use parqdb_kernels::{LvqBatchView, LvqBits, detect};

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Copy)]
struct Candidate {
    row: u32,
    distance: f32,
}

#[cfg(any(target_arch = "wasm32", test))]
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.distance.to_bits() == other.distance.to_bits()
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl Eq for Candidate {}

#[cfg(any(target_arch = "wasm32", test))]
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.row.cmp(&other.row))
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn bounded_topk(distances: &[f32], k: usize) -> Vec<Candidate> {
    let mut heap = BinaryHeap::with_capacity(k.min(distances.len()));
    for (row, distance) in distances.iter().copied().enumerate() {
        let candidate = Candidate {
            row: u32::try_from(row).expect("browser row count is bounded by u32"),
            distance,
        };
        if heap.len() < k {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|worst| candidate < *worst) {
            heap.pop();
            heap.push(candidate);
        }
    }
    let mut selected = heap.into_vec();
    selected.sort_unstable();
    selected
}

#[cfg(any(target_arch = "wasm32", test))]
fn dense_topk(values: &[f32], query: &[f32], k: usize) -> Option<Vec<Candidate>> {
    if query.is_empty()
        || k == 0
        || !values.len().is_multiple_of(query.len())
        || values.iter().chain(query).any(|value| !value.is_finite())
    {
        return None;
    }
    let rows = values.len() / query.len();
    if rows > u32::MAX as usize {
        return None;
    }
    let mut distances = vec![0.0; rows];
    detect().squared_l2_rows(values, query, &mut distances);
    Some(bounded_topk(&distances, k.min(rows)))
}

#[cfg(any(target_arch = "wasm32", test))]
fn lvq_topk(
    bits: LvqBits,
    dimension: usize,
    codes: &[u8],
    offsets: &[f32],
    scales: &[f32],
    query: &[f32],
    k: usize,
) -> Option<Vec<Candidate>> {
    if k == 0 || offsets.len() > u32::MAX as usize {
        return None;
    }
    let batch = LvqBatchView::try_new(bits, dimension, codes, offsets, scales).ok()?;
    let mut distances = vec![0.0; offsets.len()];
    detect()
        .lvq_squared_l2_rows(&batch, query, &mut distances)
        .ok()?;
    Some(bounded_topk(&distances, k.min(offsets.len())))
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::mem;
    use std::slice;

    use super::{Candidate, LvqBits, dense_topk, lvq_topk};

    const INVALID_ARGUMENT: i32 = -1;
    const SHAPE_OVERFLOW: i32 = -2;

    #[allow(unsafe_code)]
    #[unsafe(no_mangle)]
    pub extern "C" fn parqdb_alloc(byte_len: u32) -> u32 {
        if byte_len == 0 {
            return 0;
        }
        let words = (byte_len as usize).div_ceil(mem::size_of::<u32>());
        let mut allocation = vec![0_u32; words];
        let pointer = allocation.as_mut_ptr();
        mem::forget(allocation);
        pointer as u32
    }

    #[allow(unsafe_code)]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn parqdb_free(pointer: u32, byte_len: u32) {
        if pointer == 0 || byte_len == 0 {
            return;
        }
        let words = (byte_len as usize).div_ceil(mem::size_of::<u32>());
        // SAFETY: the TypeScript wrapper passes only allocations returned by
        // `parqdb_alloc`, with the same requested byte length, exactly once.
        drop(unsafe { Vec::from_raw_parts(pointer as *mut u32, words, words) });
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn parqdb_dense_topk(
        values_pointer: u32,
        query_pointer: u32,
        rows: u32,
        dimension: u32,
        k: u32,
        output_rows_pointer: u32,
        output_distances_pointer: u32,
    ) -> i32 {
        let Some(value_len) = (rows as usize).checked_mul(dimension as usize) else {
            return SHAPE_OVERFLOW;
        };
        let Some(selected) = (unsafe {
            dense_topk(
                slice::from_raw_parts(values_pointer as *const f32, value_len),
                slice::from_raw_parts(query_pointer as *const f32, dimension as usize),
                k as usize,
            )
        }) else {
            return INVALID_ARGUMENT;
        };
        // SAFETY: the wrapper allocates both output buffers for at least
        // `min(rows, k)` elements before calling this function.
        unsafe { write_candidates(&selected, output_rows_pointer, output_distances_pointer) };
        i32::try_from(selected.len()).unwrap_or(SHAPE_OVERFLOW)
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn parqdb_lvq_topk(
        codes_pointer: u32,
        offsets_pointer: u32,
        scales_pointer: u32,
        query_pointer: u32,
        rows: u32,
        dimension: u32,
        bits: u32,
        k: u32,
        output_rows_pointer: u32,
        output_distances_pointer: u32,
    ) -> i32 {
        let bits = match bits {
            4 => LvqBits::Four,
            8 => LvqBits::Eight,
            _ => return INVALID_ARGUMENT,
        };
        let code_size = match bits {
            LvqBits::Four => (dimension as usize).div_ceil(2),
            LvqBits::Eight => dimension as usize,
        };
        let Some(code_len) = (rows as usize).checked_mul(code_size) else {
            return SHAPE_OVERFLOW;
        };
        let Some(selected) = (unsafe {
            lvq_topk(
                bits,
                dimension as usize,
                slice::from_raw_parts(codes_pointer as *const u8, code_len),
                slice::from_raw_parts(offsets_pointer as *const f32, rows as usize),
                slice::from_raw_parts(scales_pointer as *const f32, rows as usize),
                slice::from_raw_parts(query_pointer as *const f32, dimension as usize),
                k as usize,
            )
        }) else {
            return INVALID_ARGUMENT;
        };
        // SAFETY: the wrapper allocates both output buffers for at least
        // `min(rows, k)` elements before calling this function.
        unsafe { write_candidates(&selected, output_rows_pointer, output_distances_pointer) };
        i32::try_from(selected.len()).unwrap_or(SHAPE_OVERFLOW)
    }

    #[allow(unsafe_code)]
    unsafe fn write_candidates(
        selected: &[Candidate],
        output_rows_pointer: u32,
        output_distances_pointer: u32,
    ) {
        let output_rows =
            unsafe { slice::from_raw_parts_mut(output_rows_pointer as *mut u32, selected.len()) };
        let output_distances = unsafe {
            slice::from_raw_parts_mut(output_distances_pointer as *mut f32, selected.len())
        };
        for (position, candidate) in selected.iter().enumerate() {
            output_rows[position] = candidate.row;
            output_distances[position] = candidate.distance;
        }
    }
}

#[cfg(test)]
mod tests {
    use parqdb_kernels::{LvqBits, encode_lvq_rows};

    use super::{dense_topk, lvq_topk};

    #[test]
    fn dense_topk_is_distance_then_row_deterministic() {
        let selected = dense_topk(&[1.0, 0.0, -1.0, 0.0, 3.0, 0.0], &[0.0, 0.0], 2).unwrap();
        assert_eq!(
            selected.iter().map(|item| item.row).collect::<Vec<_>>(),
            [0, 1]
        );
        assert!((selected[0].distance - 1.0).abs() < f32::EPSILON);
        assert!((selected[1].distance - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn lvq_topk_matches_reconstructed_order() {
        let values = [0.0, 0.5, 1.0, 3.0, 3.0, 3.0];
        for bits in [LvqBits::Four, LvqBits::Eight] {
            let encoded = encode_lvq_rows(&values, 3, bits).unwrap();
            let selected = lvq_topk(
                bits,
                3,
                encoded.codes(),
                encoded.offsets(),
                encoded.scales(),
                &[0.0, 0.0, 0.0],
                1,
            )
            .unwrap();
            assert_eq!(selected[0].row, 0);
        }
    }
}
