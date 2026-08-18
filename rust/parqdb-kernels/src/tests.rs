//! Kernel correctness and shape-validation tests.

use super::*;

#[test]
fn detected_kernel_matches_scalar_across_vector_tails() {
    for len in 0..=67 {
        let left = (0..len)
            .map(|index| index as f32 * 0.25 - 5.0)
            .collect::<Vec<_>>();
        let right = (0..len)
            .map(|index| 3.0 - index as f32 * 0.125)
            .collect::<Vec<_>>();
        assert_close(
            detect().squared_l2(&left, &right),
            squared_l2_scalar(&left, &right),
        );
    }

    assert!(!detect().name().is_empty());
    #[cfg(target_arch = "aarch64")]
    assert_eq!(detect().name(), "neon");
    #[cfg(target_arch = "x86_64")]
    assert!(matches!(detect().name(), "scalar" | "avx2"));
}

#[test]
fn detected_kernel_matches_scalar_row_norms_across_dimensions() {
    for dimension in [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33] {
        let values = (0..(3 * dimension))
            .map(|index| index as f32 * 0.1 - 2.0)
            .collect::<Vec<_>>();
        let expected = values
            .chunks_exact(dimension)
            .map(norm_squared_scalar)
            .collect::<Vec<_>>();
        for (actual, expected) in detect().row_norms(&values, dimension).iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }
}

#[test]
fn distance_propagates_non_finite_values() {
    assert!(!detect().squared_l2(&[f32::NAN], &[0.0]).is_finite());
    assert!(!detect().squared_l2(&[f32::INFINITY], &[0.0]).is_finite());
}

#[test]
fn gemm_computes_row_dot_products() {
    let left = [1.0, 2.0, 3.0, 4.0];
    let right = [2.0, 0.0, 0.0, 2.0, 1.0, 1.0];
    let mut output = [0.0_f32; 6];
    row_dot_products(&left, &right, 2, 2, 3, &mut output).unwrap();
    for (actual, expected) in output.into_iter().zip([2.0, 4.0, 3.0, 6.0, 8.0, 7.0]) {
        assert_close(actual, expected);
    }
}

#[test]
fn gemm_matches_scalar_for_rectangular_matrices() {
    let rows = 5;
    let dimension = 7;
    let columns = 11;
    let left = (0..rows * dimension)
        .map(|value| value as f32 * 0.03 - 0.5)
        .collect::<Vec<_>>();
    let right = (0..columns * dimension)
        .map(|value| value as f32 * -0.02 + 0.75)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; rows * columns];
    row_dot_products(&left, &right, rows, dimension, columns, &mut output).unwrap();

    for row in 0..rows {
        for column in 0..columns {
            let expected = left[row * dimension..(row + 1) * dimension]
                .iter()
                .zip(&right[column * dimension..(column + 1) * dimension])
                .map(|(left, right)| left * right)
                .sum();
            assert_close(output[row * columns + column], expected);
        }
    }
}

#[test]
fn gemm_validates_declared_shapes() {
    let invalid = [
        row_dot_products(&[], &[], 0, 0, 1, &mut []),
        row_dot_products(&[], &[], 0, 1, 0, &mut []),
        row_dot_products(&[1.0], &[1.0], 2, 1, 1, &mut [0.0, 0.0]),
        row_dot_products(&[1.0], &[1.0], 1, 1, 2, &mut [0.0, 0.0]),
        row_dot_products(&[1.0], &[1.0], 1, 1, 1, &mut []),
        row_dot_products(&[], &[], usize::MAX, 2, 1, &mut []),
    ];

    assert!(invalid.into_iter().all(|result| result.is_err()));
}

#[test]
fn gemm_accepts_zero_rows_with_valid_right_shape() {
    let mut output = [];
    row_dot_products(&[], &[1.0, 2.0, 3.0, 4.0], 0, 2, 2, &mut output).unwrap();
}

#[test]
fn detected_kernel_is_safe_to_share_across_threads() {
    let handles = (0..8)
        .map(|thread| {
            std::thread::spawn(move || {
                let left = vec![thread as f32; 65];
                let right = vec![1.0; 65];
                detect().squared_l2(&left, &right)
            })
        })
        .collect::<Vec<_>>();

    for (thread, handle) in handles.into_iter().enumerate() {
        assert_close(handle.join().unwrap(), 65.0 * (thread as f32 - 1.0).powi(2));
    }
}

fn assert_close(actual: f32, expected: f32) {
    let tolerance = 1e-4_f32.max(expected.abs() * 1e-5);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn batch_kernel_matches_independent_rows_across_tails() {
    let kernel = *detect();
    for rows in [0, 1, 3, 4, 5, 9] {
        for dimension in [1, 3, 4, 7, 8, 15, 16, 33, 128] {
            let values = (0..rows * dimension)
                .map(|index| (index % 251) as f32 * 0.001)
                .collect::<Vec<_>>();
            let query = (0..dimension)
                .map(|index| (index % 127) as f32 * 0.002)
                .collect::<Vec<_>>();
            let expected = values
                .chunks_exact(dimension)
                .map(|row| kernel.squared_l2(row, &query))
                .collect::<Vec<_>>();
            let mut actual = vec![0.0; rows];

            kernel.squared_l2_rows(&values, &query, &mut actual);

            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_close(actual, expected);
            }
        }
    }
}

#[test]
fn pair_kernel_matches_independent_rows_across_tails() {
    let kernel = *detect();
    for rows in [0, 1, 3, 8] {
        for dimension in [1, 3, 4, 7, 8, 15, 16, 33, 128] {
            let left = (0..rows * dimension)
                .map(|index| (index % 251) as f32 * 0.001 - 0.25)
                .collect::<Vec<_>>();
            let right = (0..rows * dimension)
                .map(|index| 0.5 - (index % 127) as f32 * 0.002)
                .collect::<Vec<_>>();
            let expected = left
                .chunks_exact(dimension)
                .zip(right.chunks_exact(dimension))
                .map(|(left, right)| squared_l2_scalar(left, right))
                .collect::<Vec<_>>();
            let mut actual = vec![0.0; rows];

            kernel.squared_l2_pairs(&left, &right, dimension, &mut actual);

            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_close(actual, expected);
            }
        }
    }
}

#[test]
fn nearest_row_kernel_matches_scalar_search() {
    let kernel = *detect();
    for rows in [0, 1, 5] {
        for reference_rows in [1, 3, 9] {
            for dimension in [1, 3, 8, 17, 64] {
                let values = (0..rows * dimension)
                    .map(|index| (index % 97) as f32 * 0.013 - 0.4)
                    .collect::<Vec<_>>();
                let references = (0..reference_rows * dimension)
                    .map(|index| 0.7 - (index % 89) as f32 * 0.009)
                    .collect::<Vec<_>>();
                let expected = values
                    .chunks_exact(dimension)
                    .map(|value| {
                        references
                            .chunks_exact(dimension)
                            .enumerate()
                            .map(|(index, reference)| (index, squared_l2_scalar(value, reference)))
                            .min_by(|left, right| left.1.total_cmp(&right.1))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let mut indices = vec![0; rows];
                let mut distances = vec![0.0; rows];

                kernel.nearest_squared_l2_rows(
                    &values,
                    &references,
                    dimension,
                    &mut indices,
                    &mut distances,
                );

                for ((actual_index, actual_distance), (expected_index, expected_distance)) in
                    indices.into_iter().zip(distances).zip(expected)
                {
                    assert_eq!(actual_index, expected_index);
                    assert_close(actual_distance, expected_distance);
                }
            }
        }
    }
}
