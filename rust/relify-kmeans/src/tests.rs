//! K-means training, sampling, assignment, and memory-bound tests.

use super::*;
use parallite::Executor;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn finds_separated_clusters() {
    let mut vectors = Vec::new();
    for index in 0..100 {
        let jitter = index as f32 * 0.001;
        vectors.extend_from_slice(&[jitter, -jitter]);
        vectors.extend_from_slice(&[10.0 + jitter, 10.0 - jitter]);
    }
    let model = fit_lloyd_kmeans(
        &vectors,
        2,
        &ParalliteContext::default(),
        KMeansOptions::new(2),
    )
    .unwrap();
    assert_eq!(model.centroids.len(), 4);
    assert!(model.iterations < KMeansOptions::new(2).max_iter);
    let mut first_coordinates = model
        .centroids
        .chunks_exact(2)
        .map(|centroid| centroid[0])
        .collect::<Vec<_>>();
    first_coordinates.sort_by(f32::total_cmp);
    assert!(first_coordinates[0] < 0.2);
    assert!(first_coordinates[1] > 9.8);
}

#[test]
fn serial_and_parallel_results_match() {
    let vectors = (0..64)
        .flat_map(|index| [index as f32, (index % 7) as f32])
        .collect::<Vec<_>>();
    let options = KMeansOptions {
        n_clusters: 4,
        max_iter: 10,
        seed: 99,
        mode: KMeansMode::Flat,
    };
    let serial = ParalliteContext::with_executor(Executor::serial());
    let parallel = ParalliteContext::builder().threads(4).unwrap().build();
    assert_eq!(
        fit_lloyd_kmeans(&vectors, 2, &serial, options).unwrap(),
        fit_lloyd_kmeans(&vectors, 2, &parallel, options).unwrap()
    );
}

#[test]
fn training_progress_reports_rows_from_every_iteration() {
    let vectors = (0..128)
        .flat_map(|index| [index as f32, (index % 11) as f32])
        .collect::<Vec<_>>();
    let options = KMeansOptions {
        n_clusters: 4,
        max_iter: 5,
        seed: 42,
        mode: KMeansMode::Flat,
    };
    let assigned_rows = AtomicUsize::new(0);
    let model = fit_lloyd_kmeans_with_progress(
        &vectors,
        2,
        &ParalliteContext::default(),
        options,
        |rows| {
            assigned_rows.fetch_add(rows, Ordering::Relaxed);
        },
    )
    .unwrap();

    assert_eq!(
        assigned_rows.load(Ordering::Relaxed),
        128 * model.iterations
    );
}

#[test]
fn hierarchical_shape_preserves_requested_cluster_count() {
    let (roots, children) = hierarchical_shape(8_192);
    assert_eq!(roots, 91);
    assert_eq!(children.iter().sum::<usize>(), 8_192);
    assert!(children.iter().all(|&count| count == 90 || count == 91));

    let (roots, children) = hierarchical_shape(65_536);
    assert_eq!(roots, 256);
    assert!(children.iter().all(|&count| count == 256));
}

#[test]
fn root_partitions_need_enough_rows_for_every_leaf_centroid() {
    assert!(has_sufficient_root_rows(&[0, 100, 200], &[64, 64]));
    assert!(!has_sufficient_root_rows(&[0, 127, 200], &[128, 64]));
}

#[test]
fn partition_rows_rejects_out_of_range_labels() {
    let error = partition_rows(&[0, 2], 2).unwrap_err();

    assert!(error.to_string().contains("label 2"));
    assert!(error.to_string().contains("cluster count 2"));
}

#[test]
fn auto_mode_uses_hierarchical_training_only_for_large_cluster_counts() {
    assert_eq!(resolved_mode(KMeansOptions::new(8_191)), KMeansMode::Flat);
    assert_eq!(
        resolved_mode(KMeansOptions::new(8_192)),
        KMeansMode::Hierarchical
    );
    assert_eq!(
        resolved_mode(KMeansOptions {
            n_clusters: 8_192,
            max_iter: 1,
            seed: 42,
            mode: KMeansMode::Flat,
        }),
        KMeansMode::Flat
    );
}

#[test]
fn hierarchical_mode_returns_flattened_leaf_centroids() {
    let vectors = (0..8)
        .flat_map(|root| {
            (0..512).flat_map(move |row| {
                [
                    root as f32 * 1_000.0 + row as f32 * 0.001,
                    root as f32 * -1_000.0 + (row % 17) as f32 * 0.001,
                ]
            })
        })
        .collect::<Vec<_>>();
    let options = KMeansOptions {
        n_clusters: 64,
        max_iter: 4,
        seed: 42,
        mode: KMeansMode::Hierarchical,
    };
    let serial = ParalliteContext::with_executor(Executor::serial());
    let parallel = ParalliteContext::builder().threads(4).unwrap().build();
    let first = fit_lloyd_kmeans(&vectors, 2, &serial, options).unwrap();
    let second = fit_lloyd_kmeans(&vectors, 2, &parallel, options).unwrap();

    assert_eq!(first.centroids.len(), 64 * 2);
    assert!(first.centroids.iter().all(|value| value.is_finite()));
    assert_eq!(first, second);
}

#[test]
fn hierarchical_failure_falls_back_only_in_auto_mode() {
    let vectors = vec![1.0_f32; 128];
    let context = ParalliteContext::with_executor(Executor::serial());
    let mut options = KMeansOptions {
        n_clusters: 64,
        max_iter: 1,
        seed: 42,
        mode: KMeansMode::Auto,
    };

    let model = fit_hierarchical_kmeans_with_progress(
        &vectors,
        1,
        vectors.len(),
        &context,
        options,
        &|_| {},
    )
    .unwrap();
    assert_eq!(model.centroids.len(), options.n_clusters);
    assert!(model.centroids.iter().all(|value| value.is_finite()));

    options.mode = KMeansMode::Hierarchical;
    let error = fit_hierarchical_kmeans_with_progress(
        &vectors,
        1,
        vectors.len(),
        &context,
        options,
        &|_| {},
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("root partitions cannot form the requested leaf centroids")
    );
}

#[test]
fn rejects_invalid_fit_inputs() {
    let context = ParalliteContext::default();
    let cases = [
        (vec![0.0, 1.0], 0, KMeansOptions::new(1)),
        (Vec::new(), 2, KMeansOptions::new(1)),
        (vec![0.0, 1.0, 2.0], 2, KMeansOptions::new(1)),
        (vec![0.0, f32::NAN], 2, KMeansOptions::new(1)),
        (vec![0.0, f32::INFINITY], 2, KMeansOptions::new(1)),
        (vec![0.0, 1.0], 2, KMeansOptions::new(0)),
        (
            vec![0.0, 1.0],
            2,
            KMeansOptions {
                n_clusters: 1,
                max_iter: 0,
                seed: 42,
                mode: KMeansMode::Flat,
            },
        ),
        (vec![0.0, 1.0], 2, KMeansOptions::new(2)),
    ];

    for (vectors, dimension, options) in cases {
        assert!(
            fit_lloyd_kmeans(&vectors, dimension, &context, options).is_err(),
            "vectors={vectors:?}, dimension={dimension}, options={options:?}"
        );
    }
}

#[test]
fn assignment_rejects_malformed_and_non_finite_matrices() {
    let context = ParalliteContext::default();
    let cases = [
        (vec![0.0, 1.0], 0, vec![0.0, 1.0]),
        (vec![0.0, 1.0, 2.0], 2, vec![0.0, 1.0]),
        (vec![0.0, 1.0], 2, vec![0.0]),
        (vec![0.0, f32::NAN], 2, vec![0.0, 1.0]),
        (vec![0.0, 1.0], 2, vec![f32::INFINITY, 1.0]),
    ];

    for (vectors, dimension, centroids) in cases {
        assert!(assign_to_centroids(&vectors, dimension, &centroids, &context).is_err());
        assert!(assign_batch_to_centroids(&vectors, dimension, &centroids).is_err());
    }
}

#[test]
fn batch_assignment_matches_parallel_assignment() {
    for (dimension, n_clusters, rows) in [(2, 4, 128), (8, 16, 2_049)] {
        let centroids = (0..n_clusters)
            .flat_map(|cluster| {
                (0..dimension).map(move |value| (cluster * dimension + value) as f32)
            })
            .collect::<Vec<_>>();
        let vectors = (0..rows)
            .flat_map(|row| {
                let cluster = row % n_clusters;
                (0..dimension)
                    .map(move |value| (cluster * dimension + value) as f32 + row as f32 * 0.001)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            assign_batch_to_centroids(&vectors, dimension, &centroids).unwrap(),
            assign_to_centroids(
                &vectors,
                dimension,
                &centroids,
                &ParalliteContext::default()
            )
            .unwrap()
        );
    }
}

#[test]
fn training_sample_is_reproducible_and_without_replacement() {
    let vectors = (0..100).map(|value| value as f32).collect::<Vec<_>>();
    let first = sample_training_rows(&vectors, 2, 10, 99).unwrap();
    let second = sample_training_rows(&vectors, 2, 10, 99).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 20);

    let rows = first.chunks_exact(2).collect::<Vec<_>>();
    let unique = rows
        .iter()
        .map(|row| (row[0].to_bits(), row[1].to_bits()))
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), rows.len());
}

#[test]
fn streaming_reservoir_is_bounded_and_batch_independent() {
    let mut left = ReservoirSampler::new(4, 42).unwrap();
    let mut right = ReservoirSampler::new(4, 42).unwrap();
    let values = (0..40).map(|value| value as f32).collect::<Vec<_>>();
    left.push(&values, 2).unwrap();
    right.push(&values[..20], 2).unwrap();
    right.push(&values[20..], 2).unwrap();

    assert_eq!(left.dimension(), Some(2));
    assert_eq!(left.seen_rows(), 20);
    assert_eq!(left.values().len(), 8);
    assert_eq!(left.values.capacity(), 8);
    assert_eq!(left.values(), right.values());
    assert_ne!(left.values(), &values[..8]);
}

#[test]
fn streaming_reservoir_validates_shape_and_limits() {
    assert!(ReservoirSampler::new(0, 42).is_err());

    let mut sampler = ReservoirSampler::new(2, 42).unwrap();
    assert!(sampler.push(&[0.0, 1.0], 0).is_err());
    assert!(sampler.push(&[0.0, 1.0, 2.0], 2).is_err());
    sampler.push(&[0.0, 1.0], 2).unwrap();
    assert!(sampler.push(&[0.0, 1.0, 2.0], 3).is_err());
}

#[test]
fn sparse_sampling_matches_partial_fisher_yates() {
    for (n, sample_size, seed) in [(2, 1, 1), (17, 8, 42), (1_000, 31, 99)] {
        let mut sparse_rng = SmallRng::new(seed);
        let sparse = sample_without_replacement(n, sample_size, &mut sparse_rng);

        let mut dense_rng = SmallRng::new(seed);
        let mut dense = (0..n).collect::<Vec<_>>();
        for index in 0..sample_size {
            let swap = index + dense_rng.gen_usize(n - index);
            dense.swap(index, swap);
        }
        dense.truncate(sample_size);

        assert_eq!(sparse, dense);
    }
}

#[test]
fn training_sample_validates_shape_and_limits() {
    for (vectors, dimension, max_rows) in [
        (Vec::new(), 2, 1),
        (vec![0.0, 1.0], 0, 1),
        (vec![0.0, 1.0, 2.0], 2, 1),
        (vec![0.0, f32::NAN], 2, 1),
        (vec![0.0, 1.0], 2, 0),
    ] {
        assert!(sample_training_rows(&vectors, dimension, max_rows, 42).is_err());
    }
}

#[test]
fn identical_vectors_keep_empty_cluster_recovery_finite() {
    let vectors = vec![0.0; 16];
    let options = KMeansOptions {
        n_clusters: 4,
        max_iter: 5,
        seed: 42,
        mode: KMeansMode::Flat,
    };
    let model = fit_lloyd_kmeans(&vectors, 2, &ParalliteContext::default(), options).unwrap();

    assert_eq!(model.centroids.len(), 8);
    assert!(model.centroids.iter().all(|value| value.is_finite()));
}

#[test]
fn one_centroid_per_vector_is_an_exact_zero_iteration_model() {
    let vectors = vec![0.0, 1.0, 10.0, 11.0];
    let model = fit_lloyd_kmeans(
        &vectors,
        2,
        &ParalliteContext::default(),
        KMeansOptions::new(2),
    )
    .unwrap();

    assert_eq!(model.centroids, vectors);
    assert_eq!(model.iterations, 0);
}

#[test]
fn gemm_assignment_matches_known_clusters() {
    let dimension = GEMM_MIN_DIMENSION;
    let n_clusters = GEMM_MIN_CLUSTERS;
    assert!(uses_gemm(dimension, n_clusters));
    assert!(!uses_gemm(dimension - 1, n_clusters));
    assert_eq!(gemm_partition_rows(n_clusters), PARTITION_ROWS);
    assert_eq!(gemm_partition_rows(GEMM_MAX_OUTPUT_VALUES), 1);

    let centroids = (0..n_clusters)
        .flat_map(|cluster| std::iter::repeat_n(cluster as f32 * 10.0, dimension))
        .collect::<Vec<_>>();
    let vectors = (0..64)
        .flat_map(|row| {
            let cluster = row % n_clusters;
            std::iter::repeat_n(cluster as f32 * 10.0 + 0.25, dimension)
        })
        .collect::<Vec<_>>();
    let assignments = assign_to_centroids(
        &vectors,
        dimension,
        &centroids,
        &ParalliteContext::default(),
    )
    .unwrap();

    assert_eq!(
        assignments,
        (0..64).map(|row| row % n_clusters).collect::<Vec<_>>()
    );
}

#[test]
fn parallel_assignment_reduces_cluster_sums() {
    let dimension = GEMM_MIN_DIMENSION;
    let n_clusters = GEMM_MIN_CLUSTERS;
    let centroids = (0..n_clusters)
        .flat_map(|cluster| std::iter::repeat_n(cluster as f32 * 10.0, dimension))
        .collect::<Vec<_>>();
    let vectors = (0..64)
        .flat_map(|row| {
            let cluster = row % n_clusters;
            std::iter::repeat_n(cluster as f32 * 10.0 + 0.25, dimension)
        })
        .collect::<Vec<_>>();
    let row_norms = detect().row_norms(&vectors, dimension);
    let mut workspace = TrainingWorkspace::new(64, n_clusters, dimension, 4);
    let batch = assign_and_sum_all(
        &ParalliteContext::default(),
        &vectors,
        64,
        dimension,
        &centroids,
        &row_norms,
        &mut workspace,
        &|_| {},
    )
    .unwrap();

    assert_eq!(batch.counts, vec![4; n_clusters]);
    for cluster in 0..n_clusters {
        let expected = 4.0 * (cluster as f32 * 10.0 + 0.25);
        assert!(
            batch.sums[cluster * dimension..(cluster + 1) * dimension]
                .iter()
                .all(|value| (*value - expected).abs() < f32::EPSILON)
        );
    }
}

#[test]
fn training_workspace_reuses_dense_buffers_between_iterations() {
    let dimension = GEMM_MIN_DIMENSION;
    let n_clusters = GEMM_MIN_CLUSTERS;
    let centroids = (0..n_clusters)
        .flat_map(|cluster| std::iter::repeat_n(cluster as f32, dimension))
        .collect::<Vec<_>>();
    let vectors = (0..64)
        .flat_map(|row| std::iter::repeat_n((row % n_clusters) as f32, dimension))
        .collect::<Vec<_>>();
    let row_norms = detect().row_norms(&vectors, dimension);
    let mut workspace = TrainingWorkspace::new(64, n_clusters, dimension, 4);
    let partial_buffers = workspace
        .partitions
        .iter()
        .map(|partition| partition.assignment_cells.as_ptr())
        .collect::<Vec<_>>();
    let merged_buffers = (
        workspace.merged.sums.as_ptr(),
        workspace.merged.counts.as_ptr(),
    );

    for _ in 0..2 {
        assign_and_sum_all(
            &ParalliteContext::default(),
            &vectors,
            64,
            dimension,
            &centroids,
            &row_norms,
            &mut workspace,
            &|_| {},
        )
        .unwrap();
    }

    assert_eq!(
        workspace
            .partitions
            .iter()
            .map(|partition| partition.assignment_cells.as_ptr())
            .collect::<Vec<_>>(),
        partial_buffers
    );
    assert_eq!(
        (
            workspace.merged.sums.as_ptr(),
            workspace.merged.counts.as_ptr()
        ),
        merged_buffers
    );
}

#[test]
fn training_partitions_follow_the_executor() {
    assert_eq!(training_partition_count(128, 32), 32);
    assert_eq!(TrainingWorkspace::new(128, 1, 1, 32).partitions.len(), 32);
    assert_eq!(training_partition_count(3, 32), 3);
}
