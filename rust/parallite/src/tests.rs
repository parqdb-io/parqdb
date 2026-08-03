use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

#[test]
fn map_flatten_partition_by_keeps_partition_order() {
    let output = ParalliteContext::default()
        .parallelize_n(vec![0, 1, 2, 3], 2)
        .flat_map(|value| Ok::<_, ()>(vec![(value % 2, value), (value % 2, value + 10)]))
        .partition_by(2, |key| Ok::<_, ()>(*key))
        .collect_partitions()
        .unwrap()
        .into_partition_iter()
        .collect::<Vec<_>>();

    assert_eq!(output[0], vec![(0, 0), (0, 10), (0, 2), (0, 12)]);
    assert_eq!(output[1], vec![(1, 1), (1, 11), (1, 3), (1, 13)]);
}

#[test]
fn partition_by_is_lazy_until_collect_partitions() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let pipeline = parallelize(vec![(0, 1), (0, 2)])
        .map(move |item| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ()>(item)
        })
        .partition_by(1, |key| Ok::<_, ()>(*key));

    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let output = pipeline.collect_partitions().unwrap().collect();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(output, vec![(0, 1), (0, 2)]);
}

#[test]
fn partition_by_rejects_invalid_partition_counts_and_ids() {
    let zero = parallelize(vec![(0, "x")])
        .partition_by(0, |key| Ok::<_, ()>(*key))
        .collect_partitions()
        .err()
        .unwrap();
    assert!(matches!(zero, ParalliteError::InvalidParameter(_)));

    let out_of_range = parallelize(vec![(3, "x")])
        .partition_by(2, |key| Ok::<_, ()>(*key))
        .collect_partitions()
        .err()
        .unwrap();
    assert!(matches!(
        out_of_range,
        ParalliteError::InvalidPartition { .. }
    ));
}

#[test]
fn user_errors_propagate_from_each_transform_boundary() {
    let map_error = parallelize(vec![0, 1, 2])
        .map(|value| if value == 1 { Err("map") } else { Ok(value) })
        .collect_partitions()
        .err()
        .unwrap();
    assert_eq!(map_error, ParalliteError::User("map"));

    let partition_error = parallelize(vec![(0, 1), (1, 2)])
        .partition_by(2, |key| if *key == 1 { Err("partition") } else { Ok(0) })
        .collect_partitions()
        .err()
        .unwrap();
    assert_eq!(partition_error, ParalliteError::User("partition"));

    let map_partitions_error = parallelize(vec![0, 1])
        .map(Ok::<_, &str>)
        .map_partitions(|_iter| Err::<Vec<i32>, _>("map_partitions"))
        .collect_partitions()
        .err()
        .unwrap();
    assert_eq!(map_partitions_error, ParalliteError::User("map_partitions"));
}

#[test]
fn map_partitions_runs_after_materialization() {
    let output = parallelize(vec![(0, 1), (1, 2)])
        .partition_by(2, |key| Ok::<_, ()>(*key))
        .map_partitions(|iter| Ok::<_, ()>(iter.map(|(key, value)| (key, value * 2))))
        .collect_partitions()
        .unwrap()
        .collect();

    assert_eq!(output, vec![(0, 2), (1, 4)]);
}

#[test]
fn source_parallelize_uses_coarser_partitions_than_elements() {
    let context = ParalliteContext::with_executor(Executor::with_threads(2).unwrap());
    let output = context
        .parallelize((0..20).collect())
        .map(Ok::<_, ()>)
        .collect_partitions()
        .unwrap();

    assert_eq!(output.partition_count(), 8);
}

#[test]
fn context_parallelize_n_uses_requested_source_partitions() {
    let output = ParalliteContext::default()
        .parallelize_n((0..10).collect(), 3)
        .map(Ok::<_, ()>)
        .collect_partitions()
        .unwrap();

    assert_eq!(output.partition_count(), 3);
}

#[test]
fn zero_source_slices_are_normalized_to_one_partition() {
    let output = ParalliteContext::default()
        .parallelize_n((0..3).collect(), 0)
        .map(Ok::<_, ()>)
        .collect_partitions()
        .unwrap();

    assert_eq!(output.partition_count(), 1);
    assert_eq!(output.collect(), [0, 1, 2]);
}

#[test]
fn context_builder_sets_default_slices() {
    let context = ParalliteContext::builder().default_slices(5).build();
    let output = context
        .parallelize((0..20).collect())
        .map(Ok::<_, ()>)
        .collect_partitions()
        .unwrap();

    assert_eq!(output.partition_count(), 5);
    assert_eq!(context.options().default_slices(), Some(5));
}

#[test]
fn executor_rejects_zero_threads_and_uses_serial_for_one() {
    assert!(Executor::with_threads(0).is_err());
    assert_eq!(Executor::with_threads(1).unwrap().thread_count(), 1);
}

#[test]
fn parallel_collect_preserves_source_order() {
    let context = ParalliteContext::builder().threads(4).unwrap().build();
    let output = context
        .parallelize_n((0..10_000).collect(), 32)
        .map(|value| Ok::<_, ()>(value * 2))
        .collect_partitions()
        .unwrap()
        .collect();

    assert_eq!(
        output,
        (0..10_000).map(|value| value * 2).collect::<Vec<_>>()
    );
}

#[test]
fn shared_executor_supports_concurrent_pipelines() {
    let context = ParalliteContext::builder().threads(4).unwrap().build();
    let handles = (0..8)
        .map(|run| {
            let context = context.clone();
            std::thread::spawn(move || {
                context
                    .parallelize_n((0..1_000).collect(), 16)
                    .map(move |value| Ok::<_, ()>(value + run * 1_000))
                    .collect_partitions()
                    .unwrap()
                    .collect()
            })
        })
        .collect::<Vec<_>>();

    for (run, handle) in handles.into_iter().enumerate() {
        let output = handle.join().unwrap();
        assert_eq!(output[0], run * 1_000);
        assert_eq!(output[999], run * 1_000 + 999);
    }
}

#[test]
fn worker_panic_propagates_without_poisoning_the_executor() {
    let context = ParalliteContext::builder().threads(2).unwrap().build();
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = context
            .parallelize_n(vec![0, 1, 2], 3)
            .map(|value| -> Result<_, ()> {
                assert_ne!(value, 1, "intentional worker panic");
                Ok(value)
            })
            .collect_partitions();
    }));
    assert!(panicked.is_err());

    let recovered = context
        .parallelize(vec![1, 2, 3])
        .map(Ok::<_, ()>)
        .collect_partitions()
        .unwrap()
        .collect();
    assert_eq!(recovered, [1, 2, 3]);
}
