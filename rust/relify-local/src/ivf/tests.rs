//! Local Arrow IVF relation tests.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float32Builder, Int32Array, Int64Array, ListBuilder,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use super::*;

fn vector_array(values: &[[f32; 2]]) -> ArrayRef {
    let mut builder = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for vector in values {
        builder.values().append_slice(vector);
        builder.append(true);
    }
    Arc::new(builder.finish())
}

#[test]
fn extracts_sliced_fixed_size_list_without_prefix_values() {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2)
        .with_field(Arc::new(Field::new("element", DataType::Float32, false)));
    for vector in [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]] {
        builder.values().append_slice(&vector);
        builder.append(true);
    }
    let array: ArrayRef = Arc::new(builder.finish());
    let sliced = array.slice(1, 2);

    let (vectors, dimension) = borrow_vectors(&sliced).unwrap();
    assert_eq!(dimension, 2);
    assert_eq!(vectors, &[2.0, 3.0, 4.0, 5.0]);
    let fixed = sliced
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let child = fixed
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(
        vectors.as_ptr(),
        child.values().as_ptr().wrapping_add(fixed.offset() * 2)
    );
}

#[test]
fn extracts_sliced_variable_list_without_prefix_values() {
    let array = vector_array(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);
    let sliced = array.slice(1, 2);

    let (vectors, dimension) = borrow_vectors(&sliced).unwrap();
    assert_eq!(dimension, 2);
    assert_eq!(vectors, &[2.0, 3.0, 4.0, 5.0]);
    let list = sliced.as_any().downcast_ref::<ListArray>().unwrap();
    let child = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let offset = usize::try_from(list.value_offsets()[0]).unwrap();
    assert_eq!(
        vectors.as_ptr(),
        child.values().as_ptr().wrapping_add(offset)
    );
}

#[test]
fn postings_resolve_composite_keys_directly() {
    let source_keys = vec![
        Arc::new(StringArray::from(vec!["b", "a"])) as ArrayRef,
        Arc::new(Int64Array::from(vec![2, 1])) as ArrayRef,
    ];
    let source_by_key = source_rows_by_key(&source_keys).unwrap();
    let postings = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int32, false),
            Field::new("key_1", DataType::Utf8, false),
            Field::new("key_2", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();

    let candidates =
        candidate_source_rows(&postings, &[true, true], 2, &source_keys, &source_by_key).unwrap();
    assert_eq!(
        candidates
            .into_iter()
            .map(|(_, source_row, _)| source_row)
            .collect::<Vec<_>>(),
        [1, 0]
    );
}

#[test]
fn postings_require_matching_non_nullable_key_columns() {
    let source_keys = vec![Arc::new(StringArray::from(vec!["a"])) as ArrayRef];
    let source_by_key = source_rows_by_key(&source_keys).unwrap();
    for (data_type, nullable) in [(DataType::Int64, false), (DataType::Utf8, true)] {
        let values: ArrayRef = if data_type == DataType::Int64 {
            Arc::new(Int64Array::from(vec![1]))
        } else {
            Arc::new(StringArray::from(vec!["a"]))
        };
        let postings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cid", DataType::Int32, false),
                Field::new("key_1", data_type, nullable),
            ])),
            vec![Arc::new(Int32Array::from(vec![0])), values],
        )
        .unwrap();
        assert!(
            candidate_source_rows(&postings, &[true], 1, &source_keys, &source_by_key).is_err()
        );
    }
}

#[test]
fn centroids_require_exact_non_nullable_schema() {
    let centroids = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int32, false),
            Field::new(
                "centroid",
                vector_array(&[[0.0, 0.0]]).data_type().clone(),
                true,
            ),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0])),
            vector_array(&[[0.0, 0.0]]),
        ],
    )
    .unwrap();

    assert!(read_centroids(&centroids, 1, 2).is_err());
}

#[test]
fn duplicate_source_keys_are_rejected() {
    let source_keys = vec![Arc::new(StringArray::from(vec!["a", "a"])) as ArrayRef];
    assert!(source_rows_by_key(&source_keys).is_err());
}

#[test]
fn cluster_ties_select_the_smaller_cid() {
    let selected = select_clusters(&[0.0, 0.0], &[1.0, 0.0, -1.0, 0.0], 2, 1);
    assert_eq!(selected, [true, false]);
}
