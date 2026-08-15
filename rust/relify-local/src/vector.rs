//! Canonical vector conversion and metric pretransforms.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, LargeListArray, ListArray};
use arrow::buffer::OffsetBuffer;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::DataFusionError;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use relify_meta::DistanceMetric;

use crate::{Error, Result};

#[derive(Debug)]
struct TransformVector {
    id: u64,
    signature: Signature,
    metric: DistanceMetric,
}

impl PartialEq for TransformVector {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TransformVector {}

impl Hash for TransformVector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl ScalarUDFImpl for TransformVector {
    fn name(&self) -> &'static str {
        match self.metric {
            DistanceMetric::L2Squared => "relify_vector_f32",
            DistanceMetric::Cosine => "relify_normalize_vector",
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        let [argument] = argument_types else {
            return Err(DataFusionError::Plan(format!(
                "{} requires exactly one argument",
                self.name()
            )));
        };
        canonical_vector_type(argument).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "{} requires a list<float> or list<double> argument",
                self.name()
            ))
        })
    }

    fn coerce_types(
        &self,
        argument_types: &[DataType],
    ) -> datafusion::common::Result<Vec<DataType>> {
        let [argument] = argument_types else {
            return Err(DataFusionError::Plan(format!(
                "{} requires exactly one argument",
                self.name()
            )));
        };
        canonical_vector_type(argument).map_or_else(
            || {
                Err(DataFusionError::Plan(format!(
                    "{} requires a list<float> or list<double> argument",
                    self.name()
                )))
            },
            |_| Ok(vec![argument.clone()]),
        )
    }

    fn return_field_from_args(
        &self,
        arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        let types = arguments
            .arg_fields
            .iter()
            .map(|field| field.data_type().clone())
            .collect::<Vec<_>>();
        Ok(Arc::new(Field::new(
            self.name(),
            self.return_type(&types)?,
            false,
        )))
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        let [argument] = arguments.args.as_slice() else {
            return Err(DataFusionError::Execution(format!(
                "{} requires exactly one argument",
                self.name()
            )));
        };
        let input = argument.to_array_of_size(arguments.number_rows)?;
        let output_type = canonical_vector_type(input.data_type()).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "{} requires a list<float> or list<double> argument",
                self.name()
            ))
        })?;
        let validation_type = canonical_vector_type_with_nullability(input.data_type(), true)
            .expect("validated vector type must have a validation type");
        let canonical = cast(input.as_ref(), &validation_type)
            .map_err(|error| invalid_source_error(error.to_string()))?;
        if canonical.is_empty() {
            return Ok(ColumnarValue::Array(canonical));
        }
        let (values, dimension) = crate::ivf::borrow_vectors_allow_nullable_elements(&canonical)
            .map_err(invalid_source_schema)?;
        if self.metric == DistanceMetric::L2Squared {
            return Ok(ColumnarValue::Array(
                cast(canonical.as_ref(), &output_type).map_err(|_| {
                    invalid_source_error(
                        "every vector must contain finite, non-null float values".into(),
                    )
                })?,
            ));
        }
        let normalized = normalize_rows(values, dimension).map_err(invalid_source_schema)?;
        Ok(ColumnarValue::Array(rebuild_vector_array(
            &output_type,
            canonical.len(),
            dimension,
            normalized,
        )?))
    }
}

fn invalid_source_schema(error: Error) -> DataFusionError {
    let message = match error {
        Error::InvalidSchema(message) | Error::InvalidArgument(message) => message,
        other => other.to_string(),
    };
    invalid_source_error(message)
}

fn invalid_source_error(message: String) -> DataFusionError {
    crate::error::invalid_schema_datafusion(message)
}

/// Returns the canonical Arrow vector type for a supported source type.
pub(crate) fn canonical_vector_type(data_type: &DataType) -> Option<DataType> {
    canonical_vector_type_with_nullability(data_type, false)
}

fn canonical_vector_type_with_nullability(
    data_type: &DataType,
    element_nullable: bool,
) -> Option<DataType> {
    match data_type {
        DataType::List(field)
            if matches!(field.data_type(), DataType::Float32 | DataType::Float64) =>
        {
            Some(DataType::List(Arc::new(Field::new(
                field.name(),
                DataType::Float32,
                element_nullable,
            ))))
        }
        DataType::LargeList(field)
            if matches!(field.data_type(), DataType::Float32 | DataType::Float64) =>
        {
            Some(DataType::LargeList(Arc::new(Field::new(
                field.name(),
                DataType::Float32,
                element_nullable,
            ))))
        }
        DataType::FixedSizeList(field, dimension)
            if *dimension > 0
                && matches!(field.data_type(), DataType::Float32 | DataType::Float64) =>
        {
            Some(DataType::FixedSizeList(
                Arc::new(Field::new(
                    field.name(),
                    DataType::Float32,
                    element_nullable,
                )),
                *dimension,
            ))
        }
        _ => None,
    }
}

/// Returns the build and query UDF for one metric's canonical transform.
pub(crate) fn transform_vector_udf(metric: DistanceMetric) -> ScalarUDF {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    ScalarUDF::new_from_impl(TransformVector {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        signature: Signature::user_defined(Volatility::Immutable),
        metric,
    })
}

/// Canonicalizes one query vector according to `metric`.
pub(crate) fn transform_query(query: &[f32], metric: DistanceMetric) -> Result<Vec<f32>> {
    if query.is_empty() || query.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(
            "query vector must contain finite float values and must not be empty".into(),
        ));
    }
    if metric == DistanceMetric::L2Squared {
        return Ok(query.to_vec());
    }
    normalize_rows(query, query.len()).map_err(|error| match error {
        Error::InvalidSchema(message) => Error::InvalidArgument(message),
        other => other,
    })
}

fn normalize_rows(values: &[f32], dimension: usize) -> Result<Vec<f32>> {
    if dimension == 0 || !values.len().is_multiple_of(dimension) {
        return Err(Error::InvalidSchema(
            "all vectors must have the same positive dimension".into(),
        ));
    }
    let mut output = Vec::with_capacity(values.len());
    for vector in values.chunks_exact(dimension) {
        let squared_norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        if !squared_norm.is_finite() || squared_norm == 0.0 {
            return Err(Error::InvalidSchema(
                "cosine vectors must have a finite, non-zero norm".into(),
            ));
        }
        let inverse_norm = squared_norm.sqrt().recip();
        output.extend(
            vector
                .iter()
                .map(|value| normalized_value(*value, inverse_norm)),
        );
    }
    Ok(output)
}

#[allow(clippy::cast_possible_truncation)]
fn normalized_value(value: f32, inverse_norm: f64) -> f32 {
    // Normalization bounds finite outputs to [-1, 1]; the canonical vector type is f32.
    (f64::from(value) * inverse_norm) as f32
}

fn rebuild_vector_array(
    data_type: &DataType,
    rows: usize,
    dimension: usize,
    values: Vec<f32>,
) -> datafusion::common::Result<ArrayRef> {
    let values: ArrayRef = Arc::new(Float32Array::from(values));
    match data_type {
        DataType::List(field) => Ok(Arc::new(ListArray::new(
            Arc::clone(field),
            OffsetBuffer::from_lengths(std::iter::repeat_n(dimension, rows)),
            values,
            None,
        ))),
        DataType::LargeList(field) => Ok(Arc::new(LargeListArray::new(
            Arc::clone(field),
            OffsetBuffer::from_lengths(std::iter::repeat_n(dimension, rows)),
            values,
            None,
        ))),
        DataType::FixedSizeList(field, declared_dimension) => {
            let declared_dimension = usize::try_from(*declared_dimension).map_err(|_| {
                DataFusionError::Execution("invalid fixed-size vector dimension".into())
            })?;
            if declared_dimension != dimension {
                return Err(DataFusionError::Execution(
                    "fixed-size vector dimension changed during normalization".into(),
                ));
            }
            Ok(Arc::new(FixedSizeListArray::new(
                Arc::clone(field),
                i32::try_from(declared_dimension).map_err(|_| {
                    DataFusionError::Execution("fixed-size vector dimension exceeds int32".into())
                })?,
                values,
                None,
            )))
        }
        _ => unreachable!("canonical vector type was validated"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_query_is_normalized_once() {
        let query = transform_query(&[3.0, 4.0], DistanceMetric::Cosine).unwrap();
        assert!((query[0] - 0.6).abs() < 1e-6);
        assert!((query[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn cosine_rejects_zero_norm() {
        assert!(transform_query(&[0.0, 0.0], DistanceMetric::Cosine).is_err());
    }
}
