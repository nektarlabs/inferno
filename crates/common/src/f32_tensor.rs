use serde::{Deserialize, Serialize};

use crate::{validate_exact_shape, Error, Result, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
}

pub trait TensorShape {
    fn into_shape(self) -> Vec<usize>;
}

impl TensorShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl TensorShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> TensorShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl TensorShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

impl TensorShape for (usize,) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0]
    }
}

impl TensorShape for (usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl TensorShape for (usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl TensorShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

pub trait TensorElement: Sized {
    fn from_f32(value: f32) -> Result<Self>;
}

pub trait IntoTensorValue {
    fn into_f32(self) -> Result<f32>;
}

impl TensorElement for f32 {
    fn from_f32(value: f32) -> Result<Self> {
        Ok(value)
    }
}

impl TensorElement for u32 {
    fn from_f32(value: f32) -> Result<Self> {
        if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= u32::MAX as f32 {
            Ok(value as u32)
        } else {
            Err(Error::model(format!(
                "cannot read non-integer tensor value {value} as u32"
            )))
        }
    }
}

impl IntoTensorValue for f32 {
    fn into_f32(self) -> Result<f32> {
        Ok(self)
    }
}

impl IntoTensorValue for u32 {
    fn into_f32(self) -> Result<f32> {
        Ok(self as f32)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    shape: Shape,
    values: Vec<f32>,
}

pub type F32Tensor = Tensor;

impl Tensor {
    pub fn new(values: Vec<f32>, shape: impl TensorShape) -> Result<Self> {
        let shape = Shape::new(shape.into_shape());
        let expected_len = element_count(shape.dims())?;
        validate_exact_shape("f32_tensor_value_count", &[values.len()], &[expected_len])?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::model("F32 tensor contains non-finite values"));
        }

        Ok(Self { shape, values })
    }

    pub fn from_vec<T: IntoTensorValue>(
        values: Vec<T>,
        shape: impl TensorShape,
        _device: &Device,
    ) -> Result<Self> {
        let values = values
            .into_iter()
            .map(IntoTensorValue::into_f32)
            .collect::<Result<Vec<_>>>()?;
        Self::new(values, shape)
    }

    pub fn zeros(shape: impl TensorShape) -> Result<Self> {
        let shape = shape.into_shape();
        let len = element_count(&shape)?;
        Self::new(vec![0.0; len], shape)
    }

    pub fn zeros_with_dtype(
        shape: impl TensorShape,
        dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        match dtype {
            DType::F32 => Self::zeros(shape),
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub fn reshape(self, shape: impl TensorShape) -> Result<Self> {
        Self::new(self.values, shape)
    }

    pub fn contiguous(&self) -> Result<Self> {
        Ok(self.clone())
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(self.clone()),
        }
    }

    pub fn flatten_all(self) -> Result<Self> {
        let len = self.values.len();
        Self::new(self.values, [len])
    }

    pub fn t(&self) -> Result<Self> {
        let dims = self.dims();
        validate_exact_shape("tensor_transpose_rank", &[dims.len()], &[2])?;
        let rows = dims[0];
        let cols = dims[1];
        let mut output = vec![0.0_f32; self.values.len()];
        for row in 0..rows {
            for col in 0..cols {
                output[col * rows + row] = self.values[row * cols + col];
            }
        }
        Self::new(output, [cols, rows])
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut shape = self.dims().to_vec();
        if dim > shape.len() {
            return Err(Error::model(format!(
                "tensor unsqueeze dim {dim} is outside rank {}",
                shape.len()
            )));
        }
        shape.insert(dim, 1);
        Self::new(self.values.clone(), shape)
    }

    pub fn to_vec0<T: TensorElement>(&self) -> Result<T> {
        validate_exact_shape("tensor_scalar_value_count", &[self.values.len()], &[1])?;
        T::from_f32(self.values[0])
    }

    pub fn to_vec1<T: TensorElement>(&self) -> Result<Vec<T>> {
        if self.dims().len() != 1 {
            return Err(Error::model(format!(
                "tensor rank must be 1 for to_vec1, got {:?}",
                self.dims()
            )));
        }
        self.values.iter().copied().map(T::from_f32).collect()
    }

    pub fn to_vec2<T: TensorElement>(&self) -> Result<Vec<Vec<T>>> {
        let dims = self.dims();
        validate_exact_shape("tensor_to_vec2_rank", &[dims.len()], &[2])?;
        let rows = dims[0];
        let cols = dims[1];
        let mut output = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row
                .checked_mul(cols)
                .ok_or_else(|| Error::model("tensor to_vec2 row offset overflow"))?;
            let end = start
                .checked_add(cols)
                .ok_or_else(|| Error::model("tensor to_vec2 row end overflow"))?;
            output.push(
                self.values[start..end]
                    .iter()
                    .copied()
                    .map(T::from_f32)
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        Ok(output)
    }

    pub fn to_vec3<T: TensorElement>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        let dims = self.dims();
        validate_exact_shape("tensor_to_vec3_rank", &[dims.len()], &[3])?;
        let batch = dims[0];
        let rows = dims[1];
        let cols = dims[2];
        let mut output = Vec::with_capacity(batch);
        for batch_index in 0..batch {
            let mut batch_rows = Vec::with_capacity(rows);
            for row in 0..rows {
                let start = ((batch_index * rows) + row)
                    .checked_mul(cols)
                    .ok_or_else(|| Error::model("tensor to_vec3 row offset overflow"))?;
                let end = start
                    .checked_add(cols)
                    .ok_or_else(|| Error::model("tensor to_vec3 row end overflow"))?;
                batch_rows.push(
                    self.values[start..end]
                        .iter()
                        .copied()
                        .map(T::from_f32)
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            output.push(batch_rows);
        }
        Ok(output)
    }

    pub fn sum_all(&self) -> Result<Self> {
        let sum = self.values.iter().copied().sum::<f32>();
        Self::new(vec![sum], [1])
    }

    pub fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| Error::model("tensor cat requires at least one input"))?;
        let rank = first.dims().len();
        if dim >= rank {
            return Err(Error::model(format!(
                "tensor cat dim {dim} is outside rank {rank}"
            )));
        }
        for tensor in tensors {
            validate_exact_shape("tensor_cat_rank", &[tensor.dims().len()], &[rank])?;
            for axis in 0..rank {
                if axis != dim && tensor.dims()[axis] != first.dims()[axis] {
                    return Err(Error::model(format!(
                        "tensor cat non-concat dim mismatch at axis {axis}: expected {}, got {}",
                        first.dims()[axis],
                        tensor.dims()[axis]
                    )));
                }
            }
        }

        let mut output_shape = first.dims().to_vec();
        output_shape[dim] =
            tensors
                .iter()
                .map(|tensor| tensor.dims()[dim])
                .try_fold(0_usize, |sum, value| {
                    sum.checked_add(value)
                        .ok_or_else(|| Error::model("tensor cat concat dimension overflow"))
                })?;

        let outer = first.dims()[..dim].iter().product::<usize>();
        let inner = first.dims()[dim + 1..].iter().product::<usize>();
        let mut output = Vec::with_capacity(element_count(&output_shape)?);
        for outer_index in 0..outer {
            for tensor in tensors {
                let chunk_values = tensor.dims()[dim]
                    .checked_mul(inner)
                    .ok_or_else(|| Error::model("tensor cat chunk value overflow"))?;
                let start = outer_index
                    .checked_mul(tensor.dims()[dim])
                    .and_then(|value| value.checked_mul(inner))
                    .ok_or_else(|| Error::model("tensor cat source offset overflow"))?;
                let end = start
                    .checked_add(chunk_values)
                    .ok_or_else(|| Error::model("tensor cat source end overflow"))?;
                output.extend_from_slice(&tensor.values[start..end]);
            }
        }

        Self::new(output, output_shape)
    }

    pub fn into_parts(self) -> (Shape, Vec<f32>) {
        (self.shape, self.values)
    }
}

fn element_count(dims: &[usize]) -> Result<usize> {
    if dims.is_empty() {
        return Err(Error::model(
            "F32 tensor shape must have at least one dimension",
        ));
    }

    dims.iter().try_fold(1_usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| Error::model("F32 tensor element count overflow"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_tensor_with_matching_shape() {
        let tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]).unwrap();

        assert_eq!(tensor.dims(), &[1, 2, 2]);
        assert_eq!(tensor.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_wrong_value_count() {
        let err =
            F32Tensor::new(vec![1.0, 2.0, 3.0], [1, 2, 2]).expect_err("shape mismatch should fail");

        assert!(err.to_string().contains("f32_tensor_value_count"));
    }

    #[test]
    fn rejects_non_finite_values() {
        let err = F32Tensor::new(vec![f32::NAN], [1]).expect_err("non-finite tensors should fail");

        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn reshapes_without_changing_values() {
        let tensor = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let reshaped = tensor.reshape([1, 4]).unwrap();

        assert_eq!(reshaped.dims(), &[1, 4]);
        assert_eq!(reshaped.values(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
