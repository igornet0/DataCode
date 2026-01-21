// Tensor type and operations for ML module
// Rewritten to match MetalNN architecture

use crate::ml::device::Device;
use ndarray::ArrayD;
use std::sync::Arc;

/// Tensor storage - wraps the actual data
#[derive(Clone)]
pub struct TensorStorage {
    data: ArrayD<f32>,
}

impl TensorStorage {
    pub fn new(data: ArrayD<f32>) -> Self {
        Self { data }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn as_array(&self) -> &ArrayD<f32> {
        &self.data
    }

    pub fn as_array_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    pub fn to_owned_array(&self) -> ArrayD<f32> {
        self.data.clone()
    }
}

/// Tensor - основной тип для работы с данными
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<TensorStorage>,
    pub shape: Vec<usize>,  // Public for compatibility with existing code
    strides: Vec<isize>,
    device: Device,
    pub data: Vec<f32>,  // Public for compatibility - computed from storage
    
    #[cfg(feature = "gpu")]
    /// Lazy GPU tensor storage (None means not yet moved to GPU)
    pub(crate) gpu_tensor: Option<candle_core::Tensor>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .field("numel", &self.numel())
            .finish()
    }
}

impl Tensor {
    /// Создать тензор из f32 массива
    pub fn from_array(array: ArrayD<f32>) -> Self {
        let shape = array.shape().to_vec();
        let strides = Self::compute_strides(&shape);
        let data_vec: Vec<f32> = array.iter().copied().collect();
        let storage = TensorStorage::new(array);
        
        Self {
            storage: Arc::new(storage),
            shape: shape.clone(),
            strides,
            device: Device::Cpu,
            data: data_vec,
    #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Создать тензор заданной формы, заполненный нулями
    pub fn zeros(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        let array = ArrayD::zeros(shape.as_slice());
        let data_vec: Vec<f32> = array.iter().copied().collect();
        let storage = TensorStorage::new(array);

        Self {
            storage: Arc::new(storage),
            shape: shape.clone(),
            strides,
            device: Device::Cpu,
            data: data_vec,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Создать тензор заданной формы, заполненный нулями, на указанном устройстве
    pub fn zeros_with_device(shape: Vec<usize>, device: Device) -> Self {
        if device.is_cpu() {
            Self::zeros(shape)
        } else {
            #[cfg(feature = "gpu")]
            {
                let candle_device = match device.as_candle() {
                    Some(d) => d,
                    None => {
                        // Fallback to CPU if device is invalid
                        return Self::zeros(shape);
                    }
                };
                
                use candle_core::{Shape, DType};
                let candle_shape = Shape::from_dims(&shape);
                let gpu_tensor = match candle_core::Tensor::zeros(candle_shape, DType::F32, &candle_device) {
                    Ok(t) => t,
                    Err(_) => {
                        // Fallback to CPU if GPU creation fails
                        return Self::zeros(shape);
                    }
                };
                
                Self::from_gpu_tensor(shape, device, Some(gpu_tensor))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Self::zeros(shape)
            }
        }
    }

    /// Создать тензор заданной формы, заполненный единицами
    pub fn ones(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        let array = ArrayD::ones(shape.as_slice());
        let data_vec: Vec<f32> = array.iter().copied().collect();
        let storage = TensorStorage::new(array);

        Self {
            storage: Arc::new(storage),
            shape: shape.clone(),
            strides,
            device: Device::Cpu,
            data: data_vec,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Создать тензор из слайса
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let array = ArrayD::from_shape_vec(shape, data.to_vec()).unwrap();
        Self::from_array(array)
    }

    /// Создать тензор с GPU tensor (для внутреннего использования)
    #[cfg(feature = "gpu")]
    pub(crate) fn from_gpu_tensor(
        shape: Vec<usize>,
        device: Device,
        gpu_tensor: Option<candle_core::Tensor>,
    ) -> Self {
        let strides = Self::compute_strides(&shape);
        let array = ArrayD::zeros(shape.as_slice());
        let data_vec: Vec<f32> = array.iter().copied().collect();
        let storage = TensorStorage::new(array);
        
        Self {
            storage: Arc::new(storage),
            shape,
            strides,
            device,
            data: data_vec,
            gpu_tensor,
        }
    }

    /// Создать тензор из Vec<f32> и shape (для совместимости)
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        // Validate shape
        if shape.is_empty() {
            return Err("Shape cannot be empty".to_string());
        }
        
        // Check for zero dimensions
        if shape.iter().any(|&s| s == 0) {
            return Err("Shape dimensions cannot be zero".to_string());
        }
        
        // Calculate expected size
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(format!(
                "Data size {} does not match shape {:?} (expected {})",
                data.len(), shape, expected_size
            ));
        }
        
        let array = ArrayD::from_shape_vec(shape.as_slice(), data).unwrap();
        Ok(Self::from_array(array))
    }

    /// Получить форму тензора
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Получить strides
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Получить устройство
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Получить количество элементов
    pub fn numel(&self) -> usize {
        self.storage.numel()
    }

    /// Получить данные как массив (копия, для операций)
    pub fn as_array(&self) -> ArrayD<f32> {
        self.storage.to_owned_array()
    }

    /// Получить ссылку на данные
    pub fn data(&self) -> &ArrayD<f32> {
        self.storage.as_array()
    }

    /// Вычислить strides для формы
    fn compute_strides(shape: &[usize]) -> Vec<isize> {
        let mut strides = vec![1isize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
        }
        strides
    }

    /// Преобразовать в вектор (для отладки и совместимости)
    pub fn to_vec(&self) -> Vec<f32> {
        self.storage.as_array().iter().copied().collect()
    }

    /// Получить данные как слайс (для операций)
    pub fn as_slice(&self) -> &[f32] {
        self.storage.as_array().as_slice().unwrap()
    }

    // ===== Совместимость методы (используются в natives.rs) =====

    /// Calculate memory size in bytes (CPU data only)
    pub fn memory_size_bytes(&self) -> usize {
        self.numel() * std::mem::size_of::<f32>()
    }
    
    /// Check if tensor has GPU buffer
            #[cfg(feature = "gpu")]
    pub fn has_gpu_buffer(&self) -> bool {
        self.gpu_tensor.is_some()
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn has_gpu_buffer(&self) -> bool {
        false
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        let size = data.len();
        Self::from_slice(&data, &[size])
    }

    pub fn total_size(&self) -> usize {
        self.numel()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let arr = self.data();
        // Use ndarray indexing
        arr.get(ndarray::IxDyn(indices)).copied()
    }

    pub fn set(&mut self, _indices: &[usize], _value: f32) -> Result<(), String> {
        // Note: With Arc<TensorStorage>, we can't mutate in place
        // This would require creating a new tensor with modified data
        // For now, return error - this method may need to be removed or redesigned
        Err("Cannot mutate tensor with Arc storage. Use operations instead.".to_string())
    }

    // GPU operations
    #[cfg(feature = "gpu")]
    pub fn to_device(&self, target_device: &Device) -> Result<Self, String> {
        if target_device.is_cpu() {
            // Moving to CPU - convert GPU tensor to CPU if needed
            if let Some(ref gpu_t) = self.gpu_tensor {
                // Convert GPU tensor to CPU (handle multi-dimensional tensors)
                let rank = gpu_t.rank();
                let data = if rank == 1 {
                    gpu_t.to_vec1::<f32>()
                        .map_err(|e| format!("Failed to convert GPU tensor to CPU: {}", e))?
                } else if rank == 2 {
                    gpu_t.to_vec2::<f32>()
                        .map_err(|e| format!("Failed to convert GPU tensor to CPU: {}", e))?
                        .into_iter()
                        .flat_map(|row| row.into_iter())
                        .collect()
                } else {
                    // Higher rank - flatten first
                    let total_size: usize = gpu_t.dims().iter().product();
                    let flattened = gpu_t.reshape((total_size,))
                        .map_err(|e| format!("Failed to reshape GPU tensor: {}", e))?;
                    flattened.to_vec1::<f32>()
                        .map_err(|e| format!("Failed to convert GPU tensor to CPU: {}", e))?
                };
                let shape = gpu_t.dims().to_vec();
                Ok(Tensor::from_slice(&data, &shape))
            } else {
                // Already on CPU
                Ok(self.clone())
            }
        } else {
            // Moving to GPU
            let candle_device = target_device.as_candle()
                .ok_or_else(|| "Invalid GPU device".to_string())?;
            
            // If already on GPU with same device, return clone
            if self.device.is_gpu() && &self.device == target_device {
                if self.gpu_tensor.is_some() {
                    return Ok(self.clone());
                }
            }
            
            // Create GPU tensor from CPU data
            use candle_core::Shape;
            let shape = Shape::from_dims(&self.shape);
            let gpu_tensor = candle_core::Tensor::from_slice(&self.data, shape, &candle_device)
                .map_err(|e| format!("Failed to create GPU tensor: {}", e))?;
            
            Ok(Tensor::from_gpu_tensor(
                self.shape.clone(),
                target_device.clone(),
                Some(gpu_tensor),
            ))
        }
    }

        #[cfg(not(feature = "gpu"))]
    pub fn to_device(&self, target_device: &Device) -> Result<Self, String> {
        if target_device.is_cpu() {
            Ok(self.clone())
        } else {
            Err("GPU support not compiled".to_string())
        }
    }

    pub fn to_cpu(&self) -> Result<Self, String> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref gpu_t) = self.gpu_tensor {
                // Convert GPU tensor to CPU
                let rank = gpu_t.rank();
                let data = if rank == 1 {
                    gpu_t.to_vec1::<f32>()
                        .map_err(|e| format!("Failed to convert GPU tensor to CPU: {}", e))?
                } else if rank == 2 {
                    gpu_t.to_vec2::<f32>()
                        .map_err(|e| format!("Failed to convert GPU tensor to CPU: {}", e))?
                        .into_iter()
                        .flat_map(|row| row.into_iter())
                        .collect()
                } else {
                    // Higher rank - flatten first
                    let total_size: usize = gpu_t.dims().iter().product();
                    let flattened = gpu_t.reshape((total_size,))
                        .map_err(|e| format!("Failed to reshape GPU tensor: {}", e))?;
                    flattened.to_vec1::<f32>()
                        .map_err(|e| format!("Failed to convert GPU tensor to CPU: {}", e))?
                };
                let shape = gpu_t.dims().to_vec();
                Ok(Tensor::from_slice(&data, &shape))
            } else {
                // Already on CPU
                Ok(self.clone())
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Ok(self.clone())
        }
    }
    
    /// Slice a batch from GPU tensor without CPU transfer
    /// This is much more efficient than copying data to CPU and back
    #[cfg(feature = "gpu")]
    pub fn slice_gpu(&self, start: usize, end: usize, dim: usize) -> Result<Self, String> {
        if !self.device.is_gpu() {
            return Err("slice_gpu() requires tensor to be on GPU".to_string());
        }
        
        if let Some(ref gpu_t) = self.gpu_tensor {
            // Use Candle's narrow() method to create a view without copying
            // narrow(dim, start, len) creates a view along dimension dim
            let len = end - start;
            let sliced_gpu = gpu_t.narrow(dim, start, len)
                .map_err(|e| format!("Failed to slice GPU tensor: {}", e))?;
            
            // Compute new shape
            let mut new_shape = self.shape.clone();
            new_shape[dim] = len;
            
            Ok(Tensor::from_gpu_tensor(
                new_shape,
                self.device.clone(),
                Some(sliced_gpu),
            ))
        } else {
            // Fallback: if no GPU tensor but device is GPU, try to create one
            // This shouldn't happen in normal flow, but handle gracefully
            Err("GPU tensor not available for slicing".to_string())
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn slice_gpu(&self, _start: usize, _end: usize, _dim: usize) -> Result<Self, String> {
        Err("GPU support not compiled".to_string())
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Tensor {
        let arr = self.data();
        let result = arr.mapv(|x| x.abs());
        Tensor::from_array(result)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor, String> {
        // Check for negative values
        let arr = self.data();
        if arr.iter().any(|&x| x < 0.0) {
            return Err("Square root of negative number".to_string());
        }
        let result = arr.mapv(|x| x.sqrt());
        Ok(Tensor::from_array(result))
    }

    /// Element-wise rounding
    pub fn round(&self) -> Tensor {
        let arr = self.data();
        let result = arr.mapv(|x| x.round());
        Tensor::from_array(result)
    }

    // Basic operations (delegating to ops module)
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        Ok(crate::ml::ops::add(self, other))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, String> {
        let neg_other = crate::ml::ops::scalar_mul(other, -1.0);
        Ok(crate::ml::ops::add(self, &neg_other))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        Ok(crate::ml::ops::mul(self, other))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> {
        Ok(crate::ml::ops::div(self, other))
    }

    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor, String> {
        Ok(crate::ml::ops::scalar_div(self, scalar))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        Ok(crate::ml::ops::matmul(self, other))
    }

    pub fn transpose(&self) -> Result<Tensor, String> {
        Ok(crate::ml::ops::transpose(self))
    }

    pub fn sum(&self) -> f32 {
        self.data().sum()
    }

    pub fn mean(&self) -> f32 {
        self.data().mean().unwrap_or(0.0)
    }

    pub fn max_idx(&self) -> Result<Vec<usize>, String> {
        let arr = self.data();
        if arr.is_empty() {
            return Err("Cannot find max index in empty tensor".to_string());
        }
        if self.shape.len() == 1 {
            let max_idx = arr.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find max index")?;
            Ok(vec![max_idx])
        } else {
            let mut result = Vec::new();
            for i in 0..self.shape[0] {
                let slice = arr.index_axis(ndarray::Axis(0), i);
                let max_idx = slice.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find max index in slice")?;
            result.push(max_idx);
        }
        Ok(result)
        }
    }

    pub fn min_idx(&self) -> Result<Vec<usize>, String> {
        let arr = self.data();
        if arr.is_empty() {
            return Err("Cannot find min index in empty tensor".to_string());
        }
        if self.shape.len() == 1 {
            let min_idx = arr.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find min index")?;
            Ok(vec![min_idx])
        } else {
            let mut result = Vec::new();
            for i in 0..self.shape[0] {
                let slice = arr.index_axis(ndarray::Axis(0), i);
                let min_idx = slice.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find min index in slice")?;
            result.push(min_idx);
        }
        Ok(result)
    }
    }

    pub fn get_row(&self, index: usize) -> Result<Tensor, String> {
        if self.shape.is_empty() {
            return Err("Cannot index empty tensor".to_string());
        }
        if index >= self.shape[0] {
            return Err(format!("Index {} out of bounds for dimension 0 (size: {})", index, self.shape[0]));
        }

        let arr = self.data();
        let slice = arr.index_axis(ndarray::Axis(0), index).to_owned();
        Ok(Tensor::from_array(slice.into_dyn()))
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        let arr = self.data();
        let broadcasted = arr.broadcast(target_shape)
            .ok_or_else(|| format!("Failed to broadcast from {:?} to {:?}", self.shape(), target_shape))?
            .to_owned();
        Ok(Tensor::from_array(broadcasted))
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, String> {
        if new_shape.is_empty() {
            return Err("New shape cannot be empty".to_string());
        }
        let current_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        if current_size != new_size {
            return Err(format!(
                "Cannot reshape from {:?} to {:?}: size mismatch ({} vs {})",
                self.shape, new_shape, current_size, new_size
            ));
        }
        let arr = self.data();
        let result = arr.clone().into_shape(ndarray::IxDyn(&new_shape)).unwrap().to_owned();
        Ok(Tensor::from_array(result))
    }

    pub fn flatten(&self) -> Result<Tensor, String> {
        if self.ndim() < 2 {
            return Err("Flatten requires at least 2 dimensions".to_string());
        }
        let batch_size = self.shape[0];
        let flattened_size: usize = self.shape[1..].iter().product();
        self.reshape(vec![batch_size, flattened_size])
    }

    pub fn relu(&self) -> Tensor {
        crate::ml::ops::relu(self)
    }

    pub fn sigmoid(&self) -> Tensor {
        crate::ml::ops::sigmoid(self)
    }

    pub fn tanh(&self) -> Tensor {
        let arr = self.data();
        let result = arr.mapv(|x| x.tanh());
        Tensor::from_array(result)
    }

    pub fn softmax(&self) -> Result<Tensor, String> {
        // Для 2D тензоров используем axis=1
        if self.ndim() == 2 {
            Ok(crate::ml::ops::softmax(self, 1))
        } else {
            // Fallback для других случаев
            Ok(crate::ml::ops::softmax(self, 0))
        }
    }

    pub fn sum_to_shape(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        crate::ml::ops::sum_to_shape(self, target_shape)
    }

    pub fn neg(&self) -> Tensor {
        let arr = self.data();
        let result = arr.mapv(|x| -x);
        Tensor::from_array(result)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.to_vec() == other.to_vec()
    }
}
