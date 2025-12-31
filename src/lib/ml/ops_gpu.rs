// GPU operations using candle-core

#[cfg(feature = "gpu")]
use crate::ml::tensor::Tensor;
#[cfg(feature = "gpu")]
use crate::ml::device::Device;

#[cfg(feature = "gpu")]
impl Tensor {
    /// Convert CPU tensor to candle tensor on GPU device
    #[allow(dead_code)] // Intended for future use
    fn to_candle_tensor(&self, device: &candle_core::Device) -> Result<candle_core::Tensor, String> {
        use candle_core::Shape;
        
        let shape = Shape::from_dims(&self.shape);
        let tensor = candle_core::Tensor::from_slice(&self.data, shape, device)
            .map_err(|e| format!("Failed to create candle tensor: {}", e))?;
        Ok(tensor)
    }
    
    /// Convert candle tensor back to CPU tensor
    #[allow(dead_code)] // Intended for future use
    fn from_candle_tensor(ct: &candle_core::Tensor, original_shape: &[usize]) -> Result<Self, String> {
        let data = ct.to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?;
        
        Ok(Tensor {
            data,
            shape: original_shape.to_vec(),
            device: Device::Cpu, // Result is always on CPU for now
            gpu_tensor: None,
        })
    }
    
    /// Matrix multiplication on GPU
    pub fn matmul_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        // Convert to candle tensors
        let shape_a = Shape::from_dims(&self.shape);
        let shape_b = Shape::from_dims(&other.shape);
        
        let a = candle_core::Tensor::from_slice(&self.data, shape_a, device)
            .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?;
        let b = candle_core::Tensor::from_slice(&other.data, shape_b, device)
            .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?;
        
        // Perform matmul on GPU
        let result = a.matmul(&b)
            .map_err(|e| format!("GPU matmul failed: {}", e))?;
        
        // Get result shape
        let result_shape = result.dims();
        
        // Convert back to CPU (for now - later we can keep on GPU)
        let data = result.to_vec2::<f32>()
            .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
        
        // Flatten 2D vec to 1D
        let flattened: Vec<f32> = data.into_iter().flat_map(|row| row.into_iter()).collect();
        
        Ok(Tensor {
            data: flattened,
            shape: result_shape.to_vec(),
            device: Device::Cpu, // For now, always return to CPU
            gpu_tensor: None,
        })
    }
    
    /// Element-wise addition on GPU
    pub fn add_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        let shape_a = Shape::from_dims(&self.shape);
        let shape_b = Shape::from_dims(&other.shape);
        
        let a = candle_core::Tensor::from_slice(&self.data, shape_a, device)
            .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?;
        let b = candle_core::Tensor::from_slice(&other.data, shape_b, device)
            .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?;
        
        let result = (&a + &b)
            .map_err(|e| format!("GPU add failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // Convert based on tensor rank
        let data = if rank == 1 {
            // 1D tensor, use to_vec1 directly
            result.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        } else if rank == 2 {
            // 2D tensor, use to_vec2 and flatten
            let data_2d = result.to_vec2::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
            data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
        } else {
            // Higher rank tensor, reshape to 1D first
            let total_size: usize = result_shape.iter().product();
            let flattened = result.reshape((total_size,))
                .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        };
        
        Ok(Tensor {
            data,
            shape: result_shape.to_vec(),
            device: Device::Cpu,
            gpu_tensor: None,
        })
    }
    
    /// Element-wise subtraction on GPU
    pub fn sub_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        let shape_a = Shape::from_dims(&self.shape);
        let shape_b = Shape::from_dims(&other.shape);
        
        let a = candle_core::Tensor::from_slice(&self.data, shape_a, device)
            .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?;
        let b = candle_core::Tensor::from_slice(&other.data, shape_b, device)
            .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?;
        
        let result = (&a - &b)
            .map_err(|e| format!("GPU sub failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // Convert based on tensor rank
        let data = if rank == 1 {
            // 1D tensor, use to_vec1 directly
            result.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        } else if rank == 2 {
            // 2D tensor, use to_vec2 and flatten
            let data_2d = result.to_vec2::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
            data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
        } else {
            // Higher rank tensor, reshape to 1D first
            let total_size: usize = result_shape.iter().product();
            let flattened = result.reshape((total_size,))
                .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        };
        
        Ok(Tensor {
            data,
            shape: result_shape.to_vec(),
            device: Device::Cpu,
            gpu_tensor: None,
        })
    }
    
    /// Element-wise multiplication on GPU
    pub fn mul_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        let shape_a = Shape::from_dims(&self.shape);
        let shape_b = Shape::from_dims(&other.shape);
        
        let a = candle_core::Tensor::from_slice(&self.data, shape_a, device)
            .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?;
        let b = candle_core::Tensor::from_slice(&other.data, shape_b, device)
            .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?;
        
        let result = (&a * &b)
            .map_err(|e| format!("GPU mul failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // Convert based on tensor rank
        let data = if rank == 1 {
            // 1D tensor, use to_vec1 directly
            result.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        } else if rank == 2 {
            // 2D tensor, use to_vec2 and flatten
            let data_2d = result.to_vec2::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
            data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
        } else {
            // Higher rank tensor, reshape to 1D first
            let total_size: usize = result_shape.iter().product();
            let flattened = result.reshape((total_size,))
                .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        };
        
        Ok(Tensor {
            data,
            shape: result_shape.to_vec(),
            device: Device::Cpu,
            gpu_tensor: None,
        })
    }
    
    /// Element-wise division on GPU
    pub fn div_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        let shape_a = Shape::from_dims(&self.shape);
        let shape_b = Shape::from_dims(&other.shape);
        
        let a = candle_core::Tensor::from_slice(&self.data, shape_a, device)
            .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?;
        let b = candle_core::Tensor::from_slice(&other.data, shape_b, device)
            .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?;
        
        let result = (&a / &b)
            .map_err(|e| format!("GPU div failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // Convert based on tensor rank
        let data = if rank == 1 {
            // 1D tensor, use to_vec1 directly
            result.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        } else if rank == 2 {
            // 2D tensor, use to_vec2 and flatten
            let data_2d = result.to_vec2::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
            data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
        } else {
            // Higher rank tensor, reshape to 1D first
            let total_size: usize = result_shape.iter().product();
            let flattened = result.reshape((total_size,))
                .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        };
        
        Ok(Tensor {
            data,
            shape: result_shape.to_vec(),
            device: Device::Cpu,
            gpu_tensor: None,
        })
    }
    
    /// Scalar division on GPU (tensor / scalar)
    pub fn div_scalar_gpu(&self, scalar: f32, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        if scalar == 0.0 {
            return Err("Division by zero".to_string());
        }
        
        let shape = Shape::from_dims(&self.shape);
        let a = candle_core::Tensor::from_slice(&self.data, shape, device)
            .map_err(|e| format!("Failed to create tensor on GPU: {}", e))?;
        
        // Create a scalar tensor and use broadcasting
        // Use multiplication by reciprocal instead of division
        let reciprocal = 1.0 / scalar;
        let scalar_tensor = candle_core::Tensor::new(&[reciprocal], device)
            .map_err(|e| format!("Failed to create scalar tensor on GPU: {}", e))?;
        
        // Broadcast scalar to match input shape and multiply
        let result = (&a * &scalar_tensor.broadcast_as(a.dims())
            .map_err(|e| format!("Failed to broadcast scalar tensor: {}", e))?)
            .map_err(|e| format!("GPU scalar div failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // Convert based on tensor rank
        let data = if rank == 1 {
            // 1D tensor, use to_vec1 directly
            result.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        } else if rank == 2 {
            // 2D tensor, use to_vec2 and flatten
            let data_2d = result.to_vec2::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
            data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
        } else {
            // Higher rank tensor, reshape to 1D first
            let total_size: usize = result_shape.iter().product();
            let flattened = result.reshape((total_size,))
                .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        };
        
        Ok(Tensor {
            data,
            shape: result_shape.to_vec(),
            device: Device::Cpu,
            gpu_tensor: None,
        })
    }
    
    /// ReLU activation on GPU
    pub fn relu_gpu(&self, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        let shape = Shape::from_dims(&self.shape);
        let a = candle_core::Tensor::from_slice(&self.data, shape, device)
            .map_err(|e| format!("Failed to create tensor on GPU: {}", e))?;
        
        let result = a.relu()
            .map_err(|e| format!("GPU ReLU failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // Convert based on tensor rank
        let data = if rank == 1 {
            // 1D tensor, use to_vec1 directly
            result.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        } else if rank == 2 {
            // 2D tensor, use to_vec2 and flatten
            let data_2d = result.to_vec2::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
            data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
        } else {
            // Higher rank tensor, reshape to 1D first
            let total_size: usize = result_shape.iter().product();
            let flattened = result.reshape((total_size,))
                .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
        };
        
        Ok(Tensor {
            data,
            shape: result_shape.to_vec(),
            device: Device::Cpu,
            gpu_tensor: None,
        })
    }
}

