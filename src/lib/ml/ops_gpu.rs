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
        
        Ok(Tensor::from_slice(&data, original_shape))
    }
    
    /// Matrix multiplication on GPU
    pub fn matmul_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        // CRITICAL OPTIMIZATION: Reuse existing GPU buffers if available
        // This prevents creating new Metal buffers on every operation
        // #region agent log
        let _a_reused = self.gpu_tensor.is_some();
        let _b_reused = other.gpu_tensor.is_some();
        // #endregion
        let a = if let Some(ref gpu_t) = self.gpu_tensor {
            // Reuse existing GPU buffer
            gpu_t.clone()
        } else {
            // Create new GPU buffer only if not exists
            let shape_a = Shape::from_dims(&self.shape);
            candle_core::Tensor::from_slice(&self.data, shape_a, device)
                .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?
        };
        
        let b = if let Some(ref gpu_t) = other.gpu_tensor {
            // Reuse existing GPU buffer
            gpu_t.clone()
        } else {
            // Create new GPU buffer only if not exists
            let shape_b = Shape::from_dims(&other.shape);
            candle_core::Tensor::from_slice(&other.data, shape_b, device)
                .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?
        };
        
        // Perform matmul on GPU
        let result = a.matmul(&b)
            .map_err(|e| format!("GPU matmul failed: {}", e))?;
        
        // Get result shape
        let result_shape = result.dims();
        
        // CRITICAL FIX: Keep result on GPU instead of converting back to CPU
        // This allows subsequent operations to reuse GPU buffers
        // Only convert to CPU when actually needed (lazy conversion)
        let result_gpu = Some(result.clone());
        
        // Lazy CPU conversion: only convert when needed, but keep GPU buffer
        // For now, we still need CPU data for compatibility, but we keep GPU buffer
        // #region agent log
        let convert_start = std::time::Instant::now();
        // #endregion
        let data = result.to_vec2::<f32>()
            .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
        // #region agent log
        let convert_time = convert_start.elapsed();
        let log_data = format!(r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"ops_gpu.rs:{}\","message":"MatMul GPU buffer usage","data":{{"a_reused":{},"b_reused":{},"convert_time_ms":{},"gpu_kept":true}},"timestamp":{}}}"#, 
            line!(), _a_reused, _b_reused, convert_time.as_millis(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/igor/Desktop/Projects/DataCode/.cursor/debug.log") {
            use std::io::Write;
            let _ = writeln!(file, "{}", log_data);
        }
        // #endregion
        
        // Flatten 2D vec to 1D
        let _flattened: Vec<f32> = data.into_iter().flat_map(|row| row.into_iter()).collect();
        
        // Use device from input tensors (prefer GPU if any input is on GPU)
        use std::sync::Arc;
        let result_device = if self.device().is_gpu() || other.device().is_gpu() {
            Device::Metal(Arc::new(device.clone()))
        } else {
            Device::Cpu
        };
        
        Ok(Tensor::from_gpu_tensor(
            result_shape.to_vec(),
            result_device,
            result_gpu, // CRITICAL: Keep GPU buffer for reuse
        ))
    }
    
    /// Element-wise addition on GPU
    pub fn add_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        // CRITICAL OPTIMIZATION: Reuse existing GPU buffers if available
        let a = if let Some(ref gpu_t) = self.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape_a = Shape::from_dims(&self.shape);
            candle_core::Tensor::from_slice(&self.data, shape_a, device)
                .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?
        };
        
        let b = if let Some(ref gpu_t) = other.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape_b = Shape::from_dims(&other.shape);
            candle_core::Tensor::from_slice(&other.data, shape_b, device)
                .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?
        };
        
        // #region agent log
        let _a_reused = self.gpu_tensor.is_some();
        let _b_reused = other.gpu_tensor.is_some();
        // #endregion
        
        let result = (&a + &b)
            .map_err(|e| format!("GPU add failed: {}", e))?;
        
        let result_shape = result.dims();
        let rank = result.rank();
        
        // CRITICAL FIX: Keep result on GPU instead of converting back to CPU
        let result_gpu = Some(result.clone());
        
        // Lazy CPU conversion: only convert when needed, but keep GPU buffer
        let _data = if rank == 1 {
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
        
        // Use device from input tensors (prefer GPU if any input is on GPU)
        use std::sync::Arc;
        let result_device = if self.device().is_gpu() || other.device().is_gpu() {
            Device::Metal(Arc::new(device.clone()))
        } else {
            Device::Cpu
        };
        
        Ok(Tensor::from_gpu_tensor(
            result_shape.to_vec(),
            result_device,
            result_gpu, // CRITICAL: Keep GPU buffer for reuse
        ))
    }
    
    /// Element-wise subtraction on GPU
    pub fn sub_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        // CRITICAL OPTIMIZATION: Reuse existing GPU buffers if available
        let a = if let Some(ref gpu_t) = self.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape_a = Shape::from_dims(&self.shape);
            candle_core::Tensor::from_slice(&self.data, shape_a, device)
                .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?
        };
        
        let b = if let Some(ref gpu_t) = other.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape_b = Shape::from_dims(&other.shape);
            candle_core::Tensor::from_slice(&other.data, shape_b, device)
                .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?
        };
        
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
        
        Ok(Tensor::from_slice(&data, &result_shape))
    }
    
    /// Element-wise multiplication on GPU
    pub fn mul_gpu(&self, other: &Tensor, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        // CRITICAL OPTIMIZATION: Reuse existing GPU buffers if available
        let a = if let Some(ref gpu_t) = self.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape_a = Shape::from_dims(&self.shape);
            candle_core::Tensor::from_slice(&self.data, shape_a, device)
                .map_err(|e| format!("Failed to create tensor A on GPU: {}", e))?
        };
        
        let b = if let Some(ref gpu_t) = other.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape_b = Shape::from_dims(&other.shape);
            candle_core::Tensor::from_slice(&other.data, shape_b, device)
                .map_err(|e| format!("Failed to create tensor B on GPU: {}", e))?
        };
        
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
        
        Ok(Tensor::from_slice(&data, &result_shape))
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
        
        Ok(Tensor::from_slice(&data, &result_shape))
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
        
        Ok(Tensor::from_slice(&data, &result_shape))
    }
    
    /// ReLU activation on GPU
    pub fn relu_gpu(&self, device: &candle_core::Device) -> Result<Tensor, String> {
        use candle_core::Shape;
        
        // CRITICAL OPTIMIZATION: Reuse existing GPU buffer if available
        let a = if let Some(ref gpu_t) = self.gpu_tensor {
            gpu_t.clone()
        } else {
            let shape = Shape::from_dims(&self.shape);
            candle_core::Tensor::from_slice(&self.data, shape, device)
                .map_err(|e| format!("Failed to create tensor on GPU: {}", e))?
        };
        
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
        
        Ok(Tensor::from_slice(&data, &result_shape))
    }
}

