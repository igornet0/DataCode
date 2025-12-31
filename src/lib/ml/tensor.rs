// Tensor type and operations for ML module

use crate::ml::device::Device;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub device: Device,
    
    #[cfg(feature = "gpu")]
    /// Lazy GPU tensor storage (None means not yet moved to GPU)
    #[allow(dead_code)] // Intended for future GPU tensor caching
    pub(crate) gpu_tensor: Option<candle_core::Tensor>,
}

// Simple PRNG for weight initialization
// Using a simple LCG (Linear Congruential Generator)
struct SimpleRNG {
    state: u64,
}

impl SimpleRNG {
    fn new(seed: u64) -> Self {
        SimpleRNG { state: seed }
    }

    fn next(&mut self) -> u64 {
        // LCG parameters (same as used in glibc)
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        // Convert to [0, 1) range
        // Take lower 32 bits and divide by 2^32 to get [0, 1)
        let state = self.next();
        let mut val = (state & 0xFFFFFFFF) as f32 / 4294967296.0; // 2^32
        // Ensure we never get exactly 0 or 1 to avoid issues with ln(0)
        if val == 0.0 {
            val = 1.0 / 4294967296.0; // Smallest non-zero value
        }
        if val >= 1.0 {
            val = 1.0 - 1.0 / 4294967296.0; // Largest value < 1.0
        }
        val
    }

    fn next_normal(&mut self) -> f32 {
        // Box-Muller transform for normal distribution
        // We need u1 and u2 in (0, 1) to avoid ln(0) and ensure valid sqrt
        let mut u1 = self.next_f32();
        let mut u2 = self.next_f32();
        
        // Ensure u1 and u2 are in (0, 1) not [0, 1)
        // This is critical for Box-Muller transform to avoid ln(0) = -Inf
        if u1 <= 0.0 {
            u1 = 1.0 / 65536.0;
        }
        if u1 >= 1.0 {
            u1 = 1.0 - 1.0 / 65536.0;
        }
        if u2 <= 0.0 {
            u2 = 1.0 / 65536.0;
        }
        if u2 >= 1.0 {
            u2 = 1.0 - 1.0 / 65536.0;
        }
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        
        // Check for NaN/Inf (should not happen with proper u1, u2)
        if z0.is_nan() || z0.is_infinite() {
            // Fallback to 0 if something went wrong
            0.0
        } else {
            z0
        }
    }
}

impl Tensor {
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
        
        Ok(Tensor { 
            data, 
            shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }
    
    /// Create tensor with specific device
    pub fn new_with_device(data: Vec<f32>, shape: Vec<usize>, device: Device) -> Result<Self, String> {
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
        
        Ok(Tensor { 
            data, 
            shape,
            device,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        let size = data.len();
        Tensor {
            data,
            shape: vec![size],
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }
    
    /// Create zeros tensor on specific device
    pub fn zeros_with_device(shape: Vec<usize>, device: Device) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
            device,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; size],
            shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Create tensor with random values from standard normal distribution N(0, 1)
    pub fn randn(shape: Vec<usize>) -> Result<Self, String> {
        if shape.is_empty() {
            return Err("Shape cannot be empty".to_string());
        }
        
        if shape.iter().any(|&s| s == 0) {
            return Err("Shape dimensions cannot be zero".to_string());
        }

        let size: usize = shape.iter().product();
        let mut rng = SimpleRNG::new(42); // Fixed seed for reproducibility
        
        let data: Vec<f32> = (0..size).map(|_| rng.next_normal()).collect();
        
        Ok(Tensor { 
            data, 
            shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    /// Xavier/Glorot uniform initialization
    /// Returns tensor with values from uniform distribution U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
    pub fn xavier_uniform(shape: Vec<usize>, fan_in: usize, fan_out: usize) -> Result<Self, String> {
        if shape.is_empty() {
            return Err("Shape cannot be empty".to_string());
        }
        
        if shape.iter().any(|&s| s == 0) {
            return Err("Shape dimensions cannot be zero".to_string());
        }

        let size: usize = shape.iter().product();
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        
        let mut rng = SimpleRNG::new(42);
        
        // Uniform distribution: U(-limit, limit)
        let data: Vec<f32> = (0..size)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit)
            .collect();
        
        Ok(Tensor { 
            data, 
            shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    /// He initialization for ReLU networks
    /// Returns tensor with values from normal distribution N(0, sqrt(2 / fan_in))
    pub fn he_normal(shape: Vec<usize>, fan_in: usize) -> Result<Self, String> {
        if shape.is_empty() {
            return Err("Shape cannot be empty".to_string());
        }
        
        if shape.iter().any(|&s| s == 0) {
            return Err("Shape dimensions cannot be zero".to_string());
        }

        let size: usize = shape.iter().product();
        let std_dev = (2.0 / fan_in as f32).sqrt();
        
        let mut rng = SimpleRNG::new(42);
        
        // Normal distribution: N(0, std_dev)
        let data: Vec<f32> = (0..size)
            .map(|_| rng.next_normal() * std_dev)
            .collect();
        
        Ok(Tensor { 
            data, 
            shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }
    
    /// Check if tensor is on GPU
    pub fn is_on_gpu(&self) -> bool {
        self.device.is_gpu()
    }
    
    /// Check if tensor is on CPU
    pub fn is_on_cpu(&self) -> bool {
        self.device.is_cpu()
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Convert tensor to different device
    /// This will move data between CPU and GPU
    pub fn to_device(&self, target_device: &Device) -> Result<Self, String> {
        if self.device == *target_device {
            return Ok(self.clone());
        }
        
        // If target is CPU, sync data from GPU if needed
        if target_device.is_cpu() {
            #[cfg(feature = "gpu")]
            {
                // If tensor is on GPU, read data from GPU tensor
                if let Some(ref gpu_t) = self.gpu_tensor {
                    // Flatten tensor to 1D before reading (to_vec1 only works for 1D tensors)
                    // Calculate total size and reshape to 1D
                    let total_size: usize = self.shape.iter().product();
                    let rank = gpu_t.rank();
                    
                    // Get data based on tensor rank
                    let gpu_data = if rank == 1 {
                        // Already 1D, use to_vec1 directly
                        gpu_t.to_vec1::<f32>()
                            .map_err(|e| format!("Failed to read 1D tensor from GPU: {}", e))?
                    } else if rank == 2 {
                        // 2D tensor, use to_vec2 and flatten
                        let data_2d = gpu_t.to_vec2::<f32>()
                            .map_err(|e| format!("Failed to read 2D tensor from GPU: {}", e))?;
                        data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
                    } else {
                        // Higher rank tensor, reshape to 1D first
                        let flattened = gpu_t.reshape((total_size,))
                            .map_err(|e| format!("Failed to reshape GPU tensor to 1D: {}", e))?;
                        flattened.to_vec1::<f32>()
                            .map_err(|e| format!("Failed to convert reshaped tensor to Vec: {}", e))?
                    };
                    return Ok(Tensor {
                        data: gpu_data,
                        shape: self.shape.clone(),
                        device: Device::Cpu,
                        gpu_tensor: None,
                    });
                }
            }
            // If already on CPU or no GPU tensor, just copy
            return Ok(Tensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
                device: Device::Cpu,
                #[cfg(feature = "gpu")]
                gpu_tensor: None,
            });
        }
        
        // Moving to GPU
        #[cfg(feature = "gpu")]
        {
            if let Some(candle_device) = target_device.as_candle() {
                use candle_core::Shape;
                
                // Create candle tensor on GPU
                let shape = Shape::from_dims(&self.shape);
                let gpu_t = candle_core::Tensor::from_slice(&self.data, shape, &candle_device)
                    .map_err(|e| format!("Failed to move tensor to GPU: {}", e))?;
                
                // Return tensor with GPU device (but keep CPU data for compatibility)
                Ok(Tensor {
                    data: self.data.clone(), // Keep CPU copy for now
                    shape: self.shape.clone(),
                    device: target_device.clone(),
                    gpu_tensor: Some(gpu_t),
                })
            } else {
                Err("Invalid GPU device".to_string())
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err("GPU support not compiled".to_string())
        }
    }
    
    /// Ensure tensor data is on CPU (for reading)
    pub fn to_cpu(&self) -> Result<Self, String> {
        self.to_device(&Device::Cpu)
    }

    pub fn total_size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn get_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        
        let mut index = 0;
        let mut stride = 1;
        
        for i in (0..indices.len()).rev() {
            if indices[i] >= self.shape[i] {
                return None;
            }
            index += indices[i] * stride;
            stride *= self.shape[i];
        }
        
        Some(index)
    }

    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        self.get_index(indices).map(|idx| self.data[idx])
    }

    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<(), String> {
        if let Some(idx) = self.get_index(indices) {
            self.data[idx] = value;
            Ok(())
        } else {
            Err("Invalid indices".to_string())
        }
    }

    /// Get a row/slice along the first dimension
    /// For tensor with shape [N, M, ...] and index i, returns tensor with shape [M, ...]
    /// For 1D tensor [N] and index i, returns the scalar value at index i
    pub fn get_row(&self, index: usize) -> Result<Tensor, String> {
        // Convert to CPU to ensure data is available
        let self_cpu = self.to_cpu()?;
        
        if self_cpu.shape.is_empty() {
            return Err("Cannot index empty tensor".to_string());
        }

        // For 1D tensors, we'll handle scalar return in VM
        // Here we return a 1-element tensor for consistency
        if self_cpu.shape.len() == 1 {
            if index >= self_cpu.shape[0] {
                return Err(format!("Index {} out of bounds for dimension 0 (size: {})", index, self_cpu.shape[0]));
            }
            return Ok(Tensor {
                data: vec![self_cpu.data[index]],
                shape: vec![1],
                device: Device::Cpu,
                #[cfg(feature = "gpu")]
                gpu_tensor: None,
            });
        }

        // For 2D+ tensors, extract the slice along first dimension
        if index >= self_cpu.shape[0] {
            return Err(format!("Index {} out of bounds for dimension 0 (size: {})", index, self_cpu.shape[0]));
        }

        // Calculate stride: size of one slice along first dimension
        let stride: usize = self_cpu.shape[1..].iter().product();
        let start = index * stride;
        let end = start + stride;

        // Extract the slice
        let slice_data: Vec<f32> = self_cpu.data[start..end].to_vec();
        
        // New shape: all dimensions except the first
        let new_shape = self_cpu.shape[1..].to_vec();

        Ok(Tensor {
            data: slice_data,
            shape: new_shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }
        
        let data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            device: Device::Cpu, // Operations default to CPU for now
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }
        
        // Try GPU if both tensors are on GPU device
        #[cfg(feature = "gpu")]
        {
            if let (Device::Cuda(_) | Device::Metal(_), 
                    Device::Cuda(_) | Device::Metal(_)) = (&self.device, &other.device) {
                if let Some(device) = self.device.as_candle() {
                    return self.sub_gpu(other, &device);
                }
            }
        }
        
        // CPU fallback
        let data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }
        
        // Try GPU if both tensors are on GPU device
        #[cfg(feature = "gpu")]
        {
            if let (Device::Cuda(_) | Device::Metal(_), 
                    Device::Cuda(_) | Device::Metal(_)) = (&self.device, &other.device) {
                if let Some(device) = self.device.as_candle() {
                    return self.mul_gpu(other, &device);
                }
            }
        }
        
        // CPU fallback
        let data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }
        
        // Try GPU if both tensors are on GPU device
        #[cfg(feature = "gpu")]
        {
            if let (Device::Cuda(_) | Device::Metal(_), 
                    Device::Cuda(_) | Device::Metal(_)) = (&self.device, &other.device) {
                if let Some(device) = self.device.as_candle() {
                    return self.div_gpu(other, &device);
                }
            }
        }
        
        // CPU fallback - ensure both tensors are on CPU before accessing .data
        let self_cpu = self.to_cpu()?;
        let other_cpu = other.to_cpu()?;
        
        // Check for division by zero
        if other_cpu.data.iter().any(|&x| x == 0.0) {
            return Err("Division by zero".to_string());
        }
        
        let data: Vec<f32> = self_cpu.data
            .iter()
            .zip(other_cpu.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor, String> {
        if scalar == 0.0 {
            return Err("Division by zero".to_string());
        }
        
        // Try GPU if tensor is on GPU device
        #[cfg(feature = "gpu")]
        {
            if let Device::Cuda(_) | Device::Metal(_) = &self.device {
                if let Some(device) = self.device.as_candle() {
                    return self.div_scalar_gpu(scalar, &device);
                }
            }
        }
        
        // CPU fallback - ensure tensor is on CPU before accessing .data
        let self_cpu = self.to_cpu()?;
        
        let data: Vec<f32> = self_cpu.data
            .iter()
            .map(|&x| x / scalar)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        // Matrix multiplication: (m, n) @ (n, p) = (m, p)
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("matmul requires 2D tensors".to_string());
        }
        
        // Check shapes for compatibility
        let n = self.shape[1];
        if other.shape[0] != n {
            return Err(format!(
                "Incompatible shapes for matmul: {:?} @ {:?}",
                self.shape, other.shape
            ));
        }
        
        // Try GPU if both tensors are on GPU device (matmul is most critical for GPU)
        #[cfg(feature = "gpu")]
        {
            if let (Device::Cuda(_) | Device::Metal(_), 
                    Device::Cuda(_) | Device::Metal(_)) = (&self.device, &other.device) {
                if let Some(device) = self.device.as_candle() {
                    return self.matmul_gpu(other, &device);
                }
            }
        }
        
        // CPU fallback - ensure both tensors are on CPU before accessing .data
        let self_cpu = self.to_cpu()?;
        let other_cpu = other.to_cpu()?;
        
        // Get shapes after CPU conversion (should be same)
        let m = self_cpu.shape[0];
        let n = self_cpu.shape[1];
        let p = other_cpu.shape[1];
        
        let mut result_data = vec![0.0; m * p];
        
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self_cpu.data[i * n + k] * other_cpu.data[k * p + j];
                }
                result_data[i * p + j] = sum;
            }
        }
        
        Ok(Tensor {
            data: result_data,
            shape: vec![m, p],
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn transpose(&self) -> Result<Tensor, String> {
        if self.ndim() != 2 {
            return Err("transpose requires 2D tensor".to_string());
        }
        
        // Ensure tensor is on CPU before accessing .data
        let self_cpu = self.to_cpu()?;
        
        let m = self_cpu.shape[0];
        let n = self_cpu.shape[1];
        let mut result_data = vec![0.0; m * n];
        
        for i in 0..m {
            for j in 0..n {
                result_data[j * m + i] = self_cpu.data[i * n + j];
            }
        }
        
        Ok(Tensor {
            data: result_data,
            shape: vec![n, m],
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn sum(&self) -> f32 {
        // Ensure tensor is on CPU before accessing .data
        match self.to_cpu() {
            Ok(cpu_tensor) => cpu_tensor.data.iter().sum(),
            Err(_) => 0.0, // Fallback to 0 on error (shouldn't happen)
        }
    }

    /// Sum tensor to target shape, summing over broadcasted dimensions
    /// This is used in backward pass when an input was broadcasted during forward pass
    /// target_shape: the shape to sum to (must be compatible with self.shape via broadcasting rules)
    pub fn sum_to_shape(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        // Pad target_shape with 1s from the left to match self.shape length if needed
        let padded_target_shape = if target_shape.len() < self.shape.len() {
            let mut padded = vec![1; self.shape.len() - target_shape.len()];
            padded.extend_from_slice(target_shape);
            padded
        } else {
            target_shape.to_vec()
        };

        if padded_target_shape.len() != self.shape.len() {
            return Err(format!(
                "Cannot sum from shape {:?} to {:?}: dimension count mismatch",
                self.shape, target_shape
            ));
        }

        // Check that target_shape can be broadcasted to self.shape
        for i in 0..self.shape.len() {
            if padded_target_shape[i] != 1 && padded_target_shape[i] != self.shape[i] {
                return Err(format!(
                    "Cannot sum from shape {:?} to {:?}: dimension {} incompatible ({} vs {})",
                    self.shape, target_shape, i, self.shape[i], padded_target_shape[i]
                ));
            }
        }

        // Calculate strides for input tensor
        let mut input_strides = vec![1; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            input_strides[i] = input_strides[i + 1] * self.shape[i + 1];
        }

        // Calculate strides for output tensor
        let mut output_strides = vec![1; padded_target_shape.len()];
        for i in (0..padded_target_shape.len() - 1).rev() {
            output_strides[i] = output_strides[i + 1] * padded_target_shape[i + 1];
        }

        // Calculate output size
        let output_size: usize = padded_target_shape.iter().product();
        let mut output_data = vec![0.0; output_size];

        // Iterate over all input elements and accumulate to output
        for input_idx in 0..self.data.len() {
            // Calculate input indices from flat index
            let mut input_indices = vec![0; self.shape.len()];
            let mut remaining = input_idx;
            for dim in (0..self.shape.len()).rev() {
                input_indices[dim] = remaining % self.shape[dim];
                remaining /= self.shape[dim];
            }

            // Calculate corresponding output indices (only non-broadcasted dimensions matter)
            let mut output_indices = vec![0; padded_target_shape.len()];
            for dim in 0..padded_target_shape.len() {
                if padded_target_shape[dim] == 1 {
                    // This dimension was broadcasted, output index is always 0
                    output_indices[dim] = 0;
                } else {
                    // This dimension was not broadcasted, use input index
                    output_indices[dim] = input_indices[dim];
                }
            }

            // Calculate flat output index
            let mut output_idx = 0;
            for dim in 0..padded_target_shape.len() {
                output_idx += output_indices[dim] * output_strides[dim];
            }

            output_data[output_idx] += self.data[input_idx];
        }

        // Use target_shape (not padded) for final output
        Ok(Tensor {
            data: output_data,
            shape: target_shape.to_vec(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    pub fn mean(&self) -> f32 {
        // Ensure tensor is on CPU before accessing .data
        match self.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    0.0
                } else {
                    cpu_tensor.data.iter().sum::<f32>() / cpu_tensor.data.len() as f32
                }
            }
            Err(_) => 0.0, // Fallback to 0 on error (shouldn't happen)
        }
    }

    /// Find index(es) of maximum element(s)
    /// For 1D tensors: returns Vec with single element (index of max)
    /// For multi-dimensional tensors: returns Vec with indices for each slice along first dimension
    pub fn max_idx(&self) -> Result<Vec<usize>, String> {
        let cpu_tensor = self.to_cpu()?;
        
        if cpu_tensor.data.is_empty() {
            return Err("Cannot find max index in empty tensor".to_string());
        }

        if cpu_tensor.shape.is_empty() {
            return Err("Cannot find max index in tensor with empty shape".to_string());
        }

        // For 1D tensors: find single max index
        if cpu_tensor.shape.len() == 1 {
            let max_idx = cpu_tensor.data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find max index")?;
            return Ok(vec![max_idx]);
        }

        // For multi-dimensional tensors: find max index for each slice along first dimension
        let first_dim_size = cpu_tensor.shape[0];
        let slice_size: usize = cpu_tensor.shape[1..].iter().product();
        let mut result = Vec::with_capacity(first_dim_size);

        for i in 0..first_dim_size {
            let start = i * slice_size;
            let end = start + slice_size;
            let slice = &cpu_tensor.data[start..end];
            
            let max_idx = slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find max index in slice")?;
            
            result.push(max_idx);
        }

        Ok(result)
    }

    /// Find index(es) of minimum element(s)
    /// For 1D tensors: returns Vec with single element (index of min)
    /// For multi-dimensional tensors: returns Vec with indices for each slice along first dimension
    pub fn min_idx(&self) -> Result<Vec<usize>, String> {
        let cpu_tensor = self.to_cpu()?;
        
        if cpu_tensor.data.is_empty() {
            return Err("Cannot find min index in empty tensor".to_string());
        }

        if cpu_tensor.shape.is_empty() {
            return Err("Cannot find min index in tensor with empty shape".to_string());
        }

        // For 1D tensors: find single min index
        if cpu_tensor.shape.len() == 1 {
            let min_idx = cpu_tensor.data
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find min index")?;
            return Ok(vec![min_idx]);
        }

        // For multi-dimensional tensors: find min index for each slice along first dimension
        let first_dim_size = cpu_tensor.shape[0];
        let slice_size: usize = cpu_tensor.shape[1..].iter().product();
        let mut result = Vec::with_capacity(first_dim_size);

        for i in 0..first_dim_size {
            let start = i * slice_size;
            let end = start + slice_size;
            let slice = &cpu_tensor.data[start..end];
            
            let min_idx = slice
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or("Failed to find min index in slice")?;
            
            result.push(min_idx);
        }

        Ok(result)
    }

    /// Broadcast tensor to target shape
    /// Broadcasting rules: dimensions can be 1 or match exactly
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        if target_shape.len() < self.shape.len() {
            return Err(format!(
                "Cannot broadcast from shape {:?} to {:?}: target has fewer dimensions",
                self.shape, target_shape
            ));
        }

        // Pad self.shape with 1s from the left to match target_shape length
        let mut padded_shape = vec![1; target_shape.len() - self.shape.len()];
        padded_shape.extend_from_slice(&self.shape);

        // Check broadcasting compatibility
        for i in 0..target_shape.len() {
            if padded_shape[i] != target_shape[i] && padded_shape[i] != 1 {
                return Err(format!(
                    "Cannot broadcast from shape {:?} to {:?}: dimension {} incompatible ({} vs {})",
                    self.shape, target_shape, i, padded_shape[i], target_shape[i]
                ));
            }
        }

        // Calculate strides for both shapes
        let mut src_strides = vec![1; padded_shape.len()];
        for i in (0..padded_shape.len() - 1).rev() {
            src_strides[i] = src_strides[i + 1] * padded_shape[i + 1];
        }

        let mut dst_strides = vec![1; target_shape.len()];
        for i in (0..target_shape.len() - 1).rev() {
            dst_strides[i] = dst_strides[i + 1] * target_shape[i + 1];
        }

        // Create result tensor
        let total_size: usize = target_shape.iter().product();
        let mut result_data = vec![0.0; total_size];

        // Broadcast by iterating over target shape and mapping to source
        for flat_idx in 0..total_size {
            let mut src_idx = 0;
            let mut remaining = flat_idx;

            // Calculate source index from target index
            for (dim, &dst_stride) in dst_strides.iter().enumerate() {
                let dst_coord = remaining / dst_stride;
                remaining %= dst_stride;

                let src_coord = if padded_shape[dim] == 1 {
                    0 // Broadcast dimension
                } else {
                    dst_coord
                };

                src_idx += src_coord * src_strides[dim];
            }

            result_data[flat_idx] = self.data[src_idx];
        }

        Ok(Tensor {
            data: result_data,
            shape: target_shape.to_vec(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    /// Expand a scalar tensor (shape [1]) to target shape
    pub fn expand(&self, target_shape: Vec<usize>) -> Result<Tensor, String> {
        if self.total_size() != 1 {
            return Err("expand() can only be used on scalar tensors (size 1)".to_string());
        }

        let value = self.data[0];
        Ok(Tensor {
            data: vec![value; target_shape.iter().product()],
            shape: target_shape,
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    /// Add another tensor and accumulate (for gradient accumulation)
    pub fn add_assign(&mut self, other: &Tensor) -> Result<(), String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch in add_assign: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }

        Ok(())
    }

    /// Negate tensor (multiply by -1)
    pub fn neg(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|&x| -x).collect(),
            shape: self.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        // Try GPU if tensor is on GPU device
        #[cfg(feature = "gpu")]
        {
            if let Device::Cuda(_) | Device::Metal(_) = &self.device {
                if let Some(device) = self.device.as_candle() {
                    if let Ok(result) = self.relu_gpu(&device) {
                        return result;
                    }
                }
            }
        }
        
        // CPU fallback - ensure tensor is on CPU before accessing .data
        let self_cpu = match self.to_cpu() {
            Ok(t) => t,
            Err(_) => return self.clone(), // Fallback to clone on error
        };
        Tensor {
            data: self_cpu.data.iter().map(|&x| x.max(0.0)).collect(),
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        // Ensure tensor is on CPU before accessing .data
        let self_cpu = match self.to_cpu() {
            Ok(t) => t,
            Err(_) => return self.clone(), // Fallback to clone on error
        };
        Tensor {
            data: self_cpu.data.iter().map(|&x| {
                if x > 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let exp_x = x.exp();
                    exp_x / (1.0 + exp_x)
                }
            }).collect(),
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        // Ensure tensor is on CPU before accessing .data
        let self_cpu = match self.to_cpu() {
            Ok(t) => t,
            Err(_) => return self.clone(), // Fallback to clone on error
        };
        Tensor {
            data: self_cpu.data.iter().map(|&x| x.tanh()).collect(),
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Tensor {
        // Ensure tensor is on CPU before accessing .data
        let self_cpu = match self.to_cpu() {
            Ok(t) => t,
            Err(_) => return self.clone(), // Fallback to clone on error
        };
        Tensor {
            data: self_cpu.data.iter().map(|&x| x.abs()).collect(),
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor, String> {
        // Ensure tensor is on CPU before accessing .data
        let self_cpu = self.to_cpu()?;
        
        // Check for negative values
        if self_cpu.data.iter().any(|&x| x < 0.0) {
            return Err("Square root of negative number".to_string());
        }
        
        Ok(Tensor {
            data: self_cpu.data.iter().map(|&x| x.sqrt()).collect(),
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    /// Element-wise rounding
    pub fn round(&self) -> Tensor {
        // Ensure tensor is on CPU before accessing .data
        let self_cpu = match self.to_cpu() {
            Ok(t) => t,
            Err(_) => return self.clone(), // Fallback to clone on error
        };
        Tensor {
            data: self_cpu.data.iter().map(|&x| x.round()).collect(),
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        }
    }

    /// Softmax activation (numerically stable)
    /// Applies softmax along the last dimension
    pub fn softmax(&self) -> Result<Tensor, String> {
        if self.ndim() < 1 {
            return Err("Softmax requires at least 1 dimension".to_string());
        }

        // Ensure tensor is on CPU before accessing .data
        let self_cpu = self.to_cpu()?;

        let last_dim = self_cpu.shape[self_cpu.shape.len() - 1];
        let other_dims: usize = if self_cpu.shape.len() > 1 {
            self_cpu.shape[0..self_cpu.shape.len() - 1].iter().product()
        } else {
            1
        };

        let mut result_data = vec![0.0; self_cpu.data.len()];

        // Process each row (or each element if 1D)
        for i in 0..other_dims {
            let start_idx = i * last_dim;
            let end_idx = start_idx + last_dim;

            // Find max for numerical stability
            let max_val = self_cpu.data[start_idx..end_idx]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0;
            let mut exp_values = Vec::new();
            for j in start_idx..end_idx {
                let exp_val = (self_cpu.data[j] - max_val).exp();
                exp_values.push(exp_val);
                exp_sum += exp_val;
            }

            // Normalize
            for (k, exp_val) in exp_values.iter().enumerate() {
                result_data[start_idx + k] = exp_val / exp_sum;
            }
        }

        Ok(Tensor {
            data: result_data,
            shape: self_cpu.shape.clone(),
            device: Device::Cpu,
            #[cfg(feature = "gpu")]
            gpu_tensor: None,
        })
    }

    /// Flatten tensor: reshape to [batch_size, -1]
    /// Preserves the first dimension (batch) and flattens all other dimensions
    /// Example: [batch, 28, 28] -> [batch, 784]
    pub fn flatten(&self) -> Result<Tensor, String> {
        if self.ndim() < 2 {
            return Err("Flatten requires at least 2 dimensions".to_string());
        }

        let batch_size = self.shape[0];
        let flattened_size: usize = self.shape[1..].iter().product();
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: vec![batch_size, flattened_size],
            device: self.device.clone(),
            #[cfg(feature = "gpu")]
            gpu_tensor: None, // Reshape doesn't preserve GPU tensor
        })
    }

    /// Reshape tensor to new shape
    /// Total size must match
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

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device.clone(),
            #[cfg(feature = "gpu")]
            gpu_tensor: None, // Reshape doesn't preserve GPU tensor
        })
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

