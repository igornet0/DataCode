// Tensor pool for memory reuse
// This module provides a pool of tensors that can be reused to avoid frequent allocations

use crate::ml::tensor::Tensor;
use crate::ml::device::Device;
use std::collections::HashMap;
use std::sync::Mutex;

/// A pool of tensors organized by shape and device
/// This allows reusing tensor memory for tensors with the same shape and device
#[derive(Debug)]
pub struct TensorPool {
    // Map from (shape_key, device_key) to a vector of available tensors
    pools: HashMap<String, Vec<Tensor>>,
    // Maximum number of tensors to keep per shape/device combination
    max_pool_size: usize,
}

impl TensorPool {
    /// Create a new tensor pool with default max size (10 tensors per shape/device)
    pub fn new() -> Self {
        TensorPool {
            pools: HashMap::new(),
            max_pool_size: 10,
        }
    }

    /// Create a new tensor pool with custom max size
    pub fn with_max_size(max_pool_size: usize) -> Self {
        TensorPool {
            pools: HashMap::new(),
            max_pool_size,
        }
    }

    /// Get a key for the pool based on shape and device
    fn get_pool_key(shape: &[usize], device: &Device) -> String {
        // Create a unique key from shape and device
        let shape_str = shape.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let device_str = match device {
            Device::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            Device::Cuda(_) => "cuda",
            #[cfg(feature = "gpu")]
            Device::Metal(_) => "metal",
        };
        format!("{}:{}", shape_str, device_str)
    }

    /// Get a tensor from the pool, or create a new one if pool is empty
    /// Returns a tensor with the specified shape and device
    pub fn get(&mut self, shape: Vec<usize>, device: Device) -> Result<Tensor, String> {
        let key = Self::get_pool_key(&shape, &device);
        let _pool_size_before = self.len();
        let _shape_for_log = shape.clone(); // Clone for logging
        
        // Try to get a tensor from the pool
        if let Some(pool) = self.pools.get_mut(&key) {
            if let Some(mut tensor) = pool.pop() {
                // Found a tensor in the pool - clear its data and reuse
                // OPTIMIZATION: Reuse the tensor's memory allocation
                let total_size: usize = shape.iter().product();
                if tensor.data.len() >= total_size {
                    // Reuse existing allocation - just clear the data
                    tensor.data.truncate(total_size);
                    tensor.data.fill(0.0);
                } else {
                    // Need to resize - create new tensor with correct shape
                    // Note: With new Tensor structure, we can't mutate in place
                    // So we'll create a new tensor instead
                }
                // Note: With new Tensor structure using Arc<TensorStorage>,
                // we can't mutate shape/device in place. This method needs redesign.
                // For now, create a new tensor with the correct device
                let new_tensor = Tensor::zeros_with_device(shape, device);
                return Ok(new_tensor);
            }
        }

        // Pool is empty or doesn't exist - create a new tensor
        // DIAG: Log new tensor creation
        // let total_size: usize = shape_for_log.iter().product();
        // let memory_size = total_size * std::mem::size_of::<f32>();
        // if memory_size > 1024 * 1024 || pool_size_before % 100 == 0 {
        //     eprintln!(
        //         "[DIAG] TensorPool: Created new tensor - shape={:?}, size={} bytes ({} MB), pool_size={}",
        //         shape_for_log, memory_size, memory_size / (1024 * 1024), pool_size_before
        //     );
        // }
        
        Ok(Tensor::zeros_with_device(shape, device))
    }

    /// Return a tensor to the pool for reuse
    /// The tensor will be cleared and stored for future use
    pub fn return_tensor(&mut self, tensor: Tensor) {
        // Only keep tensors if pool hasn't reached max size
        let key = Self::get_pool_key(tensor.shape(), tensor.device());
        
        let pool = self.pools.entry(key).or_insert_with(Vec::new);
        
        if pool.len() < self.max_pool_size {
            // Note: With new Tensor structure using Arc<TensorStorage>,
            // we can't clear data in place. This method needs redesign.
            // For now, just store the tensor as-is (it will be overwritten on next use)
            // GPU tensor will be cleared when tensor is dropped
            pool.push(tensor);
            
            // DIAG: Log pool return
            // let pool_size_after = self.len();
            // if pool_size_after % 100 == 0 || memory_size > 1024 * 1024 {
            //     eprintln!(
            //         "[DIAG] TensorPool: Returned tensor to pool - shape={:?}, size={} bytes, pool_size={} -> {}",
            //         shape_for_log, memory_size, pool_size_before, pool_size_after
            //     );
            // }
        }
        // If pool is full, just drop the tensor (let it be freed)
    }

    /// Clear all tensors from the pool
    /// This frees all memory held by the pool
    pub fn clear(&mut self) {
        self.pools.clear();
    }

    /// Get the number of tensors currently in the pool
    pub fn len(&self) -> usize {
        self.pools.values().map(|v| v.len()).sum()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.pools.values().all(|v| v.is_empty())
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> TensorPoolStats {
        let mut total_tensors = 0;
        let mut shape_counts = HashMap::new();
        
        for (key, pool) in &self.pools {
            let count = pool.len();
            total_tensors += count;
            if count > 0 {
                shape_counts.insert(key.clone(), count);
            }
        }
        
        TensorPoolStats {
            total_tensors,
            shape_counts,
            max_pool_size: self.max_pool_size,
        }
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the tensor pool
#[derive(Debug, Clone)]
pub struct TensorPoolStats {
    pub total_tensors: usize,
    pub shape_counts: HashMap<String, usize>,
    pub max_pool_size: usize,
}

// Thread-safe global tensor pool
// This can be used across threads for tensor reuse
lazy_static::lazy_static! {
    pub static ref GLOBAL_TENSOR_POOL: Mutex<TensorPool> = Mutex::new(TensorPool::new());
}

/// Get a tensor from the global pool
pub fn get_tensor_from_pool(shape: Vec<usize>, device: Device) -> Result<Tensor, String> {
    GLOBAL_TENSOR_POOL.lock().unwrap().get(shape, device)
}

/// Return a tensor to the global pool
pub fn return_tensor_to_pool(tensor: Tensor) {
    GLOBAL_TENSOR_POOL.lock().unwrap().return_tensor(tensor);
}

/// Clear the global tensor pool
pub fn clear_global_pool() {
    GLOBAL_TENSOR_POOL.lock().unwrap().clear();
}

/// Get statistics about the global tensor pool
pub fn get_pool_stats() -> TensorPoolStats {
    GLOBAL_TENSOR_POOL.lock().unwrap().stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool_basic() {
        let mut pool = TensorPool::new();
        
        // Get a tensor from empty pool - should create new one
        let tensor1 = pool.get(vec![2, 3], Device::Cpu).unwrap();
        assert_eq!(tensor1.shape, vec![2, 3]);
        assert_eq!(tensor1.data.len(), 6);
        
        // Return it to pool
        pool.return_tensor(tensor1);
        
        // Get another tensor with same shape - should reuse
        let tensor2 = pool.get(vec![2, 3], Device::Cpu).unwrap();
        assert_eq!(tensor2.shape, vec![2, 3]);
        assert_eq!(tensor2.data.len(), 6);
        
        // Pool should have stats
        let stats = pool.stats();
        assert_eq!(stats.total_tensors, 0); // All tensors are out of pool
    }

    #[test]
    fn test_tensor_pool_different_shapes() {
        let mut pool = TensorPool::new();
        
        // Get tensors with different shapes
        let tensor1 = pool.get(vec![2, 3], Device::Cpu).unwrap();
        let tensor2 = pool.get(vec![4, 5], Device::Cpu).unwrap();
        
        pool.return_tensor(tensor1);
        pool.return_tensor(tensor2);
        
        let stats = pool.stats();
        assert_eq!(stats.total_tensors, 2);
        assert_eq!(stats.shape_counts.len(), 2);
    }

    #[test]
    fn test_tensor_pool_max_size() {
        let mut pool = TensorPool::with_max_size(2);
        
        // Create and return 3 tensors - only 2 should be kept
        let tensor1 = pool.get(vec![2, 3], Device::Cpu).unwrap();
        let tensor2 = pool.get(vec![2, 3], Device::Cpu).unwrap();
        let tensor3 = pool.get(vec![2, 3], Device::Cpu).unwrap();
        
        pool.return_tensor(tensor1);
        pool.return_tensor(tensor2);
        pool.return_tensor(tensor3);
        
        let stats = pool.stats();
        assert_eq!(stats.total_tensors, 2); // Only 2 kept due to max_size
    }
}

