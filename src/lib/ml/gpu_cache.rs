// GPU tensor cache for model parameters
// This module provides caching of GPU tensors to avoid frequent CPU-GPU transfers

use crate::ml::tensor::Tensor;
use crate::ml::device::Device;
use crate::ml::graph::NodeId;
use std::collections::HashMap;
use std::sync::Mutex;

#[cfg(feature = "gpu")]
use candle_core::Tensor as CandleTensor;

/// Cache for GPU tensors indexed by node ID
/// This allows reusing GPU buffers for model parameters
#[derive(Debug)]
pub struct GpuTensorCache {
    // Map from node ID to cached GPU tensor
    cache: HashMap<NodeId, CachedGpuTensor>,
    // Device for cached tensors
    device: Device,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct CachedGpuTensor {
    gpu_tensor: CandleTensor,
    shape: Vec<usize>,
    // Hash of CPU data to detect changes
    data_hash: u64,
}

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone)]
struct CachedGpuTensor {
    _shape: Vec<usize>,
    _data_hash: u64,
}

impl GpuTensorCache {
    /// Create a new GPU cache for the specified device
    pub fn new(device: Device) -> Self {
        GpuTensorCache {
            cache: HashMap::new(),
            device,
        }
    }

    /// Compute a simple hash of tensor data for change detection
    #[allow(dead_code)]
    pub(crate) fn compute_data_hash(data: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // Hash first and last few elements plus length for quick change detection
        let sample_size = data.len().min(100);
        for i in 0..sample_size {
            data[i].to_bits().hash(&mut hasher);
        }
        if data.len() > 100 {
            for i in (data.len() - 100)..data.len() {
                data[i].to_bits().hash(&mut hasher);
            }
        }
        data.len().hash(&mut hasher);
        hasher.finish()
    }

    /// Get or create a GPU tensor for the given node ID and CPU tensor
    /// Returns the GPU tensor if cached and up-to-date, otherwise creates/updates it
    #[cfg(feature = "gpu")]
    pub fn get_or_create_gpu_tensor(
        &mut self,
        node_id: NodeId,
        cpu_tensor: &Tensor,
    ) -> Result<CandleTensor, String> {
        let data_hash = Self::compute_data_hash(&cpu_tensor.data);
        
        // Check if we have a cached tensor that's still valid
        if let Some(cached) = self.cache.get(&node_id) {
            // Check if shape and data match
            if cached.shape == cpu_tensor.shape && cached.data_hash == data_hash {
                // Cache hit - return cached GPU tensor
                return Ok(cached.gpu_tensor.clone());
            }
        }

        // Cache miss or data changed - create new GPU tensor
        let candle_device = self.device.as_candle()
            .ok_or_else(|| "Invalid GPU device".to_string())?;
        
        use candle_core::Shape;
        let shape = Shape::from_dims(&cpu_tensor.shape);
        let gpu_tensor = CandleTensor::from_slice(&cpu_tensor.data, shape, &candle_device)
            .map_err(|e| format!("Failed to create GPU tensor: {}", e))?;

        // Update cache
        let cached = CachedGpuTensor {
            gpu_tensor: gpu_tensor.clone(),
            shape: cpu_tensor.shape.clone(),
            data_hash,
        };
        self.cache.insert(node_id, cached);

        Ok(gpu_tensor)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn get_or_create_gpu_tensor(
        &mut self,
        _node_id: NodeId,
        _cpu_tensor: &Tensor,
    ) -> Result<(), String> {
        Err("GPU support not compiled".to_string())
    }

    /// Update the cached GPU tensor for a node ID
    /// This should be called when parameter values change
    #[cfg(feature = "gpu")]
    pub fn update_gpu_tensor(
        &mut self,
        node_id: NodeId,
        cpu_tensor: &Tensor,
    ) -> Result<(), String> {
        // Remove old cache entry and create new one
        self.cache.remove(&node_id);
        self.get_or_create_gpu_tensor(node_id, cpu_tensor)?;
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    pub fn update_gpu_tensor(
        &mut self,
        _node_id: NodeId,
        _cpu_tensor: &Tensor,
    ) -> Result<(), String> {
        // GPU not available - no-op
        Ok(())
    }

    /// Remove a cached tensor (e.g., when node is deleted)
    pub fn remove(&mut self, node_id: NodeId) {
        self.cache.remove(&node_id);
    }

    /// Clear all cached tensors
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> GpuCacheStats {
        GpuCacheStats {
            num_cached_tensors: self.cache.len(),
            device: self.device.clone(),
        }
    }
}

/// Statistics about the GPU cache
#[derive(Debug, Clone)]
pub struct GpuCacheStats {
    pub num_cached_tensors: usize,
    pub device: Device,
}

// Thread-safe global GPU cache
lazy_static::lazy_static! {
    static ref GLOBAL_GPU_CACHE: Mutex<Option<GpuTensorCache>> = Mutex::new(None);
}

/// Initialize the global GPU cache with a device
pub fn init_global_gpu_cache(device: Device) {
    *GLOBAL_GPU_CACHE.lock().unwrap() = Some(GpuTensorCache::new(device));
}

/// Get or create a GPU tensor from the global cache
#[cfg(feature = "gpu")]
pub fn get_gpu_tensor_from_cache(
    node_id: NodeId,
    cpu_tensor: &Tensor,
) -> Result<CandleTensor, String> {
    let mut cache_guard = GLOBAL_GPU_CACHE.lock().unwrap();
    if let Some(ref mut cache) = *cache_guard {
        cache.get_or_create_gpu_tensor(node_id, cpu_tensor)
    } else {
        Err("GPU cache not initialized. Call init_global_gpu_cache() first.".to_string())
    }
}

#[cfg(not(feature = "gpu"))]
pub fn get_gpu_tensor_from_cache(
    _node_id: NodeId,
    _cpu_tensor: &Tensor,
) -> Result<(), String> {
    Err("GPU support not compiled".to_string())
}

/// Update a GPU tensor in the global cache
#[cfg(feature = "gpu")]
pub fn update_gpu_tensor_in_cache(
    node_id: NodeId,
    cpu_tensor: &Tensor,
) -> Result<(), String> {
    let mut cache_guard = GLOBAL_GPU_CACHE.lock().unwrap();
    if let Some(ref mut cache) = *cache_guard {
        cache.update_gpu_tensor(node_id, cpu_tensor)
    } else {
        Err("GPU cache not initialized".to_string())
    }
}

#[cfg(not(feature = "gpu"))]
pub fn update_gpu_tensor_in_cache(
    _node_id: NodeId,
    _cpu_tensor: &Tensor,
) -> Result<(), String> {
    Ok(())
}

/// Clear the global GPU cache
pub fn clear_global_gpu_cache() {
    if let Some(ref mut cache) = *GLOBAL_GPU_CACHE.lock().unwrap() {
        cache.clear();
    }
}

/// Get statistics about the global GPU cache
pub fn get_gpu_cache_stats() -> Option<GpuCacheStats> {
    GLOBAL_GPU_CACHE.lock().unwrap().as_ref().map(|c| c.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_cache_basic() {
        // This test requires GPU, so we'll skip it in CI
        // In a real scenario, you'd test with actual GPU device
        let device = Device::Cpu; // Use CPU for test compatibility
        let mut cache = GpuTensorCache::new(device);
        
        // Test that cache can be created
        let stats = cache.stats();
        assert_eq!(stats.num_cached_tensors, 0);
    }

    #[test]
    fn test_data_hash() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![1.0, 2.0, 3.0];
        let data3 = vec![1.0, 2.0, 4.0];
        
        let hash1 = GpuTensorCache::compute_data_hash(&data1);
        let hash2 = GpuTensorCache::compute_data_hash(&data2);
        let hash3 = GpuTensorCache::compute_data_hash(&data3);
        
        assert_eq!(hash1, hash2); // Same data should have same hash
        assert_ne!(hash1, hash3); // Different data should have different hash
    }
}

