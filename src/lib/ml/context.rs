// ML execution context owned by VM (no global Mutex).
// VM sets this thread-local at run() start and takes it back at end.

use crate::ml::tensor::Tensor;
use crate::ml::device::Device;
use crate::ml::graph::NodeId;
use crate::ml::gpu_cache::{GpuTensorCache, GpuCacheStats};
use crate::ml::tensor_pool::{TensorPool, TensorPoolStats};
use std::cell::RefCell;

#[cfg(feature = "gpu")]
use candle_core::Tensor as CandleTensor;

/// Context for ML execution: tensor pool and optional GPU cache.
/// Owned by VM; installed thread-local during run() so ML natives use it without global Mutex.
#[derive(Debug)]
pub struct MlContext {
    pub tensor_pool: TensorPool,
    pub gpu_cache: Option<GpuTensorCache>,
}

thread_local! {
    static ML_CONTEXT: RefCell<Option<MlContext>> = RefCell::new(None);
}

impl MlContext {
    pub fn new() -> Self {
        MlContext {
            tensor_pool: TensorPool::new(),
            gpu_cache: None,
        }
    }

    pub fn with_gpu_cache(mut self, device: Device) -> Self {
        self.gpu_cache = Some(GpuTensorCache::new(device));
        self
    }

    /// Install this context for the current thread (called by VM at run() start).
    pub fn set_current(ctx: MlContext) {
        ML_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });
    }

    /// Take the context back from the current thread (called by VM at run() end).
    pub fn take_current() -> Option<MlContext> {
        ML_CONTEXT.with(|cell| cell.borrow_mut().take())
    }

    /// Run a closure with the current thread's ML context (used by ML natives).
    pub fn with_current<F, R>(f: F) -> R
    where
        F: FnOnce(&mut MlContext) -> R,
    {
        ML_CONTEXT.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let ctx = borrow.as_mut().expect("MlContext not set. VM must set context before run().");
            f(ctx)
        })
    }

    /// Check if context is set (for fallback in public API).
    pub fn is_set() -> bool {
        ML_CONTEXT.with(|cell| cell.borrow().is_some())
    }

    pub fn get_tensor_from_pool(&mut self, shape: Vec<usize>, device: Device) -> Result<Tensor, String> {
        self.tensor_pool.get(shape, device)
    }

    pub fn return_tensor_to_pool(&mut self, tensor: Tensor) {
        self.tensor_pool.return_tensor(tensor);
    }

    pub fn clear_pool(&mut self) {
        self.tensor_pool.clear();
    }

    pub fn pool_stats(&self) -> TensorPoolStats {
        self.tensor_pool.stats()
    }

    #[cfg(feature = "gpu")]
    pub fn get_gpu_tensor_from_cache(
        &mut self,
        node_id: NodeId,
        cpu_tensor: &Tensor,
    ) -> Result<CandleTensor, String> {
        match &mut self.gpu_cache {
            Some(cache) => cache.get_or_create_gpu_tensor(node_id, cpu_tensor),
            None => Err("GPU cache not initialized.".to_string()),
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn get_gpu_tensor_from_cache(
        &mut self,
        _node_id: NodeId,
        _cpu_tensor: &Tensor,
    ) -> Result<(), String> {
        Err("GPU support not compiled".to_string())
    }

    #[cfg(feature = "gpu")]
    pub fn update_gpu_tensor_in_cache(
        &mut self,
        node_id: NodeId,
        cpu_tensor: &Tensor,
    ) -> Result<(), String> {
        match &mut self.gpu_cache {
            Some(cache) => cache.update_gpu_tensor(node_id, cpu_tensor),
            None => Err("GPU cache not initialized.".to_string()),
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn update_gpu_tensor_in_cache(
        &mut self,
        _node_id: NodeId,
        _cpu_tensor: &Tensor,
    ) -> Result<(), String> {
        Ok(())
    }

    pub fn clear_gpu_cache(&mut self) {
        if let Some(cache) = &mut self.gpu_cache {
            cache.clear();
        }
    }

    pub fn gpu_cache_stats(&self) -> Option<GpuCacheStats> {
        self.gpu_cache.as_ref().map(|c| c.stats())
    }
}

impl Default for MlContext {
    fn default() -> Self {
        Self::new()
    }
}
