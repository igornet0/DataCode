//! Compute device selection and numeric kernels (CPU / Metal / future CUDA).

pub mod cpu_backend;
pub mod device;
pub mod dispatch;
pub mod runtime;

#[cfg(feature = "metal")]
pub mod metal_backend;
