// Device abstraction for CPU/GPU operations

#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Device type for tensor operations
#[derive(Debug, Clone)]
pub enum Device {
    /// CPU device (always available)
    Cpu,
    
    #[cfg(feature = "gpu")]
    /// CUDA device (NVIDIA GPUs on Linux/Windows)
    Cuda(Arc<CandleCudaDevice>),
    
    #[cfg(feature = "gpu")]
    /// Metal device (macOS GPUs)
    Metal(Arc<CandleMetalDevice>),
}

#[cfg(feature = "gpu")]
type CandleCudaDevice = candle_core::Device;

#[cfg(feature = "gpu")]
type CandleMetalDevice = candle_core::Device;

impl Device {
    /// Get the default device (auto-detect GPU if available, otherwise CPU)
    pub fn default() -> Self {
        #[cfg(feature = "gpu")]
        {
            // Try CUDA first (Linux/Windows)
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                if let Ok(device) = candle_core::Device::new_cuda(0) {
                    return Device::Cuda(Arc::new(device));
                }
            }
            
            // Try Metal (macOS)
            #[cfg(target_os = "macos")]
            {
                if let Ok(device) = candle_core::Device::new_metal(0) {
                    return Device::Metal(Arc::new(device));
                }
            }
        }
        
        // Fallback to CPU
        Device::Cpu
    }
    
    /// Create device from string specification
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Device::Cpu),
            #[cfg(feature = "gpu")]
            "cuda" | "gpu" => {
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    candle_core::Device::new_cuda(0)
                        .map(|d| Device::Cuda(Arc::new(d)))
                        .map_err(|e| format!("CUDA not available: {}", e))
                }
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                {
                    Err("CUDA only available on Linux/Windows".to_string())
                }
            }
            #[cfg(feature = "gpu")]
            "metal" => {
                #[cfg(target_os = "macos")]
                {
                    candle_core::Device::new_metal(0)
                        .map(|d| Device::Metal(Arc::new(d)))
                        .map_err(|e| format!("Metal not available: {}", e))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err("Metal only available on macOS".to_string())
                }
            }
            #[cfg(feature = "gpu")]
            "auto" => Ok(Device::default()),
            #[cfg(not(feature = "gpu"))]
            "auto" => Ok(Device::Cpu),
            _ => {
                #[cfg(feature = "gpu")]
                {
                    Err(format!("Unknown device: {}", s))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    if s == "metal" {
                        Err(format!(
                            "Unknown device: {} (GPU support not compiled). To enable Metal GPU support, reinstall DataCode: make install",
                            s
                        ))
                    } else if s == "cuda" || s == "gpu" {
                        Err(format!(
                            "Unknown device: {} (GPU support not compiled). To enable CUDA GPU support, reinstall DataCode: make install",
                            s
                        ))
                    } else {
                        Err(format!(
                            "Unknown device: {} (GPU support not compiled). To enable GPU support, reinstall DataCode: make install",
                            s
                        ))
                    }
                }
            },
        }
    }
    
    /// Check if device is GPU
    pub fn is_gpu(&self) -> bool {
        match self {
            Device::Cpu => false,
            #[cfg(feature = "gpu")]
            Device::Cuda(_) | Device::Metal(_) => true,
        }
    }
    
    /// Check if device is CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    
    /// Get candle device if GPU, return None for CPU
    #[cfg(feature = "gpu")]
    pub fn as_candle(&self) -> Option<candle_core::Device> {
        match self {
            Device::Cpu => None,
            Device::Cuda(d) | Device::Metal(d) => Some((**d).clone()),
        }
    }
    
    /// Get device name as string
    pub fn name(&self) -> &str {
        match self {
            Device::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            Device::Cuda(_) => "cuda",
            #[cfg(feature = "gpu")]
            Device::Metal(_) => "metal",
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "gpu")]
        {
            match (self, other) {
                (Device::Cpu, Device::Cpu) => true,
                (Device::Cuda(_), Device::Cuda(_)) | (Device::Metal(_), Device::Metal(_)) => {
                    // For now, consider same device type as equal
                    // In future, could compare device IDs
                    true
                }
                (Device::Cpu, Device::Cuda(_)) | (Device::Cpu, Device::Metal(_)) |
                (Device::Cuda(_), Device::Cpu) | (Device::Metal(_), Device::Cpu) |
                (Device::Cuda(_), Device::Metal(_)) | (Device::Metal(_), Device::Cuda(_)) => false,
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            // When GPU feature is not enabled, only Cpu variant exists
            matches!((self, other), (Device::Cpu, Device::Cpu))
        }
    }
}

impl Eq for Device {}

impl Default for Device {
    fn default() -> Self {
        Device::default()
    }
}

