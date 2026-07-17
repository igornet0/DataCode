//! Global compute runtime: device selection and kernel dispatch.

use crate::common::value::Value;
use crate::compute::cpu_backend::CpuBackend;
use crate::compute::device::{DeviceInfo, DeviceKind, num_cpus};
use crate::compute::dispatch::extract_f64_array;
use crate::vm::natives::utils::invoke_value_callable;

#[cfg(feature = "metal")]
use crate::compute::metal_backend::MetalBackend;

const DEFAULT_GPU_MIN_SIZE: usize = 20_000;

pub struct ComputeRuntime {
    pub device: DeviceKind,
    pub gpu_min_size: usize,
    auto_selector: Option<Value>,
    cpu: CpuBackend,
    #[cfg(feature = "metal")]
    metal: MetalBackend,
}

impl ComputeRuntime {
    pub fn new() -> Self {
        Self {
            device: DeviceKind::Cpu,
            gpu_min_size: DEFAULT_GPU_MIN_SIZE,
            auto_selector: None,
            cpu: CpuBackend::new(),
            #[cfg(feature = "metal")]
            metal: MetalBackend::new(),
        }
    }

    pub fn set_device(&mut self, kind: DeviceKind) {
        self.device = kind;
    }

    pub fn get_device(&self) -> DeviceKind {
        self.device
    }

    pub fn set_gpu_min_size(&mut self, n: usize) {
        self.gpu_min_size = n.max(1);
    }

    pub fn gpu_min_size(&self) -> usize {
        self.gpu_min_size
    }

    pub fn set_auto_selector(&mut self, cb: Option<Value>) {
        self.auto_selector = cb;
    }

    pub fn has_metal(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            return self.metal.available();
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    pub fn has_cuda(&self) -> bool {
        false
    }

    pub fn has_gpu(&self) -> bool {
        self.has_metal() || self.has_cuda()
    }

    pub fn info(&self) -> DeviceInfo {
        let mut info = DeviceInfo {
            cores: num_cpus(),
            ..DeviceInfo::default()
        };
        #[cfg(feature = "metal")]
        if self.metal.available() {
            if let Some(m) = self.metal.device_info() {
                info.backend = "metal".to_string();
                info.gpu_name = m.name;
                info.memory_mb = m.memory_mb;
                return info;
            }
        }
        info.backend = "cpu".to_string();
        info
    }

    pub fn resolve_device(&mut self, data_len: usize) -> DeviceKind {
        if let Some(ref cb) = self.auto_selector.clone() {
            if let Ok(out) = invoke_value_callable(cb, &[Value::Number(data_len as f64)]) {
                if let Some(k) = DeviceKind::from_f64(out.as_ieee_f64().unwrap_or(-1.0)) {
                    return self.normalize_device(k);
                }
            }
        }
        match self.device {
            DeviceKind::Auto => {
                if data_len >= self.gpu_min_size && self.pick_gpu_backend().is_some() {
                    DeviceKind::Gpu
                } else {
                    DeviceKind::Cpu
                }
            }
            other => self.normalize_device(other),
        }
    }

    fn normalize_device(&self, kind: DeviceKind) -> DeviceKind {
        match kind {
            DeviceKind::Gpu => {
                if self.has_metal() {
                    DeviceKind::Metal
                } else if self.has_cuda() {
                    DeviceKind::Cuda
                } else {
                    DeviceKind::Cpu
                }
            }
            DeviceKind::Metal if !self.has_metal() => DeviceKind::Cpu,
            DeviceKind::Cuda if !self.has_cuda() => DeviceKind::Cpu,
            other => other,
        }
    }

    fn pick_gpu_backend(&self) -> Option<DeviceKind> {
        if self.has_metal() {
            Some(DeviceKind::Metal)
        } else if self.has_cuda() {
            Some(DeviceKind::Cuda)
        } else {
            None
        }
    }

    pub fn should_use_gpu(&mut self, len: usize) -> bool {
        matches!(
            self.resolve_device(len),
            DeviceKind::Gpu | DeviceKind::Metal | DeviceKind::Cuda
        )
    }

    pub fn add_f64(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        let backend = self.resolve_device(a.len());
        match backend {
            #[cfg(feature = "metal")]
            DeviceKind::Metal | DeviceKind::Gpu => {
                if let Ok(v) = self.metal.add_f64(a, b) {
                    return Ok(v);
                }
                self.cpu.add_f64(a, b)
            }
            _ => self.cpu.add_f64(a, b),
        }
    }

    pub fn mul_f64(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        let backend = self.resolve_device(a.len());
        match backend {
            #[cfg(feature = "metal")]
            DeviceKind::Metal | DeviceKind::Gpu => {
                if let Ok(v) = self.metal.mul_f64(a, b) {
                    return Ok(v);
                }
                self.cpu.mul_f64(a, b)
            }
            _ => self.cpu.mul_f64(a, b),
        }
    }

    pub fn sum_f64(&mut self, data: &[f64]) -> f64 {
        let backend = self.resolve_device(data.len());
        match backend {
            #[cfg(feature = "metal")]
            DeviceKind::Metal | DeviceKind::Gpu => self
                .metal
                .sum_f64(data)
                .unwrap_or_else(|_| self.cpu.sum_f64(data)),
            _ => self.cpu.sum_f64(data),
        }
    }

    pub fn try_add_values(&mut self, a: &Value, b: &Value) -> Option<Value> {
        let (va, vb) = (extract_f64_array(a)?, extract_f64_array(b)?);
        if va.len() != vb.len() {
            return None;
        }
        let out = self.add_f64(&va, &vb).ok()?;
        Some(crate::compute::dispatch::f64_array_to_value(out))
    }

    pub fn try_mul_values(&mut self, a: &Value, b: &Value) -> Option<Value> {
        let (va, vb) = (extract_f64_array(a)?, extract_f64_array(b)?);
        if va.len() != vb.len() {
            return None;
        }
        let out = self.mul_f64(&va, &vb).ok()?;
        Some(crate::compute::dispatch::f64_array_to_value(out))
    }

    pub fn try_sum_value(&mut self, v: &Value) -> Option<f64> {
        let data = extract_f64_array(v)?;
        Some(self.sum_f64(&data))
    }
}

impl Default for ComputeRuntime {
    fn default() -> Self {
        Self::new()
    }
}
