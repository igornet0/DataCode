//! Metal compute backend (macOS / Apple Silicon). Optional: `--features metal`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub memory_mb: u64,
}

pub struct MetalBackend {
    inner: Option<&'static MetalInner>,
}

struct MetalInner {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: Mutex<HashMap<String, metal::ComputePipelineState>>,
}

static METAL_INNER: OnceLock<Option<MetalInner>> = OnceLock::new();

fn metal_inner() -> Option<&'static MetalInner> {
    METAL_INNER
        .get_or_init(|| {
            let device = metal::Device::system_default()?;
            let queue = device.new_command_queue();
            Some(MetalInner {
                device,
                queue,
                pipelines: Mutex::new(HashMap::new()),
            })
        })
        .as_ref()
}

const ADD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
"#;

const MUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void mul_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}
"#;

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            inner: metal_inner(),
        }
    }

    pub fn available(&self) -> bool {
        self.inner.is_some()
    }

    pub fn device_info(&self) -> Option<MetalDeviceInfo> {
        let inner = self.inner?;
        Some(MetalDeviceInfo {
            name: inner.device.name().to_string(),
            memory_mb: (inner.device.recommended_max_working_set_size() / (1024 * 1024)) as u64,
        })
    }

    fn pipeline(
        inner: &MetalInner,
        name: &str,
        source: &str,
        entry: &str,
    ) -> Result<metal::ComputePipelineState, String> {
        let mut guard = inner.pipelines.lock().map_err(|e| e.to_string())?;
        if let Some(p) = guard.get(name) {
            return Ok(p.clone());
        }
        let library = inner
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal library: {}", e))?;
        let function = library
            .get_function(entry, None)
            .map_err(|e| format!("Metal function: {}", e))?;
        let pipeline = inner
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Metal pipeline: {}", e))?;
        guard.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    fn run_binary_kernel(
        &self,
        pipeline_name: &str,
        source: &str,
        entry: &str,
        a: &[f64],
        b: &[f64],
    ) -> Result<Vec<f64>, String> {
        let inner = self.inner.ok_or_else(|| "Metal unavailable".to_string())?;
        let n = a.len();
        if n != b.len() {
            return Err("length mismatch".to_string());
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let pipeline = Self::pipeline(inner, pipeline_name, source, entry)?;
        let a_f32: Vec<f32> = a.iter().map(|x| *x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| *x as f32).collect();
        let buf_a = inner.device.new_buffer_with_data(
            a_f32.as_ptr() as *const _,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let buf_b = inner.device.new_buffer_with_data(
            b_f32.as_ptr() as *const _,
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let buf_out = inner.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cmd_buf = inner.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_out), 0);
        let tg_size = pipeline.max_total_threads_per_threadgroup();
        let width = tg_size.min(n as u64);
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: (n as u64 + width - 1) / width,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        let ptr = buf_out.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(slice.iter().map(|x| *x as f64).collect())
    }

    pub fn add_f64(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        self.run_binary_kernel("add_f64", ADD_SHADER, "add_arrays", a, b)
    }

    pub fn mul_f64(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        self.run_binary_kernel("mul_f64", MUL_SHADER, "mul_arrays", a, b)
    }

    pub fn sum_f64(&self, data: &[f64]) -> Result<f64, String> {
        Ok(data.iter().sum())
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}
