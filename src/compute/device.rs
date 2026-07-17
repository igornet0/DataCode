//! Compute device kinds exposed as numeric constants to DataCode (`process.cpu`, …).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeviceKind {
    Cpu = 0,
    Gpu = 1,
    Cuda = 2,
    Metal = 3,
    Auto = 4,
}

impl DeviceKind {
    pub const ALL: [DeviceKind; 5] = [
        DeviceKind::Cpu,
        DeviceKind::Gpu,
        DeviceKind::Cuda,
        DeviceKind::Metal,
        DeviceKind::Auto,
    ];

    pub fn from_f64(n: f64) -> Option<Self> {
        if !n.is_finite() || n.fract() != 0.0 {
            return None;
        }
        match n as u8 {
            0 => Some(DeviceKind::Cpu),
            1 => Some(DeviceKind::Gpu),
            2 => Some(DeviceKind::Cuda),
            3 => Some(DeviceKind::Metal),
            4 => Some(DeviceKind::Auto),
            _ => None,
        }
    }

    pub fn as_f64(self) -> f64 {
        self as u8 as f64
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
            DeviceKind::Cuda => "cuda",
            DeviceKind::Metal => "metal",
            DeviceKind::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub backend: String,
    pub gpu_name: String,
    pub memory_mb: u64,
    pub cores: u32,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            backend: "cpu".to_string(),
            gpu_name: String::new(),
            memory_mb: 0,
            cores: num_cpus(),
        }
    }
}

pub fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1)
}
