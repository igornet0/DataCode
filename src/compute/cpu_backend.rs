//! CPU numeric kernels (fallback and default backend).

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn available(&self) -> bool {
        true
    }

    pub fn add_f64(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        if a.len() != b.len() {
            return Err("length mismatch".to_string());
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    pub fn mul_f64(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        if a.len() != b.len() {
            return Err("length mismatch".to_string());
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
    }

    pub fn scale_f64(&self, a: &[f64], scalar: f64) -> Result<Vec<f64>, String> {
        Ok(a.iter().map(|x| x * scalar).collect())
    }

    pub fn sum_f64(&self, data: &[f64]) -> f64 {
        data.iter().sum()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
