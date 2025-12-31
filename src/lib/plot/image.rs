// Image structure for plot module

use crate::ml::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Image {
    pub data: Vec<u8>,  // RGBA pixels
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self { data, width, height }
    }

    /// Load image from file path
    pub fn load_from_path(path: &str) -> Result<Self, String> {
        use std::path::Path;
        let path = Path::new(path);
        
        // Load image using image crate
        let img = image::open(path)
            .map_err(|e| format!("Failed to open image: {}", e))?;
        
        // Convert to RGBA
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let data = rgba_img.into_raw();
        
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create image from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("Failed to load image from bytes: {}", e))?;
        
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let data = rgba_img.into_raw();
        
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create image from tensor
    /// Supports various tensor shapes:
    /// - 1D tensor [H*W] -> auto-detect dimensions (e.g., 784 -> 28x28 for MNIST)
    /// - 2D tensor [H, W] -> grayscale image
    /// - 3D tensor [C, H, W] or [H, W, C] -> RGB/RGBA image
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, String> {
        let shape = &tensor.shape;
        let data = &tensor.data;

        if data.is_empty() {
            return Err("Tensor data is empty".to_string());
        }

        // Find min and max for normalization
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        // Normalize function: maps [min_val, max_val] to [0, 255]
        let normalize = |val: f32| -> u8 {
            if range == 0.0 {
                0
            } else {
                (((val - min_val) / range) * 255.0).clamp(0.0, 255.0) as u8
            }
        };

        match shape.len() {
            1 => {
                // 1D tensor: try to auto-detect dimensions
                let size = shape[0];
                let (width, height) = if size == 784 {
                    // MNIST: 28x28
                    (28, 28)
                } else if size == 1024 {
                    // 32x32
                    (32, 32)
                } else if size == 4096 {
                    // 64x64
                    (64, 64)
                } else {
                    // Try to find a reasonable square dimension
                    let dim = (size as f64).sqrt() as usize;
                    if dim * dim == size {
                        (dim as u32, dim as u32)
                    } else {
                        // Not a perfect square, use as width with height 1
                        (size as u32, 1)
                    }
                };

                // Convert to grayscale RGBA
                let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
                for i in 0..(width * height) as usize {
                    let val = if i < data.len() { normalize(data[i]) } else { 0 };
                    rgba_data.push(val); // R
                    rgba_data.push(val); // G
                    rgba_data.push(val); // B
                    rgba_data.push(255); // A
                }

                Ok(Self {
                    data: rgba_data,
                    width,
                    height,
                })
            }
            2 => {
                // 2D tensor [H, W] -> grayscale
                let height = shape[0] as u32;
                let width = shape[1] as u32;

                let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
                for i in 0..(width * height) as usize {
                    let val = if i < data.len() { normalize(data[i]) } else { 0 };
                    rgba_data.push(val); // R
                    rgba_data.push(val); // G
                    rgba_data.push(val); // B
                    rgba_data.push(255); // A
                }

                Ok(Self {
                    data: rgba_data,
                    width,
                    height,
                })
            }
            3 => {
                // 3D tensor: could be [C, H, W] or [H, W, C]
                let first = shape[0];
                let second = shape[1];
                let third = shape[2];

                if first == 3 || first == 4 {
                    // [C, H, W] format
                    let channels = first;
                    let height = second as u32;
                    let width = third as u32;

                    let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
                    for h in 0..height as usize {
                        for w in 0..width as usize {
                            let base_idx = (h * width as usize + w) * channels;
                            let r = if base_idx < data.len() { normalize(data[base_idx]) } else { 0 };
                            let g = if base_idx + 1 < data.len() { normalize(data[base_idx + 1]) } else { 0 };
                            let b = if base_idx + 2 < data.len() { normalize(data[base_idx + 2]) } else { 0 };
                            let a = if channels == 4 && base_idx + 3 < data.len() {
                                normalize(data[base_idx + 3])
                            } else {
                                255
                            };
                            rgba_data.push(r);
                            rgba_data.push(g);
                            rgba_data.push(b);
                            rgba_data.push(a);
                        }
                    }

                    Ok(Self {
                        data: rgba_data,
                        width,
                        height,
                    })
                } else if third == 3 || third == 4 {
                    // [H, W, C] format
                    let height = first as u32;
                    let width = second as u32;
                    let channels = third;

                    let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
                    for h in 0..height as usize {
                        for w in 0..width as usize {
                            let base_idx = (h * width as usize + w) * channels;
                            let r = if base_idx < data.len() { normalize(data[base_idx]) } else { 0 };
                            let g = if base_idx + 1 < data.len() { normalize(data[base_idx + 1]) } else { 0 };
                            let b = if base_idx + 2 < data.len() { normalize(data[base_idx + 2]) } else { 0 };
                            let a = if channels == 4 && base_idx + 3 < data.len() {
                                normalize(data[base_idx + 3])
                            } else {
                                255
                            };
                            rgba_data.push(r);
                            rgba_data.push(g);
                            rgba_data.push(b);
                            rgba_data.push(a);
                        }
                    }

                    Ok(Self {
                        data: rgba_data,
                        width,
                        height,
                    })
                } else {
                    Err(format!(
                        "Unsupported 3D tensor shape: {:?}. Expected [C, H, W] or [H, W, C] where C is 3 or 4",
                        shape
                    ))
                }
            }
            _ => Err(format!(
                "Unsupported tensor shape: {:?}. Expected 1D, 2D, or 3D tensor",
                shape
            )),
        }
    }
}

