// Dataset structure for working with tables as ML datasets

use crate::common::table::Table;
use crate::common::value::Value;
use crate::ml::tensor::Tensor;
use std::fs::File;
use std::io::{Read, BufReader};

/// Dataset for ML operations
/// Converts table columns to tensors for training
#[derive(Debug, Clone)]
pub struct Dataset {
    features: Tensor,  // [batch_size, feature_count]
    targets: Tensor,   // [batch_size, target_count] or [batch_size, 1]
    feature_names: Vec<String>,
    target_names: Vec<String>,
}

impl Dataset {
    /// Create a dataset from a table
    /// 
    /// # Arguments
    /// * `table` - The table to convert
    /// * `feature_columns` - Names of columns to use as features
    /// * `target_columns` - Names of columns to use as targets
    /// 
    /// # Returns
    /// Dataset with features and targets as tensors
    pub fn from_table(
        table: &Table,
        feature_columns: &[String],
        target_columns: &[String],
    ) -> Result<Self, String> {
        if feature_columns.is_empty() {
            return Err("At least one feature column is required".to_string());
        }
        if target_columns.is_empty() {
            return Err("At least one target column is required".to_string());
        }

        // Validate columns exist
        for col_name in feature_columns {
            if table.get_column(col_name).is_none() {
                return Err(format!("Feature column '{}' not found in table", col_name));
            }
        }
        for col_name in target_columns {
            if table.get_column(col_name).is_none() {
                return Err(format!("Target column '{}' not found in table", col_name));
            }
        }

        let num_rows = table.len();
        if num_rows == 0 {
            return Err("Table is empty".to_string());
        }

        // Extract feature columns as tensors
        let mut feature_data = Vec::new();
        for col_name in feature_columns {
            let column = table.get_column(col_name)
                .ok_or_else(|| format!("Column '{}' not found", col_name))?;
            
            for val in column.iter() {
                match val {
                    Value::Number(n) => feature_data.push(*n as f32),
                    _ => return Err(format!(
                        "Feature column '{}' contains non-numeric values. Only numeric values are supported.",
                        col_name
                    )),
                }
            }
        }

        // Reshape: [num_rows * num_features] -> [num_rows, num_features]
        let num_features = feature_columns.len();
        let features = Tensor::new(feature_data, vec![num_rows, num_features])?;

        // Extract target columns as tensors
        let mut target_data = Vec::new();
        for col_name in target_columns {
            let column = table.get_column(col_name)
                .ok_or_else(|| format!("Column '{}' not found", col_name))?;
            
            for val in column.iter() {
                match val {
                    Value::Number(n) => target_data.push(*n as f32),
                    _ => return Err(format!(
                        "Target column '{}' contains non-numeric values. Only numeric values are supported.",
                        col_name
                    )),
                }
            }
        }

        // Reshape: [num_rows * num_targets] -> [num_rows, num_targets]
        let num_targets = target_columns.len();
        let targets = Tensor::new(target_data, vec![num_rows, num_targets])?;

        Ok(Dataset {
            features,
            targets,
            feature_names: feature_columns.to_vec(),
            target_names: target_columns.to_vec(),
        })
    }

    /// Get features tensor
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// Get targets tensor
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get target names
    pub fn target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get batch size (number of samples)
    pub fn batch_size(&self) -> usize {
        self.features.shape[0]
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.features.shape[1]
    }

    /// Get number of targets
    pub fn num_targets(&self) -> usize {
        self.targets.shape[1]
    }

    /// Get batches from the dataset
    /// Returns vector of (features_batch, targets_batch) tuples
    pub fn batches(&self, batch_size: usize, shuffle: bool) -> Result<Vec<(Tensor, Tensor)>, String> {
        if batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        let num_samples = self.batch_size();
        if batch_size > num_samples {
            return Err(format!(
                "Batch size {} is larger than dataset size {}",
                batch_size, num_samples
            ));
        }

        // Create indices
        let mut indices: Vec<usize> = (0..num_samples).collect();

        // Shuffle if requested
        if shuffle {
            // Simple Fisher-Yates shuffle with fixed seed for reproducibility
            let mut seed: u64 = 42;
            for i in (1..num_samples).rev() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (seed % (i + 1) as u64) as usize;
                indices.swap(i, j);
            }
        }

        let mut batches = Vec::new();

        // Split into batches
        for chunk in indices.chunks(batch_size) {
            let batch_size_actual = chunk.len();
            
            // Extract feature batch
            let mut feature_batch_data = Vec::new();
            let num_features = self.num_features();
            
            for &idx in chunk {
                let start_idx = idx * num_features;
                let end_idx = start_idx + num_features;
                feature_batch_data.extend_from_slice(&self.features.data[start_idx..end_idx]);
            }
            
            let feature_batch = Tensor::new(feature_batch_data, vec![batch_size_actual, num_features])?;

            // Extract target batch
            let mut target_batch_data = Vec::new();
            let num_targets = self.num_targets();
            
            for &idx in chunk {
                let start_idx = idx * num_targets;
                let end_idx = start_idx + num_targets;
                target_batch_data.extend_from_slice(&self.targets.data[start_idx..end_idx]);
            }
            
            let target_batch = Tensor::new(target_batch_data, vec![batch_size_actual, num_targets])?;

            batches.push((feature_batch, target_batch));
        }

        Ok(batches)
    }

    /// Create a dataset from MNIST files
    /// 
    /// # Arguments
    /// * `images_path` - Path to MNIST images file (idx3-ubyte format)
    /// * `labels_path` - Path to MNIST labels file (idx1-ubyte format)
    /// 
    /// # Returns
    /// Dataset with features as normalized images [num_samples, 784] and targets as labels [num_samples, 1]
    pub fn from_mnist(images_path: &str, labels_path: &str) -> Result<Self, String> {
        let images = load_mnist_images(images_path)?;
        let labels = load_mnist_labels(labels_path)?;

        if images.len() != labels.len() {
            return Err(format!(
                "Mismatch: {} images but {} labels",
                images.len(),
                labels.len()
            ));
        }

        let num_samples = images.len();
        let image_size = 28 * 28; // 784 pixels per image

        // Flatten images and normalize to [0, 1]
        let mut feature_data = Vec::with_capacity(num_samples * image_size);
        for image in &images {
            for &pixel in image {
                feature_data.push(pixel as f32 / 255.0);
            }
        }

        // Convert labels to f32
        let mut target_data = Vec::with_capacity(num_samples);
        for &label in &labels {
            target_data.push(label as f32);
        }

        let features = Tensor::new(feature_data, vec![num_samples, image_size])?;
        let targets = Tensor::new(target_data, vec![num_samples, 1])?;

        Ok(Dataset {
            features,
            targets,
            feature_names: vec!["pixel".to_string(); image_size],
            target_names: vec!["label".to_string()],
        })
    }
}

/// Load MNIST images from IDX file format
/// 
/// # Arguments
/// * `path` - Path to the idx3-ubyte file
/// 
/// # Returns
/// Vector of images, each image is a Vec<u8> of 784 pixels (28x28)
pub fn load_mnist_images(path: &str) -> Result<Vec<Vec<u8>>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open images file {}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    // Read magic number (4 bytes, big-endian)
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)
        .map_err(|e| format!("Failed to read magic number: {}", e))?;
    let magic = u32::from_be_bytes(magic_bytes);
    
    if magic != 0x00000803 {
        return Err(format!("Invalid magic number: expected 0x00000803, got 0x{:08X}", magic));
    }

    // Read number of images (4 bytes, big-endian)
    let mut num_images_bytes = [0u8; 4];
    reader.read_exact(&mut num_images_bytes)
        .map_err(|e| format!("Failed to read number of images: {}", e))?;
    let num_images = u32::from_be_bytes(num_images_bytes) as usize;

    // Read number of rows (4 bytes, big-endian)
    let mut num_rows_bytes = [0u8; 4];
    reader.read_exact(&mut num_rows_bytes)
        .map_err(|e| format!("Failed to read number of rows: {}", e))?;
    let num_rows = u32::from_be_bytes(num_rows_bytes) as usize;

    // Read number of columns (4 bytes, big-endian)
    let mut num_cols_bytes = [0u8; 4];
    reader.read_exact(&mut num_cols_bytes)
        .map_err(|e| format!("Failed to read number of columns: {}", e))?;
    let num_cols = u32::from_be_bytes(num_cols_bytes) as usize;

    let image_size = num_rows * num_cols;

    // Read all image data
    let mut images = Vec::with_capacity(num_images);
    for _ in 0..num_images {
        let mut image = vec![0u8; image_size];
        reader.read_exact(&mut image)
            .map_err(|e| format!("Failed to read image data: {}", e))?;
        images.push(image);
    }

    Ok(images)
}

/// Load MNIST labels from IDX file format
/// 
/// # Arguments
/// * `path` - Path to the idx1-ubyte file
/// 
/// # Returns
/// Vector of labels (each label is 0-9)
pub fn load_mnist_labels(path: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open labels file {}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    // Read magic number (4 bytes, big-endian)
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)
        .map_err(|e| format!("Failed to read magic number: {}", e))?;
    let magic = u32::from_be_bytes(magic_bytes);
    
    if magic != 0x00000801 {
        return Err(format!("Invalid magic number: expected 0x00000801, got 0x{:08X}", magic));
    }

    // Read number of labels (4 bytes, big-endian)
    let mut num_labels_bytes = [0u8; 4];
    reader.read_exact(&mut num_labels_bytes)
        .map_err(|e| format!("Failed to read number of labels: {}", e))?;
    let num_labels = u32::from_be_bytes(num_labels_bytes) as usize;

    // Read all labels
    let mut labels = vec![0u8; num_labels];
    reader.read_exact(&mut labels)
        .map_err(|e| format!("Failed to read labels: {}", e))?;

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_from_table() {
        let data = vec![
            vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)],
            vec![Value::Number(4.0), Value::Number(5.0), Value::Number(6.0)],
        ];
        let headers = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
        let table = Table::from_data(data, Some(headers));

        let dataset = Dataset::from_table(
            &table,
            &["x1".to_string(), "x2".to_string()],
            &["y".to_string()],
        ).unwrap();

        assert_eq!(dataset.batch_size(), 2);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.num_targets(), 1);
        assert_eq!(dataset.features().shape, vec![2, 2]);
        assert_eq!(dataset.targets().shape, vec![2, 1]);
    }
}

