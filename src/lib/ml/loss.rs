// Loss functions for ML module

use crate::ml::tensor::Tensor;

/// Mean Squared Error loss function
/// Computes MSE = mean((y_pred - y_true)^2)
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in MSE loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    // Compute (y_pred - y_true)^2
    let diff = y_pred.sub(y_true)?;
    let diff_sq = diff.mul(&diff)?;
    
    // Compute mean
    let mean_value = diff_sq.mean();
    Tensor::new(vec![mean_value], vec![1])
}

/// Binary Cross Entropy loss
/// Computes BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
pub fn binary_cross_entropy_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in BCE loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    let eps = 1e-8;
    let mut loss_sum = 0.0;
    let y_pred_arr = y_pred.data();
    let y_true_arr = y_true.data();
    let total_size = y_pred_arr.len() as f32;

    // Используем итератор для эффективного доступа к данным
    for (pred_val, true_val) in y_pred_arr.iter().zip(y_true_arr.iter()) {
        let pred_val = pred_val.max(eps).min(1.0 - eps);
        
        // BCE = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        let term1 = true_val * pred_val.ln();
        let term2 = (1.0 - true_val) * (1.0 - pred_val).ln();
        loss_sum += -(term1 + term2);
    }

    let loss = loss_sum / total_size;
    Tensor::new(vec![loss], vec![1])
}

/// Mean Absolute Error loss function (L1 loss)
/// Computes MAE = mean(|y_pred - y_true|)
pub fn mae_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in MAE loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    // Compute |y_pred - y_true|
    let diff = y_pred.sub(y_true)?;
    let abs_diff = diff.abs();
    
    // Compute mean
    let mean_value = abs_diff.mean();
    Tensor::new(vec![mean_value], vec![1])
}

/// Huber loss function (robust to outliers)
/// Computes Huber loss = 0.5 * (y_pred - y_true)^2 if |diff| <= delta
///                     else delta * |diff| - 0.5 * delta^2
pub fn huber_loss(y_pred: &Tensor, y_true: &Tensor, delta: f32) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in Huber loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    if delta <= 0.0 {
        return Err("Delta must be positive".to_string());
    }

    let diff = y_pred.sub(y_true)?;
    let diff_arr = diff.data();
    let mut loss_sum = 0.0;
    let total_size = diff_arr.len() as f32;

    for diff_val in diff_arr.iter() {
        let abs_diff = diff_val.abs();
        let loss = if abs_diff <= delta {
            0.5 * diff_val * diff_val
        } else {
            delta * abs_diff - 0.5 * delta * delta
        };
        loss_sum += loss;
    }

    let loss = loss_sum / total_size;
    Tensor::new(vec![loss], vec![1])
}

/// Hinge loss function (for SVM classification)
/// Computes Hinge = mean(max(0, 1 - y_true * y_pred))
/// where y_true should be in {-1, 1} for binary classification
pub fn hinge_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in Hinge loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    let y_pred_arr = y_pred.data();
    let y_true_arr = y_true.data();
    let mut loss_sum = 0.0;
    let total_size = y_pred_arr.len() as f32;

    // Используем итератор для эффективного доступа к данным
    for (pred_val, true_val) in y_pred_arr.iter().zip(y_true_arr.iter()) {
        let margin = 1.0 - true_val * pred_val;
        loss_sum += margin.max(0.0);
    }

    let loss = loss_sum / total_size;
    Tensor::new(vec![loss], vec![1])
}

/// Kullback-Leibler Divergence
/// Computes KL = sum(y_true * log(y_true / (y_pred + eps)))
/// Note: y_true and y_pred should be probability distributions (sum to 1)
pub fn kl_divergence(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in KL divergence: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    let eps = 1e-8;
    let mut kl_sum = 0.0;
    let y_pred_arr = y_pred.data();
    let y_true_arr = y_true.data();

    // Используем итератор для эффективного доступа к данным
    for (pred_val, true_val) in y_pred_arr.iter().zip(y_true_arr.iter()) {
        let pred_val = pred_val.max(eps);
        let true_val = true_val.max(eps);
        
        // KL = y_true * log(y_true / y_pred) = y_true * (log(y_true) - log(y_pred))
        kl_sum += true_val * (true_val.ln() - pred_val.ln());
    }

    Tensor::new(vec![kl_sum], vec![1])
}

/// Smooth L1 loss function
/// Computes Smooth L1 = mean(0.5 * x^2 if |x| < 1 else |x| - 0.5)
/// where x = y_pred - y_true
pub fn smooth_l1_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape() != y_true.shape() {
        return Err(format!(
            "Shape mismatch in Smooth L1 loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape(), y_true.shape()
        ));
    }

    let diff = y_pred.sub(y_true)?;
    let diff_arr = diff.data();
    let mut loss_sum = 0.0;
    let total_size = diff_arr.len() as f32;

    for diff_val in diff_arr.iter() {
        let abs_diff = diff_val.abs();
        let loss = if abs_diff < 1.0 {
            0.5 * diff_val * diff_val
        } else {
            abs_diff - 0.5
        };
        loss_sum += loss;
    }

    let loss = loss_sum / total_size;
    Tensor::new(vec![loss], vec![1])
}

/// Categorical Cross Entropy loss (numerically stable)
/// Computes: -mean(sum(y_true * log_softmax(logits), axis=1))
/// where logits are the raw predictions (before softmax) and y_true is one-hot encoded [batch, C]
/// 
/// For numerical stability, uses log-sum-exp trick:
/// log_softmax(x) = x - log(sum(exp(x - max(x))))
/// 
/// Contract: inputs = logits [N,C], targets = one-hot [N,C]
pub fn categorical_cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor, String> {
    if logits.ndim() != 2 || targets.ndim() != 2 {
        return Err("Categorical cross entropy loss requires 2D tensors".to_string());
    }

    let logits_shape = logits.shape();
    let targets_shape = targets.shape();
    let batch_size = logits_shape[0];
    let num_classes = logits_shape[1];

    if targets_shape[0] != batch_size {
        return Err("Batch size mismatch in categorical cross entropy loss".to_string());
    }

    if targets_shape[1] != num_classes {
        return Err(format!(
            "categorical_cross_entropy expects one-hot targets [batch, {}], got [batch, {}]. \
            Use cross_entropy for class indices [batch, 1].",
            num_classes, targets_shape[1]
        ));
    }

    let logits_arr = logits.data();
    let targets_arr = targets.data();
    let mut loss_sum = 0.0;

    // Process each sample in the batch
    for i in 0..batch_size {
        let logits_row = logits_arr.index_axis(ndarray::Axis(0), i);
        let targets_row = targets_arr.index_axis(ndarray::Axis(0), i);

        // Find max for numerical stability (log-sum-exp trick)
        let max_logit = logits_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log-sum-exp: log(sum(exp(logits - max_logit))) + max_logit
        let mut exp_sum = 0.0;
        for logit in logits_row.iter() {
            exp_sum += (logit - max_logit).exp();
        }
        // Protect against numerical issues: if exp_sum is too small, log_sum_exp becomes -Inf
        let eps = 1e-8;
        let log_sum_exp = if exp_sum > eps {
            exp_sum.ln() + max_logit
        } else {
            // Fallback: if exp_sum is too small, use max_logit directly
            max_logit
        };

        // Compute cross entropy: -sum(y_true * (logits - log_sum_exp))
        let mut sample_loss = 0.0;
        for (j, target_val) in targets_row.iter().enumerate() {
            if *target_val > 0.0 {
                let logit = logits_row[[j]];
                let log_prob = logit - log_sum_exp;
                sample_loss -= target_val * log_prob;
            }
        }

        loss_sum += sample_loss;
    }

    let loss = loss_sum / batch_size as f32;
    Tensor::new(vec![loss], vec![1])
}

/// Cross Entropy loss for sparse targets (class indices) - Canonical cross_entropy
/// Computes: -mean(log(softmax(logits)[target_class]))
/// where logits are the raw predictions [batch, C] and target_indices are class indices [batch, 1]
/// 
/// Contract: inputs = logits [N,C], targets = class indices [N,1] (int)
/// This is the canonical cross_entropy function.
pub fn sparse_softmax_cross_entropy_loss(logits: &Tensor, target_indices: &Tensor) -> Result<Tensor, String> {
    if logits.ndim() != 2 || target_indices.ndim() != 2 {
        return Err("Cross entropy loss requires 2D tensors".to_string());
    }

    let logits_shape = logits.shape();
    let target_shape = target_indices.shape();
    let batch_size = logits_shape[0];
    let num_classes = logits_shape[1];

    if target_shape[0] != batch_size {
        return Err("Batch size mismatch in cross entropy loss".to_string());
    }

    if target_shape[1] != 1 {
        return Err(format!(
            "cross_entropy expects class indices [batch, 1], got [batch, {}]. \
            Use categorical_cross_entropy for one-hot targets [batch, C].",
            target_shape[1]
        ));
    }

    let logits_arr = logits.data();
    let target_arr = target_indices.data();
    let mut loss_sum = 0.0;

    // Process each sample in the batch
    for i in 0..batch_size {
        let logits_row = logits_arr.index_axis(ndarray::Axis(0), i);
        let target_class = target_arr[[i, 0]] as usize;

        if target_class >= num_classes {
            return Err(format!(
                "Target class {} is out of range [0, {})",
                target_class, num_classes
            ));
        }

        // Find max for numerical stability
        let max_logit = logits_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log-sum-exp
        let mut exp_sum = 0.0;
        for logit in logits_row.iter() {
            exp_sum += (logit - max_logit).exp();
        }
        // Protect against numerical issues: if exp_sum is too small, log_sum_exp becomes -Inf
        // Add small epsilon to prevent log(0)
        let eps = 1e-8;
        let log_sum_exp = if exp_sum > eps {
            exp_sum.ln() + max_logit
        } else {
            // Fallback: if exp_sum is too small, use max_logit directly (all logits are very negative)
            max_logit
        };

        // Cross entropy: -(logit[target] - log_sum_exp)
        let target_logit = logits_row[[target_class]];
        let log_prob = target_logit - log_sum_exp;
        loss_sum -= log_prob;
    }

    let loss = loss_sum / batch_size as f32;
    Tensor::new(vec![loss], vec![1])
}

/// GPU version of sparse softmax cross entropy loss
/// Uses Candle GPU operations for efficient computation
#[cfg(feature = "gpu")]
pub fn sparse_softmax_cross_entropy_loss_gpu(
    logits: &Tensor,
    target_indices: &Tensor,
    device: &candle_core::Device,
) -> Result<Tensor, String> {
    use candle_core::Shape;
    
    if logits.ndim() != 2 || target_indices.ndim() != 2 {
        return Err("Cross entropy loss requires 2D tensors".to_string());
    }

    let logits_shape = logits.shape();
    let target_shape = target_indices.shape();
    let batch_size = logits_shape[0];
    let _num_classes = logits_shape[1];

    if target_shape[0] != batch_size {
        return Err("Batch size mismatch in cross entropy loss".to_string());
    }

    if target_shape[1] != 1 {
        return Err(format!(
            "cross_entropy expects class indices [batch, 1], got [batch, {}]",
            target_shape[1]
        ));
    }

    // Get GPU tensors
    let logits_gpu = if let Some(ref gpu_t) = logits.gpu_tensor {
        gpu_t.clone()
    } else {
        // Convert to GPU if not already there
        let shape = Shape::from_dims(logits_shape);
        candle_core::Tensor::from_slice(&logits.data, shape, device)
            .map_err(|e| format!("Failed to create logits GPU tensor: {}", e))?
    };

    let targets_gpu = if let Some(ref gpu_t) = target_indices.gpu_tensor {
        gpu_t.clone()
    } else {
        // Convert to GPU if not already there
        let shape = Shape::from_dims(target_shape);
        candle_core::Tensor::from_slice(&target_indices.data, shape, device)
            .map_err(|e| format!("Failed to create targets GPU tensor: {}", e))?
    };

    // Use manual cross-entropy computation on GPU
    // For numerical stability, compute log_softmax manually: logits - log(sum(exp(logits)))
    // First, find max along classes dimension (dim=1) for numerical stability
    // max(1) returns [batch, 1], we need to broadcast it to [batch, classes]
    let max_logits = logits_gpu.max_keepdim(1)
        .map_err(|e| format!("Failed to compute max: {}", e))?;
    
    // Broadcast max_logits from [batch, 1] to [batch, classes] by expanding and repeating
    // We can use unsqueeze + expand or directly subtract (Candle should handle broadcasting)
    // But to be safe, let's expand max_logits to match logits shape
    let max_logits_broadcast = max_logits.broadcast_as(logits_gpu.dims())
        .map_err(|e| format!("Failed to broadcast max_logits: {}", e))?;
    
    let logits_centered = logits_gpu.sub(&max_logits_broadcast)
        .map_err(|e| format!("Failed to center logits: {}", e))?;
    
    // Compute exp(logits_centered)
    let exp_logits = logits_centered.exp()
        .map_err(|e| format!("Failed to compute exp: {}", e))?;
    
    // Sum along classes dimension (dim=1), keeping dims -> [batch, 1]
    let sum_exp = exp_logits.sum_keepdim(1)
        .map_err(|e| format!("Failed to sum exp: {}", e))?;
    
    // Compute log(sum_exp) + max_logits = log_sum_exp
    let log_sum_exp = sum_exp.log()
        .map_err(|e| format!("Failed to compute log: {}", e))?;
    
    // Broadcast log_sum_exp and max_logits to match logits shape for final subtraction
    let log_sum_exp_broadcast = log_sum_exp.broadcast_as(logits_gpu.dims())
        .map_err(|e| format!("Failed to broadcast log_sum_exp: {}", e))?;
    let max_logits_broadcast2 = max_logits.broadcast_as(logits_gpu.dims())
        .map_err(|e| format!("Failed to broadcast max_logits again: {}", e))?;
    let log_sum_exp_final = log_sum_exp_broadcast.add(&max_logits_broadcast2)
        .map_err(|e| format!("Failed to add max: {}", e))?;
    
    // Compute log_softmax = logits - log_sum_exp
    let log_softmax = logits_gpu.sub(&log_sum_exp_final)
        .map_err(|e| format!("Failed to compute log_softmax: {}", e))?;
    
    // Convert targets to i64 and flatten from [batch, 1] to [batch]
    let targets_i64 = targets_gpu.to_dtype(candle_core::DType::I64)
        .map_err(|e| format!("Failed to convert targets to i64: {}", e))?;
    let targets_1d = targets_i64.flatten(0, 1)
        .map_err(|e| format!("Failed to flatten targets: {}", e))?;

    // Gather log_softmax values for target classes
    let targets_1d_expanded = targets_1d.unsqueeze(1)
        .map_err(|e| format!("Failed to expand targets: {}", e))?;
    let log_probs = log_softmax.gather(&targets_1d_expanded, 1)
        .map_err(|e| format!("Failed to gather log_probs: {}", e))?;
    let log_probs_flat = log_probs.flatten(0, 1)
        .map_err(|e| format!("Failed to flatten log_probs: {}", e))?;

    // Compute cross-entropy: -mean(log_probs)
    let loss_gpu = log_probs_flat.mean_all()
        .map_err(|e| format!("Failed to compute mean: {}", e))?;
    let loss_gpu_neg = loss_gpu.neg()
        .map_err(|e| format!("Failed to negate loss: {}", e))?;

    // Get loss value (mean_all returns a scalar)
    let loss_value = loss_gpu_neg.to_scalar::<f32>()
        .map_err(|e| format!("Failed to get loss value from GPU: {}", e))?;

    // Create result tensor with GPU buffer
    use crate::ml::device::Device;
    use std::sync::Arc;
    let result_device = Device::Metal(Arc::new(device.clone()));
    
    // Store GPU tensor for potential reuse
    let loss_tensor_gpu = candle_core::Tensor::from_slice(&[loss_value], &[1], device)
        .map_err(|e| format!("Failed to create loss tensor on GPU: {}", e))?;

    Ok(crate::ml::tensor::Tensor::from_gpu_tensor(
        vec![1],
        result_device,
        Some(loss_tensor_gpu),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let loss = mse_loss(&y_pred, &y_true).unwrap();
        assert_eq!(loss.to_vec()[0], 0.0);

        let y_pred = Tensor::new(vec![2.0, 3.0, 4.0], vec![3, 1]).unwrap();
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let loss = mse_loss(&y_pred, &y_true).unwrap();
        // MSE = mean((1^2, 1^2, 1^2)) = mean(1, 1, 1) = 1.0
        assert!((loss.to_vec()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_loss() {
        let y_pred = Tensor::new(vec![0.7, 0.3, 0.9], vec![3, 1]).unwrap();
        let y_true = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let loss = binary_cross_entropy_loss(&y_pred, &y_true).unwrap();
        // Loss should be positive
        assert!(loss.to_vec()[0] > 0.0);
    }
}

