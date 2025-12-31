// Loss functions for ML module

use crate::ml::tensor::Tensor;

/// Mean Squared Error loss function
/// Computes MSE = mean((y_pred - y_true)^2)
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in MSE loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
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
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in BCE loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
        ));
    }

    let eps = 1e-8;
    let mut loss_sum = 0.0;
    let total_size = y_pred.data.len() as f32;

    for i in 0..y_pred.data.len() {
        let pred_val = y_pred.data[i].max(eps).min(1.0 - eps);
        let true_val = y_true.data[i];

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
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in MAE loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
        ));
    }

    // Compute |y_pred - y_true|
    let diff = y_pred.sub(y_true)?;
    let abs_diff = Tensor {
        data: diff.data.iter().map(|&x| x.abs()).collect(),
        shape: diff.shape,
        device: diff.device.clone(),
        #[cfg(feature = "gpu")]
        gpu_tensor: None,
    };
    
    // Compute mean
    let mean_value = abs_diff.mean();
    Tensor::new(vec![mean_value], vec![1])
}

/// Huber loss function (robust to outliers)
/// Computes Huber loss = 0.5 * (y_pred - y_true)^2 if |diff| <= delta
///                     else delta * |diff| - 0.5 * delta^2
pub fn huber_loss(y_pred: &Tensor, y_true: &Tensor, delta: f32) -> Result<Tensor, String> {
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in Huber loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
        ));
    }

    if delta <= 0.0 {
        return Err("Delta must be positive".to_string());
    }

    let diff = y_pred.sub(y_true)?;
    let mut loss_sum = 0.0;
    let total_size = diff.data.len() as f32;

    for &diff_val in &diff.data {
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
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in Hinge loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
        ));
    }

    let mut loss_sum = 0.0;
    let total_size = y_pred.data.len() as f32;

    for i in 0..y_pred.data.len() {
        let pred_val = y_pred.data[i];
        let true_val = y_true.data[i];
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
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in KL divergence: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
        ));
    }

    let eps = 1e-8;
    let mut kl_sum = 0.0;

    for i in 0..y_pred.data.len() {
        let pred_val = y_pred.data[i].max(eps);
        let true_val = y_true.data[i].max(eps);
        
        // KL = y_true * log(y_true / y_pred) = y_true * (log(y_true) - log(y_pred))
        kl_sum += true_val * (true_val.ln() - pred_val.ln());
    }

    Tensor::new(vec![kl_sum], vec![1])
}

/// Smooth L1 loss function
/// Computes Smooth L1 = mean(0.5 * x^2 if |x| < 1 else |x| - 0.5)
/// where x = y_pred - y_true
pub fn smooth_l1_loss(y_pred: &Tensor, y_true: &Tensor) -> Result<Tensor, String> {
    if y_pred.shape != y_true.shape {
        return Err(format!(
            "Shape mismatch in Smooth L1 loss: y_pred {:?} vs y_true {:?}",
            y_pred.shape, y_true.shape
        ));
    }

    let diff = y_pred.sub(y_true)?;
    let mut loss_sum = 0.0;
    let total_size = diff.data.len() as f32;

    for &diff_val in &diff.data {
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

    let batch_size = logits.shape[0];
    let num_classes = logits.shape[1];

    if targets.shape[0] != batch_size {
        return Err("Batch size mismatch in categorical cross entropy loss".to_string());
    }

    if targets.shape[1] != num_classes {
        return Err(format!(
            "categorical_cross_entropy expects one-hot targets [batch, {}], got [batch, {}]. \
            Use cross_entropy for class indices [batch, 1].",
            num_classes, targets.shape[1]
        ));
    }

    let mut loss_sum = 0.0;

    // Process each sample in the batch
    for i in 0..batch_size {
        let logit_start = i * num_classes;
        let logit_end = logit_start + num_classes;
        let target_start = i * num_classes;
        let target_end = target_start + num_classes;

        let logits_row = &logits.data[logit_start..logit_end];
        let targets_row = &targets.data[target_start..target_end];

        // Find max for numerical stability (log-sum-exp trick)
        let max_logit = logits_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log-sum-exp: log(sum(exp(logits - max_logit))) + max_logit
        let mut exp_sum = 0.0;
        for &logit in logits_row {
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
        for (j, &target_val) in targets_row.iter().enumerate() {
            if target_val > 0.0 {
                let logit = logits_row[j];
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

    let batch_size = logits.shape[0];
    let num_classes = logits.shape[1];

    if target_indices.shape[0] != batch_size {
        return Err("Batch size mismatch in cross entropy loss".to_string());
    }

    if target_indices.shape[1] != 1 {
        return Err(format!(
            "cross_entropy expects class indices [batch, 1], got [batch, {}]. \
            Use categorical_cross_entropy for one-hot targets [batch, C].",
            target_indices.shape[1]
        ));
    }

    let mut loss_sum = 0.0;

    // Process each sample in the batch
    for i in 0..batch_size {
        let logit_start = i * num_classes;
        let logit_end = logit_start + num_classes;
        
        let logits_row = &logits.data[logit_start..logit_end];
        let target_class = target_indices.data[i] as usize;

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
        for &logit in logits_row {
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
        let target_logit = logits_row[target_class];
        let log_prob = target_logit - log_sum_exp;
        loss_sum -= log_prob;
    }

    let loss = loss_sum / batch_size as f32;
    Tensor::new(vec![loss], vec![1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let loss = mse_loss(&y_pred, &y_true).unwrap();
        assert_eq!(loss.data[0], 0.0);

        let y_pred = Tensor::new(vec![2.0, 3.0, 4.0], vec![3, 1]).unwrap();
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let loss = mse_loss(&y_pred, &y_true).unwrap();
        // MSE = mean((1^2, 1^2, 1^2)) = mean(1, 1, 1) = 1.0
        assert!((loss.data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_loss() {
        let y_pred = Tensor::new(vec![0.7, 0.3, 0.9], vec![3, 1]).unwrap();
        let y_true = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let loss = binary_cross_entropy_loss(&y_pred, &y_true).unwrap();
        // Loss should be positive
        assert!(loss.data[0] > 0.0);
    }
}

