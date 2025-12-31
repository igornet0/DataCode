// Linear Regression model for ML module

use crate::ml::tensor::Tensor;
use indicatif::ProgressBar;
use std::io::{self, Write};

#[derive(Debug)]
pub struct LinearRegression {
    weights: Tensor,        // Weights [feature_count, 1]
    bias: Tensor,           // Bias [1, 1]
}

impl LinearRegression {
    /// Create a new Linear Regression model
    /// feature_count: number of input features
    pub fn new(feature_count: usize) -> Result<Self, String> {
        if feature_count == 0 {
            return Err("Feature count must be greater than 0".to_string());
        }

        // Initialize weights: small values (simple initialization)
        let weights_data: Vec<f32> = vec![0.01; feature_count];
        let weights = Tensor::new(weights_data, vec![feature_count, 1])?;

        // Initialize bias to zero
        let bias = Tensor::zeros(vec![1, 1]);

        Ok(LinearRegression { weights, bias })
    }

    /// Predict outputs for given features
    /// features: [batch_size, feature_count]
    pub fn predict(&self, features: &Tensor) -> Result<Tensor, String> {
        if features.ndim() != 2 {
            return Err("Features must be 2D tensor [batch_size, feature_count]".to_string());
        }

        if features.shape[1] != self.weights.shape[0] {
            return Err(format!(
                "Feature count mismatch: expected {}, got {}",
                self.weights.shape[0],
                features.shape[1]
            ));
        }

        // Forward pass: y_pred = x @ weights + bias
        // x: [batch_size, feature_count]
        // weights: [feature_count, 1]
        // x @ weights: [batch_size, 1]
        let matmul_result = features.matmul(&self.weights)?; // [batch_size, 1]
        
        // Add bias (broadcast)
        // bias: [1, 1], matmul_result: [batch_size, 1]
        // We need to broadcast bias to [batch_size, 1]
        let bias_broadcast = self.bias.broadcast_to(&matmul_result.shape)?;
        matmul_result.add(&bias_broadcast)
    }

    /// Forward pass: predict outputs for given features
    /// Alias for predict() for consistency with NeuralNetwork
    pub fn forward(&self, features: &Tensor) -> Result<Tensor, String> {
        self.predict(features)
    }

    /// Train the model
    /// x: features [batch_size, feature_count]
    /// y: targets [batch_size, 1]
    /// epochs: number of training epochs
    /// lr: learning rate
    pub fn train(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: usize,
        lr: f32,
    ) -> Result<Vec<f32>, String> {
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err("Features and targets must be 2D tensors".to_string());
        }

        if x.shape[0] != y.shape[0] {
            return Err("Batch size mismatch between features and targets".to_string());
        }

        if y.shape[1] != 1 {
            return Err("Targets must have shape [batch_size, 1]".to_string());
        }

        let mut loss_history = Vec::new();
        let batch_size = x.shape[0] as f32;

        // Create progress bar
        let pb = ProgressBar::new(epochs as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} {bar:40.cyan/blue} {pos}/{len} ({percent}%) [{elapsed_precise}<{eta_precise}]")
                .unwrap()
                .progress_chars("##-"),
        );
        // Enable steady tick to update progress bar even during long operations
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        for epoch in 0..epochs {
            // Forward pass
            let y_pred = self.predict(x)?; // [batch_size, 1]

            // Compute loss: MSE = mean((y_pred - y)^2)
            let diff = y_pred.sub(y)?; // [batch_size, 1]
            let diff_sq = diff.mul(&diff)?; // [batch_size, 1]
            let loss = diff_sq.mean();
            loss_history.push(loss);

            // Compute gradients
            // Loss = mean((y_pred - y)^2)
            // dLoss/dy_pred = 2 * (y_pred - y) / batch_size
            // y_pred = x @ w + b
            // dLoss/dw = x^T @ (2 * (y_pred - y) / batch_size)
            // dLoss/db = sum(2 * (y_pred - y) / batch_size)

            // grad_y_pred = 2 * diff / batch_size
            let grad_scale = 2.0 / batch_size;
            let grad_y_pred_data: Vec<f32> = diff.data.iter().map(|&v| v * grad_scale).collect();
            let grad_y_pred = Tensor::new(grad_y_pred_data, diff.shape.clone())?;

            // grad_w = x^T @ grad_y_pred
            let x_t = x.transpose()?; // [feature_count, batch_size]
            let grad_w = x_t.matmul(&grad_y_pred)?; // [feature_count, 1]

            // grad_b = sum(grad_y_pred) -> scalar, broadcast to [1, 1]
            let grad_b_sum = grad_y_pred.sum();
            let grad_b = Tensor::new(vec![grad_b_sum], vec![1, 1])?;

            // Update weights: w = w - lr * grad_w
            let weights_update_data: Vec<f32> = grad_w.data.iter().map(|&v| v * lr).collect();
            let weights_update = Tensor::new(weights_update_data, grad_w.shape)?;
            self.weights = self.weights.sub(&weights_update)?;

            // Update bias: b = b - lr * grad_b
            let bias_update_data: Vec<f32> = grad_b.data.iter().map(|&v| v * lr).collect();
            let bias_update = Tensor::new(bias_update_data, grad_b.shape)?;
            self.bias = self.bias.sub(&bias_update)?;

            // Update progress bar
            pb.set_message(format!("Epoch {}/{}: Loss: {:.4}", epoch + 1, epochs, loss));
            pb.inc(1);
        }

        // Finish progress bar
        pb.finish_with_message(format!("Training completed: {} epochs", epochs));

        Ok(loss_history)
    }

    /// Evaluate the model (compute MSE)
    /// x: features [batch_size, feature_count]
    /// y: targets [batch_size, 1]
    pub fn evaluate(&self, x: &Tensor, y: &Tensor) -> Result<f32, String> {
        let y_pred = self.predict(x)?;
        
        // Compute MSE
        let diff = y_pred.sub(y)?;
        let diff_sq = diff.mul(&diff)?;
        Ok(diff_sq.mean())
    }

    /// Get current weights
    pub fn get_weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get current bias
    pub fn get_bias(&self) -> &Tensor {
        &self.bias
    }
}

// Neural Network model
use crate::ml::layer::{Sequential, Linear, LayerId};
use crate::ml::optimizer::{SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW, OptimizerType};
use crate::ml::loss::{categorical_cross_entropy_loss, sparse_softmax_cross_entropy_loss, binary_cross_entropy_loss};
use crate::ml::graph::NodeId;
use crate::ml::device::Device;
use crate::ml::scheduler::{LearningRateScheduler, AutoLRScheduler};
use std::io::Read;
use std::collections::HashMap;
use serde_json;

/// Training stage information
#[derive(Debug, Clone)]
pub struct TrainingStage {
    pub epochs: usize,
    pub loss: String,
    pub optimizer_type: String,
    pub optimizer_params: Option<serde_json::Value>, // For storing optimizer parameters (lr, beta1, beta2, etc.)
    pub frozen_layers: Vec<String>, // Names of frozen layers
    pub trainable_params: usize,
    pub frozen_params: usize,
    pub loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub val_loss_history: Option<Vec<f32>>,
    pub val_accuracy_history: Option<Vec<f32>>,
}

/// Training history for train_sh method (with early stopping and LR scheduling)
#[derive(Debug, Clone)]
pub struct TrainingHistorySH {
    pub loss: Vec<f32>,
    pub val_loss: Option<Vec<f32>>,
    pub acc: Vec<f32>,
    pub val_acc: Option<Vec<f32>>,
    pub lr: Vec<f32>,  // Learning rate history
    pub best_metric: f32,
    pub best_epoch: usize,
    pub stopped_epoch: usize,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    sequential: Sequential,
    param_node_ids: Vec<NodeId>,
    // Training metadata (legacy - for backward compatibility)
    training_epochs: Option<usize>,
    training_loss: Option<String>,
    training_optimizer: Option<String>,
    training_loss_history: Option<Vec<f32>>,
    training_accuracy_history: Option<Vec<f32>>,
    validation_loss_history: Option<Vec<f32>>,
    validation_accuracy_history: Option<Vec<f32>>,
    // Training stages history
    training_stages: Vec<TrainingStage>,
}

impl NeuralNetwork {
    /// Create a new Neural Network from a Sequential container
    /// The Sequential must have its parameters initialized
    /// Note: param_node_ids will be collected after first forward pass
    pub fn new(sequential: Sequential) -> Result<Self, String> {
        // param_node_ids will be empty initially and collected after first forward pass
        // This is because parameters are only created in the graph during forward pass
        Ok(NeuralNetwork {
            sequential,
            param_node_ids: Vec::new(),
            // Initialize legacy training metadata
            training_epochs: None,
            training_loss: None,
            training_optimizer: None,
            training_loss_history: None,
            training_accuracy_history: None,
            validation_loss_history: None,
            validation_accuracy_history: None,
            // Initialize training stages
            training_stages: Vec::new(),
        })
    }
    
    /// Update param_node_ids from sequential (call after forward pass)
    fn update_param_node_ids(&mut self) {
        self.param_node_ids = self.sequential.parameters().to_vec();
    }

    /// Forward pass: predict outputs for given inputs
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor, String> {
        // Check if input is 1D and needs batch dimension
        let (input_2d, was_1d) = if x.ndim() == 1 {
            // Reshape 1D tensor to 2D: [features] -> [1, features]
            // Ensure tensor is on CPU to access data
            let x_cpu = x.to_cpu()?;
            let new_shape = vec![1, x_cpu.shape[0]];
            let input_2d = Tensor::new(x_cpu.data.clone(), new_shape)?;
            (input_2d, true)
        } else {
            (x.clone(), false)
        };
        
        // Forward pass with 2D tensor
        let mut output = self.sequential.forward(input_2d)?;
        
        // If input was 1D, remove batch dimension from output
        if was_1d && output.ndim() == 2 && output.shape[0] == 1 {
            // Reshape output from [1, features] -> [features]
            // Ensure output is on CPU to access data
            let output_cpu = output.to_cpu()?;
            let new_shape = vec![output_cpu.shape[1]];
            output = Tensor::new(output_cpu.data.clone(), new_shape)?;
        }
        
        Ok(output)
    }

    /// Compute accuracy metric for sparse targets (class indices [N,1])
    /// Returns accuracy as a float between 0.0 and 1.0
    fn compute_accuracy_sparse(logits: &Tensor, class_indices: &Tensor) -> Result<f32, String> {
        if logits.ndim() != 2 || class_indices.ndim() != 2 {
            return Err("Accuracy computation requires 2D tensors".to_string());
        }

        if logits.shape[0] != class_indices.shape[0] {
            return Err("Batch size mismatch in accuracy computation".to_string());
        }

        if class_indices.shape[1] != 1 {
            return Err(format!(
                "compute_accuracy_sparse expects class indices [batch, 1], got [batch, {}]",
                class_indices.shape[1]
            ));
        }

        // Ensure tensors are on CPU for computation
        let logits_cpu = logits.to_cpu()?;
        let targets_cpu = class_indices.to_cpu()?;

        let batch_size = logits_cpu.shape[0];
        let num_classes = logits_cpu.shape[1];

        let mut correct = 0;

        // For each sample in the batch
        for i in 0..batch_size {
            let logit_start = i * num_classes;
            let logit_end = logit_start + num_classes;

            // Find argmax for logits
            let logits_row = &logits_cpu.data[logit_start..logit_end];
            let predicted_class = logits_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Get true class from class indices
            let true_class = targets_cpu.data[i] as usize;

            if predicted_class == true_class {
                correct += 1;
            }
        }

        Ok(correct as f32 / batch_size as f32)
    }

    /// Compute accuracy metric for categorical targets (one-hot [N,C])
    /// Returns accuracy as a float between 0.0 and 1.0
    fn compute_accuracy_categorical(logits: &Tensor, onehot_targets: &Tensor) -> Result<f32, String> {
        if logits.ndim() != 2 || onehot_targets.ndim() != 2 {
            return Err("Accuracy computation requires 2D tensors".to_string());
        }

        if logits.shape[0] != onehot_targets.shape[0] {
            return Err("Batch size mismatch in accuracy computation".to_string());
        }

        // Ensure tensors are on CPU for computation
        let logits_cpu = logits.to_cpu()?;
        let targets_cpu = onehot_targets.to_cpu()?;

        let batch_size = logits_cpu.shape[0];
        let num_classes = logits_cpu.shape[1];

        if targets_cpu.shape[1] != num_classes {
            return Err(format!(
                "compute_accuracy_categorical expects one-hot targets [batch, {}], got [batch, {}]",
                num_classes, targets_cpu.shape[1]
            ));
        }

        let mut correct = 0;

        // For each sample in the batch
        for i in 0..batch_size {
            let logit_start = i * num_classes;
            let logit_end = logit_start + num_classes;
            let target_start = i * num_classes;
            let target_end = target_start + num_classes;

            // Find argmax for logits
            let logits_row = &logits_cpu.data[logit_start..logit_end];
            let predicted_class = logits_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Find argmax for targets (one-hot encoding)
            let targets_row = &targets_cpu.data[target_start..target_end];
            let true_class = targets_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if predicted_class == true_class {
                correct += 1;
            }
        }

        Ok(correct as f32 / batch_size as f32)
    }

    /// Train the neural network using full autograd
    /// 
    /// # Arguments
    /// * `x` - Features tensor [batch_size, num_features]
    /// * `y` - Targets tensor [batch_size, num_targets] for regression or [batch_size, num_classes] for classification
    /// * `epochs` - Number of training epochs
    /// * `batch_size` - Batch size for training
    /// * `lr` - Learning rate
    /// * `loss_type` - "mse" for regression, "cross_entropy" or "sparse_cross_entropy" for classification
    /// * `x_val` - Optional validation features tensor
    /// * `y_val` - Optional validation targets tensor
    /// * `optimizer` - Optimizer name: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW" (default: "SGD")
    pub fn train(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f32,
        loss_type: &str,
        x_val: Option<&Tensor>,
        y_val: Option<&Tensor>,
        optimizer: Option<&str>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err("Features and targets must be 2D tensors".to_string());
        }

        if x.shape[0] != y.shape[0] {
            return Err("Batch size mismatch between features and targets".to_string());
        }

        // Get device from graph (default to CPU if not set)
        let device = self.sequential.graph_mut().device().clone();
        
        // Move data to GPU once at the start if using GPU
        let x_gpu = if device.is_gpu() {
            x.to_device(&device).map_err(|e| format!("Failed to move x to GPU: {}", e))?
        } else {
            x.clone()
        };
        
        let y_gpu = if device.is_gpu() {
            y.to_device(&device).map_err(|e| format!("Failed to move y to GPU: {}", e))?
        } else {
            y.clone()
        };

        // Prepare validation data if provided
        let (x_val_gpu, y_val_gpu) = if let (Some(x_val), Some(y_val)) = (x_val, y_val) {
            if x_val.ndim() != 2 || y_val.ndim() != 2 {
                return Err("Validation features and targets must be 2D tensors".to_string());
            }
            if x_val.shape[0] != y_val.shape[0] {
                return Err("Batch size mismatch between validation features and targets".to_string());
            }
            let x_val_gpu = if device.is_gpu() {
                x_val.to_device(&device).map_err(|e| format!("Failed to move x_val to GPU: {}", e))?
            } else {
                x_val.clone()
            };
            let y_val_gpu = if device.is_gpu() {
                y_val.to_device(&device).map_err(|e| format!("Failed to move y_val to GPU: {}", e))?
            } else {
                y_val.clone()
            };
            (Some(x_val_gpu), Some(y_val_gpu))
        } else {
            (None, None)
        };

        // Create optimizer based on optimizer name
        let optimizer_name = optimizer.unwrap_or("SGD");
        let mut optimizer = match optimizer_name.to_lowercase().as_str() {
            "sgd" => OptimizerType::SGD(SGD::new(lr)?),
            "momentum" => OptimizerType::Momentum(Momentum::new(lr)?),
            "nag" => OptimizerType::NAG(NAG::new(lr)?),
            "adagrad" => OptimizerType::Adagrad(Adagrad::new(lr)?),
            "rmsprop" => OptimizerType::RMSprop(RMSprop::new(lr)?),
            "adam" => OptimizerType::Adam(Adam::new(lr)?),
            "adamw" => OptimizerType::AdamW(AdamW::new(lr)?),
            _ => return Err(format!("Unknown optimizer: {}. Supported: SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW", optimizer_name)),
        };

        let total_samples = x_gpu.shape[0];
        let num_batches = (total_samples + batch_size - 1) / batch_size;

        // Check if there are any trainable parameters before starting training
        // Note: param_node_ids might be empty before first forward pass, so we'll check after first forward
        // For now, we'll check during training loop

        // Train and ensure progress bar is finished properly
        let train_result = (|| -> Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>), String> {
            let mut loss_history = Vec::new();
            let mut accuracy_history = Vec::new();
            let mut val_loss_history = Vec::new();
            let mut val_accuracy_history = Vec::new();
            for epoch in 0..epochs {
                // Create progress bar for this epoch
                let pb = ProgressBar::new(num_batches as u64);
                pb.set_style(
                    indicatif::ProgressStyle::default_bar()
                        .template("{msg} {bar:40.cyan/blue} {pos}/{len} ({percent}%) [{elapsed_precise}<{eta_precise}]")
                        .unwrap()
                        .progress_chars("##-"),
                );
                // Enable steady tick to update progress bar even during long operations
                pb.enable_steady_tick(std::time::Duration::from_millis(100));
                
                // Process epoch with error handling to ensure progress bar is finished
                let epoch_result = (|| -> Result<(), String> {
                    let mut epoch_loss_sum = 0.0;
                    let mut epoch_accuracy_sum = 0.0;
                    let mut num_batches_processed = 0;

                    // Process data in batches
                    for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * batch_size;
                    let end_idx = (start_idx + batch_size).min(total_samples);
                    let current_batch_size = end_idx - start_idx;

                    // Extract batch (from GPU tensors if using GPU)
                    let num_features = x_gpu.shape[1];
                    let num_targets = y_gpu.shape[1];

                    // For GPU, we'll create batch tensors that point to the same device
                    // For now, extract data and create new tensors on same device
                    let mut x_batch_data = Vec::new();
                    for i in start_idx..end_idx {
                        let row_start = i * num_features;
                        let row_end = row_start + num_features;
                        x_batch_data.extend_from_slice(&x_gpu.data[row_start..row_end]);
                    }
                    let mut x_batch = Tensor::new(x_batch_data, vec![current_batch_size, num_features])?;
                    if device.is_gpu() {
                        x_batch = x_batch.to_device(&device)?;
                    }

                    let mut y_batch_data = Vec::new();
                    for i in start_idx..end_idx {
                        let row_start = i * num_targets;
                        let row_end = row_start + num_targets;
                        y_batch_data.extend_from_slice(&y_gpu.data[row_start..row_end]);
                    }
                    let mut y_batch = Tensor::new(y_batch_data, vec![current_batch_size, num_targets])?;
                    if device.is_gpu() {
                        y_batch = y_batch.to_device(&device)?;
                    }
                    
                    // Validate input shape
                    let x_batch_cpu = x_batch.to_cpu().ok();
                    if let Some(x) = x_batch_cpu {
                        if x.shape.len() != 2 {
                            return Err(format!(
                                "Input x_batch must be 2D tensor, got shape {:?}",
                                x.shape
                            ));
                        }
                        if x.shape[0] != current_batch_size {
                            return Err(format!(
                                "Input x_batch batch size mismatch: expected {}, got {} (shape: {:?})",
                                current_batch_size, x.shape[0], x.shape
                            ));
                        }
                    }
                    
                    // Debug: Print input batch information

                    // Zero gradients
                    self.sequential.zero_grad();

                    // Forward pass through the model
                    let logits = self.forward(&x_batch)?;
                    
                    // Compute accuracy (for classification tasks) - will be computed after loss type is determined
                    
                    
                    // Update param_node_ids after forward pass (when parameters are created in graph)
                    // Always update to ensure we have the correct node IDs for the current graph state
                    // This is especially important after graph cleanup, as node IDs change
                    self.update_param_node_ids();

                    // Check if there are any trainable parameters (only check once per epoch, on first batch)
                    if batch_idx == 0 && epoch == 0 {
                        let trainable_params = self.trainable_parameters();
                        if trainable_params.is_empty() {
                            eprintln!("⚠️  No trainable parameters. Training will have no effect.");
                            eprintln!("   All layers are frozen. Use layer.unfreeze() to enable training.");
                        }
                    }

                    // Get output node ID from sequential
                    let output_node_id = self.sequential.output_node_id()
                        .ok_or("Sequential output node not set after forward pass")?;

                    // Add target as input node in graph
                    let target_node_id = self.sequential.graph_mut().add_input();
                    self.sequential.graph_mut().nodes[target_node_id].value = Some(y_batch.clone());

                    // Compute loss through Graph operations for proper autograd
                    let (loss_node_id, backward_start_node) = match loss_type {
                        "mse" => {
                            // MSE = mean((y_pred - y_true)^2)
                            use crate::ml::graph::OpType;
                            let diff_id = self.sequential.graph_mut().add_op(
                                OpType::Sub,
                                vec![output_node_id, target_node_id]
                            )?;
                            let diff_sq_id = self.sequential.graph_mut().add_op(
                                OpType::Mul,
                                vec![diff_id, diff_id]
                            )?;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::Mean,
                                vec![diff_sq_id]
                            )?;
                            
                            // CRITICAL: Compute intermediate values and set them on graph nodes
                            // This is required for backward pass, which needs values of all intermediate nodes
                            // Use the logits tensor we already have and y_batch directly
                            let logits_cpu = logits.to_cpu()?;
                            let y_batch_cpu = y_batch.to_cpu()?;
                            let diff = logits_cpu.sub(&y_batch_cpu)?;
                            let diff_sq = diff.mul(&diff)?;
                            let loss_value = diff_sq.mean();
                            
                            // Move intermediate values to same device as logits if needed
                            let diff_final = if logits.device() != &Device::Cpu {
                                diff.to_device(logits.device())?
                            } else {
                                diff
                            };
                            let diff_sq_final = if logits.device() != &Device::Cpu {
                                diff_sq.to_device(logits.device())?
                            } else {
                                diff_sq
                            };
                            let loss_tensor = Tensor::new(vec![loss_value], vec![1])?;
                            let loss_tensor_final = if logits.device() != &Device::Cpu {
                                loss_tensor.to_device(logits.device())?
                            } else {
                                loss_tensor
                            };
                            
                            // Set values on all intermediate nodes (required for backward pass)
                            self.sequential.graph_mut().nodes[diff_id].value = Some(diff_final);
                            self.sequential.graph_mut().nodes[diff_sq_id].value = Some(diff_sq_final);
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_tensor_final);
                            
                            (loss_node, loss_node)
                        }
                        "cross_entropy" => {
                            // Validate targets shape [batch, 1] for class indices
                            if y_batch.shape[1] != 1 {
                                return Err(format!(
                                    "cross_entropy received targets with shape {:?}, expected [batch, 1] (class indices). \
                                    Use categorical_cross_entropy for one-hot targets [batch, C].",
                                    y_batch.shape
                                ));
                            }
                            
                            // CrossEntropy loss (sparse): add as operation in graph
                            use crate::ml::graph::OpType;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::CrossEntropy,
                                vec![output_node_id, target_node_id]
                            )?;
                            
                            // Compute loss value by running forward pass on the graph
                            // The graph will compute it automatically
                            let logits_value = self.sequential.graph().get_output(output_node_id)?;
                            let targets_value = self.sequential.graph().get_output(target_node_id)?;
                            let logits_cpu = logits_value.to_cpu()?;
                            let targets_cpu = targets_value.to_cpu()?;
                            let loss_value = sparse_softmax_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                            
                            // Move loss to same device as logits if needed
                            let loss_value_final = if logits_value.device() != &Device::Cpu {
                                loss_value.to_device(logits_value.device())?
                            } else {
                                loss_value
                            };
                            
                            // Set the computed value on the loss node
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                            
                            (loss_node, loss_node)
                        }
                        "categorical_cross_entropy" => {
                            // Validate targets shape [batch, C] for one-hot
                            if y_batch.shape[1] != logits.shape[1] {
                                return Err(format!(
                                    "categorical_cross_entropy received targets with shape {:?}, expected [batch, {}] (one-hot). \
                                    Use cross_entropy for class indices [batch, 1].",
                                    y_batch.shape, logits.shape[1]
                                ));
                            }
                            
                            // CategoricalCrossEntropy loss (one-hot): add as operation in graph
                            use crate::ml::graph::OpType;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::CategoricalCrossEntropy,
                                vec![output_node_id, target_node_id]
                            )?;
                            
                            // Compute loss value by running forward pass on the graph
                            let logits_value = self.sequential.graph().get_output(output_node_id)?;
                            let targets_value = self.sequential.graph().get_output(target_node_id)?;
                            let logits_cpu = logits_value.to_cpu()?;
                            let targets_cpu = targets_value.to_cpu()?;
                            let loss_value = categorical_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                            
                            // Move loss to same device as logits if needed
                            let loss_value_final = if logits_value.device() != &Device::Cpu {
                                loss_value.to_device(logits_value.device())?
                            } else {
                                loss_value
                            };
                            
                            // Set the computed value on the loss node
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                            
                            (loss_node, loss_node)
                        }
                        "sparse_cross_entropy" => {
                            // Deprecated: redirect to cross_entropy
                            // Validate targets shape [batch, 1] for class indices
                            if y_batch.shape[1] != 1 {
                                return Err(format!(
                                    "sparse_cross_entropy (deprecated) received targets with shape {:?}, expected [batch, 1] (class indices). \
                                    Use categorical_cross_entropy for one-hot targets [batch, C].",
                                    y_batch.shape
                                ));
                            }
                            
                            // Use CrossEntropy op (same as cross_entropy)
                            use crate::ml::graph::OpType;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::CrossEntropy,
                                vec![output_node_id, target_node_id]
                            )?;
                            
                            let logits_value = self.sequential.graph().get_output(output_node_id)?;
                            let targets_value = self.sequential.graph().get_output(target_node_id)?;
                            let logits_cpu = logits_value.to_cpu()?;
                            let targets_cpu = targets_value.to_cpu()?;
                            let loss_value = sparse_softmax_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                            
                            let loss_value_final = if logits_value.device() != &Device::Cpu {
                                loss_value.to_device(logits_value.device())?
                            } else {
                                loss_value
                            };
                            
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                            
                            (loss_node, loss_node)
                        }
                        "binary_cross_entropy" => {
                            // Binary cross entropy: compute directly (not yet implemented as graph op)
                            let logits_cpu = logits.to_cpu()?;
                            let y_batch_cpu = y_batch.to_cpu()?;
                            let loss_value = binary_cross_entropy_loss(&logits_cpu, &y_batch_cpu)?;
                            
                            // Add loss as input node (constant) for tracking
                            let loss_node = self.sequential.graph_mut().add_input();
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value);
                            
                            (loss_node, loss_node)
                        }
                        _ => {
                            return Err(format!("Unknown loss type: {}. Supported: mse, cross_entropy, categorical_cross_entropy, binary_cross_entropy", loss_type));
                        }
                    };

                    // Compute accuracy based on loss type (after loss node is created)
                    let batch_accuracy = if loss_type == "cross_entropy" || loss_type == "sparse_cross_entropy" {
                        Self::compute_accuracy_sparse(&logits, &y_batch).unwrap_or(0.0)
                    } else if loss_type == "categorical_cross_entropy" {
                        Self::compute_accuracy_categorical(&logits, &y_batch).unwrap_or(0.0)
                    } else {
                        // Not applicable for regression tasks or binary_cross_entropy
                        0.0
                    };

                    // Get loss value for tracking (before backward pass)
                    let loss_value = self.sequential.graph_mut().get_output(loss_node_id)?;
                    
                    // Get current batch loss for display and check
                    // Ensure loss is on CPU for reading
                    let loss_cpu = loss_value.to_cpu()?;
                    let batch_loss = loss_cpu.data[0];
                    
                    // Check for NaN or infinite values - return error without finishing progress bar here
                    // (will be finished in the outer match statement)
                    if batch_loss.is_nan() || batch_loss.is_infinite() {
                        // Additional diagnostics: check logits and targets
                        let logits_str = if logits.shape[0] > 0 && logits.shape[1] > 0 {
                            format!("logits[0]={:?} (shape: {:?})", 
                                &logits.data[0..logits.shape[1].min(10)],
                                logits.shape)
                        } else {
                            "logits empty".to_string()
                        };
                        let targets_str = if y_batch.shape[0] > 0 && y_batch.shape[1] > 0 {
                            format!("targets[0]={:?} (shape: {:?})",
                                &y_batch.data[0..y_batch.shape[1].min(10)],
                                y_batch.shape)
                        } else {
                            "targets empty".to_string()
                        };
                        return Err(format!(
                            "Loss is NaN/Inf at epoch {}, batch {}.\nLoss value: {}\n{}\n{}",
                            epoch + 1, batch_idx + 1, batch_loss, logits_str, targets_str
                        ));
                    }

                    // Check if there are any trainable parameters before backward pass
                    let trainable_params = self.trainable_parameters();
                    let has_trainable_params = !trainable_params.is_empty();
                    
                    if has_trainable_params {
                        // Backward pass from the appropriate node
                        self.sequential.graph_mut().backward(backward_start_node)?;
                        
                        // Check gradients
                        let mut params_with_grads = 0;
                        let mut params_without_grads = 0;
                        for &param_id in &self.param_node_ids {
                            if self.sequential.graph().get_gradient(param_id).is_ok() {
                                params_with_grads += 1;
                            } else {
                                params_without_grads += 1;
                            }
                        }
                        
                        // If no parameters have gradients, something is wrong
                        if params_with_grads == 0 && params_without_grads > 0 {
                            return Err(format!(
                                "No gradients computed for any parameters at epoch {}, batch {}. \
                                This indicates backward pass did not propagate gradients to parameters. \
                                Parameters: {} total, {} with grads, {} without grads. \
                                backward_start_node: {}",
                                epoch + 1, batch_idx + 1, 
                                self.param_node_ids.len(), params_with_grads, params_without_grads,
                                backward_start_node
                            ));
                        }

                        // Check gradient magnitudes before optimizer step
                        let mut total_grad_norm = 0.0;
                        let mut param_count = 0;
                        for &param_id in self.param_node_ids.iter() {
                            if let Ok(grad) = self.sequential.graph().get_gradient(param_id) {
                                let grad_cpu = grad.to_cpu()?;
                                let grad_norm: f32 = grad_cpu.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
                                total_grad_norm += grad_norm;
                                param_count += 1;
                            }
                        }
                        
                        if param_count > 0 {
                            let avg_grad_norm = total_grad_norm / param_count as f32;
                            // If gradients are too small, warn
                            if avg_grad_norm < 1e-6 {
                                eprintln!("WARNING: Very small gradients detected (avg norm: {:.6}). Weights may not update significantly.", avg_grad_norm);
                            }
                        }

                        // Optimizer step - only update trainable parameters
                        optimizer.step(self.sequential.graph_mut(), &trainable_params)?;
                    }
                    // If no trainable parameters, skip backward pass and optimizer step
                    // Loss is still computed and tracked above
                    
                    // Save all parameter values (including frozen ones) to cache for next forward pass
                    // This must be called before clear_non_parameter_nodes() to preserve all parameter values
                    self.sequential.save_parameter_values()?;
                    
                    // Clear non-parameter nodes from graph to prevent memory leak
                    // Only do this if param_node_ids is populated (parameters have been initialized)
                    if !self.param_node_ids.is_empty() {
                        self.sequential.clear_non_parameter_nodes()?;
                        
                        // CRITICAL FIX: Update param_node_ids after graph cleanup, as node IDs change
                        // Sequential updates its internal param_node_ids, but we need to update ours too
                        // The order of parameters is preserved, only node IDs change
                        self.param_node_ids = self.sequential.parameters().to_vec();
                    }
                    
                    epoch_loss_sum += batch_loss;
                    epoch_accuracy_sum += batch_accuracy;
                    num_batches_processed += 1;
                    
                    
                    // Update progress bar during batch processing - show current batch loss and running average
                    let avg_loss = epoch_loss_sum / num_batches_processed as f32;
                    let avg_accuracy = epoch_accuracy_sum / num_batches_processed as f32;
                    let progress_msg = if loss_type == "cross_entropy" || loss_type == "sparse_cross_entropy" || loss_type == "binary_cross_entropy" {
                        format!(
                            "Epoch {}/{} | Batch {}/{} | Loss: {:.4} (avg: {:.4}) | Acc: {:.2}% (avg: {:.2}%)",
                            epoch + 1,
                            epochs,
                            batch_idx + 1,
                            num_batches,
                            batch_loss,
                            avg_loss,
                            batch_accuracy * 100.0,
                            avg_accuracy * 100.0
                        )
                    } else {
                        format!(
                            "Epoch {}/{} | Batch {}/{} | Loss: {:.4} (avg: {:.4})",
                            epoch + 1,
                            epochs,
                            batch_idx + 1,
                            num_batches,
                            batch_loss,
                            avg_loss
                        )
                    };
                    pb.set_message(progress_msg);
                    // Update progress bar position based on per-epoch batch index
                    pb.set_position((batch_idx + 1) as u64);
                    // Also flush stdout to ensure display updates
                    let _ = io::stdout().flush();
                }

                let avg_loss = epoch_loss_sum / num_batches_processed as f32;
                let avg_accuracy = epoch_accuracy_sum / num_batches_processed as f32;
                loss_history.push(avg_loss);
                accuracy_history.push(avg_accuracy);
                
                Ok(())
            })();
            
            // Finish progress bar at end of epoch (even on errors) to force new line for next epoch
            match &epoch_result {
                    Ok(_) => {
                        // Compute validation metrics if validation data is provided (only if epoch succeeded)
                        if let (Some(ref x_val_ref), Some(ref y_val_ref)) = (&x_val_gpu, &y_val_gpu) {
                            // Forward pass on validation data
                            let x_val_gpu_local = x_val_ref.clone();
                            let y_val_gpu_local = y_val_ref.clone();

                            let val_logits = self.forward(&x_val_gpu_local)?;
                            
                            // Compute validation loss
                            let val_logits_cpu = val_logits.to_cpu()?;
                            let y_val_cpu = y_val_gpu_local.to_cpu()?;
                            
                            let val_loss = match loss_type {
                                "mse" => {
                                    use crate::ml::loss::mse_loss;
                                    let loss_tensor = mse_loss(&val_logits_cpu, &y_val_cpu)?;
                                    loss_tensor.data[0]
                                }
                                "cross_entropy" => {
                                    use crate::ml::loss::sparse_softmax_cross_entropy_loss;
                                    let loss_tensor = sparse_softmax_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                    loss_tensor.data[0]
                                }
                                "categorical_cross_entropy" => {
                                    use crate::ml::loss::categorical_cross_entropy_loss;
                                    let loss_tensor = categorical_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                    loss_tensor.data[0]
                                }
                                "sparse_cross_entropy" => {
                                    use crate::ml::loss::sparse_softmax_cross_entropy_loss;
                                    let loss_tensor = sparse_softmax_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                    loss_tensor.data[0]
                                }
                                "binary_cross_entropy" => {
                                    use crate::ml::loss::binary_cross_entropy_loss;
                                    let loss_tensor = binary_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                    loss_tensor.data[0]
                                }
                                _ => {
                                    return Err(format!("Unknown loss type for validation: {}", loss_type));
                                }
                            };
                            
                            // Compute validation accuracy (for classification tasks)
                            let val_accuracy = if loss_type == "cross_entropy" || loss_type == "sparse_cross_entropy" {
                                Self::compute_accuracy_sparse(&val_logits_cpu, &y_val_cpu).unwrap_or(0.0)
                            } else if loss_type == "categorical_cross_entropy" {
                                Self::compute_accuracy_categorical(&val_logits_cpu, &y_val_cpu).unwrap_or(0.0)
                            } else {
                                0.0 // Not applicable for regression tasks
                            };
                            
                            val_loss_history.push(val_loss);
                            val_accuracy_history.push(val_accuracy);
                        }
                        let has_val_data = x_val_gpu.is_some() && y_val_gpu.is_some();
                        let epoch_msg = if loss_type == "cross_entropy" || loss_type == "categorical_cross_entropy" || loss_type == "sparse_cross_entropy" || loss_type == "binary_cross_entropy" {
                            let avg_loss = loss_history.last().copied().unwrap_or(0.0);
                            let avg_accuracy = accuracy_history.last().copied().unwrap_or(0.0);
                            if has_val_data && !val_loss_history.is_empty() {
                                let val_loss = val_loss_history.last().copied().unwrap_or(0.0);
                                let val_acc = val_accuracy_history.last().copied().unwrap_or(0.0);
                                format!("Epoch {}/{}: Loss: {:.4}, Acc: {:.2}%, Val Loss: {:.4}, Val Acc: {:.2}%", 
                                    epoch + 1, epochs, avg_loss, avg_accuracy * 100.0, val_loss, val_acc * 100.0)
                            } else {
                                format!("Epoch {}/{}: Loss: {:.4}, Acc: {:.2}%", epoch + 1, epochs, avg_loss, avg_accuracy * 100.0)
                            }
                        } else {
                            let avg_loss = loss_history.last().copied().unwrap_or(0.0);
                            if has_val_data && !val_loss_history.is_empty() {
                                let val_loss = val_loss_history.last().copied().unwrap_or(0.0);
                                format!("Epoch {}/{}: Loss: {:.4}, Val Loss: {:.4}", epoch + 1, epochs, avg_loss, val_loss)
                            } else {
                                format!("Epoch {}/{}: Loss: {:.4}", epoch + 1, epochs, avg_loss)
                            }
                        };
                        pb.finish_with_message(epoch_msg);
                    }
                    Err(e) => {
                        pb.finish_with_message(format!("Epoch {}/{} failed: {}", epoch + 1, epochs, e));
                        return Err(e.clone());
                    }
                }
                let _ = io::stdout().flush();
            }

            Ok((loss_history, accuracy_history, 
                if val_loss_history.is_empty() { None } else { Some(val_loss_history) },
                if val_accuracy_history.is_empty() { None } else { Some(val_accuracy_history) }))
        })();
        
        // After training completes successfully, save training stage
        if let Ok((ref loss_history, ref accuracy_history, ref val_loss_history, ref val_accuracy_history)) = train_result {
            // Get frozen layers and parameter counts
            let frozen_layers = self.get_frozen_layers();
            let (trainable_params, frozen_params) = self.count_trainable_frozen_params();
            
            // Serialize optimizer parameters for comparison and storage
            let optimizer_params_json = Self::serialize_optimizer_params(&optimizer);
            
            // Check if we need to show info (compare with last stage)
            let should_show_info = if let Some(last_stage) = self.last_stage() {
                let loss_changed = loss_type != last_stage.loss;
                let optimizer_type_changed = optimizer_name.to_lowercase() != last_stage.optimizer_type.to_lowercase();
                
                // Compare optimizer parameters
                let optimizer_params_changed = match &last_stage.optimizer_params {
                    Some(ref last_params) => {
                        // Compare JSON values
                        last_params != &optimizer_params_json
                    },
                    None => true, // If last stage had no params but current does
                };
                
                let frozen_layers_changed = frozen_layers != last_stage.frozen_layers;
                
                loss_changed || optimizer_type_changed || optimizer_params_changed || frozen_layers_changed
            } else {
                false
            };
            
            // Create new training stage
            let stage = TrainingStage {
                epochs,
                loss: loss_type.to_string(),
                optimizer_type: optimizer_name.to_string(),
                optimizer_params: Some(optimizer_params_json),
                frozen_layers: frozen_layers.clone(),
                trainable_params,
                frozen_params,
                loss_history: loss_history.clone(),
                accuracy_history: accuracy_history.clone(),
                val_loss_history: val_loss_history.clone(),
                val_accuracy_history: val_accuracy_history.clone(),
            };
            
            // Add stage to history
            self.training_stages.push(stage);
            
            // Update legacy fields for backward compatibility
            self.training_epochs = Some(self.total_epochs());
            self.training_loss = Some(loss_type.to_string());
            self.training_optimizer = Some(optimizer_name.to_string());
            
            // Merge histories (append to existing)
            if let Some(ref mut existing_loss) = self.training_loss_history {
                existing_loss.extend_from_slice(loss_history);
            } else {
                self.training_loss_history = Some(loss_history.clone());
            }
            
            if let Some(ref mut existing_acc) = self.training_accuracy_history {
                existing_acc.extend_from_slice(accuracy_history);
            } else {
                self.training_accuracy_history = Some(accuracy_history.clone());
            }
            
            if let Some(ref val_loss) = val_loss_history {
                if let Some(ref mut existing_val_loss) = self.validation_loss_history {
                    existing_val_loss.extend_from_slice(val_loss);
                } else {
                    self.validation_loss_history = Some(val_loss.clone());
                }
            }
            
            if let Some(ref val_acc) = val_accuracy_history {
                if let Some(ref mut existing_val_acc) = self.validation_accuracy_history {
                    existing_val_acc.extend_from_slice(val_acc);
                } else {
                    self.validation_accuracy_history = Some(val_acc.clone());
                }
            }
            
            // Show training info if there are differences
            if should_show_info {
                self.print_training_info();
            }
        }
        
        // Extract only loss_history and accuracy_history for backward compatibility
        match train_result {
            Ok((loss_history, accuracy_history, _, _)) => Ok((loss_history, accuracy_history)),
            Err(e) => Err(e),
        }
    }

    /// Train the neural network with early stopping and learning rate scheduling
    /// 
    /// # Arguments
    /// * `x` - Features tensor [batch_size, num_features]
    /// * `y` - Targets tensor [batch_size, num_targets] for regression or [batch_size, num_classes] for classification
    /// * `epochs` - Maximum number of training epochs
    /// * `batch_size` - Batch size for training
    /// * `learning_rate` - Initial learning rate (will be adjusted by scheduler)
    /// * `loss_type` - "mse" for regression, "cross_entropy" or "sparse_cross_entropy" for classification
    /// * `optimizer` - Optimizer name: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW" (default: "SGD")
    /// * `monitor` - Metric to monitor: "loss", "val_loss", "acc", "val_acc"
    /// * `patience` - Number of epochs to wait before reducing LR or stopping
    /// * `min_delta` - Minimum improvement percentage required (e.g., 1.0 means 1%)
    /// * `restore_best` - Whether to restore best weights at the end
    /// * `x_val` - Optional validation features tensor (required if monitor starts with "val_")
    /// * `y_val` - Optional validation targets tensor (required if monitor starts with "val_")
    pub fn train_sh(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        loss_type: &str,
        optimizer: Option<&str>,
        monitor: &str,
        patience: usize,
        min_delta: f32,
        restore_best: bool,
        x_val: Option<&Tensor>,
        y_val: Option<&Tensor>,
    ) -> Result<TrainingHistorySH, String> {
        // Validate inputs
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err("Features and targets must be 2D tensors".to_string());
        }

        if x.shape[0] != y.shape[0] {
            return Err("Batch size mismatch between features and targets".to_string());
        }

        // Validate monitor requires validation data
        if (monitor == "val_loss" || monitor == "val_acc") && (x_val.is_none() || y_val.is_none()) {
            return Err(format!(
                "Monitor '{}' requires validation data, but x_val or y_val is missing",
                monitor
            ));
        }

        if patience == 0 {
            return Err("Patience must be greater than 0".to_string());
        }

        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }

        // Get device from graph
        let device = self.sequential.graph_mut().device().clone();
        
        // Move data to GPU once at the start if using GPU
        let x_gpu = if device.is_gpu() {
            x.to_device(&device).map_err(|e| format!("Failed to move x to GPU: {}", e))?
        } else {
            x.clone()
        };
        
        let y_gpu = if device.is_gpu() {
            y.to_device(&device).map_err(|e| format!("Failed to move y to GPU: {}", e))?
        } else {
            y.clone()
        };

        // Prepare validation data if provided
        let (x_val_gpu, y_val_gpu) = if let (Some(x_val), Some(y_val)) = (x_val, y_val) {
            if x_val.ndim() != 2 || y_val.ndim() != 2 {
                return Err("Validation features and targets must be 2D tensors".to_string());
            }
            if x_val.shape[0] != y_val.shape[0] {
                return Err("Batch size mismatch between validation features and targets".to_string());
            }
            let x_val_gpu = if device.is_gpu() {
                x_val.to_device(&device).map_err(|e| format!("Failed to move x_val to GPU: {}", e))?
            } else {
                x_val.clone()
            };
            let y_val_gpu = if device.is_gpu() {
                y_val.to_device(&device).map_err(|e| format!("Failed to move y_val to GPU: {}", e))?
            } else {
                y_val.clone()
            };
            (Some(x_val_gpu), Some(y_val_gpu))
        } else {
            (None, None)
        };

        // Create optimizer
        let optimizer_name = optimizer.unwrap_or("SGD");
        let mut optimizer = match optimizer_name.to_lowercase().as_str() {
            "sgd" => OptimizerType::SGD(SGD::new(learning_rate)?),
            "momentum" => OptimizerType::Momentum(Momentum::new(learning_rate)?),
            "nag" => OptimizerType::NAG(NAG::new(learning_rate)?),
            "adagrad" => OptimizerType::Adagrad(Adagrad::new(learning_rate)?),
            "rmsprop" => OptimizerType::RMSprop(RMSprop::new(learning_rate)?),
            "adam" => OptimizerType::Adam(Adam::new(learning_rate)?),
            "adamw" => OptimizerType::AdamW(AdamW::new(learning_rate)?),
            _ => return Err(format!("Unknown optimizer: {}. Supported: SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW", optimizer_name)),
        };

        // Determine if monitor is a loss metric (lower is better) or accuracy metric (higher is better)
        let is_loss_metric = monitor == "loss" || monitor == "val_loss";
        
        // Create scheduler with metric type information
        let mut scheduler = AutoLRScheduler::new(learning_rate, epochs, patience, is_loss_metric)?;

        // Initialize best metric for early stopping (separate from scheduler's internal tracking)
        let mut best_metric = if is_loss_metric {
            f32::INFINITY
        } else {
            0.0
        };

        // Save initial parameter values for restoration
        let mut best_weights: HashMap<NodeId, Tensor> = HashMap::new();
        if self.param_node_ids.is_empty() {
            // Need to do a forward pass to initialize parameters
            let _ = self.forward(&x_gpu)?;
            self.update_param_node_ids();
        }
        for &param_id in &self.param_node_ids {
            if let Some(param_value) = self.sequential.get_parameter_value(param_id) {
                best_weights.insert(param_id, param_value);
            }
        }

        let total_samples = x_gpu.shape[0];
        let num_batches = (total_samples + batch_size - 1) / batch_size;

        // Training history
        let mut loss_history = Vec::new();
        let mut accuracy_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut val_accuracy_history = Vec::new();
        let mut lr_history = Vec::new();

        let mut best_epoch = 0;
        let mut wait = 0;
        let mut stopped_epoch = epochs;
        let mut previous_metric = if is_loss_metric { f32::INFINITY } else { 0.0 };

        // Training loop
        for epoch in 0..epochs {
            // Update LR at the start of epoch based on previous epoch's metric
            // For first epoch, use initial metric value
            let current_lr = scheduler.step(epoch, previous_metric);
            optimizer.set_learning_rate(current_lr);
            lr_history.push(current_lr);

            // Create progress bar for this epoch
            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{msg} {bar:40.cyan/blue} {pos}/{len} ({percent}%) [{elapsed_precise}<{eta_precise}]")
                    .unwrap()
                    .progress_chars("##-"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            
            let epoch_result = (|| -> Result<(), String> {
                let mut epoch_loss_sum = 0.0;
                let mut epoch_accuracy_sum = 0.0;
                let mut num_batches_processed = 0;

                // Process data in batches
                for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * batch_size;
                    let end_idx = (start_idx + batch_size).min(total_samples);
                    let current_batch_size = end_idx - start_idx;

                    // Extract batch
                    let num_features = x_gpu.shape[1];
                    let num_targets = y_gpu.shape[1];

                    let mut x_batch_data = Vec::new();
                    for i in start_idx..end_idx {
                        let row_start = i * num_features;
                        let row_end = row_start + num_features;
                        x_batch_data.extend_from_slice(&x_gpu.data[row_start..row_end]);
                    }
                    let mut x_batch = Tensor::new(x_batch_data, vec![current_batch_size, num_features])?;
                    if device.is_gpu() {
                        x_batch = x_batch.to_device(&device)?;
                    }

                    let mut y_batch_data = Vec::new();
                    for i in start_idx..end_idx {
                        let row_start = i * num_targets;
                        let row_end = row_start + num_targets;
                        y_batch_data.extend_from_slice(&y_gpu.data[row_start..row_end]);
                    }
                    let mut y_batch = Tensor::new(y_batch_data, vec![current_batch_size, num_targets])?;
                    if device.is_gpu() {
                        y_batch = y_batch.to_device(&device)?;
                    }
                    
                    // Zero gradients
                    self.sequential.zero_grad();

                    // Forward pass
                    let logits = self.forward(&x_batch)?;
                    
                    // Update param_node_ids after first forward pass
                    if self.param_node_ids.is_empty() {
                        self.update_param_node_ids();
                    }

                    // Get output node ID
                    let output_node_id = self.sequential.output_node_id()
                        .ok_or("Sequential output node not set after forward pass")?;

                    // Add target as input node
                    let target_node_id = self.sequential.graph_mut().add_input();
                    self.sequential.graph_mut().nodes[target_node_id].value = Some(y_batch.clone());

                    // Compute loss
                    let (loss_node_id, backward_start_node) = match loss_type {
                        "mse" => {
                            use crate::ml::graph::OpType;
                            let diff_id = self.sequential.graph_mut().add_op(
                                OpType::Sub,
                                vec![output_node_id, target_node_id]
                            )?;
                            let diff_sq_id = self.sequential.graph_mut().add_op(
                                OpType::Mul,
                                vec![diff_id, diff_id]
                            )?;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::Mean,
                                vec![diff_sq_id]
                            )?;
                            (loss_node, loss_node)
                        }
                        "cross_entropy" => {
                            if y_batch.shape[1] != 1 {
                                return Err(format!(
                                    "cross_entropy received targets with shape {:?}, expected [batch, 1]",
                                    y_batch.shape
                                ));
                            }
                            use crate::ml::graph::OpType;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::CrossEntropy,
                                vec![output_node_id, target_node_id]
                            )?;
                            let logits_value = self.sequential.graph().get_output(output_node_id)?;
                            let targets_value = self.sequential.graph().get_output(target_node_id)?;
                            let logits_cpu = logits_value.to_cpu()?;
                            let targets_cpu = targets_value.to_cpu()?;
                            let loss_value = sparse_softmax_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                            let loss_value_final = if logits_value.device() != &Device::Cpu {
                                loss_value.to_device(logits_value.device())?
                            } else {
                                loss_value
                            };
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                            (loss_node, loss_node)
                        }
                        "categorical_cross_entropy" => {
                            if y_batch.shape[1] != logits.shape[1] {
                                return Err(format!(
                                    "categorical_cross_entropy received targets with shape {:?}, expected [batch, {}]",
                                    y_batch.shape, logits.shape[1]
                                ));
                            }
                            use crate::ml::graph::OpType;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::CategoricalCrossEntropy,
                                vec![output_node_id, target_node_id]
                            )?;
                            let logits_value = self.sequential.graph().get_output(output_node_id)?;
                            let targets_value = self.sequential.graph().get_output(target_node_id)?;
                            let logits_cpu = logits_value.to_cpu()?;
                            let targets_cpu = targets_value.to_cpu()?;
                            let loss_value = categorical_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                            let loss_value_final = if logits_value.device() != &Device::Cpu {
                                loss_value.to_device(logits_value.device())?
                            } else {
                                loss_value
                            };
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                            (loss_node, loss_node)
                        }
                        "sparse_cross_entropy" => {
                            if y_batch.shape[1] != 1 {
                                return Err(format!(
                                    "sparse_cross_entropy received targets with shape {:?}, expected [batch, 1]",
                                    y_batch.shape
                                ));
                            }
                            use crate::ml::graph::OpType;
                            let loss_node = self.sequential.graph_mut().add_op(
                                OpType::CrossEntropy,
                                vec![output_node_id, target_node_id]
                            )?;
                            let logits_value = self.sequential.graph().get_output(output_node_id)?;
                            let targets_value = self.sequential.graph().get_output(target_node_id)?;
                            let logits_cpu = logits_value.to_cpu()?;
                            let targets_cpu = targets_value.to_cpu()?;
                            let loss_value = sparse_softmax_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                            let loss_value_final = if logits_value.device() != &Device::Cpu {
                                loss_value.to_device(logits_value.device())?
                            } else {
                                loss_value
                            };
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                            (loss_node, loss_node)
                        }
                        "binary_cross_entropy" => {
                            let logits_cpu = logits.to_cpu()?;
                            let y_batch_cpu = y_batch.to_cpu()?;
                            let loss_value = binary_cross_entropy_loss(&logits_cpu, &y_batch_cpu)?;
                            let loss_node = self.sequential.graph_mut().add_input();
                            self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value);
                            (loss_node, loss_node)
                        }
                        _ => {
                            return Err(format!("Unknown loss type: {}. Supported: mse, cross_entropy, categorical_cross_entropy, binary_cross_entropy", loss_type));
                        }
                    };

                    // Compute accuracy
                    let batch_accuracy = if loss_type == "cross_entropy" || loss_type == "sparse_cross_entropy" {
                        Self::compute_accuracy_sparse(&logits, &y_batch).unwrap_or(0.0)
                    } else if loss_type == "categorical_cross_entropy" {
                        Self::compute_accuracy_categorical(&logits, &y_batch).unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    // Backward pass
                    self.sequential.graph_mut().backward(backward_start_node)?;

                    // Get loss value
                    let loss_value = self.sequential.graph_mut().get_output(loss_node_id)?;
                    let loss_cpu = loss_value.to_cpu()?;
                    let batch_loss = loss_cpu.data[0];
                    
                    if batch_loss.is_nan() || batch_loss.is_infinite() {
                        return Err(format!(
                            "Loss is NaN/Inf at epoch {}, batch {}",
                            epoch + 1, batch_idx + 1
                        ));
                    }

                    // Optimizer step
                    let trainable_params = self.trainable_parameters();
                    optimizer.step(self.sequential.graph_mut(), &trainable_params)?;
                    
                    // Save updated parameter values
                    self.sequential.save_parameter_values()?;
                    
                    // Clear non-parameter nodes
                    self.sequential.clear_non_parameter_nodes()?;
                    self.param_node_ids = self.sequential.parameters().to_vec();
                    
                    epoch_loss_sum += batch_loss;
                    epoch_accuracy_sum += batch_accuracy;
                    num_batches_processed += 1;
                    
                    // Update progress bar
                    let avg_loss = epoch_loss_sum / num_batches_processed as f32;
                    let avg_accuracy = epoch_accuracy_sum / num_batches_processed as f32;
                    let progress_msg = if loss_type == "cross_entropy" || loss_type == "sparse_cross_entropy" || loss_type == "binary_cross_entropy" {
                        format!(
                            "Epoch {}/{} | Batch {}/{} | Loss: {:.4} | Acc: {:.2}% | LR: {:.6}",
                            epoch + 1, epochs, batch_idx + 1, num_batches, avg_loss, avg_accuracy * 100.0, current_lr
                        )
                    } else {
                        format!(
                            "Epoch {}/{} | Batch {}/{} | Loss: {:.4} | LR: {:.6}",
                            epoch + 1, epochs, batch_idx + 1, num_batches, avg_loss, current_lr
                        )
                    };
                    pb.set_message(progress_msg);
                    pb.set_position((batch_idx + 1) as u64);
                    let _ = io::stdout().flush();
                }

                let avg_loss = epoch_loss_sum / num_batches_processed as f32;
                let avg_accuracy = epoch_accuracy_sum / num_batches_processed as f32;
                loss_history.push(avg_loss);
                accuracy_history.push(avg_accuracy);
                
                Ok(())
            })();
            
            match &epoch_result {
                Ok(_) => {
                    // Compute validation metrics if validation data is provided
                    if let (Some(ref x_val_ref), Some(ref y_val_ref)) = (&x_val_gpu, &y_val_gpu) {
                        let x_val_gpu_local = x_val_ref.clone();
                        let y_val_gpu_local = y_val_ref.clone();

                        let val_logits = self.forward(&x_val_gpu_local)?;
                        
                        let val_logits_cpu = val_logits.to_cpu()?;
                        let y_val_cpu = y_val_gpu_local.to_cpu()?;
                        
                        let val_loss = match loss_type {
                            "mse" => {
                                use crate::ml::loss::mse_loss;
                                let loss_tensor = mse_loss(&val_logits_cpu, &y_val_cpu)?;
                                loss_tensor.data[0]
                            }
                            "cross_entropy" | "sparse_cross_entropy" => {
                                use crate::ml::loss::sparse_softmax_cross_entropy_loss;
                                let loss_tensor = sparse_softmax_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                loss_tensor.data[0]
                            }
                            "categorical_cross_entropy" => {
                                use crate::ml::loss::categorical_cross_entropy_loss;
                                let loss_tensor = categorical_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                loss_tensor.data[0]
                            }
                            "binary_cross_entropy" => {
                                use crate::ml::loss::binary_cross_entropy_loss;
                                let loss_tensor = binary_cross_entropy_loss(&val_logits_cpu, &y_val_cpu)?;
                                loss_tensor.data[0]
                            }
                            _ => {
                                return Err(format!("Unknown loss type for validation: {}", loss_type));
                            }
                        };
                        
                        let val_accuracy = if loss_type == "cross_entropy" || loss_type == "sparse_cross_entropy" {
                            Self::compute_accuracy_sparse(&val_logits_cpu, &y_val_cpu).unwrap_or(0.0)
                        } else if loss_type == "categorical_cross_entropy" {
                            Self::compute_accuracy_categorical(&val_logits_cpu, &y_val_cpu).unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        
                        val_loss_history.push(val_loss);
                        val_accuracy_history.push(val_accuracy);
                    }

                    // Get current metric value based on monitor
                    let current_metric = match monitor {
                        "loss" => *loss_history.last().unwrap(),
                        "val_loss" => *val_loss_history.last().unwrap_or(&0.0),
                        "acc" => *accuracy_history.last().unwrap(),
                        "val_acc" => *val_accuracy_history.last().unwrap_or(&0.0),
                        _ => return Err(format!("Unknown monitor: {}. Supported: loss, val_loss, acc, val_acc", monitor)),
                    };
                    
                    // Store metric for next epoch's scheduler step
                    previous_metric = current_metric;

                    // Check for improvement for early stopping (separate from scheduler's tracking)
                    let improved = if is_loss_metric {
                        // For loss metrics: lower is better
                        if current_metric < best_metric {
                            let improvement = ((best_metric - current_metric) / best_metric.abs()) * 100.0;
                            improvement >= min_delta
                        } else {
                            false
                        }
                    } else {
                        // For accuracy metrics: higher is better
                        if current_metric > best_metric {
                            let improvement = ((current_metric - best_metric) / best_metric.max(1e-10)) * 100.0;
                            improvement >= min_delta
                        } else {
                            false
                        }
                    };

                    if improved {
                        best_metric = current_metric;
                        best_epoch = epoch;
                        wait = 0;
                        // Save best weights
                        best_weights.clear();
                        for &param_id in &self.param_node_ids {
                            if let Some(param_value) = self.sequential.get_parameter_value(param_id) {
                                best_weights.insert(param_id, param_value);
                            }
                        }
                    } else {
                        wait += 1;
                    }

                    // Check early stopping (AFTER scheduler step and metric check)
                    if wait >= patience {
                        stopped_epoch = epoch + 1;
                        pb.finish_with_message(format!("Early stopping at epoch {}", epoch + 1));
                        break;
                    }

                    // Finish progress bar
                    let has_val_data = x_val_gpu.is_some() && y_val_gpu.is_some();
                    let epoch_msg = if loss_type == "cross_entropy" || loss_type == "categorical_cross_entropy" || loss_type == "sparse_cross_entropy" || loss_type == "binary_cross_entropy" {
                        let avg_loss = loss_history.last().copied().unwrap_or(0.0);
                        let avg_accuracy = accuracy_history.last().copied().unwrap_or(0.0);
                        if has_val_data && !val_loss_history.is_empty() {
                            let val_loss = val_loss_history.last().copied().unwrap_or(0.0);
                            let val_acc = val_accuracy_history.last().copied().unwrap_or(0.0);
                            format!("Epoch {}/{}: Loss: {:.4}, Acc: {:.2}%, Val Loss: {:.4}, Val Acc: {:.2}%, LR: {:.6}", 
                                epoch + 1, epochs, avg_loss, avg_accuracy * 100.0, val_loss, val_acc * 100.0, current_lr)
                        } else {
                            format!("Epoch {}/{}: Loss: {:.4}, Acc: {:.2}%, LR: {:.6}", epoch + 1, epochs, avg_loss, avg_accuracy * 100.0, current_lr)
                        }
                    } else {
                        let avg_loss = loss_history.last().copied().unwrap_or(0.0);
                        if has_val_data && !val_loss_history.is_empty() {
                            let val_loss = val_loss_history.last().copied().unwrap_or(0.0);
                            format!("Epoch {}/{}: Loss: {:.4}, Val Loss: {:.4}, LR: {:.6}", epoch + 1, epochs, avg_loss, val_loss, current_lr)
                        } else {
                            format!("Epoch {}/{}: Loss: {:.4}, LR: {:.6}", epoch + 1, epochs, avg_loss, current_lr)
                        }
                    };
                    pb.finish_with_message(epoch_msg);
                }
                Err(e) => {
                    pb.finish_with_message(format!("Epoch {}/{} failed: {}", epoch + 1, epochs, e));
                    return Err(e.clone());
                }
            }
            let _ = io::stdout().flush();
        }

        // Restore best weights if requested
        if restore_best && !best_weights.is_empty() {
            for (param_id, saved_value) in &best_weights {
                if let Some(node) = self.sequential.graph_mut().nodes.get_mut(*param_id) {
                    node.value = Some(saved_value.clone());
                }
            }
            // Update cache
            self.sequential.save_parameter_values()?;
        }

        // Update model metadata after training completes successfully
        // Get frozen layers and parameter counts
        let frozen_layers = self.get_frozen_layers();
        let (trainable_params_count, frozen_params_count) = self.count_trainable_frozen_params();
        
        // Serialize optimizer parameters for comparison and storage
        let optimizer_params_json = Self::serialize_optimizer_params(&optimizer);
        
        // Create new training stage with stopped_epoch (actual epochs trained)
        let stage = TrainingStage {
            epochs: stopped_epoch,
            loss: loss_type.to_string(),
            optimizer_type: optimizer_name.to_string(),
            optimizer_params: Some(optimizer_params_json),
            frozen_layers: frozen_layers.clone(),
            trainable_params: trainable_params_count,
            frozen_params: frozen_params_count,
            loss_history: loss_history.clone(),
            accuracy_history: accuracy_history.clone(),
            val_loss_history: if val_loss_history.is_empty() { None } else { Some(val_loss_history.clone()) },
            val_accuracy_history: if val_accuracy_history.is_empty() { None } else { Some(val_accuracy_history.clone()) },
        };
        
        // Add stage to history
        self.training_stages.push(stage);
        
        // Update legacy fields for backward compatibility
        self.training_epochs = Some(self.total_epochs());
        self.training_loss = Some(loss_type.to_string());
        self.training_optimizer = Some(optimizer_name.to_string());
        
        // Merge histories (append to existing)
        if let Some(ref mut existing_loss) = self.training_loss_history {
            existing_loss.extend_from_slice(&loss_history);
        } else {
            self.training_loss_history = Some(loss_history.clone());
        }
        
        if let Some(ref mut existing_acc) = self.training_accuracy_history {
            existing_acc.extend_from_slice(&accuracy_history);
        } else {
            self.training_accuracy_history = Some(accuracy_history.clone());
        }
        
        if !val_loss_history.is_empty() {
            if let Some(ref mut existing_val_loss) = self.validation_loss_history {
                existing_val_loss.extend_from_slice(&val_loss_history);
            } else {
                self.validation_loss_history = Some(val_loss_history.clone());
            }
        }
        
        if !val_accuracy_history.is_empty() {
            if let Some(ref mut existing_val_acc) = self.validation_accuracy_history {
                existing_val_acc.extend_from_slice(&val_accuracy_history);
            } else {
                self.validation_accuracy_history = Some(val_accuracy_history.clone());
            }
        }

        // Build return value
        Ok(TrainingHistorySH {
            loss: loss_history,
            val_loss: if val_loss_history.is_empty() { None } else { Some(val_loss_history) },
            acc: accuracy_history,
            val_acc: if val_accuracy_history.is_empty() { None } else { Some(val_accuracy_history) },
            lr: lr_history,
            best_metric,
            best_epoch,
            stopped_epoch,
        })
    }

    /// Predict outputs for given inputs (convenience method)
    pub fn predict(&mut self, x: &Tensor) -> Result<Tensor, String> {
        self.forward(x)
    }

    /// Get parameter node IDs (for optimizer)
    pub fn parameters(&self) -> &[NodeId] {
        &self.param_node_ids
    }

    /// Get trainable parameter node IDs (excluding frozen layers)
    /// Returns a filtered list of parameter node IDs from trainable layers only
    pub fn trainable_parameters(&self) -> Vec<NodeId> {
        use crate::ml::layer::with_layer;
        
        let mut trainable_params = Vec::new();
        let layer_ids = self.layers();
        let mut param_idx = 0;
        let graph = self.sequential.graph();
        
        // Iterate through layers and collect parameters only from trainable layers
        for &layer_id in layer_ids {
            if let Some(is_trainable) = with_layer(layer_id, |layer| {
                let params = layer.parameters();
                // Linear layers have 2 parameters (weight and bias)
                if params.len() == 2 {
                    Some(layer.is_trainable())
                } else {
                    None  // Not a Linear layer or no parameters
                }
            }) {
                if let Some(trainable) = is_trainable {
                    if trainable && param_idx + 1 < self.param_node_ids.len() {
                        // Validate that node IDs exist in the current graph before adding them
                        let weight_id = self.param_node_ids[param_idx];
                        let bias_id = self.param_node_ids[param_idx + 1];
                        if weight_id < graph.nodes.len() && bias_id < graph.nodes.len() {
                            // Add both weight and bias parameters
                            trainable_params.push(weight_id);
                            trainable_params.push(bias_id);
                        }
                    }
                    param_idx += 2;
                }
            }
        }
        
        trainable_params
    }

    /// Get the sequential model (mutable)
    pub fn sequential_mut(&mut self) -> &mut Sequential {
        &mut self.sequential
    }
    
    /// Get the sequential model (immutable)
    pub fn sequential(&self) -> &Sequential {
        &self.sequential
    }

    /// Get layer by index
    /// Returns LayerId if index is valid, None otherwise
    pub fn get_layer(&self, index: usize) -> Option<LayerId> {
        self.layers().get(index).copied()
    }

    /// Get all layer IDs
    /// Returns a slice of all layer IDs in the neural network
    pub fn layers(&self) -> &[LayerId] {
        self.sequential.layer_ids()
    }
    
    /// Set device for this neural network
    pub fn set_device(&mut self, device: crate::ml::device::Device) {
        self.sequential.set_device(device);
    }
    
    /// Get device for this neural network
    pub fn get_device(&self) -> crate::ml::device::Device {
        self.sequential.graph().device().clone()
    }

    /// Get total epochs across all training stages
    pub fn total_epochs(&self) -> usize {
        self.training_stages.iter().map(|stage| stage.epochs).sum()
    }

    /// Get all training stages
    pub fn training_stages(&self) -> &[TrainingStage] {
        &self.training_stages
    }

    /// Get the last training stage
    pub fn last_stage(&self) -> Option<&TrainingStage> {
        self.training_stages.last()
    }

    /// Get training epochs (legacy method)
    pub fn training_epochs(&self) -> Option<usize> {
        self.training_epochs
    }

    /// Get training loss (legacy method)
    pub fn training_loss(&self) -> Option<&str> {
        self.training_loss.as_deref()
    }

    /// Get training optimizer (legacy method)
    pub fn training_optimizer(&self) -> Option<&str> {
        self.training_optimizer.as_deref()
    }

    /// Get combined training history from all stages (legacy method)
    pub fn training_history(&self) -> Option<(&[f32], &[f32])> {
        if let (Some(ref loss), Some(ref acc)) = (&self.training_loss_history, &self.training_accuracy_history) {
            Some((loss, acc))
        } else {
            None
        }
    }

    /// Get combined validation history from all stages (legacy method)
    pub fn validation_history(&self) -> Option<(&[f32], &[f32])> {
        if let (Some(ref loss), Some(ref acc)) = (&self.validation_loss_history, &self.validation_accuracy_history) {
            Some((loss, acc))
        } else {
            None
        }
    }

    /// Get frozen layers (layers that are not trainable)
    pub fn get_frozen_layers(&self) -> Vec<String> {
        use crate::ml::layer::with_layer;
        
        let mut frozen_layers = Vec::new();
        let layer_ids = self.layers();
        
        for (idx, &layer_id) in layer_ids.iter().enumerate() {
            if let Some(is_trainable) = with_layer(layer_id, |layer| {
                // Check if this is a Linear layer by checking if it has in_features and out_features > 0
                // Linear layers have both > 0, activation layers have 0
                let in_feat = layer.in_features();
                let out_feat = layer.out_features();
                if in_feat > 0 && out_feat > 0 {
                    // This is a Linear layer - check trainable status
                    Some(layer.is_trainable())
                } else {
                    None  // Not a Linear layer, skip
                }
            }) {
                if let Some(trainable) = is_trainable {
                    if !trainable {
                        frozen_layers.push(format!("layer{}", idx));
                    }
                }
            }
        }
        
        frozen_layers
    }

    /// Serialize optimizer parameters to JSON
    fn serialize_optimizer_params(optimizer: &OptimizerType) -> serde_json::Value {
        match optimizer {
            OptimizerType::SGD(sgd) => serde_json::json!({"lr": sgd.learning_rate}),
            OptimizerType::Momentum(mom) => serde_json::json!({
                "lr": mom.learning_rate,
                "beta": mom.beta
            }),
            OptimizerType::NAG(nag) => serde_json::json!({
                "lr": nag.learning_rate,
                "beta": nag.beta
            }),
            OptimizerType::Adagrad(adagrad) => serde_json::json!({
                "lr": adagrad.learning_rate,
                "epsilon": adagrad.epsilon
            }),
            OptimizerType::RMSprop(rmsprop) => serde_json::json!({
                "lr": rmsprop.learning_rate,
                "gamma": rmsprop.gamma,
                "epsilon": rmsprop.epsilon
            }),
            OptimizerType::Adam(adam) => serde_json::json!({
                "lr": adam.learning_rate,
                "beta1": adam.beta1,
                "beta2": adam.beta2,
                "epsilon": adam.epsilon
            }),
            OptimizerType::AdamW(adamw) => serde_json::json!({
                "lr": adamw.learning_rate,
                "beta1": adamw.beta1,
                "beta2": adamw.beta2,
                "epsilon": adamw.epsilon,
                "weight_decay": adamw.weight_decay
            }),
        }
    }


    /// Count trainable and frozen parameters
    pub fn count_trainable_frozen_params(&self) -> (usize, usize) {
        use crate::ml::layer::with_layer;
        
        let mut trainable_params = 0;
        let mut frozen_params = 0;
        let layer_ids = self.layers();
        let mut param_idx = 0;
        
        for &layer_id in layer_ids {
            if let Some(result) = with_layer(layer_id, |layer| {
                let params = layer.parameters();
                let in_features = layer.in_features();
                let out_features = layer.out_features();
                
                // Linear layers have 2 parameters (weight and bias)
                if params.len() == 2 {
                    // Parameters are initialized - calculate from tensor shapes
                    let weight_params = if let Some((_, weight_tensor)) = params.get(0) {
                        weight_tensor.shape.iter().product::<usize>()
                    } else {
                        0
                    };
                    let bias_params = if let Some((_, bias_tensor)) = params.get(1) {
                        bias_tensor.shape.iter().product::<usize>()
                    } else {
                        0
                    };
                    let total_params = weight_params + bias_params;
                    Some((layer.is_trainable(), total_params))
                } else if in_features > 0 && out_features > 0 {
                    // Linear layer but parameters not yet initialized (before forward pass)
                    // Calculate from layer dimensions: weights (in_features * out_features) + bias (out_features)
                    let total_params = in_features * out_features + out_features;
                    Some((layer.is_trainable(), total_params))
                } else {
                    None
                }
            }) {
                if let Some((trainable, params_count)) = result {
                    // If param_node_ids is populated, prefer using graph values for accuracy
                    if param_idx + 1 < self.param_node_ids.len() {
                        let graph = self.sequential.graph();
                        if let (Ok(weight_tensor), Ok(bias_tensor)) = (
                            graph.get_output(self.param_node_ids[param_idx]),
                            graph.get_output(self.param_node_ids[param_idx + 1])
                        ) {
                            let layer_params = weight_tensor.shape.iter().product::<usize>() +
                                             bias_tensor.shape.iter().product::<usize>();
                            
                            if trainable {
                                trainable_params += layer_params;
                            } else {
                                frozen_params += layer_params;
                            }
                        }
                    } else {
                        // Use calculated parameters from layer dimensions
                        if trainable {
                            trainable_params += params_count;
                        } else {
                            frozen_params += params_count;
                        }
                    }
                    param_idx += 2;
                }
            }
        }
        
        (trainable_params, frozen_params)
    }


    /// Print training information with all stages
    fn print_training_info(&self) {
        if self.training_stages.is_empty() {
            return;
        }

        println!("Training:");
        println!("-------------------------------------------------");
        println!("Total epochs trained: {}", self.total_epochs());
        println!("Training stages: {}", self.training_stages.len());

        for (idx, stage) in self.training_stages.iter().enumerate() {
            println!("Stage {}:", idx + 1);
            println!("  Epochs: {}", stage.epochs);
            println!("  Loss: {}", stage.loss);
            
            // Format optimizer string
            let optimizer_str = if let Some(ref params) = stage.optimizer_params {
                if let Some(lr) = params.get("lr").and_then(|v| v.as_f64()) {
                    match stage.optimizer_type.as_str() {
                        "SGD" => format!("SGD(lr={})", lr),
                        "Momentum" => {
                            if let Some(beta) = params.get("beta").and_then(|v| v.as_f64()) {
                                format!("Momentum(lr={}, beta={})", lr, beta)
                            } else {
                                format!("Momentum(lr={})", lr)
                            }
                        },
                        "NAG" => {
                            if let Some(beta) = params.get("beta").and_then(|v| v.as_f64()) {
                                format!("NAG(lr={}, beta={})", lr, beta)
                            } else {
                                format!("NAG(lr={})", lr)
                            }
                        },
                        "Adagrad" => {
                            if let Some(eps) = params.get("epsilon").and_then(|v| v.as_f64()) {
                                format!("Adagrad(lr={}, epsilon={})", lr, eps)
                            } else {
                                format!("Adagrad(lr={})", lr)
                            }
                        },
                        "RMSprop" => {
                            let gamma = params.get("gamma").and_then(|v| v.as_f64()).unwrap_or(0.99);
                            let eps = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);
                            format!("RMSprop(lr={}, gamma={}, epsilon={})", lr, gamma, eps)
                        },
                        "Adam" => {
                            let beta1 = params.get("beta1").and_then(|v| v.as_f64()).unwrap_or(0.9);
                            let beta2 = params.get("beta2").and_then(|v| v.as_f64()).unwrap_or(0.999);
                            let eps = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);
                            format!("Adam(lr={}, beta1={}, beta2={}, epsilon={})", lr, beta1, beta2, eps)
                        },
                        "AdamW" => {
                            let beta1 = params.get("beta1").and_then(|v| v.as_f64()).unwrap_or(0.9);
                            let beta2 = params.get("beta2").and_then(|v| v.as_f64()).unwrap_or(0.999);
                            let eps = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);
                            let wd = params.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.01);
                            format!("AdamW(lr={}, beta1={}, beta2={}, epsilon={}, weight_decay={})", lr, beta1, beta2, eps, wd)
                        },
                        _ => format!("{}(lr={})", stage.optimizer_type, lr),
                    }
                } else {
                    stage.optimizer_type.clone()
                }
            } else {
                stage.optimizer_type.clone()
            };
            println!("  Optimizer: {}", optimizer_str);

            // Print frozen layers
            if stage.frozen_layers.is_empty() {
                println!("  Frozen layers: none");
            } else {
                println!("  Frozen layers:");
                for layer in &stage.frozen_layers {
                    println!("    - {}", layer);
                }
            }

            // Print parameters
            println!("  Parameters:");
            println!("    Trainable: {}", stage.trainable_params);
            println!("    Frozen:    {}", stage.frozen_params);

            // Print history (truncated if too long)
            println!("  Loss history: [{:.4}, ...]", 
                stage.loss_history.first().unwrap_or(&0.0));
            if !stage.accuracy_history.is_empty() {
                println!("  Acc history: [{:.2}%, ..]", 
                    stage.accuracy_history.first().unwrap_or(&0.0) * 100.0);
            }
            if let Some(ref val_loss) = stage.val_loss_history {
                if !val_loss.is_empty() {
                    println!("  Val Loss: [{:.4}, ..]", val_loss.first().unwrap_or(&0.0));
                }
            }
            if let Some(ref val_acc) = stage.val_accuracy_history {
                if !val_acc.is_empty() {
                    println!("  Val Acc: [{:.2}%, ..]", val_acc.first().unwrap_or(&0.0) * 100.0);
                }
            }

            println!();
        }

        // Check if frozen layers changed between stages
        if self.training_stages.len() > 1 {
            let last_frozen = &self.training_stages[self.training_stages.len() - 1].frozen_layers;
            let prev_frozen = &self.training_stages[self.training_stages.len() - 2].frozen_layers;
            if last_frozen != prev_frozen {
                println!("ℹ️ Frozen configuration changed since last training stage");
            }
        }
    }

    /// Save model to binary file
    /// Format: Magic (8 bytes) + Version (4 bytes) + JSON architecture + Binary tensors
    pub fn save(&self, path: &str) -> Result<(), String> {
        // Build architecture JSON
        let mut layers_json = Vec::new();
        let mut tensors = Vec::new(); // (name, tensor)
        
        use crate::ml::layer::with_layer;
        
        let graph = self.sequential.graph();
        
        // Get layer_ids from sequential
        let layer_ids = self.layers().to_vec();
        
        for (layer_idx, &layer_id) in layer_ids.iter().enumerate() {
            // Access layer through with_layer helper
            with_layer(layer_id, |layer| {
                // Determine layer type by checking if it has parameters
                let params = layer.parameters();
                
                if !params.is_empty() {
                    // This is a Linear layer
                    let in_features = layer.in_features();
                    let out_features = layer.out_features();
                    
                    // Get trainable status for this layer
                    let is_trainable = layer.is_trainable();
                    
                    layers_json.push(serde_json::json!({
                        "name": format!("layer{}", layer_idx),
                        "type": "Linear",
                        "in_features": in_features,
                        "out_features": out_features,
                        "trainable": is_trainable
                    }));
                    
                    // Get current weight and bias values from graph
                    // Parameters are stored as (node_id, initial_value) pairs
                    // First is weight, second is bias
                    if params.len() >= 2 {
                        let weight_node_id = params[0].0;
                        let bias_node_id = params[1].0;
                        
                        // Get current values from graph node or fall back to initial value
                        let weight_value = graph.nodes.get(weight_node_id)
                            .and_then(|n| n.value.as_ref())
                            .cloned()
                            .unwrap_or_else(|| params[0].1.clone());
                        
                        let bias_value = graph.nodes.get(bias_node_id)
                            .and_then(|n| n.value.as_ref())
                            .cloned()
                            .unwrap_or_else(|| params[1].1.clone());
                        
                        // Ensure tensors are on CPU for saving
                        if let (Ok(weight_cpu), Ok(bias_cpu)) = (weight_value.to_cpu(), bias_value.to_cpu()) {
                            tensors.push((format!("layer{}.weight", layer_idx), weight_cpu));
                            tensors.push((format!("layer{}.bias", layer_idx), bias_cpu));
                        }
                    }
                } else {
                    // This is an activation layer (ReLU, Sigmoid, etc.)
                    // Determine type by checking Debug format
                    let debug_str = format!("{:?}", layer);
                    let layer_type = if debug_str.contains("ReLU") {
                        "ReLU"
                    } else if debug_str.contains("Sigmoid") {
                        "Sigmoid"
                    } else if debug_str.contains("Tanh") {
                        "Tanh"
                    } else if debug_str.contains("Softmax") {
                        "Softmax"
                    } else if debug_str.contains("Flatten") {
                        "Flatten"
                    } else {
                        "ReLU" // Default fallback
                    };
                    
                    layers_json.push(serde_json::json!({
                        "name": format!("layer{}", layer_idx),
                        "type": layer_type
                    }));
                }
            });
        }
        
        // Get device string
        let device_str = self.sequential.graph().device().name();
        
        // Serialize training stages
        let stages_json: Vec<serde_json::Value> = self.training_stages.iter().map(|stage| {
            serde_json::json!({
                "epochs": stage.epochs,
                "loss": stage.loss,
                "optimizer_type": stage.optimizer_type,
                "optimizer_params": stage.optimizer_params,
                "frozen_layers": stage.frozen_layers,
                "trainable_params": stage.trainable_params,
                "frozen_params": stage.frozen_params,
                "loss_history": stage.loss_history,
                "accuracy_history": stage.accuracy_history,
                "val_loss_history": stage.val_loss_history,
                "val_accuracy_history": stage.val_accuracy_history,
            })
        }).collect();
        
        let architecture = serde_json::json!({
            "layers": layers_json,
            "device": device_str,
            "training": {
                "stages": stages_json,
                // Legacy fields for backward compatibility
                "epochs": self.training_epochs,
                "loss": self.training_loss,
                "optimizer": self.training_optimizer,
                "loss_history": self.training_loss_history,
                "accuracy_history": self.training_accuracy_history,
                "val_loss_history": self.validation_loss_history,
                "val_accuracy_history": self.validation_accuracy_history,
            }
        });
        
        let architecture_json = serde_json::to_string(&architecture)
            .map_err(|e| format!("Failed to serialize architecture: {}", e))?;
        
        // Write binary file
        let mut file = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        
        // Write magic number: "DATACODE" (8 bytes)
        file.write_all(b"DATACODE")
            .map_err(|e| format!("Failed to write magic: {}", e))?;
        
        // Write version: 1 (4 bytes, little-endian)
        file.write_all(&1u32.to_le_bytes())
            .map_err(|e| format!("Failed to write version: {}", e))?;
        
        // Write JSON length (4 bytes, little-endian)
        let json_len = architecture_json.len() as u32;
        file.write_all(&json_len.to_le_bytes())
            .map_err(|e| format!("Failed to write JSON length: {}", e))?;
        
        // Write JSON architecture
        file.write_all(architecture_json.as_bytes())
            .map_err(|e| format!("Failed to write JSON: {}", e))?;
        
        // Write number of tensors (4 bytes)
        let num_tensors = tensors.len() as u32;
        file.write_all(&num_tensors.to_le_bytes())
            .map_err(|e| format!("Failed to write tensor count: {}", e))?;
        
        // Write each tensor
        for (name, tensor) in tensors {
            // Write name length (4 bytes)
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len() as u32;
            file.write_all(&name_len.to_le_bytes())
                .map_err(|e| format!("Failed to write tensor name length: {}", e))?;
            
            // Write name
            file.write_all(name_bytes)
                .map_err(|e| format!("Failed to write tensor name: {}", e))?;
            
            // Write shape length (4 bytes)
            let shape_len = tensor.shape.len() as u32;
            file.write_all(&shape_len.to_le_bytes())
                .map_err(|e| format!("Failed to write shape length: {}", e))?;
            
            // Write shape (each dimension as u32, 4 bytes)
            for &dim in &tensor.shape {
                file.write_all(&(dim as u32).to_le_bytes())
                    .map_err(|e| format!("Failed to write shape dimension: {}", e))?;
            }
            
            // Write data (each f32 as 4 bytes, little-endian)
            for &val in &tensor.data {
                file.write_all(&val.to_bits().to_le_bytes())
                    .map_err(|e| format!("Failed to write tensor data: {}", e))?;
            }
        }
        
        file.sync_all()
            .map_err(|e| format!("Failed to sync file: {}", e))?;
        
        Ok(())
    }

    /// Load model from binary file
    /// Format: Magic (8 bytes) + Version (4 bytes) + JSON architecture + Binary tensors
    pub fn load(path: &str) -> Result<Self, String> {
        use crate::ml::layer::{add_layer_to_registry, ReLU, Sigmoid, Tanh, Softmax, Flatten};
        
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open model file '{}': {}. Make sure the file exists and is readable.", path, e))?;
        
        // Read magic number (8 bytes)
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read file header from '{}': {}. File may be corrupted or incomplete.", path, e))?;
        
        if &magic != b"DATACODE" {
            let magic_str = String::from_utf8_lossy(&magic);
            return Err(format!("Invalid model file format: expected 'DATACODE' magic number, but found '{}'. File '{}' may not be a valid DataCode model file.", magic_str, path));
        }
        
        // Read version (4 bytes, little-endian)
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)
            .map_err(|e| format!("Failed to read version: {}", e))?;
        let version = u32::from_le_bytes(version_bytes);
        
        if version != 1 {
            return Err(format!("Unsupported model file version: {}. This loader only supports version 1. File '{}' may have been saved with a newer version of DataCode.", version, path));
        }
        
        // Read JSON length (4 bytes, little-endian)
        let mut json_len_bytes = [0u8; 4];
        file.read_exact(&mut json_len_bytes)
            .map_err(|e| format!("Failed to read JSON length: {}", e))?;
        let json_len = u32::from_le_bytes(json_len_bytes) as usize;
        
        // Read JSON architecture
        let mut json_bytes = vec![0u8; json_len];
        file.read_exact(&mut json_bytes)
            .map_err(|e| format!("Failed to read JSON: {}", e))?;
        let architecture_json = String::from_utf8(json_bytes)
            .map_err(|e| format!("Invalid UTF-8 in JSON: {}", e))?;
        
        // Parse JSON
        let architecture: serde_json::Value = serde_json::from_str(&architecture_json)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;
        
        // Get device from JSON (but always load on CPU first to avoid Metal initialization issues)
        // The user can switch to Metal/GPU after loading if needed
        // This avoids panics during model loading that can occur with Metal device creation
        let device = Device::Cpu;
        
        // Get layers from JSON
        let layers_json = architecture.get("layers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("Missing or invalid 'layers' field in model architecture. File '{}' may be corrupted.", path))?;
        
        if layers_json.is_empty() {
            return Err(format!("Model architecture contains no layers. File '{}' is invalid.", path));
        }
        
        // Read number of tensors (4 bytes)
        let mut num_tensors_bytes = [0u8; 4];
        file.read_exact(&mut num_tensors_bytes)
            .map_err(|e| format!("Failed to read tensor count: {}", e))?;
        let num_tensors = u32::from_le_bytes(num_tensors_bytes) as usize;
        
        // Read all tensors into a map
        let mut tensor_map = std::collections::HashMap::new();
        for _ in 0..num_tensors {
            // Read name length (4 bytes)
            let mut name_len_bytes = [0u8; 4];
            file.read_exact(&mut name_len_bytes)
                .map_err(|e| format!("Failed to read tensor name length: {}", e))?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;
            
            // Read name
            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)
                .map_err(|e| format!("Failed to read tensor name: {}", e))?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| format!("Invalid UTF-8 in tensor name: {}", e))?;
            
            // Read shape length (4 bytes)
            let mut shape_len_bytes = [0u8; 4];
            file.read_exact(&mut shape_len_bytes)
                .map_err(|e| format!("Failed to read shape length: {}", e))?;
            let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;
            
            // Read shape
            let mut shape = Vec::new();
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 4];
                file.read_exact(&mut dim_bytes)
                    .map_err(|e| format!("Failed to read shape dimension: {}", e))?;
                shape.push(u32::from_le_bytes(dim_bytes) as usize);
            }
            
            // Read data
            let data_size: usize = shape.iter().product();
            let mut data = Vec::new();
            for _ in 0..data_size {
                let mut val_bytes = [0u8; 4];
                file.read_exact(&mut val_bytes)
                    .map_err(|e| format!("Failed to read tensor data: {}", e))?;
                let bits = u32::from_le_bytes(val_bytes);
                data.push(f32::from_bits(bits));
            }
            
            let tensor = Tensor::new(data, shape)
                .map_err(|e| format!("Failed to create tensor: {}", e))?;
            tensor_map.insert(name, tensor);
        }
        
        // Create layers based on architecture
        let mut layer_ids = Vec::new();
        for layer_json in layers_json {
            let layer_type = layer_json.get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing 'type' in layer".to_string())?;
            
            match layer_type {
                "Linear" => {
                    let in_features = layer_json.get("in_features")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| "Missing 'in_features' in Linear layer".to_string())? as usize;
                    let out_features = layer_json.get("out_features")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| "Missing 'out_features' in Linear layer".to_string())? as usize;
                    
                    let layer_name = layer_json.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    
                    // Get trainable status (default to true for backward compatibility)
                    let trainable = layer_json.get("trainable")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    
                    // Get weight and bias from tensor map
                    let weight_name = format!("{}.weight", layer_name);
                    let bias_name = format!("{}.bias", layer_name);
                    
                    let weight = tensor_map.get(&weight_name)
                        .ok_or_else(|| {
                            let available: Vec<String> = tensor_map.keys().map(|s| s.clone()).collect();
                            format!("Missing weight tensor '{}' in model file '{}'. Available tensors: {}", weight_name, path, 
                                available.join(", "))
                        })?;
                    
                    // Validate weight shape
                    if weight.shape != vec![in_features, out_features] {
                        return Err(format!(
                            "Weight shape mismatch for '{}': expected [{}, {}], got {:?}",
                            weight_name, in_features, out_features, weight.shape
                        ));
                    }
                    
                    let mut bias = tensor_map.get(&bias_name)
                        .ok_or_else(|| {
                            let available: Vec<String> = tensor_map.keys().map(|s| s.clone()).collect();
                            format!("Missing bias tensor '{}' in model file '{}'. Available tensors: {}", bias_name, path,
                                available.join(", "))
                        })?.clone();
                    
                    // Ensure bias has correct shape [1, out_features] for broadcasting
                    // If bias is 1D [out_features], reshape it to [1, out_features]
                    if bias.shape == vec![out_features] {
                        bias = bias.reshape(vec![1, out_features])
                            .map_err(|e| format!("Failed to reshape bias from [{}] to [1, {}]: {}", out_features, out_features, e))?;
                    } else if bias.shape != vec![1, out_features] {
                        return Err(format!(
                            "Bias shape mismatch: expected [1, {}] or [{}], got {:?}",
                            out_features, out_features, bias.shape
                        ));
                    }
                    
                    // Create Linear layer with loaded weights and trainable status
                    let linear = Linear::new_with_weights_and_trainable(
                        in_features,
                        out_features,
                        weight.clone(),
                        bias,
                        trainable,
                    )?;
                    
                    let layer_id = add_layer_to_registry(Box::new(linear));
                    layer_ids.push(layer_id);
                }
                "ReLU" => {
                    let relu = ReLU;
                    let layer_id = add_layer_to_registry(Box::new(relu));
                    layer_ids.push(layer_id);
                }
                "Sigmoid" => {
                    let sigmoid = Sigmoid;
                    let layer_id = add_layer_to_registry(Box::new(sigmoid));
                    layer_ids.push(layer_id);
                }
                "Tanh" => {
                    let tanh = Tanh;
                    let layer_id = add_layer_to_registry(Box::new(tanh));
                    layer_ids.push(layer_id);
                }
                "Softmax" => {
                    let softmax = Softmax;
                    let layer_id = add_layer_to_registry(Box::new(softmax));
                    layer_ids.push(layer_id);
                }
                "Flatten" => {
                    let flatten = Flatten;
                    let layer_id = add_layer_to_registry(Box::new(flatten));
                    layer_ids.push(layer_id);
                }
                _ => {
                    return Err(format!("Unsupported layer type '{}' in model file '{}'. Supported types: Linear, ReLU, Sigmoid, Tanh, Softmax, Flatten", layer_type, path));
                }
            }
        }
        
        // Create Sequential with loaded layers
        let sequential = Sequential::new_with_device(layer_ids, device)
            .map_err(|e| format!("Failed to create Sequential from loaded layers in '{}': {}", path, e))?;
        
        // Create NeuralNetwork
        let mut nn = NeuralNetwork::new(sequential)
            .map_err(|e| format!("Failed to create NeuralNetwork from loaded model '{}': {}", path, e))?;
        
        // Load training stages from JSON
        let training_data = architecture.get("training");
        if let Some(training) = training_data {
            // Load stages
            if let Some(stages_array) = training.get("stages").and_then(|v| v.as_array()) {
                let mut training_stages = Vec::new();
                for stage_json in stages_array {
                    let stage = TrainingStage {
                        epochs: stage_json.get("epochs")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                        loss: stage_json.get("loss")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        optimizer_type: stage_json.get("optimizer_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        optimizer_params: stage_json.get("optimizer_params").cloned(),
                        frozen_layers: stage_json.get("frozen_layers")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .collect()
                            })
                            .unwrap_or_else(Vec::new),
                        trainable_params: stage_json.get("trainable_params")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                        frozen_params: stage_json.get("frozen_params")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                        loss_history: stage_json.get("loss_history")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64())
                                    .map(|f| f as f32)
                                    .collect()
                            })
                            .unwrap_or_else(Vec::new),
                        accuracy_history: stage_json.get("accuracy_history")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64())
                                    .map(|f| f as f32)
                                    .collect()
                            })
                            .unwrap_or_else(Vec::new),
                        val_loss_history: stage_json.get("val_loss_history")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64())
                                    .map(|f| f as f32)
                                    .collect()
                            }),
                        val_accuracy_history: stage_json.get("val_accuracy_history")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64())
                                    .map(|f| f as f32)
                                    .collect()
                            }),
                    };
                    training_stages.push(stage);
                }
                nn.training_stages = training_stages;
            }
            
            // Load legacy fields for backward compatibility
            nn.training_epochs = training.get("epochs")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            nn.training_loss = training.get("loss")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            nn.training_optimizer = training.get("optimizer")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            nn.training_loss_history = training.get("loss_history")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64())
                        .map(|f| f as f32)
                        .collect()
                });
            nn.training_accuracy_history = training.get("accuracy_history")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64())
                        .map(|f| f as f32)
                        .collect()
                });
            nn.validation_loss_history = training.get("val_loss_history")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64())
                        .map(|f| f as f32)
                        .collect()
                });
            nn.validation_accuracy_history = training.get("val_accuracy_history")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64())
                        .map(|f| f as f32)
                        .collect()
                });
        }
        
        Ok(nn)
    }

    /// Gradient checking using finite differences
    /// Compares analytical gradients (from autograd) with numerical gradients
    /// Returns true if all gradients match within tolerance
    /// 
    /// # Arguments
    /// * `x` - Input features [batch_size, num_features]
    /// * `y` - Target labels [batch_size, num_classes] or [batch_size, 1]
    /// * `loss_type` - Loss type: "cross_entropy", "mse", etc.
    /// * `eps` - Epsilon for finite differences (default: 1e-5)
    /// * `tolerance` - Maximum relative error tolerance (default: 1e-4)
    pub fn gradient_check(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        loss_type: &str,
        eps: f32,
        tolerance: f32,
    ) -> Result<bool, String> {
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err("Gradient check requires 2D tensors".to_string());
        }

        // Use a small batch for gradient checking
        let batch_size = x.shape[0].min(10); // Use at most 10 samples
        let num_features = x.shape[1];
        let num_targets = y.shape[1];

        // Extract small batch
        let mut x_batch_data = Vec::new();
        for i in 0..batch_size {
            let row_start = i * num_features;
            let row_end = row_start + num_features;
            x_batch_data.extend_from_slice(&x.data[row_start..row_end]);
        }
        let x_batch = Tensor::new(x_batch_data, vec![batch_size, num_features])?;

        let mut y_batch_data = Vec::new();
        for i in 0..batch_size {
            let row_start = i * num_targets;
            let row_end = row_start + num_targets;
            y_batch_data.extend_from_slice(&y.data[row_start..row_end]);
        }
        let y_batch = Tensor::new(y_batch_data, vec![batch_size, num_targets])?;
        // Clone y_batch data and shape before loop to avoid borrow conflicts
        let y_batch_data_clone = y_batch.data.clone();
        let y_batch_shape_clone = y_batch.shape.clone();

        // Forward pass to initialize parameters
        let _ = self.forward(&x_batch)?;
        
        // Update param_node_ids if needed
        if self.param_node_ids.is_empty() {
            self.update_param_node_ids();
        }

        // Get device
        let device = self.sequential.graph_mut().device().clone();

        // Zero gradients
        self.sequential.zero_grad();

        // Forward pass
        let _ = self.forward(&x_batch)?;
        
        // Get output node ID
        let output_node_id = self.sequential.output_node_id()
            .ok_or("Sequential output node not set after forward pass")?;

        // Add target as input node
        let target_node_id = self.sequential.graph_mut().add_input();
        self.sequential.graph_mut().nodes[target_node_id].value = Some(y_batch.clone());

        // Compute loss
        let (_loss_node_id, backward_start_node) = match loss_type {
            "cross_entropy" => {
                use crate::ml::graph::OpType;
                let loss_node = self.sequential.graph_mut().add_op(
                    OpType::CrossEntropy,
                    vec![output_node_id, target_node_id]
                )?;
                // Compute loss value
                let logits_value = self.sequential.graph().get_output(output_node_id)?;
                let targets_value = self.sequential.graph().get_output(target_node_id)?;
                let logits_cpu = logits_value.to_cpu()?;
                let targets_cpu = targets_value.to_cpu()?;
                let loss_value = categorical_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                let loss_value_final = if logits_value.device() != &Device::Cpu {
                    loss_value.to_device(logits_value.device())?
                } else {
                    loss_value
                };
                self.sequential.graph_mut().nodes[loss_node].value = Some(loss_value_final);
                (loss_node, loss_node)
            }
            "mse" => {
                use crate::ml::graph::OpType;
                let diff_id = self.sequential.graph_mut().add_op(
                    OpType::Sub,
                    vec![output_node_id, target_node_id]
                )?;
                let diff_sq_id = self.sequential.graph_mut().add_op(
                    OpType::Mul,
                    vec![diff_id, diff_id]
                )?;
                let loss_node = self.sequential.graph_mut().add_op(
                    OpType::Mean,
                    vec![diff_sq_id]
                )?;
                (loss_node, loss_node)
            }
            _ => return Err(format!("Gradient check not supported for loss type: {}", loss_type)),
        };

        // Backward pass to get analytical gradients
        self.sequential.graph_mut().backward(backward_start_node)?;

        // Check gradients for each parameter
        // Clone param_node_ids to avoid borrow conflicts with self.forward()
        let param_node_ids_clone = self.param_node_ids.clone();
        let mut all_passed = true;
        let mut checked_params = 0;
        let mut failed_params = 0;

        for (param_idx, &param_id) in param_node_ids_clone.iter().enumerate() {
            // Get analytical gradient and parameter value (clone to avoid borrow conflicts)
            let (param_cpu, analytical_grad_cpu) = {
                let graph = self.sequential.graph();
                let analytical_grad = match graph.get_gradient(param_id) {
                    Ok(g) => g,
                    Err(_) => continue, // Skip if no gradient
                };
                let param_value = graph.get_output(param_id)?;
                let param_cpu = param_value.to_cpu()?;
                let analytical_grad_cpu = analytical_grad.to_cpu()?;
                (param_cpu, analytical_grad_cpu)
            };

            // Check each element of the parameter
            let mut param_passed = true;
            let mut checked_elements = 0;
            let mut failed_elements = 0;

            // Sample a few elements to check (to avoid too many computations)
            let total_elements = param_cpu.data.len();
            let sample_size = total_elements.min(100); // Check at most 100 elements
            let step = if total_elements > sample_size {
                total_elements / sample_size
            } else {
                1
            };

            for i in (0..total_elements).step_by(step) {
                checked_elements += 1;
                let original_value = param_cpu.data[i];
                let analytical_grad_val = analytical_grad_cpu.data[i];

                // Skip if gradient is zero (might be intentional)
                if analytical_grad_val.abs() < 1e-8 {
                    continue;
                }

                // Compute numerical gradient: (f(x + eps) - f(x - eps)) / (2 * eps)
                // Modify parameter
                let mut param_plus = param_cpu.data.clone();
                param_plus[i] = original_value + eps;
                let param_plus_tensor = Tensor::new(param_plus, param_cpu.shape.clone())?;
                if device.is_gpu() {
                    let _ = param_plus_tensor.to_device(&device)?;
                }

                // Set parameter value in graph (use block to avoid borrow conflicts)
                {
                    let graph = self.sequential.graph_mut();
                    graph.nodes[param_id].value = Some(
                        if device.is_gpu() {
                            param_plus_tensor.to_device(&device)?
                        } else {
                            param_plus_tensor
                        }
                    );
                }

                // Forward and compute loss
                // Clone data before forward to avoid borrow conflicts
                let y_batch_data = y_batch_data_clone.clone();
                let y_batch_shape = y_batch_shape_clone.clone();
                // Now compute forward pass (y_batch_data and y_batch_shape are owned, no borrow conflicts)
                let logits_plus = self.forward(&x_batch)?;
                // Create y_batch_for_loss after forward pass
                let y_batch_for_loss = Tensor::new(y_batch_data, y_batch_shape)?;
                let loss_plus = match loss_type {
                    "cross_entropy" => {
                        let logits_cpu = logits_plus.to_cpu()?;
                        let y_batch_cpu = y_batch_for_loss.to_cpu()?;
                        let loss = sparse_softmax_cross_entropy_loss(&logits_cpu, &y_batch_cpu)?;
                        loss.data[0]
                    }
                    "categorical_cross_entropy" => {
                        let logits_cpu = logits_plus.to_cpu()?;
                        let y_batch_cpu = y_batch_for_loss.to_cpu()?;
                        let loss = categorical_cross_entropy_loss(&logits_cpu, &y_batch_cpu)?;
                        loss.data[0]
                    }
                    "mse" => {
                        let diff = logits_plus.sub(&y_batch_for_loss)?;
                        let diff_sq = diff.mul(&diff)?;
                        diff_sq.mean()
                    }
                    _ => return Err(format!("Unsupported loss type: {}", loss_type)),
                };

                // Modify parameter in opposite direction
                let mut param_minus = param_cpu.data.clone();
                param_minus[i] = original_value - eps;
                let param_minus_tensor = Tensor::new(param_minus, param_cpu.shape.clone())?;

                // Set parameter value (use block to avoid borrow conflicts)
                {
                    let graph = self.sequential.graph_mut();
                    graph.nodes[param_id].value = Some(
                        if device.is_gpu() {
                            param_minus_tensor.to_device(&device)?
                        } else {
                            param_minus_tensor
                        }
                    );
                }

                // Forward and compute loss
                // Clone data before forward to avoid borrow conflicts
                let y_batch_data = y_batch_data_clone.clone();
                let y_batch_shape = y_batch_shape_clone.clone();
                let y_batch_for_loss = Tensor::new(y_batch_data, y_batch_shape)?;
                // Now compute forward pass
                let logits_minus = self.forward(&x_batch)?;
                let loss_minus = match loss_type {
                    "cross_entropy" => {
                        let logits_cpu = logits_minus.to_cpu()?;
                        let y_batch_cpu = y_batch_for_loss.to_cpu()?;
                        let loss = sparse_softmax_cross_entropy_loss(&logits_cpu, &y_batch_cpu)?;
                        loss.data[0]
                    }
                    "categorical_cross_entropy" => {
                        let logits_cpu = logits_minus.to_cpu()?;
                        let y_batch_cpu = y_batch_for_loss.to_cpu()?;
                        let loss = categorical_cross_entropy_loss(&logits_cpu, &y_batch_cpu)?;
                        loss.data[0]
                    }
                    "mse" => {
                        let diff = logits_minus.sub(&y_batch_for_loss)?;
                        let diff_sq = diff.mul(&diff)?;
                        diff_sq.mean()
                    }
                    _ => return Err(format!("Unsupported loss type: {}", loss_type)),
                };

                // Numerical gradient
                let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

                // Relative error
                let denominator = analytical_grad_val.abs() + numerical_grad.abs() + eps;
                let relative_error = (analytical_grad_val - numerical_grad).abs() / denominator;

                if relative_error > tolerance {
                    param_passed = false;
                    failed_elements += 1;
                    if failed_elements <= 5 { // Only print first few failures
                        eprintln!(
                            "Gradient check FAILED for param {} element {}: analytical={:.6}, numerical={:.6}, relative_error={:.6}",
                            param_idx, i, analytical_grad_val, numerical_grad, relative_error
                        );
                    }
                }

                // Restore original parameter value
                let param_original_tensor = Tensor::new(param_cpu.data.clone(), param_cpu.shape.clone())?;
                {
                    let graph = self.sequential.graph_mut();
                    graph.nodes[param_id].value = Some(
                        if device.is_gpu() {
                            param_original_tensor.to_device(&device)?
                        } else {
                            param_original_tensor
                        }
                    );
                }
            }

            if !param_passed {
                all_passed = false;
                failed_params += 1;
                eprintln!(
                    "Param {}: {}/{} elements failed gradient check",
                    param_idx, failed_elements, checked_elements
                );
            } else {
                eprintln!(
                    "Param {}: All {} checked elements passed gradient check",
                    param_idx, checked_elements
                );
            }

            checked_params += 1;
        }

        if all_passed {
            eprintln!("Gradient check PASSED: All {} parameters passed", checked_params);
        } else {
            eprintln!("Gradient check FAILED: {}/{} parameters failed", failed_params, checked_params);
        }

        Ok(all_passed)
    }
}
