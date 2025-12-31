// Optimizer module for ML
// Implements Stochastic Gradient Descent SGD, Momentum, NAG, Adagrad, RMSprop, Adam, and AdamW optimizers

use crate::ml::graph::{Graph, NodeId};
use crate::ml::tensor::Tensor;
use std::collections::HashMap;

/// Stochastic Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    /// Create a new SGD optimizer with the given learning rate
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        Ok(SGD { learning_rate })
    }

    /// Perform one optimization step
    /// Updates parameters in the graph using their gradients
    /// 
    /// # Arguments
    /// * `graph` - The computational graph containing parameters and gradients
    /// * `param_node_ids` - List of node IDs that represent parameters to optimize
    /// 
    /// # Algorithm
    /// For each parameter node:
    /// 1. Get current value: param_value = graph.get_output(node_id)
    /// 2. Get gradient: grad = graph.get_gradient(node_id)
    /// 3. Update: new_value = param_value - lr * grad
    /// 4. Set new value back to the node
    pub fn step(&self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        for &param_id in param_node_ids.iter() {
            // Validate node ID
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            // Get current parameter value
            let current_value = graph.get_output(param_id)?;

            // Get gradient for this parameter
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => {
                    // If no gradient, skip this parameter (might not have been computed yet)
                    continue;
                }
            };
            
            // Ensure gradient and value have compatible shapes
            if current_value.shape != gradient.shape {
                return Err(format!(
                    "Shape mismatch: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            // Compute update: new_value = current_value - lr * gradient
            // Use tensor operations directly to preserve device compatibility
            // First, ensure gradient is on the same device as current_value
            // Save shape before potential move
            let gradient_shape = gradient.shape.clone();
            let gradient_on_device = if *gradient.device() != *current_value.device() {
                gradient.to_device(current_value.device())?
            } else {
                gradient
            };
            
            // Scale gradient: scaled_grad = lr * gradient
            // Create a tensor filled with learning_rate on the same device as gradient
            let total_size: usize = gradient_shape.iter().product();
            let lr_data = vec![self.learning_rate; total_size];
            let mut lr_tensor = Tensor::new(lr_data, gradient_shape)?;
            
            // Move lr_tensor to same device as gradient
            if gradient_on_device.device().is_gpu() {
                lr_tensor = lr_tensor.to_device(gradient_on_device.device())?;
            }
            
            // Multiply gradient by learning rate
            let scaled_grad_tensor = gradient_on_device.mul(&lr_tensor)?;

            let new_value = current_value.sub(&scaled_grad_tensor)?;

            // Update the node's value
            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// Adam optimizer
/// Adaptive Moment Estimation optimizer with momentum and adaptive learning rates
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,      // Exponential decay rate for first moment estimates
    pub beta2: f32,      // Exponential decay rate for second moment estimates
    pub epsilon: f32,    // Small constant for numerical stability
    pub t: usize,        // Time step counter
    // State: momentum (m) and velocity (v) for each parameter
    momentum: HashMap<NodeId, Tensor>,  // First moment estimates
    velocity: HashMap<NodeId, Tensor>,  // Second moment estimates
}

impl Adam {
    /// Create a new Adam optimizer
    /// 
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.001)
    /// * `beta1` - Exponential decay rate for first moment (default: 0.9)
    /// * `beta2` - Exponential decay rate for second moment (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        Adam::new_with_params(learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Create a new Adam optimizer with custom parameters
    pub fn new_with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if beta1 < 0.0 || beta1 >= 1.0 {
            return Err("beta1 must be in [0, 1)".to_string());
        }
        if beta2 < 0.0 || beta2 >= 1.0 {
            return Err("beta2 must be in [0, 1)".to_string());
        }
        if epsilon <= 0.0 {
            return Err("epsilon must be positive".to_string());
        }

        Ok(Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        })
    }

    /// Perform one optimization step using Adam algorithm
    /// 
    /// # Algorithm
    /// For each parameter:
    /// 1. m_t = β1 * m_{t-1} + (1 - β1) * g_t  (momentum)
    /// 2. v_t = β2 * v_{t-1} + (1 - β2) * g_t² (velocity)
    /// 3. m̂_t = m_t / (1 - β1^t)  (bias correction)
    /// 4. v̂_t = v_t / (1 - β2^t)  (bias correction)
    /// 5. θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        self.t += 1;

        for &param_id in param_node_ids {
            // Validate node ID
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            // Get current parameter value
            let current_value = graph.get_output(param_id)?;

            // Get gradient for this parameter
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => {
                    // If no gradient, skip this parameter
                    continue;
                }
            };

            // Ensure gradient and value have compatible shapes
            if current_value.shape != gradient.shape {
                let param_idx = param_node_ids.iter().position(|&id| id == param_id).unwrap_or(999);
                eprintln!("ERROR Optimizer Adam: Shape mismatch at parameter node {} (index {})", param_id, param_idx);
                eprintln!("ERROR: Parameter shape: {:?}, Gradient shape: {:?}", current_value.shape, gradient.shape);
                return Err(format!(
                    "Shape mismatch: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            // Initialize momentum and velocity if not present or if shapes don't match
            // This handles the case when node IDs change after graph cleanup
            let need_new_momentum = !self.momentum.contains_key(&param_id) || 
                self.momentum.get(&param_id).map(|m| m.shape != gradient.shape).unwrap_or(true);
            let need_new_velocity = !self.velocity.contains_key(&param_id) || 
                self.velocity.get(&param_id).map(|v| v.shape != gradient.shape).unwrap_or(true);
            
            if need_new_momentum {
                self.momentum.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }
            if need_new_velocity {
                self.velocity.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }

            let m_prev = self.momentum.get(&param_id).unwrap();
            let v_prev = self.velocity.get(&param_id).unwrap();
            
            // Update momentum: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            let m_scaled = m_prev.data.iter().map(|&m| self.beta1 * m).collect::<Vec<f32>>();
            let m_scaled_tensor = Tensor::new(m_scaled, m_prev.shape.clone())?;
            
            let g_scaled = gradient.data.iter().map(|&g| (1.0 - self.beta1) * g).collect::<Vec<f32>>();
            let g_scaled_tensor = Tensor::new(g_scaled, gradient.shape.clone())?;
            
            let m_t = m_scaled_tensor.add(&g_scaled_tensor)?;

            // Update velocity: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
            let g_squared_data: Vec<f32> = gradient.data.iter().map(|&g| g * g).collect();
            let g_squared = Tensor::new(g_squared_data, gradient.shape.clone())?;
            
            let v_scaled = v_prev.data.iter().map(|&v| self.beta2 * v).collect::<Vec<f32>>();
            let v_scaled_tensor = Tensor::new(v_scaled, v_prev.shape.clone())?;
            
            let g_sq_scaled = g_squared.data.iter().map(|&gs| (1.0 - self.beta2) * gs).collect::<Vec<f32>>();
            let g_sq_scaled_tensor = Tensor::new(g_sq_scaled, g_squared.shape.clone())?;
            
            let v_t = v_scaled_tensor.add(&g_sq_scaled_tensor)?;

            // Bias correction
            let beta1_t = 1.0 - self.beta1.powi(self.t as i32);
            let beta2_t = 1.0 - self.beta2.powi(self.t as i32);

            let m_hat_data: Vec<f32> = m_t.data.iter().map(|&m| m / beta1_t).collect();
            let m_hat = Tensor::new(m_hat_data, m_t.shape.clone())?;

            let v_hat_data: Vec<f32> = v_t.data.iter().map(|&v| v / beta2_t).collect();
            let v_hat = Tensor::new(v_hat_data, v_t.shape.clone())?;

            // Compute update: θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
            let sqrt_v_hat_data: Vec<f32> = v_hat.data.iter().map(|&v| v.sqrt() + self.epsilon).collect();
            let sqrt_v_hat = Tensor::new(sqrt_v_hat_data, v_hat.shape.clone())?;

            // m_hat / sqrt_v_hat (element-wise division)
            let update_data: Vec<f32> = m_hat.data
                .iter()
                .zip(sqrt_v_hat.data.iter())
                .map(|(&m, &sv)| self.learning_rate * m / sv)
                .collect();
            let update = Tensor::new(update_data, m_hat.shape.clone())?;

            // Update parameter: θ_t = θ_{t-1} - update
            let new_value = current_value.sub(&update)?;

            // Store updated momentum and velocity
            self.momentum.insert(param_id, m_t);
            self.velocity.insert(param_id, v_t);

            // Update the node's value
            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// Momentum optimizer
/// SGD with momentum: v = β * v + (1-β) * grad, w = w - η * v
#[derive(Debug)]
pub struct Momentum {
    pub learning_rate: f32,
    pub beta: f32,      // Momentum coefficient
    velocity: HashMap<NodeId, Tensor>,  // Velocity for each parameter
}

impl Momentum {
    /// Create a new Momentum optimizer
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        Momentum::new_with_params(learning_rate, 0.9)
    }

    /// Create a new Momentum optimizer with custom parameters
    pub fn new_with_params(learning_rate: f32, beta: f32) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if beta < 0.0 || beta >= 1.0 {
            return Err("beta must be in [0, 1)".to_string());
        }

        Ok(Momentum {
            learning_rate,
            beta,
            velocity: HashMap::new(),
        })
    }

    /// Perform one optimization step using Momentum algorithm
    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        for &param_id in param_node_ids {
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            let current_value = graph.get_output(param_id)?;
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => continue,
            };

            if current_value.shape != gradient.shape {
                return Err(format!(
                    "Shape mismatch between parameter and gradient: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            if !self.velocity.contains_key(&param_id) {
                self.velocity.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }

            let v_prev = self.velocity.get(&param_id).unwrap();

            let v_scaled = v_prev.data.iter().map(|&v| self.beta * v).collect::<Vec<f32>>();
            let v_scaled_tensor = Tensor::new(v_scaled, v_prev.shape.clone())?;
            
            let g_scaled = gradient.data.iter().map(|&g| (1.0 - self.beta) * g).collect::<Vec<f32>>();
            let g_scaled_tensor = Tensor::new(g_scaled, gradient.shape.clone())?;
            
            let v_t = v_scaled_tensor.add(&g_scaled_tensor)?;

            let total_size: usize = v_t.shape.iter().product();
            let lr_data = vec![self.learning_rate; total_size];
            let mut lr_tensor = Tensor::new(lr_data, v_t.shape.clone())?;
            
            if v_t.device().is_gpu() {
                lr_tensor = lr_tensor.to_device(v_t.device())?;
            }
            
            let update = v_t.mul(&lr_tensor)?;
            let new_value = current_value.sub(&update)?;

            self.velocity.insert(param_id, v_t);
            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// Nesterov Accelerated Gradient (NAG) optimizer
#[derive(Debug)]
pub struct NAG {
    pub learning_rate: f32,
    pub beta: f32,
    velocity: HashMap<NodeId, Tensor>,
}

impl NAG {
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        NAG::new_with_params(learning_rate, 0.9)
    }

    pub fn new_with_params(learning_rate: f32, beta: f32) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if beta < 0.0 || beta >= 1.0 {
            return Err("beta must be in [0, 1)".to_string());
        }

        Ok(NAG {
            learning_rate,
            beta,
            velocity: HashMap::new(),
        })
    }

    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        for &param_id in param_node_ids {
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            let current_value = graph.get_output(param_id)?;
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => continue,
            };

            if current_value.shape != gradient.shape {
                return Err(format!(
                    "Shape mismatch between parameter and gradient: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            if !self.velocity.contains_key(&param_id) {
                self.velocity.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }

            let v_prev = self.velocity.get(&param_id).unwrap();

            let v_scaled = v_prev.data.iter().map(|&v| self.beta * v).collect::<Vec<f32>>();
            let v_scaled_tensor = Tensor::new(v_scaled, v_prev.shape.clone())?;
            
            let total_size: usize = gradient.shape.iter().product();
            let lr_data = vec![self.learning_rate; total_size];
            let mut lr_tensor = Tensor::new(lr_data, gradient.shape.clone())?;
            
            if gradient.device().is_gpu() {
                lr_tensor = lr_tensor.to_device(gradient.device())?;
            }
            
            let lr_grad = gradient.mul(&lr_tensor)?;
            let v_t = v_scaled_tensor.add(&lr_grad)?;

            let new_value = current_value.sub(&v_t)?;

            self.velocity.insert(param_id, v_t);
            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// Adagrad optimizer
#[derive(Debug)]
pub struct Adagrad {
    pub learning_rate: f32,
    pub epsilon: f32,
    accumulated_grad_sq: HashMap<NodeId, Tensor>,
}

impl Adagrad {
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        Adagrad::new_with_params(learning_rate, 1e-8)
    }

    pub fn new_with_params(learning_rate: f32, epsilon: f32) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if epsilon <= 0.0 {
            return Err("epsilon must be positive".to_string());
        }

        Ok(Adagrad {
            learning_rate,
            epsilon,
            accumulated_grad_sq: HashMap::new(),
        })
    }

    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        for &param_id in param_node_ids {
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            let current_value = graph.get_output(param_id)?;
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => continue,
            };

            if current_value.shape != gradient.shape {
                return Err(format!(
                    "Shape mismatch between parameter and gradient: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            if !self.accumulated_grad_sq.contains_key(&param_id) {
                self.accumulated_grad_sq.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }

            let g_prev = self.accumulated_grad_sq.get(&param_id).unwrap();

            let g_squared_data: Vec<f32> = gradient.data.iter().map(|&g| g * g).collect();
            let g_squared = Tensor::new(g_squared_data, gradient.shape.clone())?;
            let g_t = g_prev.add(&g_squared)?;

            let sqrt_g_eps_data: Vec<f32> = g_t.data.iter().map(|&g| (g + self.epsilon).sqrt()).collect();
            let sqrt_g_eps = Tensor::new(sqrt_g_eps_data, g_t.shape.clone())?;

            let total_size: usize = gradient.shape.iter().product();
            let lr_data = vec![self.learning_rate; total_size];
            let mut lr_tensor = Tensor::new(lr_data, gradient.shape.clone())?;
            
            if gradient.device().is_gpu() {
                lr_tensor = lr_tensor.to_device(gradient.device())?;
            }

            let adaptive_lr_data: Vec<f32> = lr_tensor.data
                .iter()
                .zip(sqrt_g_eps.data.iter())
                .map(|(&lr, &sg)| lr / sg)
                .collect();
            let adaptive_lr = Tensor::new(adaptive_lr_data, gradient.shape.clone())?;

            let update = adaptive_lr.mul(&gradient)?;
            let new_value = current_value.sub(&update)?;

            self.accumulated_grad_sq.insert(param_id, g_t);
            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// RMSprop optimizer
#[derive(Debug)]
pub struct RMSprop {
    pub learning_rate: f32,
    pub gamma: f32,
    pub epsilon: f32,
    moving_avg_sq: HashMap<NodeId, Tensor>,
}

impl RMSprop {
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        RMSprop::new_with_params(learning_rate, 0.9, 1e-8)
    }

    pub fn new_with_params(learning_rate: f32, gamma: f32, epsilon: f32) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if gamma < 0.0 || gamma >= 1.0 {
            return Err("gamma must be in [0, 1)".to_string());
        }
        if epsilon <= 0.0 {
            return Err("epsilon must be positive".to_string());
        }

        Ok(RMSprop {
            learning_rate,
            gamma,
            epsilon,
            moving_avg_sq: HashMap::new(),
        })
    }

    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        for &param_id in param_node_ids {
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            let current_value = graph.get_output(param_id)?;
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => continue,
            };

            if current_value.shape != gradient.shape {
                return Err(format!(
                    "Shape mismatch between parameter and gradient: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            if !self.moving_avg_sq.contains_key(&param_id) {
                self.moving_avg_sq.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }

            let e_prev = self.moving_avg_sq.get(&param_id).unwrap();

            let g_squared_data: Vec<f32> = gradient.data.iter().map(|&g| g * g).collect();
            let g_squared = Tensor::new(g_squared_data, gradient.shape.clone())?;
            
            let e_scaled = e_prev.data.iter().map(|&e| self.gamma * e).collect::<Vec<f32>>();
            let e_scaled_tensor = Tensor::new(e_scaled, e_prev.shape.clone())?;
            
            let g_sq_scaled = g_squared.data.iter().map(|&gs| (1.0 - self.gamma) * gs).collect::<Vec<f32>>();
            let g_sq_scaled_tensor = Tensor::new(g_sq_scaled, g_squared.shape.clone())?;
            
            let e_t = e_scaled_tensor.add(&g_sq_scaled_tensor)?;

            let sqrt_e_eps_data: Vec<f32> = e_t.data.iter().map(|&e| (e + self.epsilon).sqrt()).collect();
            let sqrt_e_eps = Tensor::new(sqrt_e_eps_data, e_t.shape.clone())?;

            let total_size: usize = gradient.shape.iter().product();
            let lr_data = vec![self.learning_rate; total_size];
            let mut lr_tensor = Tensor::new(lr_data, gradient.shape.clone())?;
            
            if gradient.device().is_gpu() {
                lr_tensor = lr_tensor.to_device(gradient.device())?;
            }

            let adaptive_lr_data: Vec<f32> = lr_tensor.data
                .iter()
                .zip(sqrt_e_eps.data.iter())
                .map(|(&lr, &se)| lr / se)
                .collect();
            let adaptive_lr = Tensor::new(adaptive_lr_data, gradient.shape.clone())?;

            let update = adaptive_lr.mul(&gradient)?;
            let new_value = current_value.sub(&update)?;

            self.moving_avg_sq.insert(param_id, e_t);
            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// AdamW optimizer
#[derive(Debug)]
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub t: usize,
    momentum: HashMap<NodeId, Tensor>,
    velocity: HashMap<NodeId, Tensor>,
}

impl AdamW {
    pub fn new(learning_rate: f32) -> Result<Self, String> {
        AdamW::new_with_params(learning_rate, 0.9, 0.999, 1e-8, 0.01)
    }

    pub fn new_with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<Self, String> {
        if learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if beta1 < 0.0 || beta1 >= 1.0 {
            return Err("beta1 must be in [0, 1)".to_string());
        }
        if beta2 < 0.0 || beta2 >= 1.0 {
            return Err("beta2 must be in [0, 1)".to_string());
        }
        if epsilon <= 0.0 {
            return Err("epsilon must be positive".to_string());
        }
        if weight_decay < 0.0 {
            return Err("weight_decay must be non-negative".to_string());
        }

        Ok(AdamW {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        })
    }

    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        self.t += 1;

        for &param_id in param_node_ids {
            if param_id >= graph.nodes.len() {
                return Err(format!("Invalid parameter node ID: {}", param_id));
            }

            let current_value = graph.get_output(param_id)?;
            let gradient = match graph.get_gradient(param_id) {
                Ok(grad) => grad,
                Err(_) => continue,
            };

            if current_value.shape != gradient.shape {
                return Err(format!(
                    "Shape mismatch between parameter and gradient: {:?} vs {:?}",
                    current_value.shape, gradient.shape
                ));
            }

            if !self.momentum.contains_key(&param_id) {
                self.momentum.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }
            if !self.velocity.contains_key(&param_id) {
                self.velocity.insert(param_id, Tensor::zeros(gradient.shape.clone()));
            }

            let m_prev = self.momentum.get(&param_id).unwrap();
            let v_prev = self.velocity.get(&param_id).unwrap();

            let m_scaled = m_prev.data.iter().map(|&m| self.beta1 * m).collect::<Vec<f32>>();
            let m_scaled_tensor = Tensor::new(m_scaled, m_prev.shape.clone())?;
            
            let g_scaled = gradient.data.iter().map(|&g| (1.0 - self.beta1) * g).collect::<Vec<f32>>();
            let g_scaled_tensor = Tensor::new(g_scaled, gradient.shape.clone())?;
            
            let m_t = m_scaled_tensor.add(&g_scaled_tensor)?;

            let g_squared_data: Vec<f32> = gradient.data.iter().map(|&g| g * g).collect();
            let g_squared = Tensor::new(g_squared_data, gradient.shape.clone())?;
            
            let v_scaled = v_prev.data.iter().map(|&v| self.beta2 * v).collect::<Vec<f32>>();
            let v_scaled_tensor = Tensor::new(v_scaled, v_prev.shape.clone())?;
            
            let g_sq_scaled = g_squared.data.iter().map(|&gs| (1.0 - self.beta2) * gs).collect::<Vec<f32>>();
            let g_sq_scaled_tensor = Tensor::new(g_sq_scaled, g_squared.shape.clone())?;
            
            let v_t = v_scaled_tensor.add(&g_sq_scaled_tensor)?;

            let beta1_t = 1.0 - self.beta1.powi(self.t as i32);
            let beta2_t = 1.0 - self.beta2.powi(self.t as i32);

            let m_hat_data: Vec<f32> = m_t.data.iter().map(|&m| m / beta1_t).collect();
            let m_hat = Tensor::new(m_hat_data, m_t.shape.clone())?;

            let v_hat_data: Vec<f32> = v_t.data.iter().map(|&v| v / beta2_t).collect();
            let v_hat = Tensor::new(v_hat_data, v_t.shape.clone())?;

            let sqrt_v_hat_data: Vec<f32> = v_hat.data.iter().map(|&v| v.sqrt() + self.epsilon).collect();
            let sqrt_v_hat = Tensor::new(sqrt_v_hat_data, v_hat.shape.clone())?;

            let adam_update_data: Vec<f32> = m_hat.data
                .iter()
                .zip(sqrt_v_hat.data.iter())
                .map(|(&m, &sv)| m / sv)
                .collect();
            let adam_update = Tensor::new(adam_update_data, m_hat.shape.clone())?;

            let weight_decay_data: Vec<f32> = current_value.data.iter().map(|&w| self.weight_decay * w).collect();
            let weight_decay_term = Tensor::new(weight_decay_data, current_value.shape.clone())?;

            let combined_update = adam_update.add(&weight_decay_term)?;

            let total_size: usize = combined_update.shape.iter().product();
            let lr_data = vec![self.learning_rate; total_size];
            let mut lr_tensor = Tensor::new(lr_data, combined_update.shape.clone())?;
            
            if combined_update.device().is_gpu() {
                lr_tensor = lr_tensor.to_device(combined_update.device())?;
            }

            let update = combined_update.mul(&lr_tensor)?;

            let new_value = current_value.sub(&update)?;

            self.momentum.insert(param_id, m_t);
            self.velocity.insert(param_id, v_t);

            graph.nodes[param_id].value = Some(new_value);
        }

        Ok(())
    }
}

/// Enum to hold different optimizer types
#[derive(Debug)]
pub enum OptimizerType {
    SGD(SGD),
    Momentum(Momentum),
    NAG(NAG),
    Adagrad(Adagrad),
    RMSprop(RMSprop),
    Adam(Adam),
    AdamW(AdamW),
}

impl OptimizerType {
    /// Perform one optimization step
    pub fn step(&mut self, graph: &mut Graph, param_node_ids: &[NodeId]) -> Result<(), String> {
        match self {
            OptimizerType::SGD(opt) => opt.step(graph, param_node_ids),
            OptimizerType::Momentum(opt) => opt.step(graph, param_node_ids),
            OptimizerType::NAG(opt) => opt.step(graph, param_node_ids),
            OptimizerType::Adagrad(opt) => opt.step(graph, param_node_ids),
            OptimizerType::RMSprop(opt) => opt.step(graph, param_node_ids),
            OptimizerType::Adam(opt) => opt.step(graph, param_node_ids),
            OptimizerType::AdamW(opt) => opt.step(graph, param_node_ids),
        }
    }
    
    /// Set the learning rate for the optimizer
    pub fn set_learning_rate(&mut self, lr: f32) {
        if lr <= 0.0 {
            return; // Silently ignore invalid LR to avoid breaking training
        }
        match self {
            OptimizerType::SGD(opt) => opt.learning_rate = lr,
            OptimizerType::Momentum(opt) => opt.learning_rate = lr,
            OptimizerType::NAG(opt) => opt.learning_rate = lr,
            OptimizerType::Adagrad(opt) => opt.learning_rate = lr,
            OptimizerType::RMSprop(opt) => opt.learning_rate = lr,
            OptimizerType::Adam(opt) => opt.learning_rate = lr,
            OptimizerType::AdamW(opt) => opt.learning_rate = lr,
        }
    }
    
    /// Get the current learning rate from the optimizer
    pub fn get_learning_rate(&self) -> f32 {
        match self {
            OptimizerType::SGD(opt) => opt.learning_rate,
            OptimizerType::Momentum(opt) => opt.learning_rate,
            OptimizerType::NAG(opt) => opt.learning_rate,
            OptimizerType::Adagrad(opt) => opt.learning_rate,
            OptimizerType::RMSprop(opt) => opt.learning_rate,
            OptimizerType::Adam(opt) => opt.learning_rate,
            OptimizerType::AdamW(opt) => opt.learning_rate,
        }
    }
}

