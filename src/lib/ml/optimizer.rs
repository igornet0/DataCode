// Optimizer module for ML
// Rewritten to match MetalNN architecture

use crate::ml::autograd::Variable;
use crate::ml::tensor::Tensor;
use crate::ml::ops;
use std::rc::Rc;

/// SGD оптимизатор (Stochastic Gradient Descent)
#[derive(Debug, Clone)]
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub velocity: Vec<(Rc<Variable>, Tensor)>,
}

impl SGD {
    pub fn new(lr: f32) -> Result<Self, String> {
        if lr <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        Ok(Self {
            lr,
            momentum: 0.0,
            velocity: vec![],
        })
    }

    pub fn with_momentum(lr: f32, momentum: f32) -> Result<Self, String> {
        if lr <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        Ok(Self {
            lr,
            momentum,
            velocity: vec![],
        })
    }

    /// Обновить параметры
    pub fn step(&mut self, parameters: &[Rc<Variable>]) {
        const MAX_GRAD_NORM: f32 = 1.0; // Gradient clipping threshold
        
        for (idx, param) in parameters.iter().enumerate() {
            let grad_opt = param.grad.borrow();
            if let Some(ref grad) = *grad_opt {
                // Проверяем градиент на NaN и Inf
                let grad_arr = grad.data();
                let mut has_nan = false;
                let mut grad_norm_sq = 0.0;
                for val in grad_arr.iter() {
                    if val.is_nan() || val.is_infinite() {
                        has_nan = true;
                        break;
                    }
                    grad_norm_sq += val * val;
                }
                
                // Пропускаем обновление, если градиент содержит NaN или Inf
                if has_nan {
                    continue;
                }
                
                // Gradient clipping для предотвращения взрывающихся градиентов
                let grad_norm = grad_norm_sq.sqrt();
                let clipped_grad = if grad_norm > MAX_GRAD_NORM {
                    ops::scalar_mul(grad, MAX_GRAD_NORM / grad_norm)
                } else {
                    grad.clone()
                };
                
                // Вычисляем new_data в отдельном блоке, чтобы освободить заимствование data
                let new_data = {
                    let data = param.data.borrow();
                    
                    // Обновление с momentum
                    if self.momentum > 0.0 {
                        // Найти или создать velocity для этого параметра
                        let velocity_grad = if idx < self.velocity.len() {
                            &self.velocity[idx].1
                        } else {
                            // Создать новую velocity
                            self.velocity.push((Rc::clone(param), Tensor::zeros(grad.shape().to_vec())));
                            &self.velocity[idx].1
                        };

                        // v = momentum * v + grad
                        let v_scaled = ops::scalar_mul(velocity_grad, self.momentum);
                        let v_new = ops::add(&v_scaled, &clipped_grad);
                        
                        // param = param - lr * v
                        let update = ops::scalar_mul(&v_new, -self.lr);
                        let new_data = ops::add(&data, &update);
                        
                        // Обновить velocity
                        self.velocity[idx].1 = v_new;
                        new_data
                    } else {
                        // Обычный SGD без momentum
                        // param = param - lr * grad
                        let update = ops::scalar_mul(&clipped_grad, -self.lr);
                        ops::add(&data, &update)
                    }
                }; // data заимствование освобождается здесь
                
                // Проверяем результат на NaN перед обновлением
                let new_data_arr = new_data.data();
                let mut has_nan_result = false;
                for val in new_data_arr.iter() {
                    if val.is_nan() || val.is_infinite() {
                        has_nan_result = true;
                        break;
                    }
                }
                
                // Обновляем только если результат валидный
                if !has_nan_result {
                    *param.data.borrow_mut() = new_data;
                }
            }
        }
    }

    /// Обнулить градиенты
    pub fn zero_grad(&self, parameters: &[Rc<Variable>]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

/// Adam оптимизатор (Adaptive Moment Estimation)
#[derive(Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub step_count: usize,
    pub m: Vec<(Rc<Variable>, Tensor)>, // Первый момент (среднее градиентов) для Variable API
    pub v: Vec<(Rc<Variable>, Tensor)>, // Второй момент (среднее квадратов градиентов) для Variable API
    // Моменты для Graph API (по node_id)
    pub graph_m: std::collections::HashMap<usize, Tensor>, // Первый момент для Graph API
    pub graph_v: std::collections::HashMap<usize, Tensor>, // Второй момент для Graph API
}

impl Adam {
    /// Создать новый Adam оптимизатор
    pub fn new(lr: f32) -> Result<Self, String> {
        Self::with_params(lr, 0.9, 0.999, 1e-8)
    }

    /// Создать Adam с кастомными параметрами
    pub fn with_params(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Result<Self, String> {
        if lr <= 0.0 {
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
        
        Ok(Self {
            lr,
            beta1,
            beta2,
            epsilon,
            step_count: 0,
            m: vec![],
            v: vec![],
            graph_m: std::collections::HashMap::new(),
            graph_v: std::collections::HashMap::new(),
        })
    }

    /// Обновить параметры
    pub fn step(&mut self, parameters: &[Rc<Variable>]) {
        const MAX_GRAD_NORM: f32 = 1.0; // Gradient clipping threshold
        
        self.step_count += 1;
        let t = self.step_count as f32;
        
        // Bias correction coefficients: 1 - beta^t
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);
        
        for (idx, param) in parameters.iter().enumerate() {
            let grad_opt = param.grad.borrow();
            if let Some(ref grad) = *grad_opt {
                // Проверяем градиент на NaN и Inf
                let grad_arr = grad.data();
                let mut has_nan = false;
                let mut grad_norm_sq = 0.0;
                for val in grad_arr.iter() {
                    if val.is_nan() || val.is_infinite() {
                        has_nan = true;
                        break;
                    }
                    grad_norm_sq += val * val;
                }
                
                // Пропускаем обновление, если градиент содержит NaN или Inf
                if has_nan {
                    continue;
                }
                
                // Gradient clipping для предотвращения взрывающихся градиентов
                let grad_norm = grad_norm_sq.sqrt();
                let clipped_grad = if grad_norm > MAX_GRAD_NORM {
                    ops::scalar_mul(grad, MAX_GRAD_NORM / grad_norm)
                } else {
                    grad.clone()
                };
                
                // Инициализируем m и v для этого параметра, если нужно
                if idx >= self.m.len() {
                    self.m.push((Rc::clone(param), Tensor::zeros(grad.shape().to_vec())));
                }
                if idx >= self.v.len() {
                    self.v.push((Rc::clone(param), Tensor::zeros(grad.shape().to_vec())));
                }
                
                // Обновляем моменты
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                let m_old = &self.m[idx].1;
                let m_scaled = ops::scalar_mul(m_old, self.beta1);
                let m_grad = ops::scalar_mul(&clipped_grad, 1.0 - self.beta1);
                let m_new = ops::add(&m_scaled, &m_grad);
                
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let v_old = &self.v[idx].1;
                let grad_squared = ops::mul(&clipped_grad, &clipped_grad);
                let v_scaled = ops::scalar_mul(v_old, self.beta2);
                let v_grad = ops::scalar_mul(&grad_squared, 1.0 - self.beta2);
                let v_new = ops::add(&v_scaled, &v_grad);
                
                // Bias correction
                // m_hat = m_t / (1 - beta1^t)
                let m_hat = ops::scalar_div(&m_new, bias_correction1);
                // v_hat = v_t / (1 - beta2^t)
                let v_hat = ops::scalar_div(&v_new, bias_correction2);
                
                // Вычисляем обновление: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
                let v_hat_sqrt = ops::sqrt(&v_hat);
                // Создаем тензор epsilon той же формы, что и v_hat_sqrt
                let epsilon_tensor = Tensor::from_slice(
                    &vec![self.epsilon; v_hat_sqrt.numel()],
                    v_hat_sqrt.shape()
                );
                let v_hat_sqrt_eps = ops::add(&v_hat_sqrt, &epsilon_tensor);
                let update_ratio = ops::div(&m_hat, &v_hat_sqrt_eps);
                let update = ops::scalar_mul(&update_ratio, -self.lr);
                
                // Вычисляем new_data в отдельном блоке, чтобы освободить заимствование data
                let new_data = {
                    let data = param.data.borrow();
                    ops::add(&data, &update)
                }; // data заимствование освобождается здесь
                
                // Проверяем результат на NaN перед обновлением
                let new_data_arr = new_data.data();
                let mut has_nan_result = false;
                for val in new_data_arr.iter() {
                    if val.is_nan() || val.is_infinite() {
                        has_nan_result = true;
                        break;
                    }
                }
                
                // Обновляем только если результат валидный
                if !has_nan_result {
                    *param.data.borrow_mut() = new_data;
                    // Обновляем моменты
                    self.m[idx].1 = m_new;
                    self.v[idx].1 = v_new;
                }
            }
        }
    }

    /// Обнулить градиенты
    pub fn zero_grad(&self, parameters: &[Rc<Variable>]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

// Placeholder types for compatibility with old API
// These will be removed or adapted later
#[derive(Debug, Clone)]
pub struct Momentum {
    pub learning_rate: f32,
    pub beta: f32,
}

impl Momentum {
    pub fn new(learning_rate: f32, beta: f32) -> Result<Self, String> {
        Ok(Self { learning_rate, beta })
    }
}

#[derive(Debug, Clone)]
pub struct NAG {
    pub learning_rate: f32,
    pub beta: f32,
}

impl NAG {
    pub fn new(learning_rate: f32, beta: f32) -> Result<Self, String> {
        Ok(Self { learning_rate, beta })
    }
}

#[derive(Debug, Clone)]
pub struct Adagrad {
    pub learning_rate: f32,
    pub epsilon: f32,
}

impl Adagrad {
    pub fn new(learning_rate: f32, epsilon: f32) -> Result<Self, String> {
        Ok(Self { learning_rate, epsilon })
    }
}

#[derive(Debug, Clone)]
pub struct RMSprop {
    pub learning_rate: f32,
    pub gamma: f32,
    pub epsilon: f32,
}

impl RMSprop {
    pub fn new(learning_rate: f32, gamma: f32, epsilon: f32) -> Result<Self, String> {
        Ok(Self { learning_rate, gamma, epsilon })
    }
}

#[derive(Debug, Clone)]
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Result<Self, String> {
        Ok(Self { learning_rate, beta1, beta2, epsilon, weight_decay })
    }
}

/// Optimizer type enum (for compatibility)
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
