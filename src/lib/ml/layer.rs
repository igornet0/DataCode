// Layer implementations for neural networks
// Rewritten to match MetalNN architecture

use crate::ml::tensor::Tensor;
use crate::ml::autograd::{Variable, requires_grad, matmul_with_grad, add_with_grad, relu_with_grad, transpose_with_grad};
use std::rc::Rc;
use std::cell::RefCell;

/// Layer ID type for referencing layers in the registry
pub type LayerId = usize;

/// Global registry for storing layer instances
/// This allows us to store layer references in Value enum via LayerId
/// Note: We use RefCell for single-threaded interior mutability
struct LayerRegistry {
    layers: Vec<Box<dyn Layer>>,
    next_id: LayerId,
}

impl LayerRegistry {
    fn new() -> Self {
        LayerRegistry {
            layers: Vec::new(),
            next_id: 0,
        }
    }

    fn add(&mut self, layer: Box<dyn Layer>) -> LayerId {
        let id = self.next_id;
        self.layers.push(layer);
        self.next_id += 1;
        id
    }

    fn get(&self, id: LayerId) -> Option<&dyn Layer> {
        self.layers.get(id).map(|l| l.as_ref())
    }
}

// Global layer registry
// Note: We use RefCell for single-threaded interior mutability
thread_local! {
    static LAYER_REGISTRY: RefCell<LayerRegistry> = RefCell::new(LayerRegistry::new());
}

/// Add a layer to the registry and return its ID
pub fn add_layer_to_registry(layer: Box<dyn Layer>) -> LayerId {
    LAYER_REGISTRY.with(|registry| registry.borrow_mut().add(layer))
}

/// Execute a closure with access to a layer from the registry
pub fn with_layer<F, R>(layer_id: LayerId, f: F) -> Option<R>
where
    F: FnOnce(&dyn Layer) -> R,
{
    LAYER_REGISTRY.with(|registry| {
        let reg = registry.borrow();
        if let Some(layer) = reg.get(layer_id) {
        Some(f(layer))
    } else {
        None
    }
    })
}

/// Forward pass for a layer by ID using Variable
pub fn forward_layer_var(layer_id: LayerId, input: Rc<Variable>) -> Option<Rc<Variable>> {
    LAYER_REGISTRY.with(|registry| {
        let reg = registry.borrow();
        if let Some(layer) = reg.get(layer_id) {
            Some(layer.forward_var(input))
        } else {
            None
        }
    })
}

/// Trait for neural network layers (for registry compatibility)
/// Note: We use a single-threaded approach, so layers don't need to be Send+Sync
/// The Mutex is used for interior mutability, not thread safety
pub trait Layer: std::fmt::Debug {
    /// Forward pass through the layer (Variable-based)
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable>;
    
    /// Get all parameters as Variables
    fn parameters_var(&self) -> Vec<Rc<Variable>>;
    
    /// Get the number of input features expected by this layer
    fn in_features(&self) -> usize;
    
    /// Get the number of output features produced by this layer
    fn out_features(&self) -> usize;
    
    /// Check if this layer is trainable
    fn is_trainable(&self) -> bool {
        true
    }
    
    /// Freeze this layer
    fn freeze(&self) {
        // Default: do nothing
    }
    
    /// Unfreeze this layer
    fn unfreeze(&self) {
        // Default: do nothing
    }
    
    /// Get parameters as (node_id, tensor) pairs (for compatibility with old API)
    /// This is a compatibility method - returns empty vec for Variable-based layers
    fn parameters(&self) -> Vec<(usize, Tensor)> {
        vec![]
    }
}

/// Linear (полносвязный) слой
#[derive(Debug)]
pub struct Linear {
    pub weight: Rc<Variable>,
    pub bias: Option<Rc<Variable>>,
    pub in_features: usize,
    pub out_features: usize,
    trainable: RefCell<bool>,
}

impl Linear {
    /// Создать новый Linear слой
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Result<Self, String> {
        if in_features == 0 || out_features == 0 {
            return Err("in_features and out_features must be greater than 0".to_string());
        }

        // Инициализация весов по Kaiming/He (для ReLU)
        let kaiming_std = (2.0 / in_features as f32).sqrt();
        
        // Инициализировать веса
        let weight_data = Self::kaiming_uniform(in_features * out_features, kaiming_std);
        let weight = requires_grad(Tensor::from_slice(&weight_data, &[out_features, in_features]));

        let bias = if use_bias {
            let bias_data = vec![0.0; out_features];
            Some(requires_grad(Tensor::from_slice(&bias_data, &[out_features])))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            trainable: RefCell::new(true),
        })
    }

    /// Kaiming uniform инициализация
    fn kaiming_uniform(size: usize, std: f32) -> Vec<f32> {
        let bound = (3.0f32).sqrt() * std;
        (0..size)
            .map(|_| {
                // Простая uniform инициализация [-bound, bound]
                (rand::random() * 2.0 - 1.0) * bound
            })
            .collect()
    }

    /// Forward pass
    pub fn forward(&self, input: Rc<Variable>) -> Rc<Variable> {
        // input: [batch, in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features]
        // Формула: input @ weight^T (как в PyTorch)
        
        // Транспонируем веса: [out_features, in_features] -> [in_features, out_features]
        let weight_t = transpose_with_grad(self.weight.clone());
        
        // input @ weight^T = [batch, in_features] @ [in_features, out_features] = [batch, out_features]
        let output = matmul_with_grad(input, weight_t);
        
        // Добавить bias если есть
        if let Some(ref bias) = self.bias {
            // Для MVP упростим - в реальности нужен broadcast
            // output = output + bias
            add_with_grad(output, bias.clone())
        } else {
            output
        }
    }

    /// Получить параметры для оптимизатора
    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    /// Create a new Linear layer with specified weights and bias
    /// Used for loading models from file
    pub fn new_with_weights_and_trainable(
        in_features: usize,
        out_features: usize,
        weight: Tensor,
        bias: Tensor,
        trainable: bool,
    ) -> Result<Self, String> {
        if in_features == 0 || out_features == 0 {
            return Err("in_features and out_features must be greater than 0".to_string());
        }

        // Validate weight shape
        if weight.shape() != &[out_features, in_features] {
            return Err(format!(
                "Weight shape mismatch: expected {:?}, got {:?}",
                vec![out_features, in_features],
                weight.shape()
            ));
        }

        // Validate bias shape
        if bias.shape() != &[out_features] && bias.shape() != &[1, out_features] {
            return Err(format!(
                "Bias shape mismatch: expected {:?} or {:?}, got {:?}",
                vec![out_features],
                vec![1, out_features],
                bias.shape()
            ));
        }

        // Reshape bias to [out_features] if needed
        let bias_reshaped = if bias.shape() == &[1, out_features] {
            bias.reshape(vec![out_features])?
            } else {
            bias
        };

        let weight_var = requires_grad(weight);
        let bias_var = requires_grad(bias_reshaped);

        Ok(Self {
            weight: weight_var,
            bias: Some(bias_var),
            in_features,
            out_features,
            trainable: RefCell::new(trainable),
        })
    }
}

impl Layer for Linear {
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable> {
        self.forward(input)
    }
    
    fn parameters_var(&self) -> Vec<Rc<Variable>> {
        self.parameters()
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn is_trainable(&self) -> bool {
        *self.trainable.borrow()
    }

    fn freeze(&self) {
        *self.trainable.borrow_mut() = false;
    }

    fn unfreeze(&self) {
        *self.trainable.borrow_mut() = true;
    }
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
            trainable: RefCell::new(*self.trainable.borrow()),
        }
    }
}

/// ReLU активационный слой
#[derive(Debug, Clone)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: Rc<Variable>) -> Rc<Variable> {
        relu_with_grad(input)
    }

    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        vec![]
    }
}

impl Layer for ReLU {
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable> {
        self.forward(input)
    }
    
    fn parameters_var(&self) -> Vec<Rc<Variable>> {
        vec![]
    }

    fn in_features(&self) -> usize {
        0  // ReLU doesn't change feature count
    }

    fn out_features(&self) -> usize {
        0  // ReLU doesn't change feature count
    }
    
    fn is_trainable(&self) -> bool {
        false  // ReLU has no parameters
    }
}

/// Sigmoid активационный слой
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Sigmoid {
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable> {
        use crate::ml::ops;
        use crate::ml::autograd::Variable;
        let input_data = input.data.borrow();
        let result_data = ops::sigmoid(&input_data);
        Variable::new(result_data, input.requires_grad)
    }
    
    fn parameters_var(&self) -> Vec<Rc<Variable>> {
        vec![]
    }

    fn in_features(&self) -> usize {
        0
    }

    fn out_features(&self) -> usize {
        0
    }
    
    fn is_trainable(&self) -> bool {
        false
    }
}

/// Tanh активационный слой
#[derive(Debug, Clone)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Tanh {
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable> {
        use crate::ml::ops;
        use crate::ml::autograd::Variable;
        let input_data = input.data.borrow();
        let result_data = ops::tanh(&input_data);
        Variable::new(result_data, input.requires_grad)
    }
    
    fn parameters_var(&self) -> Vec<Rc<Variable>> {
        vec![]
    }
    
    fn in_features(&self) -> usize {
        0
    }
    
    fn out_features(&self) -> usize {
        0
    }
    
    fn is_trainable(&self) -> bool {
        false
    }
}

/// Softmax активационный слой
#[derive(Debug, Clone)]
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Softmax {
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable> {
        use crate::ml::ops;
        use crate::ml::autograd::Variable;
        let input_data = input.data.borrow();
        let result_data = ops::softmax(&input_data, 1); // axis=1 for classes
        Variable::new(result_data, input.requires_grad)
    }
    
    fn parameters_var(&self) -> Vec<Rc<Variable>> {
        vec![]
    }
    
    fn in_features(&self) -> usize {
        0
    }
    
    fn out_features(&self) -> usize {
        0
    }
    
    fn is_trainable(&self) -> bool {
        false
    }
}

/// Flatten слой
#[derive(Debug, Clone)]
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Flatten {
    fn forward_var(&self, input: Rc<Variable>) -> Rc<Variable> {
        // Flatten: reshape to [batch, -1]
        use crate::ml::autograd::Variable;
        let shape = {
            let input_data = input.data.borrow();
            input_data.shape().to_vec()
        };
        if shape.len() < 2 {
            return input; // Can't flatten 1D tensor
        }
        let batch_size = shape[0];
        let flattened_size: usize = shape[1..].iter().product();
        let new_shape = vec![batch_size, flattened_size];
        
        // Reshape tensor
        let input_data = input.data.borrow();
        let data_vec: Vec<f32> = input_data.data().iter().copied().collect();
        let reshaped = Tensor::from_slice(&data_vec, &new_shape);
        Variable::new(reshaped, input.requires_grad)
    }
    
    fn parameters_var(&self) -> Vec<Rc<Variable>> {
        vec![]
    }
    
    fn in_features(&self) -> usize {
        0
    }
    
    fn out_features(&self) -> usize {
        0
    }
    
    fn is_trainable(&self) -> bool {
        false
    }
}

/// Enum для различных типов слоёв
#[derive(Debug, Clone)]
pub enum LayerType {
    Linear(Linear),
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    Softmax(Softmax),
    Flatten(Flatten),
}

impl LayerType {
    pub fn forward(&self, input: Rc<Variable>) -> Rc<Variable> {
        match self {
            LayerType::Linear(layer) => layer.forward(input),
            LayerType::ReLU(layer) => layer.forward(input),
            LayerType::Sigmoid(layer) => layer.forward_var(input),
            LayerType::Tanh(layer) => layer.forward_var(input),
            LayerType::Softmax(layer) => layer.forward_var(input),
            LayerType::Flatten(layer) => layer.forward_var(input),
        }
    }

    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        match self {
            LayerType::Linear(layer) => layer.parameters(),
            LayerType::ReLU(_) => vec![],
            LayerType::Sigmoid(_) => vec![],
            LayerType::Tanh(_) => vec![],
            LayerType::Softmax(_) => vec![],
            LayerType::Flatten(_) => vec![],
        }
    }
}

/// Sequential контейнер для последовательных слоёв
#[derive(Debug, Clone)]
pub struct Sequential {
    pub layers: Vec<LayerType>,
    pub layer_ids: Vec<LayerId>,  // For compatibility with old API
}

impl Sequential {
    pub fn new(layers: Vec<LayerType>) -> Self {
        // Add layers to registry and collect layer_ids
        // Note: LayerType implements Clone, so we can clone layers when adding to registry
        let mut layer_ids = Vec::new();
        for layer in &layers {
            let layer_id = match layer {
                LayerType::Linear(l) => {
                    // Clone Linear layer for registry using Clone trait
                    add_layer_to_registry(Box::new(l.clone()))
                },
                LayerType::ReLU(_) => add_layer_to_registry(Box::new(ReLU)),
                LayerType::Sigmoid(_) => add_layer_to_registry(Box::new(Sigmoid)),
                LayerType::Tanh(_) => add_layer_to_registry(Box::new(Tanh)),
                LayerType::Softmax(_) => add_layer_to_registry(Box::new(Softmax)),
                LayerType::Flatten(_) => add_layer_to_registry(Box::new(Flatten)),
            };
            layer_ids.push(layer_id);
        }
        Self { layers, layer_ids }
    }

    pub fn forward(&self, input: Rc<Variable>) -> Rc<Variable> {
        // If layers is empty, use layer_ids to get layers from registry
        if self.layers.is_empty() && !self.layer_ids.is_empty() {
            let mut output = input;
            for &layer_id in &self.layer_ids {
                // Clone output before passing to forward_layer_var in case of error
                // This is needed because output is moved into forward_layer_var
                let output_clone = output.clone();
                output = match forward_layer_var(layer_id, output) {
                    Some(layer_output) => layer_output,
                    None => {
                        // Layer not found in registry - return cloned output
                        return output_clone;
                    }
                };
            }
            output
        } else {
            // Use layers if available
            let mut output = input;
            for layer in &self.layers {
                output = layer.forward(output);
            }
            output
        }
    }

    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        let mut params = Vec::new();
        
        // If layers is empty, get parameters from registry using layer_ids
        if self.layers.is_empty() && !self.layer_ids.is_empty() {
            for &layer_id in &self.layer_ids {
                if let Some(layer_params) = with_layer(layer_id, |layer| {
                    // Only get parameters from trainable layers
                    if layer.is_trainable() {
                        Some(layer.parameters_var())
                    } else {
                        Some(vec![]) // Return empty vec for frozen layers
                    }
                }) {
                    if let Some(layer_params) = layer_params {
                        params.extend(layer_params);
                    }
                }
            }
        } else {
            // Use layers if available - filter by trainable status
            for layer in &self.layers {
                // Check if layer is trainable (for LayerType, we need to check the inner layer)
                let is_trainable = match layer {
                    LayerType::Linear(l) => l.is_trainable(),
                    _ => false, // Activation layers have no parameters
                };
                if is_trainable {
                    params.extend(layer.parameters());
                }
            }
        }
        
        params
    }
    
    /// Zero gradients for all parameters
    pub fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
    
    /// Get layer IDs (for compatibility)
    pub fn layer_ids(&self) -> &[LayerId] {
        &self.layer_ids
    }

    /// Add a layer by ID (for compatibility with old API)
    pub fn add(&mut self, layer_id: LayerId) {
        self.layer_ids.push(layer_id);
        // Note: This doesn't add the actual layer to layers vec
        // This is a compatibility method for the old API
    }

    /// Create Sequential from layer_ids (for compatibility with old API)
    /// Note: This creates Sequential with empty layers - layers will be loaded from registry when needed
    pub fn new_with_device(layer_ids: Vec<LayerId>, _device: crate::ml::device::Device) -> Result<Self, String> {
        Ok(Self {
            layers: vec![],  // Empty - will be populated from registry when needed
            layer_ids,
        })
    }
}

// Простая функция random для MVP (не криптографически безопасная)
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};

    static mut SEED: u64 = 0;

    pub fn random() -> f32 {
        unsafe {
            if SEED == 0 {
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                SEED = nanos as u64;
            }
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            // Нормализуем к [0, 1) правильно: берем младшие 16 бит и делим на 65536
            let bits = (SEED >> 16) & 0xFFFF;
            bits as f32 / 65536.0
        }
    }
}
