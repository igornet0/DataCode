// Layer trait and implementations for neural networks

use crate::ml::tensor::Tensor;
use crate::ml::graph::{Graph, NodeId};
use std::sync::Mutex;

/// Layer ID type for referencing layers in the registry
pub type LayerId = usize;

/// Global registry for storing layer instances
/// This allows us to store layer references in Value enum via LayerId
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

    #[allow(dead_code)]
    fn remove(&mut self, id: LayerId) -> Option<Box<dyn Layer>> {
        if id < self.layers.len() {
            // For now, we don't actually remove to keep IDs stable
            // Just return None to indicate layer exists
            Some(self.layers.remove(id))
        } else {
            None
        }
    }
}

// Global layer registry
lazy_static::lazy_static! {
    static ref LAYER_REGISTRY: Mutex<LayerRegistry> = Mutex::new(LayerRegistry::new());
}

/// Add a layer to the registry and return its ID
pub fn add_layer_to_registry(layer: Box<dyn Layer>) -> LayerId {
    LAYER_REGISTRY.lock().unwrap().add(layer)
}


/// Execute a closure with access to a layer from the registry
pub fn with_layer<F, R>(layer_id: LayerId, f: F) -> Option<R>
where
    F: FnOnce(&dyn Layer) -> R,
{
    let registry = LAYER_REGISTRY.lock().unwrap();
    if let Some(layer) = registry.get(layer_id) {
        Some(f(layer))
    } else {
        None
    }
}


// Execute forward pass for a layer by ID
// This locks the registry, gets the layer, and executes forward while the guard is alive
pub fn forward_layer(id: LayerId, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
    // We need to call forward while holding the lock
    // This is safe because we're in a single-threaded VM context
    let registry = LAYER_REGISTRY.lock().unwrap();
    let layer = registry.get(id)
        .ok_or_else(|| format!("Layer with ID {} not found in registry", id))?;
    
    // Call forward - this borrows graph mutably, which is fine
    layer.forward(input_node_id, graph)
}

/// Trait for neural network layers
pub trait Layer: std::fmt::Debug + Send + Sync {
    /// Forward pass through the layer
    /// Returns the NodeId of the output node in the graph
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String>;
    
    /// Get all parameter nodes and their initial values
    /// Returns (node_id, initial_tensor_value) pairs
    fn parameters(&self) -> Vec<(NodeId, Tensor)>;
    
    /// Update node IDs after graph cleanup (when node IDs have changed)
    /// old_to_new: mapping from old node ID to new node ID
    /// Default implementation does nothing (for layers without parameters)
    fn update_node_ids(&self, _old_to_new: &std::collections::HashMap<NodeId, NodeId>) {
        // Default: do nothing (for layers without parameters like ReLU)
    }
    
    /// Set expected node IDs from param_node_ids (used after graph cleanup to preserve correct order)
    /// Default implementation does nothing (for layers without parameters)
    fn set_expected_node_ids(&self, _weight_id: Option<NodeId>, _bias_id: Option<NodeId>) {
        // Default: do nothing (for layers without parameters like ReLU)
    }
    
    /// Get the number of input features expected by this layer
    fn in_features(&self) -> usize;
    
    /// Get the number of output features produced by this layer
    fn out_features(&self) -> usize;
    
    /// Check if this layer is trainable (parameters will be updated during training)
    /// Default implementation returns true (for layers without parameters like ReLU)
    fn is_trainable(&self) -> bool {
        true  // Default: all layers are trainable (layers without parameters don't matter)
    }
    
    /// Freeze this layer - parameters will not be updated during training
    /// Default implementation does nothing (for layers without parameters like ReLU)
    fn freeze(&self) {
        // Default: do nothing (layers without parameters don't need freezing)
    }
    
    /// Unfreeze this layer - parameters will be updated during training
    /// Default implementation does nothing (for layers without parameters like ReLU)
    fn unfreeze(&self) {
        // Default: do nothing (layers without parameters don't need unfreezing)
    }
}

/// Linear (Dense) layer: y = x @ W + b
#[derive(Debug)]
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Tensor,  // [in_features, out_features]
    bias: Tensor,    // [out_features]
    weight_node_id: Mutex<Option<NodeId>>,  // Use Mutex for thread-safe interior mutability
    bias_node_id: Mutex<Option<NodeId>>,
    expected_weight_node_id: Mutex<Option<NodeId>>,  // Expected node ID from param_node_ids (after graph cleanup)
    expected_bias_node_id: Mutex<Option<NodeId>>,    // Expected node ID from param_node_ids (after graph cleanup)
    trainable: Mutex<bool>,  // Whether this layer's parameters should be updated during training
}

impl Linear {
    /// Create a new Linear layer with He initialization (good for ReLU)
    pub fn new(in_features: usize, out_features: usize) -> Result<Self, String> {
        if in_features == 0 || out_features == 0 {
            return Err("in_features and out_features must be greater than 0".to_string());
        }

        // Initialize weights with He initialization (good for ReLU)
        let weight = Tensor::he_normal(vec![in_features, out_features], in_features)?;
        // Initialize bias to zeros [1, out_features] for proper broadcasting
        let bias = Tensor::zeros(vec![1, out_features]);

        Ok(Linear {
            in_features,
            out_features,
            weight,
            bias,
            weight_node_id: Mutex::new(None),
            bias_node_id: Mutex::new(None),
            expected_weight_node_id: Mutex::new(None),
            expected_bias_node_id: Mutex::new(None),
            trainable: Mutex::new(true),  // By default, layers are trainable
        })
    }

    /// Create a new Linear layer with Xavier initialization (good for tanh/sigmoid)
    pub fn new_xavier(in_features: usize, out_features: usize) -> Result<Self, String> {
        if in_features == 0 || out_features == 0 {
            return Err("in_features and out_features must be greater than 0".to_string());
        }

        // Initialize weights with Xavier initialization
        let weight = Tensor::xavier_uniform(vec![in_features, out_features], in_features, out_features)?;
        // Initialize bias to zeros [1, out_features] for proper broadcasting
        let bias = Tensor::zeros(vec![1, out_features]);

        Ok(Linear {
            in_features,
            out_features,
            weight,
            bias,
            weight_node_id: Mutex::new(None),
            bias_node_id: Mutex::new(None),
            expected_weight_node_id: Mutex::new(None),
            expected_bias_node_id: Mutex::new(None),
            trainable: Mutex::new(true),  // By default, layers are trainable
        })
    }

    /// Create a new Linear layer with specified weights and bias
    /// Used for loading models from file
    pub fn new_with_weights(
        in_features: usize,
        out_features: usize,
        weight: Tensor,
        bias: Tensor,
    ) -> Result<Self, String> {
        Self::new_with_weights_and_trainable(in_features, out_features, weight, bias, true)
    }

    /// Create a new Linear layer with specified weights, bias, and trainable status
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
        if weight.shape != vec![in_features, out_features] {
            return Err(format!(
                "Weight shape mismatch: expected {:?}, got {:?}",
                vec![in_features, out_features],
                weight.shape
            ));
        }

        // Validate bias shape (should be [1, out_features] for broadcasting)
        if bias.shape != vec![1, out_features] {
            return Err(format!(
                "Bias shape mismatch: expected {:?}, got {:?}",
                vec![1, out_features],
                bias.shape
            ));
        }

        Ok(Linear {
            in_features,
            out_features,
            weight,
            bias,
            weight_node_id: Mutex::new(None),
            bias_node_id: Mutex::new(None),
            expected_weight_node_id: Mutex::new(None),
            expected_bias_node_id: Mutex::new(None),
            trainable: Mutex::new(trainable),
        })
    }

    /// Freeze this layer - parameters will not be updated during training
    pub fn freeze(&self) {
        *self.trainable.lock().unwrap() = false;
    }

    /// Unfreeze this layer - parameters will be updated during training
    pub fn unfreeze(&self) {
        *self.trainable.lock().unwrap() = true;
    }

    /// Check if this layer is trainable
    pub fn is_trainable(&self) -> bool {
        *self.trainable.lock().unwrap()
    }

    /// Set expected node IDs from param_node_ids (used after graph cleanup to preserve correct order)
    pub fn set_expected_node_ids(&self, weight_id: Option<NodeId>, bias_id: Option<NodeId>) {
        *self.expected_weight_node_id.lock().unwrap() = weight_id;
        *self.expected_bias_node_id.lock().unwrap() = bias_id;
    }

    /// Clear expected node IDs (after they've been used)
    pub fn clear_expected_node_ids(&self) {
        *self.expected_weight_node_id.lock().unwrap() = None;
        *self.expected_bias_node_id.lock().unwrap() = None;
    }

    /// Initialize parameters in the graph (if not already initialized)
    /// This is called automatically on first forward pass
    /// 
    /// Note: This only sets initial values. Updated values from optimizer are handled
    /// by Sequential::forward() which preserves them before graph.forward() clears them.
    fn ensure_parameters_initialized(&self, graph: &mut Graph) -> Result<(), String> {
        let mut weight_guard = self.weight_node_id.lock().unwrap();
        let mut bias_guard = self.bias_node_id.lock().unwrap();
        
        // Check if expected node IDs are set (from param_node_ids after graph cleanup)
        // If they are, we should use them even if current node IDs are valid,
        // to ensure we're using the correct nodes from param_node_ids
        let expected_weight_id = *self.expected_weight_node_id.lock().unwrap();
        let expected_bias_id = *self.expected_bias_node_id.lock().unwrap();
        let has_expected_ids = expected_weight_id.is_some() || expected_bias_id.is_some();
        
        // Check if node IDs are already set and valid
        if let (Some(weight_id), Some(bias_id)) = (*weight_guard, *bias_guard) {
            // Verify that node IDs are still valid (nodes exist in graph and are Input nodes)
            let weight_valid = weight_id < graph.nodes.len() 
                && matches!(graph.nodes[weight_id].op, crate::ml::graph::OpType::Input);
            let bias_valid = bias_id < graph.nodes.len()
                && matches!(graph.nodes[bias_id].op, crate::ml::graph::OpType::Input);
            
            // If node IDs are valid AND no expected IDs are set, we can return early
            // Otherwise, we need to reinitialize to use expected IDs
            if weight_valid && bias_valid && !has_expected_ids {
                // Parameters already initialized and node IDs are valid
                // Don't restore values here - they should be handled by Sequential::forward()
                // which will use updated values from optimizer if available, or fall back to
                // initial values from layer registry if graph nodes were cleared
                return Ok(());
            } else if has_expected_ids {
                // Expected IDs are set - reset current IDs to force reinitialization with expected IDs
                *weight_guard = None;
                *bias_guard = None;
            } else {
                // Node IDs are invalid (likely due to graph cleanup), reset them
                // After graph cleanup, parameters exist in graph but with new IDs
                // We'll find existing parameter nodes by matching shape, or create new ones if not found
                *weight_guard = None;
                *bias_guard = None;
            }
        }

        // First time initialization or reinitialization after cleanup
        // Try to find existing parameter nodes by matching shape (after cleanup, they should exist)
        // OR use expected node IDs from param_node_ids if available (preserves correct order)
        let mut weight_id = None;
        let mut bias_id = None;
        
        // First, check if expected node IDs are set (from param_node_ids after graph cleanup)
        let expected_weight_id = *self.expected_weight_node_id.lock().unwrap();
        let expected_bias_id = *self.expected_bias_node_id.lock().unwrap();
        
        if let Some(exp_weight_id) = expected_weight_id {
            // Verify the expected node ID is valid
            // After graph cleanup, nodes should exist and be Input nodes
            // We check shape if value exists, but don't require it (value will be set later)
            if exp_weight_id < graph.nodes.len() {
                let node = &graph.nodes[exp_weight_id];
                if matches!(node.op, crate::ml::graph::OpType::Input) {
                    // If node has a value, verify shape matches
                    // If no value, we'll use it anyway and set the value below
                    let shape_matches = if let Some(existing_value) = &node.value {
                        existing_value.shape == self.weight.shape
                    } else {
                        true // No value yet, will set it below
                    };
                    if shape_matches {
                        weight_id = Some(exp_weight_id);
                    }
                }
            }
        }
        
        if let Some(exp_bias_id) = expected_bias_id {
            // Verify the expected node ID is valid
            // After graph cleanup, nodes should exist and be Input nodes
            // We check shape if value exists, but don't require it (value will be set later)
            if exp_bias_id < graph.nodes.len() {
                let node = &graph.nodes[exp_bias_id];
                if matches!(node.op, crate::ml::graph::OpType::Input) {
                    // If node has a value, verify shape matches
                    // If no value, we'll use it anyway and set the value below
                    let shape_matches = if let Some(existing_value) = &node.value {
                        existing_value.shape == self.bias.shape
                    } else {
                        true // No value yet, will set it below
                    };
                    if shape_matches {
                        bias_id = Some(exp_bias_id);
                    }
                }
            }
        }
        
        // If expected IDs didn't work or weren't set, fall back to shape-only matching
        if weight_id.is_none() || bias_id.is_none() {
            for (node_idx, node) in graph.nodes.iter().enumerate() {
                if matches!(node.op, crate::ml::graph::OpType::Input) {
                    if let Some(existing_value) = &node.value {
                        // Only match nodes that are in input_nodes (actual input/parameter nodes)
                        // This avoids matching temporary nodes like targets added during training
                        if !graph.input_nodes.contains(&node_idx) {
                            continue;
                        }
                        
                        // Match by shape - weight should be [in_features, out_features]
                        if existing_value.shape == self.weight.shape && weight_id.is_none() {
                            weight_id = Some(node_idx);
                        }
                        // Match by shape - bias should be [1, out_features]
                        if existing_value.shape == self.bias.shape && bias_id.is_none() {
                            bias_id = Some(node_idx);
                        }
                    }
                }
            }
        }
        
        // Get the graph device to ensure parameters are on the correct device
        let graph_device = graph.device().clone();
        
        // Move weight to graph device if needed
        let weight_on_device = if self.weight.device() != &graph_device {
            self.weight.to_device(&graph_device).map_err(|e| {
                format!("Failed to move weight to device {}: {}", graph_device.name(), e)
            })?
        } else {
            self.weight.clone()
        };
        
        // Move bias to graph device if needed
        let bias_on_device = if self.bias.device() != &graph_device {
            self.bias.to_device(&graph_device).map_err(|e| {
                format!("Failed to move bias to device {}: {}", graph_device.name(), e)
            })?
        } else {
            self.bias.clone()
        };
        
        // If weight not found in existing nodes, create new input node
        let weight_id = if let Some(id) = weight_id {
            // Found existing node - always update its value to ensure it matches layer's weight
            // This is important for loaded models where values need to be set
            // Move to graph device if needed
            graph.nodes[id].value = Some(weight_on_device);
            id
        } else {
            let new_id = graph.add_input();
            
            // Set initial value only if node doesn't have a value
            if graph.nodes[new_id].value.is_none() {
                graph.nodes[new_id].value = Some(weight_on_device);
            }
            graph.nodes[new_id].requires_grad = true;
            new_id
        };
        
        // If bias not found in existing nodes, create new input node
        let bias_id = if let Some(id) = bias_id {
            // Found existing node - always update its value to ensure it matches layer's bias
            // This is important for loaded models where values need to be set
            // Move to graph device if needed
            graph.nodes[id].value = Some(bias_on_device);
            id
        } else {
            let new_id = graph.add_input();
            // Set initial value only if node doesn't have a value
            if graph.nodes[new_id].value.is_none() {
                graph.nodes[new_id].value = Some(bias_on_device);
            }
            graph.nodes[new_id].requires_grad = true;
            new_id
        };
        
        *weight_guard = Some(weight_id);
        *bias_guard = Some(bias_id);
        
        // Clear expected node IDs after they've been used
        *self.expected_weight_node_id.lock().unwrap() = None;
        *self.expected_bias_node_id.lock().unwrap() = None;

        Ok(())
    }

    /// Get weight node ID (must be initialized first)
    pub fn weight_node_id(&self) -> Option<NodeId> {
        *self.weight_node_id.lock().unwrap()
    }

    /// Get bias node ID (must be initialized first)
    pub fn bias_node_id(&self) -> Option<NodeId> {
        *self.bias_node_id.lock().unwrap()
    }

    /// Update node IDs after graph cleanup (when node IDs have changed)
    /// old_to_new: mapping from old node ID to new node ID
    pub fn update_node_ids_internal(&self, old_to_new: &std::collections::HashMap<NodeId, NodeId>) {
        let mut weight_guard = self.weight_node_id.lock().unwrap();
        let mut bias_guard = self.bias_node_id.lock().unwrap();
        
        if let Some(old_weight_id) = *weight_guard {
            if let Some(&new_weight_id) = old_to_new.get(&old_weight_id) {
                *weight_guard = Some(new_weight_id);
            } else {
                // Old ID not found in mapping - reset to None to force reinitialization
                *weight_guard = None;
            }
        }
        
        if let Some(old_bias_id) = *bias_guard {
            if let Some(&new_bias_id) = old_to_new.get(&old_bias_id) {
                *bias_guard = Some(new_bias_id);
            } else {
                // Old ID not found in mapping - reset to None to force reinitialization
                *bias_guard = None;
            }
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
        // Ensure parameters are initialized
        self.ensure_parameters_initialized(graph)?;
        
        let weight_id = *self.weight_node_id.lock().unwrap();
        let bias_id = *self.bias_node_id.lock().unwrap();
        
        let weight_id = weight_id.ok_or("Linear layer parameters not initialized")?;
        let bias_id = bias_id.ok_or("Linear layer parameters not initialized")?;

        // y = x @ W
        use crate::ml::graph::OpType;
        let matmul_id = graph.add_op(OpType::MatMul, vec![input_node_id, weight_id])?;

        // Broadcast bias to match output shape
        // Get output shape from matmul (will be [batch_size, out_features])
        // For now, we'll add bias as element-wise addition after checking shapes in forward pass
        
        // y = (x @ W) + b
        // Note: bias needs to be broadcasted to [batch_size, out_features]
        // We'll handle broadcasting in the forward pass by adding a special add_with_broadcast operation
        // For simplicity, we'll use regular Add which should handle broadcasting
        
        let output_id = graph.add_op(OpType::Add, vec![matmul_id, bias_id])?;
        Ok(output_id)
    }

    fn parameters(&self) -> Vec<(NodeId, Tensor)> {
        let mut params = Vec::new();
        let weight_id = self.weight_node_id.lock().unwrap()
            .unwrap_or(usize::MAX); // Placeholder when not initialized
        let bias_id = self.bias_node_id.lock().unwrap()
            .unwrap_or(usize::MAX); // Placeholder when not initialized
        
        // Always return weight and bias, even if node IDs aren't initialized yet
        params.push((weight_id, self.weight.clone()));
        params.push((bias_id, self.bias.clone()));
        params
    }

    fn update_node_ids(&self, old_to_new: &std::collections::HashMap<NodeId, NodeId>) {
        self.update_node_ids_internal(old_to_new);
    }

    fn set_expected_node_ids(&self, weight_id: Option<NodeId>, bias_id: Option<NodeId>) {
        // Call the struct method - use explicit struct method call to avoid recursion
        *self.expected_weight_node_id.lock().unwrap() = weight_id;
        *self.expected_bias_node_id.lock().unwrap() = bias_id;
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn is_trainable(&self) -> bool {
        // Call the struct method directly - we know this is Linear
        *self.trainable.lock().unwrap()
    }

    fn freeze(&self) {
        *self.trainable.lock().unwrap() = false;
    }

    fn unfreeze(&self) {
        *self.trainable.lock().unwrap() = true;
    }
}

/// ReLU activation layer
#[derive(Debug, Clone)]
pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
        use crate::ml::graph::OpType;
        graph.add_op(OpType::ReLU, vec![input_node_id])
    }

    fn parameters(&self) -> Vec<(NodeId, Tensor)> {
        vec![] // No parameters
    }

    // update_node_ids uses default implementation (does nothing for layers without parameters)

    fn in_features(&self) -> usize {
        0 // Variable
    }

    fn out_features(&self) -> usize {
        0 // Variable (same as input)
    }
}

/// Sigmoid activation layer
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
        use crate::ml::graph::OpType;
        graph.add_op(OpType::Sigmoid, vec![input_node_id])
    }

    fn parameters(&self) -> Vec<(NodeId, Tensor)> {
        vec![] // No parameters
    }

    fn in_features(&self) -> usize {
        0 // Variable
    }

    fn out_features(&self) -> usize {
        0 // Variable (same as input)
    }
}

/// Tanh activation layer
#[derive(Debug, Clone)]
pub struct Tanh;

impl Layer for Tanh {
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
        use crate::ml::graph::OpType;
        graph.add_op(OpType::Tanh, vec![input_node_id])
    }

    fn parameters(&self) -> Vec<(NodeId, Tensor)> {
        vec![] // No parameters
    }

    fn in_features(&self) -> usize {
        0 // Variable
    }

    fn out_features(&self) -> usize {
        0 // Variable (same as input)
    }
}

/// Softmax activation layer
#[derive(Debug, Clone)]
pub struct Softmax;

impl Layer for Softmax {
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
        use crate::ml::graph::OpType;
        graph.add_op(OpType::Softmax, vec![input_node_id])
    }

    fn parameters(&self) -> Vec<(NodeId, Tensor)> {
        vec![] // No parameters
    }

    fn in_features(&self) -> usize {
        0 // Variable
    }

    fn out_features(&self) -> usize {
        0 // Variable (same as input)
    }
}

/// Flatten layer: reshapes input to [batch, -1]
/// Preserves the first dimension (batch) and flattens all other dimensions
#[derive(Debug, Clone)]
pub struct Flatten;

impl Layer for Flatten {
    fn forward(&self, input_node_id: NodeId, graph: &mut Graph) -> Result<NodeId, String> {
        use crate::ml::graph::OpType;
        graph.add_op(OpType::Flatten, vec![input_node_id])
    }

    fn parameters(&self) -> Vec<(NodeId, Tensor)> {
        vec![] // No parameters
    }

    fn in_features(&self) -> usize {
        0 // Variable
    }

    fn out_features(&self) -> usize {
        0 // Variable (depends on input shape)
    }
}

/// Sequential container for composing layers
#[derive(Debug, Clone)]
pub struct Sequential {
    layer_ids: Vec<LayerId>,  // Store LayerId instead of trait objects
    graph: Graph,
    input_node_id: Option<NodeId>,
    output_node_id: Option<NodeId>,
    param_node_ids: Vec<NodeId>,
    // Cache for updated parameter values (updated by optimizer)
    // Maps param_node_id -> updated Tensor value
    param_values_cache: std::collections::HashMap<NodeId, Tensor>,
}

impl Sequential {
    /// Create a new Sequential container with layer IDs
    /// Can be created empty and layers added later
    pub fn new(layer_ids: Vec<LayerId>) -> Result<Self, String> {
        let graph = Graph::new();

        Ok(Sequential {
            layer_ids,
            graph,
            input_node_id: None,
            output_node_id: None,
            param_node_ids: Vec::new(),
            param_values_cache: std::collections::HashMap::new(),
        })
    }
    
    /// Create a new Sequential container with specific device
    pub fn new_with_device(layer_ids: Vec<LayerId>, device: crate::ml::device::Device) -> Result<Self, String> {
        let graph = Graph::new_with_device(device);

        Ok(Sequential {
            layer_ids,
            graph,
            input_node_id: None,
            output_node_id: None,
            param_node_ids: Vec::new(),
            param_values_cache: std::collections::HashMap::new(),
        })
    }
    
    /// Set device for this sequential model
    pub fn set_device(&mut self, device: crate::ml::device::Device) {
        self.graph.set_device(device);
    }
    
    /// Add a layer to the sequential container
    pub fn add(&mut self, layer_id: LayerId) {
        self.layer_ids.push(layer_id);
    }

    /// Initialize all layer parameters in the graph
    /// This should be called before training
    /// For Linear layers, creates weight and bias nodes in the graph
    pub fn init_parameters(&mut self) -> Result<(), String> {
        self.param_node_ids.clear();
        
        // Iterate through all layers and initialize parameters for Linear layers
        // Linear layers need to have their weight and bias nodes created in the graph
        // We'll handle this in the forward pass - Linear::forward will check if parameters
        // are initialized and create them if needed
        // For now, we just clear param_node_ids - they'll be collected during first forward pass
        
        Ok(())
    }
    
    /// Manually add parameter node IDs (call after initializing layers)
    pub fn add_param_node_ids(&mut self, node_ids: Vec<NodeId>) {
        self.param_node_ids.extend(node_ids);
    }


    /// Forward pass through all layers
    pub fn forward(&mut self, input: Tensor) -> Result<Tensor, String> {
        // Move input to graph device if needed
        let graph_device = self.graph.device().clone();
        let input_on_device = if input.device() != &graph_device {
            input.to_device(&graph_device).map_err(|e| {
                format!("Failed to move input tensor to device {}: {}", graph_device.name(), e)
            })?
        } else {
            input
        };
        
        // Create input node if not exists
        if self.input_node_id.is_none() {
            self.input_node_id = Some(self.graph.add_input());
        }

        let input_id = self.input_node_id.unwrap();

        // CRITICAL: Restore parameter values from cache BEFORE building computation graph
        // This is especially important after graph cleanup when parameter nodes exist but may not have values
        // After cleanup, graph contains only parameter nodes (as input nodes), so we restore their values
        // from cache which contains updated values from optimizer or preserved frozen values
        // NOTE: After cleanup, param_node_ids contains NEW node IDs, and cache has been updated to use NEW IDs
        // So we can safely restore from cache using param_node_ids
        
        if !self.param_node_ids.is_empty() && !self.param_values_cache.is_empty() {
            // Restore values for ALL parameter nodes from param_node_ids
            // After cleanup, cache has been updated with new node IDs, so this should work
            for &param_node_id in &self.param_node_ids {
                if param_node_id != input_id && param_node_id < self.graph.nodes.len() {
                    // This is a parameter node (not the input data node)
                    if let Some(cached_value) = self.param_values_cache.get(&param_node_id) {
                        // Restore value from cache (cache has been updated with new IDs after cleanup)
                        self.graph.nodes[param_node_id].value = Some(cached_value.clone());
                    } else {
                        eprintln!("[WARNING] Param {} not found in cache!", param_node_id);
                    }
                }
            }
            
            // Also restore values for any parameter nodes in input_nodes that might not be in param_node_ids yet
            // (this can happen during first forward pass before param_node_ids is populated)
            // But after cleanup, all input_nodes should be in param_node_ids, so this is mainly for first forward
            for &node_id in &self.graph.input_nodes {
                if node_id != input_id && node_id < self.graph.nodes.len() {
                    // Check if this is a parameter node (Input type) and restore from cache if available
                    if matches!(self.graph.nodes[node_id].op, crate::ml::graph::OpType::Input) {
                        if self.graph.nodes[node_id].value.is_none() {
                            if let Some(cached_value) = self.param_values_cache.get(&node_id) {
                                self.graph.nodes[node_id].value = Some(cached_value.clone());
                            }
                        }
                    }
                }
            }
        }

        // Before forward pass, set expected node IDs for Linear layers if param_node_ids is available
        // This ensures parameters are matched in the correct order after graph cleanup
        if !self.param_node_ids.is_empty() {
            let registry = LAYER_REGISTRY.lock().unwrap();
            let mut param_idx = 0;
            
            for &layer_id in &self.layer_ids {
                if let Some(layer) = registry.get(layer_id) {
                    // Check if this layer has parameters by checking parameters() length
                    // Note: After graph cleanup, parameters() may return empty or old node IDs,
                    // but we can still use it to identify Linear layers (which have 2 params)
                    let params = layer.parameters();
                    // Linear layers have 2 parameters (weight and bias)
                    // We check params.len() == 2 to identify Linear layers, regardless of node ID validity
                    if params.len() == 2 && param_idx + 1 < self.param_node_ids.len() {
                        // Set expected node IDs from param_node_ids
                        let expected_weight_id = Some(self.param_node_ids[param_idx]);
                        let expected_bias_id = Some(self.param_node_ids[param_idx + 1]);
                        layer.set_expected_node_ids(expected_weight_id, expected_bias_id);
                        param_idx += 2;
                    } else if params.len() > 0 {
                        // Layer has parameters but not 2 - might be a different layer type
                        // Skip it but don't increment param_idx (shouldn't happen for our use case)
                    }
                }
            }
            drop(registry); // Release lock before forward pass
        }

        // Build the computation graph by passing through all layers
        let mut current_node_id = input_id;
        
        // Forward through all layers (this will initialize parameters for Linear layers)
        for &layer_id in &self.layer_ids {
            // Validate that current_node_id is valid before passing to next layer
            if current_node_id >= self.graph.nodes.len() {
                return Err(format!(
                    "Invalid current_node_id {} when forwarding layer {} (graph has {} nodes)",
                    current_node_id, layer_id, self.graph.nodes.len()
                ));
            }
            current_node_id = forward_layer(layer_id, current_node_id, &mut self.graph)?;
            
            // Validate that the returned node ID is valid
            if current_node_id >= self.graph.nodes.len() {
                return Err(format!(
                    "Layer {} returned invalid node ID {} (graph has {} nodes)",
                    layer_id, current_node_id, self.graph.nodes.len()
                ));
            }
        }
        
        // Validate all node references in the graph after building
        // This catches issues where nodes reference non-existent nodes
        for (node_idx, node) in self.graph.nodes.iter().enumerate() {
            for &input_id_ref in &node.inputs {
                if input_id_ref >= self.graph.nodes.len() {
                    return Err(format!(
                        "Node {} references non-existent node {} (graph has {} nodes). \
                        This may indicate stale references after graph cleanup.",
                        node_idx, input_id_ref, self.graph.nodes.len()
                    ));
                }
            }
        }
        
        // Collect parameter node IDs from Linear layers after forward (when parameters are initialized)
        // Always update param_node_ids after forward pass to ensure they match current node IDs
        // This is especially important after graph cleanup when node IDs change
        {
            let registry = LAYER_REGISTRY.lock().unwrap();
            self.param_node_ids.clear();
            for &layer_id in &self.layer_ids {
                if let Some(layer) = registry.get(layer_id) {
                    // Check if this layer has parameters (Linear layers)
                    let params = layer.parameters();
                    for (node_id, _) in params {
                        self.param_node_ids.push(node_id);
                    }
                }
            }
            drop(registry);
        }
        
        // Ensure all parameter nodes have values set in the graph
        // This is important after graph cleanup when nodes might exist but not have values
        // Also ensure any NEW parameter nodes created during forward pass have values
        // CRITICAL: Restore values from cache for all parameters (including frozen ones)
        for &param_node_id in &self.param_node_ids {
            // Validate node ID is within graph bounds
            if param_node_id >= self.graph.nodes.len() {
                return Err(format!(
                    "Invalid parameter node ID {} (graph has {} nodes). \
                    This may indicate stale references after graph cleanup.",
                    param_node_id, self.graph.nodes.len()
                ));
            }
            
            // Always try to restore from cache first (contains updated values from optimizer or preserved frozen values)
            // This is especially important for frozen parameters that weren't updated by optimizer
            // but should still have their values preserved
            if let Some(cached_value) = self.param_values_cache.get(&param_node_id) {
                // Restore value from cache (may be more up-to-date than graph node value)
                self.graph.nodes[param_node_id].value = Some(cached_value.clone());
            } else if self.graph.nodes[param_node_id].value.is_none() {
                // If not in cache and graph node has no value, try to get from registry
                if let Some(param_value) = self.get_parameter_value(param_node_id) {
                    self.graph.nodes[param_node_id].value = Some(param_value);
                } else {
                    // If we still can't get the value, try to initialize from registry by position
                    if let Some(param_index) = self.param_node_ids.iter().position(|&id| id == param_node_id) {
                        let registry = LAYER_REGISTRY.lock().unwrap();
                        let mut registry_param_index = 0;
                        
                        for &layer_id in &self.layer_ids {
                            if let Some(layer) = registry.get(layer_id) {
                                let params = layer.parameters();
                                if params.len() == 2 {
                                    if param_index >= registry_param_index && param_index < registry_param_index + 2 {
                                        let local_index = param_index - registry_param_index;
                                        if let Some((_, param_value)) = params.get(local_index) {
                                            self.graph.nodes[param_node_id].value = Some(param_value.clone());
                                            break;
                                        }
                                    }
                                    registry_param_index += 2;
                                } else if params.len() > 0 {
                                    if param_index >= registry_param_index && param_index < registry_param_index + params.len() {
                                        let local_index = param_index - registry_param_index;
                                        if let Some((_, param_value)) = params.get(local_index) {
                                            self.graph.nodes[param_node_id].value = Some(param_value.clone());
                                            break;
                                        }
                                    }
                                    registry_param_index += params.len();
                                }
                            }
                        }
                        drop(registry);
                    }
                }
            }
            
            // Final validation: ensure parameter node has a value before proceeding
            if self.graph.nodes[param_node_id].value.is_none() {
                return Err(format!(
                    "Parameter node {} has no value and could not be restored from cache or registry. \
                    param_node_ids: {:?}, cache keys: {:?}, input_nodes: {:?}",
                    param_node_id,
                    self.param_node_ids,
                    self.param_values_cache.keys().collect::<Vec<_>>(),
                    self.graph.input_nodes
                ));
            }
        }

        // Prepare input tensors for forward pass
        // IMPORTANT: Capture input_nodes AFTER graph construction, as it may have changed
        // We need to provide tensors for all input nodes in order
        // Parameter nodes (weight, bias) may have values from optimizer updates, or initial values
        let input_nodes_snapshot = self.graph.input_nodes.clone();
        
        let mut all_inputs = Vec::new();
        for (idx, &node_id) in input_nodes_snapshot.iter().enumerate() {
            if node_id == input_id {
                all_inputs.push(input_on_device.clone());
            } else {
                // For parameter nodes, use the value that was already restored in the graph node
                // We restored values from cache above, so they should be available in graph.nodes[node_id].value
                let param_value = if let Some(value) = &self.graph.nodes[node_id].value {
                    // Use value from graph node (already restored from cache)
                    value.clone()
                } else if let Some(cached_value) = self.param_values_cache.get(&node_id) {
                    // Fallback to cache if graph node doesn't have value
                    cached_value.clone()
                } else if let Some(value) = self.get_parameter_value(node_id) {
                    // Fallback to registry
                    value
                } else {
                    return Err(format!(
                        "Parameter node {} (index {}) has no value and could not be retrieved. \
                        This should not happen after restoration. \
                        param_node_ids: {:?}, input_nodes: {:?}, cache keys: {:?}",
                        node_id, idx,
                        self.param_node_ids,
                        self.graph.input_nodes,
                        self.param_values_cache.keys().collect::<Vec<_>>()
                    ));
                };
                
                // Check for NaN/Inf in parameter values
                for &val in &param_value.data {
                    if val.is_nan() || val.is_infinite() {
                        return Err(format!(
                            "NaN or Inf detected in parameter node {} before forward pass. Shape: {:?}",
                            node_id, param_value.shape
                        ));
                    }
                }
                all_inputs.push(param_value);
            }
        }
        
        // Validate that we have the correct number of inputs
        if all_inputs.len() != input_nodes_snapshot.len() {
            return Err(format!(
                "Input count mismatch: expected {} inputs (for {} input nodes), got {} inputs. input_nodes: {:?}, param_node_ids: {:?}",
                input_nodes_snapshot.len(),
                input_nodes_snapshot.len(),
                all_inputs.len(),
                input_nodes_snapshot,
                self.param_node_ids
            ));
        }
        
        // Run forward pass through graph with all inputs
        // DEBUG: Validate that all input nodes have corresponding values
        if input_nodes_snapshot.len() != all_inputs.len() {
            return Err(format!(
                "Input count mismatch before forward: expected {} inputs (for {} input nodes), got {} inputs. \
                input_nodes: {:?}, param_node_ids: {:?}",
                input_nodes_snapshot.len(),
                input_nodes_snapshot.len(),
                all_inputs.len(),
                input_nodes_snapshot,
                self.param_node_ids
            ));
        }
        
        // DEBUG: Check that all input nodes will receive values
        for (idx, &node_id) in input_nodes_snapshot.iter().enumerate() {
            if idx >= all_inputs.len() {
                return Err(format!(
                    "Input node {} (index {}) has no corresponding tensor in all_inputs. \
                    input_nodes: {:?}, all_inputs len: {}",
                    node_id, idx, input_nodes_snapshot, all_inputs.len()
                ));
            }
            // Verify node exists in graph
            if node_id >= self.graph.nodes.len() {
                return Err(format!(
                    "Input node {} (index {}) does not exist in graph (graph has {} nodes). \
                    input_nodes: {:?}",
                    node_id, idx, self.graph.nodes.len(), input_nodes_snapshot
                ));
            }
        }
        
        self.graph.forward(all_inputs)?;

        self.output_node_id = Some(current_node_id);

        // Get output
        let output = self.graph.get_output(current_node_id)?;
        
        // Check for NaN or Inf values in output for early detection of problems
        for &val in &output.data {
            if val.is_nan() || val.is_infinite() {
                return Err(format!(
                    "NaN or Inf detected in forward pass output. Output shape: {:?}, first few values: {:?}",
                    output.shape,
                    &output.data[0..output.data.len().min(10)]
                ));
            }
        }
        
        Ok(output)
    }

    /// Get all parameter node IDs
    pub fn parameters(&self) -> &[NodeId] {
        &self.param_node_ids
    }

    /// Get all layer IDs
    pub fn layer_ids(&self) -> &[LayerId] {
        &self.layer_ids
    }

    /// Get the graph (for optimizer)
    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }
    
    /// Get the graph (immutable)
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get the output node ID
    pub fn output_node_id(&self) -> Option<NodeId> {
        self.output_node_id
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.graph.zero_grad();
    }

    /// Save current parameter values from graph nodes to cache
    /// This should be called after optimizer step to preserve updated parameter values
    /// Also preserves cached values for parameters that don't have values in graph (e.g., frozen parameters)
    pub fn save_parameter_values(&mut self) -> Result<(), String> {
        
        for &param_id in &self.param_node_ids {
            // First, try to get value from graph node (updated by optimizer or from forward pass)
            if let Some(value) = &self.graph.nodes[param_id].value {
                self.param_values_cache.insert(param_id, value.clone());
                } else {
                    // If no value in graph, preserve existing cached value if available
                    // This is important for frozen parameters that weren't updated by optimizer
                    // but should still be preserved in cache
                    if !self.param_values_cache.contains_key(&param_id) {
                        // Try to get from registry as fallback
                        if let Some(param_index) = self.param_node_ids.iter().position(|&id| id == param_id) {
                        let registry = LAYER_REGISTRY.lock().unwrap();
                        let mut registry_param_index = 0;
                        
                        for &layer_id in &self.layer_ids {
                            if let Some(layer) = registry.get(layer_id) {
                                let params = layer.parameters();
                                if params.len() == 2 {
                                    if param_index >= registry_param_index && param_index < registry_param_index + 2 {
                                        let local_index = param_index - registry_param_index;
                                        if let Some((_, param_value)) = params.get(local_index) {
                                            self.param_values_cache.insert(param_id, param_value.clone());
                                            break;
                                        }
                                    }
                                    registry_param_index += 2;
                                } else if params.len() > 0 {
                                    if param_index >= registry_param_index && param_index < registry_param_index + params.len() {
                                        let local_index = param_index - registry_param_index;
                                        if let Some((_, param_value)) = params.get(local_index) {
                                            self.param_values_cache.insert(param_id, param_value.clone());
                                            break;
                                        }
                                    }
                                    registry_param_index += params.len();
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Clear all non-parameter nodes from the graph to prevent memory leaks
    /// This should be called after optimizer step and save_parameter_values
    /// Preserves only parameter nodes (weights, biases) and updates param_node_ids accordingly
    /// Parameter values are preserved in the graph nodes themselves
    pub fn clear_non_parameter_nodes(&mut self) -> Result<(), String> {
        // Save old parameter node IDs for cache mapping
        let old_param_ids = self.param_node_ids.clone();
        
        // Build mapping from old node IDs to their values for cache update
        // Use values from graph first, then fall back to cache (important for frozen parameters)
        let mut old_id_to_value: std::collections::HashMap<NodeId, Tensor> = std::collections::HashMap::new();
        for &param_id in &old_param_ids {
            // First try to get value from graph node
            if let Some(value) = &self.graph.nodes[param_id].value {
                old_id_to_value.insert(param_id, value.clone());
            } else if let Some(cached_value) = self.param_values_cache.get(&param_id) {
                // If not in graph, use cached value (important for frozen parameters)
                old_id_to_value.insert(param_id, cached_value.clone());
            }
        }

        // Clear non-parameter nodes and get new parameter node IDs
        let new_param_ids = self.graph.clear_non_parameter_nodes(&self.param_node_ids)?;
        
        // Build mapping from old IDs to new IDs (they're in the same order)
        let old_to_new: std::collections::HashMap<NodeId, NodeId> = old_param_ids
            .iter()
            .zip(new_param_ids.iter())
            .map(|(&old, &new)| (old, new))
            .collect();
        
        // Update param_node_ids with new IDs
        self.param_node_ids = new_param_ids.clone();
        
        // Update cache: map old node IDs to new node IDs
        // Parameter values are preserved in graph nodes, but we update cache keys
        // IMPORTANT: Preserve all parameter values, including frozen ones that might only be in cache
        let mut new_cache = std::collections::HashMap::new();
        
        // First, add values from old_id_to_value (from graph nodes or cache)
        for (old_id, value) in old_id_to_value {
            if let Some(&new_id) = old_to_new.get(&old_id) {
                new_cache.insert(new_id, value);
            }
        }
        
        // Then, copy any cached values that we might have missed
        // This is important for frozen parameters that weren't updated by optimizer
        // but should still be preserved in cache
        for (old_id, value) in &self.param_values_cache {
            if let Some(&new_id) = old_to_new.get(old_id) {
                // Only add if not already in new_cache (old_id_to_value takes precedence)
                if !new_cache.contains_key(&new_id) {
                    new_cache.insert(new_id, value.clone());
                }
            }
        }
        
        // Ensure all parameters have values in the new cache
        // If a parameter is missing, try to get it from registry as fallback
        for (idx, &new_id) in new_param_ids.iter().enumerate() {
            if !new_cache.contains_key(&new_id) {
                // Parameter is missing from cache, try to get from registry
                let registry = LAYER_REGISTRY.lock().unwrap();
                let mut registry_param_index = 0;
                
                for &layer_id in &self.layer_ids {
                    if let Some(layer) = registry.get(layer_id) {
                        let params = layer.parameters();
                        if params.len() == 2 {
                            if idx >= registry_param_index && idx < registry_param_index + 2 {
                                let local_index = idx - registry_param_index;
                                if let Some((_, param_value)) = params.get(local_index) {
                                    new_cache.insert(new_id, param_value.clone());
                                    break;
                                }
                            }
                            registry_param_index += 2;
                        } else if params.len() > 0 {
                            if idx >= registry_param_index && idx < registry_param_index + params.len() {
                                let local_index = idx - registry_param_index;
                                if let Some((_, param_value)) = params.get(local_index) {
                                    new_cache.insert(new_id, param_value.clone());
                                    break;
                                }
                            }
                            registry_param_index += params.len();
                        }
                    }
                }
                drop(registry);
            }
        }
        
        self.param_values_cache = new_cache;
        
        // CRITICAL: Set parameter values in graph nodes from cache
        // This ensures that all parameters (including frozen ones) have values in graph nodes
        // after cleanup, so they're available for the next forward pass
        for &new_id in &new_param_ids {
            if let Some(cached_value) = self.param_values_cache.get(&new_id) {
                // Set value in graph node if it doesn't have one or if cache has updated value
                // This is especially important for frozen parameters that weren't updated by optimizer
                // but should still have their values preserved in graph nodes
                if self.graph.nodes[new_id].value.is_none() || 
                   self.graph.nodes[new_id].value.as_ref() != Some(cached_value) {
                    self.graph.nodes[new_id].value = Some(cached_value.clone());
                }
            } else {
                eprintln!("[WARNING] New param node {} has no value in cache after cleanup!", new_id);
            }
        }
        
        // Update node IDs in Linear layers using the old_to_new mapping
        // This ensures that weight_node_id and bias_node_id point to the correct new node IDs
        let registry = LAYER_REGISTRY.lock().unwrap();
        for &layer_id in &self.layer_ids {
            if let Some(layer) = registry.get(layer_id) {
                layer.update_node_ids(&old_to_new);
            }
        }
        drop(registry); // Release lock
        
        // Reset input and output node IDs since they'll be recreated in next forward pass
        self.input_node_id = None;
        self.output_node_id = None;
        
        Ok(())
    }

    /// Get cached parameter value or fall back to initial value from registry
    /// Returns None if the node is not a parameter node (e.g., target node for loss)
    pub fn get_parameter_value(&self, param_node_id: NodeId) -> Option<Tensor> {
        // Check if this is a parameter node
        if !self.param_node_ids.contains(&param_node_id) {
            return None; // Not a parameter node
        }

        // First, try to get from cache (updated values from optimizer)
        if let Some(cached_value) = self.param_values_cache.get(&param_node_id) {
            return Some(cached_value.clone());
        }

        // If not in cache, try to get from graph node (current value, e.g., just initialized)
        if let Some(graph_value) = &self.graph.nodes[param_node_id].value {
            return Some(graph_value.clone());
        }

        // Finally, fall back to initial value from layer registry
        // Match by position in param_node_ids rather than exact node ID,
        // since node IDs change after graph cleanup but the order is preserved
        if let Some(param_index) = self.param_node_ids.iter().position(|&id| id == param_node_id) {
            let registry = LAYER_REGISTRY.lock().unwrap();
            let mut registry_param_index = 0;
            
            for &layer_id in &self.layer_ids {
                if let Some(layer) = registry.get(layer_id) {
                    let params = layer.parameters();
                    // Linear layers have 2 parameters (weight, bias)
                    if params.len() == 2 {
                        // Check if this parameter is in this layer's range
                        if param_index >= registry_param_index && param_index < registry_param_index + 2 {
                            let local_index = param_index - registry_param_index;
                            if let Some((_, param_value)) = params.get(local_index) {
                                return Some(param_value.clone());
                            }
                        }
                        registry_param_index += 2;
                    } else if params.len() > 0 {
                        // Other layer types with parameters
                        if param_index >= registry_param_index && param_index < registry_param_index + params.len() {
                            let local_index = param_index - registry_param_index;
                            if let Some((_, param_value)) = params.get(local_index) {
                                return Some(param_value.clone());
                            }
                        }
                        registry_param_index += params.len();
                    }
                }
            }
        }

        None // Parameter not found (should not happen)
    }
}


