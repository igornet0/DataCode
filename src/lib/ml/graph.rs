// Computational graph for ML module

use crate::ml::tensor::Tensor;
use crate::ml::device::Device;
use crate::ml::loss::{categorical_cross_entropy_loss, sparse_softmax_cross_entropy_loss};
use std::collections::VecDeque;

pub type NodeId = usize;

/// Types of operations in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    Input,      // Input placeholder node
    Add,        // Element-wise addition
    Sub,        // Element-wise subtraction
    Mul,        // Element-wise multiplication
    MatMul,     // Matrix multiplication
    Transpose,  // Matrix transpose
    Sum,        // Sum all elements
    Mean,       // Mean of all elements
    ReLU,       // ReLU activation
    Sigmoid,    // Sigmoid activation
    Tanh,       // Tanh activation
    Softmax,    // Softmax activation
    CrossEntropy, // Cross entropy loss operation (sparse: class indices [N,1])
    CategoricalCrossEntropy, // Categorical cross entropy loss (one-hot [N,C])
    Flatten,    // Flatten tensor: [batch, ...] -> [batch, -1]
    Broadcast,  // Broadcast tensor to target shape (takes target shape as metadata)
}

/// A node in the computational graph
#[derive(Debug, Clone)]
pub struct Node {
    pub op: OpType,
    pub inputs: Vec<NodeId>,           // Input node IDs
    pub value: Option<Tensor>,         // Computed value (after forward pass)
    pub grad: Option<Tensor>,          // Gradient (for autograd)
    pub requires_grad: bool,           // Whether this node needs gradients
}

impl Node {
    pub fn new_input() -> Self {
        Node {
            op: OpType::Input,
            inputs: Vec::new(),
            value: None,
            grad: None,
            requires_grad: false,
        }
    }

    pub fn new_op(op: OpType, inputs: Vec<NodeId>) -> Self {
        Node {
            op,
            inputs,
            value: None,
            grad: None,
            requires_grad: false,
        }
    }
}

/// Computational graph for ML operations
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub input_nodes: Vec<NodeId>,  // List of input node IDs
    pub device: Device,            // Default device for operations
}

impl Graph {
    /// Create a new empty graph with default device (CPU)
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            input_nodes: Vec::new(),
            device: Device::Cpu,
        }
    }
    
    /// Create a new empty graph with specific device
    pub fn new_with_device(device: Device) -> Self {
        Graph {
            nodes: Vec::new(),
            input_nodes: Vec::new(),
            device,
        }
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Set device (all new operations will use this device)
    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }
    

    /// Add an input placeholder node
    /// Returns the node ID
    pub fn add_input(&mut self) -> NodeId {
        let node = Node::new_input();
        let node_id = self.nodes.len();
        self.nodes.push(node);
        self.input_nodes.push(node_id);
        node_id
    }

    /// Add an operation node
    /// Returns the node ID
    pub fn add_op(&mut self, op: OpType, inputs: Vec<NodeId>) -> Result<NodeId, String> {
        // Validate input node IDs
        for &input_id in &inputs {
            if input_id >= self.nodes.len() {
                return Err(format!("Invalid input node ID: {}", input_id));
            }
        }

        let node = Node::new_op(op, inputs);
        let node_id = self.nodes.len();
        self.nodes.push(node);
        Ok(node_id)
    }

    /// Execute forward pass through the graph
    /// input_tensors: vector of tensors corresponding to input_nodes in order
    pub fn forward(&mut self, input_tensors: Vec<Tensor>) -> Result<(), String> {
        // Validate input count
        if input_tensors.len() != self.input_nodes.len() {
            eprintln!("[ERROR] Input count mismatch in graph.forward()!");
            return Err(format!(
                "Expected {} input tensors, got {}",
                self.input_nodes.len(),
                input_tensors.len()
            ));
        }

        // Clear previous values and gradients
        for node in &mut self.nodes {
            node.value = None;
            node.grad = None;
        }

        // Set input values
        for (i, &input_node_id) in self.input_nodes.iter().enumerate() {
            self.nodes[input_node_id].value = Some(input_tensors[i].clone());
        }

        // Topological sort to determine execution order
        let execution_order = self.topological_sort()?;

        // Validate that all referenced nodes exist and will be computed
        // This catches issues where nodes reference non-existent nodes or nodes that won't be computed
        for node_id in &execution_order {
            let node = &self.nodes[*node_id];
            for &input_id in &node.inputs {
                // Check that the referenced node exists
                if input_id >= self.nodes.len() {
                    return Err(format!(
                        "Node {} references non-existent node {} (graph has {} nodes)",
                        node_id, input_id, self.nodes.len()
                    ));
                }
                // Check that the referenced node is either an input node or will be computed earlier
                if !self.input_nodes.contains(&input_id) && !execution_order.contains(&input_id) {
                    return Err(format!(
                        "Node {} references node {} which is not an input node and not in execution order",
                        node_id, input_id
                    ));
                }
            }
        }

        // Execute nodes in topological order
        for node_id in execution_order.clone() {
            // Skip input nodes (already set)
            if self.input_nodes.contains(&node_id) {
                continue;
            }

            let node = &self.nodes[node_id];
            let inputs: Vec<&Tensor> = node
                .inputs
                .iter()
                .map(|&id| {
                    self.nodes[id]
                        .value
                        .as_ref()
                        .ok_or_else(|| {
                            // Debug information: find which nodes reference this node
                            let mut referencing_nodes = Vec::new();
                            for (idx, n) in self.nodes.iter().enumerate() {
                                if n.inputs.contains(&id) {
                                    referencing_nodes.push(idx);
                                }
                            }
                            // Check if this is an input node and if it should have been set
                            let is_input_node = self.input_nodes.contains(&id);
                            let node_op = format!("{:?}", self.nodes[id].op);
                            
                            eprintln!("[ERROR] Input node {} has no value!", id);
                            eprintln!("[ERROR]   Node {} references it", node_id);
                            eprintln!("[ERROR]   Node {} op: {}", id, node_op);
                            eprintln!("[ERROR]   Is input node: {}", is_input_node);
                            eprintln!("[ERROR]   Referencing nodes: {:?}", referencing_nodes);
                            eprintln!("[ERROR]   Input nodes: {:?}", self.input_nodes);
                            eprintln!("[ERROR]   Execution order: {:?}", execution_order);
                            eprintln!("[ERROR]   Total nodes: {}", self.nodes.len());
                            eprintln!("[ERROR]   Input tensors provided: {}", input_tensors.len());
                            
                            format!(
                                "Input node {} has no value. Node {} references it. \
                                Node op: {}. Is input node: {}. \
                                Referencing nodes: {:?}. Input nodes: {:?}. Execution order: {:?}. \
                                Total nodes: {}. Input tensors provided: {}",
                                id, node_id, node_op, is_input_node, referencing_nodes, 
                                self.input_nodes, execution_order, self.nodes.len(), input_tensors.len()
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Execute operation
            let result = match node.op {
                OpType::Input => {
                    return Err("Input node should not be in execution order".to_string());
                }
                OpType::Add => {
                    if inputs.len() != 2 {
                        return Err("Add operation requires 2 inputs".to_string());
                    }
                    // Try direct add first
                    match inputs[0].add(inputs[1]) {
                        Ok(result) => result,
                        Err(_) => {
                            // If shapes don't match, try broadcasting
                            // Broadcast second input to first input's shape
                            let broadcasted = inputs[1].broadcast_to(&inputs[0].shape)?;
                            inputs[0].add(&broadcasted)?
                        }
                    }
                }
                OpType::Sub => {
                    if inputs.len() != 2 {
                        return Err("Sub operation requires 2 inputs".to_string());
                    }
                    inputs[0].sub(inputs[1])?
                }
                OpType::Mul => {
                    if inputs.len() != 2 {
                        return Err("Mul operation requires 2 inputs".to_string());
                    }
                    inputs[0].mul(inputs[1])?
                }
                OpType::MatMul => {
                    if inputs.len() != 2 {
                        return Err("MatMul operation requires 2 inputs".to_string());
                    }
                    inputs[0].matmul(inputs[1])?
                }
                OpType::Transpose => {
                    if inputs.len() != 1 {
                        return Err("Transpose operation requires 1 input".to_string());
                    }
                    inputs[0].transpose()?
                }
                OpType::Sum => {
                    if inputs.len() != 1 {
                        return Err("Sum operation requires 1 input".to_string());
                    }
                    // Sum returns a scalar, but we need to wrap it in a tensor
                    let sum_value = inputs[0].sum();
                    Tensor::new(vec![sum_value], vec![1])?
                }
                OpType::Mean => {
                    if inputs.len() != 1 {
                        return Err("Mean operation requires 1 input".to_string());
                    }
                    // Mean returns a scalar, but we need to wrap it in a tensor
                    let mean_value = inputs[0].mean();
                    Tensor::new(vec![mean_value], vec![1])?
                }
                OpType::ReLU => {
                    if inputs.len() != 1 {
                        return Err("ReLU operation requires 1 input".to_string());
                    }
                    inputs[0].relu()
                }
                OpType::Sigmoid => {
                    if inputs.len() != 1 {
                        return Err("Sigmoid operation requires 1 input".to_string());
                    }
                    inputs[0].sigmoid()
                }
                OpType::Tanh => {
                    if inputs.len() != 1 {
                        return Err("Tanh operation requires 1 input".to_string());
                    }
                    inputs[0].tanh()
                }
                OpType::Softmax => {
                    if inputs.len() != 1 {
                        return Err("Softmax operation requires 1 input".to_string());
                    }
                    inputs[0].softmax()?
                }
                OpType::CrossEntropy => {
                    if inputs.len() != 2 {
                        return Err("CrossEntropy operation requires 2 inputs (logits, targets)".to_string());
                    }
                    // CrossEntropy: sparse targets (class indices [N,1])
                    // Fused Softmax-CrossEntropy: computes softmax internally, then cross-entropy
                    // This is numerically stable and avoids double softmax computation
                    // Ensure tensors are on CPU for loss computation
                    let logits_cpu = inputs[0].to_cpu()?;
                    let targets_cpu = inputs[1].to_cpu()?;
                    // Validate targets shape [N,1]
                    if targets_cpu.shape[1] != 1 {
                        return Err(format!(
                            "CrossEntropy expects class indices [batch, 1], got [batch, {}]. \
                            Use CategoricalCrossEntropy for one-hot targets [batch, C].",
                            targets_cpu.shape[1]
                        ));
                    }
                    let loss = sparse_softmax_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                    // Move loss to same device as inputs if needed
                    if inputs[0].device() != &Device::Cpu {
                        loss.to_device(inputs[0].device())?
                    } else {
                        loss
                    }
                }
                OpType::CategoricalCrossEntropy => {
                    if inputs.len() != 2 {
                        return Err("CategoricalCrossEntropy operation requires 2 inputs (logits, targets)".to_string());
                    }
                    // CategoricalCrossEntropy: one-hot targets [N,C]
                    // Fused Softmax-CrossEntropy: computes softmax internally, then cross-entropy
                    // This is numerically stable and avoids double softmax computation
                    // Ensure tensors are on CPU for loss computation
                    let logits_cpu = inputs[0].to_cpu()?;
                    let targets_cpu = inputs[1].to_cpu()?;
                    // Validate targets shape [N,C]
                    if targets_cpu.shape[1] != logits_cpu.shape[1] {
                        return Err(format!(
                            "CategoricalCrossEntropy expects one-hot targets [batch, {}], got [batch, {}]. \
                            Use CrossEntropy for class indices [batch, 1].",
                            logits_cpu.shape[1], targets_cpu.shape[1]
                        ));
                    }
                    let loss = categorical_cross_entropy_loss(&logits_cpu, &targets_cpu)?;
                    // Move loss to same device as inputs if needed
                    if inputs[0].device() != &Device::Cpu {
                        loss.to_device(inputs[0].device())?
                    } else {
                        loss
                    }
                }
                OpType::Flatten => {
                    if inputs.len() != 1 {
                        return Err("Flatten operation requires 1 input".to_string());
                    }
                    inputs[0].flatten()?
                }
                OpType::Broadcast => {
                    // Broadcast is not used in forward pass directly
                    // It's handled in Add operation
                    return Err("Broadcast operation should not be used directly".to_string());
                }
            };

            // Store result
            self.nodes[node_id].value = Some(result);
        }

        Ok(())
    }

    /// Get the output tensor of a node (after forward pass)
    pub fn get_output(&self, node_id: NodeId) -> Result<Tensor, String> {
        if node_id >= self.nodes.len() {
            return Err(format!("Invalid node ID: {}", node_id));
        }

        self.nodes[node_id]
            .value
            .as_ref()
            .cloned()
            .ok_or_else(|| format!("Node {} has no computed value. Run forward() first.", node_id))
    }

    /// Perform topological sort to determine execution order
    /// Returns vector of node IDs in execution order
    fn topological_sort(&self) -> Result<Vec<NodeId>, String> {
        // Build adjacency list and in-degree count
        let mut in_degree = vec![0; self.nodes.len()];
        let mut adjacency: Vec<Vec<NodeId>> = vec![Vec::new(); self.nodes.len()];

        for (node_id, node) in self.nodes.iter().enumerate() {
            for &input_id in &node.inputs {
                adjacency[input_id].push(node_id);
                in_degree[node_id] += 1;
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue = VecDeque::new();
        for (node_id, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        let mut result = Vec::new();
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            for &neighbor in &adjacency[node_id] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        // Check for cycles
        if result.len() != self.nodes.len() {
            return Err("Graph contains cycles".to_string());
        }

        Ok(result)
    }

    /// Execute backward pass to compute gradients
    /// output_node_id: The node from which to start backpropagation (typically the loss)
    pub fn backward(&mut self, output_node_id: NodeId) -> Result<(), String> {
        if output_node_id >= self.nodes.len() {
            return Err(format!("Invalid output node ID: {}", output_node_id));
        }

        // Ensure forward pass has been run
        if self.nodes[output_node_id].value.is_none() {
            return Err("Forward pass must be run before backward pass".to_string());
        }

        // Initialize output gradient to ones ONLY if not already set
        // This allows manual gradient setting (e.g., for sparse_cross_entropy) to work correctly
        if self.nodes[output_node_id].grad.is_none() {
            let output_shape = self.nodes[output_node_id].value.as_ref().unwrap().shape.clone();
            self.nodes[output_node_id].grad = Some(Tensor::ones(output_shape));
        }

        // Get reverse topological order (for backward pass)
        let forward_order = self.topological_sort()?;
        let mut backward_order = forward_order.clone();
        backward_order.reverse();

        // Process nodes in reverse topological order
        for &node_id in &backward_order {
            // Skip if node has no gradient or doesn't require grad
            let grad = match &self.nodes[node_id].grad {
                Some(g) => g.clone(),
                None => continue,
            };

            // Clone node inputs to avoid borrow checker issues
            let node_inputs = self.nodes[node_id].inputs.clone();
            let node_op = self.nodes[node_id].op.clone();
            
            let input_values: Vec<&Tensor> = node_inputs
                .iter()
                .map(|&id| {
                    self.nodes[id]
                        .value
                        .as_ref()
                        .ok_or_else(|| {
                            let is_input_node = self.input_nodes.contains(&id);
                            let node_op = format!("{:?}", self.nodes[id].op);
                            eprintln!("[ERROR] backward: Input node {} has no value! Node {} needs it. Is input node: {}, op: {}", 
                                      id, node_id, is_input_node, node_op);
                            eprintln!("[ERROR] backward: input_nodes: {:?}", self.input_nodes);
                            format!("Input node {} has no value (node {} needs it, is_input_node: {}, op: {})", 
                                    id, node_id, is_input_node, node_op)
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Compute gradients for each input
            let input_grads = match node_op {
                OpType::Input => {
                    // Input nodes don't propagate gradients further
                    continue;
                }
                OpType::Add => {
                    if input_values.len() != 2 {
                        return Err("Add operation requires 2 inputs".to_string());
                    }
                    // grad_a = grad, grad_b = grad (with broadcasting)
                    // If input was broadcasted during forward, we need to sum gradients over broadcasted dims
                    let grad_0 = if grad.shape == input_values[0].shape {
                        grad.clone()
                    } else {
                        grad.sum_to_shape(&input_values[0].shape)?
                    };
                    
                    let grad_1 = if grad.shape == input_values[1].shape {
                        grad.clone()
                    } else {
                        grad.sum_to_shape(&input_values[1].shape)?
                    };
                    
                    vec![grad_0, grad_1]
                }
                OpType::Sub => {
                    if input_values.len() != 2 {
                        return Err("Sub operation requires 2 inputs".to_string());
                    }
                    // grad_a = grad, grad_b = -grad (with broadcasting handling)
                    let neg_grad = grad.neg();
                    
                    let grad_0 = if grad.shape == input_values[0].shape {
                        grad.clone()
                    } else {
                        grad.sum_to_shape(&input_values[0].shape)?
                    };
                    
                    let grad_1 = if neg_grad.shape == input_values[1].shape {
                        neg_grad
                    } else {
                        neg_grad.sum_to_shape(&input_values[1].shape)?
                    };
                    
                    vec![grad_0, grad_1]
                }
                OpType::Mul => {
                    if input_values.len() != 2 {
                        return Err("Mul operation requires 2 inputs".to_string());
                    }
                    // grad_a = grad * b, grad_b = grad * a (with broadcasting handling)
                    // First, multiply grad by the other input (with broadcasting)
                    // For grad_a: multiply grad by input_values[1] (second input)
                    let grad_times_b = if grad.shape == input_values[1].shape {
                        grad.mul(input_values[1])?
                    } else {
                        // Broadcast grad to match input_values[1] shape for multiplication
                        let broadcasted_grad = grad.broadcast_to(&input_values[1].shape)?;
                        broadcasted_grad.mul(input_values[1])?
                    };
                    
                    // For grad_b: multiply grad by input_values[0] (first input)
                    let grad_times_a = if grad.shape == input_values[0].shape {
                        grad.mul(input_values[0])?
                    } else {
                        let broadcasted_grad = grad.broadcast_to(&input_values[0].shape)?;
                        broadcasted_grad.mul(input_values[0])?
                    };
                    
                    // Now sum to input shapes if they were broadcasted
                    let grad_0 = if grad_times_b.shape == input_values[0].shape {
                        grad_times_b
                    } else {
                        grad_times_b.sum_to_shape(&input_values[0].shape)?
                    };
                    
                    let grad_1 = if grad_times_a.shape == input_values[1].shape {
                        grad_times_a
                    } else {
                        grad_times_a.sum_to_shape(&input_values[1].shape)?
                    };
                    
                    vec![grad_0, grad_1]
                }
                OpType::MatMul => {
                    if input_values.len() != 2 {
                        return Err("MatMul operation requires 2 inputs".to_string());
                    }
                    // grad_a = grad @ b^T, grad_b = a^T @ grad
                    // For y = a @ b where a: [m, n], b: [n, p], y: [m, p]
                    // grad_y: [m, p]
                    // grad_a = grad_y @ b^T: [m, p] @ [p, n] = [m, n] ✓
                    // grad_b = a^T @ grad_y: [n, m] @ [m, p] = [n, p] ✓
                    // Ensure grad is on CPU for debugging and computation
                    let grad_cpu = grad.to_cpu()?;
                    
                    // Validate input shapes before computing gradients
                    let a_shape = input_values[0].shape.clone();
                    let b_shape = input_values[1].shape.clone();
                    let grad_shape = grad_cpu.shape.clone();
                    
                    // Validate that shapes are 2D
                    if a_shape.len() != 2 || b_shape.len() != 2 || grad_shape.len() != 2 {
                        return Err(format!(
                            "MatMul backward: All tensors must be 2D. Got a: {:?}, b: {:?}, grad: {:?}",
                            a_shape, b_shape, grad_shape
                        ));
                    }
                    
                    // Validate forward pass shapes: a: [m, n], b: [n, p], output: [m, p]
                    if a_shape[1] != b_shape[0] {
                        return Err(format!(
                            "MatMul backward: Incompatible forward shapes. a: {:?}, b: {:?}. Expected a[1] == b[0]",
                            a_shape, b_shape
                        ));
                    }
                    
                    // Validate gradient shape matches output shape
                    if grad_shape[0] != a_shape[0] || grad_shape[1] != b_shape[1] {
                        return Err(format!(
                            "MatMul backward: Gradient shape {:?} does not match expected output shape [{}, {}] from forward pass (a: {:?}, b: {:?})",
                            grad_shape, a_shape[0], b_shape[1], a_shape, b_shape
                        ));
                    }
                    
                    let b_t = input_values[1].transpose()?;
                    let a_t = input_values[0].transpose()?;
                    
                    // Validate transpose shapes
                    let a_t_shape = a_t.shape.clone();
                    let b_t_shape = b_t.shape.clone();
                    
                    if a_t_shape != vec![a_shape[1], a_shape[0]] {
                        return Err(format!(
                            "MatMul backward: Transpose of a failed. Expected [{}, {}], got {:?}",
                            a_shape[1], a_shape[0], a_t_shape
                        ));
                    }
                    
                    if b_t_shape != vec![b_shape[1], b_shape[0]] {
                        return Err(format!(
                            "MatMul backward: Transpose of b failed. Expected [{}, {}], got {:?}",
                            b_shape[1], b_shape[0], b_t_shape
                        ));
                    }
                    
                    // Compute gradients with error handling
                    let grad_a = grad_cpu.matmul(&b_t).map_err(|e| {
                        format!(
                            "MatMul backward: Failed to compute grad_a = grad @ b^T. grad: {:?}, b^T: {:?}. Error: {}",
                            grad_shape, b_t_shape, e
                        )
                    })?;
                    
                    let grad_b = a_t.matmul(&grad_cpu).map_err(|e| {
                        format!(
                            "MatMul backward: Failed to compute grad_b = a^T @ grad. a^T: {:?}, grad: {:?}. Error: {}",
                            a_t_shape, grad_shape, e
                        )
                    })?;
                    
                    // Validate computed gradient shapes
                    let grad_a_shape = grad_a.shape.clone();
                    let grad_b_shape = grad_b.shape.clone();
                    
                    // grad_a should have shape [m, n] (same as input a)
                    if grad_a_shape != a_shape {
                        return Err(format!(
                            "MatMul backward: grad_a shape mismatch. Expected {:?} (same as input a), got {:?}. \
                            This indicates an error in gradient computation. grad: {:?}, b^T: {:?}",
                            a_shape, grad_a_shape, grad_shape, b_t_shape
                        ));
                    }
                    
                    // grad_b should have shape [n, p] (same as input b/weight)
                    if grad_b_shape != b_shape {
                        return Err(format!(
                            "MatMul backward: grad_b shape mismatch. Expected {:?} (same as input b/weight), got {:?}. \
                            This indicates an error in gradient computation. a^T: {:?}, grad: {:?}",
                            b_shape, grad_b_shape, a_t_shape, grad_shape
                        ));
                    }
                    
                    vec![grad_a, grad_b]
                }
                OpType::Transpose => {
                    if input_values.len() != 1 {
                        return Err("Transpose operation requires 1 input".to_string());
                    }
                    // grad_input = grad^T
                    vec![grad.transpose()?]
                }
                OpType::Sum => {
                    if input_values.len() != 1 {
                        return Err("Sum operation requires 1 input".to_string());
                    }
                    // grad_input = broadcast(grad, input_shape)
                    // grad is scalar [1], need to broadcast to input shape
                    // Ensure tensor is on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let grad_val = grad_cpu.data[0];
                    let input_shape = &input_values[0].shape;
                    let total_size: usize = input_shape.iter().product();
                    vec![Tensor::new(vec![grad_val; total_size], input_shape.clone())?]
                }
                OpType::Mean => {
                    if input_values.len() != 1 {
                        return Err("Mean operation requires 1 input".to_string());
                    }
                    // grad_input = broadcast(grad / n, input_shape)
                    // Ensure tensor is on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let input_shape = &input_values[0].shape;
                    let n = input_values[0].total_size() as f32;
                    let grad_val = grad_cpu.data[0] / n;
                    let total_size: usize = input_shape.iter().product();
                    vec![Tensor::new(vec![grad_val; total_size], input_shape.clone())?]
                }
                OpType::ReLU => {
                    if input_values.len() != 1 {
                        return Err("ReLU operation requires 1 input".to_string());
                    }
                    // grad = grad * (x > 0)
                    // Ensure tensors are on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let input_cpu = input_values[0].to_cpu()?;
                    let grad_data: Vec<f32> = grad_cpu.data
                        .iter()
                        .zip(input_cpu.data.iter())
                        .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                        .collect();
                    vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                }
                OpType::Sigmoid => {
                    if input_values.len() != 1 {
                        return Err("Sigmoid operation requires 1 input".to_string());
                    }
                    // grad = grad * sigmoid(x) * (1 - sigmoid(x))
                    // Ensure tensors are on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let input_cpu = input_values[0].to_cpu()?;
                    let sigmoid_output = input_cpu.sigmoid();
                    let grad_data: Vec<f32> = grad_cpu.data
                        .iter()
                        .zip(sigmoid_output.data.iter())
                        .map(|(&g, &s)| g * s * (1.0 - s))
                        .collect();
                    vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                }
                OpType::Tanh => {
                    if input_values.len() != 1 {
                        return Err("Tanh operation requires 1 input".to_string());
                    }
                    // grad = grad * (1 - tanh²(x))
                    // Ensure tensors are on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let input_cpu = input_values[0].to_cpu()?;
                    let tanh_output = input_cpu.tanh();
                    let grad_data: Vec<f32> = grad_cpu.data
                        .iter()
                        .zip(tanh_output.data.iter())
                        .map(|(&g, &t)| g * (1.0 - t * t))
                        .collect();
                    vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                }
                OpType::Softmax => {
                    if input_values.len() != 1 {
                        return Err("Softmax operation requires 1 input".to_string());
                    }
                    // Softmax gradient is more complex: grad = softmax * (grad - sum(grad * softmax))
                    // For numerical stability, we compute it directly
                    // Ensure tensors are on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let input_cpu = input_values[0].to_cpu()?;
                    let softmax_output = input_cpu.softmax()?;
                    
                    // Compute sum(grad * softmax) along last dimension
                    let last_dim = grad_cpu.shape[grad_cpu.shape.len() - 1];
                    let other_dims: usize = if grad_cpu.shape.len() > 1 {
                        grad_cpu.shape[0..grad_cpu.shape.len() - 1].iter().product()
                    } else {
                        1
                    };
                    
                    let mut grad_data = vec![0.0; grad_cpu.data.len()];
                    
                    for i in 0..other_dims {
                        let start_idx = i * last_dim;
                        let end_idx = start_idx + last_dim;
                        
                        // Compute sum(grad * softmax) for this row
                        let sum_grad_softmax: f32 = grad_cpu.data[start_idx..end_idx]
                            .iter()
                            .zip(softmax_output.data[start_idx..end_idx].iter())
                            .map(|(&g, &s)| g * s)
                            .sum();
                        
                        // Compute gradient: softmax * (grad - sum)
                        for j in start_idx..end_idx {
                            let s = softmax_output.data[j];
                            let g = grad_cpu.data[j];
                            grad_data[j] = s * (g - sum_grad_softmax);
                        }
                    }
                    
                    vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                }
                OpType::CrossEntropy => {
                    if input_values.len() != 2 {
                        return Err("CrossEntropy operation requires 2 inputs (logits, targets)".to_string());
                    }
                    // CrossEntropy backward (sparse): 
                    // Gradient w.r.t. logits: (softmax(logits) - one_hot(targets)) / batch_size
                    // where one_hot is created from class indices
                    // Gradient w.r.t. targets: None (targets are constants)
                    // Ensure tensors are on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let logits_cpu = input_values[0].to_cpu()?;
                    let targets_cpu = input_values[1].to_cpu()?;
                    
                    // Validate targets shape [N,1]
                    if targets_cpu.shape[1] != 1 {
                        return Err(format!(
                            "CrossEntropy backward: expected class indices [batch, 1], got [batch, {}]",
                            targets_cpu.shape[1]
                        ));
                    }
                    
                    // grad is scalar [1] from loss node
                    let grad_value = grad_cpu.data[0];
                    
                    // Compute softmax of logits
                    let softmax_logits = logits_cpu.softmax()?;
                    
                    // Compute gradient: (softmax - one_hot(targets)) / batch_size
                    let batch_size = logits_cpu.shape[0];
                    let num_classes = logits_cpu.shape[1];
                    let mut grad_data = softmax_logits.data.clone();
                    
                    // Subtract one-hot encoding of targets
                    for i in 0..batch_size {
                        let target_class = targets_cpu.data[i] as usize;
                        if target_class < num_classes {
                            grad_data[i * num_classes + target_class] -= 1.0;
                        }
                    }
                    
                    // Divide by batch_size and multiply by grad_value
                    for val in &mut grad_data {
                        *val = grad_value * *val / batch_size as f32;
                    }
                    
                    // Create gradient tensor and move to same device as logits
                    let mut grad_tensor = Tensor::new(grad_data, logits_cpu.shape.clone())?;
                    if input_values[0].device() != &Device::Cpu {
                        grad_tensor = grad_tensor.to_device(input_values[0].device())?;
                    }
                    
                    // Return gradient for logits, and zeros for targets (targets are constants, don't need gradients)
                    vec![grad_tensor, Tensor::zeros(vec![1])] // Second gradient is dummy (targets don't need gradients)
                }
                OpType::CategoricalCrossEntropy => {
                    if input_values.len() != 2 {
                        return Err("CategoricalCrossEntropy operation requires 2 inputs (logits, targets)".to_string());
                    }
                    // CategoricalCrossEntropy backward: 
                    // Gradient w.r.t. logits: (softmax(logits) - targets) / batch_size
                    // Gradient w.r.t. targets: None (targets are constants)
                    // Ensure tensors are on CPU before accessing .data
                    let grad_cpu = grad.to_cpu()?;
                    let logits_cpu = input_values[0].to_cpu()?;
                    let targets_cpu = input_values[1].to_cpu()?;
                    
                    // Validate targets shape [N,C]
                    if targets_cpu.shape[1] != logits_cpu.shape[1] {
                        return Err(format!(
                            "CategoricalCrossEntropy backward: expected one-hot targets [batch, {}], got [batch, {}]",
                            logits_cpu.shape[1], targets_cpu.shape[1]
                        ));
                    }
                    
                    // grad is scalar [1] from loss node
                    let grad_value = grad_cpu.data[0];
                    
                    // Compute softmax of logits
                    let softmax_logits = logits_cpu.softmax()?;
                    
                    // Compute gradient: (softmax - targets) / batch_size
                    let batch_size = logits_cpu.shape[0] as f32;
                    let mut grad_data = Vec::with_capacity(softmax_logits.data.len());
                    
                    for i in 0..softmax_logits.data.len() {
                        let diff = softmax_logits.data[i] - targets_cpu.data[i];
                        grad_data.push(grad_value * diff / batch_size);
                    }
                    
                    // Create gradient tensor and move to same device as logits
                    let mut grad_tensor = Tensor::new(grad_data, logits_cpu.shape.clone())?;
                    if input_values[0].device() != &Device::Cpu {
                        grad_tensor = grad_tensor.to_device(input_values[0].device())?;
                    }
                    
                    // Return gradient for logits, and zeros for targets (targets are constants, don't need gradients)
                    vec![grad_tensor, Tensor::zeros(vec![1])] // Second gradient is dummy (targets don't need gradients)
                }
                OpType::Flatten => {
                    if input_values.len() != 1 {
                        return Err("Flatten operation requires 1 input".to_string());
                    }
                    // Flatten backward: reshape gradient back to input shape
                    // grad_input = reshape(grad, input_shape)
                    let input_shape = &input_values[0].shape;
                    vec![grad.reshape(input_shape.clone())?]
                }
                OpType::Broadcast => {
                    // Broadcast is handled in Add operation, not used directly
                    return Err("Broadcast operation should not be used directly in backward pass".to_string());
                }
            };

            // Accumulate gradients in input nodes
            for (i, &input_id) in node_inputs.iter().enumerate() {
                if input_id >= self.nodes.len() {
                    return Err(format!("Invalid input node ID: {}", input_id));
                }

                let input_grad = &input_grads[i];
                
                match &mut self.nodes[input_id].grad {
                    Some(existing_grad) => {
                        // Validate shapes match before accumulation
                        if existing_grad.shape != input_grad.shape {
                            return Err(format!(
                                "Gradient accumulation shape mismatch at node {}: existing gradient shape {:?} vs new gradient shape {:?}. \
                                This indicates multiple paths are contributing gradients with incompatible shapes. \
                                Check that all operations in the backward pass compute gradients with consistent shapes.",
                                input_id, existing_grad.shape, input_grad.shape
                            ));
                        }
                        
                        // Accumulate: add the new gradient to existing one
                        existing_grad.add_assign(input_grad)?;
                    }
                    None => {
                        self.nodes[input_id].grad = Some(input_grad.clone());
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the gradient of a node (after backward pass)
    pub fn get_gradient(&self, node_id: NodeId) -> Result<Tensor, String> {
        if node_id >= self.nodes.len() {
            return Err(format!("Invalid node ID: {}", node_id));
        }

        self.nodes[node_id]
            .grad
            .as_ref()
            .cloned()
            .ok_or_else(|| format!("Node {} has no gradient. Run backward() first.", node_id))
    }

    /// Zero all gradients in the graph
    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            node.grad = None;
        }
    }

    /// Set whether a node requires gradients
    pub fn set_requires_grad(&mut self, node_id: NodeId, requires_grad: bool) -> Result<(), String> {
        if node_id >= self.nodes.len() {
            return Err(format!("Invalid node ID: {}", node_id));
        }
        self.nodes[node_id].requires_grad = requires_grad;
        Ok(())
    }

    /// Clear all non-parameter nodes from the graph, keeping only parameter nodes
    /// This prevents memory leaks by removing temporary computation nodes after backward pass
    /// 
    /// # Arguments
    /// * `param_node_ids` - Slice of node IDs that represent parameters (weights, biases) to preserve
    /// 
    /// # Returns
    /// Returns a vector of new node IDs for the preserved parameter nodes (mapped to their new positions)
    pub fn clear_non_parameter_nodes(&mut self, param_node_ids: &[NodeId]) -> Result<Vec<NodeId>, String> {
        use std::collections::HashSet;
        
        // Create set of parameter node IDs for fast lookup
        let param_set: HashSet<NodeId> = param_node_ids.iter().cloned().collect();
        
        // Validate that all parameter node IDs are valid
        for &param_id in param_node_ids {
            if param_id >= self.nodes.len() {
                return Err(format!("Invalid parameter node ID: {} (graph has {} nodes)", param_id, self.nodes.len()));
            }
        }
        
        // Build new nodes vector containing only parameter nodes
        let mut new_nodes = Vec::new();
        let mut new_param_ids = Vec::new();
        
        // Copy only parameter nodes and build mapping
        for (old_id, node) in self.nodes.iter().enumerate() {
            if param_set.contains(&old_id) {
                let new_id = new_nodes.len();
                new_nodes.push(node.clone());
                new_param_ids.push(new_id);
            }
        }
        
        // Verify we found all parameters
        if new_param_ids.len() != param_node_ids.len() {
            return Err(format!(
                "Mismatch: expected {} parameter nodes, found {}",
                param_node_ids.len(),
                new_param_ids.len()
            ));
        }
        
        // Update nodes
        self.nodes = new_nodes;
        
        // Update input_nodes to only include parameter nodes (they're all Input nodes)
        self.input_nodes = new_param_ids.clone();
        
        // Clear gradients from parameter nodes (they'll be recomputed in next backward pass)
        // But preserve values - they contain the updated parameter values after optimizer step
        for node in &mut self.nodes {
            node.grad = None;
            // Note: node.value is preserved - it contains the updated parameter values
        }
        
        Ok(new_param_ids)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

