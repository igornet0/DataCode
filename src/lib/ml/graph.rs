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
    
    /// Count the number of GPU tensors in the graph (for debugging memory issues)
    #[allow(unused_mut, unused_variables)]
    pub fn count_gpu_tensors(&self) -> usize {
        let mut count = 0;
        #[cfg(feature = "gpu")]
        {
            for node in &self.nodes {
                if let Some(ref value) = node.value {
                    if value.gpu_tensor.is_some() {
                        count += 1;
                    }
                }
                if let Some(ref grad) = node.grad {
                    if grad.gpu_tensor.is_some() {
                        count += 1;
                    }
                }
            }
        }
        count
    }
    
    /// Count GPU tensors including parameter cache
    #[allow(unused_variables, unused_mut)] // param_cache and mut only used when gpu feature is enabled
    pub fn count_gpu_tensors_with_cache(&self, param_cache: &std::collections::HashMap<NodeId, Tensor>) -> usize {
        let mut count = self.count_gpu_tensors();
        
        #[cfg(feature = "gpu")]
        {
            // Count GPU tensors from cache that are not in graph nodes
            for (node_id, tensor) in param_cache {
                if *node_id < self.nodes.len() {
                    // Check if node has no value or value has no GPU buffer
                    let node_has_gpu = self.nodes[*node_id]
                        .value
                        .as_ref()
                        .map(|v| v.has_gpu_buffer())
                        .unwrap_or(false);
                    
                    if !node_has_gpu && tensor.has_gpu_buffer() {
                        count += 1;
                    }
                } else {
                    // Node doesn't exist in graph, count cache tensor if it has GPU buffer
                    if tensor.has_gpu_buffer() {
                        count += 1;
                    }
                }
            }
        }
        
        count
    }
    
    /// Calculate total memory usage of all tensors in the graph (CPU memory in bytes)
    pub fn total_memory_bytes(&self) -> usize {
        let mut total = 0;
        for node in &self.nodes {
            if let Some(ref value) = node.value {
                total += value.memory_size_bytes();
            }
            if let Some(ref grad) = node.grad {
                total += grad.memory_size_bytes();
            }
        }
        total
    }
    
    /// Calculate total memory usage including parameter cache
    /// This gives accurate memory usage after cleanup when parameters are stored in cache
    pub fn total_memory_bytes_with_cache(&self, param_cache: &std::collections::HashMap<NodeId, Tensor>) -> usize {
        let mut total = self.total_memory_bytes();
        
        // Add memory from parameter cache for parameters not in graph nodes
        for (node_id, tensor) in param_cache {
            // Only count if this parameter is not already counted in graph nodes
            // After cleanup, parameters remain in graph but may have empty values
            // So we check if the node exists and has no value, then add cache memory
            if *node_id < self.nodes.len() {
                if self.nodes[*node_id].value.is_none() {
                    total += tensor.memory_size_bytes();
                }
            } else {
                // Node ID doesn't exist in graph (shouldn't happen, but be safe)
                total += tensor.memory_size_bytes();
            }
        }
        
        total
    }
    
    /// Count total number of tensors (values + gradients) in the graph
    pub fn count_tensors(&self) -> usize {
        let mut count = 0;
        for node in &self.nodes {
            if node.value.is_some() {
                count += 1;
            }
            if node.grad.is_some() {
                count += 1;
            }
        }
        count
    }
    
    /// Log memory diagnostics for the graph
    // pub fn log_memory_diagnostics(&self, context: &str) {
    //     let total_memory = self.total_memory_bytes();
    //     let tensor_count = self.count_tensors();
    //     let gpu_tensor_count = self.count_gpu_tensors();
    //     let node_count = self.nodes.len();
        
    //     eprintln!(
    //         "[DIAG] Graph memory ({}) - Nodes: {}, Tensors: {}, GPU tensors: {}, Memory: {} bytes ({} MB)",
    //         context,
    //         node_count,
    //         tensor_count,
    //         gpu_tensor_count,
    //         total_memory,
    //         total_memory / (1024 * 1024)
    //     );
    // }
    
    /// Log memory diagnostics including parameter cache
    // pub fn log_memory_diagnostics_with_cache(&self, context: &str, param_cache: &std::collections::HashMap<NodeId, Tensor>) {
    //     let total_memory = self.total_memory_bytes_with_cache(param_cache);
    //     let tensor_count = self.count_tensors();
    //     let gpu_tensor_count = self.count_gpu_tensors_with_cache(param_cache);
    //     let node_count = self.nodes.len();
        
    //     eprintln!(
    //         "[DIAG] Graph memory ({}) - Nodes: {}, Tensors: {}, GPU tensors: {}, Memory: {} bytes ({} MB) [with cache]",
    //         context,
    //         node_count,
    //         tensor_count,
    //         gpu_tensor_count,
    //         total_memory,
    //         total_memory / (1024 * 1024)
    //     );
    // }
    

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
        #[cfg(debug_assertions)]
        let _input_nodes_count = self.input_nodes.len();
        
        // DIAG: Log memory usage at start of forward pass
        let _memory_before = self.total_memory_bytes();
        let _tensor_count_before = self.count_tensors();
        
        // Save input count for error messages (before moving input_tensors)
        let input_tensors_count = input_tensors.len();
        
        
        // DIAG: Log memory diagnostics periodically
        // if graph_size_at_start % 50 == 0 || graph_size_at_start > 100 {
        //     self.log_memory_diagnostics("forward_start");
        // }
        
        // Validate input count
        if input_tensors_count != self.input_nodes.len() {
            eprintln!("[ERROR] Input count mismatch in graph.forward()!");
            return Err(format!(
                "Expected {} input tensors, got {}",
                self.input_nodes.len(),
                input_tensors_count
            ));
        }

        // Clear previous values and gradients
        for node in &mut self.nodes {
            node.value = None;
            node.grad = None;
        }

        // Set input values
        // OPTIMIZATION: Use move semantics to avoid cloning
        // Move tensors from input_tensors vector directly into graph nodes
        for (tensor, &input_node_id) in input_tensors.into_iter().zip(self.input_nodes.iter()) {
            self.nodes[input_node_id].value = Some(tensor);
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
                            eprintln!("[ERROR]   Input tensors provided: {}", input_tensors_count);
                            
                            format!(
                                "Input node {} has no value. Node {} references it. \
                                Node op: {}. Is input node: {}. \
                                Referencing nodes: {:?}. Input nodes: {:?}. Execution order: {:?}. \
                                Total nodes: {}. Input tensors provided: {}",
                                id, node_id, node_op, is_input_node, referencing_nodes, 
                                self.input_nodes, execution_order, self.nodes.len(), input_tensors_count
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
                            let broadcasted = inputs[1].broadcast_to(inputs[0].shape())?;
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
                    let result = inputs[0].matmul(inputs[1])?;
                    
                    // OPTIMIZATION: Removed synchronization after forward MatMul - operations are asynchronous
                    // Synchronization will happen after backward pass completes
                    
                    result
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
        
        // DIAG: Log memory usage at end of forward pass
        let _memory_after = self.total_memory_bytes();
        let _tensor_count_after = self.count_tensors();
        let _memory_delta = _memory_after.saturating_sub(_memory_before);

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
        let _graph_size = self.nodes.len();
        
        // Build adjacency list and in-degree count
        let mut in_degree = vec![0; self.nodes.len()];
        let mut adjacency: Vec<Vec<NodeId>> = vec![Vec::new(); self.nodes.len()];
        
        // DIAGNOSTIC: Count node types
        let mut node_type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut nodes_with_stale_refs = Vec::new();

        for (node_id, node) in self.nodes.iter().enumerate() {
            let op_type = format!("{:?}", node.op);
            *node_type_counts.entry(op_type).or_insert(0) += 1;
            
            for &input_id in &node.inputs {
                // Check for stale references
                if input_id >= self.nodes.len() {
                    nodes_with_stale_refs.push((node_id, input_id));
                }
                adjacency[input_id].push(node_id);
                in_degree[node_id] += 1;
            }
        }
        
        
        // DIAGNOSTIC: Warn about stale references
        #[cfg(debug_assertions)]
        if !nodes_with_stale_refs.is_empty() {
            eprintln!(
                "[ERROR] topological_sort: Found {} nodes with stale references to non-existent nodes: {:?}",
                nodes_with_stale_refs.len(),
                nodes_with_stale_refs.iter().take(10).collect::<Vec<_>>()
            );
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
            eprintln!(
                "[ERROR] topological_sort: Graph contains cycles! Expected {} nodes, got {} in result",
                self.nodes.len(), result.len()
            );
            eprintln!(
                "[ERROR] Missing nodes: {:?}",
                (0..self.nodes.len())
                    .filter(|&id| !result.contains(&id))
                    .collect::<Vec<_>>()
            );
            return Err("Graph contains cycles".to_string());
        }

        Ok(result)
    }

    /// Execute backward pass to compute gradients
    /// output_node_id: The node from which to start backpropagation (typically the loss)
    pub fn backward(&mut self, output_node_id: NodeId) -> Result<(), String> {
        // OPTIMIZATION: Profile backward pass start
        let _backward_start_time = std::time::Instant::now();
        
        // OPTIMIZATION: Track profiling data for individual operations
        use std::collections::HashMap;
        let mut op_times: HashMap<String, (u128, usize)> = HashMap::new(); // (total_time_ms, count)
        #[cfg(feature = "gpu")]
        let mut _temp_gpu_buffers_created = 0usize;
        #[cfg(not(feature = "gpu"))]
        #[allow(unused_variables)]
        let _temp_gpu_buffers_created = 0usize;
        
        #[cfg(debug_assertions)]
        let _input_nodes_count = self.input_nodes.len();
        
        // DIAG: Log memory usage at start of backward pass
        let _memory_before = self.total_memory_bytes();
        let _tensor_count_before = self.count_tensors();
        
        // OPTIMIZATION: Track GPU tensors at start for profiling
        #[cfg(feature = "gpu")]
        let _gpu_tensors_before = self.count_gpu_tensors();
        #[cfg(not(feature = "gpu"))]
        let _gpu_tensors_before = 0;
        
        
        if output_node_id >= self.nodes.len() {
            eprintln!(
                "[ERROR] graph.backward: Invalid output_node_id {} (graph has {} nodes)",
                output_node_id, self.nodes.len()
            );
            return Err(format!("Invalid output node ID: {}", output_node_id));
        }

        // Ensure forward pass has been run
        if self.nodes[output_node_id].value.is_none() {
            eprintln!(
                "[ERROR] graph.backward: Output node {} has no value (forward pass not run?)",
                output_node_id
            );
            return Err("Forward pass must be run before backward pass".to_string());
        }

        // Initialize output gradient to ones ONLY if not already set
        // This allows manual gradient setting (e.g., for sparse_cross_entropy) to work correctly
        if self.nodes[output_node_id].grad.is_none() {
            let output_shape = self.nodes[output_node_id].value.as_ref().unwrap().shape.clone();
            self.nodes[output_node_id].grad = Some(Tensor::ones(output_shape));
        }

        // Get reverse topological order (for backward pass)
        // #region agent log
        let topo_sort_start = std::time::Instant::now();
        // #endregion
        let forward_order = self.topological_sort()?;
        // #region agent log
        let topo_sort_time = topo_sort_start.elapsed();
        let log_data_topo = format!(r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"graph.rs:{}","message":"Topological sort timing","data":{{"topo_sort_time_ms":{},"graph_size":{}}},"timestamp":{}}}"#, 
            line!(), topo_sort_time.as_millis(), self.nodes.len(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/igor/Desktop/Projects/DataCode/.cursor/debug.log") {
            use std::io::Write;
            let _ = writeln!(file, "{}", log_data_topo);
        }
        // #endregion
        let mut backward_order = forward_order.clone();
        backward_order.reverse();
        
        // ENHANCED LOGGING: Track nodes processed during backward
        let mut nodes_processed = 0;

        // Process nodes in reverse topological order
        for &node_id in &backward_order {
            nodes_processed += 1;
            
            // OPTIMIZATION: Periodic cleanup of intermediate GPU tensors during backward pass
            // This prevents GPU memory accumulation that causes performance degradation
            // Cleanup every 5 nodes for more aggressive memory management
            // Reduced from 10 to 5 based on analysis showing degradation in epoch 4
            #[cfg(feature = "gpu")]
            if nodes_processed % 5 == 0 {
                // Clear GPU buffers from intermediate gradients (not parameter values)
                // This helps Metal free memory incrementally during backward pass
                for (idx, node) in self.nodes.iter_mut().enumerate() {
                    // Only clear gradients from non-parameter nodes
                    // Parameters should keep their GPU buffers for next forward pass
                    if !self.input_nodes.contains(&idx) {
                        if let Some(ref mut grad) = node.grad {
                            if grad.device().is_gpu() {
                                // Clear GPU buffer from intermediate gradient
                                // It will be recreated if needed
                                #[cfg(feature = "gpu")]
                                {
                                    grad.gpu_tensor = None;
                                }
                            }
                        }
                    }
                }
                
                // CRITICAL: Synchronize GPU after periodic cleanup to ensure buffers are freed
                // This helps Metal free accumulated command buffers more effectively
                if self.device.is_gpu() {
                    if let Err(e) = self.device.synchronize() {
                        eprintln!("[WARNING] Failed to synchronize GPU during backward pass periodic cleanup (node {}): {}", node_id, e);
                    }
                }
            }
            
            // ENHANCED LOGGING: Check graph size during backward (every 10 nodes)
            
            // Skip if node has no gradient or doesn't require grad
            // OPTIMIZATION: Try to avoid cloning gradient if possible
            // We need a reference to check, but then we can work with it directly
            let grad_ref = match &self.nodes[node_id].grad {
                Some(g) => g,
                None => continue,
            };
            
            // OPTIMIZATION: Only clone gradient when we actually need to modify it
            // For operations that don't modify the gradient (like Add where shapes match),
            // we can work with references. Clone only when necessary.
            // We'll clone lazily in each operation branch where needed.

            // OPTIMIZATION: Clone node inputs (small Vec<NodeId>) to avoid borrow checker issues
            // This is acceptable since Vec<NodeId> is small (just indices)
            // The main optimization is avoiding unnecessary tensor clones
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
            // OPTIMIZATION: Profile individual operation time
            let op_start = std::time::Instant::now();
            let op_type_str = format!("{:?}", node_op);
            #[cfg(feature = "gpu")]
            let op_gpu_buffers_before = self.count_gpu_tensors();
            #[cfg(not(feature = "gpu"))]
            #[allow(unused_variables)]
            let op_gpu_buffers_before = 0;
            
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
                    // OPTIMIZATION: Avoid unnecessary clones - only clone when shapes don't match
                    let grad_shape = &grad_ref.shape;
                    let input0_shape = &input_values[0].shape;
                    let input1_shape = &input_values[1].shape;
                    
                    // Check if we can reuse the same gradient tensor for both inputs
                    // CRITICAL OPTIMIZATION: Minimize cloning to reduce Metal buffer creation
                    let (grad_0, grad_1) = if grad_shape == input0_shape && grad_shape == input1_shape {
                        // All shapes match - we can reuse the same gradient for both
                        // OPTIMIZATION: Clone only once and reuse (avoid double clone)
                        let grad_clone = grad_ref.clone();
                        (grad_clone.clone(), grad_clone)
                    } else if grad_shape == input0_shape {
                        // First input matches, second needs summing
                        // OPTIMIZATION: Clone only once, reuse for first input
                        let grad_clone = grad_ref.clone();
                        (grad_clone, grad_ref.sum_to_shape(input1_shape)?)
                    } else if grad_shape == input1_shape {
                        // Second input matches, first needs summing
                        // OPTIMIZATION: Clone only once, reuse for second input
                        let grad_clone = grad_ref.clone();
                        (grad_ref.sum_to_shape(input0_shape)?, grad_clone)
                    } else {
                        // Both need summing - compute both (no cloning needed)
                        (grad_ref.sum_to_shape(input0_shape)?, grad_ref.sum_to_shape(input1_shape)?)
                    };
                    
                    // OPTIMIZATION: Removed synchronization after Add backward - operations are asynchronous
                    // Synchronization will happen after backward pass completes
                    
                    vec![grad_0, grad_1]
                }
                OpType::Sub => {
                    if input_values.len() != 2 {
                        return Err("Sub operation requires 2 inputs".to_string());
                    }
                    // grad_a = grad, grad_b = -grad (with broadcasting handling)
                    // OPTIMIZATION: Avoid unnecessary clones - compute neg_grad only when needed
                    let grad_shape = &grad_ref.shape;
                    let input0_shape = &input_values[0].shape;
                    let input1_shape = &input_values[1].shape;
                    
                    // Compute grad_0 (grad for first input)
                    let grad_0 = if grad_shape == input0_shape {
                        grad_ref.clone()
                    } else {
                        grad_ref.sum_to_shape(input0_shape)?
                    };
                    
                    // Compute grad_1 (neg_grad for second input)
                    // Only compute negation if needed (when shape matches or needs summing)
                    let grad_1 = if grad_shape == input1_shape {
                        // Shape matches - can negate directly
                        grad_ref.neg()
                    } else {
                        // Need to sum first, then negate
                        let summed = grad_ref.sum_to_shape(input1_shape)?;
                        summed.neg()
                    };
                    
                    vec![grad_0, grad_1]
                }
                OpType::Mul => {
                    if input_values.len() != 2 {
                        return Err("Mul operation requires 2 inputs".to_string());
                    }
                    // grad_a = grad * b, grad_b = grad * a (with broadcasting handling)
                    // OPTIMIZATION: Avoid unnecessary clones and intermediate tensors
                    let grad_shape = &grad_ref.shape;
                    let input0_shape = &input_values[0].shape;
                    let input1_shape = &input_values[1].shape;
                    
                    // For grad_a: multiply grad by input_values[1] (second input)
                    // Check if broadcasting is needed
                    let grad_times_b = if grad_shape == input1_shape {
                        // Shapes match - direct multiplication
                        grad_ref.mul(input_values[1])?
                    } else {
                        // Need broadcasting - compute efficiently
                        let broadcasted_grad = grad_ref.broadcast_to(input1_shape)?;
                        broadcasted_grad.mul(input_values[1])?
                    };
                    
                    // For grad_b: multiply grad by input_values[0] (first input)
                    let grad_times_a = if grad_shape == input0_shape {
                        // Shapes match - direct multiplication
                        grad_ref.mul(input_values[0])?
                    } else {
                        // Need broadcasting
                        let broadcasted_grad = grad_ref.broadcast_to(input0_shape)?;
                        broadcasted_grad.mul(input_values[0])?
                    };
                    
                    // Now sum to input shapes if they were broadcasted
                    let grad_0 = if grad_times_b.shape == *input0_shape {
                        grad_times_b
                    } else {
                        grad_times_b.sum_to_shape(input0_shape)?
                    };
                    
                    let grad_1 = if grad_times_a.shape == *input1_shape {
                        grad_times_a
                    } else {
                        grad_times_a.sum_to_shape(input1_shape)?
                    };
                    
                    // OPTIMIZATION: Removed synchronization after Mul backward - operations are asynchronous
                    // Synchronization will happen after backward pass completes
                    
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
                    // CRITICAL OPTIMIZATION: Work on GPU if all tensors are on GPU
                    #[cfg(feature = "gpu")]
                    {
                        let grad_on_gpu = grad_ref.device().is_gpu() && grad_ref.gpu_tensor.is_some();
                        let a_on_gpu = input_values[0].device().is_gpu() && input_values[0].gpu_tensor.is_some();
                        let b_on_gpu = input_values[1].device().is_gpu() && input_values[1].gpu_tensor.is_some();
                        
                        // #region agent log
                        let log_data = format!(r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"graph.rs:{}","message":"MatMul backward GPU check","data":{{"grad_on_gpu":{},"a_on_gpu":{},"b_on_gpu":{},"all_on_gpu":{}}},"timestamp":{}}}"#, 
                            line!(), grad_on_gpu, a_on_gpu, b_on_gpu, grad_on_gpu && a_on_gpu && b_on_gpu,
                            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
                        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/igor/Desktop/Projects/DataCode/.cursor/debug.log") {
                            use std::io::Write;
                            let _ = writeln!(file, "{}", log_data);
                        }
                        // #endregion
                        
                        if grad_on_gpu && a_on_gpu && b_on_gpu {
                            // All tensors are on GPU - use GPU operations
                            // OPTIMIZATION: Removed synchronization before MatMul - operations are asynchronous
                            // Synchronization will happen after backward pass completes
                            
                            // Get GPU tensors and clone device before any operations
                            let grad_gpu = grad_ref.gpu_tensor.as_ref().unwrap();
                            let a_gpu = input_values[0].gpu_tensor.as_ref().unwrap();
                            let b_gpu = input_values[1].gpu_tensor.as_ref().unwrap();
                            let grad_device = grad_ref.device().clone(); // Clone to avoid borrow conflicts
                            
                            // Compute transpose on GPU
                            let b_t_gpu = b_gpu.transpose(0, 1)
                                .map_err(|e| format!("Failed to transpose b on GPU: {}", e))?;
                            let a_t_gpu = a_gpu.transpose(0, 1)
                                .map_err(|e| format!("Failed to transpose a on GPU: {}", e))?;
                            
                            // Compute gradients on GPU: grad_a = grad @ b^T, grad_b = a^T @ grad
                            // OPTIMIZATION: Profile each MatMul operation for detailed performance tracking
                            let _grad_shape = grad_gpu.dims();
                            let _b_t_shape = b_t_gpu.dims();
                            
                            let matmul_a_start = std::time::Instant::now();
                            let grad_a_gpu = grad_gpu.matmul(&b_t_gpu)
                                .map_err(|e| format!("Failed to compute grad_a on GPU: {}", e))?;
                            let _matmul_a_time = matmul_a_start.elapsed();
                            
                            // CRITICAL: Synchronize after first MatMul to prevent command buffer accumulation
                            // MatMul operations are the main source of command buffer accumulation
                            // Analysis shows MatMul slows down from 5-6ms to 23-31ms in epoch 5 without sync
                            if self.device.is_gpu() {
                                if let Err(e) = self.device.synchronize() {
                                    eprintln!("[WARNING] Failed to synchronize GPU after MatMul grad_a: {}", e);
                                }
                            }
                            
                            // Log detailed MatMul profiling
                            // eprintln!(
                            //     "[PROFILE] MatMul backward grad_a: time={}ms, shapes: grad={:?} @ b_t={:?} -> {:?}",
                            //     matmul_a_time.as_millis(),
                            //     grad_shape, b_t_shape, grad_a_gpu.dims()
                            // );
                            
                            let _a_t_shape = a_t_gpu.dims();
                            let matmul_b_start = std::time::Instant::now();
                            let grad_b_gpu = a_t_gpu.matmul(grad_gpu)
                                .map_err(|e| format!("Failed to compute grad_b on GPU: {}", e))?;
                            let _matmul_b_time = matmul_b_start.elapsed();
                            
                            // CRITICAL: Synchronize after second MatMul to prevent command buffer accumulation
                            // Both MatMul operations need synchronization to prevent accumulation
                            // Analysis shows degradation in epochs 4-5 without MatMul synchronization
                            if self.device.is_gpu() {
                                if let Err(e) = self.device.synchronize() {
                                    eprintln!("[WARNING] Failed to synchronize GPU after MatMul grad_b: {}", e);
                                }
                            }
                            
                            // Log detailed MatMul profiling
                            // eprintln!(
                            //     "[PROFILE] MatMul backward grad_b: time={}ms, shapes: a_t={:?} @ grad={:?} -> {:?}",
                            //     matmul_b_time.as_millis(),
                            //     a_t_shape, grad_shape, grad_b_gpu.dims()
                            // );
                            
                            // Get shapes
                            let grad_a_shape = grad_a_gpu.dims();
                            let grad_b_shape = grad_b_gpu.dims();
                            
                            // Create Tensor wrappers keeping GPU buffers
                            let grad_a = Tensor::from_gpu_tensor(
                                grad_a_shape.to_vec(),
                                grad_device.clone(),
                                Some(grad_a_gpu),
                            );
                            
                            let grad_b = Tensor::from_gpu_tensor(
                                grad_b_shape.to_vec(),
                                grad_device,
                                Some(grad_b_gpu),
                            );
                            
                            vec![grad_a, grad_b]
                        } else {
                            // Fallback to CPU (when GPU feature is enabled but tensors are not on GPU)
                            let grad_cpu = grad_ref.to_cpu()?;
                            
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
                            
                            // Compute gradients with error handling
                            let grad_a = grad_cpu.matmul(&b_t).map_err(|e| {
                                format!("MatMul backward: Failed to compute grad_a: {}", e)
                            })?;
                            
                            let grad_b = a_t.matmul(&grad_cpu).map_err(|e| {
                                format!("MatMul backward: Failed to compute grad_b: {}", e)
                            })?;
                            
                            vec![grad_a, grad_b]
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        // No GPU feature - CPU only
                        let grad_cpu = grad_ref.to_cpu()?;
                        let a_shape = input_values[0].shape.clone();
                        let b_shape = input_values[1].shape.clone();
                        let grad_shape = grad_cpu.shape.clone();
                        
                        if a_shape.len() != 2 || b_shape.len() != 2 || grad_shape.len() != 2 {
                            return Err(format!(
                                "MatMul backward: All tensors must be 2D. Got a: {:?}, b: {:?}, grad: {:?}",
                                a_shape, b_shape, grad_shape
                            ));
                        }
                        
                        if a_shape[1] != b_shape[0] {
                            return Err(format!(
                                "MatMul backward: Incompatible forward shapes. a: {:?}, b: {:?}. Expected a[1] == b[0]",
                                a_shape, b_shape
                            ));
                        }
                        
                        if grad_shape[0] != a_shape[0] || grad_shape[1] != b_shape[1] {
                            return Err(format!(
                                "MatMul backward: Gradient shape {:?} does not match expected output shape [{}, {}] from forward pass (a: {:?}, b: {:?})",
                                grad_shape, a_shape[0], b_shape[1], a_shape, b_shape
                            ));
                        }
                        
                        let b_t = input_values[1].transpose()?;
                        let a_t = input_values[0].transpose()?;
                        
                        let grad_a = grad_cpu.matmul(&b_t).map_err(|e| {
                            format!("MatMul backward: Failed to compute grad_a: {}", e)
                        })?;
                        
                        let grad_b = a_t.matmul(&grad_cpu).map_err(|e| {
                            format!("MatMul backward: Failed to compute grad_b: {}", e)
                        })?;
                        
                        vec![grad_a, grad_b]
                    }
                }
                OpType::Transpose => {
                    if input_values.len() != 1 {
                        return Err("Transpose operation requires 1 input".to_string());
                    }
                    // grad_input = grad^T
                    vec![grad_ref.transpose()?]
                }
                OpType::Sum => {
                    if input_values.len() != 1 {
                        return Err("Sum operation requires 1 input".to_string());
                    }
                    // grad_input = broadcast(grad, input_shape)
                    // grad is scalar [1], need to broadcast to input shape
                    // OPTIMIZATION: Use reference directly, only convert to CPU when needed
                    let grad_cpu = grad_ref.to_cpu()?;
                    let grad_val = grad_cpu.data[0];
                    let input_shape = &input_values[0].shape;
                    let total_size: usize = input_shape.iter().product();
                    // OPTIMIZATION: Use Vec::with_capacity and fill more efficiently
                    let mut grad_data = Vec::with_capacity(total_size);
                    grad_data.resize(total_size, grad_val);
                    vec![Tensor::new(grad_data, input_shape.clone())?]
                }
                OpType::Mean => {
                    if input_values.len() != 1 {
                        return Err("Mean operation requires 1 input".to_string());
                    }
                    // grad_input = broadcast(grad / n, input_shape)
                    // OPTIMIZATION: Use reference directly, only convert to CPU when needed
                    let grad_cpu = grad_ref.to_cpu()?;
                    let input_shape = &input_values[0].shape;
                    let n = input_values[0].total_size() as f32;
                    let grad_val = grad_cpu.data[0] / n;
                    let total_size: usize = input_shape.iter().product();
                    // OPTIMIZATION: Use Vec::with_capacity and fill more efficiently
                    let mut grad_data = Vec::with_capacity(total_size);
                    grad_data.resize(total_size, grad_val);
                    vec![Tensor::new(grad_data, input_shape.clone())?]
                }
                OpType::ReLU => {
                    if input_values.len() != 1 {
                        return Err("ReLU operation requires 1 input".to_string());
                    }
                    // grad = grad * (x > 0)
                    // CRITICAL OPTIMIZATION: Work on GPU if tensors are on GPU
                    #[cfg(feature = "gpu")]
                    {
                        let grad_on_gpu = grad_ref.device().is_gpu() && grad_ref.gpu_tensor.is_some();
                        let input_on_gpu = input_values[0].device().is_gpu() && input_values[0].gpu_tensor.is_some();
                        
                        if grad_on_gpu && input_on_gpu {
                            // OPTIMIZATION: All tensors are on GPU - use GPU operations
                            // Minimize intermediate tensors to reduce GPU memory pressure
                            let grad_gpu = grad_ref.gpu_tensor.as_ref().unwrap();
                            let input_gpu = input_values[0].gpu_tensor.as_ref().unwrap();
                            
                            // OPTIMIZATION: Create mask and compute result in fewer steps
                            // Use where_cond for efficient conditional operation
                            let zero_tensor = candle_core::Tensor::zeros_like(input_gpu)
                                .map_err(|e| format!("Failed to create zero tensor on GPU: {}", e))?;
                            
                            // OPTIMIZATION: Use where_cond to compute grad * (x > 0) efficiently
                            // This avoids creating intermediate mask_f32 tensor
                            let mask = input_gpu.gt(&zero_tensor)
                                .map_err(|e| format!("Failed to create mask on GPU: {}", e))?;
                            
                            // OPTIMIZATION: Use where_cond instead of converting mask and multiplying
                            // This is more efficient and uses fewer intermediate tensors
                            let result_gpu = candle_core::Tensor::where_cond(
                                &mask,
                                grad_gpu,
                                &zero_tensor
                            )
                            .map_err(|e| format!("Failed to compute ReLU backward on GPU: {}", e))?;
                            
                            // OPTIMIZATION: Explicitly drop intermediate tensors to free GPU memory immediately
                            // This helps prevent GPU memory accumulation that causes performance degradation
                            drop(zero_tensor);
                            drop(mask);
                            
                            // Get result shape
                            let result_shape = result_gpu.dims();
                            
                            // Create Tensor wrapper keeping GPU buffer
                            vec![Tensor::from_gpu_tensor(
                                result_shape.to_vec(),
                                grad_ref.device().clone(),
                                Some(result_gpu),
                            )]
                        } else {
                            // Fallback to CPU
                            let grad_cpu = grad_ref.to_cpu()?;
                            let input_cpu = input_values[0].to_cpu()?;
                            let grad_data: Vec<f32> = grad_cpu.data
                                .iter()
                                .zip(input_cpu.data.iter())
                                .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                                .collect();
                            vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        // CPU only
                        let grad_cpu = grad_ref.to_cpu()?;
                        let input_cpu = input_values[0].to_cpu()?;
                        let grad_data: Vec<f32> = grad_cpu.data
                            .iter()
                            .zip(input_cpu.data.iter())
                            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                            .collect();
                        vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                    }
                }
                OpType::Sigmoid => {
                    if input_values.len() != 1 {
                        return Err("Sigmoid operation requires 1 input".to_string());
                    }
                    // grad = grad * sigmoid(x) * (1 - sigmoid(x))
                    // OPTIMIZATION: Work on GPU if tensors are on GPU
                    #[cfg(feature = "gpu")]
                    {
                        let grad_on_gpu = grad_ref.device().is_gpu() && grad_ref.gpu_tensor.is_some();
                        let input_on_gpu = input_values[0].device().is_gpu() && input_values[0].gpu_tensor.is_some();
                        
                        if grad_on_gpu && input_on_gpu {
                            // OPTIMIZATION: All tensors are on GPU - use GPU operations
                            let grad_gpu = grad_ref.gpu_tensor.as_ref().unwrap();
                            let input_gpu = input_values[0].gpu_tensor.as_ref().unwrap();
                            
                            // Compute sigmoid on GPU: sigmoid(x) = 1 / (1 + exp(-x))
                            // Use exp and division operations
                            let neg_input = input_gpu.neg()
                                .map_err(|e| format!("Failed to negate input on GPU: {}", e))?;
                            let exp_neg = neg_input.exp()
                                .map_err(|e| format!("Failed to compute exp(-x) on GPU: {}", e))?;
                            let ones = candle_core::Tensor::ones_like(&exp_neg)
                                .map_err(|e| format!("Failed to create ones tensor on GPU: {}", e))?;
                            let one_plus_exp = (&ones + &exp_neg)
                                .map_err(|e| format!("Failed to compute (1+exp(-x)) on GPU: {}", e))?;
                            let sigmoid_gpu = (&ones / &one_plus_exp)
                                .map_err(|e| format!("Failed to compute sigmoid on GPU: {}", e))?;
                            
                            // Compute (1 - sigmoid) on GPU
                            let one_minus_sigmoid = (&ones - &sigmoid_gpu)
                                .map_err(|e| format!("Failed to compute (1-sigmoid) on GPU: {}", e))?;
                            
                            // Compute grad * sigmoid * (1 - sigmoid) on GPU
                            let result_gpu = (grad_gpu * &sigmoid_gpu * &one_minus_sigmoid)
                                .map_err(|e| format!("Failed to compute sigmoid backward on GPU: {}", e))?;
                            
                            // OPTIMIZATION: Explicitly drop intermediate tensors
                            drop(neg_input);
                            drop(exp_neg);
                            drop(one_plus_exp);
                            
                            // OPTIMIZATION: Explicitly drop intermediate tensors
                            drop(sigmoid_gpu);
                            drop(ones);
                            drop(one_minus_sigmoid);
                            
                            let result_shape = result_gpu.dims();
                            vec![Tensor::from_gpu_tensor(
                                result_shape.to_vec(),
                                grad_ref.device().clone(),
                                Some(result_gpu),
                            )]
                        } else {
                            // Fallback to CPU
                            let grad_cpu = grad_ref.to_cpu()?;
                            let input_cpu = input_values[0].to_cpu()?;
                            let sigmoid_output = input_cpu.sigmoid();
                            let grad_data: Vec<f32> = grad_cpu.data
                                .iter()
                                .zip(sigmoid_output.data.iter())
                                .map(|(&g, &s)| g * s * (1.0 - s))
                                .collect();
                            vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        // CPU only
                        let grad_cpu = grad_ref.to_cpu()?;
                        let input_cpu = input_values[0].to_cpu()?;
                        let sigmoid_output = input_cpu.sigmoid();
                        let grad_data: Vec<f32> = grad_cpu.data
                            .iter()
                            .zip(sigmoid_output.data.iter())
                            .map(|(&g, &s)| g * s * (1.0 - s))
                            .collect();
                        vec![Tensor::new(grad_data, grad_cpu.shape.clone())?]
                    }
                }
                OpType::Tanh => {
                    if input_values.len() != 1 {
                        return Err("Tanh operation requires 1 input".to_string());
                    }
                    // grad = grad * (1 - tanh²(x))
                    // OPTIMIZATION: Use reference directly, only convert to CPU when needed
                    let grad_cpu = grad_ref.to_cpu()?;
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
                    // Softmax gradient: grad = softmax * (grad - sum(grad * softmax))
                    // OPTIMIZATION: Work on GPU if tensors are on GPU
                    #[cfg(feature = "gpu")]
                    {
                        let grad_on_gpu = grad_ref.device().is_gpu() && grad_ref.gpu_tensor.is_some();
                        let input_on_gpu = input_values[0].device().is_gpu() && input_values[0].gpu_tensor.is_some();
                        
                        if grad_on_gpu && input_on_gpu {
                            // OPTIMIZATION: All tensors are on GPU - use GPU operations via Candle
                            use candle_nn::ops::softmax;
                            
                            let grad_gpu = grad_ref.gpu_tensor.as_ref().unwrap();
                            let input_gpu = input_values[0].gpu_tensor.as_ref().unwrap();
                            
                            // Compute softmax on GPU
                            let softmax_gpu = softmax(input_gpu, candle_core::D::Minus1)
                                .map_err(|e| format!("Failed to compute softmax on GPU: {}", e))?;
                            
                            // Compute grad * softmax
                            let grad_softmax = grad_gpu.mul(&softmax_gpu)
                                .map_err(|e| format!("Failed to multiply grad * softmax: {}", e))?;
                            
                            // Sum along last dimension (keepdim=true)
                            let sum_grad_softmax = grad_softmax.sum_keepdim(candle_core::D::Minus1)
                                .map_err(|e| format!("Failed to sum grad*softmax: {}", e))?;
                            
                            // Broadcast sum back to original shape
                            let sum_broadcast = sum_grad_softmax.broadcast_as(grad_gpu.dims())
                                .map_err(|e| format!("Failed to broadcast sum: {}", e))?;
                            
                            // Compute grad - sum
                            let grad_minus_sum = grad_gpu.sub(&sum_broadcast)
                                .map_err(|e| format!("Failed to compute grad - sum: {}", e))?;
                            
                            // Final result: softmax * (grad - sum)
                            let result_gpu = softmax_gpu.mul(&grad_minus_sum)
                                .map_err(|e| format!("Failed to compute final gradient: {}", e))?;
                            
                            // Convert result back to DataCode Tensor
                            let result_shape = result_gpu.dims().to_vec();
                            let result_rank = result_gpu.rank();
                            
                            // Convert based on tensor rank
                            let result_data = if result_rank == 1 {
                                result_gpu.to_vec1::<f32>()
                                    .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
                            } else if result_rank == 2 {
                                let data_2d = result_gpu.to_vec2::<f32>()
                                    .map_err(|e| format!("Failed to convert result to Vec: {}", e))?;
                                data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
                            } else {
                                // Higher rank tensor, reshape to 1D first
                                let total_size: usize = result_shape.iter().product();
                                let flattened = result_gpu.reshape((total_size,))
                                    .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
                                flattened.to_vec1::<f32>()
                                    .map_err(|e| format!("Failed to convert result to Vec: {}", e))?
                            };
                            
                            // Create result tensor with GPU buffer preserved
                            let result_device = input_values[0].device().clone();
                            let mut result_tensor = Tensor::from_gpu_tensor(
                                result_shape,
                                result_device,
                                Some(result_gpu),
                            );
                            
                            // Update CPU data for compatibility
                            result_tensor.data = result_data;
                            
                            vec![result_tensor]
                        } else {
                            // Fallback to CPU
                            let grad_cpu = grad_ref.to_cpu()?;
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
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        // CPU only
                        let grad_cpu = grad_ref.to_cpu()?;
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
                }
                OpType::CrossEntropy => {
                    if input_values.len() != 2 {
                        return Err("CrossEntropy operation requires 2 inputs (logits, targets)".to_string());
                    }
                    // CrossEntropy backward (sparse): 
                    // Gradient w.r.t. logits: (softmax(logits) - one_hot(targets)) / batch_size
                    // where one_hot is created from class indices
                    // Gradient w.r.t. targets: None (targets are constants)
                    // OPTIMIZATION: Use reference directly, only convert to CPU when needed
                    let grad_cpu = grad_ref.to_cpu()?;
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
                    // OPTIMIZATION: Work directly with softmax_logits.data instead of cloning
                    let batch_size = logits_cpu.shape[0];
                    let num_classes = logits_cpu.shape[1];
                    let total_size = batch_size * num_classes;
                    
                    // OPTIMIZATION: Pre-allocate grad_data with known capacity
                    let mut grad_data = Vec::with_capacity(total_size);
                    grad_data.extend_from_slice(&softmax_logits.data);
                    
                    // Subtract one-hot encoding of targets
                    for i in 0..batch_size {
                        let target_class = targets_cpu.data[i] as usize;
                        if target_class < num_classes {
                            grad_data[i * num_classes + target_class] -= 1.0;
                        }
                    }
                    
                    // Divide by batch_size and multiply by grad_value
                    let inv_batch_size = grad_value / batch_size as f32;
                    for val in &mut grad_data {
                        *val *= inv_batch_size;
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
                    // OPTIMIZATION: Use reference directly, only convert to CPU when needed
                    let grad_cpu = grad_ref.to_cpu()?;
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
                    // OPTIMIZATION: Use reference directly
                    let input_shape = &input_values[0].shape;
                    vec![grad_ref.reshape(input_shape.clone())?]
                }
                OpType::Broadcast => {
                    // Broadcast is handled in Add operation, not used directly
                    return Err("Broadcast operation should not be used directly in backward pass".to_string());
                }
            };
            // #region agent log
            let op_time = op_start.elapsed();
            // #endregion
            
            // OPTIMIZATION: Track operation profiling data
            let op_time_ms = op_time.as_millis() as u128;
            let entry = op_times.entry(op_type_str.clone()).or_insert((0, 0));
            entry.0 += op_time_ms;
            entry.1 += 1;
            
            // Track GPU buffer creation during operation
            #[cfg(feature = "gpu")]
            {
                let op_gpu_buffers_after = self.count_gpu_tensors();
                let buffers_created = op_gpu_buffers_after.saturating_sub(op_gpu_buffers_before);
                if buffers_created > 0 {
                    _temp_gpu_buffers_created += buffers_created;
                }
            }

            // OPTIMIZATION: Accumulate gradients in input nodes with minimal cloning
            // #region agent log
            let accum_start = std::time::Instant::now();
            let mut accum_count = 0;
            // #endregion
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
                        
                        // OPTIMIZATION: Accumulate on GPU if both tensors are on GPU
                        // Use add operation instead of add_assign
                        let new_grad = existing_grad.add(&input_grad)?;
                        *existing_grad = new_grad;
                        
                        // OPTIMIZATION: Removed synchronization after gradient accumulation - operations are asynchronous
                        // Synchronization will happen after backward pass completes
                        
                        // #region agent log
                        accum_count += 1;
                        // #endregion
                    }
                    None => {
                        // OPTIMIZATION: Clone gradient only when needed (first accumulation)
                        // For subsequent accumulations, we use add_assign which is more efficient
                        self.nodes[input_id].grad = Some(input_grad.clone());
                    }
                }
            }
            
            // OPTIMIZATION: Explicitly drop input_grads after accumulation to free GPU memory
            // This helps Metal free intermediate buffers immediately after backward pass operations
            // This is critical to prevent GPU memory accumulation that causes performance degradation
            drop(input_grads);
            // #region agent log
            let accum_time = accum_start.elapsed();
            // Log timing for operations (sample every 10th node to avoid too much logging)
            if nodes_processed % 10 == 0 || nodes_processed <= 3 || nodes_processed >= backward_order.len() - 3 {
                let log_data = format!(r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"graph.rs:{}","message":"Backward op timing","data":{{"node_id":{},"op_type":"{}","op_time_ms":{},"accum_time_ms":{},"accum_count":{},"nodes_processed":{}}},"timestamp":{}}}"#, 
                    line!(), node_id, op_type_str, op_time.as_millis(), accum_time.as_millis(), accum_count, nodes_processed,
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/Users/igor/Desktop/Projects/DataCode/.cursor/debug.log") {
                    use std::io::Write;
                    let _ = writeln!(file, "{}", log_data);
                }
            }
            // #endregion
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
    /// 
    /// # Performance
    /// By default, preserves GPU buffers for parameters to avoid recreation overhead in the next forward pass.
    /// This significantly improves performance by eliminating GPU buffer recreation costs (10-20ms per batch).
    pub fn clear_non_parameter_nodes(&mut self, param_node_ids: &[NodeId]) -> Result<Vec<NodeId>, String> {
        // OPTIMIZATION: Preserve GPU buffers by default to avoid recreation overhead
        // This eliminates 10-20ms overhead per batch from recreating GPU buffers
        self.clear_non_parameter_nodes_with_gpu_preserve(param_node_ids, true)
    }
    
    /// Clear all non-parameter nodes with option to preserve GPU buffers for parameters
    /// 
    /// # Arguments
    /// * `param_node_ids` - Slice of node IDs that represent parameters (weights, biases) to preserve
    /// * `preserve_param_gpu_buffers` - Whether to preserve GPU buffers for parameters
    /// * `param_values` - Optional map of parameter node IDs to their values. If provided, values are moved from this map instead of cloned from nodes.
    #[allow(unused_variables)] // preserve_param_gpu_buffers only used when gpu feature is enabled
    pub fn clear_non_parameter_nodes_with_gpu_preserve(&mut self, param_node_ids: &[NodeId], preserve_param_gpu_buffers: bool) -> Result<Vec<NodeId>, String> {
        self.clear_non_parameter_nodes_with_values(param_node_ids, preserve_param_gpu_buffers, None)
    }
    
    /// Clear all non-parameter nodes with optional parameter values to avoid cloning
    /// 
    /// # Arguments
    /// * `param_node_ids` - Slice of node IDs that represent parameters (weights, biases) to preserve
    /// * `preserve_param_gpu_buffers` - Whether to preserve GPU buffers for parameters
    /// * `param_values` - Optional mutable reference to map of parameter node IDs to their values. If provided, values are moved from this map instead of cloned from nodes.
    /// 
    /// # Performance
    /// If `param_values` is provided, values are moved instead of cloned, eliminating expensive GPU buffer cloning operations.
    #[allow(unused_variables)] // preserve_param_gpu_buffers only used when gpu feature is enabled
    pub fn clear_non_parameter_nodes_with_values(&mut self, param_node_ids: &[NodeId], preserve_param_gpu_buffers: bool, mut param_values: Option<&mut std::collections::HashMap<NodeId, Tensor>>) -> Result<Vec<NodeId>, String> {
        // DIAGNOSTIC: Log graph size before cleanup
        let nodes_before = self.nodes.len();
        #[cfg(debug_assertions)]
        let input_nodes_before = self.input_nodes.len();
        
        // DIAG: Log memory usage before cleanup
        let memory_before = self.total_memory_bytes();
        let tensor_count_before = self.count_tensors();
        let gpu_tensor_count_before = self.count_gpu_tensors();
        
        #[cfg(debug_assertions)]
        {
            // ENHANCED LOGGING: Check for invalid param_node_ids
            for &param_id in param_node_ids {
                if param_id >= nodes_before {
                    eprintln!(
                        "[ERROR] clear_non_parameter_nodes: Invalid param_node_id {} (graph has {} nodes)",
                        param_id, nodes_before
                    );
                }
            }
            
        }
        
        // Validate that all parameter node IDs are valid
        for &param_id in param_node_ids {
            if param_id >= self.nodes.len() {
                return Err(format!("Invalid parameter node ID: {} (graph has {} nodes)", param_id, self.nodes.len()));
            }
        }
        
        // OPTIMIZATION: Pre-allocate vectors with known capacity
        let num_params = param_node_ids.len();
        let mut new_nodes = Vec::with_capacity(num_params);
        let mut new_param_ids = Vec::with_capacity(num_params);
        
        // Copy only parameter nodes and build mapping
        // CRITICAL: Clear inputs field to remove any stale references to deleted nodes
        // OPTIMIZATION: Iterate only over parameter nodes instead of all nodes
        // OPTIMIZATION: Use direct access instead of get() since we validated IDs above
        for &old_id in param_node_ids {
            let node = &self.nodes[old_id];
            let new_id = new_nodes.len();
            
            // OPTIMIZATION: Move value from param_values map if provided, otherwise clone from node
            // This eliminates expensive cloning when values are already extracted via take() in layer.rs
            let param_value = if let Some(values_map) = &mut param_values {
                // Move value from map if available, otherwise fall back to cloning from node
                values_map.remove(&old_id).or_else(|| node.value.clone())
            } else {
                // No values map provided, clone from node (backward compatibility)
                node.value.clone()
            };
            
            // OPTIMIZATION: Create cleaned node directly instead of cloning then modifying
            let cleaned_node = Node {
                op: OpType::Input,
                inputs: Vec::new(), // CRITICAL: Clear inputs to remove stale references
                value: param_value, // Use moved or cloned value
                grad: None, // Clear gradient
                requires_grad: node.requires_grad,
            };
            new_nodes.push(cleaned_node);
            new_param_ids.push(new_id);
        }
        
        // Verify we found all parameters
        if new_param_ids.len() != param_node_ids.len() {
            return Err(format!(
                "Mismatch: expected {} parameter nodes, found {}",
                param_node_ids.len(),
                new_param_ids.len()
            ));
        }
        
        // Update nodes - replace entire vector to ensure complete cleanup
        self.nodes = new_nodes;
        
        // Update input_nodes to only include parameter nodes (they're all Input nodes)
        self.input_nodes = new_param_ids.clone();
        
        // CRITICAL: Clear gradients and ensure no stale references remain
        // But preserve values - they contain the updated parameter values after optimizer step
        // OPTIMIZATION: All nodes are Input type after cleanup, so we can simplify
        // After cleanup, all remaining nodes are parameters, so we can treat them all as parameters
        for node in &mut self.nodes {
            node.grad = None;
            // CRITICAL: Ensure inputs is empty (all nodes are Input type after cleanup)
            // This prevents stale references to deleted nodes
            node.inputs.clear();
            // Note: node.value is preserved - it contains the updated parameter values
            // OPTIMIZATION: Clear GPU buffers to help Metal free memory and reduce fragmentation
            // GPU buffers will be recreated lazily when needed
            // CRITICAL: Clear GPU buffers aggressively, but preserve parameter buffers if requested
            #[cfg(feature = "gpu")]
            if let Some(ref mut value) = node.value {
                if value.device().is_gpu() {
                    if !preserve_param_gpu_buffers {
                        // Aggressively clear all GPU buffers to force Metal to free memory
                        // This helps prevent memory accumulation that causes performance degradation
                        value.gpu_tensor = None;
                    }
                }
            }
        }
        
        // OPTIMIZATION: After clearing GPU buffers, try to force Metal to release memory
        // This is done by dropping all GPU tensor references explicitly
        // Note: Metal uses lazy deallocation, so this may not immediately free memory,
        // but it helps signal to Metal that buffers are no longer needed
        #[cfg(feature = "gpu")]
        if !preserve_param_gpu_buffers {
            // Force drop any remaining GPU tensor references by ensuring they're cleared
            // This is a best-effort attempt to help Metal free memory
            // The actual memory release is handled by Metal's garbage collector
            for node in &mut self.nodes {
                if let Some(ref mut value) = node.value {
                    if value.device().is_gpu() && value.gpu_tensor.is_some() {
                        // Double-check that GPU buffer is cleared
                        value.gpu_tensor = None;
                    }
                }
            }
        }
        
        // OPTIMIZATION: Combined verification and logging to avoid double iteration
        // DIAGNOSTIC: Log graph size after cleanup
        let nodes_after = self.nodes.len();
        #[cfg(debug_assertions)]
        let input_nodes_after = self.input_nodes.len();
        
        // DIAG: Log memory usage after cleanup
        // NOTE: total_memory_bytes() only counts graph nodes, not parameter cache
        // After cleanup, parameters are stored in cache, so memory_after may show 0
        // Use total_memory_bytes_with_cache() in calling code for accurate memory reporting
        let memory_after = self.total_memory_bytes();
        let tensor_count_after = self.count_tensors();
        let gpu_tensor_count_after = self.count_gpu_tensors();
        let memory_freed = memory_before.saturating_sub(memory_after);
        
        // OPTIMIZATION: Only log cleanup details in debug mode to reduce I/O overhead
        // In release mode, this logging happens 2979 times per training run and slows execution
        #[cfg(debug_assertions)]
        {
            // Combined verification and logging in single pass
            let mut invalid_nodes = Vec::new();
            for (idx, node) in self.nodes.iter().enumerate() {
                if !matches!(node.op, OpType::Input) {
                    invalid_nodes.push((idx, format!("Not Input type: {:?}", node.op)));
                }
                if !node.inputs.is_empty() {
                    invalid_nodes.push((idx, format!("Non-empty inputs: {:?}", node.inputs)));
                }
            }
            
            // eprintln!(
            //     "[DIAG] Graph cleanup - Nodes: {} -> {} (removed {}), Memory: {} -> {} bytes (freed {} bytes, {} MB), Tensors: {} -> {}, GPU tensors: {} -> {}",
            //     nodes_before, nodes_after, nodes_before - nodes_after,
            //     memory_before, memory_after, memory_freed, memory_freed / (1024 * 1024),
            //     tensor_count_before, tensor_count_after,
            //     gpu_tensor_count_before, gpu_tensor_count_after
            // );
            
            if !invalid_nodes.is_empty() {
                eprintln!(
                    "[ERROR] clear_non_parameter_nodes: Found {} invalid nodes after cleanup: {:?}",
                    invalid_nodes.len(),
                    invalid_nodes.iter().take(10).collect::<Vec<_>>()
                );
            }
            
            // Log warning if graph is not properly cleaned (more nodes than expected)
            if nodes_after > param_node_ids.len() {
                eprintln!(
                    "[WARNING] clear_non_parameter_nodes: Graph cleanup incomplete! nodes before={}, after={}, expected={} (param nodes). \
                    input_nodes before={}, after={}. Graph may not be fully cleaned!",
                    nodes_before, nodes_after, param_node_ids.len(),
                    input_nodes_before, input_nodes_after
                );
            }
        }
        
        Ok(new_param_ids)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

