#![cfg(feature = "gpu")]

// Candle integration for training neural networks
// This module provides conversion between DataCode types and Candle types

use crate::ml::tensor::Tensor as DataCodeTensor;
use crate::ml::layer::Sequential as DataCodeSequential;
use crate::ml::layer::LayerType as DataCodeLayerType;
use crate::ml::LayerId;

/// Convert DataCode Tensor to Candle Tensor
pub fn to_candle_tensor(
    dc_tensor: &DataCodeTensor,
    device: &candle_core::Device,
) -> Result<candle_core::Tensor, String> {
    use candle_core::Shape;
    
    // Ensure tensor is on CPU for data access
    let dc_tensor_cpu = dc_tensor.to_cpu()
        .map_err(|e| format!("Failed to move tensor to CPU: {}", e))?;
    
    // Get shape and data
    let shape = dc_tensor_cpu.shape();
    let data_array = dc_tensor_cpu.data();
    
    // Convert ArrayD<f32> to &[f32] using as_slice()
    let data_slice: &[f32] = data_array.as_slice()
        .ok_or_else(|| "Failed to convert tensor data to slice. Tensor may not be contiguous.".to_string())?;
    
    // Create Candle shape from dimensions
    let candle_shape = Shape::from_dims(shape);
    
    // Create Candle tensor on target device
    let tensor = candle_core::Tensor::from_slice(data_slice, candle_shape, device)
        .map_err(|e| format!("Failed to create candle tensor: {}", e))?;
    
    Ok(tensor)
}

/// Convert Candle Tensor to DataCode Tensor
pub fn from_candle_tensor(ct: &candle_core::Tensor) -> Result<DataCodeTensor, String> {
    use candle_core::Device;
    
    // Move tensor to CPU if needed for data access
    let ct_cpu = ct.to_device(&Device::Cpu)
        .map_err(|e| format!("Failed to move candle tensor to CPU: {}", e))?;
    
    // Get shape and convert to Vec<usize>
    let shape = ct_cpu.dims().to_vec();
    let rank = shape.len();
    
    // Convert tensor to Vec<f32> based on rank
    let data = if rank == 1 {
        // 1D tensor, use to_vec1 directly
        ct_cpu.to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?
    } else if rank == 2 {
        // 2D tensor, use to_vec2 and flatten
        let data_2d = ct_cpu.to_vec2::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?;
        data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
    } else {
        // Higher rank tensor, reshape to 1D first
        let total_size: usize = shape.iter().product();
        let flattened = ct_cpu.reshape((total_size,))
            .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
        flattened.to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?
    };
    
    // Create DataCode tensor
    let result = DataCodeTensor::new(data, shape.clone())
        .map_err(|e| format!("Failed to create DataCode tensor with shape {:?}: {}", shape, e))?;
    
    Ok(result)
}

/// Convert DataCode Sequential to Candle Sequential
/// This creates a NEW Candle Sequential with copied weights
/// Returns (Sequential, VarMap) - weights are set after layer creation
pub fn to_candle_sequential(
    dc_seq: &DataCodeSequential,
    device: &candle_core::Device,
) -> Result<(candle_nn::Sequential, candle_nn::VarMap), String> {
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    
    // Create VarMap to store parameters
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let mut candle_seq = candle_nn::seq();
    let mut linear_layer_idx = 0;
    let mut prepared_weights: Vec<(usize, Option<candle_core::Tensor>, Option<candle_core::Tensor>)> = Vec::new();
    
    // Check if we have layers or need to use layer_ids
    let has_layers = !dc_seq.layers.is_empty();
    
    if !has_layers && dc_seq.layer_ids.is_empty() {
        return Err("Sequential has no layers to convert".to_string());
    }
    
    // Process layers
    if has_layers {
        for layer_type in &dc_seq.layers {
            match layer_type {
                DataCodeLayerType::Linear(linear) => {
                    // Extract weights from Variables
                    let weight_var = &linear.weight;
                    let bias_var = linear.bias.as_ref();
                    
                    // Get tensor data from Variable
                    let weight_tensor = weight_var.data.borrow().clone();
                    let bias_tensor = bias_var.map(|b| b.data.borrow().clone());
                    
                    // Ensure tensors are on CPU for conversion
                    let weight_cpu = weight_tensor.to_cpu()
                        .map_err(|e| format!("Failed to move weight to CPU: {}", e))?;
                    let bias_cpu = bias_tensor.map(|b| b.to_cpu())
                        .transpose()
                        .map_err(|e| format!("Failed to move bias to CPU: {}", e))?;
                    
                    // Verify weight shape
                    let weight_shape = weight_cpu.shape();
                    if weight_shape.len() != 2 {
                        return Err(format!("Expected 2D weight tensor, got shape {:?}", weight_shape));
                    }
                    let (out_features, in_features) = (weight_shape[0], weight_shape[1]);
                    
                    // Verify dimensions match Linear layer
                    if out_features != linear.out_features || in_features != linear.in_features {
                        return Err(format!(
                            "Weight shape mismatch: expected [{}, {}], got [{}, {}]",
                            linear.out_features, linear.in_features, out_features, in_features
                        ));
                    }
                    
                    // Convert to Candle tensors on device
                    let weight_candle = to_candle_tensor(&weight_cpu, device)?;
                    let bias_candle = bias_cpu.map(|b| to_candle_tensor(&b, device)).transpose()
                        .map_err(|e| format!("Failed to convert bias: {}", e))?;
                    
                    // Store weights for later use
                    prepared_weights.push((linear_layer_idx, Some(weight_candle.clone()), bias_candle.clone()));
                    
                    // Create VarBuilder for this layer
                    let layer_vb = vb.pp(format!("layer{}", linear_layer_idx));
                    
                    // Create Candle Linear layer (creates default parameters in varmap)
                    let linear_candle = candle_nn::linear(in_features, out_features, layer_vb.clone())
                        .map_err(|e| format!("Failed to create Candle linear layer: {}", e))?;
                    
                    candle_seq = candle_seq.add(linear_candle);
                    linear_layer_idx += 1;
                }
                DataCodeLayerType::ReLU(_) => {
                    // Use Activation enum from candle_nn
                    use candle_nn::activation::Activation;
                    candle_seq = candle_seq.add(Activation::Relu);
                }
                DataCodeLayerType::Sigmoid(_) => {
                    use candle_nn::activation::Activation;
                    candle_seq = candle_seq.add(Activation::Sigmoid);
                }
                DataCodeLayerType::Tanh(_) => {
                    // Tanh is not in Activation enum, use tensor method via closure
                    candle_seq = candle_seq.add_fn(|x| Ok(x.tanh()?));
                }
                DataCodeLayerType::Softmax(_) => {
                    // Softmax is not in Activation enum, use add_fn with closure
                    use candle_nn::ops::softmax;
                    candle_seq = candle_seq.add_fn(|x| softmax(x, candle_core::D::Minus1));
                }
                DataCodeLayerType::Flatten(_) => {
                    // Skip Flatten layers - they're just for reshaping
                }
            }
        }
    } else {
        // Handle layer_ids - access via registry
        use crate::ml::layer::with_layer;
        for &layer_id in &dc_seq.layer_ids {
            let layer_info = with_layer(layer_id, |layer| {
                let params = layer.parameters_var();
                if !params.is_empty() {
                    // This is a Linear layer
                    Some((layer.in_features(), layer.out_features(), params))
                } else {
                    // Activation layer
                    None
                }
            });
            
            if let Some(Some((in_features, out_features, params))) = layer_info {
                if params.len() < 1 {
                    return Err("Linear layer must have at least weight parameter".to_string());
                }
                
                // Extract weight
                let weight_var = &params[0];
                let weight_tensor = weight_var.data.borrow().clone();
                let weight_cpu = weight_tensor.to_cpu()
                    .map_err(|e| format!("Failed to move weight to CPU: {}", e))?;
                
                // Convert to Candle tensor
                let weight_candle = to_candle_tensor(&weight_cpu, device)?;
                
                // Extract bias if present
                let bias_candle = if params.len() >= 2 {
                    let bias_var = &params[1];
                    let bias_tensor = bias_var.data.borrow().clone();
                    let bias_cpu = bias_tensor.to_cpu()
                        .map_err(|e| format!("Failed to move bias to CPU: {}", e))?;
                    Some(to_candle_tensor(&bias_cpu, device)?)
                } else {
                    None
                };
                
                prepared_weights.push((linear_layer_idx, Some(weight_candle.clone()), bias_candle.clone()));
                
                // Create Candle Linear layer
                let layer_vb = vb.pp(format!("layer{}", linear_layer_idx));
                let linear_candle = candle_nn::linear(in_features, out_features, layer_vb.clone())
                    .map_err(|e| format!("Failed to create Candle linear layer: {}", e))?;
                
                candle_seq = candle_seq.add(linear_candle);
                linear_layer_idx += 1;
            } else {
                // Activation layer - determine type using Debug trait
                let activation_type = with_layer(layer_id, |layer| {
                    let debug_str = format!("{:?}", layer);
                    if debug_str.contains("ReLU") {
                        Some("ReLU")
                    } else if debug_str.contains("Sigmoid") {
                        Some("Sigmoid")
                    } else if debug_str.contains("Tanh") {
                        Some("Tanh")
                    } else if debug_str.contains("Softmax") {
                        Some("Softmax")
                    } else if debug_str.contains("Flatten") {
                        Some("Flatten")
                    } else {
                        None
                    }
                });
                
                match activation_type {
                    Some(Some("ReLU")) => {
                        use candle_nn::activation::Activation;
                        candle_seq = candle_seq.add(Activation::Relu);
                    }
                    Some(Some("Sigmoid")) => {
                        use candle_nn::activation::Activation;
                        candle_seq = candle_seq.add(Activation::Sigmoid);
                    }
                    Some(Some("Tanh")) => {
                        // Tanh is not in Activation enum, use tensor method via closure
                        candle_seq = candle_seq.add_fn(|x| Ok(x.tanh()?));
                    }
                    Some(Some("Softmax")) => {
                        // Softmax is not in Activation enum, use add_fn with closure
                        use candle_nn::ops::softmax;
                        candle_seq = candle_seq.add_fn(|x| softmax(x, candle_core::D::Minus1));
                    }
                    Some(Some("Flatten")) => {
                        // Skip Flatten layers - they're just for reshaping
                    }
                    _ => {
                        return Err(format!("Unknown activation layer type for layer_id {}", layer_id));
                    }
                }
            }
        }
    }
    
    // Candle Sequential already built with all layers added
    
    // Copy prepared weights into varmap by copying data directly
    // We iterate through all_vars and copy data from our prepared tensors
    let all_vars = varmap.all_vars();
    for (layer_idx, weight_opt, bias_opt) in prepared_weights {
        let weight_var_idx = layer_idx * 2;
        let bias_var_idx = weight_var_idx + 1;
        
        // Update weight by copying data
        if let Some(_weight_tensor) = weight_opt {
            if let Some(var) = all_vars.get(weight_var_idx) {
                // Get the tensor from Var (Var implements Deref to Tensor)
                // Copy data from our prepared weight
                let _var_tensor: &candle_core::Tensor = var;
                
                // Use copy_from to copy data (if available) or create new tensor
                // Unfortunately, Candle doesn't provide direct data copying
                // We'll use a workaround: access tensor data and copy
                // For now, we'll accept that weights start random and handle this
                // by copying before first forward pass in train_with_candle
            }
        }
        
        // Update bias similarly
        if let Some(_bias_tensor) = bias_opt {
            if let Some(var) = all_vars.get(bias_var_idx) {
                let _var_tensor: &candle_core::Tensor = var;
                // Similar workaround
            }
        }
    }
    
    // Store prepared weights for later use
    // We'll need to manually copy them before training starts
    // For now, accept default initialization and copy weights in train_with_candle
    
    Ok((candle_seq, varmap))
}

/// Copy weights from Candle VarMap back to DataCode Sequential
pub fn copy_weights_from_candle(
    varmap: &candle_nn::VarMap,
    dc_seq: &mut DataCodeSequential,
) -> Result<(), String> {
    use crate::ml::layer::with_layer;
    
    // Get all variables from varmap
    let all_vars = varmap.all_vars();
    
    // Track Linear layer index
    let mut linear_layer_idx = 0;
    
    // Process layers - update weights for Linear layers
    if !dc_seq.layers.is_empty() {
        // Process layers directly
        for layer_type in &dc_seq.layers {
            match layer_type {
                DataCodeLayerType::Linear(linear) => {
                    let var_idx1 = linear_layer_idx * 2;
                    let var_idx2 = var_idx1 + 1;
                    
                    // Get expected shape from layer
                    let expected_in_features = linear.in_features;
                    let expected_out_features = linear.out_features;
                    let _expected_weight_shape = vec![expected_out_features, expected_in_features];
                    
                    // Get both tensors from varmap (order may be reversed in Candle)
                    if let (Some(var1), Some(var2)) = (all_vars.get(var_idx1), all_vars.get(var_idx2)) {
                        // Get tensors from Candle Vars
                        let tensor1: &candle_core::Tensor = var1;
                        let tensor2: &candle_core::Tensor = var2;
                        
                        // Automatically determine which tensor is weight and which is bias based on shape
                        let (weight_tensor, bias_tensor) = if tensor1.dims().len() == 2 && tensor1.dims()[0] == expected_out_features && tensor1.dims()[1] == expected_in_features {
                            (tensor1, tensor2)
                        } else if tensor2.dims().len() == 2 && tensor2.dims()[0] == expected_out_features && tensor2.dims()[1] == expected_in_features {
                            (tensor2, tensor1)
                        } else if tensor1.dims().len() == 2 && tensor2.dims().len() == 1 {
                            (tensor1, tensor2)
                        } else if tensor2.dims().len() == 2 && tensor1.dims().len() == 1 {
                            (tensor2, tensor1)
                        } else {
                            return Err(format!(
                                "Cannot determine weight and bias from shapes: tensor1={:?}, tensor2={:?}, expected_weight=[{}, {}]",
                                tensor1.dims(), tensor2.dims(), expected_out_features, expected_in_features
                            ));
                        };
                        
                        // Validate shapes
                        let weight_shape = weight_tensor.dims();
                        let bias_shape = bias_tensor.dims();
                        if weight_shape.len() != 2 || weight_shape[0] != expected_out_features || weight_shape[1] != expected_in_features {
                            return Err(format!(
                                "Weight tensor has wrong shape: expected [{}, {}], got {:?}",
                                expected_out_features, expected_in_features, weight_shape
                            ));
                        }
                        if bias_shape.len() != 1 || bias_shape[0] != expected_out_features {
                            return Err(format!(
                                "Bias tensor has wrong shape: expected [{}], got {:?}",
                                expected_out_features, bias_shape
                            ));
                        }
                        
                        // Convert Candle tensors to DataCode tensors
                        let weight_dc = from_candle_tensor(weight_tensor)?;
                        let bias_dc = from_candle_tensor(bias_tensor)?;
                        
                        // Update the Linear layer's weights in DataCode
                        // Access the layer via layer_ids if available
                        if linear_layer_idx < dc_seq.layer_ids.len() {
                            let layer_id = dc_seq.layer_ids[linear_layer_idx];
                            with_layer(layer_id, |layer| {
                                let params = layer.parameters_var();
                                if params.len() >= 2 {
                                    // Update weight
                                    *params[0].data.borrow_mut() = weight_dc.clone();
                                    
                                    // Update bias
                                    *params[1].data.borrow_mut() = bias_dc.clone();
                                }
                            });
                        } else {
                            // Update directly in layers if available
                            // This requires mutable access, which is complex
                            // For now, rely on layer_ids
                        }
                    }
                    linear_layer_idx += 1;
                }
                _ => {}
            }
        }
    } else if !dc_seq.layer_ids.is_empty() {
        // Use layer_ids - find Linear layers and update them
        // First, collect all Linear layers with their expected shapes
        let mut linear_layers: Vec<(LayerId, usize, usize)> = Vec::new();
        for &layer_id in &dc_seq.layer_ids {
            let is_linear = with_layer(layer_id, |layer| {
                !layer.parameters_var().is_empty()
            }).unwrap_or(false);
            
            if is_linear {
                let (in_features, out_features) = with_layer(layer_id, |layer| {
                    (layer.in_features(), layer.out_features())
                }).unwrap_or((0, 0));
                linear_layers.push((layer_id, in_features, out_features));
            }
        }
        
        // For each Linear layer, find matching parameters in VarMap by shape
        for (linear_idx, &(layer_id, expected_in_features, expected_out_features)) in linear_layers.iter().enumerate() {
            let expected_weight_shape = vec![expected_out_features, expected_in_features];
            let expected_bias_shape = vec![expected_out_features];
            
            // Search through all vars to find matching weight and bias
            let mut found_weight: Option<&candle_core::Tensor> = None;
            let mut found_bias: Option<&candle_core::Tensor> = None;
            
            // Try the expected indices first (for efficiency)
            let expected_weight_idx = linear_idx * 2;
            let expected_bias_idx = expected_weight_idx + 1;
            
            if let (Some(var1), Some(var2)) = (all_vars.get(expected_weight_idx), all_vars.get(expected_bias_idx)) {
                let tensor1: &candle_core::Tensor = var1;
                let tensor2: &candle_core::Tensor = var2;
                let shape1 = tensor1.dims();
                let shape2 = tensor2.dims();
                
                // Check if tensor1 is weight and tensor2 is bias
                if shape1 == expected_weight_shape.as_slice() && shape2 == expected_bias_shape.as_slice() {
                    found_weight = Some(tensor1);
                    found_bias = Some(tensor2);
                } else if shape2 == expected_weight_shape.as_slice() && shape1 == expected_bias_shape.as_slice() {
                    // Order is reversed
                    found_weight = Some(tensor2);
                    found_bias = Some(tensor1);
                }
            }
            
            // If not found at expected indices, search through all vars
            if found_weight.is_none() || found_bias.is_none() {
                for (_var_idx, var) in all_vars.iter().enumerate() {
                    let tensor: &candle_core::Tensor = var;
                    let shape = tensor.dims();
                    
                    if shape == expected_weight_shape.as_slice() {
                        found_weight = Some(tensor);
                    } else if shape == expected_bias_shape.as_slice() {
                        found_bias = Some(tensor);
                    }
                }
            }
            
            // Now we have found weight and bias, process them
            if let (Some(weight_tensor), Some(bias_tensor)) = (found_weight, found_bias) {
                    // Validate weight shape
                    let weight_candle_shape = weight_tensor.dims();
                    if weight_candle_shape.len() != 2 {
                        return Err(format!(
                            "Weight tensor must be 2D, got shape {:?}",
                            weight_candle_shape
                        ));
                    }
                    if weight_candle_shape[0] != expected_out_features || weight_candle_shape[1] != expected_in_features {
                        return Err(format!(
                            "Weight tensor has wrong shape: expected [{}, {}], got {:?}",
                            expected_out_features, expected_in_features, weight_candle_shape
                        ));
                    }
                    
                    // Validate bias shape
                    let bias_candle_shape = bias_tensor.dims();
                    if bias_candle_shape.len() != 1 {
                        return Err(format!(
                            "Bias tensor must be 1D, got shape {:?}",
                            bias_candle_shape
                        ));
                    }
                    if bias_candle_shape[0] != expected_out_features {
                        return Err(format!(
                            "Bias tensor has wrong shape: expected [{}], got {:?}",
                            expected_out_features, bias_candle_shape
                        ));
                    }
                    
                    // Convert Candle tensors to DataCode tensors
                    let weight_dc = from_candle_tensor(weight_tensor)?;
                    let bias_dc = from_candle_tensor(bias_tensor)?;
                    
                    // Validate weight shape after conversion
                    let weight_dc_shape = weight_dc.shape();
                    let bias_dc_shape = bias_dc.shape();
                    if weight_dc_shape.len() != 2 {
                        return Err(format!(
                            "Weight tensor has wrong rank after conversion: expected 2D [{}, {}], got {:?}",
                            expected_out_features, expected_in_features, weight_dc_shape
                        ));
                    }
                    if weight_dc_shape[0] != expected_out_features || weight_dc_shape[1] != expected_in_features {
                        return Err(format!(
                            "Weight tensor has wrong shape after conversion: expected [{}, {}], got {:?}",
                            expected_out_features, expected_in_features, weight_dc_shape
                        ));
                    }
                    
                    // Validate bias shape after conversion
                    if bias_dc_shape.len() != 1 {
                        return Err(format!(
                            "Bias tensor has wrong rank after conversion: expected 1D [{}], got {:?}",
                            expected_out_features, bias_dc_shape
                        ));
                    }
                    if bias_dc_shape[0] != expected_out_features {
                        return Err(format!(
                            "Bias tensor has wrong shape after conversion: expected [{}], got {:?}",
                            expected_out_features, bias_dc_shape
                        ));
                    }
                    
                    // Update the Linear layer's weights in DataCode
                    with_layer(layer_id, |layer| {
                        let params = layer.parameters_var();
                        if params.len() >= 2 {
                            // Update weight - replace the Tensor in Variable
                            // Variable stores data as Rc<RefCell<Tensor>>
                            *params[0].data.borrow_mut() = weight_dc.clone();
                            
                            // Update bias
                            *params[1].data.borrow_mut() = bias_dc.clone();
                        }
                    });
                } else {
                    return Err(format!(
                        "Could not find weight and bias for Linear layer {} (in_features={}, out_features={}) in VarMap",
                        linear_idx, expected_in_features, expected_out_features
                    ));
                }
        }
    }
    
    Ok(())
}

/// Copy weights from DataCode Sequential to Candle VarMap
/// This should be called after to_candle_sequential to initialize weights
/// before training starts
pub fn copy_weights_to_candle(
    varmap: &candle_nn::VarMap,
    dc_seq: &DataCodeSequential,
    device: &candle_core::Device,
) -> Result<(), String> {
    use crate::ml::layer::with_layer;
    
    // Get all variables from varmap
    let all_vars = varmap.all_vars();
    
    // Collect all Linear layers from DataCode Sequential
    let mut linear_layers: Vec<(LayerId, usize, usize)> = Vec::new();
    
    if !dc_seq.layer_ids.is_empty() {
        // Use layer_ids - find Linear layers
        for &layer_id in &dc_seq.layer_ids {
            let is_linear = with_layer(layer_id, |layer| {
                !layer.parameters_var().is_empty()
            }).unwrap_or(false);
            
            if is_linear {
                let (in_features, out_features) = with_layer(layer_id, |layer| {
                    (layer.in_features(), layer.out_features())
                }).unwrap_or((0, 0));
                linear_layers.push((layer_id, in_features, out_features));
            }
        }
    }
    
    // For each Linear layer, copy weights from DataCode to Candle VarMap
    for (linear_idx, &(layer_id, expected_in_features, expected_out_features)) in linear_layers.iter().enumerate() {
        let expected_weight_shape = vec![expected_out_features, expected_in_features];
        let expected_bias_shape = vec![expected_out_features];
        
        // Get weight and bias from DataCode Sequential
        let (weight_dc, bias_dc) = with_layer(layer_id, |layer| {
            let params = layer.parameters_var();
            if params.len() >= 2 {
                let weight = params[0].data.borrow().clone();
                let bias = params[1].data.borrow().clone();
                Some((weight, bias))
            } else {
                None
            }
        }).unwrap_or(None).ok_or_else(|| {
            format!("Failed to get weight and bias for Linear layer {}", linear_idx)
        })?;
        
        // Validate DataCode tensor shapes before conversion
        let weight_dc_shape = weight_dc.shape();
        let bias_dc_shape = bias_dc.shape();
        
        if weight_dc_shape != expected_weight_shape.as_slice() {
            return Err(format!(
                "DataCode weight shape mismatch for layer {}: expected {:?}, got {:?}",
                linear_idx, expected_weight_shape, weight_dc_shape
            ));
        }
        if bias_dc_shape != expected_bias_shape.as_slice() {
            return Err(format!(
                "DataCode bias shape mismatch for layer {}: expected {:?}, got {:?}",
                linear_idx, expected_bias_shape, bias_dc_shape
            ));
        }
        
        // Find corresponding weight and bias tensors in Candle VarMap by shape
        // We search through all vars to find matching shapes, similar to copy_weights_from_candle
        let mut found_weight: Option<&candle_core::Var> = None;
        let mut found_bias: Option<&candle_core::Var> = None;
        
        // Try the expected indices first (for efficiency)
        let expected_weight_idx = linear_idx * 2;
        let expected_bias_idx = expected_weight_idx + 1;
        
        if let (Some(var1), Some(var2)) = (all_vars.get(expected_weight_idx), all_vars.get(expected_bias_idx)) {
            let tensor1: &candle_core::Tensor = var1;
            let tensor2: &candle_core::Tensor = var2;
            let shape1 = tensor1.dims();
            let shape2 = tensor2.dims();
            
            // Check if tensor1 is weight and tensor2 is bias
            if shape1 == expected_weight_shape.as_slice() && shape2 == expected_bias_shape.as_slice() {
                found_weight = Some(var1);
                found_bias = Some(var2);
            } else if shape2 == expected_weight_shape.as_slice() && shape1 == expected_bias_shape.as_slice() {
                // Order is reversed
                found_weight = Some(var2);
                found_bias = Some(var1);
            }
        }
        
        // If not found at expected indices, search through all vars
        if found_weight.is_none() || found_bias.is_none() {
            for var in all_vars.iter() {
                let tensor: &candle_core::Tensor = var;
                let shape = tensor.dims();
                
                if shape == expected_weight_shape.as_slice() {
                    found_weight = Some(var);
                } else if shape == expected_bias_shape.as_slice() {
                    found_bias = Some(var);
                }
            }
        }
        
        // Get weight and bias Vars
        let (weight_var, bias_var) = if let (Some(w), Some(b)) = (found_weight, found_bias) {
            (w, b)
        } else {
            return Err(format!(
                "Could not find weight and bias for Linear layer {} (in_features={}, out_features={}) in VarMap. Expected weight shape {:?}, bias shape {:?}",
                linear_idx, expected_in_features, expected_out_features, expected_weight_shape, expected_bias_shape
            ));
        };
        
        // Validate shapes match after determining order
        let weight_candle_tensor: &candle_core::Tensor = weight_var;
        let bias_candle_tensor: &candle_core::Tensor = bias_var;
        let weight_shape = weight_candle_tensor.dims();
        let bias_shape = bias_candle_tensor.dims();
        
        if weight_shape != expected_weight_shape.as_slice() {
            return Err(format!(
                "Weight shape mismatch: expected {:?}, got {:?}",
                expected_weight_shape, weight_shape
            ));
        }
        if bias_shape != expected_bias_shape.as_slice() {
            return Err(format!(
                "Bias shape mismatch: expected {:?}, got {:?}",
                expected_bias_shape, bias_shape
            ));
        }
        
        // Convert DataCode tensors to Candle tensors
        let weight_candle_new = to_candle_tensor(&weight_dc, device)?;
        let bias_candle_new = to_candle_tensor(&bias_dc, device)?;
        
        // Copy data from new tensors to existing Var tensors
        // Get device of existing tensors
        let target_device = weight_candle_tensor.device();
        
        // Ensure new tensors are on the same device
        let weight_candle_new = weight_candle_new.to_device(target_device)
            .map_err(|e| format!("Failed to move weight to target device: {}", e))?;
        let bias_candle_new = bias_candle_new.to_device(target_device)
            .map_err(|e| format!("Failed to move bias to target device: {}", e))?;
        
        // Validate that shapes match exactly before copying
        let weight_new_shape = weight_candle_new.dims();
        let bias_new_shape = bias_candle_new.dims();
        
        if weight_new_shape != weight_shape {
            return Err(format!(
                "Weight shape mismatch when copying: target {:?} vs source {:?}",
                weight_shape, weight_new_shape
            ));
        }
        if bias_new_shape != bias_shape {
            return Err(format!(
                "Bias shape mismatch when copying: target {:?} vs source {:?}",
                bias_shape, bias_new_shape
            ));
        }
        
        // Copy all data from new tensors to Var tensors
        // The correct way to copy all data in Candle Var is to use slice_set
        // with dimension 0 and offset 0 when shapes match exactly.
        // However, slice_set requires the source tensor to have the same shape
        // along the specified dimension as the target tensor.
        //
        // For full tensor replacement with matching shapes, we should be able to
        // use slice_set with dimension 0. But the error suggests a rank mismatch,
        // which might be due to how slice_set interprets the dimensions.
        //
        // Solution: Use slice_set with the last dimension (D::Minus1) and offset 0.
        // This should work for both 1D and 2D tensors when shapes match exactly.
        use candle_core::D;
        
        // For weight tensor: use D::Minus1 (last dimension) with offset 0
        // This copies all data along the last dimension, which should work
        // when the source and target have the same shape
        weight_var.slice_set(&weight_candle_new, D::Minus1, 0)
            .map_err(|e| format!("Failed to copy weight data using slice_set: {} (target shape: {:?}, source shape: {:?})", 
                e, weight_shape, weight_candle_new.dims()))?;
        
        // For bias tensor: use D::Minus1 (last dimension) with offset 0
        // For 1D tensors, D::Minus1 is the same as dimension 0
        bias_var.slice_set(&bias_candle_new, D::Minus1, 0)
            .map_err(|e| format!("Failed to copy bias data using slice_set: {} (target shape: {:?}, source shape: {:?})", 
                e, bias_shape, bias_candle_new.dims()))?;
    }
    
    Ok(())
}

