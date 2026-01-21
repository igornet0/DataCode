// Tests for ML Layer module
// Tests verify that each layer can accept a tensor and return a processed tensor

use data_code::ml::tensor::Tensor;
use data_code::ml::autograd::requires_grad;
use data_code::ml::layer::{Layer, Linear, ReLU, Softmax};

/// Helper function to call a layer with an input tensor
/// Creates a Variable from input, calls layer.forward(), and returns output tensor
fn call_layer(layer: &dyn Layer, input: Tensor) -> Result<Tensor, String> {
    // Create Variable from input tensor
    let input_var = requires_grad(input);
    
    // Call layer.forward() - returns Rc<Variable>
    let output_var = layer.forward_var(input_var);
    
    // Extract tensor from Variable - clone the data to avoid borrow checker issues
    let output_tensor = {
        let borrowed = output_var.data.borrow();
        borrowed.clone()
    };
    Ok(output_tensor)
}

// ============================================================================
// Linear Layer Tests
// ============================================================================

#[test]
fn test_linear_layer_forward_1d() {
    // Test Linear layer with 1D input [batch_size=1, in_features=10]
    let layer = Linear::new(10, 2, true).unwrap();
    
    // Create input tensor [1, 10] (batch_size=1, features=10)
    let input = Tensor::ones(vec![1, 10]);
    
    let output = call_layer(&layer, input).unwrap();
    
    // Check output shape: [batch_size, out_features] = [1, 2]
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(output.data.len(), 2);
    
    // Output should be computed (not all zeros, since weights are initialized)
    // With He initialization, weights are random, so output should be non-zero
    let sum: f32 = output.data.iter().sum();
    // Sum should be non-zero (weights are initialized randomly)
    assert!(sum.abs() > 0.0);
}

#[test]
fn test_linear_layer_forward_2d() {
    // Test Linear layer with 2D input [batch_size=3, in_features=5]
    let layer = Linear::new(5, 3, true).unwrap();
    
    // Create input tensor [3, 5] (batch_size=3, features=5)
    let input = Tensor::ones(vec![3, 5]);
    
    let output = call_layer(&layer, input).unwrap();
    
    // Check output shape: [batch_size, out_features] = [3, 3]
    assert_eq!(output.shape, vec![3, 3]);
    assert_eq!(output.data.len(), 9);
}

#[test]
fn test_linear_layer_output_shape() {
    // Test that Linear layer produces correct output shape
    let layer = Linear::new(8, 4, true).unwrap();
    
    let input = Tensor::zeros(vec![2, 8]);
    let output = call_layer(&layer, input).unwrap();
    
    // Input: [2, 8], Output should be: [2, 4]
    assert_eq!(output.shape, vec![2, 4]);
    assert_eq!(output.data.len(), 8);
}

#[test]
fn test_linear_layer_parameters() {
    // Test that Linear layer has parameters (weight and bias)
    let layer = Linear::new(5, 3, true).unwrap();
    
    // Check that parameters exist
    let params = layer.parameters_var();
    assert_eq!(params.len(), 2); // weight and bias
    
    // Check parameter shapes
    // Weight: [out_features, in_features] = [3, 5]
    // Bias: [out_features] = [3]
    let weight_var = &params[0];
    let bias_var = &params[1];
    
    let weight_tensor = weight_var.data.borrow();
    let bias_tensor = bias_var.data.borrow();
    
    assert_eq!(weight_tensor.shape, vec![3, 5]);
    assert_eq!(bias_tensor.shape, vec![3]);
}

// ============================================================================
// ReLU Layer Tests
// ============================================================================

#[test]
fn test_relu_layer_forward_positive() {
    // Test ReLU with positive values (should pass through unchanged)
    let layer = ReLU;
    
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    
    let output = call_layer(&layer, input.clone()).unwrap();
    
    // ReLU should preserve positive values
    assert_eq!(output.shape, input.shape);
    assert_eq!(output.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_relu_layer_forward_negative() {
    // Test ReLU with negative values (should become 0)
    let layer = ReLU;
    
    let input = Tensor::new(vec![-1.0, -2.0, -3.0, -4.0], vec![4]).unwrap();
    
    let output = call_layer(&layer, input).unwrap();
    
    // ReLU should set negative values to 0
    assert_eq!(output.shape, vec![4]);
    assert_eq!(output.data, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_relu_layer_forward_mixed() {
    // Test ReLU with mixed positive and negative values
    let layer = ReLU;
    
    let input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
    
    let output = call_layer(&layer, input).unwrap();
    
    // ReLU: negative -> 0, zero -> 0, positive -> unchanged
    assert_eq!(output.shape, vec![5]);
    assert_eq!(output.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_relu_layer_output_shape() {
    // Test that ReLU preserves input shape
    let layer = ReLU;
    
    // Test with 1D
    let input1d = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let output1d = call_layer(&layer, input1d).unwrap();
    assert_eq!(output1d.shape, vec![3]);
    
    // Test with 2D
    let input2d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let output2d = call_layer(&layer, input2d).unwrap();
    assert_eq!(output2d.shape, vec![2, 2]);
    
    // Test with 3D
    let input3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
    let output3d = call_layer(&layer, input3d).unwrap();
    assert_eq!(output3d.shape, vec![2, 2, 2]);
}

// ============================================================================
// Softmax Layer Tests
// ============================================================================

#[test]
fn test_softmax_layer_forward_1d() {
    // Test Softmax with 1D tensor
    let layer = Softmax;
    
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    
    let output = call_layer(&layer, input).unwrap();
    
    // Check shape is preserved
    assert_eq!(output.shape, vec![3]);
    assert_eq!(output.data.len(), 3);
    
    // Check that all values are positive
    for &val in &output.data {
        assert!(val > 0.0);
    }
    
    // Check that sum is approximately 1.0
    let sum: f32 = output.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got {}", sum);
    
    // Check that largest input has highest probability
    // Input: [1.0, 2.0, 3.0], so output[2] should be largest
    assert!(output.data[2] > output.data[1]);
    assert!(output.data[1] > output.data[0]);
}

#[test]
fn test_softmax_layer_forward_2d() {
    // Test Softmax with 2D tensor (batch)
    let layer = Softmax;
    
    // Input: [2, 3] - 2 samples, 3 classes each
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![2, 3]).unwrap();
    
    let output = call_layer(&layer, input).unwrap();
    
    // Check shape is preserved
    assert_eq!(output.shape, vec![2, 3]);
    assert_eq!(output.data.len(), 6);
    
    // Check that all values are positive
    for &val in &output.data {
        assert!(val > 0.0);
    }
    
    // Check that each row sums to 1.0
    // Row 0: [1.0, 2.0, 3.0] -> softmax
    let row0_sum: f32 = output.data[0..3].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "Row 0 sum should be 1.0, got {}", row0_sum);
    
    // Row 1: [3.0, 2.0, 1.0] -> softmax
    let row1_sum: f32 = output.data[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 sum should be 1.0, got {}", row1_sum);
}

#[test]
fn test_softmax_layer_sum_to_one() {
    // Test that Softmax output always sums to 1.0
    let layer = Softmax;
    
    // Test with various inputs
    let test_cases = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![-1.0, 0.0, 1.0],
        vec![10.0, 20.0, 30.0],
    ];
    
    for test_input in test_cases {
        let input = Tensor::new(test_input.clone(), vec![test_input.len()]).unwrap();
        let output = call_layer(&layer, input).unwrap();
        
        let sum: f32 = output.data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax sum should be 1.0, got {} for input {:?}",
            sum,
            test_input
        );
    }
}

#[test]
fn test_softmax_layer_output_shape() {
    // Test that Softmax preserves input shape
    let layer = Softmax;
    
    // Test with 1D
    let input1d = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let output1d = call_layer(&layer, input1d).unwrap();
    assert_eq!(output1d.shape, vec![3]);
    
    // Test with 2D
    let input2d = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
    let output2d = call_layer(&layer, input2d).unwrap();
    assert_eq!(output2d.shape, vec![2, 3]);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_layer_chain() {
    // Test chaining layers: Linear → ReLU → Softmax
    let linear = Linear::new(10, 5, true).unwrap();
    let relu = ReLU;
    let softmax = Softmax;
    
    // Create input [1, 10]
    let input = Tensor::ones(vec![1, 10]);
    
    // Chain: Linear
    let output1 = call_layer(&linear, input).unwrap();
    assert_eq!(output1.shape, vec![1, 5]);
    
    // Chain: ReLU (on Linear output)
    let output2 = call_layer(&relu, output1).unwrap();
    assert_eq!(output2.shape, vec![1, 5]);
    
    // Chain: Softmax (on ReLU output)
    let output3 = call_layer(&softmax, output2).unwrap();
    assert_eq!(output3.shape, vec![1, 5]);
    
    // Check Softmax properties
    let sum: f32 = output3.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    for &val in &output3.data {
        assert!(val > 0.0);
    }
}

#[test]
fn test_layer_with_different_shapes() {
    // Test layers with various input shapes
    let relu = ReLU;
    
    // Test 1D
    let input1d = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3]).unwrap();
    let output1d = call_layer(&relu, input1d).unwrap();
    assert_eq!(output1d.shape, vec![3]);
    
    // Test 2D
    let input2d = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let output2d = call_layer(&relu, input2d).unwrap();
    assert_eq!(output2d.shape, vec![2, 2]);
    
    // Test 3D
    let input3d = Tensor::new(vec![-1.0; 8], vec![2, 2, 2]).unwrap();
    let output3d = call_layer(&relu, input3d).unwrap();
    assert_eq!(output3d.shape, vec![2, 2, 2]);
    // All should be 0 (ReLU of negative)
    for &val in &output3d.data {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_linear_layer_with_batch() {
    // Test Linear layer with different batch sizes
    // Note: We create a new layer for each test because layers store node IDs
    // that are specific to a graph instance
    
    // Batch size 1
    let layer1 = Linear::new(4, 2, true).unwrap();
    let input1 = Tensor::ones(vec![1, 4]);
    let output1 = call_layer(&layer1, input1).unwrap();
    assert_eq!(output1.shape, vec![1, 2]);
    
    // Batch size 3
    let layer3 = Linear::new(4, 2, true).unwrap();
    let input3 = Tensor::ones(vec![3, 4]);
    let output3 = call_layer(&layer3, input3).unwrap();
    assert_eq!(output3.shape, vec![3, 2]);
    
    // Batch size 10
    let layer10 = Linear::new(4, 2, true).unwrap();
    let input10 = Tensor::ones(vec![10, 4]);
    let output10 = call_layer(&layer10, input10).unwrap();
    assert_eq!(output10.shape, vec![10, 2]);
}

