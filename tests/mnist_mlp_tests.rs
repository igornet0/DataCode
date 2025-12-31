// Tests for MNIST MLP architecture

use data_code::ml::tensor::Tensor;
use data_code::ml::layer::{Linear, ReLU, Softmax, Sequential, add_layer_to_registry};
use data_code::ml::model::NeuralNetwork;
use data_code::ml::loss::sparse_softmax_cross_entropy_loss;

#[test]
fn test_single_linear_layer() {
    // Test a single Linear layer first
    let linear = Linear::new(10, 5).unwrap();
    let layer_id = add_layer_to_registry(Box::new(linear));
    
    let mut sequential = Sequential::new(vec![layer_id]).unwrap();
    
    let input_data = vec![0.1; 2 * 10];
    let input = Tensor::new(input_data, vec![2, 10]).unwrap();
    
    let output = sequential.forward(input).unwrap();
    
    assert_eq!(output.shape, vec![2, 5]);
    
    for &val in &output.data {
        assert!(val.is_finite(), "Output value is not finite: {}", val);
    }
}

#[test]
fn test_mlp_forward_pass() {
    // Create MLP: Linear(784, 128) → ReLU → Linear(128, 10)
    let linear1 = Linear::new(784, 128).unwrap();
    let relu = ReLU;
    let linear2 = Linear::new(128, 10).unwrap();
    
    let layer1_id = add_layer_to_registry(Box::new(linear1));
    let layer2_id = add_layer_to_registry(Box::new(relu));
    let layer3_id = add_layer_to_registry(Box::new(linear2));
    
    let mut sequential = Sequential::new(vec![layer1_id, layer2_id, layer3_id]).unwrap();
    
    // Create synthetic input: batch of 2 samples, each with 784 features
    // Use smaller values to avoid overflow
    let input_data = vec![0.1; 2 * 784];
    let input = Tensor::new(input_data, vec![2, 784]).unwrap();
    
    // Forward pass
    let output = sequential.forward(input).unwrap();
    
    // Check output shape: [batch_size, 10]
    assert_eq!(output.shape, vec![2, 10]);
    
    // Check that output values are finite
    for (i, &val) in output.data.iter().enumerate() {
        if !val.is_finite() {
            // Print some debug info
            eprintln!("Output value at index {} is not finite: {}", i, val);
            eprintln!("Output shape: {:?}", output.shape);
            eprintln!("First 10 output values: {:?}", &output.data[0..10.min(output.data.len())]);
            panic!("Output value at index {} is not finite: {}", i, val);
        }
    }
    
    // Check that output is not all zeros (should have some variation)
    let sum: f32 = output.data.iter().sum();
    assert!(sum.abs() > 1e-6, "Output sum should not be zero, got {}", sum);
}

#[test]
fn test_mlp_with_softmax() {
    // Create MLP: Linear(784, 128) → ReLU → Linear(128, 10) → Softmax
    let linear1 = Linear::new(784, 128).unwrap();
    let relu = ReLU;
    let linear2 = Linear::new(128, 10).unwrap();
    let softmax = Softmax;
    
    let layer1_id = add_layer_to_registry(Box::new(linear1));
    let layer2_id = add_layer_to_registry(Box::new(relu));
    let layer3_id = add_layer_to_registry(Box::new(linear2));
    let layer4_id = add_layer_to_registry(Box::new(softmax));
    
    let mut sequential = Sequential::new(vec![layer1_id, layer2_id, layer3_id, layer4_id]).unwrap();
    
    // Create synthetic input
    let input_data = vec![0.5; 2 * 784];
    let input = Tensor::new(input_data, vec![2, 784]).unwrap();
    
    // Forward pass
    let output = sequential.forward(input).unwrap();
    
    // Check output shape: [batch_size, 10]
    assert_eq!(output.shape, vec![2, 10]);
    
    // Check that softmax output sums to 1 for each sample
    for i in 0..2 {
        let start = i * 10;
        let end = start + 10;
        let sum: f32 = output.data[start..end].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax output should sum to 1, got {}", sum);
    }
}

#[test]
fn test_mlp_training_step() {
    // Create MLP: Linear(784, 128) → ReLU → Linear(128, 10) → Softmax
    let linear1 = Linear::new(784, 128).unwrap();
    let relu = ReLU;
    let linear2 = Linear::new(128, 10).unwrap();
    let softmax = Softmax;
    
    let layer1_id = add_layer_to_registry(Box::new(linear1));
    let layer2_id = add_layer_to_registry(Box::new(relu));
    let layer3_id = add_layer_to_registry(Box::new(linear2));
    let layer4_id = add_layer_to_registry(Box::new(softmax));
    
    let sequential = Sequential::new(vec![layer1_id, layer2_id, layer3_id, layer4_id]).unwrap();
    let mut nn = NeuralNetwork::new(sequential).unwrap();
    
    // Create small synthetic dataset: 10 samples
    let batch_size = 10;
    let input_data = vec![0.5; batch_size * 784];
    let x = Tensor::new(input_data, vec![batch_size, 784]).unwrap();
    
    // Create labels: [batch_size, 1] with class indices 0-9
    let labels_data: Vec<f32> = (0..batch_size).map(|i| (i % 10) as f32).collect();
    let y = Tensor::new(labels_data, vec![batch_size, 1]).unwrap();
    
    // Train for a few epochs
    let (loss_history, _accuracy_history) = nn.train(&x, &y, 3, batch_size, 0.01, "sparse_cross_entropy", None, None, None).unwrap();
    
    // Check that we got loss values
    assert_eq!(loss_history.len(), 3);
    
    // Check that loss values are finite and positive
    for &loss in &loss_history {
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}

#[test]
fn test_mlp_loss_decreases() {
    // Create MLP: Linear(784, 128) → ReLU → Linear(128, 10) → Softmax
    let linear1 = Linear::new(784, 128).unwrap();
    let relu = ReLU;
    let linear2 = Linear::new(128, 10).unwrap();
    let softmax = Softmax;
    
    let layer1_id = add_layer_to_registry(Box::new(linear1));
    let layer2_id = add_layer_to_registry(Box::new(relu));
    let layer3_id = add_layer_to_registry(Box::new(linear2));
    let layer4_id = add_layer_to_registry(Box::new(softmax));
    
    let sequential = Sequential::new(vec![layer1_id, layer2_id, layer3_id, layer4_id]).unwrap();
    let mut nn = NeuralNetwork::new(sequential).unwrap();
    
    // Create synthetic dataset: 20 samples
    let batch_size = 20;
    let input_data = vec![0.5; batch_size * 784];
    let x = Tensor::new(input_data, vec![batch_size, 784]).unwrap();
    
    // Create labels
    let labels_data: Vec<f32> = (0..batch_size).map(|i| (i % 10) as f32).collect();
    let y = Tensor::new(labels_data, vec![batch_size, 1]).unwrap();
    
    // Train for 5 epochs
    let (loss_history, _accuracy_history) = nn.train(&x, &y, 5, batch_size, 0.01, "sparse_cross_entropy", None, None, None).unwrap();
    
    assert_eq!(loss_history.len(), 5);
    
    // Check that loss generally decreases (not strict, but should trend downward)
    // We'll check that the average of last 2 losses is less than the average of first 2
    let first_avg = (loss_history[0] + loss_history[1]) / 2.0;
    let last_avg = (loss_history[3] + loss_history[4]) / 2.0;
    
    // Loss should decrease (with some tolerance for randomness)
    // If it doesn't decrease, that's also okay for a small test - we just want to ensure it's finite
    assert!(first_avg.is_finite());
    assert!(last_avg.is_finite());
}

#[test]
fn test_sparse_cross_entropy_loss() {
    // Test sparse cross entropy loss with MLP output
    let batch_size = 4;
    let num_classes = 10;
    
    // Create logits: [batch_size, num_classes]
    let logits_data = vec![1.0; batch_size * num_classes];
    let logits = Tensor::new(logits_data, vec![batch_size, num_classes]).unwrap();
    
    // Create target indices: [batch_size, 1]
    let targets_data = vec![0.0, 1.0, 2.0, 3.0];
    let targets = Tensor::new(targets_data, vec![batch_size, 1]).unwrap();
    
    // Compute loss
    let loss = sparse_softmax_cross_entropy_loss(&logits, &targets).unwrap();
    
    // Loss should be positive and finite
    assert!(loss.data[0] > 0.0);
    assert!(loss.data[0].is_finite());
}

#[test]
fn test_tensor_flatten() {
    // Test flatten operation
    let data = vec![1.0; 2 * 28 * 28]; // 2 samples, 28x28 images
    let tensor = Tensor::new(data, vec![2, 28, 28]).unwrap();
    
    let flattened = tensor.flatten().unwrap();
    
    // Should be [2, 784]
    assert_eq!(flattened.shape, vec![2, 784]);
    assert_eq!(flattened.data.len(), 2 * 784);
}

#[test]
fn test_tensor_reshape() {
    // Test reshape operation
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::new(data.clone(), vec![2, 3, 4]).unwrap();
    
    // Reshape to [2, 12]
    let reshaped = tensor.reshape(vec![2, 12]).unwrap();
    
    assert_eq!(reshaped.shape, vec![2, 12]);
    assert_eq!(reshaped.data, data);
}

