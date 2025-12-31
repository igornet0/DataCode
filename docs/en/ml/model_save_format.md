# ML Model Save Format in DataCode

## Overview

The ML module in DataCode saves neural networks in a binary format with `.nn` extension. The file format consists of a header, JSON metadata for architecture, and binary tensor data (weights and biases).

## Model File Structure

The model file has the following structure:

```
[Header]
├── Magic number: "DATACODE" (8 bytes)
├── Version: 1 (4 bytes, little-endian)
├── JSON length: JSON string length (4 bytes, little-endian)
│
[Architecture Metadata - JSON]
├── layers: array of layers
├── device: computation device ("cpu" or "metal")
└── training: training information
│
[Binary Tensor Data]
├── Number of tensors (4 bytes)
└── For each tensor:
    ├── Name length (4 bytes)
    ├── Tensor name (UTF-8)
    ├── Number of shape dimensions (4 bytes)
    ├── Shape dimensions (each dimension - 4 bytes, u32)
    └── Tensor data (each value f32 - 4 bytes, little-endian)
```

## Saved Data

### 1. Network Architecture (JSON)

#### Layers

For each layer, the following is saved:

**Linear layers:**
```json
{
  "name": "layer0",
  "type": "Linear",
  "in_features": 784,
  "out_features": 128,
  "trainable": true
}
```

**Activation layers:**
```json
{
  "name": "layer1",
  "type": "ReLU"  // or "Sigmoid", "Tanh", "Softmax", "Flatten"
}
```

#### Device
- `"cpu"` - CPU computations
- `"metal"` - GPU computations (macOS Metal)
- `"cuda"` - GPU computations (NVIDIA CUDA on Linux/Windows)
- `"auto"` - Automatic device selection (GPU if available, otherwise CPU)

### 2. Training Information (training)

#### Training Stages (stages)

Array of `TrainingStage` objects, each containing:

```json
{
  "epochs": 10,                    // Number of epochs in this stage
  "loss": "cross_entropy",         // Loss function type: "cross_entropy", "sparse_cross_entropy", "categorical_cross_entropy", "binary_cross_entropy", "mse"
  "optimizer_type": "Adam",        // Optimizer type: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW"
  "optimizer_params": {            // Optimizer parameters (serialized to JSON)
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
  },
  "frozen_layers": [],            // List of frozen layers
  "trainable_params": 101770,     // Number of trainable parameters
  "frozen_params": 0,             // Number of frozen parameters
  "loss_history": [0.5, 0.3, ...], // Loss history per epoch
  "accuracy_history": [0.75, ...], // Accuracy history per epoch
  "val_loss_history": [0.4, ...], // Validation loss history (optional, can be null)
  "val_accuracy_history": [0.8, ...] // Validation accuracy history (optional, can be null)
}
```

**Important**: Optimizer parameters (`optimizer_params`) are saved in JSON format and include all optimizer-specific parameters (e.g., `beta1`, `beta2` for Adam, `beta` for Momentum, `gamma` for RMSprop, `weight_decay` for AdamW, etc.). This allows exact restoration of optimizer state when loading the model.

#### Legacy Fields (for backward compatibility)

```json
{
  "epochs": 10,                    // Total number of epochs
  "loss": "categorical_cross_entropy",
  "optimizer": "Adam",
  "loss_history": [...],
  "accuracy_history": [...],
  "val_loss_history": [...],
  "val_accuracy_history": [...]
}
```

### 3. Model Parameters (Tensors)

For each Linear layer, two tensors are saved:

1. **Weights**
   - Name: `"layer{N}.weight"`
   - Shape: `[in_features, out_features]`
   - Data: array of `f32` weight values

2. **Biases**
   - Name: `"layer{N}.bias"`
   - Shape: `[out_features]` or `[1, out_features]` (automatically converted to `[1, out_features]` on load for correct broadcasting)
   - Data: array of `f32` bias values

**Important:** 
- All tensors are converted to CPU before saving, even if the model was trained on GPU.
- Bias format can be either `[out_features]` or `[1, out_features]` — both formats are automatically converted to `[1, out_features]` on load to ensure correct broadcasting during computations.

## Example JSON Architecture Structure

```json
{
  "device": "cpu",  // Can be "cpu", "metal", "cuda" or "auto"
  "layers": [
    {
      "name": "layer0",
      "type": "Linear",
      "in_features": 784,
      "out_features": 128,
      "trainable": true
    },
    {
      "name": "layer1",
      "type": "ReLU"
    },
    {
      "name": "layer2",
      "type": "Linear",
      "in_features": 128,
      "out_features": 10,
      "trainable": true
    }
  ],
  "training": {
    "stages": [
      {
        "epochs": 1,
        "loss": "cross_entropy",  // Can be "cross_entropy", "sparse_cross_entropy", "categorical_cross_entropy", "binary_cross_entropy", "mse"
        "optimizer_type": "Adam",
        "optimizer_params": {
          "lr": 0.001,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        },
        "frozen_layers": [],
        "trainable_params": 101770,
        "frozen_params": 0,
        "loss_history": [0.7523],
        "accuracy_history": [0.7523],
        "val_loss_history": null,
        "val_accuracy_history": null
      }
    ],
    "epochs": 1,
    "loss": "cross_entropy",  // Legacy field for backward compatibility
    "optimizer": "Adam",      // Legacy field for backward compatibility
    "loss_history": [0.7523],
    "accuracy_history": [0.7523],
    "val_loss_history": null,
    "val_accuracy_history": null
  }
}
```

## Supported Optimizers and Their Parameters

### SGD
```json
{"lr": 0.01}
```

### Momentum
```json
{"lr": 0.01, "beta": 0.9}
```

### NAG (Nesterov Accelerated Gradient)
```json
{"lr": 0.01, "beta": 0.9}
```

### Adagrad
```json
{"lr": 0.01, "epsilon": 1e-8}
```

### RMSprop
```json
{"lr": 0.001, "gamma": 0.99, "epsilon": 1e-8}
```

### Adam
```json
{"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
```

### AdamW
```json
{"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.01}
```

## Supported Loss Functions

- `"cross_entropy"` - Sparse cross entropy for multi-class classification (class indices [N,1])
- `"sparse_cross_entropy"` - Deprecated alias for `cross_entropy` (saved as `"sparse_cross_entropy"`)
- `"categorical_cross_entropy"` - Categorical cross entropy for multi-class classification (one-hot [N,C])
- `"binary_cross_entropy"` - for binary classification
- `"mse"` - Mean Squared Error for regression

**Note**: `"cross_entropy"` is saved in the model file as `"cross_entropy"` (not as `"sparse_softmax_cross_entropy"`). When loading the model, these values are used to restore training history, but do not affect loss function selection during further training.

## Model File Size

Model file size depends on:

1. **Architecture size:**
   - Number of layers
   - Layer sizes (in_features × out_features)

2. **Number of tensors:**
   - For each Linear layer: 2 tensors (weights + bias)
   - Tensor size = product of shape dimensions × 4 bytes (f32)

3. **Training history:**
   - Number of epochs
   - Number of training stages
   - Size of loss_history and accuracy_history arrays

**Calculation example:**
- Layer: 784 → 128
  - weights: 784 × 128 × 4 bytes = 401,408 bytes
  - bias: 128 × 4 bytes = 512 bytes
  - Total: ~402 KB per layer

## Versioning

Current format version: **1**

When loading a model, the following is checked:
- Magic number must be "DATACODE"
- Version must be 1 (support for other versions may be added in the future)

## Usage

### Saving a Model

```datacode
import ml

model = ml.nn([784, 128, 10])
# ... training the model ...
ml.save_model(model, "my_model.nn")
```

### Loading a Model

```datacode
import ml

model = ml.load("my_model.nn")
ml.model_info(model)  // Show model information
```

## Notes

1. **Device on load:** Models are always loaded on CPU, even if they were saved with GPU. This is done to avoid Metal initialization issues on some systems.

2. **Frozen layers:** Information about frozen layers is saved in training stages, but on load all layers will be trainable. It's necessary to explicitly freeze layers after loading if required.

3. **Training history:** Complete training history is saved, allowing analysis of the training process after loading the model.

4. **Backward compatibility:** The format supports legacy fields for compatibility with older model versions.
