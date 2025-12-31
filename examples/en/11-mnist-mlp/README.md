# MNIST MLP Example

This example demonstrates training a minimal Multi-Layer Perceptron (MLP) on the MNIST dataset for digit classification.

## Architecture

The model uses a simple MLP architecture:

```
Input: [batch, 784]
  ↓
Linear(784, 128) + He initialization
  ↓
ReLU
  ↓
Linear(128, 10)
  ↓
Softmax
  ↓
SparseCrossEntropyLoss
  ↓
SGD optimizer (lr=0.001)
```

## Code Example

The example demonstrates the new layer API using `ml.layer.*`:

```datacode
import ml

# Create layers using the new API
layer1 = ml.layer.linear(784, 128)  # Dense layer: 784 → 128
layer2 = ml.layer.relu()            # ReLU activation
layer3 = ml.layer.linear(128, 10)   # Dense layer: 128 → 10
layer4 = ml.layer.softmax()         # Softmax activation

# Create Sequential container
layers = [layer1, layer2, layer3, layer4]
model_seq = ml.sequential(layers)

# Create Neural Network
model = ml.neural_network(model_seq)

# Load and prepare data
dataset_train = ml.load_mnist("train")
x_train = ml.dataset_features(dataset_train)
y_train = ml.dataset_targets(dataset_train)

# Train the model
loss_history = ml.nn_train(model, x_train, y_train, epochs=5, batch_size=32, learning_rate=0.001, loss="sparse_cross_entropy")
```

## Expected Results

- **Accuracy**: 92-95% on test set after 5-10 epochs
- **Training time**: Depends on hardware, typically a few minutes on CPU, significantly faster on GPU

## Usage

### Basic Usage (CPU)
```bash
# If installed globally
datacode mnist_mlp.dc

# Or in development mode
cargo run -- examples/en/10-mnist-mlp/mnist_mlp.dc
```

### With GPU Acceleration

**macOS (Metal):**
```bash
# Development mode with Metal GPU support
cargo run --features metal -- examples/en/10-mnist-mlp/mnist_mlp.dc

# Or using Makefile
make run-metal FILE=examples/en/10-mnist-mlp/mnist_mlp.dc
```

**Linux/Windows (CUDA):**
```bash
# Development mode with CUDA GPU support
cargo run --features cuda -- examples/en/10-mnist-mlp/mnist_mlp.dc

# Or using Makefile
make run-cuda FILE=examples/en/10-mnist-mlp/mnist_mlp.dc
```

**Note:** The example code sets `model.device("metal")` on line 41. If you run without GPU support compiled in, the code will automatically fall back to CPU with a warning message. For optimal performance, compile with the appropriate GPU feature flag (`--features metal` for macOS or `--features cuda` for Linux/Windows).

## Requirements

- MNIST dataset files must be present in `src/lib/ml/datasets/mnist/`:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## Layer API

This example uses the new `ml.layer.*` API for creating layers:

- `ml.layer.linear(in_features, out_features)` - Creates a Linear (Dense) layer
- `ml.layer.relu()` - Creates a ReLU activation layer
- `ml.layer.softmax()` - Creates a Softmax activation layer
- `ml.layer.flatten()` - Creates a Flatten layer (not used in this example)

Layers can also be called directly as functions:
```datacode
layer = ml.layer.linear(10, 5)
output = layer(input_tensor)  # Direct layer call
```

## GPU Support

This example is configured to use GPU acceleration when available. The code includes:
- `model.device("metal")` - Sets the device to Metal (macOS) or can be changed to `"cuda"` (Linux/Windows) or `"cpu"`

**Important:** To enable GPU support, you must compile the project with the appropriate feature flag:
- `--features metal` for macOS
- `--features cuda` for Linux/Windows with NVIDIA GPU

If GPU support is not compiled in, the code will automatically fall back to CPU execution, but training will be significantly slower.

## Notes

- The model uses sparse cross-entropy loss (targets are class indices 0-9)
- SGD optimizer with learning rate 0.001
- Batch size of 32 for training
- Input images are normalized to [0, 1] range (automatically done by `load_mnist`)
- Layers use He initialization by default (good for ReLU activations)

