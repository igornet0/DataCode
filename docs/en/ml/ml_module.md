# DataCode ML Module

The `ml` module provides a complete set of functions for machine learning in DataCode, including tensor operations, neural network creation, model training, and data handling.

**📚 Usage examples:**
- Basic examples: [`examples/en/11-mnist-mlp/`](../../examples/en/11-mnist-mlp/)
- Training process: [Neural Network Training Flow](./training_flow.md)
- Model save format: [Model Save Format](./model_save_format.md)

## Contents

1. [Introduction](#introduction)
2. [Tensor Operations](#tensor-operations)
3. [Graph Operations](#graph-operations)
4. [Linear Regression](#linear-regression)
5. [Optimizers](#optimizers)
6. [Loss Functions](#loss-functions)
7. [Dataset Functions](#dataset-functions)
8. [Layer Functions](#layer-functions)
9. [Neural Network Functions](#neural-network-functions)
10. [Object Methods](#object-methods)
11. [Usage Examples](#usage-examples)

---

## Introduction

The `ml` module is imported at the beginning of the program:

```datacode
import ml
```

After import, all module functions are available through the `ml.` prefix, for example:
- `ml.tensor()` - create tensor
- `ml.neural_network()` - create neural network
- `ml.layer.linear()` - create linear layer

The module supports work on CPU and GPU (Metal for macOS, CUDA for Linux/Windows).

---

## Tensor Operations

### `ml.tensor(data, shape?)`

Creates a tensor from data. Shape can be automatically determined from data structure.

**Arguments:**
- `data` (array | number) - data for tensor (array of numbers or nested arrays)
- `shape` (array, optional) - explicit tensor shape `[dim1, dim2, ...]`

**Returns:** `tensor` - tensor object

**Examples:**
```datacode
# Automatic shape determination
t1 = ml.tensor([1.0, 2.0, 3.0])  # Shape: [3]
t2 = ml.tensor([[1, 2], [3, 4]])  # Shape: [2, 2]

# Explicit shape specification
t3 = ml.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])  # Shape: [2, 2]
```

---

### `ml.shape(tensor)`

Returns tensor shape.

**Arguments:**
- `tensor` (tensor) - tensor

**Returns:** `array` - array of dimensions for each axis

**Examples:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
shape = ml.shape(t)  # [2, 2]
```

---

### `ml.data(tensor)`

Returns tensor data as a flat array.

**Arguments:**
- `tensor` (tensor) - tensor

**Returns:** `array` - array of numbers

**Examples:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
data = ml.data(t)  # [1.0, 2.0, 3.0, 4.0]
```

---

### `ml.add(t1, t2)`

Element-wise addition of two tensors.

**Arguments:**
- `t1` (tensor) - first tensor
- `t2` (tensor) - second tensor

**Returns:** `tensor` - addition result

**Examples:**
```datacode
t1 = ml.tensor([1, 2, 3])
t2 = ml.tensor([4, 5, 6])
result = ml.add(t1, t2)  # [5, 7, 9]
```

---

### `ml.sub(t1, t2)`

Element-wise subtraction of two tensors.

**Arguments:**
- `t1` (tensor) - first tensor
- `t2` (tensor) - second tensor

**Returns:** `tensor` - subtraction result

**Examples:**
```datacode
t1 = ml.tensor([5, 7, 9])
t2 = ml.tensor([1, 2, 3])
result = ml.sub(t1, t2)  # [4, 5, 6]
```

---

### `ml.mul(t1, t2)`

Element-wise multiplication of two tensors.

**Arguments:**
- `t1` (tensor) - first tensor
- `t2` (tensor) - second tensor

**Returns:** `tensor` - multiplication result

**Examples:**
```datacode
t1 = ml.tensor([2, 3, 4])
t2 = ml.tensor([5, 6, 7])
result = ml.mul(t1, t2)  # [10, 18, 28]
```

---

### `ml.matmul(t1, t2)`

Matrix multiplication of two tensors.

**Arguments:**
- `t1` (tensor) - first tensor (shape `[n, m]`)
- `t2` (tensor) - second tensor (shape `[m, k]`)

**Returns:** `tensor` - matrix multiplication result (shape `[n, k]`)

**Examples:**
```datacode
t1 = ml.tensor([[1, 2], [3, 4]])  # [2, 2]
t2 = ml.tensor([[5, 6], [7, 8]])  # [2, 2]
result = ml.matmul(t1, t2)  # [[19, 22], [43, 50]]
```

---

### `ml.transpose(tensor)`

Transposes a tensor.

**Arguments:**
- `tensor` (tensor) - tensor (shape `[n, m]`)

**Returns:** `tensor` - transposed tensor (shape `[m, n]`)

**Examples:**
```datacode
t = ml.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
result = ml.transpose(t)  # [[1, 4], [2, 5], [3, 6]] - shape [3, 2]
```

---

### `ml.sum(tensor)`

Sum of all tensor elements.

**Arguments:**
- `tensor` (tensor) - tensor

**Returns:** `number` - sum of all elements

**Examples:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
s = ml.sum(t)  # 10.0
```

---

### `ml.mean(tensor)`

Mean value of all tensor elements.

**Arguments:**
- `tensor` (tensor) - tensor

**Returns:** `number` - mean value

**Examples:**
```datacode
t = ml.tensor([[1, 2], [3, 4]])
m = ml.mean(t)  # 2.5
```

---

### `ml.max_idx(tensor)`

Finds indices of maximum elements. For 1D tensors returns one index, for multidimensional - array of indices for each slice along the first dimension.

**Arguments:**
- `tensor` (tensor) - tensor

**Returns:** `array` - array of indices

**Examples:**
```datacode
t = ml.tensor([3, 1, 4, 1, 5])
idx = ml.max_idx(t)  # [4] - index of maximum element
```

---

### `ml.min_idx(tensor)`

Finds indices of minimum elements. For 1D tensors returns one index, for multidimensional - array of indices for each slice along the first dimension.

**Arguments:**
- `tensor` (tensor) - tensor

**Returns:** `array` - array of indices

**Examples:**
```datacode
t = ml.tensor([3, 1, 4, 1, 5])
idx = ml.min_idx(t)  # [1] or [3] - index of minimum element
```

---

## Graph Operations

Computation graph is used for automatic differentiation (autograd).

### `ml.graph()`

Creates a new computation graph.

**Arguments:** none

**Returns:** `graph` - graph object

**Examples:**
```datacode
g = ml.graph()
```

---

### `ml.graph_add_input(graph)`

Adds an input node to the graph.

**Arguments:**
- `graph` (graph) - computation graph

**Returns:** `number` - node ID

**Examples:**
```datacode
g = ml.graph()
input_id = ml.graph_add_input(g)
```

---

### `ml.graph_add_op(graph, op_name, input_node_ids)`

Adds an operation to the graph.

**Arguments:**
- `graph` (graph) - computation graph
- `op_name` (string) - operation name: `"add"`, `"sub"`, `"mul"`, `"matmul"`, `"transpose"`, `"sum"`, `"mean"`
- `input_node_ids` (array) - array of input node IDs

**Returns:** `number` - new node ID

**Examples:**
```datacode
g = ml.graph()
input1 = ml.graph_add_input(g)
input2 = ml.graph_add_input(g)
add_node = ml.graph_add_op(g, "add", [input1, input2])
```

---

### `ml.graph_forward(graph, input_tensors)`

Executes forward pass through the graph.

**Arguments:**
- `graph` (graph) - computation graph
- `input_tensors` (array) - array of input tensors

**Returns:** `null`

**Examples:**
```datacode
g = ml.graph()
input_id = ml.graph_add_input(g)
t1 = ml.tensor([1, 2, 3])
ml.graph_forward(g, [t1])
```

---

### `ml.graph_get_output(graph, node_id)`

Gets output tensor of a node after forward pass.

**Arguments:**
- `graph` (graph) - computation graph
- `node_id` (number) - node ID

**Returns:** `tensor` - output tensor

**Examples:**
```datacode
output = ml.graph_get_output(g, node_id)
```

---

### `ml.graph_backward(graph, output_node_id)`

Executes backward pass (backpropagation) to compute gradients.

**Arguments:**
- `graph` (graph) - computation graph
- `output_node_id` (number) - output node ID

**Returns:** `null`

**Examples:**
```datacode
ml.graph_backward(g, output_node_id)
```

---

### `ml.graph_get_gradient(graph, node_id)`

Gets gradient of a node after backward pass.

**Arguments:**
- `graph` (graph) - computation graph
- `node_id` (number) - node ID

**Returns:** `tensor` - gradient

**Examples:**
```datacode
grad = ml.graph_get_gradient(g, node_id)
```

---

### `ml.graph_zero_grad(graph)`

Zeros all gradients in the graph.

**Arguments:**
- `graph` (graph) - computation graph

**Returns:** `null`

**Examples:**
```datacode
ml.graph_zero_grad(g)
```

---

### `ml.graph_set_requires_grad(graph, node_id, requires_grad)`

Sets whether gradient computation is required for a node.

**Arguments:**
- `graph` (graph) - computation graph
- `node_id` (number) - node ID
- `requires_grad` (bool) - whether gradient is required

**Returns:** `null`

**Examples:**
```datacode
ml.graph_set_requires_grad(g, node_id, true)
```

---

## Linear Regression

### `ml.linear_regression(feature_count)`

Creates a linear regression model.

**Arguments:**
- `feature_count` (number) - number of features

**Returns:** `linear_regression` - model object

**Examples:**
```datacode
model = ml.linear_regression(3)  # 3 features
```

---

### `ml.lr_predict(model, features)`

Prediction for linear regression.

**Arguments:**
- `model` (linear_regression) - model
- `features` (tensor) - features (shape `[batch_size, feature_count]`)

**Returns:** `tensor` - predictions (shape `[batch_size, 1]`)

**Examples:**
```datacode
x = ml.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
predictions = ml.lr_predict(model, x)
```

---

### `ml.lr_train(model, x, y, epochs, lr)`

Trains a linear regression model.

**Arguments:**
- `model` (linear_regression) - model
- `x` (tensor) - features (shape `[batch_size, feature_count]`)
- `y` (tensor) - target values (shape `[batch_size, 1]`)
- `epochs` (number) - number of epochs
- `lr` (number) - learning rate

**Returns:** `array` - loss history

**Examples:**
```datacode
loss_history = ml.lr_train(model, x_train, y_train, 100, 0.01)
```

---

### `ml.lr_evaluate(model, x, y)`

Evaluates a linear regression model (computes MSE).

**Arguments:**
- `model` (linear_regression) - model
- `x` (tensor) - features
- `y` (tensor) - target values

**Returns:** `number` - MSE (Mean Squared Error)

**Examples:**
```datacode
mse = ml.lr_evaluate(model, x_test, y_test)
```

---

## Optimizers

### `ml.sgd(learning_rate)`

Creates an SGD (Stochastic Gradient Descent) optimizer.

**Arguments:**
- `learning_rate` (number) - learning rate

**Returns:** `sgd` - optimizer object

**Examples:**
```datacode
optimizer = ml.sgd(0.001)
```

---

### `ml.sgd_step(optimizer, graph, param_node_ids)`

Executes one SGD optimization step.

**Arguments:**
- `optimizer` (sgd) - optimizer
- `graph` (graph) - computation graph
- `param_node_ids` (array) - array of parameter node IDs

**Returns:** `null`

**Examples:**
```datacode
ml.sgd_step(optimizer, graph, [weight_id, bias_id])
```

---

### `ml.sgd_zero_grad(graph)`

Zeros gradients in the graph (convenience function for SGD).

**Arguments:**
- `graph` (graph) - computation graph

**Returns:** `null`

**Examples:**
```datacode
ml.sgd_zero_grad(graph)
```

---

### `ml.adam(learning_rate, beta1?, beta2?, epsilon?)`

Creates an Adam optimizer.

**Arguments:**
- `learning_rate` (number) - learning rate
- `beta1` (number, optional) - first moment coefficient (default 0.9)
- `beta2` (number, optional) - second moment coefficient (default 0.999)
- `epsilon` (number, optional) - small value for numerical stability (default 1e-8)

**Returns:** `adam` - optimizer object

**Examples:**
```datacode
optimizer = ml.adam(0.001)  # With default parameters
optimizer2 = ml.adam(0.001, 0.9, 0.999, 1e-8)  # With explicit parameters
```

---

### `ml.adam_step(optimizer, graph, param_node_ids)`

Executes one Adam optimization step.

**Arguments:**
- `optimizer` (adam) - optimizer
- `graph` (graph) - computation graph
- `param_node_ids` (array) - array of parameter node IDs

**Returns:** `null`

**Examples:**
```datacode
ml.adam_step(optimizer, graph, [weight_id, bias_id])
```

---

## Loss Functions

### `ml.mse_loss(y_pred, y_true)`

Computes Mean Squared Error.

**Arguments:**
- `y_pred` (tensor) - predicted values
- `y_true` (tensor) - true values

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.mse_loss(predictions, targets)
```

---

### `ml.cross_entropy_loss(logits, class_indices)`

**DEPRECATED**: Use `sparse_softmax_cross_entropy_loss` or `categorical_cross_entropy_loss`.

---

### `ml.binary_cross_entropy_loss(y_pred, y_true)`

Computes Binary Cross Entropy Loss (for binary classification).

**Arguments:**
- `y_pred` (tensor) - predicted probabilities (shape `[batch_size, 1]`)
- `y_true` (tensor) - true labels (shape `[batch_size, 1]`, values 0 or 1)

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.binary_cross_entropy_loss(predictions, targets)
```

---

### `ml.mae_loss(y_pred, y_true)`

Computes Mean Absolute Error.

**Arguments:**
- `y_pred` (tensor) - predicted values
- `y_true` (tensor) - true values

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.mae_loss(predictions, targets)
```

---

### `ml.huber_loss(y_pred, y_true, delta?)`

Computes Huber Loss (combination of MSE and MAE).

**Arguments:**
- `y_pred` (tensor) - predicted values
- `y_true` (tensor) - true values
- `delta` (number, optional) - threshold value (default 1.0)

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.huber_loss(predictions, targets, 1.0)
```

---

### `ml.hinge_loss(y_pred, y_true)`

Computes Hinge Loss (for SVM).

**Arguments:**
- `y_pred` (tensor) - predicted values
- `y_true` (tensor) - true values (usually -1 or 1)

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.hinge_loss(predictions, targets)
```

---

### `ml.kl_divergence(y_pred, y_true)`

Computes Kullback-Leibler Divergence.

**Arguments:**
- `y_pred` (tensor) - predicted probability distributions
- `y_true` (tensor) - true probability distributions

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.kl_divergence(predictions, targets)
```

---

### `ml.smooth_l1_loss(y_pred, y_true)`

Computes Smooth L1 Loss (combination of L1 and L2).

**Arguments:**
- `y_pred` (tensor) - predicted values
- `y_true` (tensor) - true values

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
loss = ml.smooth_l1_loss(predictions, targets)
```

---

### `ml.categorical_cross_entropy_loss(logits, targets)`

Computes Categorical Cross Entropy Loss for multi-class classification with one-hot labels.

**Arguments:**
- `logits` (tensor) - model logits (shape `[batch_size, num_classes]`)
- `targets` (tensor) - one-hot labels (shape `[batch_size, num_classes]`)

**Returns:** `tensor` - loss tensor

**Examples:**
```datacode
# Converting labels to one-hot
y_onehot = ml.onehot(y_train, 10)
loss = ml.categorical_cross_entropy_loss(logits, y_onehot)
```

**Note:** For class index labels, use `sparse_softmax_cross_entropy_loss` through `model.train()` with `loss="cross_entropy"`.

---

## Dataset Functions

### `ml.dataset(table, feature_columns, target_columns)`

Creates a dataset from a table.

**Arguments:**
- `table` (table) - data table
- `feature_columns` (array) - array of feature column names
- `target_columns` (array) - array of target column names

**Returns:** `dataset` - dataset object

**Examples:**
```datacode
ds = ml.dataset(data_table, ["feature1", "feature2"], ["target"])
```

---

### `ml.dataset_features(dataset)`

Extracts feature tensor from dataset.

**Arguments:**
- `dataset` (dataset) - dataset

**Returns:** `tensor` - feature tensor (shape `[num_samples, num_features]`)

**Examples:**
```datacode
x = ml.dataset_features(dataset)
```

---

### `ml.dataset_targets(dataset)`

Extracts target tensor from dataset.

**Arguments:**
- `dataset` (dataset) - dataset

**Returns:** `tensor` - target tensor (shape `[num_samples, num_targets]`)

**Examples:**
```datacode
y = ml.dataset_targets(dataset)
```

---

### `ml.onehot(labels, num_classes?)`

Converts class labels to one-hot encoding.

**Arguments:**
- `labels` (tensor) - class labels (shape `[N]` or `[N, 1]`)
- `num_classes` (number, optional) - number of classes (if not specified, determined as `max(labels) + 1`)

**Returns:** `tensor` - one-hot tensor (shape `[N, num_classes]`)

**Examples:**
```datacode
labels = ml.tensor([[0], [1], [2], [0]])  # [4, 1]
y_onehot = ml.onehot(labels, 3)  # [4, 3]
# Result: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
```

---

### `ml.load_mnist(split)`

Loads the MNIST dataset.

**Arguments:**
- `split` (string) - dataset split: `"train"` or `"test"`

**Returns:** `dataset` - MNIST dataset

**Examples:**
```datacode
train_dataset = ml.load_mnist("train")
test_dataset = ml.load_mnist("test")
```

**Note:** MNIST files must be located in `src/lib/ml/datasets/mnist/`.

---

## Layer Functions

Layers are created through the `ml.layer` submodule:

### `ml.layer.linear(in_features, out_features)`

Creates a linear (fully connected) layer.

**Arguments:**
- `in_features` (number) - number of input features
- `out_features` (number) - number of output features

**Returns:** `layer` - layer object

**Examples:**
```datacode
layer = ml.layer.linear(784, 128)  # 784 inputs, 128 outputs
```

---

### `ml.layer.relu()`

Creates a ReLU activation layer.

**Arguments:** none

**Returns:** `layer` - layer object

**Examples:**
```datacode
relu_layer = ml.layer.relu()
```

---

### `ml.layer.softmax()`

Creates a Softmax activation layer.

**Arguments:** none

**Returns:** `layer` - layer object

**Examples:**
```datacode
softmax_layer = ml.layer.softmax()
```

---

### `ml.layer.flatten()`

Creates a Flatten layer (converts multidimensional tensor to 1D).

**Arguments:** none

**Returns:** `layer` - layer object

**Examples:**
```datacode
flatten_layer = ml.layer.flatten()
```

---

## Neural Network Functions

### `ml.sequential(layers)`

Creates a sequential container of layers.

**Arguments:**
- `layers` (array) - array of layers

**Returns:** `sequential` - sequential container object

**Examples:**
```datacode
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
seq = ml.sequential([layer1, layer2, layer3])
```

---

### `ml.sequential_add(sequential, layer)`

Adds a layer to the sequential container.

**Arguments:**
- `sequential` (sequential) - sequential container
- `layer` (layer) - layer to add

**Returns:** `null`

**Examples:**
```datacode
ml.sequential_add(seq, ml.layer.softmax())
```

---

### `ml.neural_network(sequential)`

Creates a neural network from a sequential container.

**Arguments:**
- `sequential` (sequential) - sequential container of layers

**Returns:** `neural_network` - neural network object

**Examples:**
```datacode
model = ml.neural_network(seq)
```

---

### `ml.nn_forward(model, x)`

Executes forward pass through the neural network.

**Arguments:**
- `model` (neural_network | sequential | linear_regression) - model
- `x` (tensor) - input data (shape `[batch_size, input_features]`)

**Returns:** `tensor` - output data

**Examples:**
```datacode
output = ml.nn_forward(model, x)
```

---

### `ml.nn_train(model, x, y, epochs, batch_size, lr, loss_type, optimizer?, x_val?, y_val?)`

Trains a neural network.

**Arguments:**
- `model` (neural_network) - model
- `x` (tensor) - training features (shape `[num_samples, num_features]`)
- `y` (tensor) - training labels (shape `[num_samples, 1]` for `cross_entropy` or `[num_samples, num_classes]` for `categorical_cross_entropy`)
- `epochs` (number) - number of epochs
- `batch_size` (number) - batch size
- `lr` (number) - learning rate
- `loss_type` (string) - loss function type: `"cross_entropy"`, `"categorical_cross_entropy"`, `"mse"`, `"binary_cross_entropy"`
- `optimizer` (string, optional) - optimizer: `"SGD"`, `"Adam"`, `"Momentum"`, `"NAG"`, `"Adagrad"`, `"RMSprop"`, `"AdamW"` (default `"SGD"`)
- `x_val` (tensor, optional) - validation features
- `y_val` (tensor, optional) - validation labels

**Returns:** `array` - loss history

**Examples:**
```datacode
# Basic training
loss_history = ml.nn_train(model, x_train, y_train, 10, 32, 0.001, "cross_entropy")

# With optimizer and validation
loss_history = ml.nn_train(model, x_train, y_train, 10, 32, 0.001, 
                           "cross_entropy", "Adam", x_val, y_val)
```

---

### `ml.nn_train_sh(model, x, y, epochs, batch_size, lr, loss_type, optimizer, monitor, patience, min_delta, restore_best, x_val?, y_val?)`

Trains a neural network with early stopping and learning rate scheduler.

**Arguments:**
- `model` (neural_network) - model
- `x` (tensor) - training features
- `y` (tensor) - training labels
- `epochs` (number) - maximum number of epochs
- `batch_size` (number) - batch size
- `lr` (number) - initial learning rate
- `loss_type` (string) - loss function type
- `optimizer` (string | null) - optimizer (or `null` for default SGD)
- `monitor` (string) - metric to monitor: `"loss"`, `"val_loss"`, `"acc"`, `"val_acc"`
- `patience` (number) - number of epochs without improvement before stopping
- `min_delta` (number) - minimum change for improvement
- `restore_best` (bool) - whether to restore best weights after stopping
- `x_val` (tensor, optional) - validation features
- `y_val` (tensor, optional) - validation labels

**Returns:** `object` - object with training history:
- `loss` (array) - training loss history
- `val_loss` (array | null) - validation loss history
- `acc` (array) - training accuracy history
- `val_acc` (array | null) - validation accuracy history
- `lr` (array) - learning rate history
- `best_metric` (number) - best metric value
- `best_epoch` (number) - epoch with best metric
- `stopped_epoch` (number) - stopping epoch

**Examples:**
```datacode
history = ml.nn_train_sh(model, x_train, y_train, 50, 32, 0.001,
                         "cross_entropy", "Adam", "val_loss", 5, 0.0001, true,
                         x_val, y_val)
print("Best epoch:", history.best_epoch)
print("Best metric:", history.best_metric)
```

---

### `ml.nn_save(model, path)`

Saves a neural network to a file.

**Arguments:**
- `model` (neural_network) - model
- `path` (path | string) - file path

**Returns:** `null`

**Examples:**
```datacode
ml.nn_save(model, "model.nn")
ml.nn_save(model, path("models/my_model.nn"))
```

---

### `ml.nn_load(path)`

Loads a neural network from a file.

**Arguments:**
- `path` (path | string) - file path

**Returns:** `neural_network` - loaded model

**Examples:**
```datacode
model = ml.nn_load("model.nn")
model = ml.nn_load(path("models/my_model.nn"))
```

---

### `ml.save_model(model, path)`

Alternative name for `ml.nn_save()`.

**Arguments:**
- `model` (neural_network) - model
- `path` (path | string) - file path

**Returns:** `null`

---

### `ml.load(path)`

Alternative name for `ml.nn_load()`.

**Arguments:**
- `path` (path | string) - file path

**Returns:** `neural_network` - loaded model

---

### `ml.set_device(device_name)`

Sets the default device for ML operations.

**Arguments:**
- `device_name` (string) - device name: `"cpu"`, `"cuda"`, `"metal"`, `"auto"`

**Returns:** `string` - name of set device

**Examples:**
```datacode
ml.set_device("metal")  # macOS
ml.set_device("cuda")   # Linux/Windows with NVIDIA GPU
ml.set_device("cpu")    # CPU
```

**Note:** If GPU device is unavailable, automatic fallback to CPU is performed.

---

### `ml.get_device()`

Returns the current default device.

**Arguments:** none

**Returns:** `string` - device name

**Examples:**
```datacode
device = ml.get_device()  # "cpu", "metal", "cuda"
```

---

### `ml.nn_set_device(model, device_name)`

Sets device for a specific model.

**Arguments:**
- `model` (neural_network) - model
- `device_name` (string) - device name

**Returns:** `string` - name of set device

**Examples:**
```datacode
model.device("metal")
```

---

### `ml.nn_get_device(model)`

Returns model device.

**Arguments:**
- `model` (neural_network) - model

**Returns:** `string` - device name

**Examples:**
```datacode
device = ml.nn_get_device(model)
```

---

### `ml.validate_model(model)`

Validates model validity.

**Arguments:**
- `model` (neural_network) - model

**Returns:** `bool` - `true` if model is valid, `false` otherwise

**Examples:**
```datacode
if ml.validate_model(model) {
    print("Model is valid")
}
```

---

### `ml.model_info(model, verbose?, format?, show_graph?)`

Outputs model information.

**Arguments:**
- `model` (neural_network | linear_regression) - model
- `verbose` (bool, optional) - whether to show detailed information (default `false`)
- `format` (string, optional) - output format: `"text"` or `"json"` (default `"text"`)
- `show_graph` (bool, optional) - whether to show computation graph (default `false`)

**Returns:** `null` for `"text"` format, `string` (JSON) for `"json"` format

**Examples:**
```datacode
ml.model_info(model)  # Text information
ml.model_info(model, true, "json")  # JSON format with details
```

---

### `ml.model_get_layer(model, index)`

Gets a model layer by index.

**Arguments:**
- `model` (neural_network) - model
- `index` (number) - layer index (starting from 0)

**Returns:** `layer` - layer object

**Examples:**
```datacode
first_layer = ml.model_get_layer(model, 0)
```

---

### `ml.layer_freeze(layer)`

Freezes a layer (disables parameter updates during training).

**Arguments:**
- `layer` (layer) - layer

**Returns:** `null`

**Examples:**
```datacode
ml.layer_freeze(first_layer)
```

---

### `ml.layer_unfreeze(layer)`

Unfreezes a layer (enables parameter updates during training).

**Arguments:**
- `layer` (layer) - layer

**Returns:** `null`

**Examples:**
```datacode
ml.layer_unfreeze(first_layer)
```

---

## Object Methods

### Tensor

Tensors have the following properties and methods:

#### Properties

- **`tensor.shape`** - returns tensor shape as array
- **`tensor.data`** - returns tensor data as flat array

#### Methods

- **`tensor.max_idx()`** - returns index of maximum element
- **`tensor.min_idx()`** - returns index of minimum element

#### Indexing

- **`tensor[i]`** - access element by index:
  - For 1D tensors: returns scalar value
  - For multidimensional tensors: returns slice along first dimension

**Examples:**
```datacode
t = ml.tensor([[1, 2, 3], [4, 5, 6]])

# Properties
shape = t.shape  # [2, 3]
data = t.data    # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Methods
max_idx = t.max_idx()  # [5] - index of maximum element
min_idx = t.min_idx()  # [0] - index of minimum element

# Indexing
first_row = t[0]  # tensor [1, 2, 3]
second_row = t[1]  # tensor [4, 5, 6]
```

---

### NeuralNetwork

Neural networks have the following methods:

#### Methods

- **`model.train(x, y, epochs, batch_size, lr, loss_type, optimizer?, x_val?, y_val?)`** - train model
- **`model.train_sh(x, y, epochs, batch_size, lr, loss_type, optimizer, monitor, patience, min_delta, restore_best, x_val?, y_val?)`** - train with early stopping
- **`model.save(path)`** - save model
- **`model.device(device_name)`** - set device (equivalent to `ml.nn_set_device()`)
- **`model.get_device()`** - get device (equivalent to `ml.nn_get_device()`)

#### Properties

- **`model.layers[i]`** - access layer by index

**Examples:**
```datacode
# Training
loss_history = model.train(x_train, y_train, 10, 32, 0.001, "cross_entropy")

# Training with validation and optimizer
loss_history = model.train(x_train, y_train, 10, 32, 0.001, 
                           "cross_entropy", "Adam", x_val, y_val)

# Saving
model.save("my_model.nn")

# Setting device
model.device("metal")

# Accessing layers
first_layer = model.layers[0]
```

---

### Dataset

Datasets support indexing:

- **`dataset[i]`** - returns `[features, target]` for i-th sample

**Examples:**
```datacode
dataset = ml.load_mnist("train")
sample = dataset[0]  # [features_tensor, target_value]
features = sample[0]
target = sample[1]
```

---

## Usage Examples

### Complete MLP Training Example on MNIST

```datacode
import ml

# Loading data
train_dataset = ml.load_mnist("train")
test_dataset = ml.load_mnist("test")

x_train = ml.dataset_features(train_dataset)  # [60000, 784]
y_train = ml.dataset_targets(train_dataset)   # [60000, 1]

x_test = ml.dataset_features(test_dataset)    # [10000, 784]
y_test = ml.dataset_targets(test_dataset)     # [10000, 1]

# Creating model
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model_seq = ml.sequential([layer1, layer2, layer3])
model = ml.neural_network(model_seq)

# Setting device (optional)
model.device("metal")  # or "cuda" for Linux/Windows

# Training
loss_history = model.train(x_train, y_train, 
                           10, 64, 0.001, 
                           "cross_entropy", "Adam", x_test, y_test)

# Prediction
predictions = ml.nn_forward(model, x_test)

# Saving model
model.save("mnist_model.nn")
```

### Example with Early Stopping

```datacode
import ml

# ... creating model and loading data ...

# Training with early stopping
history = model.train_sh(x_train, y_train, 50, 32, 0.001,
                         "cross_entropy", "Adam", "val_loss", 5, 0.0001, true,
                         x_val, y_val)

print("Training stopped at epoch:", history.stopped_epoch)
print("Best epoch:", history.best_epoch)
print("Best metric:", history.best_metric)
```

### Tensor Operations Example

```datacode
import ml

# Creating tensors
a = ml.tensor([[1, 2], [3, 4]])
b = ml.tensor([[5, 6], [7, 8]])

# Operations
c = ml.add(a, b)        # Element-wise addition
d = ml.matmul(a, b)     # Matrix multiplication
e = ml.transpose(a)     # Transposition

# Properties and methods
shape = a.shape         # [2, 2]
data = a.data           # [1.0, 2.0, 3.0, 4.0]
max_idx = a.max_idx()   # [3]
sum_val = ml.sum(a)     # 10.0
mean_val = ml.mean(a)   # 2.5
```

### Linear Regression Example

```datacode
import ml

# Creating model
model = ml.linear_regression(3)  # 3 features

# Training
x_train = ml.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = ml.tensor([[6], [15], [24]])
loss_history = ml.lr_train(model, x_train, y_train, 100, 0.01)

# Prediction
x_test = ml.tensor([[2, 3, 4]])
prediction = ml.lr_predict(model, x_test)

# Evaluation
mse = ml.lr_evaluate(model, x_test, y_test)
```

---

## Related Documents

- [Neural Network Training Flow](./training_flow.md) - detailed description of training process
- [Model Save Format](./model_save_format.md) - description of model file format
- [Built-in Functions](../builtin_functions.md) - other DataCode built-in functions
- [MNIST MLP Examples](../../examples/en/11-mnist-mlp/) - practical usage examples

---

**Note:** This documentation is created based on analysis of DataCode source code. All functions are tested and work in the current version of the language.
