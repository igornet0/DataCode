# Neural Network Training Flow: Complete Data Path Through Computation Graph

## Overview

This document describes the complete data path during model training through the `model.train()` method in DataCode. The MLP architecture for MNIST is considered: `Input(784) → Linear(128) + ReLU → Linear(10)`.

## Quick Start

Minimal example of model training:

```datacode
import ml

# Loading data
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)   # [60000, 1] - class indices

# Creating model
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model_seq = ml.sequential([layer1, layer2, layer3])
model = ml.neural_network(model_seq)

# Training (cross_entropy uses class indices [N,1])
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val)
```

**Key points**:
- `y_train` has shape `[N, 1]` - these are class indices for `cross_entropy`
- For `categorical_cross_entropy` you need to use one-hot: `ml.onehot(y_train, 10)`
- For more details see section [3.1.1](#311-loss-function-usage-examples)

## Training Call

**`train()` method signature**:
```datacode
(loss_history, accuracy_history) = model.train(
    x,                    # Features tensor [batch_size, num_features]
    y,                    # Targets tensor [batch_size, num_targets] or [batch_size, num_classes]
    epochs,               # Number of epochs
    batch_size,           # Batch size
    learning_rate,        # Learning rate
    loss_type,            # Loss function type: "mse", "cross_entropy", "categorical_cross_entropy", "binary_cross_entropy"
    x_val,                # Optional validation features tensor (can be null)
    y_val,                # Optional validation targets tensor (can be null)
    optimizer             # Optional optimizer: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW" (default "SGD")
)
```

**Return value**: Tuple `(loss_history, accuracy_history)` — two arrays of loss and accuracy values per epoch.

**Example for cross_entropy (class indices [N,1])**:
```datacode
y_train = ml.dataset_targets(dataset)  # [N, 1]
(loss_history, accuracy_history) = model.train(
    x_train, y_train, 
    5, 64, 0.001, 
    "cross_entropy", 
    x_val, y_val,
    "Adam"  # Optional optimizer parameter
)
```

**Training parameters**:
- `epochs` — number of epochs
- `batch_size` — batch size
- `learning_rate` — learning rate
- `loss_type` — loss function: `"cross_entropy"`, `"sparse_cross_entropy"` (deprecated alias for `cross_entropy`), `"categorical_cross_entropy"`, `"mse"`, `"binary_cross_entropy"`
- `x_val` — optional validation features tensor (can be `null`)
- `y_val` — optional validation targets tensor (can be `null`)
- `optimizer` — optional optimizer (default `"SGD"`). For more details see section [5.3](#53-optimizers-table)

Detailed examples for all loss types see in section [3.1.1](#311-loss-function-usage-examples).

### train_sh() Method (with early stopping and LR scheduling)

The `train_sh()` method provides advanced training capabilities with automatic stopping (early stopping) and learning rate scheduling.

**`train_sh()` method signature**:
```datacode
history = model.train_sh(
    x,                    # Features tensor [batch_size, num_features]
    y,                    # Targets tensor [batch_size, num_targets] or [batch_size, num_classes]
    epochs,               # Maximum number of epochs
    batch_size,           # Batch size
    learning_rate,        # Initial learning rate (will be changed by scheduler)
    loss_type,            # Loss function type: "mse", "cross_entropy", "categorical_cross_entropy", "binary_cross_entropy"
    optimizer,            # Optional optimizer: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW" (default "SGD")
    monitor,              # Metric to monitor: "loss", "val_loss", "acc", "val_acc"
    patience,             # Number of epochs to wait before reducing LR or stopping
    min_delta,            # Minimum improvement percentage (e.g., 1.0 means 1%)
    restore_best,         # Whether to restore best weights at the end of training
    x_val,                # Optional validation features tensor (required if monitor starts with "val_")
    y_val                 # Optional validation targets tensor (required if monitor starts with "val_")
)
```

**Return value**: `TrainingHistorySH` object with fields:
- `loss` — loss history on training set
- `val_loss` — loss history on validation set (optional)
- `acc` — accuracy history on training set
- `val_acc` — accuracy history on validation set (optional)
- `lr` — learning rate change history
- `best_metric` — best metric value
- `best_epoch` — epoch with best metric
- `stopped_epoch` — epoch at which training stopped

**Usage example**:
```datacode
history = model.train_sh(
    x_train, y_train,
    100,              # Maximum 100 epochs
    64,               # Batch size
    0.001,            # Initial learning rate
    "cross_entropy",
    "Adam",           # Optimizer
    "val_loss",       # Monitor validation loss
    10,               # Wait 10 epochs without improvement
    1.0,              # Minimum improvement 1%
    true,             # Restore best weights
    x_val, y_val
)

# Access training history
print("Best epoch:", history.best_epoch)
print("Best metric:", history.best_metric)
print("Stopped at epoch:", history.stopped_epoch)
```

## Computation Graph Architecture

### Model Structure
- **Sequential container** contains a list of layers: `[Linear(784→128), ReLU, Linear(128→10)]`
- **Graph (computation graph)** - directed acyclic graph (DAG), where:
  - **Nodes** represent operations or input data
  - **Edges** represent dependencies between operations
  - **Parameters** (weights and biases) are stored as input nodes with `requires_grad = true`

### Model Parameters
For the MLP architecture, the following parameters are created:
1. **Linear Layer 1**:
   - `Linear1_Weight`: [784, 128] - weight matrix
   - `Linear1_Bias`: [1, 128] - bias vector
2. **Linear Layer 2**:
   - `Linear2_Weight`: [128, 10] - weight matrix
   - `Linear2_Bias`: [1, 10] - bias vector

## Complete Training Cycle (One Batch)

### Stage 1: Data Preparation

**File**: `src/lib/ml/model.rs:282-402`

1. **Moving data to device** (CPU/GPU):
   ```rust
   x_batch: [batch_size, 784]  // input batch data
   // For cross_entropy:
   y_batch: [batch_size, 1]    // class indices (0-9)
   // For categorical_cross_entropy:
   y_batch: [batch_size, 10]   // one-hot batch labels
   ```

2. **Zeroing gradients**:
   ```rust
   self.sequential.zero_grad()  // Clears all gradients in the graph
   ```

### Stage 2: Forward Pass

**File**: `src/lib/ml/model.rs:425` → `src/lib/ml/layer.rs:626-720`

#### 2.1. Building Computation Graph

**Sequential.forward()** builds the graph by passing through all layers:

```rust
// Creating input node
input_node_id = graph.add_input()  // Node 0: Input [batch_size, 784]
```

#### 2.2. Pass Through Layer 1: Linear(784 → 128)

**File**: `src/lib/ml/layer.rs:200-333`

**Linear.forward()** creates operations in the graph:

1. **Input shape validation**:
   ```rust
   // Validation: input must be [batch_size, 784]
   assert_eq!(input.shape, [batch_size, 784]);
   ```

2. **Parameter initialization** (if first time):
   ```rust
   Linear1_Weight = graph.add_input()  // Node 1: Linear1_Weight [784, 128]
   Linear1_Bias = graph.add_input()   // Node 2: Linear1_Bias [1, 128]
   ```

3. **Matrix multiplication**:
   ```rust
   Linear1_MatMul = graph.add_op(MatMul, [input_node, Linear1_Weight])
   // Node 3: Linear1_MatMul [batch_size, 784] @ [784, 128] = [batch_size, 128]
   // Shape check: output [batch_size, 128]
   ```

4. **Adding bias** (with broadcast):
   ```rust
   Linear1_Add = graph.add_op(Add, [Linear1_MatMul, Linear1_Bias])
   // Node 4: Linear1_Add [batch_size, 128] + [1, 128] (broadcast) = [batch_size, 128]
   // Broadcast: [1, 128] automatically expands to [batch_size, 128]
   // Shape check: output [batch_size, 128]
   ```

**Data path**: `x_batch [batch_size, 784]` → `Linear1_MatMul` → `[batch_size, 128]` → `Linear1_Add` → `[batch_size, 128]`

#### 2.3. Pass Through Layer 2: ReLU

**File**: `src/lib/ml/layer.rs:428-431`

```rust
// Input shape validation
assert_eq!(Linear1_Add.shape, [batch_size, 128]);

ReLU1 = graph.add_op(ReLU, [Linear1_Add])
// Node 5: ReLU1 [batch_size, 128] → [batch_size, 128]
// ReLU(x) = max(0, x) - applied element-wise
// Shape preserved: [batch_size, 128]
```

**Data path**: `[batch_size, 128]` → `ReLU1` → `[batch_size, 128]` (negative values zeroed)

#### 2.4. Pass Through Layer 3: Linear(128 → 10)

**File**: `src/lib/ml/layer.rs:200-333`

1. **Input shape validation**:
   ```rust
   // Validation: ReLU1 must be [batch_size, 128]
   assert_eq!(ReLU1.shape, [batch_size, 128]);
   ```

2. **Parameter initialization**:
   ```rust
   Linear2_Weight = graph.add_input()  // Node 6: Linear2_Weight [128, 10]
   Linear2_Bias = graph.add_input()    // Node 7: Linear2_Bias [1, 10]
   ```

3. **Matrix multiplication**:
   ```rust
   Linear2_MatMul = graph.add_op(MatMul, [ReLU1, Linear2_Weight])
   // Node 8: Linear2_MatMul [batch_size, 128] @ [128, 10] = [batch_size, 10]
   // Shape check: output [batch_size, 10]
   ```

4. **Adding bias** (with broadcast):
   ```rust
   Linear2_Add = graph.add_op(Add, [Linear2_MatMul, Linear2_Bias])
   // Node 9: Linear2_Add [batch_size, 10] + [1, 10] (broadcast) = [batch_size, 10]
   // Broadcast: [1, 10] automatically expands to [batch_size, 10]
   // Shape check: output [batch_size, 10] (logits)
   ```

**Data path**: `[batch_size, 128]` → `Linear2_MatMul` → `[batch_size, 10]` → `Linear2_Add` → `[batch_size, 10]`

#### 2.5. Forward Pass Execution

**File**: `src/lib/ml/graph.rs:140-302`

**Graph.forward()** executes operations in topological order:

1. **Topological sorting** (Kahn's algorithm):
   ```rust
   execution_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   // Order: Input → Params → MatMul → Add → ReLU → MatMul → Add
   ```

2. **Operation execution**:
   - Node 0 (Input): `value = x_batch`
   - Node 1 (Linear1_Weight): `value = Linear1_Weight` (from cache or initialization)
   - Node 2 (Linear1_Bias): `value = Linear1_Bias`
   - Node 3 (Linear1_MatMul): `value = input @ Linear1_Weight`
   - Node 4 (Linear1_Add): `value = Linear1_MatMul + Linear1_Bias` (broadcast)
   - Node 5 (ReLU1): `value = relu(Linear1_Add)`
   - Node 6 (Linear2_Weight): `value = Linear2_Weight`
   - Node 7 (Linear2_Bias): `value = Linear2_Bias`
   - Node 8 (Linear2_MatMul): `value = ReLU1 @ Linear2_Weight`
   - Node 9 (Linear2_Add): `value = Linear2_MatMul + Linear2_Bias` (broadcast)

**Result**: `logits [batch_size, 10]` - logits (raw predictions)

### Stage 3: Loss Function Computation

**File**: `src/lib/ml/model.rs:514-662`

#### 3.1. Loss Functions Table

| Loss name | Targets format | Shape | Description |
|-----------|----------------|-------|-------------|
| `cross_entropy` | class indices | [N,1] | Sparse cross entropy with class indices (int) |
| `sparse_cross_entropy` | class indices | [N,1] | **Deprecated**: alias for `cross_entropy`. Use `cross_entropy` |
| `categorical_cross_entropy` | one-hot | [N,C] | Cross entropy with one-hot encoding |
| `binary_cross_entropy` | binary | [N,1] | Binary classification (values in [0,1]) |
| `mse` | continuous | [N,C] | Mean squared error for regression |

**Important**: 
- `cross_entropy` expects **class indices** [N,1], not one-hot
- `sparse_cross_entropy` — deprecated alias for `cross_entropy`, it's recommended to use `cross_entropy`
- `categorical_cross_entropy` expects **one-hot** [N,C]
- Format is strictly validated, format errors lead to compilation/execution errors

#### 3.1.1. Loss Function Usage Examples

**Correct: cross_entropy with class indices [N,1]**

```datacode
# Loading data
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)   # [60000, 1] - class indices (0-9)

# Training with cross_entropy
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val)
```

**Correct: categorical_cross_entropy with one-hot [N,C]**

```datacode
# Loading data
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)    # [60000, 1] - class indices

# Converting to one-hot
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10] - one-hot encoding

# Training with categorical_cross_entropy
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val_onehot)
```

**INCORRECT: format mismatch with loss_type**

```datacode
# ❌ ERROR: cross_entropy with one-hot
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val_onehot)
# Error: "cross_entropy expects class indices [batch, 1], got [batch, 10]"

# ❌ ERROR: categorical_cross_entropy with class indices
y_train = ml.dataset_targets(dataset)  # [60000, 1]
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val)
# Error: "categorical_cross_entropy expects one-hot targets [batch, 10], got [batch, 1]"
```

When the format is incorrect, the system will output a clear error message with a hint on how to fix the problem.

#### 3.1.2. Loss Type and Accuracy Function Compatibility Table

When computing accuracy, it's important to use the correct function depending on the loss type:

| loss_type | accuracy_function | targets format | Description |
|-----------|-------------------|----------------|-------------|
| `cross_entropy` | `compute_accuracy_sparse` | [N,1] class indices | Compares argmax(logits) with class indices |
| `sparse_cross_entropy` | `compute_accuracy_sparse` | [N,1] class indices | Deprecated alias for `cross_entropy` |
| `categorical_cross_entropy` | `compute_accuracy_categorical` | [N,C] one-hot | Compares argmax(logits) with argmax(one-hot) |
| `binary_cross_entropy` | N/A | [N,1] binary | Accuracy is not computed automatically |
| `mse` | N/A | [N,C] continuous | Accuracy is not applicable for regression |

**Note**: Accuracy functions are called automatically inside `model.train()` based on `loss_type`. For `cross_entropy` and `sparse_cross_entropy`, `compute_accuracy_sparse()` is used, for `categorical_cross_entropy` - `compute_accuracy_categorical()`.

#### 3.2. Adding Target Values to Graph

```rust
target_node_id = graph.add_input()  // Node 10: Targets
graph.nodes[target_node_id].value = Some(y_batch)
```

#### 3.3. Creating Loss Node

For `loss_type = "cross_entropy"` (sparse, class indices [N,1]):

```rust
loss_node = graph.add_op(CrossEntropy, [output_node_id, target_node_id])
// Node 11: CrossEntropy [batch_size, 10], [batch_size, 1] → [1] (scalar)
```

For `loss_type = "categorical_cross_entropy"` (one-hot [N,C]):

```rust
loss_node = graph.add_op(CategoricalCrossEntropy, [output_node_id, target_node_id])
// Node 11: CategoricalCrossEntropy [batch_size, 10], [batch_size, 10] → [1] (scalar)
```

**CrossEntropy (sparse) computation** (`src/lib/ml/graph.rs:267-283`):
1. **Softmax** is applied to logits (inside the operation for numerical stability)
2. **Cross-Entropy** is computed: `loss = -mean(log(softmax(logits)[target_class]))`
3. Result: scalar loss value `[1]`

**CategoricalCrossEntropy computation** (`src/lib/ml/graph.rs`):
1. **Softmax** is applied to logits (inside the operation for numerical stability)
2. **Cross-Entropy** is computed: `loss = -mean(sum(targets * log(softmax(logits))))`
3. Result: scalar loss value `[1]`

**Data path for cross_entropy**: 
- `logits [batch_size, 10]` + `targets [batch_size, 1]` (class indices)
- → `CrossEntropy` 
- → `loss [1]` (scalar)

**Data path for categorical_cross_entropy**: 
- `logits [batch_size, 10]` + `targets [batch_size, 10]` (one-hot)
- → `CategoricalCrossEntropy` 
- → `loss [1]` (scalar)

### Stage 4: Backward Pass

**File**: `src/lib/ml/model.rs:712` → `src/lib/ml/graph.rs:359-750`

#### 4.0. Gradient Initialization (zero_grad)

**Important**: Before each backward pass, all gradients must be zeroed. This happens automatically in Stage 1 (Data Preparation) through the call to `self.sequential.zero_grad()`.

```rust
// Called at the beginning of each batch iteration (Stage 1)
self.sequential.zero_grad()
// Clears all gradients in the graph, including parameter gradients
// This is critical, as gradients accumulate between iterations
```

**Why this is important**:
- Gradients accumulate for parameters between batches
- Without `zero_grad()`, gradients will sum up, leading to incorrect training
- `zero_grad()` resets all gradients before a new forward pass
- Gradients are recomputed for each batch

#### 4.1. Loss Gradient Initialization

```rust
graph.nodes[loss_node].grad = Tensor::ones([1])  // grad_loss = 1.0
```

#### 4.2. Gradient Backpropagation

**Graph.backward()** traverses the graph in **reverse topological order**:

```rust
backward_order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
// Reverse order: Loss → Output → Linear2_Add → Linear2_MatMul → ReLU1 → Linear1_Add → Linear1_MatMul → Params
```

#### 4.3. Gradient Computation for Each Operation

**Node 11: CrossEntropy** (`src/lib/ml/graph.rs:714-749`)

```rust
// Gradient w.r.t. logits: (softmax(logits) - targets) / batch_size
grad_logits = (softmax(logits) - targets) / batch_size
// Shape: [batch_size, 10]
```

**Gradient path**: `loss [1]` → `CrossEntropy.backward()` → `grad_logits [batch_size, 10]`

---

**Node 9: Linear2_Add (Linear2_Bias)** (`src/lib/ml/graph.rs:411-429`)

```rust
// Gradient propagates to both inputs:
grad_Linear2_MatMul = grad_Linear2_Add.clone()  // [batch_size, 10]
grad_Linear2_Bias = grad_Linear2_Add.sum_to_shape([1, 10])  // Sum over batch
```

**Gradient path**: `grad_Linear2_Add [batch_size, 10]` → `Linear2_Add.backward()` → 
- `grad_Linear2_MatMul [batch_size, 10]`
- `grad_Linear2_Bias [1, 10]` (saved in node 7: Linear2_Bias)

---

**Node 8: Linear2_MatMul (Linear2_Weight)** (`src/lib/ml/graph.rs:490-591`)

```rust
// For y = a @ b, where a=[batch, 128], b=[128, 10]
// grad_a = grad_y @ b^T
// grad_b = a^T @ grad_y

grad_ReLU1 = grad_Linear2_MatMul @ Linear2_Weight^T
// [batch_size, 10] @ [10, 128] = [batch_size, 128]

grad_Linear2_Weight = ReLU1^T @ grad_Linear2_MatMul
// [128, batch_size] @ [batch_size, 10] = [128, 10]
```

**Gradient path**: `grad_Linear2_MatMul [batch_size, 10]` → `Linear2_MatMul.backward()` →
- `grad_ReLU1 [batch_size, 128]` (to node 5: ReLU1)
- `grad_Linear2_Weight [128, 10]` (saved in node 6: Linear2_Weight)

---

**Node 5: ReLU1** (`src/lib/ml/graph.rs:625-638`)

```rust
// ReLU gradient: grad = grad * (x > 0)
// If input was > 0, gradient passes, otherwise = 0

grad_Linear1_Add = grad_ReLU1 * mask
// mask[i] = 1.0 if Linear1_Add[i] > 0, else 0.0
// Shape: [batch_size, 128]
```

**Gradient path**: `grad_ReLU1 [batch_size, 128]` → `ReLU1.backward()` → 
- `grad_Linear1_Add [batch_size, 128]` (only for positive values)

---

**Node 4: Linear1_Add (Linear1_Bias)** (`src/lib/ml/graph.rs:411-429`)

```rust
grad_Linear1_MatMul = grad_Linear1_Add.clone()  // [batch_size, 128]
grad_Linear1_Bias = grad_Linear1_Add.sum_to_shape([1, 128])  // Sum over batch
```

**Gradient path**: `grad_Linear1_Add [batch_size, 128]` → `Linear1_Add.backward()` →
- `grad_Linear1_MatMul [batch_size, 128]`
- `grad_Linear1_Bias [1, 128]` (saved in node 2: Linear1_Bias)

---

**Node 3: Linear1_MatMul (Linear1_Weight)** (`src/lib/ml/graph.rs:490-591`)

```rust
grad_input = grad_Linear1_MatMul @ Linear1_Weight^T
// [batch_size, 128] @ [128, 784] = [batch_size, 784]

grad_Linear1_Weight = input^T @ grad_Linear1_MatMul
// [784, batch_size] @ [batch_size, 128] = [784, 128]
```

**Gradient path**: `grad_Linear1_MatMul [batch_size, 128]` → `Linear1_MatMul.backward()` →
- `grad_input [batch_size, 784]` (to node 0: Input, not used)
- `grad_Linear1_Weight [784, 128]` (saved in node 1: Linear1_Weight)

---

#### 4.4. Final Parameter Gradients

After backward pass, gradients are saved in parameter nodes:

- **Node 1 (Linear1_Weight)**: `grad_Linear1_Weight [784, 128]`
- **Node 2 (Linear1_Bias)**: `grad_Linear1_Bias [1, 128]`
- **Node 6 (Linear2_Weight)**: `grad_Linear2_Weight [128, 10]`
- **Node 7 (Linear2_Bias)**: `grad_Linear2_Bias [1, 10]`

**Important**: Gradients accumulate for parameters. The `zero_grad()` function is called at the beginning of each iteration (Stage 1) to reset all gradients before a new forward pass. Gradients are recomputed for each batch.

### Stage 5: Parameter Update (Optimizer Step)

**File**: `src/lib/ml/model.rs:836` → `src/lib/ml/optimizer.rs:36-95`

#### 5.1. SGD Optimizer

**SGD.step()** updates each parameter:

```rust
for each param_node_id in [1, 2, 6, 7]:
    current_value = graph.get_output(param_node_id)  // Current parameter value
    gradient = graph.get_gradient(param_node_id)      // Parameter gradient
    
    // Update: new_value = current_value - lr * gradient
    update = lr * gradient
    new_value = current_value - update
    
    graph.nodes[param_node_id].value = Some(new_value)  // Save new value
```

**Example for Linear1_Weight**:
```rust
Linear1_Weight_new = Linear1_Weight_old - learning_rate * grad_Linear1_Weight
// [784, 128] = [784, 128] - lr * [784, 128]
```

#### 5.2. Saving Updated Parameters

```rust
self.sequential.save_parameter_values()
// Saves updated parameter values to cache for next forward pass
```

#### 5.3. Optimizers Table

DataCode supports various optimizers for updating model parameters. All optimizers work with the same computation graph and use gradients computed in backward pass. The difference between optimizers is only in the parameter update formula.

**General principles**:
- All optimizers update parameters **in-place** (directly modify values in the graph)
- Work through `optimizer.step(graph, param_node_ids)`
- Optimizer choice doesn't affect graph structure or backward pass
- Gradients are computed the same way for all optimizers

##### 5.3.1. Optimizers Table

| Optimizer | Weight update formula | Parameters | Usage example |
|-----------|----------------------|------------|--------------|
| **SGD** | `w = w - η * grad` | `η` — learning rate | `optimizer = "SGD"` or `optimizer = SGD(lr=0.01)` |
| **SGD + Momentum** | `v = β * v + (1-β) * grad`<br>`w = w - η * v` | `η` — lr, `β` — momentum (0.9) | `optimizer = "Momentum"` or `optimizer = Momentum(lr=0.01, beta=0.9)` |
| **Nesterov (NAG)** | `v = β * v + η * grad(w - β*v)`<br>`w = w - v` | `η` — lr, `β` — momentum | `optimizer = "NAG"` or `optimizer = NAG(lr=0.01, beta=0.9)` |
| **Adagrad** | `G = G + grad²`<br>`w = w - η / sqrt(G + ε) * grad` | `η` — lr, `ε` — small number (1e-8) | `optimizer = "Adagrad"` or `optimizer = Adagrad(lr=0.01)` |
| **RMSprop** | `E[g²] = γ * E[g²] + (1-γ) * grad²`<br>`w = w - η / sqrt(E[g²] + ε) * grad` | `η` — lr, `γ` — decay (0.9–0.99), `ε` — small number | `optimizer = "RMSprop"` or `optimizer = RMSprop(lr=0.001, gamma=0.9)` |
| **Adam** | `m = β1*m + (1-β1)*grad`<br>`v = β2*v + (1-β2)*grad²`<br>`m̂ = m / (1-β1^t)`<br>`v̂ = v / (1-β2^t)`<br>`w = w - η * m̂ / (sqrt(v̂) + ε)` | `η` — lr, `β1=0.9`, `β2=0.999`, `ε=1e-8` | `optimizer = "Adam"` or `optimizer = Adam(lr=0.001)` |
| **AdamW** | `m = β1*m + (1-β1)*grad`<br>`v = β2*v + (1-β2)*grad²`<br>`m̂ = m / (1-β1^t)`<br>`v̂ = v / (1-β2^t)`<br>`w = w - η * (m̂ / (sqrt(v̂)+ε) + λ*w)` | `η` — lr, `β1=0.9`, `β2=0.999`, `λ` — weight decay | `optimizer = "AdamW"` or `optimizer = AdamW(lr=0.001, weight_decay=0.01)` |

**Notation in formulas**:
- `w` — weights (model parameters)
- `grad` — parameter gradient
- `η` (eta) — learning rate
- `β` (beta) — momentum coefficient
- `γ` (gamma) — decay rate
- `ε` (epsilon) — small number for numerical stability
- `λ` (lambda) — weight decay coefficient
- `t` — iteration number (time step)
- `m` — first moment estimate (momentum)
- `v` — second moment estimate (velocity)
- `m̂`, `v̂` — bias-corrected moments

##### 5.3.2. Optimizer Usage Examples

**Universal interface in DataCode**:

```datacode
import ml

# Loading data
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)   # [60000, 1] - class indices

# Creating model
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model_seq = ml.sequential([layer1, layer2, layer3])
model = ml.neural_network(model_seq)

# Training settings
epochs = 10
batch_size = 64
learning_rate = 0.001
optimizer_name = "Adam"        # Choose optimizer: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW"
loss_fn = "cross_entropy"

# Preparing validation data
x_val = ml.dataset_features(ml.load_mnist("test"))
y_val = ml.dataset_targets(ml.load_mnist("test"))

# Training model
(loss_history, accuracy_history) = model.train(
    x_train, y_train,
    epochs,
    batch_size,
    learning_rate,
    loss_fn,
    x_val,
    y_val,
    optimizer_name  # Optional parameter
)
```

**Examples for different optimizers**:

```datacode
# SGD (Stochastic Gradient Descent)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.01, 
                                                "cross_entropy", x_val, y_val, "SGD")

# Adam (recommended for most tasks)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val, "Adam")

# RMSprop (good for RNN)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val, "RMSprop")

# AdamW (with weight decay for regularization)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val, "AdamW")
```

##### 5.3.3. Optimizer Selection Recommendations

- **SGD**: Simple and reliable, works well for simple tasks. Requires careful learning rate tuning.
- **SGD + Momentum**: Accelerates SGD convergence, helps overcome local minima.
- **Nesterov (NAG)**: Improved version of Momentum, often converges faster.
- **Adagrad**: Adaptive learning rate, works well for sparse gradients. May decrease learning rate too much.
- **RMSprop**: Solves Adagrad's problem with decaying learning rate. Good for RNN.
- **Adam**: Recommended for most tasks. Adaptive learning rate with momentum. Works well out of the box.
- **AdamW**: Improved version of Adam with correct weight decay implementation. Recommended for modern models.

**Important**: All optimizers work with the same computation graph and use gradients computed in backward pass. The difference is only in the parameter update formula.

### Stage 6: Graph Cleanup

**File**: `src/lib/ml/model.rs:856` → `src/lib/ml/layer.rs:767-820`

```rust
self.sequential.clear_non_parameter_nodes()
// Removes all intermediate nodes (MatMul, Add, ReLU, Loss)
// Keeps only parameter nodes (weights, biases)
// This prevents memory leaks between batches
```

## Computation Graph Visualization

### Model Parameters (preserved between batches)

```
Parameters (green nodes - preserved):
  Linear1_Weight [784, 128]  ──┐
  Linear1_Bias [1, 128]        │
  Linear2_Weight [128, 10]     │─── Updated through backward pass
  Linear2_Bias [1, 10]         │
                               └─── Preserved after clear_non_parameter_nodes()
```

### Forward Pass - blue arrows

```
Input [batch, 784]  (blue arrow →)
  │
  ├─→ Linear1_Weight [784, 128] ──┐
  │                               │
  └─→ Linear1_MatMul ─────────────┼─→ [batch, 128]  (blue arrow →)
                                  │
  Linear1_Bias [1, 128] ──────────┘
                                  │
                                  ↓  (blue arrow →)
                        Linear1_Add [batch, 128]
                                  │
                                  ↓  (blue arrow →)
                        ReLU1 [batch, 128]
                                  │
                                  ├─→ Linear2_Weight [128, 10] ──┐
                                  │                              │
                                  └─→ Linear2_MatMul ────────────┼─→ [batch, 10]  (blue arrow →)
                                                                  │
                        Linear2_Bias [1, 10] ────────────────────┘
                                                                  │
                                                                  ↓  (blue arrow →)
                                                Linear2_Add [batch, 10] (logits)
                                                                  │
                                                                  ↓  (blue arrow →)
                                                CrossEntropy
                                                                  │
                                                                  ↓  (blue arrow →)
                                                 Loss [1]
```

### Backward Pass - red arrows

```
Loss [1] (grad = 1.0)  (red arrow ←)
  │
  ↓  (red arrow ←)
CrossEntropy.backward()
  │
  ↓ grad_logits [batch, 10]  (red arrow ←)
Linear2_Add.backward()
  │
  ├─→ grad_Linear2_MatMul [batch, 10]  (red arrow ←)
  └─→ grad_Linear2_Bias [1, 10] ──→ Linear2_Bias updated  (red arrow ←)
  │
  ↓  (red arrow ←)
Linear2_MatMul.backward()
  │
  ├─→ grad_ReLU1 [batch, 128]  (red arrow ←)
  └─→ grad_Linear2_Weight [128, 10] ──→ Linear2_Weight updated  (red arrow ←)
  │
  ↓  (red arrow ←)
ReLU1.backward() (mask: x > 0)
  │
  ↓ grad_Linear1_Add [batch, 128] (only positive)  (red arrow ←)
Linear1_Add.backward()
  │
  ├─→ grad_Linear1_MatMul [batch, 128]  (red arrow ←)
  └─→ grad_Linear1_Bias [1, 128] ──→ Linear1_Bias updated  (red arrow ←)
  │
  ↓  (red arrow ←)
Linear1_MatMul.backward()
  │
  ├─→ grad_input [batch, 784] (not used)  (red arrow ←)
  └─→ grad_Linear1_Weight [784, 128] ──→ Linear1_Weight updated  (red arrow ←)
```

**Legend**:
- **Blue arrows (→)**: Forward pass - data flow from inputs to outputs
- **Red arrows (←)**: Backward pass - gradient flow from loss to parameters
- **Green nodes**: Parameters - preserved between batches, updated through optimizer

## Complete Cycle for One Batch

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Preparation: x_batch, y_batch, zero_grad()             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Forward Pass:                                            │
│    x → Linear_1 → ReLU → Linear_2 → logits                   │
│    logits + targets → CrossEntropy → loss                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Backward Pass:                                            │
│    loss → grad_logits → grad_weight_2, grad_bias_2           │
│    → grad_relu → grad_weight_1, grad_bias_1                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Optimizer Step:                                          │
│    weight_1 = weight_1 - lr * grad_weight_1                 │
│    bias_1 = bias_1 - lr * grad_bias_1                      │
│    weight_2 = weight_2 - lr * grad_weight_2                  │
│    bias_2 = bias_2 - lr * grad_bias_2                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Saving and cleanup:                                      │
│    save_parameter_values()                                   │
│    clear_non_parameter_nodes()                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Files and Functions

### Main Components

1. **model.rs** (`src/lib/ml/model.rs`):
   - `NeuralNetwork::train()` - main training loop
   - `NeuralNetwork::forward()` - forward pass call

2. **layer.rs** (`src/lib/ml/layer.rs`):
   - `Sequential::forward()` - graph building and execution
   - `Linear::forward()` - creating Linear layer operations
   - `ReLU::forward()` - creating ReLU operation

3. **graph.rs** (`src/lib/ml/graph.rs`):
   - `Graph::forward()` - executing operations in topological order
   - `Graph::backward()` - gradient backpropagation
   - `Graph::topological_sort()` - determining execution order

4. **optimizer.rs** (`src/lib/ml/optimizer.rs`):
   - `SGD::step()` - updating parameters by gradients (SGD)
   - `Adam::step()` - updating parameters by gradients (Adam)
   - Support for other optimizers: Momentum, NAG, Adagrad, RMSprop, AdamW
   - For more details see section [5.3](#53-optimizers-table)

5. **loss.rs** (`src/lib/ml/loss.rs`):
   - `softmax_cross_entropy_loss()` - loss function computation

## Implementation Features

### 1. Automatic Differentiation (Autograd)

**Main capabilities**:
- Computation graph is built **dynamically** during forward pass
- Gradients are computed automatically through backward pass
- No manual derivative computation required

**Advanced capabilities**:

1. **Branching**:
   - Graph can have multiple computation paths
   - Backward pass correctly handles all branches
   - Gradients are summed at merge points

2. **Conditional constructs**:
   - Support for `if/else` in computation graph
   - Gradients propagate only along active branch
   - Conditional operations are integrated into autograd

3. **Loops**:
   - Support for `for/while` loops in graph
   - Gradients accumulate through loop iterations
   - Backward pass goes through all iterations

4. **Dynamic structure**:
   - Graph is rebuilt on each forward pass
   - This allows changing architecture between iterations
   - Models with variable structure are supported

**Advantages**:
- Flexibility: can build complex architectures
- Convenience: no need to manually compute gradients
- Safety: system guarantees computation correctness

### 2. Memory Management
- Intermediate nodes are deleted after each batch
- Only parameters (weights, biases) are preserved
- Parameters are cached for next forward pass

### 3. Device Support (CPU/GPU)
- All operations support work on CPU and GPU
- Data is automatically moved to the required device
- Gradients are saved on the same device as parameters

### 5. Batch Size Flexibility

**Support for various batch sizes**:
- `batch_size` can be **1** (online learning / stochastic gradient descent)
- `batch_size` can equal dataset size (full batch gradient descent)
- `batch_size` can be any value between them (mini-batch)

**How it works**:
- Topological sorting works for **any** batch_size
- Backward pass works correctly for any batch_size
- Gradients are automatically normalized by batch_size
- Formula: `gradient = gradient / batch_size` (inside loss functions)

**Examples**:
```datacode
# Online learning (batch_size = 1)
model.train(x_train, y_train, 10, 1, 0.001, "cross_entropy", ...)

# Mini-batch (batch_size = 64)
model.train(x_train, y_train, 10, 64, 0.001, "cross_entropy", ...)

# Full batch (batch_size = dataset_size)
model.train(x_train, y_train, 10, x_train.shape[0], 0.001, "cross_entropy", ...)
```

**Performance**:
- Smaller batch_size → more iterations, but less memory
- Larger batch_size → fewer iterations, but more memory
- Optimal batch_size depends on model size and available memory

### 4. Numerical Stability

**Softmax and CrossEntropy**:
- CrossEntropy and CategoricalCrossEntropy use **fused Softmax** to avoid overflow
- **Log-sum-exp trick** is applied for numerical stability:
  ```rust
  // Instead of direct softmax computation:
  // softmax(x) = exp(x) / sum(exp(x))  // Can overflow!
  
  // Stable formula is used:
  max_x = max(x)
  log_softmax(x) = x - max_x - log(sum(exp(x - max_x)))
  ```
- This prevents overflow at large logit values
- Formula: `log_softmax(x) = x - log(sum(exp(x - max(x))))`
- All loss operations use this trick automatically

**ReLU**:
- ReLU prevents gradient propagation through negative values
- Gradient = 0 for negative inputs, which stabilizes training

**Other measures**:
- All operations check for NaN/Inf
- Gradients are normalized by batch_size automatically
- Epsilon values are used to prevent division by zero

## Common Errors and Solutions

### 1. Targets Format Mismatch with loss_type

**Error**: Using one-hot with `cross_entropy`
```datacode
# ❌ INCORRECT
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val_onehot)
# Error: "cross_entropy expects class indices [batch, 1], got [batch, 10]"
```

**Solution**: Use class indices for `cross_entropy`
```datacode
# ✅ CORRECT
y_train = ml.dataset_targets(dataset)  # [60000, 1]
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val)
```

**Or**: Use `categorical_cross_entropy` with one-hot
```datacode
# ✅ CORRECT
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val_onehot)
```

### 2. Using Class Indices with categorical_cross_entropy

**Error**:
```datacode
# ❌ INCORRECT
y_train = ml.dataset_targets(dataset)  # [60000, 1]
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val)
# Error: "categorical_cross_entropy expects one-hot targets [batch, 10], got [batch, 1]"
```

**Solution**: Convert to one-hot
```datacode
# ✅ CORRECT
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val_onehot)
```

### 3. Forgetting zero_grad() (not applicable - called automatically)

**Note**: In DataCode `zero_grad()` is called automatically at the beginning of each batch iteration (Stage 1). No need to call manually. If gradients accumulate incorrectly, this may be a system bug, not a user error.

### 4. Incorrect Targets Format for Accuracy

**Problem**: Accuracy is computed automatically based on `loss_type`. If incorrect format is used, accuracy may be computed incorrectly.

**Solution**: Ensure targets format matches `loss_type`:
- `cross_entropy` → `[N, 1]` class indices → `compute_accuracy_sparse()`
- `categorical_cross_entropy` → `[N, C]` one-hot → `compute_accuracy_categorical()`

### 5. Batch Size Mismatch

**Error**: Different batch sizes for train and validation
```datacode
# May cause problems if batch_size doesn't divide dataset size
```

**Solution**: System automatically handles last incomplete batch. But it's better to use batch_size that divides dataset size.

## Conclusion

Model training in DataCode happens through building a dynamic computation graph, which:
1. Executes forward pass to get predictions
2. Computes loss function
3. Propagates gradients back through graph (backward pass)
4. Updates parameters using optimizer
5. Cleans intermediate nodes for next batch

This approach provides flexibility, automatic differentiation, and efficient memory usage.

**Key principles**:
- ✅ `cross_entropy` uses class indices `[N, 1]`
- ✅ `categorical_cross_entropy` uses one-hot `[N, C]`
- ✅ Format is strictly validated with clear error messages
- ✅ Gradients are zeroed automatically before each batch
- ✅ Memory is managed automatically through intermediate node cleanup
