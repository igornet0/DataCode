# `ml` Module (Installable Package)

← [Modules](./README.md) · import syntax: [11-modules-and-imports](../../0-syntax/11-modules-and-imports.md)

The **`ml`** module is **not built in** to DataCode. Install it separately as a DPM package from [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib).

ML sources in the main DataCode repository are mirrored in `src/lib/ml/`; the published user package is a separate git repository.

---

## Installation

```bash
dpm add ml
```

The package is listed in the DataCode registry (`datacode_registry_index/config.json`):

- **Name:** `ml`
- **Source:** `git+https://github.com/igornet0/ML-Datacode-lib.git`
- **Minimum DataCode version:** `>=2.0.0`

After cloning, DPM places the package in `<environment>/packages/ml/`. For native `import ml` you need a built library:

- macOS: `libml.dylib`
- Linux: `libml.so`
- Windows: `ml.dll`

Build (in the `ml` package directory):

```bash
cargo build --release
```

The VM looks for `libml.*` in the script base path or in `<dpm>/packages/ml/`. Alternative — `.dcmodule` artifact (see `setup.dcmodule` in [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib)).

---

## Import and usage

```datacode
import ml

# Tensor
t = ml.tensor([[1, 2], [3, 4]])

# Simple neural network
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model = ml.neural_network(ml.sequential([layer1, layer2, layer3]))

# Training
loss_history = model.train(x_train, y_train, 10, 32, 0.001, "cross_entropy")
```

Without the installed package, `import ml` fails: module not found in base path or DPM packages.

---

## Capabilities

The package provides machine learning APIs (exact function list — in [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) documentation):

| Area | Examples |
|------|----------|
| Tensors | creation, arithmetic, matrix multiply |
| Computation graph | automatic differentiation |
| Linear regression | create and train |
| Optimizers | SGD, Adam, … |
| Loss functions | MSE, Cross Entropy, MAE, … |
| Datasets | MNIST loading, etc. |
| Layers and networks | Linear, ReLU, Softmax, Sequential, train/save |

---

## Examples and documentation

| Resource | Description |
|----------|-------------|
| [ML-Datacode-lib / examples](https://github.com/igornet0/ML-Datacode-lib/tree/main/examples) | Examples in the package repository |
| [ML-Datacode-lib / docs](https://github.com/igornet0/ML-Datacode-lib/tree/main/docs) | Package documentation |
| [1 — Examples / MNIST](../../1-examples/11-mnist-mlp.md) | MNIST MLP scenario overview (after installing `ml`) |

---

## Difference from built-in modules

| | Built-in (`plot`, `uuid`, …) | `ml` |
|---|-------------------------------|------|
| Installation | already in runtime | `dpm add ml` + native build |
| Documentation | [2-language/modules/](./README.md) | this file + [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) |
| Global functions | none (only `import ml`) | none |

See also: [module loading](../../200-developers/module_import_system.md), [`.dcmodule`](../../200-developers/dcmodule_artifact.md).
