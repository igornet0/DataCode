# Modules and Imports

This document describes how to **import** and **use modules** in DataCode: syntax, types of modules (single file vs package), dotted names, base path, and built-in modules. For implementation details (bytecode, ModuleObject, remapping), see [Core / Internals — Module Import System](./internals/module_import_system.md).

---

## Import syntax

### Import a module (module object)

```datacode
import ml
import plot
import settings_env
```

This loads the module and binds its **namespace object** to the given name. You then access exports via the module name:

```datacode
import ml
let x = ml.load_mnist()
```

### Import specific names from a module

```datacode
from ml import load_mnist, NeuralNetwork
from settings_env import load_env, Settings
from core.config import get_settings, load_settings
```

- **Named**: `from M import X, Y` — imports `X` and `Y` from module `M` into the current scope.
- **Aliased**: `from M import X as Z` — imports `X` under the name `Z`.
- **All**: `from M import *` — imports all names exported by `M` (subject to runtime behavior).

You can mix: `from M import a, b as c, *`.

---

## How modules are found (base path)

When you run a script from the **command line** (e.g. `datacode path/to/script.dc`), the **base path** is set to the **directory containing that script**. All `import` / `from` resolve relative to this base path (and optionally [DPM package paths](#dpm-package-paths)).

- Running `datacode examples/en/15-modules/main.dc` → base path = `examples/en/15-modules/`.
- So `from core.config import get_settings` looks for `examples/en/15-modules/core/config/__lib__.dc`.

If you run code **without** a script file (e.g. REPL or `run("from foo import bar")` without a base path), local `.dc` modules are **not** found; only [built-in modules](#built-in-modules) are available. When using the library API, pass a `base_path` so that local imports work (see [Internals](./internals/module_import_system.md)).

---

## Two kinds of local modules

Local modules are resolved under the **base path** (and DPM paths). There are two forms:

### 1. Single-file module

A file named `<name>.dc` is imported as module `name`:

- Path: `<base_path>/<name>.dc`
- Example: `utils.dc` → `import utils` or `from utils import foo`

### 2. Package (directory with `__lib__.dc`)

A **package** is a directory that contains a file `__lib__.dc`. The package is imported by the directory name; the code that runs is the content of `__lib__.dc`.

- Path: `<base_path>/<name>/__lib__.dc`
- Example: directory `core/config/` with `__lib__.dc` inside → imported as `core.config` (see [Dotted names](#dotted-names-packages)).

The loader **prefers** a package over a file: if both `<name>/__lib__.dc` and `<name>.dc` exist, the package is used.

---

## Dotted names (packages)

You can import a **submodule** with a dotted name. Each segment is resolved in order:

- `from core.config import get_settings`:
  1. Find package or file `core` under base path → directory `core/` with `core/__lib__.dc` (or file `core.dc`).
  2. Under that directory, find `config` → `core/config/__lib__.dc` (or `core/config.dc`).
  3. The loaded module is `core.config`; its exports (e.g. `get_settings`) come from `core/config/__lib__.dc`.

So for **nested packages**, every segment except the last must be a **package** (directory with `__lib__.dc`). For example, for `core.config` to work you need:

- `core/__lib__.dc` (can be minimal or empty),
- `core/config/__lib__.dc` (actual exports).

---

## What gets exported

For a **local .dc module** (single file or `__lib__.dc`), the module’s **global variables and functions** form the export namespace. Everything defined at top level (variables, functions, classes) is available to importers.

- `from M import f` — `f` must be a global in that module.
- `import M` — the namespace object gives access to all those globals (e.g. `M.f`).

---

## Built-in modules

These modules are built into the runtime; they do **not** require a corresponding `.dc` file or package under base path:

| Module           | Description |
|------------------|-------------|
| `ml`             | Machine learning (tensors, graphs, neural networks, datasets, etc.) |
| `plot`           | Images, windows, charts (bar, line, pie, heatmap, subplots) |
| `settings_env`   | Loading .env, Settings, Config, Field |
| `uuid`           | UUID generation (v4, v7), parse, to_string, bytes, v3/v5 |
| `database`       | Database engine and DatabaseCluster |

Example:

```datacode
from database import engine, DatabaseCluster
from uuid import v4, v7
```

If a name is **not** a built-in module and no local module is found (under base path or DPM paths), you get a runtime error such as: *"Module 'X' not found. Searched in base path and DPM packages."*

---

## DPM package paths

In addition to the script’s base path, the runtime can use extra search paths (e.g. for **DPM**-installed packages). Those paths are used when resolving module names the same way as the base path (file `<name>.dc` or directory `<name>/__lib__.dc`). How to set them depends on the host (CLI, HTTP server, or library API).

---

## Full example (local package)

Layout:

```
my_app/
├── main.dc
└── core/
    ├── __lib__.dc
    └── config/
        ├── __lib__.dc
        ├── base.dc
        ├── config.dc
        ├── dev_config.dc
        └── prod_config.dc
```

- `main.dc`: `from core.config import get_settings, load_settings`
- `core/__lib__.dc`: can be empty or a short comment (so that `core` is a package).
- `core/config/__lib__.dc`: defines `get_settings`, `load_settings`, and any other exports; it may use `from base import ...`, `from config import ...`, etc., which resolve relative to `core/config/`.

Run from project root:

```bash
cargo run --bin datacode my_app/main.dc
```

A complete runnable example is in the repository:

- **EN**: [examples/en/15-modules/](../../examples/en/15-modules/) (see [README](../../examples/en/15-modules/README.md))

---

## Summary

| Topic | Description |
|-------|-------------|
| **Syntax** | `import M` / `from M import X, Y as Z, *` |
| **Base path** | Set to the directory of the script when run from CLI; local modules are resolved relative to it. |
| **Single-file module** | `<base_path>/<name>.dc` → module `name`. |
| **Package** | `<base_path>/<name>/__lib__.dc` → module `name`; preferred over a file with the same name. |
| **Dotted name** | `core.config` → walk segments (each segment = package or file); last segment is the loaded module. |
| **Built-in modules** | `ml`, `plot`, `settings_env`, `uuid`, `database` — no file needed. |
| **Exports** | For .dc modules, top-level globals (variables, functions, classes) are the exports. |

For bytecode-level and VM details (ModuleObject, function remapping, cache), see [Module Import System (Internals)](./internals/module_import_system.md).
