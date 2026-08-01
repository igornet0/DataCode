# Modules and Imports

This document describes how to **import** and **use modules** in DataCode: syntax, module kinds (file and package), dotted names, base path, and built-in modules. Implementation details (bytecode, ModuleObject, remapping) are in [200 — Developers / Module import system](../200-developers/module_import_system.md).

---

## Import syntax

### Import a module (module object)

```datacode
import ml
import plot
import settings_env

```

The module is loaded and its **namespace object** is bound to the given name. Access exports through the module name:

```datacode
import ml
x = ml.load_mnist()

```

### Import individual names from a module

```datacode
from ml import load_mnist, NeuralNetwork
from settings_env import load_env, Settings
from core.config import get_settings, load_settings

```

- **Named**: `from M import X, Y` — imports names `X` and `Y` from module `M` into the current scope.
- **Alias**: `from M import X as Z` — imports `X` under the name `Z`.
- **All**: `from M import *` — imports all names exported by the module (per runtime behavior).

You can combine: `from M import a, b as c, *`.

---

## How modules are resolved (base path)

When running a script from the **command line** (for example, `datacode path/to/script.dc`), the **base path** is set to the **directory containing that script**. All `import` and `from` statements resolve relative to this path (and, when needed, [DPM package paths](#dpm-package-paths)).

- Running `datacode examples/en/14-modules/main.dc` → base path = `examples/en/14-modules/`.
- So `from core.config import get_settings` looks for `examples/en/14-modules/core/config/__lib__.dc`.

If code runs **without a script file** (REPL or `run("from foo import bar")` without a base path), local `.dc` modules are **not found**; only [built-in modules](#built-in-modules) are available. When using the library API, pass `base_path` so local imports work (see [Internals](../200-developers/module_import_system.md)).

---

## Two kinds of local modules

Local modules are searched relative to the **base path** (and DPM paths). Two variants are supported:

### 1. File module

A file named `<name>.dc` is imported as module `name`:

- Path: `<base_path>/<name>.dc`
- Example: `utils.dc` → `import utils` or `from utils import foo`

### 2. Package (directory with `__lib__.dc`)

A **package** is a directory containing `__lib__.dc`. Import the package by directory name; code from `__lib__.dc` is executed.

- Path: `<base_path>/<name>/__lib__.dc`
- Example: directory `core/config/` with `__lib__.dc` inside → import as `core.config` (see [Dotted names](#dotted-names-packages)).

The loader **prefers** a package over a file: if both `<name>/__lib__.dc` and `<name>.dc` exist, the package is used.

---

## Dotted names (packages)

You can import a **submodule** with a dotted name. Each segment is resolved in order:

- `from core.config import get_settings`:
  1. Find package or file `core` in the base path → directory `core/` with `core/__lib__.dc` (or file `core.dc`).
  2. Inside it find `config` → `core/config/__lib__.dc` (or `core/config.dc`).
  3. Loaded module is `core.config`; its exports (for example, `get_settings`) come from `core/config/__lib__.dc`.

### Packages and namespace folders

For **intermediate** segments of a dotted import (all except the last), these are allowed:

| Kind | Path | `__lib__.dc` | Example |
|------|------|--------------|---------|
| **Package** | `<root>/<name>/__lib__.dc` | required | `core/__lib__.dc` for `core.config` |
| **Namespace folder** | `<root>/<name>/` (ordinary directory) | not needed | `advanced/union_find.dc` for `from advanced.union_find import …` |

The **last** segment loads as a file module (`<name>.dc`) or a package with `__lib__.dc` — same as a normal import.

A namespace folder is **not** a module by itself: `import advanced` still requires `advanced.dc` or `advanced/__lib__.dc`. A folder without `__lib__.dc` is only used as a path prefix.

If an intermediate segment is a **file** (`foo.dc`), not a directory, `from foo.bar import …` fails: the segment is a file, not a package.

Example with a namespace folder:

```
graphs/
├── connected_components.dc   # from advanced.union_find import UnionFind
└── advanced/
    └── union_find.dc         # without advanced/__lib__.dc
```

For **nested packages** with re-export via `__lib__.dc` (for example `core.config`), `__lib__.dc` is still required at each level:

- `core/__lib__.dc` (may be empty or a comment),
- `core/config/__lib__.dc` (actual exports).

---

## What is exported

For a **local .dc module** (file or `__lib__.dc`), exports are the module's **global variables and functions**. Everything defined at the top level (variables, functions, classes) is available on import.

- `from M import f` — `f` must be global in that module.
- `import M` — the namespace object exposes all those globals (for example, `M.f`).

---

## Built-in modules

These modules are built into the runtime; no matching `.dc` file or package is needed in the base path:

| Module | Description |
|--------|-------------|
| `plot` | Images, windows, charts (bar, line, pie, heatmap, subplots) |
| `settings_env` | Load .env, Settings, Config, Field |
| `uuid` | UUID generation (v4, v7), parse, to_string, bytes, v3/v5 |
| `system` | OS, runtime, paths, hardware, network, processes, FS, logs, permissions ([documentation](../2-language/modules/system/README.md)) |
| `database_engine` | DB engine and DatabaseCluster |
| `web` | HTTP, browser automation, HTML→table ([documentation](../2-language/modules/web/README.md)) |
| `websocket` / `ws` | WebSocket server and DCP session helpers |

Example:

```datacode
from database_engine import engine, DatabaseCluster
from uuid import v4, v7

```

## Installable packages (DPM)

Not built into the runtime — install via **DPM**:

| Package | Install | Documentation |
|---------|---------|---------------|
| `ml` | `dpm add ml` | [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) · [overview](./11-modules-and-imports.md) |

```datacode
import ml
x = ml.load_mnist()

```

If a name is **not** a built-in module and no local module is found (in the base path or DPM paths), a runtime error occurs, for example: *"Module 'X' not found. Searched in base path and DPM packages."*

---

## DPM package paths

Besides the script base path, the runtime may use additional search paths (for example, for packages installed via **DPM**). Module names resolve the same way as on the base path (file `<name>.dc` or directory `<name>/__lib__.dc`). How they are set depends on the environment (CLI, HTTP server, or library API).

---

## Full example (local package)

Structure:

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
- `core/__lib__.dc`: may be empty or a short comment (so `core` is a package).
- `core/config/__lib__.dc`: defines `get_settings`, `load_settings`, and other exports; may use `from base import ...`, `from config import ...`, etc., resolved relative to `core/config/`.

Run from the project root:

```bash
cargo run --bin datacode my_app/main.dc
```

Ready-made example in the repository:

- **EN**: [examples/en/14-modules/](../../../examples/en/14-modules/) (see [README](../../../examples/en/14-modules/README.md))

---

## `fn __main__` and module import

If a `.dc` file declares `fn __main__()`, it runs **only when that file is explicitly run as a program** (`datacode path/to/script.dc`). When the same file is loaded as a module (`import M` / `from M import …`), the `__main__` body is **not** called automatically: classes, functions, and variables enter the module namespace, and `__main__` remains an ordinary function (you can call it manually if needed).

Top-level code outside `fn __main__` still runs on first module load (global initialization). Put demos and module tests in `__main__` so they do not run on import.

---

## Summary

| Topic | Description |
|-------|-------------|
| **Syntax** | `import M` / `from M import X, Y as Z, *` |
| **`fn __main__`** | Only when running the file as a script; not on `import` / `from … import`. |
| **Base path** | Set to the script directory when running from CLI; local modules resolve relative to it. |
| **File module** | `<base_path>/<name>.dc` → module `name`. |
| **Package** | `<base_path>/<name>/__lib__.dc` → module `name`; takes priority over a same-named file. |
| **Dotted name** | `core.config` — walk segments; intermediate = package or namespace folder; last = file or package. |
| **Built-in modules** | `plot`, `settings_env`, `uuid`, `system`, `database_engine` — no files required. |
| **DPM packages** | For example `ml` — `dpm add ml`, see [modules_and_imports](./11-modules-and-imports.md). |
| **Exports** | For .dc modules, top-level globals (variables, functions, classes). |

Bytecode and VM details (ModuleObject, function remapping, cache) are in [Module import system (Internals)](../200-developers/module_import_system.md).
