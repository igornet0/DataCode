# Native call lifecycle (bytecode → VM → ABI → plugin)

This document describes how a call from DataCode source reaches a native plugin and returns, using the **existing** C ABI (`datacode_abi`). The only cross-boundary value type is `AbiValue` (see [`datacode_abi/src/value.rs`](../datacode_abi/src/value.rs)), not Rust `enum` layout from `data-code`.

## 1. Load native module (`import`)

1. The VM resolves `import <name>` (or `from <name> import ...`) via [`import_handler`](../src/vm/module_system/import_handler.rs).
2. If no `.dc` / built-in module matches, [`try_load_native_module`](../src/vm/native_loader.rs) runs:
   - Optional: `name.dcmodule` (zip + `manifest.json`) → cache extract → dylib path.
   - Else: `lib<name>.dylib` / `lib<name>.so` / `<name>.dll` in base path, DPM package roots, current directory.
3. `dlopen` and `dlsym` on **`datacode_module_entry`** (preferred) or **`datacode_module`** (fallback) — see [`datacode_abi::module`](../datacode_abi/src/module.rs).
4. Exports are turned into `Value::NativeFunction(…)` entries on a module object; the module is stored like any other import.

Artifact layout (`.dcmodule`) is documented in [dcmodule_artifact.md](./dcmodule_artifact.md).

## 2. Call native function from bytecode

```mermaid
flowchart LR
  OpCall[Opcode call]
  Dispatch[call_dispatch]
  NativeExec[execute_native_call]
  Bridge[call_abi_native]
  AbiFn[NativeAbiFn in .so]
  OpCall --> Dispatch
  Dispatch --> NativeExec
  NativeExec --> Bridge
  Bridge --> AbiFn
```

1. **Opcode** pushes callee and arguments; the VM dispatches to [`call_dispatch`](../src/vm/runtime/call_engine/call_dispatch.rs).
2. When the callee is `Value::NativeFunction(native_index)` and `native_index >= builtin_count`, [`execute_native_call`](../src/vm/runtime/call_engine/native_call.rs) delegates to **`call_abi_native`** in [`native_loader.rs`](../src/vm/native_loader.rs).
3. **`call_abi_native`** converts each `Value` to `AbiValue` with [`AbiBridgeContext`](../src/vm/abi_bridge.rs), builds a [`VmContext`](../datacode_abi/src/vm_context.rs) (alloc, `throw_error`, `register_native`), and invokes:

```text
extern "C" fn(*mut VmContext, *const AbiValue, argc) -> AbiValue
```

4. The return `AbiValue` is converted back to `Value` in the bridge; ABI errors set via `throw_error` are surfaced as runtime exceptions.

## 3. `VmContext` and pointer lifetime

| Field | Role |
|-------|------|
| `alloc` | Allocate bytes from the VM-side allocator (C-compatible). |
| `throw_error` | Report `DatacodeError` + UTF-8 message to the VM. |
| `register_native` | Used when a module uses legacy `register` (not for normal descriptor entry). |

Pointers inside `AbiValue` (e.g. strings) are only valid for the duration of the native call unless documented otherwise.

## 4. Authoring plugins

Use `datacode_sdk` + `cdylib`; see [datacode_sdk/docs/modules.md](../datacode_sdk/docs/modules.md). Prefer **`datacode_module_entry`** via `define_module_entry!` for static export tables.
