# DataCode Internals — Core Developer Documentation

This section describes the internal architecture and implementation of the DataCode VM and runtime. It is intended for **core developers** working on the interpreter, compiler, or runtime (kernel) code.

**Related:** [Main English docs](../README.md) (user-facing language and library documentation).

---

## Contents

| Document | Description |
|----------|-------------|
| [VM Architecture](vm_architecture.md) | VM overview: `Vm`, `CallFrame`, executor; bytecode (OpCode, Chunk, DCB); execution loop. |
| [Execution Model](execution_model.md) | Internal execution: operand stack, frames (slots, ip, stack_start), constants, call/return, exceptions. |
| [Globals and Namespace](globals_and_namespace.md) | `GlobalSlot`, builtins vs module globals, `global_names`, `merge_global_names`, `update_chunk_indices_from_names`. |
| [Module Import System](module_import_system.md) | Compilation of Import/ImportFrom; runtime resolution (file_import); ModuleObject; remap after merge. |
| [Profiling](profiling.md) | Feature `profile`, ProfileStats, how to build and interpret output. |

---

## Key source locations

- **VM:** `src/vm/vm.rs`, `src/vm/executor.rs`, `src/vm/frame.rs`
- **Bytecode:** `src/bytecode/opcode.rs`, `src/bytecode/chunk.rs`, `src/vm/dcb.rs`
- **Globals:** `src/vm/global_slot.rs`, `src/vm/globals.rs`
- **Modules:** `src/vm/module_object.rs`, `src/vm/file_import.rs`
- **Remap:** `src/lib.rs` (`remap_module_function_ids_in_exports`, `remap_native_indices_in_exports`)
- **Compiler:** `src/compiler/stmt/import.rs`, `src/compiler/variable/resolver.rs`, `src/compiler/scope.rs`
- **Profile:** `src/vm/profile.rs`
