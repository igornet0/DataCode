# DataCode — Developer Documentation (EN)

Internal VM/runtime documentation and native plugin ABI. For **core developers** (interpreter, compiler, runtime).

**User docs:** [English](../README.md) · [Русский](../../ru/README.md)

---

## VM and runtime

| Document | Description |
|----------|-------------|
| [VM Architecture](vm_architecture.md) | `Vm`, `CallFrame`, executor; bytecode (OpCode, Chunk, DCB) |
| [Execution Model](execution_model.md) | Stack, frames, call/return, exceptions |
| [Globals and Namespace](globals_and_namespace.md) | `GlobalSlot`, `global_names`, remap |
| [Module Import System](module_import_system.md) | Import/ImportFrom, `ModuleObject`, `file_import` |
| [Profiling](profiling.md) | Feature `profile`, `ProfileStats` |
| [Text rendering](text-rendering.md) | Plot module font/glyph pipeline |

## Native plugins (ABI)

| Document | Description |
|----------|-------------|
| [`.dcmodule` artifact](dcmodule_artifact.md) | ZIP manifest + native libraries |
| [Native call lifecycle](native_call_lifecycle.md) | Bytecode → VM → ABI → plugin |
| [User-defined operators](operator_descriptor.md) | `operator_descriptor` export |
| [Mutating natives audit](vm_mutating_natives_audit.md) | `load_value` / fast-path audit |

## SDK (external)

| Resource | Description |
|----------|-------------|
| [Datacode-sdk](https://github.com/igornet0/Datacode-sdk) | `define_module!`, examples |
| [Datacode-abi](https://github.com/igornet0/Datacode-abi) | C ABI contract (`DATACODE_ABI_VERSION`) |

**Russian:** same topics in [docs/ru/200-разработчикам](../../ru/200-разработчикам/README.md).

## Key source files

- `src/vm/vm.rs`, `src/vm/executor.rs`, `src/vm/frame.rs`
- `src/bytecode/opcode.rs`, `src/vm/native_loader.rs`
- `src/vm/runtime/call_engine/native_call.rs`
