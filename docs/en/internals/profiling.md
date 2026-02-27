# Profiling

The VM can collect lightweight execution statistics when built with the **profile** feature. This document describes the metrics, where they are recorded, and how to build and use the feature.

**Source:** [src/vm/profile.rs](../../../src/vm/profile.rs), [src/vm/executor.rs](../../../src/vm/executor.rs), [src/common/value_store.rs](../../../src/common/value_store.rs), [src/vm/vm.rs](../../../src/vm/vm.rs).

---

## Feature and cost

- **Feature flag:** `profile`. Enable with `cargo build --features profile` or add `profile` to `[features]` in Cargo.toml and build normally.
- **When disabled:** All profile API calls are no-op stubs (empty functions). No thread_locals or allocations; zero runtime cost.
- **When enabled:** Thread-local state holds **ProfileStats** and the current opcode name. Each executed instruction and each store allocate/get is recorded (see below).

---

## ProfileStats

**ProfileStats** ([src/vm/profile.rs](../../../src/vm/profile.rs)) is the struct holding one run’s statistics:

| Field | Meaning |
|-------|---------|
| **opcodes_executed** | Total number of instructions executed. |
| **store_allocations** | Number of value_store allocations (e.g. when storing a new value). |
| **store_get_count** | Number of value_store get/load operations. |
| **alloc_by_opcode** | Map opcode variant name → count of allocations that occurred while that opcode was current. |
| **get_by_opcode** | Map opcode variant name → count of store gets that occurred while that opcode was current. |

Opcode names are the **variant name** without parameters (e.g. `MakeArray(8)` and `MakeArray(3)` both contribute to `"MakeArray"`), from **OpCode::variant_name()**.

---

## Where metrics are recorded

- **record_opcode()** — Called at the start of **execute_instruction** ([src/vm/executor.rs](../../../src/vm/executor.rs)). Increments **opcodes_executed**.
- **set_current_opcode(op)** — Called immediately after record_opcode with the current instruction. Sets the thread-local “current opcode” used to attribute the next allocate/get to that opcode.
- **record_allocate()** — Called from **value_store** when an allocation happens ([src/common/value_store.rs](../../../src/common/value_store.rs), e.g. in allocate or when storing a new cell). Increments **store_allocations** and **alloc_by_opcode[current_opcode]**.
- **record_store_get()** — Called from value_store on get/load. Increments **store_get_count** and **get_by_opcode[current_opcode]**.

So hot paths (e.g. Constant, LoadLocal, MakeArray) can be correlated with store pressure: which opcodes cause the most allocations or loads.

---

## Lifecycle in run()

1. **Start of run()** ([src/vm/vm.rs](../../../src/vm/vm.rs)): **profile::set()** initializes thread-local ProfileStats (with empty alloc_by_opcode and get_by_opcode maps).
2. **During execution:** Each step calls **execute_instruction**; it calls **record_opcode()** and **set_current_opcode(&instruction)**. Store operations call **record_allocate()** / **record_store_get()**.
3. **End of run():** When the main loop exits (either by **VMStatus::Return** or **VMStatus::FrameEnded**), the VM calls **profile::take()** to consume the thread-local stats. If **Some(stats)** is returned, **profile::print_stats(&stats)** is called.

**take()** removes the stats from the thread-local; a second take in the same run returns None. So only one run’s stats are printed per VM run.

---

## Building and running with profile

```bash
cargo build --features profile
# or
cargo run --features profile -- path/to/script.dc
```

Run your script or REPL as usual. At the end of execution, profile output is printed to stderr, for example:

```
[profile] opcodes_executed   = 1234567
[profile] store_allocations  = 50000
[profile] store_get_count    = 200000
[profile] top 15 by alloc: [("MakeArray", 10000), ("Constant", 8000), ...]
[profile] top 15 by get:  [("LoadGlobal", 50000), ("GetArrayElement", 30000), ...]
[profile] top 15 by alloc+get: [("LoadGlobal", 55000), ("MakeArray", 12000), ...]
```

---

## Interpreting the output

- **opcodes_executed** — Total instruction count; with a known script size, gives a rough idea of loop/function weight.
- **store_allocations** / **store_get_count** — High values suggest heavy use of the value_store (objects, arrays, non-inline globals). Use **alloc_by_opcode** and **get_by_opcode** to see which instructions dominate.
- **Top by alloc** — Opcodes that cause the most allocations (e.g. MakeObject, MakeArray, Constant for non-immediate constants).
- **Top by get** — Opcodes during which the most store loads happen (e.g. LoadGlobal, GetArrayElement, load_value in various paths).
- **Top by alloc+get** — Combined store traffic by opcode; useful to find the hottest opcodes for store-related optimization.

The profile is per run and per thread (thread-local). It does not include wall time or CPU time; for that use external tools (e.g. `perf`, `cargo flamegraph`).
