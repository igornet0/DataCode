# User-defined infix operators (`operator_descriptor`)

Native modules (dylibs) can declare extra infix operators **before** parse time by exporting **`operator_descriptor`**: a native function with **no arguments** that returns an **array of rows**, each row:

`[symbol, name, precedence, associativity]`

- **symbol** (string): source token(s), e.g. `"@"`.
- **name** (string): logical name passed to `opaque_binop` / VM dispatch, e.g. `"matmul"`.
- **precedence** (number): binding tier (higher binds tighter; align with built-ins: `+`/`-` ≈ 50, `*` ≈ 60).
- **associativity**: `"left"` / `"right"` or `0` / `1`.

Example (conceptual): register `@` → `matmul` with the same precedence tier as multiplication.

## Import order

The host **preloads** every `import` / `from … import` module name found in the token stream, loads matching **native** dylibs, and merges `operator_descriptor` into one [`OperatorRegistry`](crate::vm::operator_registry::OperatorRegistry). Pure `.dc` packages (no dylib) are skipped.

Using an operator symbol in an expression **without** importing a module that registers it first will fail at parse time (e.g. unregistered `@`).

## Conflicts

Registering the same **symbol** twice (from different modules or duplicate rows) is an **error** — deterministic, no silent override.

## Runtime

Arithmetic / plugin dispatch uses the single ABI hook **`opaque_binop(left, right, op_name)`** on `PluginOpaque` values; the VM does not hardcode module names.

## Introspection

After a normal run, `import debug` and call `debug.operators()` to print the operator table (symbol, name, precedence, associativity, source module).

## Opcode note

Legacy bytecode may use `MatMul`; new emits should prefer **`BinaryOp`** with logical name `"matmul"`. `MatMul` remains for `.dcb` compatibility.
