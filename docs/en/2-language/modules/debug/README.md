# debug Module — Runtime Debugging

Built-in module for **VM introspection**. Intended for developers and debugging, not production logic.

## Import

```datacode
import debug
```

## debug.operators() -> string

Returns a text table of registered **infix operators** (symbol, name, precedence, associativity, source module).

Requires the VM host to set an operator registry snapshot at startup (`Vm::set_operator_registry_snapshot`). With a normal `datacode` launch, the table is available after `import debug`.

```datacode
import debug
print(debug.operators())
```

If the snapshot is not set, an explanatory message is returned instead of the table.

## Related

| Topic | Documentation |
|-------|---------------|
| Operator descriptors | [200-developers/operator_descriptor](../../../200-developers/operator_descriptor.md) |
| Operator registry (Rust) | `src/vm/operator_registry.rs` |
