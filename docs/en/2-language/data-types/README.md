# Data Types in DataCode

Readable descriptions of types: what a value is, how to access it (`[]`, methods), and which **built-in functions** work with it.

**Technical reference:** [type-reference.md](../type-reference.md) · **Built-in functions:** [functions/](../functions/README.md) · **Syntax:** [0-syntax](../../0-syntax/README.md)

Implementation details: the `Value` type in `src/common/value.rs`, indexing in `src/vm/interpreter/element/`, built-ins in `src/vm/native_registry.rs` and `src/vm/globals.rs`.

## Where to start

1. **[typeof and isinstance](typeof-and-isinstance.md)** — how to inspect and check types for **any** value, with examples.
2. Then open the section you need from the table below.

## Contents

| Topic | File |
|-------|------|
| **Type checks:** `typeof`, `isinstance` | [typeof-and-isinstance.md](typeof-and-isinstance.md) |
| Numbers, logic, `null`, `...` | [numbers-logic-null.md](numbers-logic-null.md) |
| Strings, date/money in string form | [string.md](string.md) |
| Arrays, slices, `enum(...)` | [arrays.md](arrays.md) |
| Bytes (hex digest, binary files) | [bytes.md](bytes.md) |
| Tuple | [tuple.md](tuple.md) |
| File or directory path | [path.md](path.md) |
| ZIP/7Z/RAR archive | [archive.md](archive.md) |
| UUID | [uuid.md](uuid.md) |
| Table and column | [table.md](table.md) |
| Object, class, dictionary | [object.md](object.md) |
| Functions | [functions.md](functions.md) |
| Date and duration | [date-and-duration.md](date-and-duration.md) |
| Window, plot, database, plugins | [graphics-databases-plugins.md](graphics-databases-plugins.md) |

## Type summary

| In your program | `typeof` (usually) | Details |
|-----------------|-------------------|---------|
| Number | `int` / `float` | [numbers-logic-null.md](numbers-logic-null.md) |
| `true` / `false` | `bool` | [numbers-logic-null.md](numbers-logic-null.md) |
| Quoted text | `string` / `date` / `money` | [string.md](string.md) |
| `[...]` | `array` | [arrays.md](arrays.md) |
| `sha256`, … | `bytes` | [bytes.md](bytes.md) |
| Tuple | `tuple` | [tuple.md](tuple.md) |
| Result of `enum(...)` | `enumerate` | [arrays.md](arrays.md) |
| `path(...)` | `path` | [path.md](path.md) |
| `archive(...)` | `archive` | [archive.md](archive.md) |
| UUID | `uuid` | [uuid.md](uuid.md) |
| Table | `table` | [table.md](table.md) |
| Column `t["col"]` | `column` | [table.md](table.md) |
| Columns `t[["a","b"]]` | `columns` | [table.md](table.md) |
| Dictionary / class instance | `object` (or plugin name) | [object.md](object.md) |
| Function | `function` | [functions.md](functions.md) |
| Plugin object | from plugin / `plugin_opaque` | [graphics-databases-plugins.md](graphics-databases-plugins.md) |
| Window, image, plot | `window`, `image`, `figure`, `axis` | [graphics-databases-plugins.md](graphics-databases-plugins.md) |
| Database | `database_engine`, `database_cluster` | [graphics-databases-plugins.md](graphics-databases-plugins.md) |
| `null` | `null` | [numbers-logic-null.md](numbers-logic-null.md) |
| `...` | `ellipsis` | [numbers-logic-null.md](numbers-logic-null.md) |
