# DataCode Built-in Functions

Reference for **117** global built-in functions (source: [`BUILTIN_GLOBAL_NAMES`](../../../../src/vm/globals.rs)).  
Code aliases: `read` → `read_file`, `read_bin` → `read_file_bin`; also `save`, `save_tables_sqlite` (extended builtins).

**📚 Examples:**
- Basics: [`examples/en/01-basics/`](../../../examples/en/01-basics/)
- Functions: [`examples/en/04-functions/`](../../../examples/en/04-functions/)
- Data and tables: [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)

## Contents

| Category | Document | Functions |
|----------|----------|-----------|
| Utilities | [utilities.md](./utilities.md) | `print`, `len`, `copy`, `range` |
| Type conversion | [type-conversion.md](./type-conversion.md) | `int`, `float`, `bool`, `str`, `array`, `date`, `money` |
| Type operations | [types.md](./types.md) | `typeof`, `isinstance` |
| Date and time | [date-and-time.md](./date-and-time.md) | `now`, `parse_date`, `format_date`, `date_to_unix`, `duration` |
| Paths and files | [paths.md](./paths.md) | `path`, …, `getcwd`, `list_files` |
| Math | [math.md](./math.md) | `abs`, …, `divmod`, `isinf` |
| Strings | [strings.md](./strings.md) | `upper`, …, `starts_with`, `ends_with`, `ord` |
| Arrays and collections | [arrays.md](./arrays.md) | `push`, …, `any`, `all`, `enum`, `set`, … |
| Tables | [tables.md](./tables.md) | `table`, `read`/`read_file`, `table_*`, `archive`, `datasource`, … |
| JOIN | [join.md](./join.md) | `inner_join`, `left_join`, … (13 functions) |
| Cryptography and RNG | [cryptography-and-randomness.md](./cryptography-and-randomness.md) | `sha256`, `random`, `random_int`, … |

## Full list (from source)

<details>
<summary>117 global functions — expand</summary>

Full name list is in [`BUILTIN_GLOBAL_NAMES`](../../../../src/vm/globals.rs) (indices `0..116`). Summary by group:

| Indices | Group | Key names |
|---------|-------|-----------|
| 0–11 | utilities / types | `print`, `len`, `range`, `int`…`money` |
| 12–20 | paths | `path`, `path_exists`, … `path_len` |
| 21–26 | math | `abs`, `sqrt`, `pow`, `min`, `max`, `round`, `ceil`, `floor` |
| 27–40 | strings | `upper`…`capitalize`, `starts_with`, `ends_with` |
| 41–50 | arrays | `push`…`all` |
| 51–79 | tables | `table`, `read_file`, `table_*`, `merge_tables`, `show_table` |
| 80–82 | date / paths | `now`, `getcwd`, `list_files` |
| 83–95 | JOIN | `inner_join`…`table_suffixes` |
| 96–98 | relations | `relate`, `primary_key`, `enum` |
| 99–103 | collections | `Table`, `array_with_capacity`, `map`, `filter`, `reduce` |
| 104–111 | cryptography / RNG | `sha256`…`random` |
| 112–120 | date / utilities | `date_to_unix`, `parse_date`, `format_date`, `duration`, `set`, `divmod`, `isinf`, `copy`, `ord` |
| 121–126 | tables (extended) | `table_row_number`, `table_distinct`, `table_value_map`, `table_aggregate`, `table_aggregate_group` |
| 127–128 | files / API | `archive`, `datasource` |

Aliases: `read` = `read_file`, `read_bin` = `read_file_bin`. Extended: `save`, `save_tables_sqlite`.

</details>

## Do not confuse with modules

After the 117 builtins, **built-in modules** are registered in globals (`import plot`, `import uuid`, …) — separate APIs, not global functions.

**DPM installable packages** (for example `ml` via `dpm add ml`) are also not builtins; see [ml-module.md](../modules/ml-module.md) and [modules/README.md](../modules/README.md).

## Quick lookup

| Task | Where |
|------|-------|
| Random float [0, 1) | [random()](./cryptography-and-randomness.md#random) |
| Random integer | [random_int](./cryptography-and-randomness.md#random_intmin-max) |
| String hash | [sha256](./cryptography-and-randomness.md#sha256data) |
| Current time | [now()](./date-and-time.md#now) |
| Copy a container | [copy()](./utilities.md#copyvalue) |
| Table JOIN | [join.md](./join.md) |
| Filter a table | [tables.md](./tables.md) |

## Related documents

- [Data types](../data-types/README.md)
- [Working with tables](../tables/creation-and-operations.md)
- [ML — MNIST example](../../1-examples/11-mnist-mlp.md)
