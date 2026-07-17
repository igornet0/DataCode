# Table Functions

← [Built-in Functions](./README.md)

> **Read/write aliases:** examples and scripts often use `read(...)` and `save(...)` — these are the same functions as **`read_file`** and canonical table writing. The VM registers `read_file`, `read_file_bin`; `read` is a compatibility alias (`src/vm/natives/file_io_compat.rs`).

**📚 Examples:**
- [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)
- See also: [Working with Tables](../tables/creation-and-operations.md)

### `table(data, headers)`

Creates a table from data and headers.

**Arguments:**
- `data` (array) — array of arrays, where each inner array represents a row
- `headers` (array, optional) — array of strings with column names

**Returns:** `table` — table object, or `null` if arguments are of the wrong type

**Examples:**
```datacode
table([[1, "a"], [2, "b"]], ["ID", "Name"])
table([[1, 2, 3], [4, 5, 6]])  # Without headers

```

---

### `read(path)` / `read(path, header_row)` / `read(path, sheet_name="sheet_name")` / `read(path, header_row, sheet_name)` / `read(path, header_row, sheet_name, header)` / `read(path, header_row, headerT=...)`

Reads a file and returns a table (for CSV/XLSX) or a string (for TXT).

**Arguments:**
- `path` (path | string) — path to the file
- `header_row` (number, optional) — row number containing headers (0-based, default 0). When `headerT` is used, applies to the **transposed** table
- `sheet_name` (string, optional) — sheet name for XLSX files (default: first sheet)
- `header` (array | object, optional) — column filter or rename mapping for a normal table
- `headerT` (array | object, optional) — same as `header`, but for a transposed table. Cannot be used together with `header`

**Returns:**
- `table` — for CSV and XLSX files
- `string` — for TXT files
- `null` — if the file is not found or an error occurred

**Examples:**
```datacode
read("data.csv")
read("report.xlsx", "Sales")
read("data.csv", 2)
read(path("report.xlsx"), 1, "DataSheet")
read(path("notes.txt"))  # Returns a string

# Load only specified columns
sample_table = read(path("sample.csv"), header_row=0, header=["Name", "Age", "City", "Salary"])

# Rename columns on load
sample_table = read(path("sample.csv"), header_row=0, header={"Name": "Name_A", "Age": null, "City": null, "Salary": null})

# Combined with sheet_name for XLSX
data = read(path("report.xlsx"), header_row=1, sheet_name="Data", header=["ID", "Value"])

# Wide format → table after transposition
metrics = read(path("wide.csv"), header_row=0, headerT=["Metric", "Revenue", "Cost"])

```

**Notes:**
- For CSV files, data types are detected automatically
- For XLSX files, a specific sheet can be specified
- For XLSX files, the header row can be specified (if it is not the first row)
- Argument order: `read(path, header_row, sheet_name, header, headerT)`
- When `header` is an array, only the specified columns are loaded (non-existent columns are ignored)
- When `header` is an object, columns are renamed according to the mapping (columns not listed in the object keep their original names)
- `header` and `headerT` are mutually exclusive
- When `headerT` is used, the `header_row` parameter refers to the transposed table

---

### `table_info(table)`

Returns information about a table (row count, column count, data types).

**Arguments:**
- `table` (table) — table

**Returns:** `string` — string with table information

**Examples:**
```datacode
data = read(path("data.csv"))
print(table_info(data))

```

---

### `table_head(table, n)`

Returns the first n rows of a table.

**Arguments:**
- `table` (table) — table
- `n` (number, optional) — number of rows (default 5)

**Returns:** `table` — new table with the first n rows, or `null` if the argument is not a table

**Examples:**
```datacode
table_head(data)      # First 5 rows
table_head(data, 10)  # First 10 rows

```

---

### `table_tail(table, n)`

Returns the last n rows of a table.

**Arguments:**
- `table` (table) — table
- `n` (number, optional) — number of rows (default 5)

**Returns:** `table` — new table with the last n rows, or `null` if the argument is not a table

**Examples:**
```datacode
table_tail(data)      # Last 5 rows
table_tail(data, 10)  # Last 10 rows

```

---

### `table_select(table, columns)`

Selects specified columns from a table.

**Arguments:**
- `table` (table) — table
- `columns` (array) — array of strings with column names to select

**Returns:** `table` — new table with only the selected columns, or `null` if a column is not found or arguments are of the wrong type

**Examples:**
```datacode
table_select(data, ["Name", "Age"])
table_select(data, ["ID"])
data.select(["Name", "Age"])

```

**Method:** `data.select(columns)` — same as `table_select`.

---

### `table_rename(table, mapping)` / `table_rename(table, old, new)`

Renames table columns.

**Arguments:**
- `table` (table)
- `mapping` (object) — object `{old: new}`; `null` as a value keeps the original name
- **or** `old` (string), `new` (string) — rename a single column

**Returns:** `table` — new table (the original is not modified)

**Examples:**
```datacode
table_rename(data, {"Name": "EmployeeName", "Age": null})
table_rename(data, "Name", "EmployeeName")
data.rename("Name", "EmployeeName")
```

**Errors:** `KeyError` if a column is not found; `TypeError` for wrong argument types.

---

### `table_drop_column(table, column)`

Removes one or more columns.

**Arguments:**
- `table` (table)
- `column` (string | array[string])

**Returns:** `table` — new table without the specified columns

**Examples:**
```datacode
table_drop_column(data, "Notes")
table_drop_column(data, ["Notes", "Temp"])
data.drop_column("Notes")
```

**Errors:** `KeyError` if a column is not found; `ValueError` if all columns are removed.

---

### `table_add_column(table, name, value?)`

Adds a column at the end of a table.

**Arguments:**
- `table` (table)
- `name` (string) — name of the new column
- `value` (any, optional) — scalar (one value for all rows) or `array` (one value per row). Without this argument — `null`.

**Returns:** `table` — new table with the added column

**Examples:**
```datacode
table_add_column(data, "Region", "EMEA")
table_add_column(data, "RowNum", [1, 2, 3])
data.add_column("Region", "EMEA")
data.add_column("Extra")   # null in all rows
```

**Errors:** `ValueError` if the column already exists or the array length does not match the number of rows.

---

### `table_map(table, column, function)` / `table.map(column, function)`

Applies a function to each cell in the specified column and returns a **new table** with transformed values in that column (other columns unchanged).

**Arguments:**
- `table` (table)
- `column` (string) — column name
- `function` (callable) — built-in function (`int`, `float`, `str`, `bool`, `round`, …) or user-defined `fn` with arity 1

**Returns:** `table` — new table

**Examples:**
```datacode
table_map(data, "Age", int)
table_map(data, "Amount", float)
data.map("Salary", str)
data.map("Active", to_bool_flag)

fn to_bool_flag(s) {
    if s == "yes" { return true }
    if s == "no" { return false }
    return bool(s)
}
typed = employees.map("Age", int)
```

**Errors:** `KeyError` if the column is not found; `TypeError` if the third argument is not callable or the user function has arity ≠ 1.

**Note:** the global `map(collection, fn)` for tables iterates **rows**, not columns. To transform a column, use `table.map(column, fn)`.

---

### `table_split_column(table, column, iter_fn | delimiter, new_columns)` / `table.split_column(...)`

Splits one column into several new columns (the original column is preserved). For each row:

- **callback:** `iter_fn(cell)` → `array` of parts;
- **or delimiter string:** `column, " ", ["Part1", "Part2"]` — built-in split without a callback.

Missing parts are filled with an empty string `""`. New columns are appended to the end of the table.

**Arguments:**
- `table` (table)
- `column` (string) — source column
- `iter_fn` (callable, arity 1) **or** `delimiter` (string)
- `new_columns` (array of string) — names of new columns (non-empty array)

**Examples:**
```datacode
table_split_column(data, "FullName", " ", ["First", "Last"])
data.split_column("Tag", fn(x) => [x, ""], ["A", "B"])
```

**Errors:** `KeyError`, `ValueError` (empty `new_columns`, duplicate name), `TypeError` (callback does not return an array).

---

### `table_join_columns(table, source_columns, new_column, delimiter)` / `table.join_columns(...)`

Joins several columns into one new column (`join` by row values). Original columns are preserved.

**Arguments:**
- `table` (table)
- `source_columns` (array of string) — columns in join order
- `new_column` (string) — name of the new column
- `delimiter` (string) — separator

**Examples:**
```datacode
table_join_columns(employees, ["Name", "Department"], "NameDept", " — ")
employees.join_columns(["Name", "Department"], "NameDept", " - ")
```

**Errors:** `KeyError`, `ValueError` (empty column list, name already exists).

---

### `table_sort(table, column, ascending)`

Sorts a table by the specified column.

**Arguments:**
- `table` (table) — table
- `column` (string) — column name to sort by
- `ascending` (bool, optional) — sort direction (default `true`)

**Returns:** `table` — new sorted table, or `null` if the column is not found or arguments are of the wrong type

**Examples:**
```datacode
table_sort(data, "Age")              # Ascending
table_sort(data, "Name", true)       # Ascending
table_sort(data, "Salary", false)    # Descending

```

---

### `table_where(table, column, operator, value)`

Filters a table by a condition.

**Arguments:**
- `table` (table) — table
- `column` (string) — column name to filter on
- `operator` (string) — comparison operator (">", "<", ">=", "<=", "==", "=", "!=", "<>")
- `value` (any) — value to compare against

**Returns:** `table` — new table with filtered rows, or `null` if the column is not found or arguments are of the wrong type

**Examples:**
```datacode
table_where(data, "Age", ">", 18)
table_where(data, "Name", "==", "John")
table_where(data, "Salary", ">=", 50000)
table_where(data, "Status", "!=", "inactive")

```

**Alternative syntax:** filter via brackets — `data["Age" > 25]`. Supports `in` / `not in`, string conditions `& contains(...)`, and compound conditions:

```datacode
data["City" in ["Houston", "Chicago"]]
data["Name" & starts_with("Jo")]
data["Name" & ends_with("son")]
data["Salary" & contains("50000")]
data["Age" > 25 or "City" == "Houston"]
data["Age" > 25 and "City" == "Houston"]
data[("Age" > 25 or "City" == "Houston") and "Salary" > 40000]
```

Precedence: `and` is higher than `or`. A chain `[...][...]` is equivalent to `and`.

---

### `table_drop_nulls(table, column?)`

Removes rows containing `null` in the specified columns.

**Arguments:**
- `table` (table) — table
- `column` (string | array[string], optional) — single column or list of columns to check

**Behavior:**
- Without `column` — a row is removed if **at least one** column contains `null`
- With `column` — a row is removed if `null` appears in **any** of the specified columns

**Returns:** `table` — new table without removed rows (the original is not modified)

**Examples:**
```datacode
table_drop_nulls(data)
table_drop_nulls(data, "City")
table_drop_nulls(data, ["City", "Age"])

data.drop_nulls()
data.drop_nulls("City")
data.drop_nulls(["City", "Age"])
```

**Errors:** `TypeError` if `column` is not a string or array of strings; `KeyError` if a column is not found.

---

### `table_replace_nulls(table, column_or_replacement, replacement?)`

Replaces cells with `null` values with a constant or callback result.

**Arguments:**
- `table` (table) — table
- `column_or_replacement` (string | array[string] | any | function) — column selector or replacement value/function
- `replacement` (any | function, optional) — replacement value/function when the second argument specifies column(s)

**Behavior:**
- `table_replace_nulls(data, replacement)` — replace `null` in **all** columns
- `table_replace_nulls(data, "City", replacement)` — replace `null` only in column `City`
- `table_replace_nulls(data, ["City", "Age"], replacement)` — replace `null` only in the listed columns
- `replacement` can be:
  - a scalar value
  - a callback `fn(row)`, where `row` is the current row object (`{"Column": value, ...}`)

**Returns:** `table` — new table with replacements (the original is not modified)

**Examples:**
```datacode
table_replace_nulls(data, "Unknown")
table_replace_nulls(data, "City", "Unknown")
table_replace_nulls(data, ["City", "Age"], "Unknown")

table_replace_nulls(data, ["City", "Age"], fn(row) => if row["Amount"] < 1000 { null } else { "Unknown" })

data.replace_nulls("Unknown")
data.replace_nulls("City", "Unknown")
data.replace_nulls(["City", "Age"], "Unknown")
```

**Errors:** `TypeError` for wrong `column` type, wrong callback arity, or wrong number of arguments; `KeyError` if a column is not found.

---

### `table_row_number(table, column_name?, start_from?)`

Adds a new column with sequential row numbering to a table.

**Arguments:**
- `table` (table) — table
- `column_name` (string, optional) — name of the new column (default: `"RowNumber"`)
- `start_from` (number, optional) — starting value (default: `1`)

**Returns:** `table` — new table with the numbering column added

**Examples:**
```datacode
table_row_number(data)
table_row_number(data, "Num")
table_row_number(data, "Num", 100)

data.row_number()
data.row_number("Num", 100)
```

**Errors:** `TypeError` for wrong argument types; `ValueError` if column `column_name` already exists.

---

### `table_distinct(table, columns?)`

Removes duplicate rows while preserving the order of first occurrence.

**Arguments:**
- `table` (table) — table
- `columns` (string | array[string], optional) — column or list of columns for the uniqueness key

**Behavior:**
- Without `columns` — removes duplicates across the **entire row**
- With `columns` — removes duplicates based on the values in the specified columns

**Returns:** `table` — new table without duplicates

**Examples:**
```datacode
table_distinct(data)
table_distinct(data, "ID")
table_distinct(data, ["ID", "Category"])

data.distinct()
data.distinct(["ID", "Category"])
```

**Errors:** `TypeError` if `columns` is not a string or array of strings; `KeyError` if a column is not found.

---

### `table_value_map(table, column, mappings)`

Transforms values in one table column according to mapping rules.

**Arguments:**
- `table` (table) — table
- `column` (string) — target column name
- `mappings` (object | array[object]) — value replacement rules

**Supported `mappings` formats:**
- object: `{old_value: new_value, ...}`
- array: `[{"from": x, "to": y}, ...]` or `[{old: x, new: y}, ...]`

Values not present in `mappings` remain unchanged.

**Returns:** `table` — new table with the transformed column

**Examples:**
```datacode
table_value_map(data, "Active", {"yes": "active", "no": "inactive"})
table_value_map(data, "Tag", [{"from": "sale", "to": "promo"}, {"from": "new", "to": "fresh"}])
data.value_map("Tag", [{old: "old", new: "archive"}])
```

**Errors:** `TypeError` for wrong argument types or wrong mapping element format; `KeyError` if `column` is not found.

---

### `table_aggregate(table, spec)`

Aggregates an entire table into a new table with a single result row.

**Arguments:**
- `table` (table) — source table
- `spec` (object) — object of the form `{output_column: aggSpec, ...}`

`aggSpec` forms:
- string: `"count"` (for operations without a column)
- object: `{op: "...", column: "Amount", ...}`

Supported operations:
- `count`, `count_distinct`, `sum`, `avg`, `min`, `max`, `first`, `last`, `median`, `mode`, `stddev`, `variance`, `percentile`, `list`, `any`

Special fields:
- `percentile`: `{op: "percentile", column: "Amount", p: 0.9}`
- `any`: `{op: "any", column: "Amount", where: fn(x) => x > 100}`

**Returns:** `table` — new table with one row and columns from the keys of `spec`

**Examples:**
```datacode
summary = table_aggregate(employees, {
  rows: "count",
  total_amount: {op: "sum", column: "Amount"},
  p90_amount: {op: "percentile", column: "Amount", p: 0.9},
  has_large: {op: "any", column: "Amount", where: fn(x) => x > 6000}
})

summary2 = employees.aggregate({
  rows: "count",
  avg_amount: {op: "avg", column: "Amount"}
})
```

**Errors:** `TypeError` for invalid `spec`/`op`/field types; `KeyError` if `column` is not found.

---

### `table_aggregate_group(table, spec)`

Groups table rows and aggregates each group into a separate result row.

**Arguments:**
- `table` (table) — source table
- `spec` (object) — object with a required `group` field and aggregations (same `aggSpec` format as `table_aggregate`)

**Required field:**
- `group` (string | array[string]) — column or list of columns to group by

**Returns:** `table` — new table with one row per unique group; group columns first, then aggregation columns

**Examples:**
```datacode
by_city = table_aggregate_group(employees, {
  group: "City",
  count: "count",
  sum_amount: {op: "sum", column: "Amount"},
})

by_dept_city = employees.aggregate_group({
  group: ["Department", "City"],
  count: "count",
  avg_amount: {op: "avg", column: "Amount"},
})
```

**Errors:** `TypeError` for invalid `spec`/`group`/aggregation fields; `KeyError` if a group column or aggregation column is not found.

---

### `show_table(table)`

Prints a table to the console in a formatted layout.

**Arguments:**
- `table` (table) — table to display

**Returns:** `null`

**Examples:**
```datacode
data = read(path("data.csv"))
show_table(data)
show_table(table_head(data, 10))

```

**Notes:**
- Displays at most 20 rows for large tables
- Automatically adjusts column widths
- Uses Unicode characters for table borders

---

### `read_bin(path)`

Reads a file as **raw bytes** (does not parse CSV/XLSX). Path rules are the same as for `read`.

**Arguments:**
- `path` (path | string)

**Returns:** `bytes`, or `null`

---

### `merge_tables(tables, mode?)`

Combines several tables into one (vertical stack / union by rows).

**Arguments:**
- `tables` (array of table)
- `mode` (string, optional)

**Returns:** `table`

---

### `table.add_row(row)` (method)

Adds a row to an existing table **in-place**. There is no global `table_add_row` function — only method syntax.

**Arguments:**
- `row` (array) — array of cell values; length must match the number of columns

**Returns:** `table` — the same table (in-place mutation)

**Examples:**
```datacode
t = table([[1, "a"]], ["id", "name"])
t.add_row([2, "b"])
len(t)   # 2

# Empty table with headers
t = table([], ["x", "y"])
t.add_row([1, 2])

```

**Errors:** `TypeError` if the argument is not an array; `ValueError` if the row length does not match the number of columns.

**📚 Example:** [`examples/en/03-data-types/table_push.dc`](../../../examples/en/03-data-types/table_push.dc)

---

### `Table(path)`

Loads a table from a file path (alternative to `read` for the typed constructor).

**Arguments:**
- `path` (path | string)

**Returns:** `table`

---

### `relate(col1, col2)`

Declares a relationship between columns (for subsequent JOINs).

**Arguments:**
- `col1`, `col2` — column references

---

### `primary_key(col)`

Marks a column as the primary key of a table.

**Arguments:**
- `col` — column reference

---

### `save_tables_sqlite(tables, filename="db", **kwargs)`

Exports an array of tables to a single SQLite file. `relate` and `primary_key` relationships are reflected in the schema. Additional kwargs are stored in `_datacode_variables`.

**Arguments:**
- `tables` (array) — array of tables (each must be in a named variable)
- `filename` (string, optional) — base path (`.sqlite` is appended)
- `**kwargs` — arbitrary metadata

**Returns:** `string` — final path to the file

**Example:**
```datacode
users = table([[1, "Alice"]], ["id", "name"])
orders = table([[1, 100]], ["user_id", "amount"])
relate(users["id"], orders["user_id"])
path = save_tables_sqlite([users, orders], filename="model", env="dev")
```

See also: [creation-and-operations.md](../tables/creation-and-operations.md)

---

### JOIN functions

`inner_join`, `left_join`, `right_join`, `full_join`, `cross_join`, `semi_join`, `anti_join`, `zip_join`, `asof_join`, `apply_join`, `join_on`, `table_suffixes` — see [join.md](./join.md).

---
