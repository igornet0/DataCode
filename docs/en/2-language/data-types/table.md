# Table and column

A table is named **columns** and **rows** of data (like a simplified Excel or SQL result). The separate **column** type appears when you take one column from a table: you can use it like a "virtual array" in `sum`, `len`, and similar functions.

Full type checks for **any** value, including tables: **[typeof and isinstance](typeof-and-isinstance.md)**.

---

## Table (`table`)

### Purpose

- store data in rows and columns;
- filter, sort, take first/last rows;
- join with other tables;
- inherit classes from built-in **`Table`** in the object model.

`typeof(table)` → string `"table"`.

### How to create

Usually via **`table(...)`** (see project function docs). Example:

```dc
# Rows first (array of arrays), then optional headers
t = table([["Ann", 20], ["Bob", 25]], ["name", "age"])
print(len(t))                    # row count
print(t["columns"])              # array of column names

```

### Index access `t[...]`

| Expression | Result |
|------------|--------|
| `t["column_name"]` | **Column** (lazy reference): then `col[i]` for a cell |
| `t[["col1", "col2"]]` | **Columns** reference for row-wise `.map(fn)` (not a sub-table; use `.select([...])` to project) |
| `t["rows"]` | Array of rows (each row — array or object of fields) |
| `t["columns"]` | Array of strings — column names |
| `t[0]`, `t[1]`, … | **Object** (dictionary) for one row: key = column name |

Example:

```dc
col = t["age"]
print(len(col))                  # rows in this column
row0 = t[0]
print(row0["name"])              # value in column name for first row

```

If the column does not exist — error like "column not found".

### Table methods

#### `add_row(row)`

Adds one row **in-place** (like `push` on an array). Argument — array of cell values.

```dc
t = table([[1, "Alice"]], ["id", "name"])
t.add_row([2, "Bob"])
print(len(t))   # 2
```

**Rules:**
- length of `row` must **exactly** match the number of columns;
- for empty table with headers (`table([], ["a", "b"])`) expected length = `len(headers)`;
- for table without columns (`table([])`) — `ValueError`;
- argument must be an array, otherwise `TypeError`;
- returns the same table; aliases (`b = t`) see changes.

**Errors:**
- `ValueError: row length N does not match table columns M` — wrong element count;
- `ValueError: cannot add_row to table without columns` — no headers and no rows.

**Example:** [`examples/en/03-data-types/table_push.dc`](../../../examples/en/03-data-types/table_push.dc)

### Built-in functions for tables

**View and transform:** `table`, `table_info`, `table_head`, `table_tail`, `table_select`, `table_sort`, `table_where`, `show_table`, `merge_tables`.

**Join tables:** `inner_join`, `left_join`, `right_join`, `full_join`, `cross_join`, `semi_join`, `anti_join`, `zip_join`, `asof_join`, `apply_join`, `join_on`, `table_suffixes`.

**Relations:** `relate`, `primary_key`.

**Class for inheritance:** global **`Table`** — used in class declarations inheriting the table model; for type checks see below.

### `join` function

- If the first argument is a **table** and there are at least three arguments, **table join** is invoked.
- If the first argument is a **string array** and the second is a separator, strings are **joined** into one string (array "join").

---

## Table column (`column`)

Appears as the result of `t["col"]`. Not a separate literal in the language — an internal "column reference" type.

`typeof(column)` → `"column"`.

### Indexing

- `col[i]` — cell value in row `i` (zero-based, like table rows).

### Built-in functions

Where implemented, you can pass a column as the first argument:

- **`len(col)`** — row count;
- **`sum`**, **`average`**, **`count`**, **`unique`** — column aggregates (for numbers and unique values — see implementation);
- **`set(col)`** — set of unique hashable column values.

Example:

```dc
ages = t["age"]
print(sum(ages))
print(unique(ages))
print(set(ages))

```

---

## Several columns (`columns`)

Appears as the result of `t[["col1", "col2", ...]]` (at least two names). Used to build a per-row formula from several fields.

`typeof(cols)` → `"columns"`.

```dc
orders!["avg_check"] = orders[["total", "quantity"]].map(fn(v, q) => v / q)
```

Indexing cells of a columns reference (`cols[0]`) is not supported; use `.map(fn)` or `map(cols, fn)`.

---

## Type checks: table and classes

**`isinstance`** for **all types** is in [typeof-and-isinstance.md](typeof-and-isinstance.md).

Briefly for tables:

```dc
print(isinstance(t, "table"))     # true for built-in table
print(isinstance(t, Table))       # same for table value (via Table class)

```

For a **class object** declared with inheritance from **`Table`**, `isinstance(obj, "table")` may be **true** if the object has the internal table inheritance flag (`__extends_table`). Details — [object.md](object.md).
