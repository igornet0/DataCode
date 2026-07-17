# Table transformations (Transformation widget)

Examples match the UI transformation catalog: each group is a separate `.dc` file with CSV in `data/`.

## CSV files

| File | Purpose |
|------|---------|
| `data/employees.csv` | Employees: text, numbers, empty cells, dates, flags |
| `data/text_products.csv` | Text fields for trim/upper/split/replace |
| `data/orders.csv` | Orders: amounts and ISO dates |
| `data/duplicates.csv` | Duplicate rows for distinct |

## Examples by group

| File | UI group | Operations |
|------|----------|------------|
| `01-columns.dc` | Columns | rename, select, drop, reorder, add |
| `02-data-types.dc` | Data types | changeType |
| `03-text.dc` | Text | trim, upper, lower, capitalize, replace, split, join |
| `04-numbers.dc` | Numbers | round, abs; ceil/floor — workaround |
| `05-dates.dc` | Dates | extractDatePart |
| `06-null-values.dc` | Null values | fillEmpty, dropEmptyRows |
| `07-rows.dc` | Rows | sort, rowNumber, distinct |
| `08-values.dc` | Values | valueMap |

Helper functions: `_helpers.dc` (copy into your own scripts).

## Run

From repository root:

```bash
datacode "examples/en/03-data-types/tables/01-transformation/01-columns.dc"
datacode "examples/en/03-data-types/tables/01-transformation/06-null-values.dc"
```

Or from the examples folder:

```bash
cd "examples/en/03-data-types/tables/01-transformation"
datacode 03-text.dc
```

Paths `path("data/...")` resolve relative to the script file.

## Available in DataCode today

| Operation | DataCode now |
|----------|-----------------|
| selectColumns | `table_select(table, ["A", "B"])` / `data.select(cols)` |
| dropColumn | `table_drop_column(table, col)` / `data.drop_column(col)` |
| reorderColumns | `table_select` with desired order / `data.select(cols)` |
| renameColumn | `table_rename(table, map)` / `data.rename(old, new)` / `read(..., header={...})` |
| addColumn | `table_add_column(table, name, value?)` / `data.add_column(name, value?)` |
| changeType | `table_map(table, col, fn)` / `data.map(col, fn)` — `int`, `float`, `str`, `bool`, `date`, … |
| sort | `table_sort(table, "Col", true/false)` |
| dropEmptyRows | `table_drop_nulls(table [, column])` / `table.drop_nulls()` |
| rowNumber | `table_row_number(table [, column_name [, start_from]])` / `table.row_number(...)` |
| distinct | `table_distinct(table [, columns])` / `table.distinct(...)` |
| valueMap | `table_value_map(table, column, mappings)` / `table.value_map(column, mappings)` |
| trim / upper / lower / capitalize / replace | `trim()`, `upper()`, … + `data.map(col, fn)` |
| split / join columns | `table_split_column`, `table_join_columns` / `.split_column`, `.join_columns` |
| round / abs | `round()`, `abs()` |
| extractDatePart | `date()` + `.year`, `.month`, `.day`, … |
| split / join | `split()`, `join()` per row |

## Not yet implemented (workarounds in other example groups)

- `table_change_type` (with `dateFormat` / `decimalSeparator`), `capitalize()`, `replace()`, `ceil()`/`floor()`
- `table_extract_date(..., "quarter"|"weekday"|...)`
- `table_fill_empty`
- `table_sort_multi`
- `table_add_column(..., fn(row) => ...)` — per-row callback
