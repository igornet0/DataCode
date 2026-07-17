# Table Creation Functions in DataCode

## Overview

DataCode provides two equivalent functions for creating tables from array data:
- `table(data, headers?)` - Original table creation function
- `table_create(data, headers?)` - Alternative name for clarity

Both create table structures from two-dimensional array data with optional column headers.

**📚 Usage examples:**
- Data model creation: [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)
- File operations: [`examples/en/01-basics/`](../../../examples/en/01-basics/)

## Syntax

```datacode
table(data)
table(data, headers)

table_create(data)
table_create(data, headers)

```

### Parameters

- **data** (Array): Two-dimensional array where each sub-array is a table row
- **headers** (Array, optional): Array of strings — column names

### Return value

- **Table**: DataCode table structure with rows and columns

## Examples

### Basic table creation

```datacode
# Create a simple numeric table
global data = [[1, 25], [2, 30], [3, 35]]
global my_table = table_create(data)
show_table(my_table)

```

Output:
```
┌──────────┬──────────┐
│ Column_0 │ Column_1 │
├──────────┼──────────┤
│ 1        │ 25       │
│ 2        │ 30       │
│ 3        │ 35       │
└──────────┴──────────┘
```

### Table with custom headers

```datacode
# Create table with custom column names
global employee_data = [
    [1, "Alice", 28, 75000],
    [2, "Bob", 35, 82000],
    [3, "Charlie", 42, 68000]
]
global headers = ["id", "name", "age", "salary"]
global employees = table_create(employee_data, headers)
show_table(employees)

```

Output:
```
┌────┬─────────┬─────┬────────┐
│ id │ name    │ age │ salary │
├────┼─────────┼─────┼────────┤
│ 1  │ Alice   │ 28  │ 75000  │
│ 2  │ Bob     │ 35  │ 82000  │
│ 3  │ Charlie │ 42  │ 68000  │
└────┴─────────┴─────┴────────┘
```

### Mixed data types

```datacode
# Table with various data types
global mixed_data = [
    [1, "Active", true, "$50000"],
    [2, "Inactive", false, "$60000"],
    [3, "Pending", true, "$55000"]
]
global headers = ["id", "status", "enabled", "budget"]
global status_table = table_create(mixed_data, headers)
show_table(status_table)

```

### Department summary example

```datacode
# Create department summary table
global summary = [
    ["Engineering", 5, 425000],
    ["Marketing", 3, 242500], 
    ["HR", 2, 126000]
]
global summary_headers = ["department", "count", "total_salary"]
global summary_table = table_create(summary, summary_headers)

print("Department Summary:")
show_table(summary_table)

```

Output:
```
Department Summary:
┌─────────────┬───────┬──────────────┐
│ department  │ count │ total_salary │
├─────────────┼───────┼──────────────┤
│ Engineering │ 5     │ 425000       │
│ Marketing   │ 3     │ 242500       │
│ HR          │ 2     │ 126000       │
└─────────────┴───────┴──────────────┘
```

## Working with created tables

After creating a table, you can use various table functions:

```datacode
# Create table
global data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
global headers = ["id", "name", "age"]
global my_table = table_create(data, headers)

# Show table info
table_info(my_table)

# Show first 2 rows
global head_table = table_head(my_table, 2)
show_table(head_table)

# Select specific columns
global names_only = table_select(my_table, ["name", "age"])
show_table(names_only)

# Filter data (two syntax variants)
global adults = table_where(my_table, "age", ">", 30)
# or bracket syntax: table["column" operator value]
global adults2 = my_table["age" > 30]
global houston_or_adult = my_table["age" > 30 or "city" == "Houston"]
global houston_and_adult = my_table["age" > 30 and "city" == "Houston"]
show_table(adults)

# Sort by age
global sorted_table = table_sort(my_table, "age")
show_table(sorted_table)

```

### Incremental row addition (`add_row`)

Instead of a `push` loop and rebuilding with `table()`, grow the table with **`add_row`**:

```datacode
global t = table([], ["id", "name", "age"])
t.add_row([1, "Alice", 28])
t.add_row([2, "Bob", 35])
show_table(t)

```

The array length in `add_row` must match the column count. To combine ready-made tables use **`merge_tables([t1, t2])`**.

**📚 Example:** [`examples/en/03-data-types/table_push.dc`](../../../examples/en/03-data-types/table_push.dc)

**📚 Examples:** [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)

## Error handling

`table_create` returns errors in these cases:

1. **No arguments provided**:
   ```datacode
   global my_table = table_create()  # Error: Invalid argument count
   
```

2. **Data is not an array**:
   ```datacode
   global my_table = table_create("not an array")  # Error: Type error
   
```

3. **Inconsistent row lengths**:
   ```datacode
   global bad_data = [[1, 2], [3, 4, 5]]  # Warning: Row length mismatch
   global my_table = table_create(bad_data)
   
```

## Recommendations

1. **Use descriptive headers**: Always provide meaningful column names
   ```datacode
   # Good
   global headers = ["employee_id", "full_name", "department", "salary"]
   
   # Avoid
   global headers = ["col1", "col2", "col3", "col4"]
   
```

2. **Consistent data types**: Keep data types consistent within columns
   ```datacode
   # Good - consistent numeric data
   global ages = [[25], [30], [35]]
   
   # Avoid - mixed types in one column
   global mixed = [[25], ["thirty"], [35]]
   
```

3. **Missing data**: Use null for missing values
   ```datacode
   global data_with_nulls = [
       [1, "Alice", 28],
       [2, "Bob", null],
       [3, "Charlie", 42]
   ]
   
```

## Function equivalence

Both `table` and `table_create` are functionally identical:

```datacode
global data = [[1, 2], [3, 4]]

# These calls are equivalent
global table1 = table(data)
global table2 = table_create(data)

# Both create the same table structure

```

Use whichever name feels more natural. `table_create` may be more self-documenting for new users.

## Related functions

- `show_table(table)` - Display formatted table
- `table_info(table)` - Show metadata and statistics
- `table_head(table, n)` - First n rows
- `table_tail(table, n)` - Last n rows
- `table_select(table, columns)` - Select columns
- `table_where(table, column, operator, value)` - Filter rows (alternative: `data["column" op value]`, e.g. `data["age" = 28]` or `data["age" > 30]`)
- `table_sort(table, column)` - Sort by column
- `table.add_row(row)` - Add row in-place (method, not global)
- `table_row_number(table, column_name?, start_from?)` - Add row number column (method: `table.row_number(...)`)
- `table_distinct(table, columns?)` - Remove duplicate rows (method: `table.distinct(...)`)
- `table_sample(table, n)` - Random row sample

**📚 More:** See [Table functions](../functions/tables.md)

---

**See also:**
- [Data types](../type-reference.md) - more on the Table type
- [Table functions](../functions/tables.md) — full list of built-in table functions
- [Data model creation examples](../../../examples/en/08-data-model-creation/) - practical examples
