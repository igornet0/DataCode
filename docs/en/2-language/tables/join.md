# Specification: Table Join Operations (JOIN)

This document describes the specification for table join operations (JOIN) in the DataCode language.

**📚 Usage examples:**
- Table joins: [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)

## 1. General

JOIN operations combine two tables based on:
- key equality,
- arbitrary logical conditions,
- temporal proximity of values,
- row position.

All JOIN operations return a new table and do not modify source data.

## 2. Basic types

- **Table** — tabular data structure
- **Row** — table row
- **Column** — column identifier
- **Expr** — logical expression
- **JoinType** — join type
- **JoinKey** — Column | (Column, Column)
- **JoinKeys** — JoinKey | List<JoinKey>

## 3. Universal JOIN function

### 3.1 Signature

```datacode
join(
    left: Table,
    right: Table,
    on: JoinKeys | Expr,
    type: JoinType = "inner",
    suffixes: (string, string) = ("_left", "_right"),
    nulls_equal: boolean = false
) -> Table

```

### 3.2 Supported JOIN types

```datacode
JoinType :=
    "inner"
  | "left"
  | "right"
  | "full"
  | "cross"
  | "semi"
  | "anti"

```

## 4. Specialized JOIN functions

All specialized functions are syntactic sugar over `join()`.

### 4.1 INNER JOIN

```datacode
inner_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Semantics:** Returns rows with a match in both tables.

**Equivalent:** `join(left, right, on, type="inner")`

**Example:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
global orders = table([[1, 100], [1, 200]], ["user_id", "amount"])
global result = inner_join(users, orders, "id", "user_id")

```

### 4.2 LEFT JOIN

```datacode
left_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Semantics:** All rows from `left` are kept. Missing values from `right` are filled with NULL.

**Example:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
global orders = table([[1, 100]], ["user_id", "amount"])
global result = left_join(users, orders, "id", "user_id")
# Bob will have NULL in order columns

```

### 4.3 RIGHT JOIN

```datacode
right_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Semantics:** All rows from `right` are kept. Missing values from `left` are filled with NULL.

### 4.4 FULL JOIN

```datacode
full_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Semantics:** Returns all rows from both tables. Missing values are filled with NULL.

### 4.5 CROSS JOIN

```datacode
cross_join(
    left: Table,
    right: Table
) -> Table

```

**Semantics:** Returns the Cartesian product of the tables.

**Example:**
```datacode
global table1 = table([[1], [2]], ["col1"])
global table2 = table([["a"], ["b"]], ["col2"])
global result = cross_join(table1, table2)
# Result: 4 rows (2 * 2)

```

### 4.6 SEMI JOIN

```datacode
semi_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Semantics:**
- Only rows from `left` are returned
- Columns from `right` are not included

**Example:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
global orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
global result = semi_join(users, orders, "id", "user_id")
# Result: only Alice and Charlie (without order columns)

```

### 4.7 ANTI JOIN

```datacode
anti_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Semantics:** Returns rows from `left` with no match in `right`.

**Example:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
global orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
global result = anti_join(users, orders, "id", "user_id")
# Result: only Bob (no orders)

```

## 5. JOIN on arbitrary condition

### 5.1 NON-EQUI JOIN

```datacode
join_on(
    left: Table,
    right: Table,
    condition: Expr,
    type: JoinType = "inner"
) -> Table

```

**Examples:**

```datacode
# Shorthand: array ["left_col", "operator", "right_col"]
global result = join_on(orders, prices, ["date", ">=", "start_date"])

# Or string: "left_col >= right_col"
global result = join_on(orders, prices, "date >= start_date")

```

## 6. Multi-key JOIN

```datacode
global on = [
    ["user_id", "id"],
    ["region", "region"]
]

```

Join is performed on all keys with logical AND.

**Example:**
```datacode
global result = inner_join(table1, table2, [["id", "id"], ["region", "region"]])

```

## 7. Temporal JOIN (ASOF)

```datacode
asof_join(
    left: Table,
    right: Table,
    on: Column,
    by: Column | List<Column>,
    direction: "backward" | "forward" | "nearest" = "backward"
) -> Table

```

**Use cases:**
- Time series
- Financial data
- Events and logs

**Example:**
```datacode
global prices = table([[100, 10.5], [200, 11.0]], ["time", "price"])
global trades = table([[150, 100], [250, 200]], ["time", "amount"])
global result = asof_join(trades, prices, "time", direction="backward")
# For each trade, find nearest price <= trade time

```

## 8. Index JOIN

```datacode
zip_join(
    left: Table,
    right: Table
) -> Table

```

**Semantics:** Joins rows by index (positionally).

**Example:**
```datacode
global table1 = table([[1], [2], [3]], ["col1"])
global table2 = table([["a"], ["b"], ["c"]], ["col2"])
global result = zip_join(table1, table2)
# Result: [[1, "a"], [2, "b"], [3, "c"]]

```

## 9. APPLY / LATERAL JOIN

```datacode
apply_join(
    left: Table,
    fn: (Row) -> Table,
    type: "inner" | "left"
) -> Table

```

**Semantics:** For each row in `left`, a sub-table is computed.

**Note:** Requires additional infrastructure for functions as values.

## 10. Column name collisions

When column names match:
- Suffixes are applied (`suffixes`)
- Or explicit rename via `as`

**Example:**
```datacode
global result = left_join(users, orders, "id", suffixes=["_user", "_order"])
# Column "id" becomes "id_user" and "id_order" if there is a collision

```

## 11. Recommended language syntax

### 11.1 Object style

```datacode
users.left_join(orders, on="user_id")

```

### 11.2 Functional style

```datacode
left_join(users, orders, on="user_id")

```

### 11.3 Pipeline style (DSL)

```datacode
users
|> left_join(orders, on="user_id")
|> anti_join(bans, on="user_id")

```

**Note:** Pipeline operator (`|>`) is not implemented yet.

## 12. Implementation guarantees

- JOIN is deterministic
- Row order is preserved when possible
- NULL is not equal to NULL when `nulls_equal = false`

## 13. Implementation note

The implementation may choose an optimal algorithm:
- Hash Join
- Merge Join
- Nested Loop

without changing language semantics.

## 14. Implementation algorithms

### Hash Join (for equi-joins)

1. Build hash table from right table by keys
2. Scan left table and look up matches in hash table
3. For each match, create result row

### Nested Loop Join (for non-equi and small tables)

1. For each row in left:
   - For each row in right:
     - Check condition
     - If true, add to result

### ASOF Join algorithm

1. Sort both tables by time column
2. For each row in left:
   - Use binary search to find nearest row in right
   - Respect direction (backward/forward/nearest)
   - If `by` is specified, limit search to matching group

## 15. Edge cases

- Empty tables
- Missing columns in keys
- NULL values in keys
- Duplicate keys
- Incompatible data types in keys
- Very large tables (possible optimization)

---

**See also:**
- [Working with tables](./creation-and-operations.md) - table creation and basic operations
- [Data types](../type-reference.md) - more on the Table type
- [Data model creation examples](../../../examples/en/08-data-model-creation/) - practical join examples
