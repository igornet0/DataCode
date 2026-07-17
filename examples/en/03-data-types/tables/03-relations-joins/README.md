# Relations and table joins

Examples of declaring relations (`relate`, `primary_key`) and all JOIN types in DataCode.

Data in `data/`:
- `employees.csv` — employees and departments
- `orders.csv` — orders
- `duplicates.csv` — duplicates by composite key
- `text_products.csv` — product lookup

Run:

```bash
dcr examples/en/03-data-types/tables/03-relations-joins/01-relations.dc
dcr examples/en/03-data-types/tables/03-relations-joins/02-join-basics.dc
dcr examples/en/03-data-types/tables/03-relations-joins/03-join-advanced.dc
dcr examples/en/03-data-types/tables/03-relations-joins/04-save_tables_sqlite.dc
```

## 01-relations.dc

- `primary_key(col)` — mark primary key
- `relate(col1, col2)` — declare relation between columns (for data model / export)

## 02-join-basics.dc

Equi-join on keys with expected row count checks:

| Function | Meaning |
|---------|--------|
| `inner_join` | matching rows only |
| `left_join` | all left rows |
| `right_join` | all right rows |
| `full_join` | all rows from both sides |
| `cross_join` | Cartesian product |
| `semi_join` | left rows with match (no right columns) |
| `anti_join` | left rows without match |

## 03-join-advanced.dc

- `join(left, right, on, type)` — universal function
- composite key: `[["col1", "col1"], ["col2", "col2"]]`
- `join_on` — non-equi condition (`>=`, `<=`, …)
- `zip_join` — join by row position
- `asof_join` — nearest value by time
- `apply_join` — lateral join with `fn(row) => table`
- `table.suffixes(left_suffix, right_suffix)` — suffixes on name conflicts
