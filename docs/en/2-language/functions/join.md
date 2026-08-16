# JOIN Functions

← [Built-in Functions](./README.md)

Global table join functions. Full specification: [tables/join.md](../tables/join.md).

**📚 Examples:** [`examples/en/08-data-model-creation/`](../../../examples/en/08-data-model-creation/)

| Function | Purpose |
|----------|---------|
| `inner_join(left, right, on, …)` | Intersection by keys |
| `left_join` | All rows from left table |
| `right_join` | All rows from right table |
| `full_join` | Full outer join |
| `cross_join(left, right)` | Cartesian product |
| `semi_join` | Rows from left with a match in right |
| `anti_join` | Rows from left without a match in right |
| `zip_join` | By row position |
| `asof_join` | By nearest time |
| `apply_join` | JOIN with custom logic |
| `join_on(left, right, on, …)` | Universal JOIN |
| `table_suffixes(left, right, left_suffix, right_suffix)` | Set column suffixes on name conflicts |

**Common arguments** (except `cross_join`):
- `left`, `right` (table)
- `on` — key(s) or expression
- `type` (string, optional) — join type
- `suffixes` (tuple, optional)

**Related functions:**
- [`relate(pk_col, fk_col, ...)`](./tables.md#relatepk_col-fk_col--relatepk_col-fk_col-) — declare relationships (star: first = PK, rest = FKs)
- [`primary_key(col)`](./tables.md#primary_keycol) — mark a primary key

---
