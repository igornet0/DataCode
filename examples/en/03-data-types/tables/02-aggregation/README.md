# Aggregation: table aggregation methods

This set shows how to compute common aggregates over table columns.

Data used:
- `data/employees.csv` — numeric and categorical columns
- `data/orders.csv` — order amounts
- `data/duplicates.csv` — distinct behavior check

Run:

```bash
dcr examples/en/03-data-types/tables/02-aggregation/01-aggregations.dc
dcr examples/en/03-data-types/tables/02-aggregation/02-table_aggregate.dc
```

In `01-aggregations.dc` — `.aggregate` / `.aggregate_group` methods.  
In `02-table_aggregate.dc` — global `table_aggregate` / `table_aggregate_group`.

Aggregates covered:
- `count`
- `count_distinct`
- `sum`
- `avg`
- `min`
- `max`
- `first`
- `last`
- `median`
- `mode`
- `stddev`
- `variance`
- `percentile` (p90 example)
- `list`
- `any`
