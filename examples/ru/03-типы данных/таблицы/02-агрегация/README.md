# Агрегация: методы обработки таблиц

Этот набор показывает, как получить популярные агрегаты по колонке таблицы.

Используемые данные:
- `data/employees.csv` — числовые и категориальные колонки
- `data/orders.csv` — суммы заказов
- `data/duplicates.csv` — проверка distinct-поведения

Запуск:

```bash
dcr examples/ru/03-типы\ данных/таблицы/02-агрегация/01-агрегации.dc
dcr examples/ru/03-типы\ данных/таблицы/02-агрегация/02-table_aggregate.dc
```

В примере `01-агрегации.dc` покрыты методы `.aggregate` / `.aggregate_group`.  
В `02-table_aggregate.dc` — глобальные `table_aggregate` / `table_aggregate_group`.

В примере покрыты агрегаты:
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
- `percentile` (пример p90)
- `list`
- `any`
