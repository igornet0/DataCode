# JOIN-функции

← [Встроенные функции](./README.md)

Глобальные функции объединения таблиц. Полная спецификация: [таблицы/join.md](../таблицы/join.md).

**📚 Примеры:** [`examples/ru/08-создание модели данных/`](../../../examples/ru/08-создание%20модели%20данных/)

| Функция | Назначение |
|---------|------------|
| `inner_join(left, right, on, …)` | Пересечение по ключам |
| `left_join` | Все строки левой таблицы |
| `right_join` | Все строки правой |
| `full_join` | Полное внешнее |
| `cross_join(left, right)` | Декартово произведение |
| `semi_join` | Строки left, для которых есть match в right |
| `anti_join` | Строки left без match в right |
| `zip_join` | По позиции строк |
| `asof_join` | По ближайшему времени |
| `apply_join` | JOIN с пользовательской логикой |
| `join_on(left, right, on, …)` | Универсальный JOIN |
| `table_suffixes(left, right, left_suffix, right_suffix)` | Задать суффиксы колонок при конфликте имён |

**Общие аргументы** (кроме `cross_join`):
- `left`, `right` (table)
- `on` — ключ(и) или выражение
- `type` (string, опционально) — тип соединения
- `suffixes` (tuple, опционально)

**Связанные функции:**
- [`relate(pk_col, fk_col, ...)`](./таблицы.md#relatepk_col-fk_col--relatepk_col-fk_col-) — объявить связи (звезда: первая = PK, остальные = FK)
- [`primary_key(col)`](./таблицы.md#primary_keycol) — пометить первичный ключ

---
