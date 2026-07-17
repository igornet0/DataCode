# Связи и соединения таблиц

Примеры объявления связей (`relate`, `primary_key`) и всех типов JOIN в DataCode.

Данные в `data/`:
- `employees.csv` — сотрудники и отделы
- `orders.csv` — заказы
- `duplicates.csv` — дубли по составному ключу
- `text_products.csv` — справочник товаров

Запуск:

```bash
dcr examples/ru/03-типы\ данных/таблицы/03-связи-соединения/01-связи.dc
dcr examples/ru/03-типы\ данных/таблицы/03-связи-соединения/02-join-базовые.dc
dcr examples/ru/03-типы\ данных/таблицы/03-связи-соединения/03-join-продвинутые.dc
dcr examples/ru/03-типы\ данных/таблицы/03-связи-соединения/04-save_tables_sqlite.dc
```

## 01-связи.dc

- `primary_key(col)` — пометить первичный ключ
- `relate(col1, col2)` — объявить связь между колонками (для модели данных / экспорта)

## 02-join-базовые.dc

Equi-join по ключам с проверкой ожидаемого числа строк:

| Функция | Смысл |
|---------|--------|
| `inner_join` | только совпадающие строки |
| `left_join` | все строки слева |
| `right_join` | все строки справа |
| `full_join` | все строки с обеих сторон |
| `cross_join` | декартово произведение |
| `semi_join` | строки left с match (без колонок right) |
| `anti_join` | строки left без match |

## 03-join-продвинутые.dc

- `join(left, right, on, type)` — универсальная функция
- составной ключ: `[["col1", "col1"], ["col2", "col2"]]`
- `join_on` — non-equi условие (`>=`, `<=`, …)
- `zip_join` — соединение по позиции строк
- `asof_join` — ближайшее значение по времени
- `apply_join` — lateral join с функцией `fn(row) => table`
- `table.suffixes(left_suffix, right_suffix)` — суффиксы при конфликте имён
