# Трансформации таблиц (виджет «Трансформация»)

Примеры соответствуют каталогу операций UI-трансформаций: каждая группа — отдельный `.dc` файл и CSV в `data/`.

## CSV-файлы

| Файл | Назначение |
|------|------------|
| `data/employees.csv` | Сотрудники: текст, числа, пустые ячейки, даты, флаги |
| `data/text_products.csv` | Текстовые поля для trim/upper/split/replace |
| `data/orders.csv` | Заказы: суммы и ISO-даты |
| `data/duplicates.csv` | Дубликаты строк для distinct |

## Примеры по группам

| Файл | Группа UI | Операции |
|------|-----------|----------|
| `01-колонки.dc` | Колонки | rename, select, drop, reorder, add |
| `02-типы_данных.dc` | Типы данных | changeType |
| `03-текст.dc` | Текст | trim, upper, lower, capitalize, replace, split, join |
| `04-числа.dc` | Числа | round, abs; ceil/floor — обходной путь |
| `05-даты.dc` | Даты | extractDatePart |
| `06-пустые_значения.dc` | Пустые значения | fillEmpty, dropEmptyRows |
| `07-строки.dc` | Строки | sort, rowNumber, distinct |
| `08-значения.dc` | Значения | valueMap |

Вспомогательные функции: `_helpers.dc` (можно копировать в свои скрипты).

## Запуск

Из корня репозитория:

```bash
datacode "examples/ru/03-типы данных/таблицы/01-колонки.dc"
datacode "examples/ru/03-типы данных/таблицы/06-пустые_значения.dc"
```

Или из папки примеров:

```bash
cd "examples/ru/03-типы данных/таблицы"
datacode 03-текст.dc
```

Пути `path("data/...")` разрешаются относительно файла скрипта.

## Что уже есть в DataCode

| Операция | DataCode сейчас |
|----------|-----------------|
| selectColumns | `table_select(table, ["A", "B"])` / `data.select(cols)` |
| dropColumn | `table_drop_column(table, col)` / `data.drop_column(col)` |
| reorderColumns | `table_select` с нужным порядком / `data.select(cols)` |
| renameColumn | `table_rename(table, map)` / `data.rename(old, new)` / `read(..., header={...})` |
| addColumn | `table_add_column(table, name, value?)` / `data.add_column(name, value?)` |
| changeType | `table_map(table, col, fn)` / `data.map(col, fn)` — `int`, `float`, `str`, `bool`, `date`, … |
| sort | `table_sort(table, "Col", true/false)` |
| dropEmptyRows | `table_drop_nulls(table [, column])` / `table.drop_nulls()` |
| rowNumber | `table_row_number(table [, column_name [, start_from]])` / `table.row_number(...)` |
| distinct | `table_distinct(table [, columns])` / `table.distinct(...)` |
| valueMap | `table_value_map(table, column, mappings)` / `table.value_map(column, mappings)` |
| trim / upper / lower / capitalize / replace | `trim()`, `upper()`, … + `data.map(col, fn)` |
| split / join колонок | `table_split_column`, `table_join_columns` / `.split_column`, `.join_columns` |
| round / abs | `round()`, `abs()` |
| extractDatePart | `date()` + `.year`, `.month`, `.day`, … |
| split / join | `split()`, `join()` построчно |

## Пока не реализовано (в других группах примеров)

- `table_change_type` (с `dateFormat` / `decimalSeparator`), `capitalize()`, `replace()`, `ceil()`/`floor()`
- `table_extract_date(..., "quarter"|"weekday"|...)`
- `table_fill_empty`
- `table_sort_multi`
- `table_add_column(..., fn(row) => ...)` — callback по строке
