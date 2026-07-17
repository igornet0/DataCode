# Встроенные функции DataCode

Справочник **117** глобальных встроенных функций языка (источник: [`BUILTIN_GLOBAL_NAMES`](../../../../src/vm/globals.rs)).  
Алиасы в коде: `read` → `read_file`, `read_bin` → `read_file_bin`; также `save`, `save_tables_sqlite` (extended builtins).

**📚 Примеры:**
- Базовые: [`examples/ru/01-основы/`](../../../examples/ru/01-основы/)
- Функции: [`examples/ru/04-функции/`](../../../examples/ru/04-функции/)
- Данные и таблицы: [`examples/ru/08-создание модели данных/`](../../../examples/ru/08-создание%20модели%20данных/)

## Оглавление

| Категория | Документ | Функции |
|-----------|----------|---------|
| Утилиты | [утилиты.md](./утилиты.md) | `print`, `len`, `copy`, `range` |
| Преобразование типов | [преобразование-типов.md](./преобразование-типов.md) | `int`, `float`, `bool`, `str`, `array`, `date`, `money` |
| Работа с типами | [типы.md](./типы.md) | `typeof`, `isinstance` |
| Дата и время | [дата-и-время.md](./дата-и-время.md) | `now`, `parse_date`, `format_date`, `date_to_unix`, `duration` |
| Пути и файлы | [пути.md](./пути.md) | `path`, …, `getcwd`, `list_files` |
| Математика | [математика.md](./математика.md) | `abs`, …, `divmod`, `isinf` |
| Строки | [строки.md](./строки.md) | `upper`, …, `starts_with`, `ends_with`, `ord` |
| Массивы и коллекции | [массивы.md](./массивы.md) | `push`, …, `any`, `all`, `enum`, `set`, … |
| Таблицы | [таблицы.md](./таблицы.md) | `table`, `read`/`read_file`, `table_*`, `archive`, `datasource`, … |
| JOIN | [join.md](./join.md) | `inner_join`, `left_join`, … (13 функций) |
| Криптография и RNG | [криптография-и-случайность.md](./криптография-и-случайность.md) | `sha256`, `random`, `random_int`, … |

## Полный список (из исходников)

<details>
<summary>117 глобальных функций — развернуть</summary>

Полный список имён — в [`BUILTIN_GLOBAL_NAMES`](../../../../src/vm/globals.rs) (индексы `0..116`). Кратко по группам:

| Индексы | Группа | Ключевые имена |
|---------|--------|----------------|
| 0–11 | утилиты / типы | `print`, `len`, `range`, `int`…`money` |
| 12–20 | пути | `path`, `path_exists`, … `path_len` |
| 21–26 | математика | `abs`, `sqrt`, `pow`, `min`, `max`, `round`, `ceil`, `floor` |
| 27–40 | строки | `upper`…`capitalize`, `starts_with`, `ends_with` |
| 41–50 | массивы | `push`…`all` |
| 51–79 | таблицы | `table`, `read_file`, `table_*`, `merge_tables`, `show_table` |
| 80–82 | дата / пути | `now`, `getcwd`, `list_files` |
| 83–95 | JOIN | `inner_join`…`table_suffixes` |
| 96–98 | связи | `relate`, `primary_key`, `enum` |
| 99–103 | коллекции | `Table`, `array_with_capacity`, `map`, `filter`, `reduce` |
| 104–111 | криптография / RNG | `sha256`…`random` |
| 112–120 | дата / утилиты | `date_to_unix`, `parse_date`, `format_date`, `duration`, `set`, `divmod`, `isinf`, `copy`, `ord` |
| 121–126 | таблицы (расшир.) | `table_row_number`, `table_distinct`, `table_value_map`, `table_aggregate`, `table_aggregate_group` |
| 127–128 | файлы / API | `archive`, `datasource` |

Алиасы: `read` = `read_file`, `read_bin` = `read_file_bin`. Extended: `save`, `save_tables_sqlite`.

</details>

## Не путать с модулями

После 117 builtins в globals регистрируются **встроенные модули** (`import plot`, `import uuid`, …) — это отдельные API, не глобальные функции.

**Устанавливаемые пакеты DPM** (например `ml` через `dpm add ml`) — тоже не builtins; см. [ml-модуль.md](../модули/ml-модуль.md) и [модули/README.md](../модули/README.md).

## Быстрый выбор

| Задача | Куда |
|--------|------|
| Случайное float [0, 1) | [random()](./криптография-и-случайность.md#random) |
| Случайное целое | [random_int](./криптография-и-случайность.md#random_intmin-max) |
| Хеш строки | [sha256](./криптография-и-случайность.md#sha256data) |
| Текущее время | [now()](./дата-и-время.md#now) |
| Скопировать контейнер | [copy()](./утилиты.md#copyvalue) |
| JOIN таблиц | [join.md](./join.md) |
| Фильтр таблицы | [таблицы.md](./таблицы.md) |

## Связанные документы

- [Типы данных](../справочник-типов.md)
- [Работа с таблицами](../таблицы/создание-и-операции.md)
- [ML — пример MNIST](../../1-примеры/11-mnist-mlp.md)
