# Функции создания таблиц в DataCode

## Обзор

DataCode предоставляет две эквивалентные функции для создания таблиц из данных массивов:
- `table(data, headers?)` - Оригинальная функция создания таблицы
- `table_create(data, headers?)` - Альтернативное имя для лучшей ясности

Обе функции создают структуры таблиц из двумерных массивов данных с опциональными заголовками колонок.

**📚 Примеры использования:**
- Создание моделей данных: [`examples/ru/09-создание модели данных/`](../../examples/ru/09-создание%20модели%20данных/)
- Работа с файлами: [`examples/ru/01-основы/`](../../examples/ru/01-основы/)

## Синтаксис

```datacode
table(data)
table(data, headers)

table_create(data)
table_create(data, headers)
```

### Параметры

- **data** (Array): Двумерный массив, где каждый подмассив представляет строку таблицы
- **headers** (Array, опционально): Массив строк, представляющих имена колонок

### Возвращаемое значение

- **Table**: Структура таблицы DataCode со строками и колонками

## Примеры

### Базовое создание таблицы

```datacode
# Создать простую числовую таблицу
global data = [[1, 25], [2, 30], [3, 35]]
global my_table = table_create(data)
show_table(my_table)
```

Вывод:
```
┌──────────┬──────────┐
│ Column_0 │ Column_1 │
├──────────┼──────────┤
│ 1        │ 25       │
│ 2        │ 30       │
│ 3        │ 35       │
└──────────┴──────────┘
```

### Таблица с пользовательскими заголовками

```datacode
# Создать таблицу с пользовательскими именами колонок
global employee_data = [
    [1, "Alice", 28, 75000],
    [2, "Bob", 35, 82000],
    [3, "Charlie", 42, 68000]
]
global headers = ["id", "name", "age", "salary"]
global employees = table_create(employee_data, headers)
show_table(employees)
```

Вывод:
```
┌────┬─────────┬─────┬────────┐
│ id │ name    │ age │ salary │
├────┼─────────┼─────┼────────┤
│ 1  │ Alice   │ 28  │ 75000  │
│ 2  │ Bob     │ 35  │ 82000  │
│ 3  │ Charlie │ 42  │ 68000  │
└────┴─────────┴─────┴────────┘
```

### Смешанные типы данных

```datacode
# Таблица с различными типами данных
global mixed_data = [
    [1, "Active", true, "$50000"],
    [2, "Inactive", false, "$60000"],
    [3, "Pending", true, "$55000"]
]
global headers = ["id", "status", "enabled", "budget"]
global status_table = table_create(mixed_data, headers)
show_table(status_table)
```

### Пример сводки по отделам

```datacode
# Создать сводную таблицу по отделам
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

Вывод:
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

## Работа с созданными таблицами

После создания таблицы вы можете использовать различные функции для работы с таблицами:

```datacode
# Создать таблицу
global data = [[1, "Alice", 28], [2, "Bob", 35], [3, "Charlie", 42]]
global headers = ["id", "name", "age"]
global my_table = table_create(data, headers)

# Показать информацию о таблице
table_info(my_table)

# Показать первые 2 строки
global head_table = table_head(my_table, 2)
show_table(head_table)

# Выбрать определенные колонки
global names_only = table_select(my_table, ["name", "age"])
show_table(names_only)

# Фильтровать данные (два варианта записи)
global adults = table_where(my_table, "age", ">", 30)
# или через синтаксис скобок: таблица["колонка" оператор значение]
global adults2 = my_table["age" > 30]
show_table(adults)

# Сортировать по возрасту
global sorted_table = table_sort(my_table, "age")
show_table(sorted_table)
```

**📚 Примеры:** [`examples/ru/09-создание модели данных/`](../../examples/ru/09-создание%20модели%20данных/)

## Обработка ошибок

Функция `table_create` вернет ошибки в следующих случаях:

1. **Не предоставлены аргументы**:
   ```datacode
   global my_table = table_create()  # Ошибка: Неверное количество аргументов
   ```

2. **Данные не являются массивом**:
   ```datacode
   global my_table = table_create("not an array")  # Ошибка: Ошибка типа
   ```

3. **Несогласованные длины строк**:
   ```datacode
   global bad_data = [[1, 2], [3, 4, 5]]  # Предупреждение: Несоответствие длины строк
   global my_table = table_create(bad_data)
   ```

## Рекомендации

1. **Используйте описательные заголовки**: Всегда предоставляйте осмысленные имена колонок
   ```datacode
   # Хорошо
   global headers = ["employee_id", "full_name", "department", "salary"]
   
   # Избегайте
   global headers = ["col1", "col2", "col3", "col4"]
   ```

2. **Согласованные типы данных**: Сохраняйте типы данных согласованными внутри колонок
   ```datacode
   # Хорошо - согласованные числовые данные
   global ages = [[25], [30], [35]]
   
   # Избегайте - смешанные типы в одной колонке
   global mixed = [[25], ["thirty"], [35]]
   ```

3. **Обработка отсутствующих данных**: Используйте значения null для отсутствующих данных
   ```datacode
   global data_with_nulls = [
       [1, "Alice", 28],
       [2, "Bob", null],
       [3, "Charlie", 42]
   ]
   ```

## Эквивалентность функций

Обе функции `table` и `table_create` функционально идентичны:

```datacode
global data = [[1, 2], [3, 4]]

# Эти вызовы эквивалентны
global table1 = table(data)
global table2 = table_create(data)

# Оба создают одинаковую структуру таблицы
```

Используйте то имя, которое кажется более естественным в вашем коде. `table_create` может быть более самодокументируемым для новых пользователей.

## Связанные функции

- `show_table(table)` - Отобразить таблицу в форматированном виде
- `table_info(table)` - Показать метаданные и статистику таблицы
- `table_head(table, n)` - Получить первые n строк
- `table_tail(table, n)` - Получить последние n строк
- `table_select(table, columns)` - Выбрать определенные колонки
- `table_where(table, column, operator, value)` - Фильтровать строки (альтернатива: `data["колонка" op значение]`, например `data["age" = 28]` или `data["age" > 30]`)
- `table_sort(table, column)` - Сортировать по колонке
- `table_distinct(table, column)` - Получить уникальные значения
- `table_sample(table, n)` - Случайная выборка строк

**📚 Подробнее:** См. [Встроенные функции](./builtin_functions.md#функции-работы-с-таблицами)

---

**См. также:**
- [Типы данных](./data_types.md) - подробнее о типе Table
- [Встроенные функции](./builtin_functions.md) - полный список функций для работы с таблицами
- [Примеры создания моделей данных](../../examples/ru/09-создание%20модели%20данных/) - практические примеры

