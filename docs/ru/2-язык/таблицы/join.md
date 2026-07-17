# Спецификация: Операции объединения таблиц (JOIN)

Этот документ описывает спецификацию операций объединения таблиц (JOIN) в языке DataCode.

**📚 Примеры использования:**
- Объединение таблиц: [`examples/ru/08-создание модели данных/`](../../../../examples/ru/08-создание%20модели%20данных/)

## 1. Общие положения

Операции JOIN предназначены для объединения двух таблиц на основе:
- равенства ключей,
- произвольных логических условий,
- временной близости значений,
- позиции строк.

Все JOIN-операции возвращают новую таблицу и не изменяют исходные данные.

## 2. Базовые типы

- **Table** — табличная структура данных
- **Row** — строка таблицы
- **Column** — идентификатор колонки
- **Expr** — логическое выражение
- **JoinType** — тип соединения
- **JoinKey** — Column | (Column, Column)
- **JoinKeys** — JoinKey | List<JoinKey>

## 3. Универсальная функция JOIN

### 3.1 Сигнатура

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

### 3.2 Поддерживаемые типы JOIN

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

## 4. Специализированные JOIN-функции

Все специализированные функции являются синтаксическим сахаром над `join()`.

### 4.1 INNER JOIN

```datacode
inner_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Семантика:** Возвращает строки, для которых существует совпадение в обеих таблицах.

**Эквивалент:** `join(left, right, on, type="inner")`

**Пример:**
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

**Семантика:** Все строки из `left` сохраняются. Отсутствующие значения из `right` заполняются NULL.

**Пример:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
global orders = table([[1, 100]], ["user_id", "amount"])
global result = left_join(users, orders, "id", "user_id")
# Bob будет иметь NULL в колонках orders

```

### 4.3 RIGHT JOIN

```datacode
right_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Семантика:** Все строки из `right` сохраняются. Отсутствующие значения из `left` заполняются NULL.

### 4.4 FULL JOIN

```datacode
full_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Семантика:** Возвращает все строки из обеих таблиц. Отсутствующие значения заполняются NULL.

### 4.5 CROSS JOIN

```datacode
cross_join(
    left: Table,
    right: Table
) -> Table

```

**Семантика:** Возвращает декартово произведение таблиц.

**Пример:**
```datacode
global table1 = table([[1], [2]], ["col1"])
global table2 = table([["a"], ["b"]], ["col2"])
global result = cross_join(table1, table2)
# Результат: 4 строки (2 * 2)

```

### 4.6 SEMI JOIN

```datacode
semi_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Семантика:**
- Возвращаются только строки `left`
- Колонки `right` не включаются

**Пример:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
global orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
global result = semi_join(users, orders, "id", "user_id")
# Результат: только Alice и Charlie (без колонок orders)

```

### 4.7 ANTI JOIN

```datacode
anti_join(
    left: Table,
    right: Table,
    on: JoinKeys
) -> Table

```

**Семантика:** Возвращает строки `left`, не имеющие совпадений в `right`.

**Пример:**
```datacode
global users = table([[1, "Alice"], [2, "Bob"], [3, "Charlie"]], ["id", "name"])
global orders = table([[1, 100], [3, 300]], ["user_id", "amount"])
global result = anti_join(users, orders, "id", "user_id")
# Результат: только Bob (без заказов)

```

## 5. JOIN по произвольному условию

### 5.1 NON-EQUI JOIN

```datacode
join_on(
    left: Table,
    right: Table,
    condition: Expr,
    type: JoinType = "inner"
) -> Table

```

**Примеры:**

```datacode
# Упрощенный синтаксис: массив ["left_col", "operator", "right_col"]
global result = join_on(orders, prices, ["date", ">=", "start_date"])

# Или строка: "left_col >= right_col"
global result = join_on(orders, prices, "date >= start_date")

```

## 6. JOIN по нескольким ключам

```datacode
global on = [
    ["user_id", "id"],
    ["region", "region"]
]

```

Соединение выполняется по всем ключам с логическим AND.

**Пример:**
```datacode
global result = inner_join(table1, table2, [["id", "id"], ["region", "region"]])

```

## 7. Временной JOIN (ASOF)

```datacode
asof_join(
    left: Table,
    right: Table,
    on: Column,
    by: Column | List<Column>,
    direction: "backward" | "forward" | "nearest" = "backward"
) -> Table

```

**Назначение:**
- Временные ряды
- Финансовые данные
- События и логи

**Пример:**
```datacode
global prices = table([[100, 10.5], [200, 11.0]], ["time", "price"])
global trades = table([[150, 100], [250, 200]], ["time", "amount"])
global result = asof_join(trades, prices, "time", direction="backward")
# Для каждого trade находится ближайшая цена <= времени trade

```

## 8. Индексный JOIN

```datacode
zip_join(
    left: Table,
    right: Table
) -> Table

```

**Семантика:** Соединяет строки по индексу (позиционно).

**Пример:**
```datacode
global table1 = table([[1], [2], [3]], ["col1"])
global table2 = table([["a"], ["b"], ["c"]], ["col2"])
global result = zip_join(table1, table2)
# Результат: [[1, "a"], [2, "b"], [3, "c"]]

```

## 9. APPLY / LATERAL JOIN

```datacode
apply_join(
    left: Table,
    fn: (Row) -> Table,
    type: "inner" | "left"
) -> Table

```

**Семантика:** Для каждой строки `left` вычисляется подтаблица.

**Примечание:** Требует дополнительной инфраструктуры для поддержки функций как значений.

## 10. Коллизии имён колонок

Если имена колонок совпадают:
- Применяется суффиксация (`suffixes`)
- Либо явное переименование через `as`

**Пример:**
```datacode
global result = left_join(users, orders, "id", suffixes=["_user", "_order"])
# Колонка "id" станет "id_user" и "id_order" если есть коллизия

```

## 11. Рекомендуемый синтаксис ЯП

### 11.1 Объектный стиль

```datacode
users.left_join(orders, on="user_id")

```

### 11.2 Функциональный стиль

```datacode
left_join(users, orders, on="user_id")

```

### 11.3 Pipeline-стиль (DSL)

```datacode
users
|> left_join(orders, on="user_id")
|> anti_join(bans, on="user_id")

```

**Примечание:** Pipeline-оператор (`|>`) пока не реализован.

## 12. Гарантии реализации

- JOIN является детерминированным
- Порядок строк сохраняется, если это возможно
- NULL не равен NULL, если `nulls_equal = false`

## 13. Примечание для реализации

Реализация может выбирать оптимальный алгоритм:
- Hash Join
- Merge Join
- Nested Loop

без изменения семантики языка.

## 14. Алгоритмы реализации

### Hash Join (для equi-joins)

1. Построить хеш-таблицу из правой таблицы по ключам
2. Пройти по левой таблице и искать совпадения в хеш-таблице
3. Для каждого совпадения создать результирующую строку

### Nested Loop Join (для non-equi и малых таблиц)

1. Для каждой строки left:
   - Для каждой строки right:
     - Проверить условие
     - Если условие истинно, добавить в результат

### ASOF Join алгоритм

1. Отсортировать обе таблицы по временной колонке
2. Для каждой строки left:
   - Используя бинарный поиск, найти ближайшую строку right
   - Учесть direction (backward/forward/nearest)
   - Если указан `by`, ограничить поиск соответствующей группой

## 15. Обработка краевых случаев

- Пустые таблицы
- Отсутствующие колонки в ключах
- NULL значения в ключах
- Дублирующиеся ключи
- Несовместимые типы данных в ключах
- Очень большие таблицы (возможная оптимизация)

---

**См. также:**
- [Работа с таблицами](./таблицы/создание-и-операции.md) - создание и базовые операции с таблицами
- [Типы данных](./справочник-типов.md) - подробнее о типе Table
- [Примеры создания моделей данных](../../../../examples/ru/08-создание%20модели%20данных/) - практические примеры объединения таблиц

