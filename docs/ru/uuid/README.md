# Модуль uuid — генерация уникальных идентификаторов

Модуль `uuid` предназначен для генерации уникальных идентификаторов (UUID): случайных (v4), время-упорядоченных (v7), детерминированных (v3, v5), разбора строк, работы с байтовым представлением и извлечения метаданных.

## Импорт модуля

Модуль импортируется:

```datacode
import uuid
```

## Генерация UUID

### uuid.v4() -> UUID
### uuid.random() -> UUID

Генерирует случайный UUID версии 4. Алиас `random()` эквивалентен `v4()`.

**Возвращает:** UUID (случайный).

**Пример:**
```datacode
let u = uuid.v4()
print(uuid.to_string(u))
# или
let r = uuid.random()
print(uuid.to_string(r))
```

### uuid.v7() -> UUID
### uuid.new() -> UUID

Генерирует время-упорядоченный UUID версии 7. Удобен для индексов и сортировки по времени создания. Алиас `new()` эквивалентен `v7()`.

**Возвращает:** UUID (время-упорядоченный).

**Пример:**
```datacode
let u = uuid.v7()
print(uuid.to_string(u))
let n = uuid.new()
print(uuid.to_string(n))
```

## Разбор и строки

### uuid.parse(s: string) -> UUID | null

Разбирает строку в формате с дефисами (например, `550e8400-e29b-41d4-a716-446655440000`) и возвращает UUID. При неверном формате возвращает `null`.

**Параметры:**
- `s` (string): строка в стандартном формате UUID.

**Возвращает:** UUID или `null` при ошибке разбора.

**Пример:**
```datacode
let parsed = uuid.parse("550e8400-e29b-41d4-a716-446655440000")
if parsed != null {
    print(uuid.to_string(parsed))
} else {
    print("Ошибка разбора")
}
```

### uuid.to_string(u: UUID) -> string

Преобразует UUID в строку в стандартном формате с дефисами.

**Параметры:**
- `u` (UUID): значение UUID.

**Возвращает:** строка вида `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.

**Пример:**
```datacode
let u = uuid.v4()
print(uuid.to_string(u))
```

## Байты

### uuid.to_bytes(u: UUID) -> array

Возвращает массив из 16 чисел (0–255) — байтовое представление UUID.

**Параметры:**
- `u` (UUID): значение UUID.

**Возвращает:** массив из 16 чисел.

**Пример:**
```datacode
let u = uuid.v4()
let bytes = uuid.to_bytes(u)
print(len(bytes))  # 16
```

### uuid.from_bytes(arr: array) -> UUID | null

Собирает UUID из массива из 16 чисел (0–255). Массив должен содержать ровно 16 целых чисел в диапазоне 0–255. При неверных данных возвращает `null` и устанавливает нативную ошибку.

**Параметры:**
- `arr` (array): массив из 16 чисел 0–255.

**Возвращает:** UUID или `null` при ошибке.

**Пример:**
```datacode
let u = uuid.v4()
let bytes = uuid.to_bytes(u)
let restored = uuid.from_bytes(bytes)
print(uuid.to_string(u) == uuid.to_string(restored))  # true
```

## Детерминированные UUID (v3, v5)

### uuid.v3(namespace: UUID, name: string) -> UUID

Генерирует UUID версии 3 (MD5) по namespace и имени. Одинаковые namespace и name дают один и тот же UUID.

**Параметры:**
- `namespace` (UUID): namespace (используйте uuid.DNS, uuid.URL, uuid.OID или свой UUID).
- `name` (string): имя.

**Возвращает:** UUID.

**Пример:**
```datacode
let u = uuid.v3(uuid.URL, "https://example.com/page")
print(uuid.to_string(u))
```

### uuid.v5(namespace: UUID, name: string) -> UUID

Генерирует UUID версии 5 (SHA-1) по namespace и имени. Одинаковые namespace и name дают один и тот же UUID. Рекомендуется вместо v3.

**Параметры:**
- `namespace` (UUID): namespace.
- `name` (string): имя.

**Возвращает:** UUID.

**Пример:**
```datacode
let u = uuid.v5(uuid.DNS, "example.com")
print(uuid.to_string(u))
```

### Стандартные namespace: uuid.DNS, uuid.URL, uuid.OID

Константы UUID для использования в v3 и v5:

- **uuid.DNS** — namespace для DNS имён.
- **uuid.URL** — namespace для URL.
- **uuid.OID** — namespace для OID (иерархических идентификаторов).

**Пример:**
```datacode
print(uuid.to_string(uuid.DNS))
print(uuid.to_string(uuid.URL))
let u5 = uuid.v5(uuid.OID, "1.2.3.4.5")
```

## Метаданные

### uuid.version(u: UUID) -> number

Возвращает номер версии UUID (1–7).

**Параметры:**
- `u` (UUID): значение UUID.

**Возвращает:** число 1–7.

**Пример:**
```datacode
let u4 = uuid.v4()
print(uuid.version(u4))  # 4
let u7 = uuid.v7()
print(uuid.version(u7))  # 7
```

### uuid.variant(u: UUID) -> number

Возвращает вариант UUID: 0=NCS, 1=RFC4122, 2=Microsoft, 3=Future. Обычно используется 1 (RFC4122).

**Параметры:**
- `u` (UUID): значение UUID.

**Возвращает:** число 0–3.

### uuid.timestamp(u: UUID) -> number | null

Возвращает Unix-время (секунды с дробной частью) для UUID версий 1 и 7. Для v4 возвращает `null`.

**Параметры:**
- `u` (UUID): значение UUID.

**Возвращает:** число (Unix-время) или `null` для v4.

**Пример:**
```datacode
let u7 = uuid.v7()
let ts = uuid.timestamp(u7)
if ts != null {
    print("Время создания: " + ts)
}
```

## Примеры

Практические примеры расположены в каталогах:

- **Русский:** [examples/ru/13-uuid/](../../../examples/ru/13-uuid/)
- **English:** [examples/en/13-uuid/](../../../examples/en/13-uuid/)

### Содержание примеров (ru)

1. **01-базовое_использование.dc** — импорт, v4/v7, to_string, parse, typeof.
2. **02-детерминированные_uuid.dc** — v3 и v5 с namespace DNS, URL, OID.
3. **03-байты.dc** — to_bytes и from_bytes, круговая проверка.
4. **04-метаданные.dc** — version, variant, timestamp.
5. **05-практический_пример.dc** — генерация id для записей, массив, объект.

### Содержание примеров (en)

1. **01-basic-usage.dc** — import, v4, v7, to_string, parse, typeof.
2. **02-deterministic-uuid.dc** — v3, v5 with namespace.
3. **03-bytes.dc** — to_bytes, from_bytes, round-trip.
4. **04-metadata.dc** — version, variant, timestamp.
5. **05-practical-example.dc** — record ids, array, object.

**Запуск примера (из корня проекта):**
```bash
datacode --bin datacode examples/ru/13-uuid/01-базовое_использование.dc
```
