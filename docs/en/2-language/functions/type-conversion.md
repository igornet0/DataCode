# Type Conversion

← [Built-in Functions](./README.md)

**📚 Examples:** [`examples/en/03-data-types/`](../../../examples/en/03-data-types/)

### `int(value)`

Converts a value to an integer.

**Arguments:**
- `value` (any) — value to convert

**Returns:** `number` — integer

**Examples:**
```datacode
int(42.7)      # 42
int("123")     # 123
int(true)      # 1
int(false)     # 0
int(null)      # 0

```

---

### `float(value)`

Converts a value to a floating-point number.

**Arguments:**
- `value` (any) — value to convert

**Returns:** `number` — float

**Examples:**
```datacode
float(42)        # 42.0
float("3.14")    # 3.14
float(true)      # 1.0
float(false)     # 0.0

```

---

### `bool(value)`

Converts a value to a boolean.

**Arguments:**
- `value` (any) — value to convert

**Returns:** `bool` — boolean value

**Examples:**
```datacode
bool(1)          # true
bool(0)          # false
bool(42)         # true
bool("")         # false
bool(null)       # false
bool([1, 2, 3])  # true
bool([])         # false

```

---

### `str(value)`

Converts a value to a string.

**Arguments:**
- `value` (any) — value to convert

**Returns:** `string` — string representation of the value

**Examples:**
```datacode
str(42)          # "42"
str(3.14)        # "3.14"
str(true)        # "true"
str(false)       # "false"
str(null)        # "null"
str([1, 2, 3])   # "[1, 2, 3]"

```

---

### `array(...)`

Creates an array from the passed arguments.

**Arguments:**
- `...` — any number of values of any type

**Returns:** `array` — array of values

**Examples:**
```datacode
array(1, 2, 3)                    # [1, 2, 3]
array("a", "b", "c")              # ["a", "b", "c"]
array(100.50, 11, "Hello")         # [100.5, 11, "Hello"]
array([1, 3], [1, 5, 6])          # [[1, 3], [1, 5, 6]]

```

---

### `date(value)`

Converts a value to **`date`**.

**Arguments:**
- `value` (`date` | string | number) — existing date, Unix seconds, or string (ISO / built-in formats)

**Returns:** `date` or `null` (if string is not recognized)

For arbitrary string formats use [`parse_date`](./date-and-time.md#parse_datestring-format).

**Examples:**
```datacode
typeof(date("2024-01-15T10:30:00Z"))   # "date"
date("2023-12-25").year                 # 2023
```

---

### `money(amount, format)`

Formats a number as a monetary amount.

**Arguments:**
- `amount` (number | string) — amount to format
- `format` (string, optional) — format string (for example, "$0.00", "0,0 $", "0 EUR")

**Returns:** `string` — formatted monetary amount

**Examples:**
```datacode
money(100.50, "$0.00")      # "$100.50"
money(250.75, "0,0 $")      # "250,75 $"
money(50, "0 EUR")          # "50 EUR"
money(100.50)               # "100.5" (no formatting)

```

---
