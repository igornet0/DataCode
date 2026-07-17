# Utilities

← [Built-in Functions](./README.md)

**📚 Examples:** [`examples/en/01-basics/`](../../../examples/en/01-basics/)

### `print(...)`

Prints values to the console. Accepts any number of arguments.

**Arguments:**
- `...` — any number of values of any type

**Returns:** `null`

**Examples:**
```datacode
print("Hello, World!")
print("Number:", 42, "String:", "test")
print()  # Empty line

```

---

### `len(value)`

Returns the length of a string, array, table, or object.

**Arguments:**
- `value` (string | array | table | object) — value to get length of

**Returns:** `number` — length of value, or `null` if type is not supported

**Examples:**
```datacode
len("Hello")        # 5
len([1, 2, 3])      # 3
len([])             # 0

```

---

### `copy(value)`

Returns a **deep copy** of a container: array, tuple, set, object, or table. Scalar values (numbers, strings, `null`, dates, etc.) are returned unchanged.

**Important:** ordinary assignment (`b = a`) for containers **does not copy** data — both variables refer to the same object. To get an independent copy, use `copy(a)`, the `.clone()` method on arrays/tuples, or `set.copy()` on sets — all perform the same deep copy.

**Arguments:**
- `value` (any) — value to copy

**Returns:** deep copy of container or the same scalar value; `null` when called with no arguments

**Errors:** `TypeError` for non-copyable types (functions, generators, iterators, database connections, etc.)

**Examples:**
```datacode
a = [1, [2, 3]]
b = copy(a)
b[1][0] = 99
print(a[1][0])   # 2 — original array unchanged

x = [1, 2]
y = x            # reference: y and x share one array
z = copy(x)      # independent copy

s = set([1, 2])
t = copy(s)      # same as s.copy()

copy(42)         # 42
copy("hello")    # "hello"

```

---

### `range(end)` / `range(start, end)` / `range(start, end, step)`

Creates an array of numbers in the specified range.

**Arguments:**
- `end` (number) — end value (not included)
- `start` (number, optional) — start value (default 0)
- `step` (number, optional) — step (default 1)

**Returns:** `array` — array of numbers

**Examples:**
```datacode
range(5)              # [0, 1, 2, 3, 4]
range(1, 5)           # [1, 2, 3, 4]
range(1, 10, 2)       # [1, 3, 5, 7, 9]
range(10, 0, -1)      # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

```

---
