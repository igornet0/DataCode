# Mathematical Functions

← [Built-in Functions](./README.md)

**📚 Examples:** [`examples/en/01-basics/arithmetic.dc`](../../../examples/en/01-basics/arithmetic.dc)

### `abs(n)`

Returns the absolute value of a number.

**Arguments:**
- `n` (number) — number

**Returns:** `number` — absolute value, or `null` if argument is not a number

**Examples:**
```datacode
abs(-5)      # 5
abs(5)       # 5
abs(-3.14)   # 3.14

```

---

### `sqrt(n)`

Returns the square root of a number.

**Arguments:**
- `n` (number) — number (must be non-negative)

**Returns:** `number` — square root, or `null` if number is negative or argument is not a number

**Examples:**
```datacode
sqrt(16)     # 4.0
sqrt(9)      # 3.0
sqrt(-1)     # null

```

---

### `pow(base, exp)`

Raises a number to a power.

**Arguments:**
- `base` (number) — base
- `exp` (number) — exponent

**Returns:** `number` — result, or `null` if arguments are not numbers

**Examples:**
```datacode
pow(2, 3)      # 8.0
pow(10, 2)     # 100.0
pow(2, 0.5)    # 1.4142135623730951 (square root of 2)

```

---

### `min(...)`

Returns the minimum of the passed numbers or of a single array of numbers (non-numeric elements are skipped).

**Arguments:**
- `...` — any number of numbers **or** one array of numbers

**Returns:** `number` — minimum value, or `null` if there are no numeric values (for example empty array or only non-numbers)

**Examples:**
```datacode
min(1, 2, 3)        # 1
min(5, 2, 8, 1)     # 1
min(-5, -2, -10)    # -10
min([3, 1, 2])      # 1
min([])             # null

```

---

### `max(...)`

Returns the maximum of the passed numbers or of a single array of numbers (non-numeric elements are skipped).

**Arguments:**
- `...` — any number of numbers **or** one array of numbers

**Returns:** `number` — maximum value, or `null` if there are no numeric values (for example empty array or only non-numbers)

**Examples:**
```datacode
max(1, 2, 3)        # 3
max(5, 2, 8, 1)     # 8
max(-5, -2, -10)    # -2
max([3, 1, 2])      # 3
max([])             # null

```

---

### `round(n)`

Rounds a number to the nearest integer.

**Arguments:**
- `n` (number) — number to round

**Returns:** `number` — rounded number, or `null` if argument is not a number

**Examples:**
```datacode
round(3.5)      # 4
round(3.4)      # 3
round(-3.5)     # -3
round(-3.6)     # -4

```

---

### `ceil(n)`

Rounds a number up to the nearest integer.

**Arguments:**
- `n` (number) — number

**Returns:** `number` — rounded-up number, or `null` if argument is not a number

**Examples:**
```datacode
ceil(3.1)      # 4
ceil(-3.1)     # -3
```

---

### `floor(n)`

Rounds a number down to the nearest integer.

**Arguments:**
- `n` (number) — number

**Returns:** `number` — rounded-down number, or `null` if argument is not a number

**Examples:**
```datacode
floor(3.9)      # 3
floor(-3.1)     # -4
```

---

### `divmod(a, b)`

Returns pair `(q, r)` — quotient and remainder with Python semantics (floor division for negatives).

**Arguments:**
- `a`, `b` (int | float)

**Returns:** tuple of two numbers of the same kind, or error when `b == 0`

**Examples:**
```datacode
q, r = divmod(10, 3)    # q=3, r=1
q, r = divmod(-10, 3)   # floor semantics
```

---

### `isinf(x)`

Checks whether a number is infinity (`±∞`).

**Arguments:**
- `x` (int | float | number)

**Returns:** `bool`

**Examples:**
```datacode
isinf(42)          # false
```

---
