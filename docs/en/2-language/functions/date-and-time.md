# Date and Time

← [Built-in Functions](./README.md)

Functions for the current moment, intervals, and Unix time. String-to-date conversion — [`date()`](./type-conversion.md#datevalue).

**📚 More on types:** [date and duration](../data-types/date-and-duration.md)

### `now()`

Current date and time (UTC).

**Arguments:** none

**Returns:** `date`

**Examples:**
```datacode
now()
typeof(now())   # "date"
```

---

### `date_to_unix(value)`

Unix time in **seconds** (fractional part — nanoseconds).

**Arguments:**
- `value` (date | string | number) — date, ISO string, or ready-made Unix number

**Returns:** `number`, or `null`

**Examples:**
```datacode
date_to_unix(now())
date_to_unix("2024-01-15T10:30:00Z")
```

---

### `parse_date(string, format)`

Parse a string into **`date`** using an explicit chrono/strftime template (`%Y`, `%m`, `%d`, …).

**Arguments:** `string`, `format` (both string)

**Returns:** `date` or `null`

**Examples:**
```datacode
parse_date("15.03.2024", "%d.%m.%Y")
parse_date("12/10/2020", "%m/%d/%Y")
```

---

### `format_date(date, format)`

Format **`date`** as a string. Equivalent to `d.format(format)`.

**Arguments:** `date` (or recognizable string), `format` (string)

**Returns:** `string`

**Examples:**
```datacode
d = parse_date("15.03.2024", "%d.%m.%Y")
format_date(d, "%Y-%m-%d")
d.format("%d.%m.%Y")
```

---

### `duration(seconds, minutes, hours, days, milliseconds)`

Creates a `duration` value as the sum of components (all arguments optional, default 0).

**Arguments:**
- `seconds`, `minutes`, `hours`, `days`, `milliseconds` (number)

**Returns:** `duration`

**Examples:**
```datacode
duration(seconds=30)
duration(minutes=5, seconds=30)
duration(hours=1)
```

---
