# Date and duration (`date` and `duration`)

At runtime these are **separate** values: a point in time (`date`) and an interval (`duration`). Do not confuse them with **string** `typeof` heuristics for quoted text that looks like a date — see [string.md](string.md) and the section below.

---

## `typeof` and `isinstance`

- For a **value** from `date(...)` / `now()`: `typeof` → `"date"`, `isinstance(x, date)` → `true`.
- For a **value** from `duration(...)`: `typeof` → `"duration"`, `isinstance(x, duration)` → `true`.
- Literal `"2024-01-15"` in quotes is still a **string**; by content `typeof` may return `"date"` (see [string.md](string.md)). That is **not** the same as `date("2024-01-15")` — a separate **`date`** value.

---

## Getting a date

- **`now()`** — current moment in UTC, type **`date`** (internally fixed offset 0, RFC 3339-style output).
- **`date(value)`** — parse or normalize:
  - already **`date`** → same moment;
  - **number** — Unix time in seconds; fractional part is sub-second (nanoseconds);
  - **string** — if recognized by the date parser, otherwise **`null`**.
- **`date_to_unix(value)`** — for **`date`** returns seconds as a **number** (with fractional part); for a recognizable **date string** — same; for a **number** returns it as-is; otherwise **`null`**.

Built-in details: [type conversion](../functions/type-conversion.md) (`date`), [date and time](../functions/date-and-time.md) (`now`, `date_to_unix`, `parse_date`, `format_date`, `duration`).

---

## Parsing and formatting with patterns

For strings with **non-standard** format (not ISO and not the built-in auto-parser of `date()`) use **`parse_date(string, format)`**. Pattern uses [chrono strftime](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) directives (`%Y`, `%m`, `%d`, `%H`, `%M`, `%S`, `%z`, …).

```dc
d = parse_date("15.03.2024", "%d.%m.%Y")
d = parse_date("12/10/2020", "%m/%d/%Y")
```

On format mismatch → **`null`** (like `date(string)`).

**`format_date(date, format)`** — date to string by pattern. First argument — **`date`** value (or string recognized by `date()`). Method **`d.format(format)`** is equivalent.

```dc
d = parse_date("15.03.2024", "%d.%m.%Y")
format_date(d, "%Y-%m-%d")   # "2024-03-15"
d.format("%d.%m.%Y")          # "15.03.2024"
```

In a table column: `data.map("SignedUp", fn(s) => parse_date(s, "%Y-%m-%d"))` or a wrapper function with the needed format.

| Directive | Meaning |
|-----------|---------|
| `%Y` | Year (4 digits) |
| `%m` | Month 01–12 |
| `%d` | Day 01–31 |
| `%H` | Hour 00–23 |
| `%M` | Minute 00–59 |
| `%S` | Second 00–59 |
| `%z` | UTC offset `+0000` |

Full list and differences from Python — in chrono documentation.

---

## Fields of a `date` value

A **`date`** value has **fields** (no separate global functions). Parentheses optional: **`d.year`** and **`d.year()`** are the same.

| Field | Result |
|-------|--------|
| `year`, `month`, `day` | Number: calendar year / month (1–12) / day of month in the **date's own offset** |
| `quarter` | Number: calendar quarter (**1..4**) in the **date's own offset** |
| `hour`, `minute`, `second` | Number: hours (0–23), minutes and seconds (0–59) in that offset |
| `weekday` | Number: ISO 8601 weekday (**1** = Monday … **7** = Sunday) in the calendar date of the date's offset |
| `to_utc` or `utc` | **`date`** value: same instant with **UTC (+00:00)** offset |
| `format` | Method: `d.format("%Y-%m-%d")` — see "Parsing and formatting" |

If the instant is given as a string with non-UTC offset (e.g. `+10:00`), components `year` … `second` match **local** time in that offset; after `to_utc` / `utc` the calendar date may shift relative to UTC midnight.

```dc
d = date("2024-03-15T13:45:30Z")
print(d.year, d.month, d.day)
u = d.utc
print(typeof(u))   # date

```

---

## Duration

**`duration(seconds=, minutes=, hours=, days=, milliseconds=)`** — all named arguments optional (default 0). Sum defines the interval (including fractional parts where applicable).

```dc
h = duration(hours=1)
print(typeof(h))   # duration

```

---

## Arithmetic

- **`date + duration`** and **`duration + date`** → **`date`** (forward shift).
- **`date - duration`** → **`date`** (backward shift).
- **`date - date`** → **`duration`** (signed difference).

```dc
t = now()
h = duration(hours=1)
print(t + h)
print((t + h) - t)              # same interval as h
print(date("2024-01-02") - date("2024-01-01"))

```

---

## SQLite export

When exporting to SQLite, **`duration`** values are stored as **REAL** (seconds, with fractional part).
