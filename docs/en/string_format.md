# String Formatting (Interpolation)

DataCode supports **string interpolation**: embedding variables and expressions inside double-quoted strings using `${...}`. You can optionally output as "name=value" and format numbers.

## Contents

1. [Basic interpolation](#basic-interpolation)
2. [Name=value output](#namevalue-output)
3. [Number formatting](#number-formatting)
4. [Combining options](#combining-options)
5. [Escaping](#escaping)

---

## Basic interpolation

Inside a double-quoted string, use `${expression}`. The expression is evaluated and its value is converted to string and inserted.

```datacode
let name = "Igor"
"Hello, ${name}!"   # "Hello, Igor!"

let a = 2
let b = 3
"result: ${a + b}"   # "result: 5"
```

Any expression is allowed: variables, function calls, arithmetic, property access, etc.

---

## Name=value output

The **`${variable=}`** syntax outputs both the name and the value as `name=value`. Useful for debugging and logging.

```datacode
let n = 42
let b = true
"${b=} ${n=}"   # "b=true n=42"
```

For complex expressions, the expression source is used as the "name" (e.g. `a+b=5`).

---

## Number formatting

The **`${expression:format}`** syntax lets you control how numbers are formatted. The format is given after a colon, similar to a float format specifier (e.g. Python-style).

| Format | Description              | Example (n = 42.23425) |
|--------|--------------------------|-------------------------|
| `.0f`  | Integer (0 decimal places) | `42`   |
| `.2f`  | Two decimal places       | `42.23`|
| `.4f`  | Four decimal places      | `42.2343` |

```datacode
let n = 42.23425
"${n:.2f}"    # "42.23"
"${n:.0f}"    # "42"
```

For non-numeric values the format is ignored and the usual string representation is used.

---

## Combining options

You can output the name and apply a format at the same time: **`${variable=:format}`**.

```datacode
let n = 42.23425
"${n=:.0f}"   # "n=42"
"${n=:.2f}"   # "n=42.23"
```

Order: expression, optional `=` suffix, then optional `:format`.

---

## Escaping

To include a literal `${` in the string (no interpolation), escape the dollar sign: **`\${`**. The result is the literal characters `${`.

```datacode
"Syntax: \${expr}"   # "Syntax: ${expr}"
```

---

## Quick reference

| Syntax        | Result (example)        |
|---------------|-------------------------|
| `${x}`        | value of expression     |
| `${x=}`       | `x=value`               |
| `${x:.2f}`    | number with 2 decimals  |
| `${x=:.0f}`   | `x=42` (name and integer) |

**See also:**
- [Data Types — String](./data_types.md#string)
- [Built-in Functions](./builtin_functions.md) — `str()`, string functions
