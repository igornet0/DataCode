# String Formatting (Interpolation)

DataCode strings support **interpolation**: substituting variable and expression values inside a string using the `${...}` format. Variants with variable-name output and number formatting are available.

## Contents

1. [Basic interpolation](#basic-interpolation)
2. [Name=value output](#namevalue-output)
3. [Number formatting](#number-formatting)
4. [Combining](#combining)
5. [Escaping](#escaping)

---

## Basic interpolation

Inside a double-quoted string you can use `${expression}`. The expression is evaluated and its value is inserted into the string (as when converting to string).

```datacode
name = "Igor"
"Hello, ${name}!"   # "Hello, Igor!"

a = 2
b = 3
"result: ${a + b}"   # "result: 5"

```

Any expression is supported: variables, function calls, arithmetic, property access, etc.

---

## Name=value output

The **`${variable=}`** syntax outputs not only the value but also the name as `name=value`. Convenient for debug output and logs.

```datacode
n = 42
b = true
"${b=} ${n=}"   # "b=true n=42"

```

For complex expressions, the expression text is used as the "name" (for example, `a+b=5`).

---

## Number formatting

The **`${expression:format}`** syntax lets you set how a number is printed. The format is given after a colon and is similar to a Python-style float specifier.

| Format | Description | Example (n = 42.23425) |
|--------|-------------|------------------------|
| `.0f` | Integer (0 decimal places) | `42` |
| `.2f` | Two decimal places | `42.23` |
| `.4f` | Four decimal places | `42.2343` |

```datacode
n = 42.23425
"${n:.2f}"    # "42.23"
"${n:.0f}"    # "42"

```

For non-numeric values the format is ignored and the usual string representation is used.

---

## Combining

You can output the name and apply a format at the same time: **`${variable=:format}`**.

```datacode
n = 42.23425
"${n=:.0f}"   # "n=42"
"${n=:.2f}"   # "n=42.23"

```

Order: expression first, then optional `=` suffix, then optional `:format`.

---

## Escaping

To insert a literal `${` sequence into a string (without interpolation), escape the dollar sign: **`\${`**. The result is the literal `${`.

```datacode
"Syntax: \${expr}"   # "Syntax: ${expr}"

```

---

## Quick reference

| Syntax | Result (example) |
|--------|------------------|
| `${x}` | value of expression |
| `${x=}` | `x=value` |
| `${x:.2f}` | number with 2 decimal places |
| `${x=:.0f}` | `x=42` (name and integer) |

**See also:**
- [Data Types — String](../2-language/data-types/string.md)
- [Built-in Functions](../2-language/functions/README.md) — `str()`, string functions
