# String Formatting (Interpolation)

DataCode strings support **interpolation**: substituting variable and expression values inside a string using the `${...}` format. Variants with variable-name output and formatting (width, alignment, precision, thousands separators, percent, exponent) are available.

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

The **`${expression:format}`** syntax controls how a value is printed. The spec is a subset of Python's format-spec mini-language:

```
[[fill]align][sign][#][0][width][grouping][.precision][type]
```

| Part | Values | Description |
|------|--------|-------------|
| align | `<` `>` `^` `=` | left, right, center; `=` pads between the sign and the digits |
| fill | any character | goes before align (`*^10`), default is a space |
| sign | `+` `-` | `+` always prints the sign, `-` only for negatives (default) |
| `#` | — | keeps the decimal point with `.0f`/`.0e` (`3.`) |
| `0` | — | zero padding after the sign (`08d` → `-0000042`) |
| width | number | **minimum** field width; longer values are never truncated |
| grouping | `,` `_` | thousands separator in the integer part |
| `.precision` | number | digits after the point; for strings, the maximum length |
| type | `f` `F` `e` `E` `d` `%` `s` | see the table below |

| Type | Description | Example | Result |
|------|-------------|---------|--------|
| `f` / `F` | fixed point (6 digits by default) | `${42.23425:.2f}` | `42.23` |
| `e` / `E` | exponent notation | `${150:.2e}` | `1.50e+02` |
| `d` | integer (a float with no fractional part also works) | `${42:08d}` | `00000042` |
| `%` | multiplies by 100 and appends `%` | `${0.256:.1%}` | `25.6%` |
| `s` | string (precision truncates) | `${w:.4s}` with `w = "datacode"` | `data` |
| none | plain string form | `${s:>6}` with `s = "ab"` | `    ab` |

```datacode
n = 42.23425
"${n:.2f}"            # "42.23"
"${n:.0f}"            # "42"
"[${n:9.2f}]"         # "[    42.23]"  — numbers align right by default
"[${n:<9.2f}]"        # "[42.23    ]"
"[${n:^9.1f}]"        # "[  42.2   ]"

s = "ab"
"[${s:5}]"            # "[ab   ]"      — non-numbers align left
"[${s:>6}]"           # "[    ab]"
"[${s:*^6}]"          # "[**ab**]"

"${1234567.891:,.2f}" # "1,234,567.89"
"${1234567:_d}"       # "1_234_567"
"${7:+d}"             # "+7"
"[${-3.5:=+8.1f}]"    # "[-    3.5]"
"${0.256:.1%}"        # "25.6%"
"${150:.2E}"          # "1.50E+02"
"${3:#.0f}"           # "3."
```

Rules:

- If the value does not fit the type (a string with `f`, a fractional number with `d`), its plain string form is printed, still padded to the width: `${s:.2f}` → `ab`, `${5.5:d}` → `5.5`.
- A colon inside `[...]` is not a format separator: `${path[:5]}` is a slice.
- A spec that is not recognised as a format (for example `${n:abc}`) is a string parse error.

---

## Combining

You can output the name and apply a format at the same time: **`${variable=:format}`**.

```datacode
n = 42.23425
"${n=:.0f}"   # "n=42"
"${n=:.2f}"   # "n=42.23"
"${n=:>10.2f}"  # "n=     42.23"

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
| `${x:9.2f}` | width 9, 2 decimals, right-aligned |
| `${s:*^10}` | centered, `*` fill |
| `${x:,.2f}` | `1,234,567.89` |
| `${x:08d}` | `00000042` |
| `${x:.1%}` | `25.6%` |
| `${x:.2e}` | `1.50e+02` |

**See also:**
- [Data Types — String](../2-language/data-types/string.md)
- [Built-in Functions](../2-language/functions/README.md) — `str()`, string functions
