# Numbers, logic (`bool`), `null`, ellipsis (`...`)

## Numbers (`int` and `float`)

You write numbers as usual: `42`, `-1`, `3.14`. **Exponential notation** is supported: `1e-9`, `2.5E+10`, `6e3` (like Python). Internally the VM stores them as floating point, but **`typeof`** distinguishes integers and floats:

- no fractional part → **`"int"`**;
- with fractional part → **`"float"`**.

### Examples

```dc
print(typeof(7))        # int
print(typeof(7.0))      # int (fractional part is zero)
print(typeof(0.5))      # float
print(10 + 3 * 2)       # arithmetic as expected

```

### Bitwise operations

Only for **`int`**: `&`, `|`, `^`, `~`, `<<`, `>>`. See [bitwise operators](../../0-syntax/10-bitwise-operators.md).

```dc
print(5 & 3)      # 1
print(1 << 10)    # 1024
```

### Useful functions

| Function | Purpose |
|----------|---------|
| `int(x)` | Integer part / parse from string |
| `float(x)` | Floating-point number |
| `abs`, `sqrt`, `pow`, `min`, `max`, `round` | Math |

Type check: `isinstance(x, "int")`, `isinstance(x, "float")` — see [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Boolean values (`bool`)

Only **`true`** and **`false`**. In `if`, `while` conditions, **truthiness** applies (see below).

```dc
print(typeof(true))     # bool
print(isinstance(false, "bool"))   # true

```

---

## `null`

Means "no value". In conditions **`null` is false**.

```dc
x = null
print(typeof(x))        # null
print(isinstance(x, "null"))       # true

```

---

## Ellipsis `...` (`ellipsis`)

Literal **`...`** — service value (e.g. in field metadata). Rare in ordinary scripts.

`typeof` → `"ellipsis"`.

---

## Indexing `x["name"]`

Numbers, `bool`, `null`, and `...` have **no** bracket properties — only operators and functions.

---

## Truthiness in conditions

For `if x { ... }`:

- **false:** `null`, `false`, number **0**, **empty** string, **empty** array or slice, empty table, empty object;
- **true:** non-zero number, non-empty string, non-empty collections, `true`, most environment objects (windows, DB engines, etc.).

Exact rules are in `is_truthy` in `src/common/value.rs`.
