# Strings

Text in **double quotes** is a string. The same string type is used for plain text and (after processing by functions) for **dates** and **money** in string form.

Type checks: [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## What `typeof` returns for strings

Depends on **content**:

| Content | `typeof` |
|---------|----------|
| Looks like a date (`YYYY-MM-DD` at start) | `"date"` |
| Contains `$`, `EUR`, `€` | `"money"` |
| Everything else | `"string"` |

There are no separate "Date" or "Money" machine types — they are strings, **named** differently in `typeof`.

```dc
print(typeof("hello"))           # string
print(typeof("2024-01-15"))      # date
print(typeof("$10.00"))          # money

```

---

## Index: character and methods

### Character by index `s[i]`

Index is by **Unicode characters** (not bytes). Negative indices are **not** supported.

```dc
s = "Hello"
print(s[0])            # one letter as a string

```

### Slice `s[start:stop:step]`

Same syntax as arrays (see [arrays](../../0-syntax/03-arrays.md#slices-arrstartstopstep)). Bounds are **Unicode characters** (like `s[i]`); negative indices and step are supported.

| Syntax | Meaning |
|--------|---------|
| `s[a:b]` | characters from index `a` up to but **not including** `b` |
| `s[:]` | entire string (new copy) |
| `s[::2]` | every second character |
| `s[::-1]` | characters in reverse order |

```dc
s = "Hello, World!"
print(s[1:4])     # ell
print(s[-3:])     # ld!
print(s[::2])     # Hlo,Wrd!

```

Result is a **new string** (not a view, unlike array slices). Slice assignment `s[a:b] = ...` is **not supported** — strings are immutable.

**Note:** `len(s)` returns length in **UTF-8 bytes**, while `s[i]` and `s[a:b]` work by **characters**; for non-ASCII text these numbers may differ.

**Examples:** [`examples/en/01-basics/strings.dc`](../../../examples/en/01-basics/strings.dc)

### Methods via string key `s["name"]`

Returns a built-in function with the same meaning as the global (convenient to write `s.upper()` instead of `upper(s)` for the first argument):

| Property | Manual call |
|----------|-------------|
| `upper` | `upper(s)` |
| `lower` | `lower(s)` |
| `trim` | `trim(s)` |
| `split` | `split(s, separator)` |
| `join` | for **array** of strings: `join(array, glue)` |
| `contains` | `contains(s, substring)` |
| `isupper` | `isupper(s)` |
| `islower` | `islower(s)` |
| `replace` | `replace(s, find, replace_with)` |
| `capitalize` | `capitalize(s)` |

Example:

```dc
t = "  a,b,c  "
print(t.trim())
print(t.split(","))

```

---

## Built-in functions

- **`str(x)`** — string representation of a value.
- **`len(s)`** — length (in the Rust implementation this is UTF-8 **byte** length; for non-ASCII the number may differ from "letter count").
- **`date(...)`**, **`money(...)`** — normalize/format as string.
- **`enum(s)`** — iterate by character in loop `(i, ch)` — see [arrays.md](arrays.md).

---

## Construct `str[N]` (not a character string)

If you index the **global function** `str` as a value: **`str[5]`** with integer **`N ≥ 0`**, you get a **descriptor object** with fields `__type`, `__length` — for typing "fixed-length string" in ORM/schemas, not an ordinary string. Details in [object.md](object.md).
