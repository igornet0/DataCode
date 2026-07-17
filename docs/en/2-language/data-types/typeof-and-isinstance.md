# `typeof` and `isinstance`: type checks

In short: **`typeof(x)`** returns a **string** with the type name for debugging and logic. **`isinstance(x, type)`** returns **`true`** or **`false`**, convenient in `if`.

Both work with **all values** in the language: numbers, strings, arrays, tables, functions, `null`, class objects, types from graphics/databases/plugins, and so on.

---

## `typeof(x)`

The result is a string such as `"int"`, `"array"`, `"table"`.

### Notes

- **Numbers:** no fractional part → `"int"`, with fractional part → `"float"`.
- **Strings:** the same string in memory may be classified as `"date"` (if the start looks like `YYYY-MM-DD`), as `"money"` (if it contains `$`, `EUR`, `€`), or as plain `"string"`.
- **Plugin object:** if `__plugin_namespace` is set, `typeof` may return that name instead of `"object"`.
- **Non-obvious names:** database engine — `"database_engine"`, cluster — `"database_cluster"`, table column reference — `"column"`.

### Examples

```dc
print(typeof(42))           # int
print(typeof(3.14))         # float
print(typeof("hello"))      # string
print(typeof([1, 2]))       # array
print(typeof(null))         # null

```

---

## `isinstance(x, second_argument)`

Returns **`true`** if value `x` matches the specified type, and **`false`** otherwise (or if there are fewer than two arguments).

### How to specify the second argument

1. **String** — type name (case insensitive):  
   `isinstance(x, "array")`, `isinstance(x, "table")`, `isinstance(x, "null")`.

2. **Type "constructor" from globals** — for convenience you can pass a built-in function treated as a type name:  
   `isinstance(x, int)`, `isinstance(x, str)`, `isinstance(x, array)`, `isinstance(x, Table)`, etc. (see implementation: native indices `int`, `str`, `array`, `Table` in `native_isinstance`).

3. **Class object** — for **inheritance** checks (class instance, table as `Table`):  
   the second argument is a class value with `__class_name`. Then for **`Value::Table`** it checks match with class `"Table"`, for an **object** — the class chain / table inheritance flag.

### String reference for `isinstance`

| Value `x` | Examples of matching type names (string or synonym) |
|-----------|------------------------------------------------------|
| Integer | `"int"`, `"integer"`, `"number"`, `"num"` |
| Float | `"float"` (only if the number has a fractional part); for any number also `"int"`, `"number"`, `"num"`; `"money"` — for any number |
| `true` / `false` | `"bool"`, `"boolean"` |
| Plain string | `"string"`, `"str"` |
| Date string (`YYYY-MM-DD`…) | `"date"` |
| Currency string | `"money"` |
| Array or array slice (view) | `"array"`, `"list"` |
| Bytes (`sha256`, …) | `"bytes"` (also `"array"`, `"list"`) — see [bytes.md](bytes.md) |
| Tuple | `"tuple"` |
| Table | `"table"` |
| Dictionary object | `"object"`, `"dict"`, `"dictionary"` |
| Class object inheriting Table | `"table"` (if `__extends_table` is set on the object) |
| Table column | `"column"` |
| `null` | `"null"`, `"none"` |
| Any function | `"function"` |
| Path | `"path"` |
| UUID | `"uuid"` |
| `enum(...)` | `"enumerate"` |
| `...` (ellipsis) | `"ellipsis"` |
| Window / image / figure / axis | `"window"`, `"image"`, `"figure"`, `"axis"` |
| DB engine / cluster | `"database_engine"`, `"database_cluster"` |
| Plugin object (opaque) | `"plugin_opaque"` or the type name returned by the plugin |
| Object with `__plugin_namespace` | string must match that namespace |

### Examples

```dc
# Basic checks
print(isinstance(10, int))           # true
print(isinstance(10, float))         # false
print(isinstance(2.5, float))        # true
print(isinstance("ab", str))         # true
print(isinstance([1], array))        # true
print(isinstance(null, null))        # true

# Table
t = table(["a"], [[1]])
print(isinstance(t, table))          # true

```

Class checks (inheritance) are described further in [object.md](object.md) and in the table section of [table.md](table.md).

---

## Relationship between `typeof` and `isinstance`

The name from `typeof(x)` can **often** be passed as the second argument to `isinstance`, but the rules are not always identical (for example, for numbers `typeof(3)` is `"int"`, while `isinstance(3, "number")` is also `true`). For strings `"date"` / `"money"`, `isinstance` looks at **content**, not only `typeof`.

If unsure, print for debugging:

```dc
print("typeof =", typeof(x))
print("same as isinstance =", isinstance(x, typeof(x)))

```

For many types the second call returns `true`, but for numbers and strings with subtypes there can be exceptions — use the table above.
