# Type Operations

← [Built-in Functions](./README.md)

More on type checking in language context: [Type Recognition and Conversion](../data-types/typeof-and-isinstance.md).

**📚 Examples:** [`examples/en/03-data-types/`](../../../examples/en/03-data-types/)

### `typeof(value)`

Returns a string with the type of the value.

**Arguments:**
- `value` (any) — value to inspect

**Returns:** `string` — type name ("int", "float", "string", "bool", "array", "table", "path", "null", "function", "date", "money", "object")

**Examples:**
```datacode
typeof(42)              # "int"
typeof(3.14)            # "float"
typeof("hello")         # "string"
typeof(true)            # "bool"
typeof([1, 2, 3])       # "array"
typeof(null)            # "null"
typeof(path("test.txt")) # "path"

```

---

### `isinstance(value, type_name)`

Checks whether a value is an instance of the specified type.

**Arguments:**
- `value` (any) — value to check
- `type_name` (string) — type name to check against

**Returns:** `bool` — `true` if value matches the type, otherwise `false`

**Examples:**
```datacode
isinstance(42, "int")           # true
isinstance(3.14, "float")       # true
isinstance("hello", "string")   # true
isinstance([1, 2], "array")     # true
isinstance(42, "string")        # false

```

---
