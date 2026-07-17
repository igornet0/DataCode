# Object (dictionary, class, service descriptors)

An **object** is a set of **name → value** pairs. In DataCode it is used as a plain **dictionary**, as a **class instance** or **class**, and sometimes as a service structure (plugins, type descriptors).

`typeof` is usually **`"object"`**, or a string from **`__plugin_namespace`** for plugin objects.

Type and class checks: [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Field access

- **`obj["key"]`**
- **`obj.key`** (when the language syntax allows it for that identifier)

For **classes**, **private** / **protected** rules and special rules for **`metadata`** apply — see `get_object` in `object_fields.rs`.

Dictionary example:

```dc
user = {"name": "Ann", "age": 20}
print(user["name"])

```

---

## Literal `{ … }` (keys)

Key semantics depend on whether a variable with that name is **declared** in the current scope:

| Syntax | When | Runtime key |
|--------|------|-------------|
| `{id: 2, name: "Bob"}` | `id`, `name` **not** declared as variables | strings `"id"`, `"name"` (convenient for `table.push`) |
| `{start_id: 100}` | `start_id` already assigned / declared | value of variable `start_id` (e.g. `0`) |
| `{"field": value}` | always | string `"field"` |
| `{1: true}`, `{true: "yes"}` | literal or expression | number / bool, etc. |

If a local variable **shadows** a field name (e.g. `id = 5` but you need field `"id"`), use an explicit string key: `{"id": 42}`.

---

## Built-in functions

| Function | Purpose |
|----------|---------|
| `len(obj)` | Number of keys |
| `isinstance(obj, Class)` | Class / inheritance check (second argument — class object with `__class_name`) |
| `isinstance(obj, "object")` | Any plain dictionary object |
| `isinstance(obj, "table")` | **Class** object inheriting the table model (`Table`) |

Exception: constructing **`ValueError(...)`** via built-in native (see language docs on `raise`).

---

## Tables and classes

Classes may inherit the built-in **`Table`**. Then an instance may behave like a table row with column fields, and `isinstance(..., "table")` — see [table.md](table.md).

---

## Descriptor `str[N]`

The construct **`str[5]`** on global **`str`** does not produce a five-character string; it returns an **object** with fields like **`__type`**, **`__length`** — for describing a fixed-length string type in ORM/schemas. See [string.md](string.md).
