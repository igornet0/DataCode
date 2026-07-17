# Tuple (`tuple`)

A tuple is a **fixed-length** sequence of values: order matters, like a mathematical pair `(x, y)` or a multi-value return from enumeration logic.

`typeof` → **`"tuple"`**.

Check: `isinstance(x, "tuple")` — [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Indexing

Only **non-negative** integer indices: `t[0]`, `t[1]`, …

**No negative indices** (unlike arrays) — `t[-1]` causes an error.

No string methods like `t["name"]` on tuples.

```dc
# Example: enum element is a tuple (index, value)
e = enum([10, 20])
pair = e[0]
print(pair[0])   # 0
print(pair[1])   # 10

```

---

## Built-in functions

- **`len(t)`** — number of elements.
- **`enum(t)`** — wrapper for indexed loop (see [arrays.md](arrays.md)).

---

## Array vs tuple (in practice)

| | Array | Tuple |
|---|--------|--------|
| `map` / `filter` / `reduce` | yes | no (use array or view) |
| Negative index | yes | no |
| `push` / list mutations | yes | not via typical array API |

For flexible processing, use an **array**; a tuple is convenient as a **small composite result**.
