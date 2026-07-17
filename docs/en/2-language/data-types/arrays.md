# Arrays, slices, and `enum(...)`

An **array** is an ordered list you can modify (add elements, sort, and so on). A **slice** (view) is a "window" into the same array without copying data. **`enum`** is a wrapper for indexed loops.

**Syntax** (literals, slices, comprehensions): [0-syntax/arrays](../../0-syntax/03-arrays.md).

Type checks: [typeof-and-isinstance.md](typeof-and-isinstance.md).

---

## Array

### How to create

```dc
a = [1, 2, 3]              # literal
b = array(10, 20)          # array(...) function
c = range(5)               # [0,1,2,3,4]
d = array_with_capacity(100)   # empty, with reserved capacity

```

References to the same array behave as **one** collection: changes are visible everywhere.

### Index `arr[i]`

- **`i`** — integer.
- **Negative index** counts from the end (like Python): `[-1]` is the last element.

```dc
v = [10, 20, 30]
print(v[0])       # 10
print(v[-1])      # 30

```

**Slices** with step — see [0-syntax/arrays](../../0-syntax/03-arrays.md#slices-arrstartstopstep); implemented in `array_ops.rs`.

### Methods `arr["name"]`

Names match **global** functions:

`push`, `pop`, `unique`, `reverse`, `sort`, `sum`, `average`, `count`, `any`, `all`, `chunk`.

Both styles are equivalent in meaning:

```dc
x = [1]
push(x, 2)
x.push(3)          # get method first, then call
print(x)

```

More examples: `examples/en/03-data-types/arrays.dc`.

### Built-in functions

- creation: `array`, `array_with_capacity`, `range`;
- size and aggregates: `len`, `sum`, `average`, `count`, `min`, `max`, `unique`, `reverse`, `sort`, `any`, `all`;
- **`map`**, **`filter`**, **`reduce`** — only for array or array slice (not tuple);
- **`enum(array)`** — see section below;
- **`chunk(arr, n)`** — split into chunks of length `n`.

Aggregates **`sum`**, **`average`**, **`unique`** also accept a **table column** — see [table.md](table.md).

---

## Array view (slice, `ArrayView`)

Looks like an array for `typeof` (**`"array"`**), but points to **part** of another array. Useful after operations like `chunk` on a slice — without extra copying.

Same methods: `push`, `pop`, …, `chunk`.

---

## `enum(...)` and enumeration

**`enum(x)`** builds a wrapper for the loop **`for (i, item) in enum(x)`**:

- for **array** or **tuple** — pairs `(index, element)`;
- for **string** — pairs `(index, character as string)`.

Accessing **`e[i]`** on the result of `enum` yields a **tuple** of two elements: index and value.

`typeof` for this wrapper — **`"enumerate"`**. **`len(e)`** — same as the length of the source collection.

---

## Quick reference

| Task | Example |
|------|---------|
| Append | `push(arr, x)` or `arr.push(x)` |
| Pop from end | `pop(arr)` |
| Any truthy element? | `any(arr)` |
| All elements truthy? | `all(arr)` |
| Transform each element | `map(arr, f)` |
