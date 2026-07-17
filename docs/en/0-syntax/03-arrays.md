# Arrays

An ordered, mutable list of values. Elements are listed comma-separated in square brackets; indexing and slicing follow a Python-like style.

**Type semantics, methods, and built-in functions** (`push`, `map`, `enum`, …): [../2-language/data-types/README.md](../2-language/data-types/README.md) (Array type).

---

## Literal `[ … ]`

```datacode
empty = []
nums = [1, 2, 3]
mixed = [1, "two", true, null]
nested = [[1, 2], [3, 4]]

```

A trailing comma after the last element is allowed: `[1, 2, 3,]`.

Elements can be any expression:

```datacode
n = 3
row = [n * 2, n + 1, "x"]

```

---

## Other ways to create an array

```datacode
a = array(10, 20, 30)           # from arguments
b = array_with_capacity(100)    # empty, with reserved capacity
c = range(5)                    # lazy iterator 0..4 (not an array)
d = range(1, 11)                # 1..10
e = range(0, 10, 2)             # 0, 2, 4, 6, 8
arr = array(range(n))           # materialize into an indexable array arr[i]

```

`range(...)` returns a **lazy iterator** (`Iterable`): O(1) memory; elements are produced in `for … in range(…)` and in list comprehension.  
`len(range(a, b))` and `len(range(a, b, step))` compute length by formula without building an array.  
Arguments must be integer `int` or integer `number` (for example `1 << n`); a fractional literal `5.0` is not accepted.  
`range(stop)` — from `0` up to `stop` **not including** `stop`.  
`range(start, stop)` and `range(start, stop, step)` work the same way: the upper bound is not included. Step cannot be `0`.

---

## Index `arr[i]`

```datacode
v = [10, 20, 30, 40]
print(v[0])       # 10
print(v[2])       # 30
print(v[-1])      # 40 — last element
print(v[-2])      # 30

```

A negative index counts from the end of the array.

**Assignment by index** modifies the original array:

```datacode
arr = [1, 2, 3]
arr[0] = 9
print(arr)        # [9, 2, 3]

```

---

## Slices `arr[start:stop:step]`

The same syntax works for **strings** (`str`); see [Data Types — String](../2-language/data-types/string.md).

Syntax as in Python. Omitted bounds mean "from the start" / "to the end":

| Syntax | Meaning |
|--------|---------|
| `arr[a:b]` | elements from index `a` up to `b` **not including** `b` |
| `arr[:]` | entire array (view — a "window" without copying data) |
| `arr[::2]` | every second element |
| `arr[::-1]` | elements in reverse order |
| `arr[2:6:2]` | from index 2 to 6 with step 2 |

```datacode
a = [0, 1, 2, 3, 4, 5, 6, 7]
print(a[1:4])     # [1, 2, 3]
print(a[-3:])     # last three
print(a[::2])     # [0, 2, 4, 6]

```

**Slice assignment** replaces a segment of the array; an empty list on the right removes elements:

```datacode
arr = [1, 2, 3, 4, 5]
arr[1:3] = [20, 30]    # [1, 20, 30, 4, 5]
arr[1:4] = []          # remove elements 1..3 → [1, 5]

```

More on slice representation (view): [Data Types — Array](../2-language/data-types/README.md).

---

## List comprehension — `[ expression for … ]`

A Python-style list generator; the result is a **new** array.

### Simple forms

```datacode
squares = [x * x for x in [1, 2, 3]]           # [1, 4, 9]
zeros = [0 for _ in range(5)]                  # [0, 0, 0, 0, 0]
indexed = [x for x in range(10)]               # [0, 1, …, 9]
ones = [1 for _ in range(n)]                    # n ones

```

The `_` variable in the `for` header is an ordinary name: the iterator value is unused, but the loop runs the required number of times.

### `if` filter

```datacode
evens = [x for x in range(10) if x % 2 == 0]   # [0, 2, 4, 6, 8]

```

### Multiple `for` loops

Nested loops are iterated left to right (as in Python):

```datacode
grid = [x * y for x in [1, 2] for y in [10, 20]]
# [10, 20, 20, 40]

matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
# outer and inner range are lazy; result is an array of arrays

```

Nested comprehension where each element is another comprehension (`[[e for _ in range(c)] for _ in range(r)]`) is supported the same way as in Python.

`if` between two `for` clauses applies to the nearest left loop:

```datacode
pairs = [100 * x + y for x in [1, 2] if x == 2 for y in [1, 2]]
# [201, 202]

```

### Unpacking in the header

```datacode
sums = [a + b for a, b in [[1, 2], [3, 4]]]    # [3, 7]
picked = [weights[i] for i in items]

```

The same patterns as in `for … in …` are supported: `for x in`, `for x, y in`, `for (x, y) in`, `for [x, y] in`, `for x, _, y in`.

### Scope

Variables introduced in a comprehension **do not** change same-named variables outside:

```datacode
x = 100
a = [x for x in [1, 2]]
print(x)    # 100

```

---

## Iteration with `for`

An array is an iterable value:

```datacode
for i in [1, 2, 3, 4, 5] {
    print(i)
}

for i in range(5) {
    print(i)
}

for ch in parts {
    print(len(ch), ch)
}

```

With an index — via `enum`:

```datacode
for i, item in enum([10, 20, 30]) {
    print(i, item)
}

```

More: [loops](./05-loops.md).

---

## Mutating an array

An array is **mutable**: multiple variables can refer to the same list.

```datacode
a = [1]
b = a
push(a, 2)
print(b)        # [1, 2] — visible in b too

stack = [1, 2, 3]
push(stack, 4)
stack.push(5)   # method or global function

```

Adding, removing, sorting, and aggregates — in [Data Types](../2-language/data-types/README.md) and the `arrays.dc` example.

---

## Examples in the repository

| File | Topic |
|------|-------|
| [`arrays.dc`](../../../examples/en/03-data-types/arrays.dc) | literals, `push`/`pop`, `map`/`filter`/`reduce` |
| [`for_loops.dc`](../../../examples/en/06-loops/for_loops.dc) | `for i in […]`, `for i in range(…)` |
| [`edit_distance.dc`](../../../examples/en/09-advanced/dp/edit_distance.dc) | nested comprehension, DP tables |
| [`knapsack_01.dc`](../../../examples/en/09-advanced/dp/knapsack_01.dc) | `[0 for _ in range(n)]`, index filtering |
| [`coin_change.dc`](../../../examples/en/09-advanced/dp/coin_change.dc) | fixed-size arrays, `dp[0] = 0` |

Overview: [1 — Examples / 03-data-types](../1-examples/03-data-types.md).

## Related sections

- [loops](./05-loops.md) — `for … in …`, `range`, `enum`
- [functions](./06-functions.md) — `fn`, lambdas `fn(x) => …` for `map`/`filter`
- [Data Types](../2-language/data-types/README.md) — methods, `typeof`, `chunk`, tuples vs arrays
