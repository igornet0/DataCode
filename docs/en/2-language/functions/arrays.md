# Array Functions

← [Built-in Functions](./README.md)

**📚 Examples:** [`examples/en/01-basics/`](../../../examples/en/01-basics/), [`examples/en/06-loops/`](../../../examples/en/06-loops/)

### `push(array, item)`

Appends an element to the end of an array (mutates the original array).

**Arguments:**
- `array` (array) — array to modify
- `item` (any) — element to append

**Returns:** `array` — the same array (for chaining)

**Examples:**
```datacode
arr = [1, 2, 3]
push(arr, 4)        # arr is now [1, 2, 3, 4]
push(arr, "hello")  # arr is now [1, 2, 3, 4, "hello"]

```

---

### `pop(array)`

Removes and returns the last element of an array.

**Arguments:**
- `array` (array) — array to modify

**Returns:** `any` — last element, or `null` if array is empty or argument is not an array

**Examples:**
```datacode
arr = [1, 2, 3]
pop(arr)     # returns 3, arr is now [1, 2]
pop(arr)     # returns 2, arr is now [1]
pop([])      # null

```

---

### `unique(array)`

Returns a new array with unique elements (preserves order of first occurrence).

**Arguments:**
- `array` (array) — array to process

**Returns:** `array` — new array with unique elements, or `null` if argument is not an array

**Examples:**
```datacode
unique([1, 2, 2, 3, 1])        # [1, 2, 3]
unique(["a", "b", "a", "c"])   # ["a", "b", "c"]

```

---

### `reverse(array)`

Reverses element order in an array (mutates the original array).

**Arguments:**
- `array` (array) — array to modify

**Returns:** `array` — same array with reversed order

**Examples:**
```datacode
arr = [1, 2, 3]
reverse(arr)  # arr is now [3, 2, 1]

```

---

### `sort(array)`

Sorts array elements by string representation (mutates the original array).

**Arguments:**
- `array` (array) — array to sort

**Returns:** `array` — sorted array

**Examples:**
```datacode
arr = [3, 1, 2]
sort(arr)  # arr is now [1, 2, 3]

arr2 = ["c", "a", "b"]
sort(arr2)  # arr2 is now ["a", "b", "c"]

```

---

### `sum(array)`

Computes the sum of all numbers in an array.

**Arguments:**
- `array` (array) — array of numbers

**Returns:** `number` — sum, or `0` if there are no numbers or argument is not an array

**Examples:**
```datacode
sum([1, 2, 3])        # 6
sum([10, 20, 30])     # 60
sum([1.5, 2.5, 3.0])  # 7.0

```

---

### `average(array)`

Computes the arithmetic mean of numbers in an array.

**Arguments:**
- `array` (array) — array of numbers

**Returns:** `number` — average, or `0` if there are no numbers or argument is not an array

**Examples:**
```datacode
average([1, 2, 3])        # 2.0
average([10, 20, 30])     # 20.0
average([1.5, 2.5, 3.0])  # 2.3333333333333335

```

---

### `count(array)`

Returns the number of elements in an array.

**Arguments:**
- `array` (array) — array

**Returns:** `number` — element count, or `0` if argument is not an array

**Examples:**
```datacode
count([1, 2, 3])      # 3
count([])             # 0
count(["a", "b"])     # 2

```

---

### `map(collection, fn | native)`

Transforms each element of an array (or array view) and returns a **new array of the same length**.

- `fn(x) => ...` — one argument per element.
- `fn(x, i) => ...` — value and numeric index `i` (zero-based).
- `str` (or any **single-argument** builtin call) — applied to each element.

**Returns:** `array` or error on callback failure.

---

### `filter(collection, predicate)`

Keeps elements in order where the predicate is truthy. **User functions only** (not raw natives).

- `fn(x) => ...` or `fn(x, i) => ...` — same arity rules as `map`.

**Returns:** new `array` (length not greater than source).

---

### `reduce(collection, fn, initial)`

Folds an array with **required** `initial`. Callback must have arity 2: `(acc, item)`.

`reduce([], fn, initial)` returns `initial` without calling the function.

A `reduce` variant **without** initial value is **not** supported.

**Returns:** single value.

---

### `any(collection)` / `all(collection)`

- `any` — `true` if **at least one** element is truthy.
- `all` — `true` if **all** elements are truthy and collection is **not empty**.

Support arrays and lazy iterables (for example result of `map`).

**Examples:**
```datacode
any([0, false, 1])    # true
all([1, 2, 3])        # true
all([])               # false
```

---

### `enum(iterable)`

Lazy wrapper `(index, element)` for `for i, x in enum(arr)` loops.

**Arguments:**
- `iterable` (array | tuple | string | table)

**Returns:** enumerate-iterable, or `null`

---

### `array_with_capacity(n)`

Empty array with pre-allocated capacity (handy before a loop with `push`).

**Arguments:**
- `n` (int) — capacity, `0 … 1_000_000_000`

**Returns:** `array`

**Examples:**
```datacode
buf = array_with_capacity(1000)
```

---

### `set()` / `set(array)`

Creates a set: empty or from array elements (elements must be hashable).

**Arguments:**
- optionally one `array`

**Returns:** `set`

**Examples:**
```datacode
s = set()
s = set([1, 2, 2, 3])   # set with unique elements
```

Set methods: `.add()`, `.remove()`, `.copy()`, etc. — see [Object](../data-types/object.md).

---
