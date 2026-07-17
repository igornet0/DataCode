# Loops

Repeat a block of code: **`while`** (while a condition is true) and **`for … in …`** (iterate over a collection).

## while

```datacode
counter = 0
while counter < 5 {
    print("Counter:", counter)
    counter = counter + 1
}

```

The loop runs while the condition is **truthy**. Remember to update variables in the body — otherwise the loop never ends.

```datacode
sum = 0
i = 1
while i <= 10 {
    sum = sum + i
    i = i + 1
}

```

The condition may never be true — the body is simply skipped:

```datacode
x = 10
while x > 20 {
    print("will not run")
}
print("x =", x)

```

## for … in …

The main loop form is **iteration over an iterable value**:

```datacode
for i in [0, 1, 2, 3, 4] {
    print("i =", i)
}

```

The **iterator variable** (`i`, `ch`, unpack variables) exists **only inside the loop** — after `}` it is not accessible. Assignments in the loop body (`sum = sum + x`) remain in the function scope. More: [scoping and closures](../2-language/scoping-and-closures.md).

### Arrays and `range`

```datacode
sum = 0
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
    sum = sum + i
}

for i in range(5) {
    print(i)          # 0, 1, 2, 3, 4
}

for i in range(1, 11) {
    print(i)          # 1 … 10
}

```

Array syntax, slices, and comprehension: [arrays](./03-arrays.md). Methods and `enum`: [Data Types](../2-language/data-types/README.md).

### Iterator variable

```datacode
fn print_table(numbers) {
    for i in numbers {
        print(i, "* 2 =", i * 2)
    }
}

print_table([1, 2, 3, 4, 5])

```

### Unpacking in the header

```datacode
for i, ch in enum("abc") {
    print(i, ch)
}

for x, y in pairs {
    print(x, y)
}

for _ in range(n) {
    # body runs n times; counter value not needed
}

```

Supported patterns: `for x in`, `for x, y in`, `for (x, y) in`, `for [x, y] in`, `for x, _, y in`.

### One-line for

```datacode
for i in items: print(i)

```

## break and continue

```datacode
for i in items {
    if i == skip: continue
    if i < 0: break
    print(i)
}

```

## Nested loops

```datacode
for i in [1, 2, 3] {
    for j in [1, 2] {
        print("i =", i, ", j =", j)
    }
}

```

```datacode
i = 1
while i <= 3 {
    j = 1
    while j <= 2 {
        print(i, j)
        j = j + 1
    }
    i = i + 1
}

```

Mixed variant — `for` with `while` inside:

```datacode
for i in [1, 2, 3] {
    j = 1
    while j <= i {
        print("i =", i, ", j =", j)
        j = j + 1
    }
}

```

## Counter via while

If you need an explicit counter with an arbitrary step, use **`while`**:

```datacode
i = 0
while i < 10 {
    print(i)
    i = i + 1
}

```

C-style syntax `for (init; cond; step)` is **not supported**.

## Stream functions and for

In `stream fn`, a `for` loop iterates over values produced via `return` / `ireturn`. See [stream-functions](./08-stream-functions.md).

## Examples

| File | Topic |
|------|-------|
| [`while_loops.dc`](../../../examples/en/06-loops/while_loops.dc) | `while`, counter, sum |
| [`for_loops.dc`](../../../examples/en/06-loops/for_loops.dc) | `for i in …`, `range`, functions |
| [`nested_loops.dc`](../../../examples/en/06-loops/nested_loops.dc) | nesting, `while` + `for` |
| [`arrays.dc`](../../../examples/en/03-data-types/arrays.dc) | `for i in range(…)` + `push` |

More: [1 — Examples / 06-loops](../1-examples/06-loops.md)

## Next

- [if / else](./04-if-else.md) — conditions inside loops
- [functions](./06-functions.md)
- [Data Types](../2-language/data-types/README.md)
