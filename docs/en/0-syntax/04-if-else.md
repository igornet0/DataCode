# Conditions: if / else

Control execution flow by condition. A condition can be an explicit comparison (`x > 0`) or any value — in that case **truthiness** is used; see [below](#truthiness-in-conditions).

## Basic syntax

### `if` without `else`

```datacode
if x > 5 {
    print("x is greater than 5")
}

```

The block runs only when the condition is true. If the condition is false, execution simply continues.

### `if` / `else`

```datacode
if x > y {
    print("x is greater than y")
} else {
    print("x is not greater than y")
}

```

### `else if` chain

```datacode
if a > 0 {
    print("positive")
} else if a < 0 {
    print("negative")
} else {
    print("zero")
}

```

`else if` is an `else` immediately followed by a new `if`. There can be any number of branches; the final `else` is optional.

## One-line `if` (with `:`)

For short branches you can write a single statement after a colon — without curly braces:

```datacode
if x > 0: return 1
if x == 0: x = 10
if !(0 <= nr < rows and 0 <= nc < cols): continue

```

Convenient in functions and loops for early exit (`return`, `continue`, `break`) or simple assignment. A full block is still written with `{ ... }`.

## Truthiness in conditions

In `if`, `while`, and ternary expressions, a value is treated as "true" or "false":

| Falsy | Truthy |
|-------|--------|
| `null`, `false` | `true` |
| number `0` | any non-zero number |
| empty string `""` | non-empty string |
| empty array `[]`, empty object | non-empty collections |
| empty table | runtime objects (windows, DB engines, etc.) |

```datacode
items = []
if items {
    print("has elements")   # does not run
} else {
    print("list is empty")
}

if flag {                    # flag: bool
    print("enabled")
}

```

More on types: [Data Types](../2-language/data-types/README.md).

## Comparison operators

| Operator | Meaning |
|----------|---------|
| `==` | equal |
| `!=` | not equal |
| `>`, `<` | greater than, less than |
| `>=`, `<=` | greater or equal, less or equal |

```datacode
if name1 == name2 {
    print("Names are the same")
}

if value != null {
    print(value)
}

```

### Chained comparisons

As in Python, you can write several comparisons in a row — they are combined with `and`:

```datacode
if 0 <= age and age < 18 {
    print("minor")
}

if 0 <= nr and nr < rows and 0 <= nc and nc < cols {
    // cell inside the grid
}

```

Shorthand for bounds: `0 <= x < n` is equivalent to `0 <= x and x < n`.

## Logical operators

```datacode
ok = true
not_ok = !ok

if x != null and x > 0 {
    print(x)
}

if blocked or timeout {
    print("abort")
}

if !(a < b and b < c) {
    print("chain broken")
}

```

| Operator | Meaning |
|----------|---------|
| `and` | logical AND (short-circuit) |
| `or` | logical OR (short-circuit) |
| `!` | negation |

For **bit masks** use bitwise operators `&`, `|`, `^`, `~`, `<<`, `>>` (see [bitwise-operators.md](./10-bitwise-operators.md)), not `and` / `or`.

Literals: `true`, `false`.

The **`in`** operator checks membership in a collection:

```datacode
if 5 in [1, 2, 3, 4, 5] {
    print("found")
}

```

The **`not in`** operator checks that an element is **not** in a collection:

```datacode
if 5 not in [1, 2, 3, 4, 5] {
    print("found")
}

```

## `if` as an expression

When you need to **produce a value** rather than run a block of statements, use `if` / `else` with curly braces in **expression position**:

```datacode
status = if got == expected { "OK" } else { "FAIL" }
cost = if a[i - 1] == b[j - 1] { 0 } else { 1 }
label = if n > 0 { "present" } else { "empty" }

print("Result: ${if x > 0 { "plus" } else { "minus" }}")

```

Both branches are required. Inside `{ ... }` there can be multiple lines; the branch value is the result of the **last** expression in the block.

In `return` and other expressions:

```datacode
fn sign(n) {
    return if n > 0 { 1 } else { if n < 0 { -1 } else { 0 } }
}

```

### Alternative: Python-style ternary

The form `value_if_true if condition else value_if_false` is also supported:

```datacode
x = 1 if true else 2
return dp[amount] if dp[amount] != INF else -1

```

`if` and `else` in this form must be on the **same line** as the expression (no implicit line break). For multi-line branches, prefer the block form `if ... { ... } else { ... }`.

### C-style ternary `? :`

The form `condition ? value_if_true : value_if_false` (as in C, JavaScript, Rust):

```datacode
x = true ? 1 : 2
to_bool = fn(s) => s == "yes" ? true : s == "no" ? false : bool(s)

```

| Python | C-style |
|--------|---------|
| `1 if cond else 2` | `cond ? 1 : 2` |
| `a if b else c if d else e` | `b ? a : d ? c : e` |

Properties:

- **Right-associative:** `a ? b : c ? d : e` → `a ? b : (c ? d : e)`
- **Precedence** below arithmetic: `1 + 2 ? 3 : 4` → `(1 + 2) ? 3 : 4`
- **Short-circuit:** the unchosen branch is not evaluated
- Line breaks are **allowed** (unlike the Python form with `if`/`else` on one line)
- Both forms can be combined in one expression

## Common patterns

### Nested conditions

```datacode
if age >= 18 {
    if age < 65 {
        print("Adult")
    } else {
        print("Retiree")
    }
} else {
    print("Minor")
}

```

### Check for `null` before use

```datacode
if env != null {
    print(env["HOST"])
}

```

### Early return from a function

```datacode
fn divide(a, b) {
    if b == 0: return null
    return a / b
}

```

### Condition inside a loop

```datacode
for i in items {
    if i == skip: continue
    if i < 0: break
    print(i)
}

```

## Examples

| File | Topic |
|------|-------|
| [`conditionals.dc`](../../../examples/en/02-syntax/conditionals.dc) | `if` / `else`, nesting, string comparison |
| [`booleans.dc`](../../../examples/en/02-syntax/booleans.dc) | `true` / `false`, `!`, null checks |
| [`expressions.dc`](../../../examples/en/02-syntax/expressions.dc) | compound expressions |
| [`tarjan_scc.dc`](../../../examples/en/09-advanced/data-structures/graphs/advanced/tarjan_scc.dc) | `if` as expression (`status = if ... { "OK" } else { "FAIL" }`) |

More: [1 — Examples / 02-syntax](../1-examples/02-syntax.md)

## Next

- [loops](./05-loops.md)
- [error handling](./12-error-handling.md) — `try` / `catch`
- [Data Types](../2-language/data-types/README.md)
