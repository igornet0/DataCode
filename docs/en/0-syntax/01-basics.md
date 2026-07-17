# Syntax Basics

The minimum set of constructs for your first DataCode programs.

## Output

```datacode
print("Hello, DataCode!")

```

## Variables

```datacode
x = 42
name = "DataCode"
flag = true

```

### Three ways to declare a variable

| Syntax | Where | Effect |
|--------|-------|--------|
| `x = …` | top level of a script | implicit **global** variable |
| `global x = …` | anywhere | explicit **global** variable |
| `let x = …` | top level | also **global** (same as `x = …`) |
| `let x = …` | inside a function | new **local** variable in that function |

At the top level of a script, `x = 42` and `let x = 42` behave the same: the variable goes into the shared global namespace.

### `global` — explicit global variable

```datacode
global counter = 0

fn inc() {
    global counter = counter + 1
}

```

`global counter = …` inside a function is needed when the variable is **not yet declared** at the top level, or when you explicitly want to write to a global (as in the increment example).

If a variable is already declared globally (`global x = …` or `let x = …` / `x = …` at the top level), assigning **`x = …` inside a function changes the global `x`**, rather than creating a local copy.

### `let` — local variable inside a function

The **`let`** keyword inside a function creates a **new local** variable that **shadows** a global with the same name. Changes to the local variable **do not affect** the global.

### Example: nested functions (522 vs 900)

A classic example from VM tests — two versions of the same code. The only difference is whether `let` is used for `y` inside the functions.

**With `let` — result `522`:**

```datacode
global x = 1
let y = 2
fn outer() {
    x = 10
    let y = 20          # local y in outer; global y stays 2
    fn inner() {
        x = 100         # global x
        let y = 200     # local y in inner
        return x + y    # 100 + 200 = 300
    }
    return inner() + x + y   # 300 + 100 + 20 = 420
}
outer() + x + y              # 420 + 100 + 2 = 522

```

**Without `let` (only `y = …`) — result `900`:**

```datacode
global x = 1
let y = 2
fn outer() {
    x = 10
    y = 20              # y is already global → y = 20 for everyone
    fn inner() {
        x = 100
        y = 200         # global y = 200
        return x + y    # 300
    }
    return inner() + x + y   # 300 + 100 + 200 = 600
}
outer() + x + y              # 600 + 100 + 200 = 900

```

Step by step for the **with `let`** variant:

| Step | What happens |
|------|--------------|
| `inner()` | `x = 100` (global), local `y = 200` → returns **300** |
| end of `outer()` | 300 + `x`(100) + local `y`(20) = **420** |
| top level | 420 + `x`(100) + global `y`(2) = **522** |

Step by step for the **without `let`** variant:

| Step | What happens |
|------|--------------|
| `inner()` | global `x = 100`, `y = 200` → **300** |
| end of `outer()` | 300 + 100 + 200 = **600** |
| top level | 600 + 100 + 200 = **900** |

**Rule:** inside a function, **`let y = …`** always creates a local `y`. Assigning **`y = …`** without `let` updates the **global** `y` if it already exists (declared via `global`, `let`, or `=` at the top level). If there is no global `y`, `y = …` creates a new local variable.

For **`x`** in both variants: `global x = 1` makes `x` global, so `x = 10` and `x = 100` inside functions change the same global variable.

### Quick cheat sheet

```datacode
global shared = 0     # explicit global

fn example() {
    let local = 1     # local, not visible outside
    shared = 2        # changes global shared (already declared)
    temp = 3          # new local (no global temp)
}

```

## Arithmetic and strings

```datacode
sum = 10 + 3 * 2
text = "Hello, " + name
length = len(text)

```

Operators: `+`, `-`, `*`, `/`, unary `-`.

## Comments

```datacode
# single-line comment

```

## Code blocks

Code is grouped with curly braces `{ ... }` — in conditions, loops, functions, and classes.

## Related sections

- [if / else](./04-if-else.md) — conditions
- [loops](./05-loops.md) — repetition
- [functions](./06-functions.md) — named blocks of code, nested `fn`
- [strings and interpolation](./02-strings-and-interpolation.md) — `${expr}` in strings

## Examples

| File | Topic |
|------|-------|
| [`hello.dc`](../../../examples/en/01-basics/hello.dc) | `print()` |
| [`variables.dc`](../../../examples/en/01-basics/variables.dc) | assignment |
| [`arithmetic.dc`](../../../examples/en/01-basics/arithmetic.dc) | operators |
| [`strings.dc`](../../../examples/en/01-basics/strings.dc) | strings, `len()` |
| [`global_local.dc`](../../../examples/en/01-basics/global_local.dc) | `global`, recursion and global `x` |

More: [1 — Examples / 01-basics](../1-examples/01-basics.md)
