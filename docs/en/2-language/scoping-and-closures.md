# Scoping and closures

In DataCode, closures work through **slot copying on call**: when a nested function or lambda is invoked, captured variable values are copied from the parent frame into the callee frame. Assignments inside the nested function **do not write back** to the parent frame.

To keep behavior predictable (especially for recursion and graph algorithms), the compiler enforces:

## Rules

1. **Reads are allowed.** A nested function or lambda may read variables from enclosing functions (parameters and locals assigned with `=`).

2. **Assigning to an outer variable is not allowed.** If a name already exists in the parent function's scope, you cannot assign to it from a nested function or lambda without creating a new local variable (shadowing) — see rule 3.

3. **Explicit state passing.** Pass updatable state through parameters and return updated values from functions (for example tuples like `[visited, stack]`).

Plain assignment (`x = …`) inside a function always applies **within that function**: either to an existing local/capture slot, or it creates a new implicit local. The outer frame is not modified.

## `for` loop variables

The **name in the header** `for x in iterable { … }` (including unpack patterns like `for a, b in pairs`) has **block** scope: it applies only inside that loop. After the loop, the name is not available to later statements in the same function.

Assignments **inside the loop body** (for example `elapsed = i`) still belong to the **function scope** and remain visible after the loop.

Because the pattern name does not leak into the outer function, a nested function **after** the loop may reuse the same name without `let`:

```dc
fn build(freq) {
    for ch in freq {
        heap = push(heap, ch)
    }
    fn walk(node) {
        ch = node[2]   // new local in walk, not the outer variable
        ...
    }
}
```

## Anti-pattern

Do not accumulate recursion state by assigning to outer locals from a nested function:

```dc
fn bad(graph) {
    visited = []
    fn dfs(node) {
        visited = push(visited, node)
        ...
    }
}
```

This fails at compile time with a message about assigning to an outer variable and a suggestion to pass state explicitly.

## Recommended pattern

As in other graph algorithm examples (Kosaraju, connected components): pass state through parameters and return updated values:

```dc
fn dfs(graph, node, visited, stack) {
    visited = push(visited, node)
    ...
    return [visited, stack]
}
```

## Note on `push`

`push(array, item)` returns an array value suitable for rebinding; relying on "shared mutation" through closures is unreliable under the current rules. Prefer returning updated arrays and state tuples from helper functions.

## Future

A dedicated keyword (for example `nonlocal`) for mutating outer bindings may come later; it is not implemented today.
