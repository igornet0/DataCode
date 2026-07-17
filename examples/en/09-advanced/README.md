# 🎯 Advanced DataCode Features

This section contains advanced language usage examples.

## 📋 Contents

### 1. `complex.dc` - Complex example
**Description**: Demonstrates combining various language features.

**What you'll learn**:
- Combining functions, loops, and conditions
- Recursive functions
- Complex computations
- Practical usage examples

**Run**:
```bash
datacode examples/en/09-advanced/complex.dc
```

### 2. `scope_demo.dc` - Variable scope
**Description**: Demonstrates global and local variables.

**What you'll learn**:
- Global variables
- Local variables in functions
- Variables in loops and blocks
- Scope in recursive functions

**Run**:
```bash
datacode examples/en/09-advanced/scope_demo.dc
```

### 3. `error_handling.dc` - Error handling
**Description**: Demonstrates error handling with try/catch/throw/finally blocks.

**What you'll learn**:
- Basic try/catch blocks
- try/catch with and without error variable
- try/finally blocks
- Combined try/catch/finally
- Using throw for custom exceptions
- Nested try/catch blocks
- Error handling in functions
- File operations with error handling
- Error propagation

**Run**:
```bash
datacode examples/en/09-advanced/error_handling.dc
```

## Subdirectories

| Folder | Topics |
|--------|--------|
| `dp/` | Dynamic programming |
| `algorithms/` | Geometry, greedy, arrays, math, search, practical, sorting, strings |
| `data-structures/` | Graphs, trees, queues, caches, hash structures |

Run all examples:

```bash
./run_all.sh
```

## 🎯 Concepts Covered

### Variable scope
- **Global variables**: Declared at top level, accessible everywhere
- **Local variables**: Declared inside functions, accessible only inside the function
- **Function parameters**: Local to the function
- **Block variables**: Accessible only inside the block (if, while, for)

### Error handling
- **try/catch**: Catch and handle exceptions
- **try/finally**: Always run cleanup code
- **throw**: Raise custom exceptions
- **Nested blocks**: Error handling at different levels
- **Propagation**: Errors bubble up through function calls

### Function composition
Functions can be combined for more complex solutions:
- Functions can call other functions
- Functions can be arguments to other functions
- Recursive functions can use local variables

## 🔗 Navigation

### Next steps
After advanced features, continue with:
- **[05-demonstrations](../05-demonstrations/)** - comprehensive showcases of all features

---

**Keep learning!** 🚀
