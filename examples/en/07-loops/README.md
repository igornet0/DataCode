# 🔄 Loops in DataCode

This section contains examples of using `while` and `for` loops.

## 📋 Contents

### 1. `while_loops.dc` - While Loops
**Description**: Demonstrates using the `while` loop.

**What you'll learn**:
- `while` loop syntax
- Loop continuation conditions
- Changing variables in loops
- Loop exit conditions

**Run**:
```bash
datacode examples/en/07-loops/while_loops.dc
```

### 2. `for_loops.dc` - For Loops
**Description**: Demonstrates using the `for` loop.

**What you'll learn**:
- `for` loop syntax: `for i in array { ... }`
- Iterating over arrays
- Using for loops in functions

**Run**:
```bash
datacode examples/en/07-loops/for_loops.dc
```

### 3. `nested_loops.dc` - Nested Loops
**Description**: Demonstrates nested loops and their combinations.

**What you'll learn**:
- Nested `while` loops
- Nested `for` loops
- Combinations of loops and conditions
- Practical examples (multiplication table)

**Run**:
```bash
datacode examples/en/07-loops/nested_loops.dc
```

## 🎯 Concepts Covered

### While Loop
```dc
while condition {
    // code that executes while condition is true
}
```

### For Loop
```dc
for i in iter {
    // loop body
}
```

**Examples**:
- `for i in range(1, 11) { ... }`
- `for i in [1, 2, 3] { ... }`
- `for item in array { ... }`

### Important Points
1. **Exit condition**: Make sure the loop condition will eventually become false
2. **Changing variables**: Don't forget to change variables in the loop
3. **Infinite loops**: Be careful with conditions that are always true

## 🔗 Navigation

### Next Steps
After learning loops, move on to:
- **[04-advanced](../04-advanced/)** - advanced programming techniques
- **[06-demonstrations](../06-demonstrations/)** - comprehensive examples

---

**Learn loops!** 🔄

