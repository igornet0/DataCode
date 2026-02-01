# 🚀 DataCode Basics

This section contains basic examples for learning the fundamentals of the DataCode programming language.

## 📋 Contents

### 1. `hello.dc` - Simple Example
**Description**: Demonstrates basic text output.

**What you'll learn**:
- The `print()` function
- String literals

**Run**:
```bash
datacode examples/en/01-basics/hello.dc
```

### 2. `variables.dc` - Variables
**Description**: Demonstrates variable declaration and usage.

**What you'll learn**:
- Variable declaration with `let`
- Value assignment
- Using variables in expressions

**Run**:
```bash
datacode examples/en/01-basics/variables.dc
```

### 3. `arithmetic.dc` - Arithmetic Operations
**Description**: Demonstrates mathematical operations.

**What you'll learn**:
- Arithmetic operators: `+`, `-`, `*`, `/`
- Operator precedence
- Unary operators

**Run**:
```bash
datacode examples/en/01-basics/arithmetic.dc
```

### 4. `strings.dc` - Working with Strings
**Description**: Demonstrates string operations.

**What you'll learn**:
- String literals
- String concatenation (`+`)
- The `len()` function for getting string length

**Run**:
```bash
datacode examples/en/01-basics/strings.dc
```

### 5. `global_local.dc` - Global and Local Variables
**Description**: Demonstrates global and local variable scope.

**What you'll learn**:
- Global variables with `global`
- Local variables with `let`
- Variable scope in functions

**Run**:
```bash
datacode examples/en/01-basics/global_local.dc
```

### 6. `classes.dc` - Working with Classes
**Description**: Demonstrates class declaration, constructors, fields and methods.

**What you'll learn**:
- Simple class with public fields and constructor (`cls Point`, `new Point`)
- Class with private fields and methods (`Counter`, `inc()`, `get()`)
- Class with method that takes arguments (`Calculator`, `add(n)`)

**Run**:
```bash
datacode examples/en/01-basics/classes.dc
```

### 7. `inheritance.dc` - Class Inheritance
**Description**: Demonstrates child classes, `super`, and access to inherited/protected fields.

**What you'll learn**:
- Parent class with public and protected fields
- Child class with `super()` in constructor
- Access via methods vs direct access; `ProtectError` for protected fields

**Run**:
```bash
datacode examples/en/01-basics/inheritance.dc
```

## 🎯 Concepts Covered

### Variables
- **Declaration**: `let name = value`
- **Data types**: numbers, strings, boolean values
- **Automatic type inference**

### Operators
- **Arithmetic**: `+`, `-`, `*`, `/`
- **String**: concatenation with `+`
- **Unary**: `-` (negation)

### Built-in Functions
- **`print(...)`**: output data to console
- **`len(str)`**: get string length

## ⚠️ Important Points

1. **Variable declaration**: Use `let` to declare variables
2. **Data types**: DataCode automatically infers types
3. **Strings**: Use double quotes `"text"`
4. **Comments**: Start with `#` symbol

## 🔗 Navigation

### Next Steps
After learning the basics, move on to:
- **[02-syntax](../02-syntax/)** - conditional constructs and expressions
- **[05-functions](../05-functions/)** - user-defined functions
- **[07-loops](../07-loops/)** - while and for loops

---

**Happy learning DataCode!** 🧠✨

