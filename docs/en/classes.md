# Working with Classes

This document describes how to define and use classes in DataCode: declaration, fields, constructors, methods, inheritance, and visibility (private, protected, public).

**Usage examples:**
- Classes, constructors, fields, methods (EN): [`examples/en/01-basics/classes.dc`](../../examples/en/01-basics/classes.dc)
- Inheritance, super, visibility (EN): [`examples/en/01-basics/inheritance.dc`](../../examples/en/01-basics/inheritance.dc)
- Classes, constructors, fields, methods (RU): [`examples/ru/01-основы/классы.dc`](../../examples/ru/01-основы/классы.dc)
- Inheritance, super, visibility (RU): [`examples/ru/01-основы/наследование.dc`](../../examples/ru/01-основы/наследование.dc)

---

## Table of Contents

1. [Class Declaration](#class-declaration)
2. [Visibility Sections](#visibility-sections)
3. [Fields](#fields)
4. [Class-Level Variables](#class-level-variables)
5. [Constructor](#constructor)
6. [Inheritance](#inheritance)
7. [Methods](#methods)
8. [Field Access](#field-access)

---

## Class Declaration

Use the `cls` keyword to declare a class. The body is enclosed in braces.

**Syntax without inheritance:**

```datacode
cls ClassName {
    # fields, constructors, methods
}
```

**Syntax with inheritance:**

```datacode
cls ChildClass(ParentClass) {
    # fields, constructors, methods
}
```

The superclass name in parentheses must refer to a previously defined class.

---

## Visibility Sections

Class members are grouped by visibility: `private:`, `protected:`, and `public:`.

- **private** — accessible only within the class that defines the member.
- **protected** — accessible in the defining class and in subclasses (not from outside).
- **public** — accessible from anywhere (instance fields and methods).

Sections are optional. Members declared without a preceding section default to private. You can repeat sections (e.g. multiple `public:` blocks).

```datacode
cls Example {
    # default (private) section
    secret: int

    protected:
    internal: str

    public:
    name: str
    new Example(name) {
        this.name = name
    }
}
```

---

## Fields

Fields are declared with an optional type and optional default value.

**Forms:**

- `name: type` — field with type, no default
- `name: type = expression` — field with type and default (constant expressions are evaluated at compile time where possible)

Fields are listed under `private:`, `protected:`, or `public:`.

```datacode
cls Point {
    public:
        x: int
        y: int
    new Point(x, y) {
        this.x = x
        this.y = y
    }
}
```

```datacode
cls Counter {
    private:
        n: int = 0
    public:
        name: str
    new Counter(name) {
        this.name = name
    }
}
```

---

## Class-Level Variables

Class-level variables are assignments without a type annotation: `name = expression`. They are stored on the class object (e.g. for configuration). They follow the same visibility sections as fields.

```datacode
cls Config {
    public:
        model_config = Settings.config(env_prefix="APP__")
}
```

---

## Constructor

A constructor is declared with `new ClassName(params)` and a body in braces.

**Syntax:**

```datacode
new ClassName(param1, param2, ...) {
    # body: assign this.field, call super(...) if subclass, etc.
}
```

**Creating an instance:** call the class as a function with the same number of arguments. The runtime selects the constructor by arity (e.g. `ClassName::new_2` for two parameters).

```datacode
let p = Point(10, 20)
let c = Counter("MyCounter")
```

**Delegating constructor:** to call another constructor of the same class, use `: this(...)` after the parameter list and leave the body empty:

```datacode
new ClassName(a, b) : this(a + b) { }
```

---

## Inheritance

To inherit from another class, declare `cls Child(Parent) { ... }`.

**Superclass constructor:** in a subclass constructor, you must call the parent constructor exactly once, as the first statement, using `super(args)`.

```datacode
cls Child(Parent) {
    public:
    new Child(v) {
        super(v, 202)
    }
}
```

**Calling a parent method:** use `super.methodName(args)` inside a method. This is only allowed inside a class that has a superclass.

```datacode
fn overridden() {
    return super.overridden() + 1
}
```

Private members of the parent are not accessible in the child. Protected members are accessible in the child class (e.g. in methods); the VM enforces that protected fields are not read or written from outside the class hierarchy.

---

## Methods

Methods are declared with `fn methodName(params) { body }`. Optional return type: `fn methodName(params) -> type { body }`. The first implicit parameter is `this` (the current instance).

```datacode
cls Calculator {
    public:
        base: int
    new Calculator(base) {
        this.base = base
    }
    fn add(n) {
        return this.base + n
    }
}
```

**Calling a method:** `instance.methodName(args)`.

```datacode
let calc = Calculator(100)
print(calc.add(25))  # 125
```

---

## Field Access

- **Inside the class:** use `this.fieldName` to read or assign instance fields in constructors and methods.
- **Outside:** use `object.fieldName` for public fields. Access to private or protected fields from outside the class (or from the wrong class in the hierarchy) causes a runtime error (e.g. ProtectError).

```datacode
# In constructor or method
this.x = 10
let v = this.x

# From outside (public only)
let p = Point(5, 10)
print(p.x)
print(p.y)
```

---

## See Also

**Usage examples:**
- Classes, constructors, fields, methods (EN): [`examples/en/01-basics/classes.dc`](../../examples/en/01-basics/classes.dc)
- Inheritance, super, visibility (EN): [`examples/en/01-basics/inheritance.dc`](../../examples/en/01-basics/inheritance.dc)
- Classes, constructors, fields, methods (RU): [`examples/ru/01-основы/классы.dc`](../../examples/ru/01-основы/классы.dc)
- Inheritance, super, visibility (RU): [`examples/ru/01-основы/наследование.dc`](../../examples/ru/01-основы/наследование.dc)
