# Working with Classes

This document describes how to define and use classes in DataCode: declaration, fields, constructors, methods, inheritance, and visibility (private, protected, public).

**Usage examples:**
- Classes, constructors, fields, methods: [`examples/en/01-basics/classes.dc`](../../../examples/en/01-basics/classes.dc)
- Inheritance, super, visibility: [`examples/en/01-basics/inheritance.dc`](../../../examples/en/01-basics/inheritance.dc)
- Special methods `@add`, `@string`, …: [`examples/en/02-syntax/special-methods.dc`](../../../examples/en/02-syntax/special-methods.dc)

---

## Contents

1. [Class declaration](#class-declaration)
2. [Abstract classes](#abstract-classes)
3. [Visibility sections](#visibility-sections)
4. [Fields](#fields)
5. [Class-level variables](#class-level-variables)
6. [Constructor](#constructor)
7. [Inheritance](#inheritance)
8. [Parameter type annotations](#parameter-type-annotations)
9. [Methods](#methods)
10. [The @class parameter](#the-class-parameter)
11. [Field access](#field-access)
12. [Special methods (@-methods)](#special-methods--methods)

---

## Class declaration

A class is declared with the `cls` keyword. The class body is enclosed in curly braces.

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

The parent class name in parentheses must refer to a previously defined class.

---

## Abstract classes

A class can be marked abstract with the `@Abstract` attribute immediately before `cls`. An abstract class cannot be instantiated directly; calling it (for example, `AbstractBase()`) raises a runtime error: `Cannot instantiate abstract class 'AbstractBase'`. Subclasses of an abstract class are ordinary classes and can be instantiated. Constructors and `super(...)` in subclasses work as usual.

**Syntax:**

```datacode
@Abstract
cls AbstractBase {
    public:
        name: str
    new AbstractBase(name) {
        this.name = name
    }
}

cls Concrete(AbstractBase) {
    new Concrete(name) {
        super(name)
    }
}

c = Concrete("ok")   # allowed
# b = AbstractBase("x")  # runtime error: Cannot instantiate abstract class 'AbstractBase'

```

---

## Visibility sections

Class members are grouped by visibility: `private:`, `protected:`, and `public:`.

- **private** — accessible only inside the class that defines the member.
- **protected** — accessible in the defining class and in subclasses (not from outside).
- **public** — accessible from anywhere (instance fields and methods).

Sections are optional. Fields, methods, and class-level variables **without** a preceding section default to **public**. To hide a member, explicitly use `private:` or `protected:`. Sections can be repeated (for example, open `public:` again after a `private:` block).

```datacode
cls Example {
    name: str              # public (default)

    private:
    secret: int

    protected:
    internal: str

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

Fields can be declared **without a section** (then they are public) or in `private:`, `protected:`, `public:` sections.

```datacode
cls Point {
    x: int
    y: int
    new Point(x, y) {
        this.x = x
        this.y = y
    }
}

```

An explicit `public:` section is needed only if a `private:` / `protected:` block came before:

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

## Class-level variables

Class-level variables are assignments without a type annotation: `name = expression`. They are stored on the class object (for example, for configuration). They follow the same visibility sections as fields.

```datacode
cls Config {
    public:
        model_config = Settings.config(env_prefix="APP__")
}

```

---

## Constructor

A constructor is declared as `new ClassName(params)` with a body in curly braces.

**Syntax:**

```datacode
new ClassName(param1, param2, ...) {
    # body: assign this.field, call super(...) when inheriting, etc.
}

```

**Creating an instance:** call the class as a function with the same number of arguments. The runtime selects the constructor by argument count (for example, `ClassName::new_2` for two parameters).

If all constructor parameters have types (`new Foo(x: int)`), the internal constructor name includes a type suffix (`Foo::new_1_int`). When calling `Foo(42)` or `Foo(x)` from another file (`from module import Foo`), the compiler and import resolve the typed name automatically — you do not write `Foo::new_1_int` in code.

```datacode
p = Point(10, 20)
c = Counter("MyCounter")

```

**Delegating constructor:** to call another constructor overload in the same class, after the parameter list write `: this(...)`. The body may be empty or contain additional initialization that runs after the delegated constructor returns:

```datacode
new ClassName(a, b) : this(a + b) { }

new FenwickTree(arr: array[int]) : this(len(arr)) {
    for i in range(len(arr)) {
        this.add(i, arr[i])
    }
}
```

With typed constructor overloads (`new Foo(n: int)` and `new Foo(arr: array[int])`), `: this(...)` picks the matching overload by argument types (for example, `len(arr)` delegates to the constructor with an `int` parameter).

**Note:** for bitwise operations on integers (for example, Fenwick tree lowbit: `i & (-i)`) use the `&` operator, not logical `and`.

---

## Inheritance

To inherit from a class, declare `cls Child(Parent) { ... }`.

**Parent constructor:** in a subclass constructor you must call the parent constructor exactly once, as the first statement: `super(args)`.

```datacode
cls Child(Parent) {
    public:
    new Child(v) {
        super(v, 202)
    }
}

```

**Calling a parent method:** inside a method use `super.methodName(args)`. This is only allowed inside a class that has a parent.

```datacode
fn overridden() {
    return super.overridden() + 1
}

```

Private members of the parent are not accessible in the child. Protected members are accessible in the subclass (for example, in methods); the runtime forbids reading and writing protected fields from outside the class hierarchy.

---

## Parameter type annotations

In a function or method signature you can specify a class name as a parameter type: `fn merge(a: HashSet, b: HashSet) { ... }`. At runtime the VM checks arguments against instance metadata:

- **`object` / `dict`** — any dictionary-like object, including a user class instance.
- **Class name** (`HashSet`, `MyClass`, …) — the value must be `Value::Object` with `__class_name` matching the name, or with `__superclass` equal to that name (direct parent). Names are case-sensitive.
- Built-in types (`int`, `str`, `array`, …) — a fixed language set; they do not overlap with user class names.

`typeof(instance)` for a class instance returns the class name (for example `"HashSet"`), not `"object"`. `isinstance(x, "MyClass")` and `isinstance(x, MyClass)` use the same matching logic.

**Limitation:** checking along an ancestor chain deeper than one level (`GrandChild` → parameter `Base` when the instance only has `__superclass = "Child"`) is not performed in the current version — hierarchy traversal via class objects in globals is planned.

---

## Methods

Methods are declared as `fn methodName(params) { body }`. Optionally specify a return type: `fn methodName(params) -> type { body }`. The first implicit parameter is `this` (the current instance).

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
calc = Calculator(100)
print(calc.add(25))  # 125

```

---

## The @class parameter

A method can declare an optional first parameter `@class` to receive the class object (meta-information about the class). It must be the first parameter in the list. The VM automatically supplies the class object when the method is called; calling code does not pass it.

**Syntax:** `fn methodName(@class, otherParams) { body }`

**Example:**

```datacode
cls User {
    public:
        name: str
    new User(name) {
        this.name = name
    }
    fn info(@class) {
        print(@class.name)           # "User"
        print(@class.full_name)      # "User"
        print(@class.method_names)   # ["info", ...]
    }
}

u = User("Alice")
u.info()   # @class is injected; prints class name and method list

```

**@class object API (class object properties):**

- **Identity:** `@class.name` (short name), `@class.full_name` (currently same as name)
- **Hierarchy:** `@class.parent` (superclass name or null)
- **Modifiers:** `@class.is_abstract` (true if the class is abstract)
- **Structure:** `@class.method_names` (array of method names)

Use `@class` for ORM registration, serialization, DI, or any logic that needs the class at runtime without changing call sites.

---

## Field access

- **Inside the class:** use `this.fieldName` to read and assign instance fields in constructors and methods.
- **Outside:** use `object.fieldName` for public fields. Access to private or protected fields from outside the class (or from the wrong class in the hierarchy) raises a runtime error (for example, ProtectError).

```datacode
# In constructor or method
this.x = 10
v = this.x

# Outside (public only)
p = Point(5, 10)
print(p.x)
print(p.y)

```

---

## Special methods (@-methods)

Methods like `fn @add(...)`, `fn @string()`, etc. connect class instances to operators (`+`, `==`, `in`, …), built-in functions (`str`, `len`, `print`), and the `for-in` protocol. You cannot call them directly (`obj.@string()`).

Full method table, signatures, and a `Vector` example: **[Special Methods](../2-language/special-methods.md)**.

---

## See also

**Usage examples:**
- Classes, constructors, fields, methods: [`examples/en/01-basics/classes.dc`](../../../examples/en/01-basics/classes.dc)
- Inheritance, super, visibility: [`examples/en/01-basics/inheritance.dc`](../../../examples/en/01-basics/inheritance.dc)
- Special methods `@add`, `@string`, …: [`examples/en/02-syntax/special-methods.dc`](../../../examples/en/02-syntax/special-methods.dc)
