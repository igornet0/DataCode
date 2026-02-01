# 📚 DataCode Documentation in English

Welcome to the complete documentation of the **DataCode** programming language in English!

## 🎯 About DataCode

**DataCode** is a simple interactive programming language designed for fast data processing and easy learning. It features intuitive syntax, powerful array support, built-in functions, and user-defined functions with local scope.

## 📖 Documentation Contents

### 1. [Built-in Functions](./builtin_functions.md)

Complete description of all DataCode built-in functions, organized by category:

- **Utilities** - `print()`, `len()`, `range()`
- **Type conversion** - `int()`, `float()`, `bool()`, `str()`, `array()`, `date()`, `money()`
- **Type operations** - `typeof()`, `isinstance()`
- **Path operations** - `path()`, `path_name()`, `path_parent()`, `path_exists()`, `path_is_file()`, `path_is_dir()`, `path_extension()`, `path_stem()`, `path_len()`
- **Mathematical functions** - `abs()`, `sqrt()`, `pow()`, `min()`, `max()`, `round()`
- **String functions** - `upper()`, `lower()`, `trim()`, `split()`, `join()`, `contains()`
- **Array functions** - `push()`, `pop()`, `unique()`, `reverse()`, `sort()`, `sum()`, `average()`, `count()`
- **Table functions** - `table()`, `read_file()`, `table_info()`, `table_head()`, `table_tail()`, `table_select()`, `table_sort()`, `table_where()`, `show_table()`

**📚 Usage examples:**
- Basic functions: [`examples/en/01-basics/`](../../examples/en/01-basics/)
- Advanced functions: [`examples/en/05-functions/`](../../examples/en/05-functions/)
- Data work: [`examples/en/09-data-model-creation/`](../../examples/en/09-data-model-creation/)

---

### 2. [Data Types](./data_types.md)

Detailed description of all data types supported by the language:

- **Basic types**: Integer, Float, String, Bool, Null
- **Composite types**: Array, Object
- **Special table types**: Table, TableColumn, TableIndexer, LazyTable
- **File operation types**: Path, PathPattern
- **Special types**: Date, Currency, Mixed

**📚 Usage examples:**
- Working with types: [`examples/en/03-data-types/`](../../examples/en/03-data-types/)
- Basic types: [`examples/en/01-basics/`](../../examples/en/01-basics/)

---

### 3. [User-Defined Functions with Type Annotations](./user_functions.md)

Complete guide to creating functions with type annotations:

- **Type annotations** - Parameter and return type annotations
- **Union types** - Multiple types with `str | int` syntax
- **Type checking** - Runtime type validation and error handling
- **Partial typing** - Mixing typed and untyped parameters
- **Default values** - Combining types with default parameters
- **Supported types** - All available types for annotations

**📚 Usage examples:**
- Typed functions: [`examples/en/05-functions/typed_functions.dc`](../../examples/en/05-functions/typed_functions.dc)
- All function examples: [`examples/en/05-functions/`](../../examples/en/05-functions/)

---

### 4. [Working with Classes](./classes.md)

Declaration, fields, constructors, methods, inheritance, this/super, and visibility (private, protected, public):

- **Class declaration** - `cls Name { }` and `cls Child(Parent) { }`
- **Visibility** - `private:`, `protected:`, `public:` sections
- **Constructor** - `new ClassName(params) { body }`, instance creation `ClassName(args)`
- **Inheritance** - `super(args)` in constructor, `super.method(args)` in methods

**📚 Usage examples:**
- Classes, constructors, fields, methods (EN): [`examples/en/01-basics/classes.dc`](../../examples/en/01-basics/classes.dc)
- Inheritance, super, visibility (EN): [`examples/en/01-basics/inheritance.dc`](../../examples/en/01-basics/inheritance.dc)
- Classes (RU): [`examples/ru/01-основы/классы.dc`](../../examples/ru/01-основы/классы.dc)
- Inheritance (RU): [`examples/ru/01-основы/наследование.dc`](../../examples/ru/01-основы/наследование.dc)

---

### 5. [Working with Tables](./table_create_function.md)

Creating and working with tables in DataCode:

- Functions `table()` and `table_create()`
- Table operations (filtering, sorting, selection)
- File system integration (CSV, XLSX)
- Automatic column typing

**📚 Usage examples:**
- Data model creation: [`examples/en/09-data-model-creation/`](../../examples/en/09-data-model-creation/)
- File operations: [`examples/en/01-basics/`](../../examples/en/01-basics/)

---

### 6. [WebSocket Server](./websocket_server.md)

Remote execution of DataCode code via WebSocket:

- Server startup and configuration
- Message exchange protocol
- Connecting to SMB share
- Usage examples (JavaScript, Python)
- Virtual environment mode

**📚 Usage examples:**
- WebSocket clients: [`examples/en/08-websocket/`](../../examples/en/08-websocket/)

---

### 7. [JOIN Specification](./join_specification.md)

Table join operations:

- JOIN types (inner, left, right, full, cross, semi, anti)
- JOIN by arbitrary conditions
- Temporal JOIN (ASOF)
- Index JOIN
- Implementation algorithms

**📚 Usage examples:**
- Table joins: [`examples/en/09-data-model-creation/`](../../examples/en/09-data-model-creation/)

---

### 8. [ML Module](./ml/ml_module.md)

Machine learning module for DataCode:

- **Tensor operations** (12 functions) - working with tensors, operations on them
- **Graph operations** (9 functions) - computation graph for automatic differentiation
- **Linear Regression** (4 functions) - creating and training linear regression models
- **Optimizers** (5 functions) - optimizers (SGD, Adam, etc.)
- **Loss functions** (9 functions) - loss functions (MSE, Cross Entropy, MAE, etc.)
- **Dataset functions** (5 functions) - working with datasets, loading MNIST
- **Layer functions** (4 functions) - creating neural network layers
- **Neural Network functions** (20 functions) - creating, training and saving neural networks
- **Object methods** - methods of Tensor, NeuralNetwork, Dataset and other objects

**📚 Usage examples:**
- MNIST MLP: [`examples/en/11-mnist-mlp/`](../../examples/en/11-mnist-mlp/)
- Training process: [Neural Network Training Flow](./ml/training_flow.md)
- Model save format: [Model Save Format](./ml/model_save_format.md)

---

### 9. [Plot Module - Charts and Visualization](./plot/README.md)

Complete description of the `plot` module for working with images and creating charts:

- **Working with images** - loading, displaying, creating windows
- **Creating charts** - bar charts, line charts, pie charts, heatmaps
- **Subplots** - creating a grid of plots for simultaneous display
- **Axis configuration** - axis labels, titles, visibility control
- **Data types** - Image, Window, Figure, Axis

**📚 Usage examples:**
- Charts and visualization: [`examples/en/10-plot/`](../../examples/en/10-plot/)

---

### 10. [settings_env Module](./settings_env/README.md)

Loading environment variables from .env files, type coercion, prefix filtering, configuration, and Settings subclasses with Field descriptors:

- **Loading .env** - load_env, Settings, path resolution
- **Configuration** - Settings.config (env_prefix, extra, case_sensitive, etc.)
- **Field descriptors** - default, default_factory, required, min_length, max_length, alias, regex
- **Settings subclasses** - model_config, nested configs via default_factory

**📚 Usage examples:**
- settings_env module: [`examples/en/12-settings-env/`](../../examples/en/12-settings-env/), [`examples/ru/12-settings_env/`](../../examples/ru/12-settings_env/)

---

### 11. [uuid Module](./uuid/README.md)

UUID generation: v4, v7, deterministic v3/v5, string parsing, byte representation, metadata:

- **Generation** - v4, v7, new (alias v7), random (alias v4)
- **Parsing and strings** - parse, to_string
- **Bytes** - to_bytes, from_bytes
- **Deterministic UUIDs** - v3, v5, namespace DNS, URL, OID
- **Metadata** - version, variant, timestamp

**📚 Usage examples:**
- uuid module: [`examples/en/13-uuid/`](../../examples/en/13-uuid/), [`examples/ru/13-uuid/`](../../examples/ru/13-uuid/)

---

### 12. [Text Rendering in Plot Module](./text_rendering.md)

Detailed description of the text rendering system in the plot module:

- **System architecture** - components and data flow
- **Rendering process** - from font initialization to bitmap rendering
- **Fontdue coordinate model** - baseline-oriented model
- **Glyph caching** - performance optimization via FontAtlas
- **Alpha blending** - text edge smoothing
- **Implementation details** - DPI-aware rendering, monospace vs proportional fonts
- **Common issues and solutions** - typical problems and solutions

**📚 Technical documentation:**
- Complete process description: [Text Rendering](./text_rendering.md)
- Problem analysis: [`TEXT_RENDERING_ANALYSIS.md`](../../TEXT_RENDERING_ANALYSIS.md)

---

## 🚀 Quick Start

If you're just starting to learn DataCode, we recommend the following order:

### 1. Language Basics
Start with basic concepts (including classes):
- **Documentation:** [`examples/en/01-basics/README.md`](../../examples/en/01-basics/README.md)
- **Examples:** [`examples/en/01-basics/`](../../examples/en/01-basics/)
- **Classes:** [Working with Classes](./classes.md)

### 2. Syntax
Study syntactic constructs:
- **Documentation:** [`examples/en/02-syntax/README.md`](../../examples/en/02-syntax/README.md)
- **Examples:** [`examples/en/02-syntax/`](../../examples/en/02-syntax/)

### 3. Data Types
Get familiar with the type system:
- **Documentation:** [`examples/en/03-data-types/`](../../examples/en/03-data-types/)
- **Examples:** [`examples/en/03-data-types/`](../../examples/en/03-data-types/)

### 4. Functions
Master creating and using functions:
- **Documentation:** [`examples/en/05-functions/README.md`](../../examples/en/05-functions/README.md)
- **Type Annotations:** [`docs/en/user_functions.md`](./user_functions.md) - User-defined functions with type annotations
- **Examples:** [`examples/en/05-functions/`](../../examples/en/05-functions/)

### 5. Loops
Study loops and iterations:
- **Documentation:** [`examples/en/07-loops/README.md`](../../examples/en/07-loops/README.md)
- **Examples:** [`examples/en/07-loops/`](../../examples/en/07-loops/)

### 6. Advanced Features
Study advanced techniques:
- **Documentation:** [`examples/en/04-advanced/README.md`](../../examples/en/04-advanced/README.md)
- **Examples:** [`examples/en/04-advanced/`](../../examples/en/04-advanced/)

### 7. Comprehensive Examples
View complete demonstrations:
- **Documentation:** [`examples/en/06-demonstrations/README.md`](../../examples/en/06-demonstrations/README.md)
- **Examples:** [`examples/en/06-demonstrations/`](../../examples/en/06-demonstrations/)

---

## 📚 Complete Example Collection

All examples are organized in the [`examples/en/`](../../examples/en/) folder:

- **[01-basics](../../examples/en/01-basics/)** - Basic language concepts (including classes)
- **[02-syntax](../../examples/en/02-syntax/)** - Syntactic constructs
- **[03-data-types](../../examples/en/03-data-types/)** - Working with data types
- **[04-advanced](../../examples/en/04-advanced/)** - Advanced features
- **[05-functions](../../examples/en/05-functions/)** - User-defined functions
- **[06-demonstrations](../../examples/en/06-demonstrations/)** - Comprehensive examples
- **[07-loops](../../examples/en/07-loops/)** - Loops and iterations
- **[08-websocket](../../examples/en/08-websocket/)** - Working with WebSocket
- **[09-data-model-creation](../../examples/en/09-data-model-creation/)** - Data model creation
- **[10-plot](../../examples/en/10-plot/)** - Charts and visualization
- **[11-mnist-mlp](../../examples/en/11-mnist-mlp/)** - MNIST MLP example
- **[12-settings-env](../../examples/en/12-settings-env/)** - settings_env module
- **[13-uuid](../../examples/en/13-uuid/)** - uuid module

**Complete example documentation:** [`examples/en/README.md`](../../examples/en/README.md)

---

## 🔗 Useful Links

- **[Main Documentation](../README.md)** - Navigation for all documentation
- **[Main Project README](../../README.md)** - General information about DataCode
- **[Installation](../../INSTALL.md)** - Installation instructions
- **[License](../../LICENSE)** - Project license

---

## 📝 Documentation Structure

```
docs/
├── README.md                    # Main navigation
└── en/                          # English documentation
    ├── README.md                # This file
    ├── builtin_functions.md     # Built-in functions
    ├── data_types.md            # Data types
    ├── user_functions.md        # User-defined functions with type annotations
    ├── classes.md               # Working with classes
    ├── table_create_function.md # Working with tables
    ├── websocket_server.md      # WebSocket server
    ├── join_specification.md    # JOIN specification
    ├── ml_module.md             # ML module (machine learning)
    ├── text_rendering.md        # Text rendering in plot module
    ├── plot/                    # Plot module documentation
    │   └── README.md            # Plot module - charts and visualization
    ├── settings_env/            # settings_env module documentation
    │   └── README.md            # settings_env module - environment variables
    ├── uuid/                    # uuid module documentation
    │   └── README.md            # uuid module - UUID generation
    └── ml/                      # ML module documentation
        ├── training_flow.md     # Neural network training flow
        └── model_save_format.md # Model save format
```

---

## 💡 Tips for Using Documentation

1. **Start with examples** - Practical examples help understand concepts faster
2. **Use search** - Most editors allow searching documentation (Ctrl+F / Cmd+F)
3. **Follow links** - Documentation contains cross-references for easy navigation
4. **Experiment** - Modify examples and see what happens

---

**Note:** This documentation is created based on analysis of DataCode source code and usage examples. All functions are tested and work in the current version of the language.

**Happy learning DataCode!** 🧠✨

