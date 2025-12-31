# 📚 DataCode Examples

Welcome to the DataCode programming language examples collection!

## 🎯 Example Structure

Examples are organized by topics, progressing from simple to complex:

### 🚀 [01-basics](01-basics/) - Language Basics
Basic examples to start learning:
- `hello.dc` - Simplest example
- `variables.dc` - Working with variables
- `arithmetic.dc` - Arithmetic operations
- `strings.dc` - Working with strings
- `global_local.dc` - Global and local variables

### 🔧 [02-syntax](02-syntax/) - Syntax Constructs
Language syntax examples:
- `conditionals.dc` - Conditional constructs (if/else)
- `expressions.dc` - Complex expressions
- `booleans.dc` - Boolean values and logical operations

### 📦 [03-data-types](03-data-types/) - Data Types
Working with different data types:
- `type_conversion_functions.dc` - Type conversion functions
- `type_conversion_guide.dc` - Type conversion guide
- `type_date.dc` - Working with dates
- `objects.dc` - Working with objects (dictionaries)

### 📦 [05-functions](05-functions/) - Functions
Examples of creating and using functions:
- `simple_functions.dc` - Simple functions
- `recursion.dc` - Recursive functions
- `nested_functions.dc` - Nested function calls

### 🔄 [07-loops](07-loops/) - Loops
Examples of using loops:
- `while_loops.dc` - While loops
- `for_loops.dc` - For loops
- `nested_loops.dc` - Nested loops

### 🎯 [04-advanced](04-advanced/) - Advanced Features
Advanced programming techniques:
- `complex.dc` - Complex example
- `scope_demo.dc` - Variable scope
- `error_handling.dc` - Error handling with try/catch/throw

### 🎪 [06-demonstrations](06-demonstrations/) - Demonstrations
Comprehensive demonstrations of all features:
- `showcase.dc` - Complete language demonstration

### 📊 [09-data-model-creation](09-data-model-creation/) - Data Model Creation
Building database models from CSV files:
- `01-file-operations.dc` - Working with files and directories
- `02-merge-tables.dc` - Merging multiple tables
- `03-create-relations.dc` - Creating relations between tables
- `04-load-quarterly-data.dc` - Loading quarterly aggregated data
- `05-table-joins.dc` - Table JOIN operations (inner, left, right, full, cross, semi, anti)
- `load_model_data.dc` - Complete example loading all data and creating SQLite database

### 🎨 [10-plot](10-plot/) - Plots and Visualization
Working with the `plot` module for loading and displaying images, as well as creating charts:
- `01-load-image.dc` - Loading an image from a file
- `02-show-image.dc` - Quick image viewing
- `03-create-window.dc` - Creating a window for drawing
- `04-draw-image.dc` - Drawing an image in a window
- `05-full-example.dc` - Comprehensive example of working with the plot module
- `06-subplots.dc` - Creating a subplots grid
- `07-bar-chart.dc` - Creating a bar chart
- `08-heatmap.dc` - Creating a heatmap
- `09-line-chart.dc` - Creating a line chart
- `10-pie-chart.dc` - Creating a pie chart

### 🧠 [11-mnist-mlp](11-mnist-mlp/) - MNIST MLP Example
Training a Multi-Layer Perceptron (MLP) on the MNIST dataset:
- `mnist_mlp.dc` - Complete MLP training example
- `mnist_model_demo.dc` - Model demonstration
- `mnist_mlp_sh.dc` - Shell script example

## 🚀 Quick Start

### Recommended Learning Order

1. **Start with basics**:
   ```bash
   cargo run examples/en/01-basics/hello.dc
   cargo run examples/en/01-basics/variables.dc
   ```

2. **Learn syntax**:
   ```bash
   cargo run examples/en/02-syntax/conditionals.dc
   cargo run examples/en/02-syntax/expressions.dc
   ```

3. **Understand data types**:
   ```bash
   cargo run examples/en/03-data-types/type_conversion_functions.dc
   cargo run examples/en/03-data-types/objects.dc
   ```

4. **Master functions**:
   ```bash
   cargo run examples/en/05-functions/simple_functions.dc
   cargo run examples/en/05-functions/recursion.dc
   ```

5. **Study loops**:
   ```bash
   cargo run examples/en/07-loops/while_loops.dc
   cargo run examples/en/07-loops/for_loops.dc
   ```

6. **Advanced techniques**:
   ```bash
   cargo run examples/en/04-advanced/complex.dc
   cargo run examples/en/04-advanced/error_handling.dc
   ```

7. **Data model creation**:
   ```bash
   cargo run examples/en/09-data-model-creation/05-table-joins.dc
   ```

8. **Plots and visualization**:
   ```bash
   cargo run examples/en/10-plot/01-load-image.dc
   cargo run examples/en/10-plot/07-bar-chart.dc
   cargo run examples/en/10-plot/09-line-chart.dc
   cargo run examples/en/10-plot/10-pie-chart.dc
   cargo run examples/en/10-plot/08-heatmap.dc
   ```

9. **MNIST MLP example**:
   ```bash
   cargo run examples/en/11-mnist-mlp/mnist_mlp.dc
   ```

10. **Complete demonstration**:
   ```bash
   cargo run examples/en/06-demonstrations/showcase.dc
   ```

## 📖 Documentation

Each section contains its own `README.md` with detailed descriptions of examples and concepts covered.

## 💡 Tips

- **Start simple**: Begin with the `01-basics` section
- **Experiment**: Modify examples and see what happens
- **Read comments**: Comments in examples explain the code
- **Use REPL**: Run `cargo run` for interactive mode

## 🔗 Useful Links

- **[Main Documentation](../../README.md)** - Project documentation
- **[Russian Examples](../ru/README.md)** - Russian language examples

---

**Happy learning DataCode!** 🧠✨

