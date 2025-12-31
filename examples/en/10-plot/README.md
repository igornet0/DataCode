# 🎨 Plots and Visualization (plot module)

This section contains examples of working with the `plot` module for loading and displaying images, as well as creating various types of charts in DataCode.

## 📋 Contents

### 1. `01-load-image.dc` - Loading an Image
**Description**: Demonstrates basic image loading from a file.

**What you'll learn**:
- The `plot.image(path)` function for loading images
- Checking for successful loading
- Supported image formats

**Run**:
```bash
cargo run examples/en/10-plot/01-load-image.dc
```

### 2. `02-show-image.dc` - Quick Image Viewing
**Description**: Demonstrates using the convenient `plot.show()` function for automatic image display.

**What you'll learn**:
- The `plot.show(path)` function for quick viewing
- Automatic window creation with image dimensions
- Blocking wait for window closure

**Run**:
```bash
cargo run examples/en/10-plot/02-show-image.dc
```

### 3. `03-create-window.dc` - Creating a Window
**Description**: Demonstrates creating a window with specified parameters.

**What you'll learn**:
- The `plot.window(width, height, title)` function for creating a window
- Window parameters (width, height, title)
- Checking for successful window creation

**Run**:
```bash
cargo run examples/en/10-plot/03-create-window.dc
```

### 4. `04-draw-image.dc` - Drawing in a Window
**Description**: Demonstrates the full cycle of working with images: loading, creating a window, rendering.

**What you'll learn**:
- Sequence of operations for displaying an image
- The `plot.draw(window, image)` function for rendering
- The `plot.wait(window)` function for waiting for window closure

**Run**:
```bash
cargo run examples/en/10-plot/04-draw-image.dc
```

### 5. `05-full-example.dc` - Comprehensive Example
**Description**: A complete example demonstrating all capabilities of the `plot` module.

**What you'll learn**:
- All functions of the `plot` module in one example
- Combining various operations
- Error handling when working with images

**Run**:
```bash
cargo run examples/en/10-plot/05-full-example.dc
```

### 6. `06-subplots.dc` - Creating a Subplots Grid
**Description**: Demonstrates using `plot.subplots()` to create a grid of plots and display multiple images simultaneously.

**What you'll learn**:
- The `plot.subplots(rows, cols, figsize=(width, height))` function for creating a grid
- Access to axes via `fig.axes[row][col]`
- The `axis.imshow(image, cmap='gray')` method for displaying images in axes
- The `axis.set_title(title)` method for setting titles
- The `axis.axis('off')` method for hiding coordinate axes
- The `plot.tight_layout(fig)` function for automatic padding adjustment
- Displaying a figure via `plot.show(fig, title="...")`

**Run**:
```bash
cargo run examples/en/10-plot/06-subplots.dc
```

### 7. `07-bar-chart.dc` - Bar Chart
**Description**: Demonstrates creating a bar chart using categorical data.

**What you'll learn**:
- The `plot.bar(x, y)` function for creating bar charts
- The `plot.xlabel()` and `plot.ylabel()` functions for setting axis labels
- Working with categorical data (strings) and numeric values
- The `plot.show(title="...")` function for displaying a chart with a title

**Run**:
```bash
cargo run examples/en/10-plot/07-bar-chart.dc
```

### 8. `08-heatmap.dc` - Heatmap
**Description**: Demonstrates creating a heatmap for visualizing matrix data.

**What you'll learn**:
- The `plot.heatmap(data)` function for creating heatmaps
- The `min` and `max` parameters for setting value ranges
- The `palette` parameter for choosing color schemes
- Visualizing two-dimensional data arrays

**Run**:
```bash
cargo run examples/en/10-plot/08-heatmap.dc
```

### 9. `09-line-chart.dc` - Line Chart
**Description**: Demonstrates creating line charts with multiple data series.

**What you'll learn**:
- The `plot.line(x, y, connected, color)` function for creating line charts
- The `connected` parameter for connecting points with a line
- The `color` parameter for setting line color (named colors and hex codes)
- Plotting multiple charts on the same axis
- Supported colors: blue, green, red, black, white, as well as hex codes

**Run**:
```bash
cargo run examples/en/10-plot/09-line-chart.dc
```

### 10. `10-pie-chart.dc` - Pie Chart
**Description**: Demonstrates creating a pie chart for visualizing proportions.

**What you'll learn**:
- The `plot.pie(x, y)` function for creating pie charts
- Working with categorical labels and numeric values
- Visualizing data proportions and shares

**Run**:
```bash
cargo run examples/en/10-plot/10-pie-chart.dc
```

## 🎯 Concepts Covered

### The plot Module
The `plot` module provides functionality for working with images, displaying them in windows, as well as creating various types of charts and diagrams. The module is available as a global variable `plot`.

### plot Module Functions

#### `plot.image(path) -> Image | null`
Loads an image from a file at the specified path.

**Parameters**:
- `path` (string): path to the image file

**Returns**:
- `Image`: image object on successful loading
- `null`: if loading failed

**Example**:
```datacode
let img = plot.image("path/to/image.png")
if img != null {
    print("Image loaded!")
}
```

#### `plot.window(width, height, title) -> Window | null`
Creates a new window for displaying graphics.

**Parameters**:
- `width` (number): window width in pixels
- `height` (number): window height in pixels
- `title` (string): window title

**Returns**:
- `Window`: window object on successful creation
- `null`: if creation failed

**Example**:
```datacode
let window = plot.window(800, 600, "My Window")
```

#### `plot.draw(window, image) -> null`
Draws an image in the specified window.

**Parameters**:
- `window` (Window): window object
- `image` (Image): image object

**Returns**:
- `null`: always returns null

**Example**:
```datacode
plot.draw(window, image)
```

#### `plot.wait(window) -> null`
Blocks program execution until the user closes the window.

**Parameters**:
- `window` (Window): window object

**Returns**:
- `null`: always returns null

**Example**:
```datacode
plot.wait(window)
print("Window closed")
```

#### `plot.show(path) -> null`
#### `plot.show(figure, title="...") -> null`
Convenient function for quick viewing of an image or figure. Automatically loads the image, creates a window with image dimensions, displays it, and waits for closure. Can also display a figure with subplots.

**Parameters**:
- `path` (string): path to the image file
- `figure` (Figure): figure object with subplots
- `title` (string, optional): window title

**Returns**:
- `null`: always returns null

**Example**:
```datacode
plot.show("path/to/image.png")
# or
fig = plot.subplots(2, 2)
plot.show(fig, title="My Figure")
```

#### `plot.subplots(rows, cols, figsize=(width, height)) -> Figure`
Creates a figure with a subplots grid for displaying multiple plots simultaneously.

**Parameters**:
- `rows` (number): number of rows in the grid
- `cols` (number): number of columns in the grid
- `figsize` (tuple, optional): figure size in inches (width, height), default (10, 10)

**Returns**:
- `Figure`: figure object with a two-dimensional array of axes `axes[row][col]`

**Example**:
```datacode
fig = plot.subplots(3, 3, figsize=(10, 10))
axis = fig.axes[0][0]  # Access to the first axis
```

#### `plot.tight_layout(figure) -> null`
Automatically adjusts padding between subplots for a neater display.

**Parameters**:
- `figure` (Figure): figure object

**Returns**:
- `null`: always returns null

**Example**:
```datacode
fig = plot.subplots(2, 2)
# ... configure subplots ...
plot.tight_layout(fig)
plot.show(fig)
```

#### `plot.bar(x, y) -> null`
Creates a bar chart.

**Parameters**:
- `x` (array of strings): categories for the X axis
- `y` (array of numbers): values for the Y axis

**Returns**:
- `null`: always returns null

**Example**:
```datacode
x = ["Category 1", "Category 2", "Category 3"]
y = [10, 20, 15]
plot.bar(x, y)
plot.show(title="Bar Chart")
```

#### `plot.line(x, y, connected, color) -> null`
Creates a line chart.

**Parameters**:
- `x` (array of numbers): values for the X axis
- `y` (array of numbers): values for the Y axis
- `connected` (boolean): whether to connect points with a line
- `color` (string, optional): line color (blue, green, red, black, white, or hex code)

**Returns**:
- `null`: always returns null

**Example**:
```datacode
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 6, 3]
plot.line(x, y, true, color="blue")
plot.show(title="Line Chart")
```

#### `plot.pie(x, y) -> null`
Creates a pie chart.

**Parameters**:
- `x` (array of strings): labels for sectors
- `y` (array of numbers): values for sectors

**Returns**:
- `null`: always returns null

**Example**:
```datacode
x = ["Item 1", "Item 2", "Item 3"]
y = [30, 40, 30]
plot.pie(x, y)
plot.show(title="Pie Chart")
```

#### `plot.heatmap(data, min, max, palette) -> null`
Creates a heatmap from a data matrix.

**Parameters**:
- `data` (array of arrays of numbers): two-dimensional data array
- `min` (number, optional): minimum value for the color scale
- `max` (number, optional): maximum value for the color scale
- `palette` (string, optional): color palette (e.g., "red")

**Returns**:
- `null`: always returns null

**Example**:
```datacode
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
plot.heatmap(data)
plot.show(title="Heatmap")
```

#### `plot.xlabel(label) -> null`
#### `plot.ylabel(label) -> null`
Sets the label for the X or Y axis.

**Parameters**:
- `label` (string): label text

**Returns**:
- `null`: always returns null

**Example**:
```datacode
plot.xlabel("Time")
plot.ylabel("Value")
```

### Axis Object Methods

#### `axis.imshow(image, cmap='gray') -> null`
Displays an image in the axis.

**Parameters**:
- `image` (Image | Tensor): image to display
- `cmap` (string, optional): color map ('gray', 'viridis', etc.), default 'gray'

**Returns**:
- `null`: always returns null

**Example**:
```datacode
axis = fig.axes[0][0]
axis.imshow(image, cmap='gray')
```

#### `axis.set_title(title) -> null`
Sets the title for the axis.

**Parameters**:
- `title` (string): title text

**Returns**:
- `null`: always returns null

**Example**:
```datacode
axis.set_title("My Image")
```

#### `axis.axis('off') -> null`
#### `axis.axis('on') -> null`
Controls the visibility of coordinate axes.

**Parameters**:
- `mode` (string): 'off' to hide axes, 'on' to show

**Returns**:
- `null`: always returns null

**Example**:
```datacode
axis.axis('off')  # Hide axes
axis.axis('on')   # Show axes
```

### Supported Image Formats

The `plot` module supports the following image formats:
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **GIF** (.gif)
- **BMP** (.bmp)
- And other formats supported by the `image` library in Rust

### Data Types

#### Image
Image object containing:
- Pixel data in RGBA format
- Image width
- Image height

#### Window
Window object containing:
- Window dimensions (width, height)
- Window title
- Window state (open/closed)

#### Figure
Figure object with subplots, containing:
- Two-dimensional array of axes `axes[row][col]`
- Figure size `figsize` (width, height)
- Layout settings

#### Axis
Axis object for displaying plots, containing:
- Image to display
- Axis title
- Coordinate axes visibility settings
- Color map (cmap)

## ⚠️ Important Notes

1. **File Paths**: Use correct paths to image files. Relative paths are specified relative to the current program execution directory.

2. **Blocking Functions**: The `plot.show()` and `plot.wait()` functions block program execution until the user closes the window.

3. **Result Checking**: Always check the result of image loading and window creation for `null` to handle possible errors.

4. **Centering**: The image is automatically centered in the window if it is smaller than the window size.

5. **Current Status**: Some module functions may be partially implemented. Check the current documentation for information on available capabilities.

## 🔗 Navigation

### Previous Sections
- **[09-data-model-creation](../09-data-model-creation/)** - working with data and tables
- **[08-websocket](../08-websocket/)** - working with WebSocket
- **[07-loops](../07-loops/)** - loops and iterations

### Next Steps
After studying the `plot` module, you can:
- Integrate visualization into your projects
- Create tools for viewing images
- Develop applications with a graphical interface

## 💡 Tips

- **Start Simple**: Use `plot.show()` for quick image viewing
- **Handle Errors**: Always check operation results for `null`
- **Use Correct Paths**: Make sure file paths are specified correctly
- **Experiment**: Try different window sizes and image formats

## 📚 Additional Information

For more detailed information about the `plot` module and its capabilities, refer to the main project documentation.

---

**Good luck working with graphics in DataCode!** 🎨✨
