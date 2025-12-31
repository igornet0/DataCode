# Plot Module - Charts and Visualization

The `plot` module provides functionality for working with images, displaying them in windows, as well as creating various types of charts and diagrams in DataCode.

## Module Import

The `plot` module is available as a global variable after import:

```datacode
import plot
```

## Working with Images

### `plot.image(path) -> Image | null`

Loads an image from a file at the specified path.

**Parameters:**
- `path` (string): path to the image file

**Returns:**
- `Image`: image object on successful loading
- `null`: if loading failed

**Example:**
```datacode
let img = plot.image("path/to/image.png")
if img != null {
    print("Image loaded!")
} else {
    print("Error loading image")
}
```

**Supported formats:**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- And other formats supported by the `image` library in Rust

### `plot.window(width, height, title) -> Window | null`

Creates a new window for displaying graphics.

**Parameters:**
- `width` (number): window width in pixels
- `height` (number): window height in pixels
- `title` (string): window title

**Returns:**
- `Window`: window object on successful creation
- `null`: if creation failed

**Example:**
```datacode
let window = plot.window(800, 600, "My Window")
if window != null {
    print("Window created successfully")
}
```

### `plot.draw(window, image) -> null`

Draws an image in the specified window.

**Parameters:**
- `window` (Window): window object
- `image` (Image): image object

**Returns:**
- `null`: always returns null

**Example:**
```datacode
let window = plot.window(800, 600, "Viewer")
let image = plot.image("image.png")
if window != null && image != null {
    plot.draw(window, image)
}
```

### `plot.wait(window) -> null`

Blocks program execution until the user closes the window.

**Parameters:**
- `window` (Window): window object

**Returns:**
- `null`: always returns null

**Example:**
```datacode
let window = plot.window(800, 600, "Waiting")
plot.wait(window)
print("Window closed")
```

### `plot.show(path) -> null`
### `plot.show(figure, title="...") -> null`
### `plot.show(title="...") -> null`

Convenient function for quick viewing of an image or figure. Automatically loads the image, creates a window with image dimensions, displays it, and waits for closure. Can also display a figure with subplots or the current chart.

**Parameters:**
- `path` (string, optional): path to the image file
- `figure` (Figure, optional): figure object with subplots
- `title` (string, optional): window title

**Returns:**
- `null`: always returns null

**Examples:**
```datacode
# Show image from file
plot.show("path/to/image.png")

# Show figure with subplots
fig = plot.subplots(2, 2)
# ... configure subplots ...
plot.show(fig, title="My Figure")

# Show current chart with title
plot.bar(x, y)
plot.show(title="My Chart")
```

## Working with Subplots

### `plot.subplots(rows, cols, figsize=(width, height)) -> Figure`

Creates a figure with a subplots grid for displaying multiple plots simultaneously.

**Parameters:**
- `rows` (number): number of rows in the grid
- `cols` (number): number of columns in the grid
- `figsize` (tuple, optional): figure size in inches (width, height), default (10, 10)

**Returns:**
- `Figure`: figure object with a two-dimensional array of axes `axes[row][col]`

**Example:**
```datacode
# Create 3x3 grid
fig = plot.subplots(3, 3, figsize=(10, 10))

# Access to axes
axis = fig.axes[0][0]  # First axis (top-left corner)
axis = fig.axes[1][2]  # Axis in second row, third column
```

### `plot.tight_layout(figure) -> null`

Automatically adjusts padding between subplots for a neater display.

**Parameters:**
- `figure` (Figure): figure object

**Returns:**
- `null`: always returns null

**Example:**
```datacode
fig = plot.subplots(2, 2)
# ... configure subplots ...
plot.tight_layout(fig)
plot.show(fig)
```

## Axis Object Methods

The `Axis` object is obtained via `fig.axes[row][col]` from a `Figure` object.

### `axis.imshow(image, cmap='gray') -> null`

Displays an image in the axis.

**Parameters:**
- `image` (Image | Tensor): image to display
- `cmap` (string, optional): color map ('gray', 'viridis', etc.), default 'gray'

**Returns:**
- `null`: always returns null

**Example:**
```datacode
fig = plot.subplots(1, 1)
axis = fig.axes[0][0]
let img = plot.image("image.png")
if img != null {
    axis.imshow(img, cmap='gray')
}
plot.show(fig)
```

### `axis.set_title(title) -> null`

Sets the title for the axis.

**Parameters:**
- `title` (string): title text

**Returns:**
- `null`: always returns null

**Example:**
```datacode
axis.set_title("My Image")
```

### `axis.axis('off') -> null`
### `axis.axis('on') -> null`

Controls the visibility of coordinate axes.

**Parameters:**
- `mode` (string): 'off' to hide axes, 'on' to show

**Returns:**
- `null`: always returns null

**Example:**
```datacode
axis.axis('off')  # Hide axes
axis.axis('on')   # Show axes
```

## Creating Charts

### `plot.bar(x, y) -> null`

Creates a bar chart.

**Parameters:**
- `x` (array of strings): categories for the X axis
- `y` (array of numbers): values for the Y axis

**Returns:**
- `null`: always returns null

**Example:**
```datacode
x = ["Category 1", "Category 2", "Category 3"]
y = [10, 20, 15]
plot.xlabel("Category")
plot.ylabel("Value")
plot.bar(x, y)
plot.show(title="Bar Chart")
```

### `plot.line(x, y, connected, color) -> null`

Creates a line chart.

**Parameters:**
- `x` (array of numbers): values for the X axis
- `y` (array of numbers): values for the Y axis
- `connected` (boolean): whether to connect points with a line
- `color` (string, optional): line color

**Returns:**
- `null`: always returns null

**Supported colors:**
- Named colors: `"blue"`, `"green"`, `"red"`, `"black"`, `"white"`
- Hex codes: e.g., `"#a434eb"`

**Example:**
```datacode
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 6, 3]
y1 = [7, 12, 2, 9, 1]

plot.xlabel("Time")
plot.ylabel("Value")

plot.line(x, y, true, color="blue")
plot.line(x, y1, true, color="green")
plot.show(title="Line Chart")
```

### `plot.pie(x, y) -> null`

Creates a pie chart.

**Parameters:**
- `x` (array of strings): labels for sectors
- `y` (array of numbers): values for sectors

**Returns:**
- `null`: always returns null

**Example:**
```datacode
x = ["Item 1", "Item 2", "Item 3"]
y = [30, 40, 30]
plot.pie(x, y)
plot.show(title="Pie Chart")
```

### `plot.heatmap(data, min, max, palette) -> null`

Creates a heatmap from a data matrix.

**Parameters:**
- `data` (array of arrays of numbers): two-dimensional data array
- `min` (number, optional): minimum value for the color scale
- `max` (number, optional): maximum value for the color scale
- `palette` (string, optional): color palette (e.g., "red")

**Returns:**
- `null`: always returns null

**Example:**
```datacode
data = [
  [1, 2, 3, 4, 5],
  [2, 4, 6, 8, 10],
  [3, 6, 9, 12, 15]
]

# Basic usage
plot.heatmap(data)
plot.show(title="Heatmap")

# With range setting
plot.heatmap(data, min=0, max=100)
plot.show(title="Heatmap with Range")

# With palette selection
plot.heatmap(data, palette="red")
plot.show(title="Red Heatmap")
```

## Axis Configuration

### `plot.xlabel(label) -> null`

Sets the label for the X axis.

**Parameters:**
- `label` (string): label text

**Returns:**
- `null`: always returns null

**Example:**
```datacode
plot.xlabel("Time")
plot.bar(x, y)
plot.show()
```

### `plot.ylabel(label) -> null`

Sets the label for the Y axis.

**Parameters:**
- `label` (string): label text

**Returns:**
- `null`: always returns null

**Example:**
```datacode
plot.ylabel("Value")
plot.line(x, y, true)
plot.show()
```

## Data Types

### Image

Image object containing:
- Pixel data in RGBA format
- Image width
- Image height

### Window

Window object containing:
- Window dimensions (width, height)
- Window title
- Window state (open/closed)

### Figure

Figure object with subplots, containing:
- Two-dimensional array of axes `axes[row][col]`
- Figure size `figsize` (width, height)
- Layout settings

### Axis

Axis object for displaying plots, containing:
- Image to display
- Axis title
- Coordinate axes visibility settings
- Color map (cmap)

## Important Notes

1. **File Paths**: Use correct paths to image files. Relative paths are specified relative to the current program execution directory.

2. **Blocking Functions**: The `plot.show()` and `plot.wait()` functions block program execution until the user closes the window.

3. **Result Checking**: Always check the result of image loading and window creation for `null` to handle possible errors.

4. **Centering**: The image is automatically centered in the window if it is smaller than the window size.

5. **Operation Sequence**: When creating charts, first configure the axes (`plot.xlabel()`, `plot.ylabel()`), then create the chart (`plot.bar()`, `plot.line()`, etc.), and finally call `plot.show()` to display.

6. **Multiple Charts**: You can create multiple charts on the same axis by calling chart creation functions multiple times before `plot.show()`.

## Usage Examples

### Complete Image Working Example

```datacode
import plot

# Load image
let img = plot.image("image.png")
if img == null {
    print("Error loading image")
    exit(1)
}

# Create window
let window = plot.window(1024, 768, "Image Viewer")
if window == null {
    print("Error creating window")
    exit(1)
}

# Display image
plot.draw(window, img)

# Wait for window closure
plot.wait(window)
print("Window closed")
```

### Subplots Working Example

```datacode
import plot, ml

# Create 2x2 grid
fig = plot.subplots(2, 2, figsize=(10, 10))

# Load data
data = ml.load_mnist("test")
images = []
for x, y in data {
    push(images, x)
    if len(images) >= 4 {
        break
    }
}

# Fill subplots
for i in range(4) {
    row = i // 2
    col = i % 2
    axis = fig.axes[row][col]
    axis.imshow(images[i], cmap='gray')
    axis.set_title("Image " + str(i))
    axis.axis('off')
}

# Configure padding and display
plot.tight_layout(fig)
plot.show(fig, title="Subplots Example")
```

### Various Charts Creation Example

```datacode
import plot

# Bar chart
x = ["Cheese", "Milk", "Vegetables", "Fruits", "Water"]
y = [2, 4, 1, 6, 3]
plot.xlabel("Category")
plot.ylabel("Value")
plot.bar(x, y)
plot.show(title="Bar Chart")

# Line chart
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 6, 3]
y1 = [7, 12, 2, 9, 1]
plot.xlabel("Time")
plot.ylabel("Value")
plot.line(x, y, true, color="blue")
plot.line(x, y1, true, color="green")
plot.show(title="Line Chart")

# Pie chart
x = ["Cheese", "Milk", "Vegetables", "Fruits", "Water"]
y = [2, 4, 1, 6, 3]
plot.pie(x, y)
plot.show(title="Pie Chart")

# Heatmap
data = [
  [1, 2, 3, 4, 5],
  [2, 4, 6, 8, 10],
  [3, 6, 9, 12, 15]
]
plot.heatmap(data)
plot.show(title="Heatmap")
```

## See Also

- [Plot module usage examples](../../../examples/en/10-plot/)
- [Built-in Functions](../builtin_functions.md)
- [Data Types](../data_types.md)

