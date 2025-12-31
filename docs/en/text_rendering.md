# Text Rendering in Plot Module

## Overview

The `plot` module uses the `fontdue` library to render text on graphs and images. The system supports UTF-8, including Cyrillic, and uses glyph caching for performance optimization.

## Architecture

### Main Components

1. **Renderer** (`src/lib/plot/renderer.rs`) - main rendering class
2. **FontAtlas** (`src/lib/plot/font_atlas.rs`) - cache for rasterized glyphs
3. **Font** (from `fontdue`) - loaded font
4. **Buffer** - pixel buffer for rendering (BGRA format)

### Data Flow

```
draw_figure() 
  → draw_text_improved() 
    → draw_text_with_font() 
      → FontAtlas.get_or_rasterize() 
        → font.rasterize() (if not cached)
```

## Rendering Process

### 1. Font Initialization

**Location:** `renderer.rs:525-692` (`load_font()`)

Font loading priority:
1. Monospace fonts (Monaco, Menlo, Consolas, etc.)
2. Proportional fonts (Inter, Roboto, system fonts)
3. Bitmap fallback if no font found

### 2. Baseline Calculation

**Location:** `renderer.rs:426-435`

The baseline is the imaginary line on which characters "sit". Correct baseline calculation is critical for proper text alignment:

```rust
let text_y = if let Some(font) = self.font.as_ref() {
    if let Some(line_metrics) = font.horizontal_line_metrics(font_size) {
        cell_y + line_metrics.ascent.round() as usize
    } else {
        cell_y + 14 // fallback
    }
} else {
    cell_y + 14
};
```

**Important:** 
- `cell_y` is the Y-coordinate of the cell's top edge
- `line_metrics.ascent` is the distance from baseline to top of characters
- Baseline ≠ top of text!

### 3. Text Width Calculation

**Location:** `renderer.rs:694-710` (`calculate_text_width()`)

For text centering, exact width is needed:

```rust
fn calculate_text_width(font: Option<&Font>, text: &str, font_size: f32) -> f32 {
    if let Some(font) = font {
        let char_width = Self::calculate_max_char_width(font, font_size);
        text.chars().count() as f32 * char_width
    } else {
        text.chars().count() as f32 * font_size * 0.6
    }
}
```

### 4. Main Rendering Function

**Location:** `renderer.rs:801-895` (`draw_text_with_font()`)

#### Glyph Positioning Formula

**Horizontal position:**
```
glyph_x = current_x + normalized_xmin
normalized_xmin = metrics.xmin - min_xmin
```

**Vertical position:**
```
glyph_y = baseline_y + metrics.ymin
```

**Important:** 
- `metrics.ymin` can be negative for descenders (p, d, q, g, y)
- This is normal - different characters have different `ymin`
- With correct baseline, all characters align properly

#### Rendering Loop

```rust
for ch in text.chars() {
    // Get or rasterize glyph (with caching)
    let (metrics, bitmap) = atlas.get_or_rasterize(ch, rasterize_size, |c, size| {
        font.rasterize(c, size)
    });
    
    // Calculate glyph position
    let normalized_xmin = metrics.xmin as f32 - min_xmin;
    let glyph_x = (current_x + normalized_xmin).round() as i32;
    let glyph_y = (baseline_y + metrics.ymin as f32).round() as i32;
    
    // Render bitmap with alpha blending
    // ...
    
    // Advance to next character
    current_x += char_width;
}
```

### 5. Alpha Blending

**Location:** `renderer.rs:713-734` (`blend()`)

Text is rendered with alpha channel for smooth anti-aliasing:

```rust
fn blend(bg: u32, fg_r: u8, fg_g: u8, fg_b: u8, alpha: u8) -> u32 {
    // Integer blending: (fg * a + bg * inv_a) / 255
    // ...
}
```

**Pixel format:** BGRA (Blue, Green, Red, Alpha)
- Text is rendered in white (255, 255, 255)
- Alpha channel provides edge smoothing

### 6. Glyph Caching (FontAtlas)

**Location:** `font_atlas.rs`

`FontAtlas` caches rasterized glyphs for performance:

```rust
pub struct FontAtlas {
    cache: HashMap<(char, u32), GlyphData>,
}
```

**Cache key:** `(character, rounded_font_size)`

**Size rounding:**
```rust
let rounded_size = (font_size * 2.0).round() as u32;
```

This reduces cache size while maintaining quality (e.g., 32.0, 32.5, 33.0).

## Fontdue Coordinate Model

### Baseline-Oriented Model

Fontdue uses a baseline-oriented coordinate model:

1. **Baseline** - fixed horizontal line for the entire line
2. **metrics.ymin** - minimum Y-coordinate relative to baseline (can be negative)
3. **metrics.ymax** - maximum Y-coordinate relative to baseline (positive)
4. **metrics.xmin** - minimum X-coordinate relative to character start
5. **metrics.advance_width** - character width (distance to next character)

### Important Glyph Metrics

```rust
struct Metrics {
    advance_width: f32,  // Character width
    advance_height: f32, // Character height
    bearing_x: f32,     // Horizontal offset
    bearing_y: f32,     // Vertical offset
    xmin: i32,           // Minimum X-coordinate of bounding box
    ymin: i32,           // Minimum Y-coordinate relative to baseline
    xmax: i32,           // Maximum X-coordinate of bounding box
    ymax: i32,           // Maximum Y-coordinate relative to baseline
    width: usize,        // Bitmap width
    height: usize,       // Bitmap height
}
```

## Implementation Details

### 1. DPI-Aware Rendering

Font size scales with screen DPI:

```rust
let scale_factor = window_ref.scale_factor() as f32;
let font_size = base_font_size * self.scale_factor;
```

### 2. Monospace vs Proportional Fonts

- **Monospace:** all characters have same width (`advance_width`)
- **Proportional:** character widths vary

For monospace fonts, fixed character width is used for uniform spacing.

### 3. xmin Normalization

For uniform alignment, all characters are normalized by minimum `xmin`:

```rust
let normalized_xmin = metrics.xmin as f32 - min_xmin;
```

This ensures all characters start at the same visual position.

### 4. Different ymin for Different Characters

**This is normal!** Different characters have different `ymin`:
- Characters without descenders (x, a, e, o): `ymin ≈ -2`
- Characters with descenders (p, d, q, g, y): `ymin ≈ -8`
- Digits: `ymin ≈ -1`
- Punctuation: may have their own values

With correct baseline, all characters align properly.

## Common Issues and Solutions

### Issue: "Floating" Text

**Cause:** Incorrect baseline setup (using `cell_y + 14` instead of font metrics)

**Solution:** Use `font.horizontal_line_metrics()` for correct baseline:

```rust
let text_y = if let Some(font) = self.font.as_ref() {
    if let Some(line_metrics) = font.horizontal_line_metrics(font_size) {
        cell_y + line_metrics.ascent.round() as usize
    } else {
        cell_y + 14 // fallback
    }
} else {
    cell_y + 14
};
```

### Issue: Different ymin for Different Characters

**This is not an issue!** Different `ymin` values are normal font behavior. With correct baseline, all characters align properly.

**Don't:**
- ❌ Normalize ymin
- ❌ Use fixed ymin
- ❌ Special cases for different characters

**Do:**
- ✅ Use `baseline_y + metrics.ymin` as-is
- ✅ Calculate baseline correctly via `horizontal_line_metrics()`

## Performance

### Optimizations

1. **Glyph caching:** `FontAtlas` caches rasterized glyphs
2. **Size rounding:** font size rounded to 0.5 to reduce cache size
3. **Integer blending:** fast integer alpha blending
4. **Boundary checks:** buffer boundary checks before pixel writes

### Limitations

- Maximum text length in bitmap fallback: 50 characters
- Maximum image scale: 10x
- Cache size is unlimited (may grow with many characters)

## Conclusion

The text rendering system in the plot module uses modern approaches:

1. ✅ Baseline-oriented coordinate model
2. ✅ Glyph caching for optimization
3. ✅ DPI-aware rendering
4. ✅ UTF-8 and Cyrillic support
5. ✅ Alpha blending for anti-aliasing
6. ✅ Proper use of font metrics

Key point: correct baseline calculation via `horizontal_line_metrics()` ensures proper alignment of all characters regardless of their individual metrics.



