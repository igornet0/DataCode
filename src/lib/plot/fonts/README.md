# Fonts Directory

This directory contains embedded fonts used by the plot module.

## Current Fonts

The following fonts are already integrated and support **both Russian and English** languages:

### Inter
- **Location**: `Inter/static/Inter_18pt-Regular.ttf` (primary)
- **Fallback**: `Inter/Inter-VariableFont_opsz,wght.ttf`
- **Description**: Modern, clean sans-serif font optimized for UI and numbers
- **Language Support**: Russian (Cyrillic), English (Latin), and many other languages
- **License**: SIL Open Font License (see `Inter/OFL.txt`)

### Roboto
- **Location**: `roboto/static/Roboto-Regular.ttf` (fallback)
- **Fallback**: `roboto/Roboto-VariableFont_wdth,wght.ttf`
- **Description**: Google's font family - excellent readability
- **Language Support**: Russian (Cyrillic), English (Latin), and many other languages
- **License**: Apache License 2.0 (see `roboto/OFL.txt`)

## Font Loading Priority

The renderer tries to load fonts in this order:
1. Inter Regular (static) - **Primary choice**
2. Inter Variable Font - Fallback if static not available
3. Roboto Regular (static) - Second choice
4. Roboto Variable Font - Fallback
5. System fonts (platform-specific)

## Adding New Fonts

To add a new embedded font:

1. Place TTF or OTF font files in this directory
2. Update `load_font()` in `renderer.rs` to include the new font
3. Ensure the font supports the languages you need (check font documentation)
4. Verify the license allows embedding in your application

