// Font atlas for caching rasterized glyphs

use std::collections::HashMap;

/// Cached glyph data - stores the result of font.rasterize()
/// Type matches: (Metrics, Vec<u8>) from fontdue
type GlyphData = (fontdue::Metrics, Vec<u8>);

/// Font atlas for caching rasterized glyphs
pub struct FontAtlas {
    cache: HashMap<(char, u32), GlyphData>,
}

impl FontAtlas {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
    
    /// Get cached glyph or rasterize and cache it
    /// Returns (metrics, bitmap) for the glyph
    /// font_size is rounded to nearest 0.5 for caching purposes
    pub fn get_or_rasterize<F>(
        &mut self,
        ch: char,
        font_size: f32,
        rasterize_fn: F,
    ) -> GlyphData
    where
        F: FnOnce(char, f32) -> GlyphData,
    {
        // Round font_size to nearest 0.5 for caching (e.g., 32.0, 32.5, 33.0)
        // This reduces cache size while maintaining quality
        let rounded_size = (font_size * 2.0).round() as u32;
        let key = (ch, rounded_size);
        
        if let Some(cached) = self.cache.get(&key) {
            // Clone the cached data
            (cached.0.clone(), cached.1.clone())
        } else {
            let result = rasterize_fn(ch, font_size);
            self.cache.insert(key, result.clone());
            result
        }
    }
    
    /// Clear the cache (useful if memory becomes an issue)
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for FontAtlas {
    fn default() -> Self {
        Self::new()
    }
}

