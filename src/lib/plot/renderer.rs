// Renderer for plot module

use crate::plot::{Window, Image, Figure, FontAtlas};
use crate::plot::window::ImageViewState;
use crate::plot::command::FigureData;
use softbuffer::{Context, Surface};
use std::rc::Rc;
use std::cell::RefCell;
use fontdue::Font;

pub struct Renderer {
    _context: Context<&'static winit::window::Window>,
    surface: Surface<&'static winit::window::Window, &'static winit::window::Window>,
    font: Option<Font>,
    scale_factor: f32,
    atlas: FontAtlas,
    window_ref: &'static winit::window::Window, // Keep reference to window for recreating surface
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, softbuffer::SoftBufferError> {
        // We need to extend the lifetime, but this is unsafe
        // In practice, the window will outlive the renderer
        let window_ref: &'static winit::window::Window = unsafe {
            std::mem::transmute(&window.window_handle)
        };
        let context = Context::new(window_ref)?;
        let surface = Surface::new(&context, window_ref)?;
        
        // Get scale factor from window for DPI-aware rendering
        let scale_factor = window_ref.scale_factor() as f32;
        
        // Try to load a system font or use built-in font
        // For now, we'll create a simple font from built-in data
        // In the future, you can embed a TTF font file here
        let font = Self::load_font();
        
        Ok(Self { 
            _context: context,
            surface,
            font,
            scale_factor,
            atlas: FontAtlas::new(),
            window_ref,
        })
    }
    
    /// Update scale factor from window (called on resize or DPI change)
    pub fn update_scale_factor(&mut self, window: &Window) {
        let window_ref: &'static winit::window::Window = unsafe {
            std::mem::transmute(&window.window_handle)
        };
        self.scale_factor = window_ref.scale_factor() as f32;
    }
    
    /// Get current scale factor (for debugging)
    pub fn get_scale_factor(&self) -> f32 {
        self.scale_factor
    }
    
    /// Force buffer resize by recreating surface if needed
    /// This ensures the buffer is updated to the new window size
    /// Note: softbuffer may not auto-resize, so we need to recreate surface
    pub fn ensure_buffer_resized(&mut self) -> Result<(u32, u32), softbuffer::SoftBufferError> {
        // Get current window size
        let window_size = self.window_ref.inner_size();
        let expected_width = window_size.width;
        let expected_height = window_size.height;
        
        // Check current buffer size
        let buffer = self.surface.buffer_mut()?;
        let current_width = buffer.width().get();
        let current_height = buffer.height().get();
        
        // If buffer size doesn't match window size, we need to recreate surface
        // softbuffer doesn't automatically resize the buffer, so we recreate the surface
        if current_width != expected_width || current_height != expected_height {
            // Recreate surface to match new window size
            let new_surface = Surface::new(&self._context, self.window_ref)?;
            self.surface = new_surface;
            
            // Get new buffer size
            let new_buffer = self.surface.buffer_mut()?;
            let new_width = new_buffer.width().get();
            let new_height = new_buffer.height().get();
            Ok((new_width, new_height))
        } else {
            Ok((current_width, current_height))
        }
    }
    
    /// Get current buffer size (for debugging)
    pub fn get_buffer_size(&mut self) -> Result<(u32, u32), softbuffer::SoftBufferError> {
        let buffer = self.surface.buffer_mut()?;
        Ok((buffer.width().get(), buffer.height().get()))
    }
    
    /// Calculate adaptive margins based on window size
    /// Returns (left, right, top, bottom) margins
    /// Uses percentage of window size with minimum values for small windows
    fn calculate_adaptive_margins(scale_factor: f32, buffer_width: u32, buffer_height: u32) -> (u32, u32, u32, u32) {
        // Calculate margins as percentage of window size, with minimum values
        // Left margin: 12% of width, minimum 150px (for y-axis labels)
        let left_margin = ((buffer_width as f32 * 0.12).max(150.0) * scale_factor).min(buffer_width as f32 * 0.25) as u32;
        
        // Right margin: 3% of width, minimum 30px
        let right_margin = ((buffer_width as f32 * 0.03).max(30.0) * scale_factor).min(buffer_width as f32 * 0.1) as u32;
        
        // Top margin: 5% of height, minimum 50px (for title)
        let top_margin = ((buffer_height as f32 * 0.05).max(50.0) * scale_factor).min(buffer_height as f32 * 0.15) as u32;
        
        // Bottom margin: 8% of height, minimum 70px (for x-axis labels)
        let bottom_margin = ((buffer_height as f32 * 0.08).max(70.0) * scale_factor).min(buffer_height as f32 * 0.2) as u32;
        
        (left_margin, right_margin, top_margin, bottom_margin)
    }

    pub fn draw_image(&mut self, image: &Image, _window: &Window) -> Result<(), softbuffer::SoftBufferError> {
        let image_width = image.width;
        let image_height = image.height;

        // Get buffer - it will be resized automatically based on window size
        let mut buffer = self.surface.buffer_mut()?;
        
        // Get actual buffer dimensions
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        // BGRA format: 0xAABBGGRR
        let bg_color = 0xFF1E1E1E; // Dark gray
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate adaptive scale to fit image in window while maintaining aspect ratio
        // Leave small margins (2% on each side) for better appearance
        let margin_factor = 0.96;
        let available_width = (buffer_width as f32 * margin_factor) as u32;
        let available_height = (buffer_height as f32 * margin_factor) as u32;
        
        // Calculate scale factors for width and height
        let scale_x = available_width as f32 / image_width as f32;
        let scale_y = available_height as f32 / image_height as f32;
        
        // Use minimum scale to ensure image fits in both dimensions (maintain aspect ratio)
        // But ensure minimum scale for very small images
        let min_base_scale = if image_width <= 100 && image_height <= 100 {
            10.0
        } else if image_width <= 200 && image_height <= 200 {
            5.0
        } else {
            4.0
        };
        
        let adaptive_scale = scale_x.min(scale_y);
        let scale_f = adaptive_scale.max(min_base_scale);
        let scale = scale_f as usize;

        // Calculate scaled image dimensions
        let scaled_width = (image_width as f32 * scale_f) as u32;
        let scaled_height = (image_height as f32 * scale_f) as u32;

        // Center the scaled image in the buffer (maintain aspect ratio)
        let offset_x = if scaled_width < buffer_width {
            (buffer_width - scaled_width) / 2
        } else {
            0
        };
        let offset_y = if scaled_height < buffer_height {
            (buffer_height - scaled_height) / 2
        } else {
            0
        };

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let image_width_usize = image_width as usize;
        let image_height_usize = image_height as usize;
        let offset_x_usize = offset_x as usize;
        let offset_y_usize = offset_y as usize;
        let scale_usize = scale as usize;

        // Nearest neighbor scaling (like matplotlib interpolation='nearest')
        // Each source pixel becomes a scale×scale block in the output
        for src_y in 0..image_height_usize {
            for src_x in 0..image_width_usize {
                let image_idx = (src_y * image_width_usize + src_x) * 4;
                if image_idx + 3 < image.data.len() {
                    // Image data is in RGBA format
                    let r = image.data[image_idx] as u32;
                    let g = image.data[image_idx + 1] as u32;
                    let b = image.data[image_idx + 2] as u32;
                    let a = image.data[image_idx + 3] as u32;
                    
                    // Convert RGBA to BGRA format (softbuffer uses BGRA on most platforms)
                    // Format: 0xAABBGGRR
                    let pixel = (a << 24) | (b << 16) | (g << 8) | r;
                    
                    // Draw scale×scale block (nearest neighbor - no interpolation)
                    let dst_y_start = offset_y_usize + src_y * scale_usize;
                    let dst_x_start = offset_x_usize + src_x * scale_usize;
                    
                    for dy in 0..scale_usize {
                        for dx in 0..scale_usize {
                            let dst_y = dst_y_start + dy;
                            let dst_x = dst_x_start + dx;
                            
                            if dst_y < buffer_height_usize && dst_x < buffer_width_usize {
                                let buffer_idx = dst_y * buffer_width_usize + dst_x;
                                if buffer_idx < buffer.len() {
                                    buffer[buffer_idx] = pixel;
                                }
                            }
                        }
                    }
                }
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Draw image with zoom and pan transformations
    pub fn draw_image_with_transform(
        &mut self,
        image: &Image,
        _window: &Window,
        view_state: &ImageViewState,
    ) -> Result<(), softbuffer::SoftBufferError> {
        let image_width = image.width;
        let image_height = image.height;

        // Get buffer - it will be resized automatically based on window size
        let mut buffer = self.surface.buffer_mut()?;
        
        // Get actual buffer dimensions
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Calculate adaptive base scale to fit image in window while maintaining aspect ratio
        // Leave small margins (2% on each side) for better appearance
        let margin_factor = 0.96;
        let available_width = (buffer_width as f32 * margin_factor) as u32;
        let available_height = (buffer_height as f32 * margin_factor) as u32;
        
        // Calculate scale factors for width and height
        let scale_x = available_width as f32 / image_width as f32;
        let scale_y = available_height as f32 / image_height as f32;
        
        // Use minimum scale to ensure image fits in both dimensions (maintain aspect ratio)
        // But ensure minimum scale for very small images
        let min_base_scale = if image_width <= 100 && image_height <= 100 {
            10.0
        } else if image_width <= 200 && image_height <= 200 {
            5.0
        } else {
            4.0
        };
        
        let adaptive_base_scale = scale_x.min(scale_y).max(min_base_scale);

        // Apply zoom to scale
        let effective_scale = (adaptive_base_scale * view_state.zoom) as usize;
        let effective_scale_f = adaptive_base_scale * view_state.zoom;

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        // BGRA format: 0xAABBGGRR
        let bg_color = 0xFF1E1E1E; // Dark gray
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate scaled image dimensions
        let scaled_width = (image_width as f32 * effective_scale_f) as u32;
        let scaled_height = (image_height as f32 * effective_scale_f) as u32;

        // Calculate image position with pan offset
        // Center the image in the buffer, then apply pan
        let base_offset_x = if scaled_width < buffer_width {
            (buffer_width - scaled_width) / 2
        } else {
            0
        };
        let base_offset_y = if scaled_height < buffer_height {
            (buffer_height - scaled_height) / 2
        } else {
            0
        };

        let offset_x = (base_offset_x as f32 + view_state.pan_x) as i32;
        let offset_y = (base_offset_y as f32 + view_state.pan_y) as i32;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let image_width_usize = image_width as usize;
        let image_height_usize = image_height as usize;

        // Calculate visible region to optimize rendering
        let start_x = (-offset_x / effective_scale as i32).max(0) as usize;
        let start_y = (-offset_y / effective_scale as i32).max(0) as usize;
        let end_x = ((buffer_width as i32 - offset_x) / effective_scale as i32).min(image_width as i32) as usize;
        let end_y = ((buffer_height_usize as i32 - offset_y) / effective_scale as i32).min(image_height as i32) as usize;

        // Draw only visible pixels
        for src_y in start_y..end_y.min(image_height_usize) {
            for src_x in start_x..end_x.min(image_width_usize) {
                let image_idx = (src_y * image_width_usize + src_x) * 4;
                if image_idx + 3 < image.data.len() {
                    // Image data is in RGBA format
                    let r = image.data[image_idx] as u32;
                    let g = image.data[image_idx + 1] as u32;
                    let b = image.data[image_idx + 2] as u32;
                    let a = image.data[image_idx + 3] as u32;
                    
                    // Convert RGBA to BGRA format (softbuffer uses BGRA on most platforms)
                    // Format: 0xAABBGGRR
                    let pixel = (a << 24) | (b << 16) | (g << 8) | r;
                    
                    // Draw effective_scale×effective_scale block (nearest neighbor - no interpolation)
                    let dst_y_start = offset_y + (src_y as i32 * effective_scale as i32);
                    let dst_x_start = offset_x + (src_x as i32 * effective_scale as i32);
                    
                    for dy in 0..effective_scale {
                        for dx in 0..effective_scale {
                            let dst_y = dst_y_start + dy as i32;
                            let dst_x = dst_x_start + dx as i32;
                            
                            if dst_y >= 0 && dst_y < buffer_height as i32 &&
                               dst_x >= 0 && dst_x < buffer_width as i32 {
                                let buffer_idx = (dst_y as usize) * buffer_width_usize + (dst_x as usize);
                                if buffer_idx < buffer.len() {
                                    buffer[buffer_idx] = pixel;
                                }
                            }
                        }
                    }
                }
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Draw multiple images in a grid layout
    /// images: vector of images to display (Arc<Mutex<Image>> for Send + Sync)
    /// rows: number of rows in grid
    /// cols: number of columns in grid
    /// titles: optional vector of titles for each image
    pub fn draw_image_grid(
        &mut self,
        images: &[std::sync::Arc<std::sync::Mutex<Image>>],
        rows: usize,
        cols: usize,
        _titles: Option<&[String]>,
    ) -> Result<(), softbuffer::SoftBufferError> {
        if images.is_empty() {
            return Ok(());
        }

        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate cell dimensions with adaptive padding
        // Padding scales with window size but has minimum value
        let padding = ((buffer_width.min(buffer_height) as f32 * 0.01).max(8.0) * self.scale_factor).min(20.0) as u32;
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        let cell_width = (buffer_width - padding * (cols_u32 + 1)) / cols_u32;
        let cell_height = (buffer_height - padding * (rows_u32 + 1)) / rows_u32;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let cell_width_usize = cell_width as usize;
        let cell_height_usize = cell_height as usize;
        let padding_usize = padding as usize;

        // Draw each image in its grid cell
        for (idx, image_arc) in images.iter().enumerate() {
            if idx >= rows * cols {
                break; // Don't draw more images than grid cells
            }

            let row = idx / cols;
            let col = idx % cols;

            let image = image_arc.lock().unwrap();
            let img_width = image.width as usize;
            let img_height = image.height as usize;

            // Calculate scale to fit image in cell (maintain aspect ratio)
            let scale_x = (cell_width_usize - 2 * padding_usize) as f32 / img_width as f32;
            let scale_y = (cell_height_usize - 2 * padding_usize) as f32 / img_height as f32;
            let scale = scale_x.min(scale_y).min(10.0); // Max scale 10x
            let scale_usize = scale as usize;

            // Calculate scaled dimensions
            let scaled_width = (img_width as f32 * scale) as usize;
            let scaled_height = (img_height as f32 * scale) as usize;

            // Calculate cell position
            let cell_x = padding_usize + col * (cell_width_usize + padding_usize);
            let cell_y = padding_usize + row * (cell_height_usize + padding_usize);

            // Center image in cell
            let offset_x = cell_x + (cell_width_usize - scaled_width) / 2;
            let offset_y = cell_y + (cell_height_usize - scaled_height) / 2;

            // Draw image with nearest neighbor scaling
            for src_y in 0..img_height {
                for src_x in 0..img_width {
                    let image_idx = (src_y * img_width + src_x) * 4;
                    if image_idx + 3 < image.data.len() {
                        let r = image.data[image_idx] as u32;
                        let g = image.data[image_idx + 1] as u32;
                        let b = image.data[image_idx + 2] as u32;
                        let a = image.data[image_idx + 3] as u32;
                        
                        let pixel = (a << 24) | (b << 16) | (g << 8) | r;
                        
                        // Draw scale×scale block
                        for dy in 0..scale_usize {
                            for dx in 0..scale_usize {
                                let dst_y = offset_y + src_y * scale_usize + dy;
                                let dst_x = offset_x + src_x * scale_usize + dx;
                                
                                if dst_y < buffer_height_usize && dst_x < buffer_width_usize {
                                    let buffer_idx = dst_y * buffer_width_usize + dst_x;
                                    if buffer_idx < buffer.len() {
                                        buffer[buffer_idx] = pixel;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Draw figure with axes (grid of images with titles)
    /// figure: reference to figure
    /// rows: number of rows
    /// cols: number of columns
    /// titles: titles for each cell
    pub fn draw_figure(
        &mut self,
        figure: &Rc<RefCell<Figure>>,
        rows: usize,
        cols: usize,
        titles: &[String],
    ) -> Result<(), softbuffer::SoftBufferError> {
        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate cell dimensions with padding (tight layout)
        let (padding, axes_len) = {
            let figure_ref = figure.borrow();
            let pad = if figure_ref.tight_layout { 5u32 } else { 10u32 };
            let len = figure_ref.axes.len() * (if rows > 0 && !figure_ref.axes.is_empty() { figure_ref.axes[0].len() } else { 0 });
            (pad, len)
        };
        let title_height = 20u32; // Space for title above each image
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        
        // Available space for images (excluding titles)
        let available_height = buffer_height - (rows_u32 + 1) * padding - rows_u32 * title_height;
        let available_width = buffer_width - (cols_u32 + 1) * padding;
        
        let cell_width = available_width / cols_u32;
        let cell_height = available_height / rows_u32;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let cell_width_usize = cell_width as usize;
        let cell_height_usize = cell_height as usize;
        let padding_usize = padding as usize;
        let title_height_usize = title_height as usize;

        // Draw each axis
        let mut idx = 0;
        for row_idx in 0..rows {
            for col_idx in 0..cols {
                if idx >= axes_len {
                    break;
                }

                let axis_rc = {
                    let figure_ref = figure.borrow();
                    figure_ref.axes[row_idx][col_idx].clone()
                };
                let axis_ref = axis_rc.borrow();

                // Calculate cell position
                let cell_x = padding_usize + col_idx * (cell_width_usize + padding_usize);
                let cell_y = padding_usize + row_idx * (cell_height_usize + padding_usize + title_height_usize);

                // Draw title if present
                if idx < titles.len() && !titles[idx].is_empty() {
                    let title = &titles[idx];
                    // Calculate text width using exact font metrics for proper centering
                    let base_font_size = 20.0; // Reduced from 32.0 for smaller text
                    let font_size = base_font_size * self.scale_factor;
                    let text_width = Self::calculate_text_width(self.font.as_ref(), title, font_size);
                    // Center text: (cell_width - text_width) / 2, rounded to integer
                    let text_x = (cell_x as f32 + (cell_width_usize as f32 - text_width) / 2.0).round() as usize;
                    // Calculate baseline using font metrics for proper text alignment
                    let text_y = if let Some(font) = self.font.as_ref() {
                        if let Some(line_metrics) = font.horizontal_line_metrics(font_size) {
                            // baseline_y = cell_y + ascent (distance from top of cell to baseline)
                            cell_y + line_metrics.ascent.round() as usize
                        } else {
                            cell_y + 14 // fallback if metrics unavailable
                        }
                    } else {
                        cell_y + 14 // fallback if no font loaded
                    };
                    let font_ref = self.font.as_ref();
                    let scale_factor = self.scale_factor;
                    Self::draw_text_improved(
                        &mut self.atlas,
                        font_ref,
                        &mut buffer,
                        text_x,
                        text_y,
                        title,
                        font_size,
                        scale_factor,
                        buffer_width_usize,
                        buffer_height_usize,
                    );
                }

                // Draw image if present
                if let Some(ref img_rc) = axis_ref.image {
                    let image = img_rc.borrow();
                    let img_width = image.width as usize;
                    let img_height = image.height as usize;

                    if img_width > 0 && img_height > 0 {
                        // Calculate scale to fit image in cell (maintain aspect ratio)
                        let image_area_width = cell_width_usize;
                        let image_area_height = cell_height_usize;
                        let scale_x = image_area_width as f32 / img_width as f32;
                        let scale_y = image_area_height as f32 / img_height as f32;
                        let scale = scale_x.min(scale_y).min(10.0); // Max scale 10x
                        let scale_usize = scale as usize;

                        // Calculate scaled dimensions
                        let scaled_width = (img_width as f32 * scale) as usize;
                        let scaled_height = (img_height as f32 * scale) as usize;

                        // Center image in cell (below title)
                        let image_y = cell_y + title_height_usize;
                        let offset_x = cell_x + (cell_width_usize - scaled_width) / 2;
                        let offset_y = image_y + (cell_height_usize - scaled_height) / 2;

                        // Draw image with nearest neighbor scaling
                        for src_y in 0..img_height {
                            for src_x in 0..img_width {
                                let image_idx = (src_y * img_width + src_x) * 4;
                                if image_idx + 3 < image.data.len() {
                                    let r = image.data[image_idx] as u32;
                                    let g = image.data[image_idx + 1] as u32;
                                    let b = image.data[image_idx + 2] as u32;
                                    let a = image.data[image_idx + 3] as u32;
                                    
                                    // Apply grayscale colormap if needed
                                    let (final_r, final_g, final_b) = if axis_ref.cmap == "gray" {
                                        // Convert to grayscale
                                        let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u32;
                                        (gray, gray, gray)
                                    } else {
                                        (r, g, b)
                                    };
                                    
                                    let pixel = (a << 24) | (final_b << 16) | (final_g << 8) | final_r;
                                    
                                    // Draw scale×scale block
                                    for dy in 0..scale_usize {
                                        for dx in 0..scale_usize {
                                            let dst_y = offset_y + src_y * scale_usize + dy;
                                            let dst_x = offset_x + src_x * scale_usize + dx;
                                            
                                            if dst_y < buffer_height_usize && dst_x < buffer_width_usize {
                                                let buffer_idx = dst_y * buffer_width_usize + dst_x;
                                                if buffer_idx < buffer.len() {
                                                    buffer[buffer_idx] = pixel;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                idx += 1;
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Draw figure with axes from FigureData (grid of images with titles)
    /// figure_data: extracted figure data (Send + Sync)
    /// rows: number of rows
    /// cols: number of columns
    /// titles: titles for each cell
    pub fn draw_figure_from_data(
        &mut self,
        figure_data: FigureData,
        rows: usize,
        cols: usize,
        titles: &[String],
    ) -> Result<(), softbuffer::SoftBufferError> {
        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate cell dimensions with padding (tight layout)
        let padding = if figure_data.tight_layout { 5u32 } else { 10u32 };
        let axes_len = figure_data.axes.len() * (if rows > 0 && !figure_data.axes.is_empty() { figure_data.axes[0].len() } else { 0 });
        let title_height = 20u32; // Space for title above each image
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        
        // Available space for images (excluding titles)
        let available_height = buffer_height - (rows_u32 + 1) * padding - rows_u32 * title_height;
        let available_width = buffer_width - (cols_u32 + 1) * padding;
        
        let cell_width = available_width / cols_u32;
        let cell_height = available_height / rows_u32;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let cell_width_usize = cell_width as usize;
        let cell_height_usize = cell_height as usize;
        let padding_usize = padding as usize;
        let title_height_usize = title_height as usize;

        // Draw each axis
        let mut idx = 0;
        for row_idx in 0..rows {
            for col_idx in 0..cols {
                if idx >= axes_len || row_idx >= figure_data.axes.len() || col_idx >= figure_data.axes[row_idx].len() {
                    break;
                }

                let axis_data = &figure_data.axes[row_idx][col_idx];

                // Calculate cell position
                let cell_x = padding_usize + col_idx * (cell_width_usize + padding_usize);
                let cell_y = padding_usize + row_idx * (cell_height_usize + padding_usize + title_height_usize);

                // Draw title if present
                if idx < titles.len() && !titles[idx].is_empty() {
                    let title = &titles[idx];
                    // Calculate text width using exact font metrics for proper centering
                    let base_font_size = 20.0; // Reduced from 32.0 for smaller text
                    let font_size = base_font_size * self.scale_factor;
                    let text_width = Self::calculate_text_width(self.font.as_ref(), title, font_size);
                    // Center text: (cell_width - text_width) / 2, rounded to integer
                    let text_x = (cell_x as f32 + (cell_width_usize as f32 - text_width) / 2.0).round() as usize;
                    // Calculate baseline using font metrics for proper text alignment
                    let text_y = if let Some(font) = self.font.as_ref() {
                        if let Some(line_metrics) = font.horizontal_line_metrics(font_size) {
                            // baseline_y = cell_y + ascent (distance from top of cell to baseline)
                            cell_y + line_metrics.ascent.round() as usize
                        } else {
                            cell_y + 14 // fallback if metrics unavailable
                        }
                    } else {
                        cell_y + 14 // fallback if no font loaded
                    };
                    let font_ref = self.font.as_ref();
                    let scale_factor = self.scale_factor;
                    Self::draw_text_improved(
                        &mut self.atlas,
                        font_ref,
                        &mut buffer,
                        text_x,
                        text_y,
                        title,
                        font_size,
                        scale_factor,
                        buffer_width_usize,
                        buffer_height_usize,
                    );
                }

                // Draw image if present
                if let Some(ref image_arc) = axis_data.image {
                    let image = image_arc.lock().unwrap();
                    let img_width = image.width as usize;
                    let img_height = image.height as usize;

                    if img_width > 0 && img_height > 0 {
                        // Calculate scale to fit image in cell (maintain aspect ratio)
                        let image_area_width = cell_width_usize;
                        let image_area_height = cell_height_usize;
                        let scale_x = image_area_width as f32 / img_width as f32;
                        let scale_y = image_area_height as f32 / img_height as f32;
                        let scale = scale_x.min(scale_y).min(10.0); // Max scale 10x
                        let scale_usize = scale as usize;

                        // Calculate scaled dimensions
                        let scaled_width = (img_width as f32 * scale) as usize;
                        let scaled_height = (img_height as f32 * scale) as usize;

                        // Center image in cell (below title)
                        let image_y = cell_y + title_height_usize;
                        let offset_x = cell_x + (cell_width_usize - scaled_width) / 2;
                        let offset_y = image_y + (cell_height_usize - scaled_height) / 2;

                        // Draw image with nearest neighbor scaling
                        for src_y in 0..img_height {
                            for src_x in 0..img_width {
                                let image_idx = (src_y * img_width + src_x) * 4;
                                if image_idx + 3 < image.data.len() {
                                    let r = image.data[image_idx] as u32;
                                    let g = image.data[image_idx + 1] as u32;
                                    let b = image.data[image_idx + 2] as u32;
                                    let a = image.data[image_idx + 3] as u32;
                                    
                                    // Apply grayscale colormap if needed
                                    let (final_r, final_g, final_b) = if axis_data.cmap == "gray" {
                                        // Convert to grayscale
                                        let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u32;
                                        (gray, gray, gray)
                                    } else {
                                        (r, g, b)
                                    };
                                    
                                    let pixel = (a << 24) | (final_b << 16) | (final_g << 8) | final_r;
                                    
                                    // Draw scale×scale block
                                    for dy in 0..scale_usize {
                                        for dx in 0..scale_usize {
                                            let dst_y = offset_y + src_y * scale_usize + dy;
                                            let dst_x = offset_x + src_x * scale_usize + dx;
                                            
                                            if dst_y < buffer_height_usize && dst_x < buffer_width_usize {
                                                let buffer_idx = dst_y * buffer_width_usize + dst_x;
                                                if buffer_idx < buffer.len() {
                                                    buffer[buffer_idx] = pixel;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                idx += 1;
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Find the nearest point to cursor position
    /// Returns the index of the nearest point if within threshold distance
    pub fn find_nearest_point(
        x_data: &[f64],
        y_data: &[f64],
        cursor_x: f32,
        cursor_y: f32,
        plot_x: usize,
        plot_y: usize,
        plot_width: usize,
        plot_height: usize,
        x_min_plot: f64,
        x_max_plot: f64,
        y_min_plot: f64,
        y_max_plot: f64,
        point_size: usize,
    ) -> Option<usize> {
        if x_data.is_empty() || y_data.is_empty() {
            return None;
        }

        let cursor_x_usize = cursor_x as usize;
        let cursor_y_usize = cursor_y as usize;

        // Check if cursor is within plot area
        if cursor_x_usize < plot_x || cursor_x_usize >= plot_x + plot_width ||
           cursor_y_usize < plot_y || cursor_y_usize >= plot_y + plot_height {
            return None;
        }

        let x_range_plot = x_max_plot - x_min_plot;
        let y_range_plot = y_max_plot - y_min_plot;

        // Helper function to convert data coordinates to screen coordinates
        let to_screen_x = |x: f64| -> usize {
            let normalized = (x - x_min_plot) / x_range_plot;
            plot_x + (normalized * plot_width as f64) as usize
        };

        let to_screen_y = |y: f64| -> usize {
            let normalized = (y - y_min_plot) / y_range_plot;
            plot_y + plot_height - (normalized * plot_height as f64) as usize
        };

        let threshold = (point_size + 10) as f32; // Threshold distance in pixels
        let mut min_distance = threshold;
        let mut nearest_index = None;

        for i in 0..x_data.len() {
            let point_x = to_screen_x(x_data[i]) as f32;
            let point_y = to_screen_y(y_data[i]) as f32;

            let dx = cursor_x - point_x;
            let dy = cursor_y - point_y;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance < min_distance {
                min_distance = distance;
                nearest_index = Some(i);
            }
        }

        nearest_index
    }

    /// Find the nearest point to cursor position across all lines
    /// Returns (line_index, point_index) if a point is found within threshold distance
    pub fn find_nearest_point_across_lines(
        lines: &[(Vec<f64>, Vec<f64>, bool, usize, usize, u32)],
        cursor_x: f32,
        cursor_y: f32,
        plot_x: usize,
        plot_y: usize,
        plot_width: usize,
        plot_height: usize,
        x_min_plot: f64,
        x_max_plot: f64,
        y_min_plot: f64,
        y_max_plot: f64,
    ) -> Option<(usize, usize)> {
        if lines.is_empty() {
            return None;
        }

        let cursor_x_usize = cursor_x as usize;
        let cursor_y_usize = cursor_y as usize;

        // Check if cursor is within plot area
        if cursor_x_usize < plot_x || cursor_x_usize >= plot_x + plot_width ||
           cursor_y_usize < plot_y || cursor_y_usize >= plot_y + plot_height {
            return None;
        }

        let x_range_plot = x_max_plot - x_min_plot;
        let y_range_plot = y_max_plot - y_min_plot;

        // Helper function to convert data coordinates to screen coordinates
        let to_screen_x = |x: f64| -> usize {
            let normalized = (x - x_min_plot) / x_range_plot;
            plot_x + (normalized * plot_width as f64) as usize
        };

        let to_screen_y = |y: f64| -> usize {
            let normalized = (y - y_min_plot) / y_range_plot;
            plot_y + plot_height - (normalized * plot_height as f64) as usize
        };

        let mut min_distance = f32::INFINITY;
        let mut nearest = None;

        // Search across all lines
        for (line_idx, (x_data, y_data, _, point_size, _, _)) in lines.iter().enumerate() {
            if x_data.is_empty() || y_data.is_empty() || x_data.len() != y_data.len() {
                continue;
            }

            // Use a reasonable threshold to detect points on hover
            // Threshold should be large enough to catch points but not too large to avoid false positives
            // Reduced to 15 pixels for more accurate detection
            let threshold = if *point_size == 0 {
                15.0 // Threshold when points are not normally shown
            } else {
                ((*point_size + 10) as f32).max(15.0).min(20.0) // Between 15 and 20 pixels
            };

            for point_idx in 0..x_data.len() {
                let point_x = to_screen_x(x_data[point_idx]) as f32;
                let point_y = to_screen_y(y_data[point_idx]) as f32;

                let dx = cursor_x - point_x;
                let dy = cursor_y - point_y;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance < threshold && distance < min_distance {
                    min_distance = distance;
                    nearest = Some((line_idx, point_idx));
                }
            }
        }

        nearest
    }

    /// Helper function to draw a line segment
    fn draw_line_segment(
        buffer: &mut [u32],
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        line_color: u32,
        line_width: usize,
        buffer_width: usize,
        buffer_height: usize,
    ) {
        if line_width == 1 {
            // Simple Bresenham's algorithm for single-pixel line
            let dx = (x2 as i32 - x1 as i32).abs();
            let dy = (y2 as i32 - y1 as i32).abs();
            let sx = if x1 < x2 { 1 } else { -1 };
            let sy = if y1 < y2 { 1 } else { -1 };
            let mut err = dx - dy;

            let mut x = x1 as i32;
            let mut y = y1 as i32;

            loop {
                if x >= 0 && x < buffer_width as i32 && y >= 0 && y < buffer_height as i32 {
                    let idx = (y as usize) * buffer_width + (x as usize);
                    if idx < buffer.len() {
                        buffer[idx] = line_color;
                    }
                }

                if x == x2 as i32 && y == y2 as i32 {
                    break;
                }

                let e2 = 2 * err;
                if e2 > -dy {
                    err -= dy;
                    x += sx;
                }
                if e2 < dx {
                    err += dx;
                    y += sy;
                }
            }
        } else {
            // Draw thick line by filling circles along the line path
            let dx = (x2 as i32 - x1 as i32).abs();
            let dy = (y2 as i32 - y1 as i32).abs();
            let sx = if x1 < x2 { 1 } else { -1 };
            let sy = if y1 < y2 { 1 } else { -1 };
            let mut err = dx - dy;

            let mut x = x1 as i32;
            let mut y = y1 as i32;
            let half_width = (line_width / 2) as i32;

            loop {
                // Draw a filled circle (or square) at each point along the line
                for dy_offset in -half_width..=half_width {
                    for dx_offset in -half_width..=half_width {
                        // Use square pattern for simplicity and better coverage
                        let px = x + dx_offset;
                        let py = y + dy_offset;
                        
                        if px >= 0 && px < buffer_width as i32 && py >= 0 && py < buffer_height as i32 {
                            let idx = (py as usize) * buffer_width + (px as usize);
                            if idx < buffer.len() {
                                buffer[idx] = line_color;
                            }
                        }
                    }
                }

                if x == x2 as i32 && y == y2 as i32 {
                    break;
                }

                let e2 = 2 * err;
                if e2 > -dy {
                    err -= dy;
                    x += sx;
                }
                if e2 < dx {
                    err += dx;
                    y += sy;
                }
            }
        }
    }

    /// Draw line chart with axes, grid, and labels
    /// lines: array of lines, each as (x_data, y_data, show_points, point_size, line_width, color)
    /// xlabel: optional x-axis label
    /// ylabel: optional y-axis label
    /// window: window reference
    /// cursor_pos: optional cursor position (x, y) in screen coordinates for hover tooltip
    /// hovered_point_index: optional index of point under cursor (for first line)
    /// selected_point_index: optional index of selected point (for first line)
    pub fn draw_line_chart(
        &mut self,
        lines: &[(Vec<f64>, Vec<f64>, bool, usize, usize, u32)],
        xlabel: Option<&str>,
        ylabel: Option<&str>,
        _window: &Window,
        cursor_pos: Option<(f32, f32)>,
        hovered_point_index: Option<(usize, usize)>, // (line_index, point_index)
        selected_point_index: Option<(usize, usize)>, // (line_index, point_index)
    ) -> Result<(), softbuffer::SoftBufferError> {
        if lines.is_empty() {
            return Ok(());
        }

        // Validate all lines have matching x and y lengths
        for (x_data, y_data, _, _, _, _) in lines {
            if x_data.is_empty() || y_data.is_empty() || x_data.len() != y_data.len() {
                return Ok(());
            }
        }

        // Save scale_factor before mutable borrow
        let scale_factor = self.scale_factor;

        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate plot area with adaptive margins for labels
        let (left_margin, right_margin, top_margin, bottom_margin) = 
            Self::calculate_adaptive_margins(scale_factor, buffer_width, buffer_height);
        let plot_width = buffer_width.saturating_sub(left_margin + right_margin);
        let plot_height = buffer_height.saturating_sub(top_margin + bottom_margin);
        let plot_x = left_margin;
        let plot_y = top_margin;

        // Calculate data bounds from all lines
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        
        for (x_data, y_data, _, _, _, _) in lines {
            for &x_val in x_data {
                x_min = x_min.min(x_val);
                x_max = x_max.max(x_val);
            }
            for &y_val in y_data {
                y_min = y_min.min(y_val);
                y_max = y_max.max(y_val);
            }
        }

        // Handle edge cases
        let x_range = if x_max > x_min { x_max - x_min } else { 1.0 };
        let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };

        // Add small padding to ranges
        let x_padding = x_range * 0.05;
        let y_padding = y_range * 0.05;
        let x_min_plot = x_min - x_padding;
        let x_max_plot = x_max + x_padding;
        let y_min_plot = y_min - y_padding;
        let y_max_plot = y_max + y_padding;
        let x_range_plot = x_max_plot - x_min_plot;
        let y_range_plot = y_max_plot - y_min_plot;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let plot_width_usize = plot_width as usize;
        let plot_height_usize = plot_height as usize;
        let plot_x_usize = plot_x as usize;
        let plot_y_usize = plot_y as usize;

        // Helper function to convert data coordinates to screen coordinates
        let to_screen_x = |x: f64| -> usize {
            let normalized = (x - x_min_plot) / x_range_plot;
            plot_x_usize + (normalized * plot_width_usize as f64) as usize
        };

        let to_screen_y = |y: f64| -> usize {
            let normalized = (y - y_min_plot) / y_range_plot;
            // Flip y-axis (screen y increases downward)
            plot_y_usize + plot_height_usize - (normalized * plot_height_usize as f64) as usize
        };

        // Draw grid lines (light gray)
        let grid_color = 0xFF404040; // Darker gray for grid
        let num_grid_lines_x = 5;
        let num_grid_lines_y = 5;

        // Vertical grid lines
        for i in 0..=num_grid_lines_x {
            let x_val = x_min_plot + (x_range_plot * i as f64 / num_grid_lines_x as f64);
            let x_screen = to_screen_x(x_val);
            if x_screen >= plot_x_usize && x_screen < plot_x_usize + plot_width_usize {
                for y_screen in plot_y_usize..plot_y_usize + plot_height_usize {
                    if y_screen < buffer_height_usize && x_screen < buffer_width_usize {
                        let idx = y_screen * buffer_width_usize + x_screen;
                        if idx < buffer.len() {
                            buffer[idx] = grid_color;
                        }
                    }
                }
            }
        }

        // Horizontal grid lines
        for i in 0..=num_grid_lines_y {
            let y_val = y_min_plot + (y_range_plot * i as f64 / num_grid_lines_y as f64);
            let y_screen = to_screen_y(y_val);
            if y_screen >= plot_y_usize && y_screen < plot_y_usize + plot_height_usize {
                for x_screen in plot_x_usize..plot_x_usize + plot_width_usize {
                    if y_screen < buffer_height_usize && x_screen < buffer_width_usize {
                        let idx = y_screen * buffer_width_usize + x_screen;
                        if idx < buffer.len() {
                            buffer[idx] = grid_color;
                        }
                    }
                }
            }
        }

        // Draw axes (white)
        let axis_color = 0xFFFFFFFF;
        // X-axis - draw at y=0 or at the bottom if y_max_plot < 0
        let x_axis_y_val = if 0.0 < y_max_plot { 0.0 } else { y_max_plot };
        let x_axis_y = to_screen_y(x_axis_y_val);
        for x_screen in plot_x_usize..plot_x_usize + plot_width_usize {
            if x_screen < buffer_width_usize && x_axis_y < buffer_height_usize {
                let idx = x_axis_y * buffer_width_usize + x_screen;
                if idx < buffer.len() {
                    buffer[idx] = axis_color;
                }
            }
        }
        // Y-axis - draw at x=0 or at the left if x_min_plot > 0
        let y_axis_x_val = if 0.0 > x_min_plot { 0.0 } else { x_min_plot };
        let y_axis_x = to_screen_x(y_axis_x_val);
        for y_screen in plot_y_usize..plot_y_usize + plot_height_usize {
            if y_screen < buffer_height_usize && y_axis_x < buffer_width_usize {
                let idx = y_screen * buffer_width_usize + y_axis_x;
                if idx < buffer.len() {
                    buffer[idx] = axis_color;
                }
            }
        }

        // Draw all lines first (so points will be drawn on top)
        for (_line_idx, (x_data, y_data, _, _, line_width, line_color)) in lines.iter().enumerate() {
            // Draw line segments
            for i in 0..x_data.len() - 1 {
                let x1 = to_screen_x(x_data[i]);
                let y1 = to_screen_y(y_data[i]);
                let x2 = to_screen_x(x_data[i + 1]);
                let y2 = to_screen_y(y_data[i + 1]);

                Self::draw_line_segment(
                    &mut buffer,
                    x1,
                    y1,
                    x2,
                    y2,
                    *line_color,
                    *line_width,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Draw all points after all lines (so they appear on top)
        let selected_color = 0xFFFFFF00; // Yellow for selected point
        let hovered_color = 0xFFFF69B4; // Hot pink for hovered point
        let hovered_outline_color = 0xFFFFFFFF; // White outline for hovered point
        
        for (line_idx, (x_data, y_data, show_points, point_size, _, line_color)) in lines.iter().enumerate() {
            for i in 0..x_data.len() {
                let x_center = to_screen_x(x_data[i]);
                let y_center = to_screen_y(y_data[i]);
                
                // Determine point size and color based on state (works for all lines)
                let is_hovered = hovered_point_index == Some((line_idx, i));
                let is_selected = selected_point_index == Some((line_idx, i));
                
                // Draw point if show_points is true OR if point is hovered/selected
                if *show_points || is_hovered || is_selected {
                    let current_point_size = if is_hovered || is_selected {
                        // Increase size significantly when hovered or selected (6x for better visibility)
                        // Use a minimum size of 10 pixels if point_size is 0 (when show_points=false)
                        let base_size = if *point_size == 0 { 10 } else { *point_size };
                        ((base_size as f32) * 6.0).max(20.0) as usize // At least 20 pixels when hovered
                    } else {
                        *point_size
                    };
                    
                    let current_color = if is_selected {
                        selected_color
                    } else if is_hovered {
                        hovered_color
                    } else {
                        *line_color // Use line color for points
                    };
                    
                    // Draw filled circle for each point
                    let point_radius_i32 = current_point_size as i32;
                    let radius_squared = point_radius_i32 * point_radius_i32;
                    
                    // Draw outline for hovered/selected points (slightly larger circle)
                    if is_hovered || is_selected {
                        let outline_radius = point_radius_i32 + 2;
                        let outline_radius_squared = outline_radius * outline_radius;
                        for dy in -outline_radius..=outline_radius {
                            for dx in -outline_radius..=outline_radius {
                                let dist_sq = dx * dx + dy * dy;
                                // Draw outline pixels (between outer and inner radius)
                                if dist_sq <= outline_radius_squared && dist_sq > radius_squared {
                                    let x = x_center as i32 + dx;
                                    let y = y_center as i32 + dy;
                                    
                                    if x >= 0 && x < buffer_width as i32 && y >= 0 && y < buffer_height as i32 {
                                        let idx = (y as usize) * buffer_width_usize + (x as usize);
                                        if idx < buffer.len() {
                                            buffer[idx] = hovered_outline_color;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Draw main point circle
                    for dy in -point_radius_i32..=point_radius_i32 {
                        for dx in -point_radius_i32..=point_radius_i32 {
                            if dx * dx + dy * dy <= radius_squared {
                                let x = x_center as i32 + dx;
                                let y = y_center as i32 + dy;
                                
                                if x >= 0 && x < buffer_width as i32 && y >= 0 && y < buffer_height as i32 {
                                    let idx = (y as usize) * buffer_width_usize + (x as usize);
                                    if idx < buffer.len() {
                                        buffer[idx] = current_color;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Draw axis labels (numbers)
        // Note: Using white color directly in draw_text_improved
        let font_size = 16.0 * self.scale_factor;

        // X-axis labels
        for i in 0..=num_grid_lines_x {
            let x_val = x_min_plot + (x_range_plot * i as f64 / num_grid_lines_x as f64);
            let x_screen = to_screen_x(x_val);
            let label = if x_val.fract() == 0.0 {
                format!("{}", x_val as i64)
            } else {
                format!("{:.2}", x_val)
            };
            let text_width = Self::calculate_text_width(self.font.as_ref(), &label, font_size);
            let text_x = (x_screen as f32 - text_width / 2.0).max(0.0) as usize;
            let text_y = x_axis_y + 20;
            if text_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    &label,
                    font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Y-axis labels
        for i in 0..=num_grid_lines_y {
            let y_val = y_min_plot + (y_range_plot * i as f64 / num_grid_lines_y as f64);
            let y_screen = to_screen_y(y_val);
            let label = if y_val.fract() == 0.0 {
                format!("{}", y_val as i64)
            } else {
                format!("{:.2}", y_val)
            };
            let text_width = Self::calculate_text_width(self.font.as_ref(), &label, font_size);
            let text_x = (plot_x_usize as f32 - text_width - 10.0).max(0.0) as usize;
            let text_y = y_screen;
            if text_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    &label,
                    font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Draw axis text labels (xlabel, ylabel)
        let label_font_size = 20.0 * self.scale_factor;

        // X-axis label
        if let Some(xlabel_text) = xlabel {
            let text_width = Self::calculate_text_width(self.font.as_ref(), xlabel_text, label_font_size);
            let text_x = plot_x_usize + (plot_width_usize as f32 - text_width) as usize / 2;
            let text_y = buffer_height_usize - 50; // Increased from 20 to 40 pixels from bottom
            if text_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    xlabel_text,
                    label_font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Y-axis label (rotated 90 degrees would be ideal, but for simplicity we'll draw it vertically)
        if let Some(ylabel_text) = ylabel {
            // Draw vertically by drawing each character
            let char_height = label_font_size as usize;
            let start_y = plot_y_usize + (plot_height_usize - ylabel_text.chars().count() * char_height) / 2;
            let mut current_y = start_y;
            for ch in ylabel_text.chars() {
                let char_str = ch.to_string();
                let text_width = Self::calculate_text_width(self.font.as_ref(), &char_str, label_font_size);
                let text_x = (40.0 - text_width / 2.0).max(0.0) as usize; // Increased to 80 pixels from left
                if current_y < buffer_height_usize {
                    Self::draw_text_improved(
                        &mut self.atlas,
                        self.font.as_ref(),
                        &mut buffer,
                        text_x,
                        current_y,
                        &char_str,
                        label_font_size,
                        self.scale_factor,
                        buffer_width_usize,
                        buffer_height_usize,
                    );
                }
                current_y += char_height;
            }
        }

        // Draw hover tooltip with x, y coordinates
        // Priority: selected point > hovered point > cursor position
        let tooltip_data = if let Some((line_idx, point_idx)) = selected_point_index {
            // Show selected point coordinates from the correct line
            if line_idx < lines.len() {
                let (x_data, y_data, _, _, _, _) = &lines[line_idx];
                if point_idx < x_data.len() {
                    Some((x_data[point_idx], y_data[point_idx]))
                } else {
                    None
                }
            } else {
                None
            }
        } else if let Some((line_idx, point_idx)) = hovered_point_index {
            // Show hovered point coordinates
            if line_idx < lines.len() {
                let (x_data, y_data, _, _, _, _) = &lines[line_idx];
                if point_idx < x_data.len() {
                    Some((x_data[point_idx], y_data[point_idx]))
                } else {
                    None
                }
            } else {
                None
            }
        } else if let Some((cursor_x, cursor_y)) = cursor_pos {
            let cursor_x_usize = cursor_x as usize;
            let cursor_y_usize = cursor_y as usize;
            
            // Check if cursor is within plot area
            if cursor_x_usize >= plot_x_usize && cursor_x_usize < plot_x_usize + plot_width_usize &&
               cursor_y_usize >= plot_y_usize && cursor_y_usize < plot_y_usize + plot_height_usize {
                
                // Convert screen coordinates to data coordinates
                let screen_x_rel = (cursor_x_usize - plot_x_usize) as f64;
                let screen_y_rel = (cursor_y_usize - plot_y_usize) as f64;
                
                let normalized_x = screen_x_rel / plot_width_usize as f64;
                let normalized_y = 1.0 - (screen_y_rel / plot_height_usize as f64); // Flip y-axis
                
                let data_x = x_min_plot + normalized_x * x_range_plot;
                let data_y = y_min_plot + normalized_y * y_range_plot;
                
                Some((data_x, data_y))
            } else {
                None
            }
        } else {
            None
        };
        
        if let Some((data_x, data_y)) = tooltip_data {
            // Format tooltip text with better precision
            let tooltip_text = if data_x.fract() == 0.0 && data_y.fract() == 0.0 {
                format!("x: {}, y: {}", data_x as i64, data_y as i64)
            } else {
                // Use more precision for better readability
                let x_str = if data_x.abs() >= 1000.0 || (data_x.abs() < 0.01 && data_x != 0.0) {
                    format!("{:.2e}", data_x)
                } else {
                    format!("{:.2}", data_x)
                };
                let y_str = if data_y.abs() >= 1000.0 || (data_y.abs() < 0.01 && data_y != 0.0) {
                    format!("{:.2e}", data_y)
                } else {
                    format!("{:.2}", data_y)
                };
                format!("x: {}, y: {}", x_str, y_str)
            };
            
            // Draw tooltip at the top of the plot area
            let tooltip_font_size = 16.0 * self.scale_factor;
            let text_width = Self::calculate_text_width(self.font.as_ref(), &tooltip_text, tooltip_font_size);
            
            // Center tooltip horizontally, position at top of plot area
            let tooltip_x = plot_x_usize + ((plot_width_usize as f32 - text_width) / 2.0).max(0.0) as usize;
            let tooltip_y = plot_y_usize + 10; // 10 pixels from top of plot area
            
            if tooltip_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    tooltip_x,
                    tooltip_y,
                    &tooltip_text,
                    tooltip_font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Draw bar chart with axes, grid, and labels
    /// bars: array of bar series, each as (x_labels, y_data, color)
    /// xlabel: optional x-axis label
    /// ylabel: optional y-axis label
    /// window: window reference
    pub fn draw_bar_chart(
        &mut self,
        bars: &[(Vec<String>, Vec<f64>, u32)],
        xlabel: Option<&str>,
        ylabel: Option<&str>,
        _window: &Window,
    ) -> Result<(), softbuffer::SoftBufferError> {
        if bars.is_empty() {
            return Ok(());
        }

        // Validate all bars have matching x_labels and y_data lengths
        for (x_labels, y_data, _) in bars {
            if x_labels.is_empty() || y_data.is_empty() || x_labels.len() != y_data.len() {
                return Ok(());
            }
        }

        // Save scale_factor before mutable borrow
        let scale_factor = self.scale_factor;

        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate plot area with adaptive margins for labels
        let (left_margin, right_margin, top_margin, bottom_margin) = 
            Self::calculate_adaptive_margins(scale_factor, buffer_width, buffer_height);
        // Increase bottom margin for bar charts to accommodate category labels
        let bottom_margin = bottom_margin.max((buffer_height as f32 * 0.1).max(90.0) as u32);
        let plot_width = buffer_width.saturating_sub(left_margin + right_margin);
        let plot_height = buffer_height.saturating_sub(top_margin + bottom_margin);
        let plot_x = left_margin;
        let plot_y = top_margin;

        // Calculate y bounds from all bars
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        
        for (_, y_data, _) in bars {
            for &y_val in y_data {
                y_min = y_min.min(y_val);
                y_max = y_max.max(y_val);
            }
        }

        // Handle edge cases
        let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };

        // Add small padding to y range
        let y_padding = y_range * 0.05;
        let y_min_plot = y_min - y_padding;
        let y_max_plot = y_max + y_padding;
        let y_range_plot = y_max_plot - y_min_plot;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let plot_width_usize = plot_width as usize;
        let plot_height_usize = plot_height as usize;
        let plot_x_usize = plot_x as usize;
        let plot_y_usize = plot_y as usize;

        // Helper function to convert y data coordinate to screen coordinate
        let to_screen_y = |y: f64| -> usize {
            let normalized = (y - y_min_plot) / y_range_plot;
            // Flip y-axis (screen y increases downward)
            plot_y_usize + plot_height_usize - (normalized * plot_height_usize as f64) as usize
        };

        // Get all unique x labels from all bar series (use first series for x positions)
        let (first_x_labels, _, _) = &bars[0];
        let num_categories = first_x_labels.len();

        // Calculate bar width and spacing (adaptive based on plot width)
        let bar_spacing = ((plot_width_usize as f32 * 0.01).max(8.0) * self.scale_factor).min(20.0) as usize; // Adaptive spacing
        let total_bar_width = plot_width_usize.saturating_sub((num_categories - 1) * bar_spacing);
        let bar_width = if num_categories > 0 {
            // Adaptive bar width: min 5px, max 10% of plot width per bar
            let max_bar_width = ((plot_width_usize as f32 / num_categories as f32) * 0.8).min(100.0) as usize;
            (total_bar_width / num_categories).max(5).min(max_bar_width)
        } else {
            20
        };
        let actual_bar_width = bar_width.min(total_bar_width / num_categories.max(1));

        // Draw grid lines (horizontal only for bar charts)
        let grid_color = 0xFF404040; // Darker gray for grid
        let num_grid_lines_y = 5;

        // Horizontal grid lines
        for i in 0..=num_grid_lines_y {
            let y_val = y_min_plot + (y_range_plot * i as f64 / num_grid_lines_y as f64);
            let y_screen = to_screen_y(y_val);
            if y_screen >= plot_y_usize && y_screen < plot_y_usize + plot_height_usize {
                for x_screen in plot_x_usize..plot_x_usize + plot_width_usize {
                    if y_screen < buffer_height_usize && x_screen < buffer_width_usize {
                        let idx = y_screen * buffer_width_usize + x_screen;
                        if idx < buffer.len() {
                            buffer[idx] = grid_color;
                        }
                    }
                }
            }
        }

        // Draw x-axis (at y=0 or at the bottom if y_max_plot < 0)
        let x_axis_y_val = if 0.0 < y_max_plot { 0.0 } else { y_max_plot };
        let x_axis_y = to_screen_y(x_axis_y_val);
        for x_screen in plot_x_usize..plot_x_usize + plot_width_usize {
            if x_screen < buffer_width_usize && x_axis_y < buffer_height_usize {
                let idx = x_axis_y * buffer_width_usize + x_screen;
                if idx < buffer.len() {
                    buffer[idx] = 0xFFFFFFFF; // White axis
                }
            }
        }

        // Draw y-axis (at left edge)
        let y_axis_x = plot_x_usize;
        for y_screen in plot_y_usize..plot_y_usize + plot_height_usize {
            if y_screen < buffer_height_usize && y_axis_x < buffer_width_usize {
                let idx = y_screen * buffer_width_usize + y_axis_x;
                if idx < buffer.len() {
                    buffer[idx] = 0xFFFFFFFF; // White axis
                }
            }
        }

        // Draw bars for each series
        for (x_labels, y_data, bar_color) in bars {
            // Calculate x position for each bar
            for (i, (_, &y_val)) in x_labels.iter().zip(y_data.iter()).enumerate() {
                // Calculate bar position (centered in its slot)
                let slot_width = if num_categories > 0 {
                    plot_width_usize / num_categories
                } else {
                    actual_bar_width + bar_spacing
                };
                let bar_x = plot_x_usize + i * slot_width + (slot_width - actual_bar_width) / 2;
                let bar_top = to_screen_y(y_val);
                let bar_bottom = to_screen_y(0.0);

                // Draw bar rectangle
                let bar_height = if bar_bottom > bar_top {
                    bar_bottom - bar_top
                } else {
                    bar_top - bar_bottom
                };

                for dy in 0..bar_height {
                    let y_pos = if bar_bottom > bar_top {
                        bar_top + dy
                    } else {
                        bar_bottom + dy
                    };
                    
                    if y_pos >= plot_y_usize && y_pos < plot_y_usize + plot_height_usize {
                        for dx in 0..actual_bar_width {
                            let x_pos = bar_x + dx;
                            if x_pos < plot_x_usize + plot_width_usize && x_pos < buffer_width_usize && y_pos < buffer_height_usize {
                                let idx = y_pos * buffer_width_usize + x_pos;
                                if idx < buffer.len() {
                                    buffer[idx] = *bar_color;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Draw value labels on top of bars
        let value_font_size = 16.0 * self.scale_factor;
        let bar_bottom_y = to_screen_y(0.0); // Y position of x-axis (y=0)
        for (x_labels, y_data, _) in bars {
            for (i, (_, &y_val)) in x_labels.iter().zip(y_data.iter()).enumerate() {
                // Calculate bar position (same as when drawing bars)
                let slot_width = if num_categories > 0 {
                    plot_width_usize / num_categories
                } else {
                    actual_bar_width + bar_spacing
                };
                let bar_x = plot_x_usize + i * slot_width + (slot_width - actual_bar_width) / 2;
                let bar_top = to_screen_y(y_val);
                
                // Format value label
                let value_label = if y_val.fract() == 0.0 {
                    format!("{}", y_val as i64)
                } else {
                    format!("{:.2}", y_val)
                };
                
                // Calculate text position (centered above bar)
                let text_width = Self::calculate_text_width(self.font.as_ref(), &value_label, value_font_size);
                let text_x = (bar_x as f32 + (actual_bar_width as f32 - text_width) / 2.0).max(plot_x_usize as f32) as usize;
                // Position text above bar top (15 pixels above)
                let text_y = if bar_bottom_y > bar_top {
                    bar_top.saturating_sub(15) // 15 pixels above bar top
                } else {
                    bar_top.saturating_sub(15) // For negative values, still above bar top
                };
                
                // Only draw if text is within plot area bounds
                if text_y >= plot_y_usize && text_y < plot_y_usize + plot_height_usize &&
                   text_x < plot_x_usize + plot_width_usize {
                    Self::draw_text_improved(
                        &mut self.atlas,
                        self.font.as_ref(),
                        &mut buffer,
                        text_x,
                        text_y,
                        &value_label,
                        value_font_size,
                        self.scale_factor,
                        buffer_width_usize,
                        buffer_height_usize,
                    );
                }
            }
        }

        // Draw y-axis labels (numbers)
        let font_size = 16.0 * self.scale_factor;
        for i in 0..=num_grid_lines_y {
            let y_val = y_min_plot + (y_range_plot * i as f64 / num_grid_lines_y as f64);
            let y_screen = to_screen_y(y_val);
            let label = if y_val.fract() == 0.0 {
                format!("{}", y_val as i64)
            } else {
                format!("{:.2}", y_val)
            };
            let text_width = Self::calculate_text_width(self.font.as_ref(), &label, font_size);
            let text_x = (plot_x_usize as f32 - text_width - 10.0).max(0.0) as usize;
            let text_y = y_screen;
            if text_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    &label,
                    font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Draw x-axis category labels (under bars)
        let label_font_size = 14.0 * self.scale_factor;
        let (first_x_labels, _, _) = &bars[0];
        for (i, label) in first_x_labels.iter().enumerate() {
            let slot_width = if num_categories > 0 {
                plot_width_usize / num_categories
            } else {
                actual_bar_width + bar_spacing
            };
            let label_x_center = plot_x_usize + i * slot_width + slot_width / 2;
            let text_width = Self::calculate_text_width(self.font.as_ref(), label, label_font_size);
            let text_x = (label_x_center as f32 - text_width / 2.0).max(plot_x_usize as f32) as usize;
            let text_y = plot_y_usize + plot_height_usize + 20; // 20 pixels below plot area
            
            // Truncate long labels if needed
            let display_label = if text_width > slot_width as f32 {
                // Try to truncate label
                let max_chars = (slot_width as f32 / (label_font_size * 0.6)) as usize;
                if label.chars().count() > max_chars {
                    let truncated: String = label.chars().take(max_chars.saturating_sub(3)).collect();
                    format!("{}...", truncated)
                } else {
                    label.clone()
                }
            } else {
                label.clone()
            };
            
            if text_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    &display_label,
                    label_font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Draw axis text labels (xlabel, ylabel)
        let axis_label_font_size = 20.0 * self.scale_factor;

        // X-axis label
        if let Some(xlabel_text) = xlabel {
            let text_width = Self::calculate_text_width(self.font.as_ref(), xlabel_text, axis_label_font_size);
            let text_x = plot_x_usize + (plot_width_usize as f32 - text_width) as usize / 2;
            let text_y = buffer_height_usize - 50;
            if text_y < buffer_height_usize {
                Self::draw_text_improved(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    xlabel_text,
                    axis_label_font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                );
            }
        }

        // Y-axis label (rotated vertically)
        if let Some(ylabel_text) = ylabel {
            let char_height = axis_label_font_size as usize;
            let start_y = plot_y_usize + (plot_height_usize - ylabel_text.chars().count() * char_height) / 2;
            let mut current_y = start_y;
            for ch in ylabel_text.chars() {
                let char_str = ch.to_string();
                let text_width = Self::calculate_text_width(self.font.as_ref(), &char_str, axis_label_font_size);
                let text_x = (40.0 - text_width / 2.0).max(0.0) as usize;
                if current_y < buffer_height_usize {
                    Self::draw_text_improved(
                        &mut self.atlas,
                        self.font.as_ref(),
                        &mut buffer,
                        text_x,
                        current_y,
                        &char_str,
                        axis_label_font_size,
                        self.scale_factor,
                        buffer_width_usize,
                        buffer_height_usize,
                    );
                }
                current_y += char_height;
            }
        }

        buffer.present()?;
        Ok(())
    }

    /// Convert HSL to RGB
    /// h: hue in degrees [0, 360)
    /// s: saturation [0, 1]
    /// l: lightness [0, 1]
    /// Returns (r, g, b) values in range [0, 255]
    fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
        let h = h % 360.0;
        let s = s.clamp(0.0, 1.0);
        let l = l.clamp(0.0, 1.0);

        if s == 0.0 {
            // Grayscale
            let gray = (l * 255.0).round() as u8;
            return (gray, gray, gray);
        }

        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = l - c / 2.0;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        (
            ((r + m) * 255.0).round().clamp(0.0, 255.0) as u8,
            ((g + m) * 255.0).round().clamp(0.0, 255.0) as u8,
            ((b + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        )
    }

    /// Convert RGB to BGRA format (0xAABBGGRR)
    /// r, g, b: color components [0, 255]
    /// Returns u32 in BGRA format with alpha = 0xFF
    fn rgb_to_bgra(r: u8, g: u8, b: u8) -> u32 {
        (0xFF << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32)
    }

    /// Generate colors for pie chart segments
    /// num_segments: number of segments in the pie chart
    /// Returns vector of colors in BGRA format
    fn generate_pie_colors(num_segments: usize) -> Vec<u32> {
        // d3 Category10 palette (excellent for up to 10 segments)
        // Colors are in RGB format, convert to BGRA
        const D3_CATEGORY10_RGB: [(u8, u8, u8); 10] = [
            (0x1F, 0x77, 0xB4), // Blue
            (0xFF, 0x7F, 0x0E), // Orange
            (0x2C, 0xA0, 0x2C), // Green
            (0xD6, 0x27, 0x28), // Red
            (0x94, 0x67, 0xBD), // Purple
            (0x8C, 0x56, 0x4B), // Brown
            (0xE3, 0x77, 0xC2), // Pink
            (0x7F, 0x7F, 0x7F), // Gray
            (0xBC, 0xBD, 0x22), // Olive
            (0x17, 0xBE, 0xCF), // Cyan
        ];

        // For small number of segments, use fixed palette
        if num_segments <= 12 {
            let mut colors = Vec::with_capacity(num_segments);
            for i in 0..num_segments {
                let (r, g, b) = D3_CATEGORY10_RGB[i % D3_CATEGORY10_RGB.len()];
                colors.push(Self::rgb_to_bgra(r, g, b));
            }
            return colors;
        }

        // For more segments, generate HSL colors
        let mut colors = Vec::with_capacity(num_segments);
        let saturation = 0.7; // Good for dark background
        let base_lightness = 0.65; // Good for dark background

        for i in 0..num_segments {
            // Uniform hue distribution
            let hue = (360.0 / num_segments as f64) * i as f64;

            // For very large number of segments, add lightness variation
            let lightness = if num_segments > 30 {
                let level = i as f64 / num_segments as f64;
                0.55 + 0.15 * (level * std::f64::consts::PI * 2.0).sin()
            } else {
                base_lightness
            };

            let (r, g, b) = Self::hsl_to_rgb(hue, saturation, lightness);
            colors.push(Self::rgb_to_bgra(r, g, b));
        }

        colors
    }

    /// Get contrasting text color for a segment
    /// color: segment color in BGRA format
    /// Returns white (0xFFFFFFFF) or black (0xFF000000) based on lightness
    fn get_text_color_for_segment(color: u32) -> u32 {
        // Extract RGB components from BGRA format
        let r = ((color >> 0) & 0xFF) as f64;
        let g = ((color >> 8) & 0xFF) as f64;
        let b = ((color >> 16) & 0xFF) as f64;

        // Calculate relative luminance (perceived brightness)
        // Using standard formula: 0.299*R + 0.587*G + 0.114*B
        let luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

        // If luminance > 0.6, use black text, otherwise white
        if luminance > 0.6 {
            0xFF000000 // Black
        } else {
            0xFFFFFFFF // White
        }
    }

    /// Draw pie chart with labels and percentages
    /// pies: array of pie data, each as (x_labels, y_data, color)
    /// xlabel: optional x-axis label (not used for pie charts, but kept for consistency)
    /// ylabel: optional y-axis label (not used for pie charts, but kept for consistency)
    /// window: window reference
    /// rotation: rotation angle in radians (0 = right, π/2 = top, π = left, 3π/2 = bottom)
    pub fn draw_pie_chart(
        &mut self,
        pies: &[(Vec<String>, Vec<f64>, u32)],
        _xlabel: Option<&str>,
        _ylabel: Option<&str>,
        _window: &Window,
        rotation: f64,
    ) -> Result<(), softbuffer::SoftBufferError> {
        if pies.is_empty() {
            return Ok(());
        }

        // Validate all pies have matching x_labels and y_data lengths
        for (x_labels, y_data, _) in pies {
            if x_labels.is_empty() || y_data.is_empty() || x_labels.len() != y_data.len() {
                return Ok(());
            }
        }

        // Save scale_factor before mutable borrow
        let scale_factor = self.scale_factor;

        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Use first pie data for rendering (pie charts typically show one dataset)
        let (x_labels, y_data, _base_color) = &pies[0];

        // Calculate total sum for normalization
        let total: f64 = y_data.iter().sum();
        if total == 0.0 {
            return Ok(());
        }

        // Calculate plot area with adaptive margins (pie charts need less margin)
        let left_margin = ((buffer_width as f32 * 0.03).max(30.0) * scale_factor) as u32;
        let right_margin = ((buffer_width as f32 * 0.03).max(30.0) * scale_factor) as u32;
        let top_margin = ((buffer_height as f32 * 0.05).max(50.0) * scale_factor) as u32;
        let bottom_margin = ((buffer_height as f32 * 0.1).max(90.0) * scale_factor) as u32; // Space for legend
        let plot_width = buffer_width.saturating_sub(left_margin + right_margin);
        let plot_height = buffer_height.saturating_sub(top_margin + bottom_margin);
        let plot_x = left_margin;
        let plot_y = top_margin;

        // Calculate center and radius
        let center_x = plot_x + plot_width / 2;
        let center_y = plot_y + plot_height / 2;
        let radius = (plot_width.min(plot_height) / 2 - 20).max(50); // At least 50px radius

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let center_x_usize = center_x as usize;
        let center_y_usize = center_y as usize;
        let radius_usize = radius as usize;

        // Generate colors based on number of segments
        let colors = Self::generate_pie_colors(y_data.len());

        // Draw pie slices
        // Start angle is rotation (default -π/2 = top, but can be rotated)
        let mut current_angle = rotation;
        
        for (i, &value) in y_data.iter().enumerate() {
            if value <= 0.0 {
                continue; // Skip zero or negative values
            }

            // Calculate slice angle
            let slice_angle = (value / total) * 2.0 * std::f64::consts::PI;
            let end_angle = current_angle + slice_angle;

            // Get color for this slice
            let slice_color = if i < colors.len() {
                colors[i]
            } else {
                // Fallback (shouldn't happen, but just in case)
                colors[i % colors.len()]
            };

            // Draw pie slice by filling pixels within the slice
            // Use a scanline approach: for each y, find x range within the slice
            for y_offset in -(radius_usize as i32)..=radius_usize as i32 {
                for x_offset in -(radius_usize as i32)..=radius_usize as i32 {
                    let distance_squared = (x_offset * x_offset + y_offset * y_offset) as f64;
                    if distance_squared > (radius_usize * radius_usize) as f64 {
                        continue; // Outside circle
                    }

                    // Calculate angle for this pixel
                    // atan2(y, x) returns angle in [-π, π] range
                    let pixel_angle_raw = (y_offset as f64).atan2(x_offset as f64);
                    
                    // Normalize to [0, 2π) range
                    let pixel_angle = if pixel_angle_raw < 0.0 {
                        pixel_angle_raw + 2.0 * std::f64::consts::PI
                    } else {
                        pixel_angle_raw
                    };
                    
                    // Normalize start and end angles to [0, 2π) range
                    let normalized_start = if current_angle < 0.0 {
                        current_angle + 2.0 * std::f64::consts::PI
                    } else if current_angle >= 2.0 * std::f64::consts::PI {
                        current_angle - 2.0 * std::f64::consts::PI
                    } else {
                        current_angle
                    };
                    
                    let normalized_end = if end_angle < 0.0 {
                        end_angle + 2.0 * std::f64::consts::PI
                    } else if end_angle >= 2.0 * std::f64::consts::PI {
                        end_angle - 2.0 * std::f64::consts::PI
                    } else {
                        end_angle
                    };

                    // Check if pixel is within slice
                    // Handle wrap-around case (when slice crosses 0/2π boundary)
                    let in_slice = if normalized_end >= normalized_start {
                        // Normal case: slice doesn't wrap around
                        pixel_angle >= normalized_start && pixel_angle <= normalized_end
                    } else {
                        // Wrap-around case: slice crosses 0/2π boundary
                        pixel_angle >= normalized_start || pixel_angle <= normalized_end
                    };

                    if in_slice {
                        let x = center_x_usize as i32 + x_offset;
                        let y = center_y_usize as i32 + y_offset;
                        
                        if x >= 0 && x < buffer_width as i32 && y >= 0 && y < buffer_height as i32 {
                            let idx = (y as usize) * buffer_width_usize + (x as usize);
                            if idx < buffer.len() {
                                buffer[idx] = slice_color;
                            }
                        }
                    }
                }
            }

            // Draw slice border (white line)
            // Draw line from center to edge at start angle
            let start_x = center_x_usize as i32 + (radius_usize as f64 * current_angle.cos()) as i32;
            let start_y = center_y_usize as i32 + (radius_usize as f64 * current_angle.sin()) as i32;
            Self::draw_line_segment(
                &mut buffer,
                center_x_usize,
                center_y_usize,
                start_x.max(0) as usize,
                start_y.max(0) as usize,
                0xFFFFFFFF, // White border
                2,
                buffer_width_usize,
                buffer_height_usize,
            );

            // Draw line from center to edge at end angle
            let end_x = center_x_usize as i32 + (radius_usize as f64 * end_angle.cos()) as i32;
            let end_y = center_y_usize as i32 + (radius_usize as f64 * end_angle.sin()) as i32;
            Self::draw_line_segment(
                &mut buffer,
                center_x_usize,
                center_y_usize,
                end_x.max(0) as usize,
                end_y.max(0) as usize,
                0xFFFFFFFF, // White border
                2,
                buffer_width_usize,
                buffer_height_usize,
            );

            // Draw arc for slice edge
            let num_arc_points = (slice_angle * radius_usize as f64 / 2.0) as usize;
            for j in 0..=num_arc_points {
                let angle = current_angle + (slice_angle * j as f64 / num_arc_points as f64);
                let arc_x = center_x_usize as i32 + (radius_usize as f64 * angle.cos()) as i32;
                let arc_y = center_y_usize as i32 + (radius_usize as f64 * angle.sin()) as i32;
                
                if arc_x >= 0 && arc_x < buffer_width as i32 && arc_y >= 0 && arc_y < buffer_height as i32 {
                    let idx = (arc_y as usize) * buffer_width_usize + (arc_x as usize);
                    if idx < buffer.len() {
                        buffer[idx] = 0xFFFFFFFF; // White border
                    }
                    // Draw thicker border
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let px = arc_x + dx;
                            let py = arc_y + dy;
                            if px >= 0 && px < buffer_width as i32 && py >= 0 && py < buffer_height as i32 {
                                let idx = (py as usize) * buffer_width_usize + (px as usize);
                                if idx < buffer.len() {
                                    buffer[idx] = 0xFFFFFFFF; // White border
                                }
                            }
                        }
                    }
                }
            }

            // Calculate label position (on the edge of the slice, halfway through the angle)
            let label_angle = current_angle + slice_angle / 2.0;
            let label_distance = radius_usize as f64 * 1.15; // Slightly outside the pie
            let label_x = center_x_usize as f32 + (label_distance * label_angle.cos()) as f32;
            let label_y = center_y_usize as f32 + (label_distance * label_angle.sin()) as f32;

            // Calculate percentage
            let percentage = (value / total) * 100.0;
            let label_text = if percentage >= 1.0 {
                format!("{} ({:.1}%)", x_labels[i], percentage)
            } else {
                format!("{} ({:.2}%)", x_labels[i], percentage)
            };

            // Draw label with contrasting color
            let label_font_size = 14.0 * self.scale_factor;
            let text_width = Self::calculate_text_width(self.font.as_ref(), &label_text, label_font_size);
            let text_x = (label_x - text_width / 2.0).max(0.0) as usize;
            let text_y = label_y as usize;

            // Get contrasting text color based on segment color
            let text_color = Self::get_text_color_for_segment(slice_color);

            if text_y < buffer_height_usize && text_x < buffer_width_usize {
                // Draw text shadow/outline for better visibility (black outline)
                let outline_color = 0xFF000000; // Black outline
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx != 0 || dy != 0 {
                            // Calculate outline position with bounds checking
                            let outline_x = if dx < 0 {
                                text_x.saturating_sub((-dx) as usize)
                            } else {
                                text_x.saturating_add(dx as usize)
                            };
                            let outline_y = if dy < 0 {
                                text_y.saturating_sub((-dy) as usize)
                            } else {
                                text_y.saturating_add(dy as usize)
                            };
                            
                            // Only draw if within bounds
                            if outline_x < buffer_width_usize && outline_y < buffer_height_usize {
                                Self::draw_text_improved_with_color(
                                    &mut self.atlas,
                                    self.font.as_ref(),
                                    &mut buffer,
                                    outline_x,
                                    outline_y,
                                    &label_text,
                                    label_font_size,
                                    self.scale_factor,
                                    buffer_width_usize,
                                    buffer_height_usize,
                                    Some(outline_color),
                                );
                            }
                        }
                    }
                }
                
                // Draw main text with contrasting color
                Self::draw_text_improved_with_color(
                    &mut self.atlas,
                    self.font.as_ref(),
                    &mut buffer,
                    text_x,
                    text_y,
                    &label_text,
                    label_font_size,
                    self.scale_factor,
                    buffer_width_usize,
                    buffer_height_usize,
                    Some(text_color),
                );
            }

            current_angle = end_angle;
        }

        buffer.present()?;
        Ok(())
    }

    /// Load font - try monospace fonts first for uniform character width, then proportional fonts as fallback
    /// Monospace fonts ensure all characters have the same width for better text alignment
    fn load_font() -> Option<Font> {
        // Optimal font settings for quality rendering
        // Note: fontdue 0.7 doesn't have explicit hinting support in FontSettings
        // Hinting is handled automatically by the font rasterizer based on the scale
        // Using scale: 40.0 (default) provides optimal hinting for most use cases
        let font_settings = fontdue::FontSettings {
            scale: 40.0, // Default scale provides optimal hinting
            collection_index: 0,
        };
        
        // Try to load system monospace fonts first (for uniform character width)
        // Monospace fonts ensure all symbols have the same width
        
        // Try system monospace fonts on macOS
        #[cfg(target_os = "macos")]
        {
            let monospace_font_paths = [
                "/System/Library/Fonts/Monaco.ttf",
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Courier New.ttf",
                "/Library/Fonts/Courier New.ttf",
                "/System/Library/Fonts/Supplemental/Courier New.ttf",
            ];
            
            for path in &monospace_font_paths {
                if let Ok(font_data) = std::fs::read(path) {
                    if let Ok(font) = Font::from_bytes(font_data, font_settings) {
                        return Some(font);
                    }
                }
            }
        }
        
        // Try system monospace fonts on Linux
        #[cfg(target_os = "linux")]
        {
            let monospace_font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
                "/usr/share/fonts/truetype/courier/Courier New.ttf",
            ];
            
            for path in &monospace_font_paths {
                if let Ok(font_data) = std::fs::read(path) {
                    if let Ok(font) = Font::from_bytes(font_data, font_settings) {
                        return Some(font);
                    }
                }
            }
        }
        
        // Try system monospace fonts on Windows
        #[cfg(target_os = "windows")]
        {
            use std::env;
            if let Ok(windows_dir) = env::var("WINDIR") {
                let monospace_font_paths = [
                    format!("{}\\Fonts\\consola.ttf", windows_dir),
                    format!("{}\\Fonts\\cour.ttf", windows_dir),
                    format!("{}\\Fonts\\courbd.ttf", windows_dir),
                    format!("{}\\Fonts\\lucon.ttf", windows_dir),
                    format!("{}\\Fonts\\consolab.ttf", windows_dir),
                ];
                
                for path in &monospace_font_paths {
                    if let Ok(font_data) = std::fs::read(path) {
                        if let Ok(font) = Font::from_bytes(font_data, font_settings) {
                            return Some(font);
                        }
                    }
                }
            }
        }
        
        // Fallback to proportional fonts if monospace fonts are not available
        // Try to load embedded Inter font (supports Russian and English)
        // Inter is a modern, high-quality font optimized for UI
        const INTER_FONT: &[u8] = include_bytes!("fonts/Inter/static/Inter_18pt-Regular.ttf");
        if let Ok(font) = Font::from_bytes(INTER_FONT, font_settings) {
            return Some(font);
        }
        
        // Fallback to Inter variable font if static version not available
        const INTER_VARIABLE_FONT: &[u8] = include_bytes!("fonts/Inter/Inter-VariableFont_opsz,wght.ttf");
        if let Ok(font) = Font::from_bytes(INTER_VARIABLE_FONT, font_settings) {
            return Some(font);
        }
        
        // Try Roboto as second choice (also supports Russian and English)
        const ROBOTO_FONT: &[u8] = include_bytes!("fonts/roboto/static/Roboto-Regular.ttf");
        if let Ok(font) = Font::from_bytes(ROBOTO_FONT, font_settings) {
            return Some(font);
        }
        
        // Fallback to Roboto variable font
        const ROBOTO_VARIABLE_FONT: &[u8] = include_bytes!("fonts/roboto/Roboto-VariableFont_wdth,wght.ttf");
        if let Ok(font) = Font::from_bytes(ROBOTO_VARIABLE_FONT, font_settings) {
            return Some(font);
        }
        
        // Try to load system proportional fonts as last resort
        
        // Try system fonts on macOS
        #[cfg(target_os = "macos")]
        {
            // Try SF Pro (San Francisco) - macOS system font
            let font_paths = [
                "/System/Library/Fonts/Supplemental/SF-Pro-Text-Regular.otf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/HelveticaNeue.ttc",
            ];
            
            for path in &font_paths {
                if let Ok(font_data) = std::fs::read(path) {
                    if let Ok(font) = Font::from_bytes(font_data, font_settings) {
                        return Some(font);
                    }
                }
            }
        }
        
        // Try system fonts on Linux
        #[cfg(target_os = "linux")]
        {
            let font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            ];
            
            for path in &font_paths {
                if let Ok(font_data) = std::fs::read(path) {
                    if let Ok(font) = Font::from_bytes(font_data, font_settings) {
                        return Some(font);
                    }
                }
            }
        }
        
        // Try system fonts on Windows
        #[cfg(target_os = "windows")]
        {
            use std::env;
            if let Ok(windows_dir) = env::var("WINDIR") {
                let font_paths = [
                    format!("{}\\Fonts\\segoeui.ttf", windows_dir),
                    format!("{}\\Fonts\\arial.ttf", windows_dir),
                    format!("{}\\Fonts\\calibri.ttf", windows_dir),
                ];
                
                for path in &font_paths {
                    if let Ok(font_data) = std::fs::read(path) {
                        if let Ok(font) = Font::from_bytes(font_data, font_settings) {
                            return Some(font);
                        }
                    }
                }
            }
        }
        
        // Fallback: return None to use bitmap font
        None
    }
    
    /// Calculate exact text width using font metrics
    /// For monospace fonts, uses maximum character width for accurate calculation
    /// Returns the actual pixel width of the text
    fn calculate_text_width(font: Option<&Font>, text: &str, font_size: f32) -> f32 {
        if let Some(font) = font {
            // Calculate fixed character width using maximum width from sample characters
            // This ensures accurate width calculation matching the rendering
            let char_width = Self::calculate_max_char_width(font, font_size);
            
            // Use fixed width for all characters to ensure uniform spacing
            text.chars().count() as f32 * char_width
        } else {
            // Fallback estimation for bitmap font
            let avg_char_width = font_size * 0.6;
            text.chars().count() as f32 * avg_char_width
        }
    }
    
    
    /// Optimized integer-based alpha blending
    /// bg: background pixel in BGRA format (0xAABBGGRR)
    /// fg: foreground color components (r, g, b)
    /// alpha: alpha value (0-255)
    /// Returns blended pixel in BGRA format
    fn blend(bg: u32, fg_r: u8, fg_g: u8, fg_b: u8, alpha: u8) -> u32 {
        let a = alpha as u32;
        let inv_a = 255 - a;
        
        // Extract background components (BGRA format: 0xAABBGGRR)
        let bg_r = (bg >> 0) & 0xFF;
        let bg_g = (bg >> 8) & 0xFF;
        let bg_b = (bg >> 16) & 0xFF;
        
        // Integer blending: (fg * a + bg * inv_a) / 255
        let r = ((fg_r as u32 * a + bg_r * inv_a) / 255) as u8;
        let g = ((fg_g as u32 * a + bg_g * inv_a) / 255) as u8;
        let b = ((fg_b as u32 * a + bg_b * inv_a) / 255) as u8;
        
        // Format: 0xAABBGGRR
        (0xFF << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32)
    }
    
    /// Improved text rendering with fontdue for proper UTF-8 and Cyrillic support
    fn draw_text_improved(
        atlas: &mut FontAtlas,
        font: Option<&Font>,
        buffer: &mut [u32],
        x: usize,
        y: usize,
        text: &str,
        font_size: f32,
        scale_factor: f32,
        buffer_width: usize,
        buffer_height: usize,
    ) {
        Self::draw_text_improved_with_color(
            atlas,
            font,
            buffer,
            x,
            y,
            text,
            font_size,
            scale_factor,
            buffer_width,
            buffer_height,
            None, // Default to white
        );
    }

    fn draw_text_improved_with_color(
        atlas: &mut FontAtlas,
        font: Option<&Font>,
        buffer: &mut [u32],
        x: usize,
        y: usize,
        text: &str,
        font_size: f32,
        scale_factor: f32,
        buffer_width: usize,
        buffer_height: usize,
        text_color: Option<u32>, // BGRA format, None = white
    ) {
        if let Some(font) = font {
            // Use fontdue for proper text rendering with atlas caching
            Self::draw_text_with_font(
                font,
                atlas,
                buffer,
                x,
                y,
                text,
                font_size,
                scale_factor,
                buffer_width,
                buffer_height,
                text_color,
            );
        } else {
            // Fallback to improved bitmap font
            Self::draw_text_bitmap(
                buffer,
                x,
                y,
                text,
                buffer_width,
                buffer_height,
            );
        }
    }
    
    /// Calculate maximum character width from sample characters
    /// This ensures uniform spacing for all characters
    fn calculate_max_char_width(font: &Font, font_size: f32) -> f32 {
        // Test characters: digits, uppercase, lowercase, space, and common symbols
        let test_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz МеткаPredict:";
        let mut max_width = 0.0f32;
        
        for ch in test_chars.chars() {
            let (metrics, _) = font.rasterize(ch, font_size);
            max_width = max_width.max(metrics.advance_width);
        }
        
        // Ensure minimum width (fallback if all characters fail)
        if max_width <= 0.0 {
            let (space_metrics, _) = font.rasterize(' ', font_size);
            max_width = space_metrics.advance_width;
        }
        
        max_width
    }
    
    /// Draw text using fontdue font with DPI-aware rendering and atlas caching
    /// For monospace fonts, uses fixed character width for uniform spacing
    /// Uses proper fontdue coordinate model: baseline is a property of the line, not individual glyphs
    /// Position formula: glyph_y = baseline_y + metrics.ymin (no normalization)
    fn draw_text_with_font(
        font: &Font,
        atlas: &mut FontAtlas,
        buffer: &mut [u32],
        x: usize,
        y: usize,
        text: &str,
        font_size: f32,
        _scale_factor: f32,
        buffer_width: usize,
        buffer_height: usize,
        text_color: Option<u32>, // BGRA format, None = white
    ) {
        // Use DPI-aware font size for rasterization
        let rasterize_size = font_size;
        
        // Calculate fixed character width using maximum width from sample characters
        // This ensures all characters occupy the same horizontal space
        let char_width = Self::calculate_max_char_width(font, rasterize_size);
        
        // In fontdue, y=0 is the baseline
        // The parameter 'y' represents the baseline position in buffer coordinates
        // Each glyph's ymin is relative to this baseline
        // Use integer baseline to prevent "sinking" of characters
        let baseline_y = (y as f32).floor();
        
        // Round starting coordinates to avoid sub-pixel positioning (blur)
        // Note: x and y are already in buffer pixel coordinates, scale_factor is already in font_size
        let base_x = x as f32;
        let mut current_x = base_x;
        
        // Pre-calculate minimum xmin for all characters in the text to ensure uniform alignment
        // This ensures all characters align consistently regardless of their individual xmin values
        // Use atlas to avoid re-rasterizing characters
        let min_xmin = text.chars()
            .map(|ch| {
                let (metrics, _) = atlas.get_or_rasterize(ch, rasterize_size, |c, size| {
                    font.rasterize(c, size)
                });
                metrics.xmin as f32
            })
            .fold(0.0f32, |acc, xmin| acc.min(xmin));
        
        for ch in text.chars() {
            // Use atlas to cache glyphs
            let (metrics, bitmap) = atlas.get_or_rasterize(ch, rasterize_size, |c, size| {
                font.rasterize(c, size)
            });
            
            // Calculate glyph position: normalize xmin to ensure uniform alignment
            // Subtract min_xmin from each character's xmin to align all characters consistently
            // This ensures all characters start at the same visual position within their cells
            let normalized_xmin = metrics.xmin as f32 - min_xmin;
            let glyph_x = (current_x + normalized_xmin).round() as i32;
            
            // Proper fontdue coordinate model:
            // - baseline_y is the fixed baseline position for the entire line
            // - metrics.ymin is the glyph's bounding box minimum relative to baseline (can be negative for descenders)
            // - No normalization or offset calculation - just baseline + ymin
            let glyph_y = (baseline_y + metrics.ymin as f32).round() as i32;
            
            // Draw the character bitmap
            if metrics.width > 0 {
                for (row_idx, row) in bitmap.chunks(metrics.width).enumerate() {
                    let pixel_y = glyph_y.saturating_add(row_idx as i32);
                    if pixel_y < 0 || pixel_y as usize >= buffer_height {
                        continue;
                    }
                    let pixel_y = pixel_y as usize;
                    
                    for (col_idx, &alpha) in row.iter().enumerate() {
                        let pixel_x = glyph_x.saturating_add(col_idx as i32);
                        if pixel_x < 0 || pixel_x as usize >= buffer_width {
                            continue;
                        }
                        
                        if alpha > 0 {
                            // Blend text color with background using optimized integer blending
                            let idx = pixel_y * buffer_width + pixel_x as usize;
                            if idx < buffer.len() {
                                let current_pixel = buffer[idx];
                                // Extract RGB from text_color (BGRA format) or use white
                                let (fg_r, fg_g, fg_b) = if let Some(color) = text_color {
                                    (
                                        ((color >> 0) & 0xFF) as u8,
                                        ((color >> 8) & 0xFF) as u8,
                                        ((color >> 16) & 0xFF) as u8,
                                    )
                                } else {
                                    (255, 255, 255) // Default white
                                };
                                buffer[idx] = Self::blend(current_pixel, fg_r, fg_g, fg_b, alpha);
                            }
                        }
                    }
                }
            }
            
            // Advance to next character position using fixed width for uniform spacing
            // Always use fixed width regardless of individual character advance_width
            current_x += char_width;
            
            if current_x.round() as usize >= buffer_width {
                break;
            }
        }
    }
    
    /// Fallback bitmap font rendering with improved patterns
    fn draw_text_bitmap(
        buffer: &mut [u32],
        x: usize,
        y: usize,
        text: &str,
        buffer_width: usize,
        buffer_height: usize,
    ) {
        let _text_color = 0xFFFFFFFF; // White
        let font_width = 16; // Increased from 14
        let font_height = 24; // Increased from 20
        let scale = 2; // Scale factor
        
        let mut current_x = x;
        
        for ch in text.chars().take(50) {
            if current_x + font_width * scale >= buffer_width {
                break;
            }
            if y + font_height * scale >= buffer_height {
                break;
            }

            Self::draw_char_scaled(
                buffer,
                current_x,
                y,
                ch,
                font_width,
                font_height,
                scale,
                _text_color,
                buffer_width,
                buffer_height,
            );
            
            current_x += font_width * scale + 2;
        }
    }
    
    /// Draw a single character scaled up
    fn draw_char_scaled(
        buffer: &mut [u32],
        x: usize,
        y: usize,
        ch: char,
        base_width: usize,
        base_height: usize,
        scale: usize,
        color: u32,
        buffer_width: usize,
        buffer_height: usize,
    ) {
        // Simple bitmap patterns for common characters (Latin + Cyrillic + numbers)
        let pattern = Self::get_char_pattern(ch, base_width, base_height);
        
        for (row_idx, row) in pattern.iter().enumerate() {
            for (col_idx, &pixel) in row.iter().enumerate() {
                if pixel {
                    // Draw scaled pixel
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = x + col_idx * scale + sx;
                            let py = y + row_idx * scale + sy;
                            
                            if px < buffer_width && py < buffer_height {
                                let idx = py * buffer_width + px;
                                if idx < buffer.len() {
                                    buffer[idx] = color;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Get bitmap pattern for a character with better Cyrillic support
    fn get_char_pattern(ch: char, width: usize, height: usize) -> Vec<Vec<bool>> {
        let mut pattern = vec![vec![false; width]; height];
        
        // Helper to set pixel safely
        fn set(pattern: &mut Vec<Vec<bool>>, x: usize, y: usize, width: usize, height: usize) {
            if x < width && y < height {
                pattern[y][x] = true;
            }
        }
        
        // Helper to draw vertical line
        fn vline(pattern: &mut Vec<Vec<bool>>, x: usize, y1: usize, y2: usize, width: usize, height: usize) {
            for y in y1..=y2.min(height-1) {
                set(pattern, x, y, width, height);
            }
        }
        
        // Helper to draw horizontal line
        fn hline(pattern: &mut Vec<Vec<bool>>, x1: usize, x2: usize, y: usize, width: usize, height: usize) {
            for x in x1..=x2.min(width-1) {
                set(pattern, x, y, width, height);
            }
        }
        
        match ch {
            // Numbers
            '0' => {
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                    set(&mut pattern, width-1, y, width, height);
                }
                for x in 1..width-1 {
                    set(&mut pattern, x, 0, width, height);
                    set(&mut pattern, x, height-1, width, height);
                }
            }
            '1' => {
                // Number 1 - vertical line with small top
                let mid_x = width / 2;
                vline(&mut pattern, mid_x, 0, height-1, width, height);
                if width > 2 {
                    hline(&mut pattern, mid_x.saturating_sub(1), mid_x, 0, width, height);
                }
            }
            '2' => {
                // Number 2 - S-like shape
                hline(&mut pattern, 1, width-2, 0, width, height);
                hline(&mut pattern, 1, width-2, height/2, width, height);
                hline(&mut pattern, 1, width-2, height-1, width, height);
                set(&mut pattern, width-1, 0, width, height);
                set(&mut pattern, 0, height/2, width, height);
                set(&mut pattern, 0, height-1, width, height);
            }
            '3' => {
                // Number 3 - two curves
                hline(&mut pattern, 1, width-2, 0, width, height);
                hline(&mut pattern, 1, width-2, height/2, width, height);
                hline(&mut pattern, 1, width-2, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
            }
            '4' => {
                // Number 4
                vline(&mut pattern, 0, 0, height/2, width, height);
                hline(&mut pattern, 0, width-1, height/2, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
            }
            '5' => {
                // Number 5
                hline(&mut pattern, 0, width-1, 0, width, height);
                vline(&mut pattern, 0, 0, height/2, width, height);
                hline(&mut pattern, 0, width-1, height/2, width, height);
                vline(&mut pattern, width-1, height/2, height-1, width, height);
                hline(&mut pattern, 0, width-1, height-1, width, height);
            }
            '6' => {
                // Number 6
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                }
                hline(&mut pattern, 1, width-2, 0, width, height);
                hline(&mut pattern, 1, width-2, height/2, width, height);
                hline(&mut pattern, 1, width-2, height-1, width, height);
                vline(&mut pattern, width-1, height/2, height-1, width, height);
            }
            '7' => {
                // Number 7
                hline(&mut pattern, 0, width-1, 0, width, height);
                for y in 1..height {
                    let x = width - 1 - ((y * (width - 1)) / height.max(1));
                    set(&mut pattern, x, y, width, height);
                }
            }
            '8' => {
                // Number 8 - two circles
                for y in 1..height/2 {
                    set(&mut pattern, 0, y, width, height);
                    set(&mut pattern, width-1, y, width, height);
                }
                for y in height/2..height-1 {
                    set(&mut pattern, 0, y, width, height);
                    set(&mut pattern, width-1, y, width, height);
                }
                hline(&mut pattern, 1, width-2, 0, width, height);
                hline(&mut pattern, 1, width-2, height/2, width, height);
                hline(&mut pattern, 1, width-2, height-1, width, height);
            }
            '9' => {
                // Number 9 - inverted 6
                for y in 0..height/2 {
                    set(&mut pattern, 0, y, width, height);
                }
                hline(&mut pattern, 1, width-2, 0, width, height);
                hline(&mut pattern, 1, width-2, height/2, width, height);
                hline(&mut pattern, 1, width-2, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
            }
            // Latin letters
            'A'..='Z' | 'a'..='z' => {
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                    set(&mut pattern, width-1, y, width, height);
                }
                for x in 0..width {
                    set(&mut pattern, x, 0, width, height);
                }
                if height > 4 {
                    let mid = height / 2;
                    hline(&mut pattern, 1, width-2, mid, width, height);
                }
            }
            // Cyrillic letters - detailed patterns
            'М' | 'м' => {
                // M shape - two vertical lines with V in middle
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                let mid_x = width / 2;
                // Draw V shape in middle
                for y in 0..height/2 {
                    let offset = (y * mid_x) / (height/2).max(1);
                    if offset < mid_x && mid_x - offset < width && mid_x + offset < width {
                        set(&mut pattern, mid_x - offset, y, width, height);
                        set(&mut pattern, mid_x + offset, y, width, height);
                    }
                }
            }
            'Е' | 'е' => {
                // E shape - vertical line with three horizontals
                vline(&mut pattern, 0, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
                hline(&mut pattern, 0, width-1, height-1, width, height);
                if height > 4 {
                    hline(&mut pattern, 0, width*3/4, height/2, width, height);
                }
            }
            'Т' | 'т' => {
                // T shape - horizontal line on top, vertical in middle
                hline(&mut pattern, 0, width-1, 0, width, height);
                let mid_x = width / 2;
                vline(&mut pattern, mid_x, 0, height-1, width, height);
            }
            'К' | 'к' => {
                // K shape - vertical line, two diagonals
                vline(&mut pattern, 0, 0, height-1, width, height);
                let mid_y = height / 2;
                // Top diagonal
                for y in 0..mid_y {
                    let x = width - 1 - ((mid_y - y) * (width - 2)) / mid_y.max(1);
                    if x < width {
                        set(&mut pattern, x, y, width, height);
                    }
                }
                // Bottom diagonal
                for y in mid_y..height {
                    let x = width - 1 - ((y - mid_y) * (width - 2)) / (height - mid_y).max(1);
                    if x < width {
                        set(&mut pattern, x, y, width, height);
                    }
                }
            }
            'А' | 'а' => {
                // A shape - triangle with horizontal line
                let mid_x = width / 2;
                // Draw triangle sides
                for y in 0..height {
                    let offset = (y * mid_x) / height.max(1);
                    if offset < mid_x {
                        set(&mut pattern, mid_x - offset, y, width, height);
                        set(&mut pattern, mid_x + offset, y, width, height);
                    }
                }
                // Horizontal line in middle
                if height > 4 {
                    hline(&mut pattern, 1, width-2, height/2, width, height);
                }
            }
            // Other Cyrillic letters - use similar patterns
            'Б' | 'б' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
                hline(&mut pattern, 0, width*2/3, height/2, width, height);
                hline(&mut pattern, 0, width-1, height-1, width, height);
            }
            'В' | 'в' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
                hline(&mut pattern, 0, width*2/3, height/2, width, height);
                hline(&mut pattern, 0, width-1, height-1, width, height);
            }
            'Г' | 'г' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
            }
            'Д' | 'д' => {
                for x in 0..width {
                    set(&mut pattern, x, height-1, width, height);
                }
                let mid_x = width / 2;
                for y in 0..height-1 {
                    let offset = ((height-1-y) * mid_x) / height.max(1);
                    if offset < mid_x {
                        set(&mut pattern, mid_x - offset, y, width, height);
                        set(&mut pattern, mid_x + offset, y, width, height);
                    }
                }
            }
            'Ж' | 'ж' => {
                let mid_x = width / 2;
                let mid_y = height / 2;
                vline(&mut pattern, mid_x, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, mid_y, width, height);
                for y in 0..mid_y {
                    if width > 2 {
                        set(&mut pattern, 0, y, width, height);
                        set(&mut pattern, width-1, y, width, height);
                    }
                }
            }
            'З' | 'з' => {
                hline(&mut pattern, 1, width-2, 0, width, height);
                hline(&mut pattern, 1, width-2, height-1, width, height);
                hline(&mut pattern, 1, width*2/3, height/2, width, height);
                set(&mut pattern, width-1, 0, width, height);
                set(&mut pattern, width-1, height-1, width, height);
            }
            'И' | 'и' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                for y in 0..height {
                    let x = (y * (width-1)) / height.max(1);
                    set(&mut pattern, x, y, width, height);
                }
            }
            'Й' | 'й' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                for y in 0..height {
                    let x = (y * (width-1)) / height.max(1);
                    set(&mut pattern, x, y, width, height);
                }
                if width > 2 && height > 2 {
                    set(&mut pattern, width-2, 0, width, height);
                }
            }
            'Л' | 'л' => {
                let mid_x = width / 2;
                for y in 0..height {
                    let offset = ((height-y) * mid_x) / height.max(1);
                    if offset < mid_x {
                        set(&mut pattern, mid_x - offset, y, width, height);
                        set(&mut pattern, mid_x + offset, y, width, height);
                    }
                }
            }
            'Н' | 'н' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                if height > 4 {
                    hline(&mut pattern, 1, width-2, height/2, width, height);
                }
            }
            'О' | 'о' => {
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height); set(&mut pattern, width-1, y, width, height);
                }
                for x in 1..width-1 {
                    set(&mut pattern, x, 0, width, height); set(&mut pattern, x, height-1, width, height);
                }
            }
            'П' | 'п' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
            }
            'Р' | 'р' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
                hline(&mut pattern, 0, width*2/3, height/2, width, height);
            }
            'С' | 'с' => {
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                }
                for x in 1..width-1 {
                    set(&mut pattern, x, 0, width, height); set(&mut pattern, x, height-1, width, height);
                }
            }
            'У' | 'у' => {
                let mid_x = width / 2;
                for y in 0..height*2/3 {
                    let offset = (y * mid_x) / (height*2/3).max(1);
                    if offset < mid_x {
                        set(&mut pattern, mid_x - offset, y, width, height);
                        set(&mut pattern, mid_x + offset, y, width, height);
                    }
                }
                vline(&mut pattern, mid_x, height*2/3, height-1, width, height);
            }
            'Ф' | 'ф' => {
                let mid_x = width / 2;
                vline(&mut pattern, mid_x, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
                hline(&mut pattern, 0, width-1, height-1, width, height);
                if height > 4 {
                    hline(&mut pattern, 0, width-1, height/2, width, height);
                }
            }
            'Х' | 'х' => {
                for y in 0..height {
                    let x1 = (y * (width-1)) / height.max(1);
                    let x2 = width - 1 - x1;
                    set(&mut pattern, x1, y, width, height);
                    set(&mut pattern, x2, y, width, height);
                }
            }
            'Ц' | 'ц' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                if width > 2 {
                    vline(&mut pattern, width-2, height*3/4, height-1, width, height);
                }
            }
            'Ч' | 'ч' => {
                vline(&mut pattern, 0, 0, height*2/3, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                if height > 4 {
                    hline(&mut pattern, 0, width-1, height*2/3, width, height);
                }
            }
            'Ш' | 'ш' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width/2, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
            }
            'Щ' | 'щ' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width/2, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, 0, width, height);
                if width > 2 {
                    vline(&mut pattern, width-2, height*3/4, height-1, width, height);
                }
            }
            'Ъ' | 'ъ' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                hline(&mut pattern, 0, width-1, height/2, width, height);
                if width > 2 {
                    vline(&mut pattern, width-1, height/2, height-1, width, height);
                }
            }
            'Ы' | 'ы' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                vline(&mut pattern, width-1, 0, height-1, width, height);
                if width > 2 {
                    vline(&mut pattern, width-2, height/2, height-1, width, height);
                }
            }
            'Ь' | 'ь' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                if width > 2 {
                    vline(&mut pattern, width-1, 0, height/2, width, height);
                }
            }
            'Э' | 'э' => {
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                }
                for x in 1..width-1 {
                    set(&mut pattern, x, 0, width, height); set(&mut pattern, x, height-1, width, height);
                }
                if width > 2 {
                    set(&mut pattern, width-1, height/2, width, height);
                }
            }
            'Ю' | 'ю' => {
                vline(&mut pattern, 0, 0, height-1, width, height);
                let mid = width / 2;
                for y in 1..height-1 {
                    set(&mut pattern, mid, y, width, height);
                }
                for x in mid+1..width-1 {
                    set(&mut pattern, x, 0, width, height); set(&mut pattern, x, height-1, width, height);
                }
            }
            'Я' | 'я' => {
                vline(&mut pattern, width-1, 0, height-1, width, height);
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                }
                for x in 1..width-1 {
                    set(&mut pattern, x, 0, width, height);
                }
                if height > 4 {
                    hline(&mut pattern, 1, width-2, height/2, width, height);
                }
            }
            'Ё' | 'ё' => {
                for y in 1..height-1 {
                    set(&mut pattern, 0, y, width, height);
                }
                for x in 1..width-1 {
                    set(&mut pattern, x, 0, width, height); set(&mut pattern, x, height-1, width, height);
                }
                if height > 4 {
                    hline(&mut pattern, 0, width*2/3, height/2, width, height);
                }
                if width > 2 && height > 2 {
                    set(&mut pattern, width-2, 0, width, height);
                }
            }
            // Other Cyrillic lowercase - handle individually or use uppercase
            ch if ch.is_lowercase() && ('а'..='я').contains(&ch) && 
                ch != 'а' && ch != 'е' && ch != 'к' && ch != 'м' && ch != 'т' => {
                // Use uppercase pattern for lowercase
                let upper = ch.to_uppercase().next().unwrap_or(ch);
                pattern = Self::get_char_pattern(upper, width, height);
            }
            ' ' => {
                // Space - empty pattern
            }
            ':' => {
                // Colon - two dots
                let dot_y1 = height / 3;
                let dot_y2 = 2 * height / 3;
                let dot_x = width / 2;
                if dot_y1 < height && dot_y2 < height && dot_x < width {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if (dot_x as i32 + dx) >= 0 && (dot_x as i32 + dx) < width as i32 &&
                               (dot_y1 as i32 + dy) >= 0 && (dot_y1 as i32 + dy) < height as i32 {
                                set(&mut pattern, (dot_x as i32 + dx) as usize, (dot_y1 as i32 + dy) as usize, width, height);
                            }
                            if (dot_x as i32 + dx) >= 0 && (dot_x as i32 + dx) < width as i32 &&
                               (dot_y2 as i32 + dy) >= 0 && (dot_y2 as i32 + dy) < height as i32 {
                                set(&mut pattern, (dot_x as i32 + dx) as usize, (dot_y2 as i32 + dy) as usize, width, height);
                            }
                        }
                    }
                }
            }
            '-' => {
                // Dash - horizontal line in middle
                let mid_y = height / 2;
                for x in 1..width-1 {
                    set(&mut pattern, x, mid_y, width, height);
                }
            }
            _ => {
                // Default: draw a simple box for unknown characters
                for y in 0..height {
                    set(&mut pattern, 0, y, width, height);
                    if width > 1 {
                        set(&mut pattern, width-1, y, width, height);
                    }
                }
                for x in 0..width {
                    set(&mut pattern, x, 0, width, height);
                    if height > 1 {
                        set(&mut pattern, x, height-1, width, height);
                    }
                }
            }
        }
        
        pattern
    }

    /// Get color from palette based on normalized value [0.0, 1.0]
    /// Returns color in BGRA format (0xAABBGGRR)
    fn get_color_from_palette(value: f64, palette: &str) -> u32 {
        // Clamp value to [0.0, 1.0]
        let t = value.max(0.0).min(1.0);
        
        let (r, g, b) = match palette {
            "green" => {
                // Cold → warm: blue → green → yellow
                if t < 0.5 {
                    // Blue to green
                    let t2 = t * 2.0;
                    let r = 0;
                    let g = (t2 * 255.0) as u8;
                    let b = ((1.0 - t2) * 255.0) as u8;
                    (r, g, b)
                } else {
                    // Green to yellow
                    let t2 = (t - 0.5) * 2.0;
                    let r = (t2 * 255.0) as u8;
                    let g = 255;
                    let b = 0;
                    (r, g, b)
                }
            }
            "red" => {
                // Black → dark red → red
                let r = (t * 255.0) as u8;
                let g = 0;
                let b = 0;
                (r, g, b)
            }
            "blue" => {
                // Black → dark blue → blue
                let r = 0;
                let g = 0;
                let b = (t * 255.0) as u8;
                (r, g, b)
            }
            "bw" => {
                // Black → white
                let gray = (t * 255.0) as u8;
                (gray, gray, gray)
            }
            _ => {
                // Default to green palette
                if t < 0.5 {
                    let t2 = t * 2.0;
                    let r = 0;
                    let g = (t2 * 255.0) as u8;
                    let b = ((1.0 - t2) * 255.0) as u8;
                    (r, g, b)
                } else {
                    let t2 = (t - 0.5) * 2.0;
                    let r = (t2 * 255.0) as u8;
                    let g = 255;
                    let b = 0;
                    (r, g, b)
                }
            }
        };
        
        // Convert RGB to BGRA format: 0xAABBGGRR
        (0xFF << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32)
    }

    /// Draw heatmap chart
    /// heatmaps: array of heatmap data, each as (data, min, max, palette)
    /// xlabel: optional x-axis label
    /// ylabel: optional y-axis label
    /// window: window reference
    pub fn draw_heatmap_chart(
        &mut self,
        heatmaps: &[(Vec<Vec<f64>>, Option<f64>, Option<f64>, String)],
        _xlabel: Option<&str>,
        _ylabel: Option<&str>,
        _window: &Window,
    ) -> Result<(), softbuffer::SoftBufferError> {
        if heatmaps.is_empty() {
            return Ok(());
        }

        // Use first heatmap data for rendering
        let (data, min_override, max_override, palette) = &heatmaps[0];

        // Validate data
        if data.is_empty() || data[0].is_empty() {
            return Ok(());
        }

        let rows = data.len();
        let cols = data[0].len();

        // Validate all rows have the same length
        for row in data.iter() {
            if row.len() != cols {
                return Ok(());
            }
        }

        // Save scale_factor before mutable borrow
        let scale_factor = self.scale_factor;

        let mut buffer = self.surface.buffer_mut()?;
        let buffer_width = buffer.width().get();
        let buffer_height = buffer.height().get();

        // Dark gray background (like matplotlib: rgb(30, 30, 30))
        let bg_color = 0xFF1E1E1E;
        for pixel in buffer.iter_mut() {
            *pixel = bg_color;
        }

        // Calculate min and max values
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for row in data.iter() {
            for &val in row.iter() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Use override values if provided
        let min_val = min_override.unwrap_or(min_val);
        let max_val = max_override.unwrap_or(max_val);

        // Handle edge case where all values are the same
        let value_range = if max_val > min_val {
            max_val - min_val
        } else {
            1.0
        };

        // Calculate plot area with adaptive margins (heatmap needs space for colorbar)
        let left_margin = ((buffer_width as f32 * 0.12).max(150.0) * scale_factor).min(buffer_width as f32 * 0.25) as u32;
        let right_margin = ((buffer_width as f32 * 0.08).max(70.0) * scale_factor).min(buffer_width as f32 * 0.15) as u32; // Space for colorbar
        let top_margin = ((buffer_height as f32 * 0.05).max(50.0) * scale_factor).min(buffer_height as f32 * 0.15) as u32;
        let bottom_margin = ((buffer_height as f32 * 0.08).max(70.0) * scale_factor).min(buffer_height as f32 * 0.2) as u32;
        let plot_width = buffer_width.saturating_sub(left_margin + right_margin);
        let plot_height = buffer_height.saturating_sub(top_margin + bottom_margin);
        let plot_x = left_margin;
        let plot_y = top_margin;

        let buffer_width_usize = buffer_width as usize;
        let buffer_height_usize = buffer_height as usize;
        let plot_width_usize = plot_width as usize;
        let plot_height_usize = plot_height as usize;
        let plot_x_usize = plot_x as usize;
        let plot_y_usize = plot_y as usize;

        // Calculate cell dimensions
        let cell_width = plot_width_usize / cols.max(1);
        let cell_height = plot_height_usize / rows.max(1);

        // Draw heatmap cells with borders
        let cell_border_color = 0xFF2A2A2A; // Slightly lighter than background for subtle borders
        
        for (row_idx, row) in data.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                // Normalize value to [0.0, 1.0]
                let normalized = if value_range > 0.0 {
                    ((value - min_val) / value_range).max(0.0).min(1.0)
                } else {
                    0.5 // If all values are the same, use middle color
                };

                // Get color from palette
                let color = Self::get_color_from_palette(normalized, palette);

                // Calculate cell position
                let cell_x = plot_x_usize + col_idx * cell_width;
                let cell_y = plot_y_usize + row_idx * cell_height;

                // Draw cell rectangle
                for y_offset in 0..cell_height {
                    for x_offset in 0..cell_width {
                        let x = cell_x + x_offset;
                        let y = cell_y + y_offset;
                        
                        // Draw border on edges
                        let is_border = x_offset == 0 || x_offset == cell_width - 1 || 
                                        y_offset == 0 || y_offset == cell_height - 1;
                        
                        if x < buffer_width_usize && y < buffer_height_usize {
                            let idx = y * buffer_width_usize + x;
                            if idx < buffer.len() {
                                if is_border {
                                    buffer[idx] = cell_border_color;
                                } else {
                                    buffer[idx] = color;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Draw axes
        let axis_color = 0xFFFFFFFF; // White

        // Draw x-axis (bottom)
        let x_axis_y = plot_y_usize + plot_height_usize;
        if x_axis_y < buffer_height_usize {
            for x in plot_x_usize..plot_x_usize + plot_width_usize {
                if x < buffer_width_usize {
                    let idx = x_axis_y * buffer_width_usize + x;
                    if idx < buffer.len() {
                        buffer[idx] = axis_color;
                    }
                }
            }
        }

        // Draw y-axis (left)
        let y_axis_x = plot_x_usize;
        if y_axis_x < buffer_width_usize {
            for y in plot_y_usize..plot_y_usize + plot_height_usize {
                if y < buffer_height_usize {
                    let idx = y * buffer_width_usize + y_axis_x;
                    if idx < buffer.len() {
                        buffer[idx] = axis_color;
                    }
                }
            }
        }

        // Draw colorbar (vertical bar on the right)
        let colorbar_x = plot_x_usize + plot_width_usize + 20;
        let colorbar_width = 30;
        let colorbar_height = plot_height_usize;

        if colorbar_x + colorbar_width < buffer_width_usize {
            for y_offset in 0..colorbar_height {
                let normalized = 1.0 - (y_offset as f64 / colorbar_height as f64);
                let color = Self::get_color_from_palette(normalized, palette);
                
                for x_offset in 0..colorbar_width {
                    let x = colorbar_x + x_offset;
                    let y = plot_y_usize + y_offset;
                    
                    if x < buffer_width_usize && y < buffer_height_usize {
                        let idx = y * buffer_width_usize + x;
                        if idx < buffer.len() {
                            buffer[idx] = color;
                        }
                    }
                }
            }

            // Draw colorbar border
            let border_color = 0xFFFFFFFF;
            // Left border
            for y_offset in 0..colorbar_height {
                let x = colorbar_x;
                let y = plot_y_usize + y_offset;
                if x < buffer_width_usize && y < buffer_height_usize {
                    let idx = y * buffer_width_usize + x;
                    if idx < buffer.len() {
                        buffer[idx] = border_color;
                    }
                }
            }
            // Right border
            for y_offset in 0..colorbar_height {
                let x = colorbar_x + colorbar_width - 1;
                let y = plot_y_usize + y_offset;
                if x < buffer_width_usize && y < buffer_height_usize {
                    let idx = y * buffer_width_usize + x;
                    if idx < buffer.len() {
                        buffer[idx] = border_color;
                    }
                }
            }
            // Top border
            for x_offset in 0..colorbar_width {
                let x = colorbar_x + x_offset;
                let y = plot_y_usize;
                if x < buffer_width_usize && y < buffer_height_usize {
                    let idx = y * buffer_width_usize + x;
                    if idx < buffer.len() {
                        buffer[idx] = border_color;
                    }
                }
            }
            // Bottom border
            for x_offset in 0..colorbar_width {
                let x = colorbar_x + x_offset;
                let y = plot_y_usize + colorbar_height - 1;
                if x < buffer_width_usize && y < buffer_height_usize {
                    let idx = y * buffer_width_usize + x;
                    if idx < buffer.len() {
                        buffer[idx] = border_color;
                    }
                }
            }
        }

        // Draw colorbar labels (min and max values)
        let label_font_size = 14.0 * self.scale_factor;
        let text_color = 0xFFFFFFFF; // White
        
        // Format min and max values
        let min_label = if min_val.fract() == 0.0 {
            format!("{}", min_val as i64)
        } else {
            format!("{:.2}", min_val)
        };
        let max_label = if max_val.fract() == 0.0 {
            format!("{}", max_val as i64)
        } else {
            format!("{:.2}", max_val)
        };
        
        // Draw min label (bottom of colorbar)
        let min_text_width = Self::calculate_text_width(self.font.as_ref(), &min_label, label_font_size);
        let min_text_x = (colorbar_x as f32 + (colorbar_width as f32 - min_text_width) / 2.0).round() as usize;
        let min_text_y = plot_y_usize + colorbar_height + 5; // 5 pixels below colorbar
        if min_text_y < buffer_height_usize && min_text_x < buffer_width_usize {
            Self::draw_text_improved_with_color(
                &mut self.atlas,
                self.font.as_ref(),
                &mut buffer,
                min_text_x,
                min_text_y,
                &min_label,
                label_font_size,
                self.scale_factor,
                buffer_width_usize,
                buffer_height_usize,
                Some(text_color),
            );
        }
        
        // Draw max label (top of colorbar)
        let max_text_width = Self::calculate_text_width(self.font.as_ref(), &max_label, label_font_size);
        let max_text_x = (colorbar_x as f32 + (colorbar_width as f32 - max_text_width) / 2.0).round() as usize;
        // Position at top, accounting for font ascent
        let max_text_y = if let Some(font) = self.font.as_ref() {
            if let Some(line_metrics) = font.horizontal_line_metrics(label_font_size) {
                (plot_y_usize as f32 - line_metrics.ascent - 5.0).max(0.0) as usize
            } else {
                plot_y_usize.saturating_sub(20)
            }
        } else {
            plot_y_usize.saturating_sub(20)
        };
        if max_text_y < buffer_height_usize && max_text_x < buffer_width_usize {
            Self::draw_text_improved_with_color(
                &mut self.atlas,
                self.font.as_ref(),
                &mut buffer,
                max_text_x,
                max_text_y,
                &max_label,
                label_font_size,
                self.scale_factor,
                buffer_width_usize,
                buffer_height_usize,
                Some(text_color),
            );
        }
        
        // Draw middle label (optional, for better readability)
        let mid_val = min_val + (max_val - min_val) / 2.0;
        let mid_label = if mid_val.fract() == 0.0 {
            format!("{}", mid_val as i64)
        } else {
            format!("{:.2}", mid_val)
        };
        let mid_text_width = Self::calculate_text_width(self.font.as_ref(), &mid_label, label_font_size);
        let mid_text_x = (colorbar_x as f32 + (colorbar_width as f32 - mid_text_width) / 2.0).round() as usize;
        let mid_text_y = plot_y_usize + colorbar_height / 2;
        if mid_text_y < buffer_height_usize && mid_text_x < buffer_width_usize {
            Self::draw_text_improved_with_color(
                &mut self.atlas,
                self.font.as_ref(),
                &mut buffer,
                mid_text_x,
                mid_text_y,
                &mid_label,
                label_font_size,
                self.scale_factor,
                buffer_width_usize,
                buffer_height_usize,
                Some(text_color),
            );
        }

        // Note: xlabel and ylabel rendering would require font rendering
        // For now, we'll skip it as it's optional

        buffer.present()?;
        Ok(())
    }
}
