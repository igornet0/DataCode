// Window state stored in GUI thread

use crate::plot::Window;
use crate::plot::Image;
use crate::plot::window::ImageViewState;
use crate::plot::renderer::Renderer;
use crate::plot::command::{ChartData, FigureData};
use std::sync::{Arc, Mutex, Condvar};

/// Render content type for a window
#[derive(Debug)]
pub enum RenderContent {
    None,
    Image(Arc<Mutex<Image>>, ImageViewState),
    Chart(ChartData),
    ImageGrid {
        images: Vec<Arc<Mutex<Image>>>,
        rows: usize,
        cols: usize,
        titles: Vec<String>,
    },
    Figure(FigureData), // Figure data extracted from Figure (Send + Sync)
}

/// State of a window stored in GUI thread
/// Windows NEVER leave the GUI thread - runtime communicates via commands
/// All rendering data lives here - no thread-local storage needed
pub struct WindowState {
    pub window: Window,
    pub renderer: Option<Renderer>,
    pub content: RenderContent,
    pub wait: Option<Arc<(Mutex<bool>, Condvar)>>, // For blocking runtime until window closes
    pub cursor_pos: Option<(f32, f32)>, // Cursor position (x, y) in screen coordinates
    pub selected_point: Option<(usize, usize)>, // Selected point (line_index, point_index) for line charts
    pub hovered_point: Option<(usize, usize)>, // Hovered point (line_index, point_index) for line charts
}

impl WindowState {
    pub fn new(window: Window) -> Self {
        Self {
            window,
            renderer: None,
            content: RenderContent::None,
            wait: None,
            cursor_pos: None,
            selected_point: None,
            hovered_point: None,
        }
    }
}

