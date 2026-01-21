// Window management for plot module

#[derive(Debug, Clone)]
pub struct ImageViewState {
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub is_dragging: bool,
    pub last_mouse_x: f32,
    pub last_mouse_y: f32,
}

impl ImageViewState {
    pub fn new() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            is_dragging: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
        self.is_dragging = false;
    }
}

#[derive(Debug)]
pub struct Window {
    pub(crate) window_handle: winit::window::Window,
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub is_open: bool,
    pub image_view_state: Option<ImageViewState>,
}

impl Window {
    pub fn new(window_handle: winit::window::Window, width: u32, height: u32, title: String) -> Self {
        Self {
            window_handle,
            width,
            height,
            title,
            is_open: true,
            image_view_state: Some(ImageViewState::new()),
        }
    }
    
    pub fn request_redraw(&self) {
        self.window_handle.request_redraw();
    }
    
    pub fn id(&self) -> winit::window::WindowId {
        self.window_handle.id()
    }
    
    /// Get physical size of the window (accounting for DPI scaling)
    pub fn physical_size(&self) -> (u32, u32) {
        let size = self.window_handle.inner_size();
        (size.width, size.height)
    }
    
    /// Update window size (called on resize event)
    pub fn update_size(&mut self) {
        let size = self.window_handle.inner_size();
        self.width = size.width;
        self.height = size.height;
    }
}

