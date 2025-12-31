// Window handle for runtime thread
// Runtime thread NEVER owns Window - only WindowId

use winit::window::WindowId;

/// Window handle that can be sent between threads
/// Runtime uses this to reference windows in GUI thread
/// This is a lightweight handle - Window itself stays in GUI thread
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlotWindowHandle {
    pub id: WindowId,
}

impl PlotWindowHandle {
    pub fn new(id: WindowId) -> Self {
        Self { id }
    }
    
    pub fn id(&self) -> WindowId {
        self.id
    }
}

