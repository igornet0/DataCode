// Plot system for managing EventLoopProxy and window communication

use winit::event_loop::EventLoopProxy;
use winit::window::WindowId;
use crate::plot::command::GuiCommand;
use std::sync::{Arc, Mutex, Condvar, LazyLock};
use std::collections::HashMap;

/// Global window registry mapping WindowId to waiters
/// Accessed from both main thread (on close) and runtime thread (on wait)
static WINDOW_WAITERS: LazyLock<Mutex<HashMap<WindowId, Arc<(Mutex<bool>, Condvar)>>>> = 
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Plot system that manages communication with the main thread's EventLoop
pub struct PlotSystem {
    proxy: EventLoopProxy<GuiCommand>,
}

impl PlotSystem {
    /// Create a new PlotSystem with the given EventLoopProxy
    pub fn new(proxy: EventLoopProxy<GuiCommand>) -> Self {
        Self { proxy }
    }

    /// Get the EventLoopProxy
    pub fn proxy(&self) -> &EventLoopProxy<GuiCommand> {
        &self.proxy
    }

    /// Register a waiter for a window close event
    pub fn register_waiter(window_id: WindowId, waiter: Arc<(Mutex<bool>, Condvar)>) {
        WINDOW_WAITERS.lock().unwrap().insert(window_id, waiter);
    }

    /// Get waiter for a window and remove it from registry
    pub fn get_waiter(window_id: WindowId) -> Option<Arc<(Mutex<bool>, Condvar)>> {
        WINDOW_WAITERS.lock().unwrap().remove(&window_id)
    }

    /// Notify that a window has closed
    pub fn notify_window_closed(window_id: WindowId) {
        if let Some(waiter) = WINDOW_WAITERS.lock().unwrap().remove(&window_id) {
            let (lock, cvar) = &*waiter;
            let mut closed = lock.lock().unwrap();
            *closed = true;
            cvar.notify_all();
        }
    }
}

/// Global PlotSystem instance (set by main.rs)
static PLOT_SYSTEM: Mutex<Option<PlotSystem>> = Mutex::new(None);

/// Initialize the plot system with an EventLoopProxy
/// Must be called from main.rs after creating EventLoop
pub fn init_plot_system(proxy: EventLoopProxy<GuiCommand>) {
    *PLOT_SYSTEM.lock().unwrap() = Some(PlotSystem::new(proxy));
}

/// Get the PlotSystem (panics if not initialized)
pub fn get_plot_system() -> PlotSystem {
    let system = PLOT_SYSTEM.lock().unwrap();
    if let Some(ref sys) = *system {
        // Clone the proxy (EventLoopProxy is Clone)
        PlotSystem::new(sys.proxy().clone())
    } else {
        panic!("PlotSystem not initialized. Call init_plot_system() first.");
    }
}

