// Plot system for managing EventLoopProxy and window communication

use winit::event_loop::EventLoopProxy;
use crate::plot::command::GuiCommand;
use std::sync::Mutex;
use std::cell::RefCell;

/// Plot system: communication with the main thread's EventLoop.
/// Window waiters are handled in the event loop closure (no global Mutex).
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

}

/// Global PlotSystem instance (set by gui when event loop starts).
/// EventLoopProxy is not Sync on some platforms (e.g. macOS), so Mutex is required for static storage.
/// Contention: init once; get_plot_system() locks only on first call per thread, then CACHED_PROXY is used (no lock).
/// See docs/gil_bottlenecks.md.
static PLOT_SYSTEM: Mutex<Option<PlotSystem>> = Mutex::new(None);

// Thread-local cache: first get_plot_system() in a thread locks once and caches proxy; later calls use cache (no lock).
thread_local! {
    static CACHED_PROXY: RefCell<Option<EventLoopProxy<GuiCommand>>> = RefCell::new(None);
}

/// Initialize the plot system with an EventLoopProxy (called from gui when event loop starts).
pub fn init_plot_system(proxy: EventLoopProxy<GuiCommand>) {
    *PLOT_SYSTEM.lock().unwrap() = Some(PlotSystem::new(proxy));
}

/// Get the PlotSystem (panics if not initialized). Uses thread-local cache after first call to avoid lock on every command.
pub fn get_plot_system() -> PlotSystem {
    CACHED_PROXY.with(|cell| {
        if let Some(ref proxy) = *cell.borrow() {
            return PlotSystem::new(proxy.clone());
        }
        let proxy = {
            let system = PLOT_SYSTEM.lock().unwrap();
            system.as_ref().expect("PlotSystem not initialized. Call init_plot_system() first.")
                .proxy().clone()
        };
        *cell.borrow_mut() = Some(proxy.clone());
        PlotSystem::new(proxy)
    })
}

