// GUI event loop and window management

use crate::plot::{GuiCommand, system, natives, WindowState, RenderContent};
use crate::plot::window::Window as PlotWindow;
use crate::plot::window::ImageViewState;
use winit::event_loop::EventLoopBuilder;
use winit::event::{Event, WindowEvent};
use winit::window::WindowBuilder;
use winit::event_loop::ControlFlow;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;

use crate::vm::window_events;

/// Global window storage in GUI thread
/// Windows NEVER leave this thread - runtime communicates via commands
static WINDOWS: LazyLock<Mutex<HashMap<winit::window::WindowId, WindowState>>> = 
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Run DataCode code with EventLoop (for plot support)
/// Creates EventLoop in main thread, spawns runtime in separate thread
pub fn run_with_event_loop<F>(code_runner: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    // Create EventLoop in main thread (required for macOS)
    let event_loop = EventLoopBuilder::<GuiCommand>::with_user_event()
        .build()
        .map_err(|e| format!("Failed to create event loop: {:?}", e))?;
    
    let proxy = event_loop.create_proxy();
    
    // Initialize PlotSystem
    system::init_plot_system(proxy.clone());
    
    // Shared flag to track if runtime thread has finished
    let runtime_finished = Arc::new(Mutex::new(false));
    let runtime_finished_clone = runtime_finished.clone();
    
    // Spawn runtime thread
    // DIAG: Log thread creation
    // eprintln!("[DIAG] Thread: Spawning runtime thread");
    // let thread_start_time = std::time::Instant::now();
    
    let runtime_handle = thread::spawn(move || {
        // let thread_id = std::thread::current().id();
        // eprintln!("[DIAG] Thread: Runtime thread started - thread_id={:?}", thread_id);
        
        let result = code_runner();
        
        // let thread_duration = thread_start_time.elapsed();
        
        // Mark runtime as finished when done
        *runtime_finished_clone.lock().unwrap() = true;
        result
    });
    
    // Run event loop (blocking)
    // All windows are stored in WINDOWS HashMap in GUI thread
    let runtime_finished_event_loop = runtime_finished.clone();
    event_loop.run(move |event, elwt| {
        // Check if we should exit before processing events
        let all_windows_closed = WINDOWS.lock().unwrap().is_empty();
        let runtime_done = *runtime_finished_event_loop.lock().unwrap();
        if all_windows_closed && runtime_done {
            elwt.exit();
            return;
        }
        
        // Use Poll when runtime is done to check for exit more frequently
        if runtime_done {
            elwt.set_control_flow(ControlFlow::Poll);
        } else {
            elwt.set_control_flow(ControlFlow::Wait);
        }
        
        match event {
            Event::UserEvent(cmd) => {
                match cmd {
                    GuiCommand::CreateWindow { width, height, title, icon, response } => {
                        
                        let mut window_builder = WindowBuilder::new()
                            .with_title(&title)
                            .with_inner_size(winit::dpi::LogicalSize::new(width as f64, height as f64));
                        
                        if let Some(icon) = icon {
                            window_builder = window_builder.with_window_icon(Some(icon));
                        }
                        
                        match window_builder.build(elwt) {
                            Ok(window) => {
                                let window_id = window.id();
                                
                                #[cfg(target_os = "macos")]
                                {
                                    // Load and set icon on macOS if needed
                                    if let Some(_icon) = natives::load_window_icon() {
                                        // Icon is already set via window_builder
                                    }
                                }
                                
                                // Create PlotWindow wrapper
                                let plot_window = PlotWindow::new(window, width, height, title.clone());
                                
                                // Create WindowState and store in global HashMap
                                let window_state = WindowState::new(plot_window);
                                WINDOWS.lock().unwrap().insert(window_id, window_state);
                                
                                // Note: Windows are stored in WINDOWS HashMap, not in thread-local
                                // Thread-local storage is kept for backward compatibility but not actively used
                                
                                // Send response with window_id
                                let _ = response.send(Ok(window_id));
                            }
                            Err(e) => {
                                let _ = response.send(Err(format!("Failed to build window: {:?}", e)));
                            }
                        }
                    }
                    GuiCommand::DrawImage { window_id, image } => {
                        // Store image in WindowState and request redraw
                        if let Ok(mut windows) = WINDOWS.lock() {
                            if let Some(state) = windows.get_mut(&window_id) {
                                let view_state = ImageViewState::new();
                                state.content = RenderContent::Image(image, view_state);
                                state.window.request_redraw();
                            }
                        }
                    }
                    GuiCommand::UpdateChart { window_id, chart_data } => {
                        // Store chart data in WindowState and request redraw
                        if let Ok(mut windows) = WINDOWS.lock() {
                            if let Some(state) = windows.get_mut(&window_id) {
                                state.content = RenderContent::Chart(chart_data);
                                state.window.request_redraw();
                            }
                        }
                    }
                    GuiCommand::UpdateImageGrid { window_id, images, rows, cols, titles } => {
                        // Store image grid in WindowState and request redraw
                        if let Ok(mut windows) = WINDOWS.lock() {
                            if let Some(state) = windows.get_mut(&window_id) {
                                state.content = RenderContent::ImageGrid {
                                    images,
                                    rows,
                                    cols,
                                    titles,
                                };
                                state.window.request_redraw();
                            }
                        }
                    }
                    GuiCommand::UpdateFigure { window_id, figure_data } => {
                        // Store figure data in WindowState and request redraw
                        if let Ok(mut windows) = WINDOWS.lock() {
                            if let Some(state) = windows.get_mut(&window_id) {
                                state.content = RenderContent::Figure(figure_data);
                                state.window.request_redraw();
                            }
                        }
                    }
                    GuiCommand::Redraw { window_id } => {
                        // Request redraw
                        if let Ok(mut windows) = WINDOWS.lock() {
                            if let Some(state) = windows.get_mut(&window_id) {
                                state.window.request_redraw();
                            }
                        }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::Resized(_new_size), window_id, .. } => {
                window_events::handle_window_resize(&WINDOWS, window_id);
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, window_id, .. } => {
                let should_exit = window_events::handle_window_close(&WINDOWS, window_id);
                
                // If all windows are closed and runtime has finished, exit event loop
                if should_exit {
                    let runtime_done = *runtime_finished_event_loop.lock().unwrap();
                    if runtime_done {
                        elwt.exit();
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, window_id, .. } => {
                window_events::handle_cursor_moved(&WINDOWS, window_id, position);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: button_state, button: winit::event::MouseButton::Left, .. }, window_id, .. } => {
                // Handle mouse click - select point on line chart
                if button_state == winit::event::ElementState::Pressed {
                    window_events::handle_mouse_click(&WINDOWS, window_id);
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, window_id, .. } => {
                window_events::handle_redraw(&WINDOWS, window_id);
            }
            Event::AboutToWait => {
                // Check if we should exit: all windows closed and runtime finished
                let all_windows_closed = WINDOWS.lock().unwrap().is_empty();
                let runtime_done = *runtime_finished_event_loop.lock().unwrap();
                if all_windows_closed && runtime_done {
                    elwt.exit();
                    return;
                }
            }
            _ => {}
        }
    }).map_err(|e| format!("Event loop error: {:?}", e))?;
    
    // Wait for runtime to finish
    // DIAG: Log thread join
    let join_start = std::time::Instant::now();
    let result = runtime_handle.join().map_err(|_| "Runtime thread panicked".to_string())??;
    let _join_duration = join_start.elapsed();
    
    Ok(result)
}

