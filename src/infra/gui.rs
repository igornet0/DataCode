// GUI event loop and window management

use crate::plot::{GuiCommand, system, natives, WindowState, RenderContent};
use crate::plot::window::Window as PlotWindow;
use crate::plot::window::ImageViewState;
use winit::event_loop::EventLoopBuilder;
use winit::event::{Event, WindowEvent};
use winit::window::WindowBuilder;
use winit::event_loop::ControlFlow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Condvar};
use std::thread;

use crate::infra::window_events;

/// Run DataCode code with EventLoop (for plot support)
/// Creates EventLoop in main thread, spawns runtime in separate thread.
/// Window state lives in event-loop closure (no global Mutex).
pub fn run_with_event_loop<F>(code_runner: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    let event_loop = EventLoopBuilder::<GuiCommand>::with_user_event()
        .build()
        .map_err(|e| format!("Failed to create event loop: {:?}", e))?;
    
    let proxy = event_loop.create_proxy();
    system::init_plot_system(proxy.clone());
    
    let runtime_finished = Arc::new(AtomicBool::new(false));
    let runtime_finished_clone = runtime_finished.clone();
    
    let runtime_handle = thread::spawn(move || {
        let result = code_runner();
        runtime_finished_clone.store(true, Ordering::SeqCst);
        // Wake the event loop so it can exit when there are no windows (otherwise it may block in Wait forever).
        let _ = system::get_plot_system().proxy().send_event(GuiCommand::RuntimeFinished);
        result
    });
    
    // Window state and waiters are local to this closure; no global WINDOW_WAITERS Mutex
    let mut windows: HashMap<winit::window::WindowId, WindowState> = HashMap::new();
    let mut window_waiters: HashMap<winit::window::WindowId, Arc<(Mutex<bool>, Condvar)>> = HashMap::new();
    let runtime_finished_el = runtime_finished.clone();

    event_loop.run(move |event, elwt| {
        let all_windows_closed = windows.is_empty();
        let runtime_done = runtime_finished_el.load(Ordering::SeqCst);
        if all_windows_closed && runtime_done {
            elwt.exit();
            return;
        }
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
                                    if let Some(_icon) = natives::load_window_icon() {}
                                }
                                let plot_window = PlotWindow::new(window, width, height, title.clone());
                                let window_state = WindowState::new(plot_window);
                                windows.insert(window_id, window_state);
                                let _ = response.send(Ok(window_id));
                            }
                            Err(e) => {
                                let _ = response.send(Err(format!("Failed to build window: {:?}", e)));
                            }
                        }
                    }
                    GuiCommand::DrawImage { window_id, image } => {
                        if let Some(state) = windows.get_mut(&window_id) {
                            let view_state = ImageViewState::new();
                            state.content = RenderContent::Image(image, view_state);
                            state.window.request_redraw();
                        }
                    }
                    GuiCommand::UpdateChart { window_id, chart_data } => {
                        if let Some(state) = windows.get_mut(&window_id) {
                            state.content = RenderContent::Chart(chart_data);
                            state.window.request_redraw();
                        }
                    }
                    GuiCommand::UpdateImageGrid { window_id, images, rows, cols, titles } => {
                        if let Some(state) = windows.get_mut(&window_id) {
                            state.content = RenderContent::ImageGrid {
                                images, rows, cols, titles,
                            };
                            state.window.request_redraw();
                        }
                    }
                    GuiCommand::UpdateFigure { window_id, figure_data } => {
                        if let Some(state) = windows.get_mut(&window_id) {
                            state.content = RenderContent::Figure(figure_data);
                            state.window.request_redraw();
                        }
                    }
                    GuiCommand::Redraw { window_id } => {
                        if let Some(state) = windows.get_mut(&window_id) {
                            state.window.request_redraw();
                        }
                    }
                    GuiCommand::RegisterWaiter { window_id, waiter } => {
                        window_waiters.insert(window_id, waiter);
                    }
                    GuiCommand::RuntimeFinished => {
                        if windows.is_empty() {
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::Resized(_new_size), window_id, .. } => {
                window_events::handle_window_resize(&mut windows, window_id);
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, window_id, .. } => {
                let should_exit = window_events::handle_window_close(&mut windows, window_id, &mut window_waiters);
                if should_exit {
                    let runtime_done = runtime_finished_el.load(Ordering::SeqCst);
                    if runtime_done {
                        elwt.exit();
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, window_id, .. } => {
                window_events::handle_cursor_moved(&mut windows, window_id, position);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: button_state, button: winit::event::MouseButton::Left, .. }, window_id, .. } => {
                if button_state == winit::event::ElementState::Pressed {
                    window_events::handle_mouse_click(&mut windows, window_id);
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, window_id, .. } => {
                window_events::handle_redraw(&mut windows, window_id);
            }
            Event::AboutToWait => {
                let all_windows_closed = windows.is_empty();
                let runtime_done = runtime_finished_el.load(Ordering::SeqCst);
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
