// Native functions for plot module

use crate::common::value::Value;
use crate::plot::{Image, Window, Figure, GuiCommand, system, PlotWindowHandle};
use crate::plot::command::{ChartData as CommandChartData, ChartType as CommandChartType, FigureData, AxisData};
use crate::plot::window::ImageViewState;
use crate::plot::renderer::Renderer;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use winit::window::{WindowId, Icon};
use std::path::Path;
use std::sync::{Arc, Mutex, Condvar, mpsc};

// Note: WindowHandle was removed as it's no longer needed

// Note: Window icon is set via WindowBuilder in main.rs

// Thread-local storage for EventLoop and window management
thread_local! {
    static WINDOW_DATA: RefCell<HashMap<WindowId, WindowData>> = RefCell::new(HashMap::new());
    static RENDERERS: RefCell<HashMap<WindowId, Renderer>> = RefCell::new(HashMap::new());
    static IMAGE_VIEW_STATES: RefCell<HashMap<WindowId, ImageViewState>> = RefCell::new(HashMap::new());
    static WINDOW_IMAGES: RefCell<HashMap<WindowId, Rc<RefCell<Image>>>> = RefCell::new(HashMap::new());
    static CURSOR_POSITIONS: RefCell<HashMap<WindowId, (f32, f32)>> = RefCell::new(HashMap::new());
    static PLOT_STATE: RefCell<PlotState> = RefCell::new(PlotState {
        xlabel: None,
        ylabel: None,
        line_data: Vec::new(),
        bar_data: Vec::new(),
        pie_data: Vec::new(),
        heatmap_data: Vec::new(),
    });
    static WINDOW_CHART_DATA: RefCell<HashMap<WindowId, ChartData>> = RefCell::new(HashMap::new());
    static WINDOW_POINT_STATES: RefCell<HashMap<WindowId, PointState>> = RefCell::new(HashMap::new());
}

// Thread-local storage for window management
// Note: Windows are stored in main thread's thread-local storage
// Runtime thread uses WindowId for communication
thread_local! {
    static ACTIVE_WINDOWS: RefCell<HashMap<WindowId, Rc<RefCell<Window>>>> = RefCell::new(HashMap::new());
    static CLOSED_WINDOWS: RefCell<HashSet<WindowId>> = RefCell::new(HashSet::new());
}

// Helper functions to access windows from main.rs (main thread can use thread-local)
pub fn with_active_windows<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<WindowId, Rc<RefCell<Window>>>) -> R,
{
    ACTIVE_WINDOWS.with(|windows| {
        f(&mut *windows.borrow_mut())
    })
}

pub fn with_closed_windows<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashSet<WindowId>) -> R,
{
    CLOSED_WINDOWS.with(|closed| {
        f(&mut *closed.borrow_mut())
    })
}

// Helper function to get PlotSystem (panics if not initialized)
fn get_plot_system() -> system::PlotSystem {
    system::get_plot_system()
}

// EventLoop is now created in main.rs - this function is no longer needed

// WindowData - kept for thread-local storage (temporary, will be removed)
#[allow(dead_code)]
struct WindowData {
    renderer: Option<Renderer>,
}

// ChartType and ChartData - kept for thread-local storage compatibility
// Using command types for new code, but keeping these for thread-local
#[allow(dead_code)]
type ChartType = CommandChartType;
#[allow(dead_code)]
type ChartData = CommandChartData;

struct PlotState {
    xlabel: Option<String>,
    ylabel: Option<String>,
    line_data: Vec<(Vec<f64>, Vec<f64>, bool, usize, usize, u32)>, // Множество линий для одного графика: (x_data, y_data, show_points, point_size, line_width, color)
    bar_data: Vec<(Vec<String>, Vec<f64>, u32)>, // Данные столбчатых диаграмм: (x_labels, y_data, color)
    pie_data: Vec<(Vec<String>, Vec<f64>, u32)>, // Данные круговых диаграмм: (x_labels, y_data, color)
    heatmap_data: Vec<(Vec<Vec<f64>>, Option<f64>, Option<f64>, String)>, // Данные тепловых карт: (data, min, max, palette)
}

// ChartData removed - using command::ChartData instead

// PointState - kept for potential future use
#[allow(dead_code)]
struct PointState {
    hovered: Option<(usize, usize)>, // (line_index, point_index)
    selected: Option<(usize, usize)>, // (line_index, point_index)
}

impl PointState {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            hovered: None,
            selected: None,
        }
    }
}

/// Load window icon from file
/// Tries multiple paths to find the icon file
pub fn load_window_icon() -> Option<Icon> {
    // Try different possible paths for the icon
    let mut icon_paths = Vec::new();
    
    // Try relative to current working directory first (most common case)
    if let Ok(cwd) = std::env::current_dir() {
        icon_paths.push(cwd.join("src/lib/plot/icon/datacode-plot.png").to_string_lossy().to_string());
        icon_paths.push(cwd.join("icon/datacode-plot.png").to_string_lossy().to_string());
    }
    
    // Relative to project root (for development)
    icon_paths.push("src/lib/plot/icon/datacode-plot.png".to_string());
    icon_paths.push("./src/lib/plot/icon/datacode-plot.png".to_string());
    
    // Relative to executable (for installed version)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // Try relative to executable directory
            icon_paths.push(exe_dir.join("icon/datacode-plot.png").to_string_lossy().to_string());
            icon_paths.push(exe_dir.join("src/lib/plot/icon/datacode-plot.png").to_string_lossy().to_string());
        }
    }

    for path_str in &icon_paths {
        let path = Path::new(path_str);
        if path.exists() {
            // Load image using image crate
            if let Ok(img) = image::open(path) {
                // Convert to RGBA8
                let rgba_img = img.to_rgba8();
                let (width, height) = rgba_img.dimensions();
                
                // Resize icon to appropriate size for window icon
                // macOS requires square icons, preferably powers of 2: 16, 32, 64, 128, 256, 512
                // Use 512x512 for best quality on macOS (retina displays)
                let target_size = if width > 512 || height > 512 {
                    512
                } else if width > 256 || height > 256 {
                    256
                } else if width > 128 || height > 128 {
                    128
                } else if width > 64 || height > 64 {
                    64
                } else {
                    // Ensure minimum size of 32x32
                    32.max(width.max(height))
                };
                
                // Resize to square maintaining aspect ratio, then center on square canvas
                let scale = (target_size as f32 / width.max(height) as f32).min(1.0);
                let new_width = (width as f32 * scale) as u32;
                let new_height = (height as f32 * scale) as u32;
                
                // Resize image
                let resized = image::imageops::resize(&rgba_img, new_width, new_height, image::imageops::FilterType::Lanczos3);
                
                // Create square canvas and center the resized image
                let mut square_img = image::RgbaImage::new(target_size, target_size);
                // Fill with transparent background
                for pixel in square_img.pixels_mut() {
                    *pixel = image::Rgba([0, 0, 0, 0]);
                }
                
                // Calculate offset to center the image
                let offset_x = (target_size - new_width) / 2;
                let offset_y = (target_size - new_height) / 2;
                
                // Copy resized image to center of square canvas
                image::imageops::overlay(&mut square_img, &resized, offset_x as i64, offset_y as i64);
                
                let rgba_data = square_img.into_raw();
                
                // Create Icon from RGBA data (must be square)
                match Icon::from_rgba(rgba_data, target_size, target_size) {
                    Ok(icon) => {
                        return Some(icon);
                    }
                    Err(_e) => {
                        // Icon creation failed, try next path
                    }
                }
            }
        }
    }
    
    None
}

// Note: The ensure_global_event_loop function was removed as we're using a simpler approach
// The global event loop is managed directly in native_plot_window and native_plot_wait

/// Load image from file path
/// plot.image(path) -> Image
pub fn native_plot_image(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let path = match &args[0] {
        Value::String(s) => s.clone(),
        Value::Path(p) => p.to_string_lossy().to_string(),
        _ => return Value::Null,
    };

    match Image::load_from_path(&path) {
        Ok(image) => Value::Image(Rc::new(RefCell::new(image))),
        Err(_) => Value::Null,
    }
}

/// Create a window
/// plot.window(width, height, title) -> Window
pub fn native_plot_window(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }

    // Parse arguments
    let width = match &args[0] {
        Value::Number(n) => *n as u32,
        _ => return Value::Null,
    };
    
    let height = match &args[1] {
        Value::Number(n) => *n as u32,
        _ => return Value::Null,
    };
    
    let title = match &args[2] {
        Value::String(s) => s.clone(),
        _ => "Window".to_string(),
    };

    // Get PlotSystem (EventLoop is created in main.rs)
    let plot_system = get_plot_system();
    let proxy = plot_system.proxy();
    
    // Create channel for response
        let (tx, rx) = mpsc::channel();
        
            // Load window icon
            let icon = load_window_icon();
            
    // Send request to create window via EventLoopProxy
    let command = GuiCommand::CreateWindow {
                width,
                height,
                title: title.clone(),
                icon,
                response: tx,
            };
            
    if let Err(_e) = proxy.send_event(command) {
        return Value::Null;
    }
    
    // Wait for response from main thread
                match rx.recv() {
                    Ok(Ok(window_id)) => {
            // Return PlotWindowHandle (contains only WindowId) - Window stays in GUI thread
            let handle = PlotWindowHandle::new(window_id);
            return Value::Window(handle);
                    }
                    Ok(Err(_e)) => {
                        return Value::Null;
                    }
                    Err(_) => {
                        return Value::Null;
                    }
                }
}

/// Draw image to window
/// window.draw(image) -> Null
/// Runtime sends command to GUI thread - never touches Window directly
pub fn native_window_draw(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let window_handle = match &args[0] {
        Value::Window(handle) => *handle,
        _ => return Value::Null,
    };

    let image_rc = match &args[1] {
        Value::Image(img) => img.clone(),
        _ => return Value::Null,
    };

    // Convert Rc<RefCell<Image>> to Arc<Mutex<Image>> for Send
    // Image doesn't implement Clone, so we need to manually clone the data
    let image = {
        let img_ref = image_rc.borrow();
        let cloned_image = Image {
            width: img_ref.width,
            height: img_ref.height,
            data: img_ref.data.clone(),
        };
        Arc::new(Mutex::new(cloned_image))
    };

    // Send command to GUI thread to draw image
    let plot_system = get_plot_system();
    let proxy = plot_system.proxy();
    
    let command = GuiCommand::DrawImage {
        window_id: window_handle.id(),
        image,
    };
    
    if let Err(_e) = proxy.send_event(command) {
        return Value::Null;
    }
    
        Value::Null
}

/// Wait for window to close (blocking)
/// plot.wait(window) -> Null
/// Runtime waits for GUI thread to notify via Condvar
pub fn native_plot_wait(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let window_handle = match &args[0] {
        Value::Window(handle) => *handle,
        _ => return Value::Null,
    };

    let window_id = window_handle.id();

    // Create a waiter for this window's close event
    let waiter = Arc::new((Mutex::new(false), Condvar::new()));
    
    // Register waiter in PlotSystem (accessible from main thread)
    system::PlotSystem::register_waiter(window_id, waiter.clone());

    // Event loop is running in main thread - just wait for window to close
        let (lock, cvar) = &*waiter;
        let mut closed = lock.lock().unwrap();
        while !*closed {
            closed = cvar.wait(closed).unwrap();
        }
    Value::Null
}

/// Convenience function: show image in window
/// plot.show(path) -> Null
/// plot.show(tensor) -> Null
/// plot.show(path, title) -> Null
/// plot.show(tensor, title) -> Null
/// plot.show(figure) -> Null
/// plot.show(title="...") -> Null (for line charts)
pub fn native_plot_show(args: &[Value]) -> Value {
    // Check if first argument is a Figure
    if !args.is_empty() {
        if let Value::Figure(_) = &args[0] {
            return native_plot_show_figure(args);
        }
    }

    // Check if we have pie data in plot state
    let has_pie_data = PLOT_STATE.with(|state| {
        let plot_state = state.borrow();
        !plot_state.pie_data.is_empty()
    });

    // Check if we have bar data in plot state
    let has_bar_data = PLOT_STATE.with(|state| {
        let plot_state = state.borrow();
        !plot_state.bar_data.is_empty()
    });

    // Check if we have heatmap data in plot state
    let has_heatmap_data = PLOT_STATE.with(|state| {
        let plot_state = state.borrow();
        !plot_state.heatmap_data.is_empty()
    });

    // Check if we have line data in plot state
    let has_line_data = PLOT_STATE.with(|state| {
        let plot_state = state.borrow();
        !plot_state.line_data.is_empty()
    });

    // If we have pie data and no image arguments, render pie chart
    if has_pie_data {
        // Check if first argument is an object with title key (named arguments)
        let title = if !args.is_empty() {
            match &args[0] {
                Value::Object(map) => {
                    if let Some(Value::String(s)) = map.get("title") {
                        Some(s.clone())
                    } else {
                        None
                    }
                }
                Value::String(s) => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        };

        // Get plot state data
        let (xlabel, ylabel, pie_data) = PLOT_STATE.with(|state| {
            let plot_state = state.borrow();
            (
                plot_state.xlabel.clone(),
                plot_state.ylabel.clone(),
                plot_state.pie_data.clone(),
            )
        });

        // Clear plot state after reading
        PLOT_STATE.with(|state| {
            let mut plot_state = state.borrow_mut();
            plot_state.xlabel = None;
            plot_state.ylabel = None;
            plot_state.pie_data.clear();
        });

        // Check if we have pie data
        if pie_data.is_empty() {
            return Value::Null;
        }

        // Clone all pie data for storage
        let pies_clone: Vec<(Vec<String>, Vec<f64>, u32)> = pie_data.iter().map(|(x, y, c)| {
            (x.clone(), y.clone(), *c)
        }).collect();
        let xlabel_clone = xlabel.clone();
        let ylabel_clone = ylabel.clone();

        // Create window with default size for charts
        let window_width = 800;
        let window_height = 600;
        let window_title = title.unwrap_or_else(|| "Pie Chart".to_string());

        let window_args = vec![
            Value::Number(window_width as f64),
            Value::Number(window_height as f64),
            Value::String(window_title),
        ];
        let window_result = native_plot_window(&window_args);
        let window = match window_result {
            Value::Window(w) => w,
            _ => return Value::Null,
        };

        // Send command to GUI thread to update chart data
        // Rendering will happen in GUI thread on RedrawRequested
        let plot_system = get_plot_system();
        let proxy = plot_system.proxy();
        
        // Convert to command::ChartData
        let chart_data = CommandChartData {
            chart_type: CommandChartType::Pie,
                lines: Vec::new(),
                bars: Vec::new(),
                pies: pies_clone,
                heatmaps: Vec::new(),
                xlabel: xlabel_clone,
                ylabel: ylabel_clone,
            pie_rotation: -std::f64::consts::PI / 2.0,
        };
        
        let command = GuiCommand::UpdateChart {
            window_id: window.id(),
            chart_data,
        };
        
        if let Err(_e) = proxy.send_event(command) {
            return Value::Null;
        }

        // Wait for window to close
        let wait_args = vec![Value::Window(window)];
        native_plot_wait(&wait_args);

        return Value::Null;
    }

    // If we have bar data and no image arguments, render bar chart
    if has_bar_data {
        // Check if first argument is an object with title key (named arguments)
        let title = if !args.is_empty() {
            match &args[0] {
                Value::Object(map) => {
                    if let Some(Value::String(s)) = map.get("title") {
                        Some(s.clone())
                    } else {
                        None
                    }
                }
                Value::String(s) => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        };

        // Get plot state data
        let (xlabel, ylabel, bar_data) = PLOT_STATE.with(|state| {
            let plot_state = state.borrow();
            (
                plot_state.xlabel.clone(),
                plot_state.ylabel.clone(),
                plot_state.bar_data.clone(),
            )
        });

        // Clear plot state after reading
        PLOT_STATE.with(|state| {
            let mut plot_state = state.borrow_mut();
            plot_state.xlabel = None;
            plot_state.ylabel = None;
            plot_state.bar_data.clear();
        });

        // Check if we have bar data
        if bar_data.is_empty() {
            return Value::Null;
        }

        // Clone all bar data for storage
        let bars_clone: Vec<(Vec<String>, Vec<f64>, u32)> = bar_data.iter().map(|(x, y, c)| {
            (x.clone(), y.clone(), *c)
        }).collect();
        let xlabel_clone = xlabel.clone();
        let ylabel_clone = ylabel.clone();

        // Create window with default size for charts
        let window_width = 800;
        let window_height = 600;
        let window_title = title.unwrap_or_else(|| "Bar Chart".to_string());

        let window_args = vec![
            Value::Number(window_width as f64),
            Value::Number(window_height as f64),
            Value::String(window_title),
        ];
        let window_result = native_plot_window(&window_args);
        let window = match window_result {
            Value::Window(w) => w,
            _ => return Value::Null,
        };

        // Send command to GUI thread to update chart data
        let plot_system = get_plot_system();
        let proxy = plot_system.proxy();
        
        let chart_data = CommandChartData {
            chart_type: CommandChartType::Bar,
                lines: Vec::new(),
            bars: bars_clone.clone(),
                pies: Vec::new(),
                heatmaps: Vec::new(),
            xlabel: xlabel_clone.clone(),
            ylabel: ylabel_clone.clone(),
                pie_rotation: 0.0,
        };
        
        let cmd = GuiCommand::UpdateChart {
            window_id: window.id(),
            chart_data,
        };
        
        if let Err(_e) = proxy.send_event(cmd) {
            return Value::Null;
        }

        // Wait for window to close
        let wait_args = vec![Value::Window(window)];
        native_plot_wait(&wait_args);

        return Value::Null;
    }

    // If we have heatmap data and no image arguments, render heatmap
    if has_heatmap_data {
        // Check if first argument is an object with title key (named arguments)
        let title = if !args.is_empty() {
            match &args[0] {
                Value::Object(map) => {
                    if let Some(Value::String(s)) = map.get("title") {
                        Some(s.clone())
                    } else {
                        None
                    }
                }
                Value::String(s) => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        };

        // Get plot state data
        let (xlabel, ylabel, heatmap_data) = PLOT_STATE.with(|state| {
            let plot_state = state.borrow();
            (
                plot_state.xlabel.clone(),
                plot_state.ylabel.clone(),
                plot_state.heatmap_data.clone(),
            )
        });

        // Clear plot state after reading
        PLOT_STATE.with(|state| {
            let mut plot_state = state.borrow_mut();
            plot_state.xlabel = None;
            plot_state.ylabel = None;
            plot_state.heatmap_data.clear();
        });

        // Check if we have heatmap data
        if heatmap_data.is_empty() {
            return Value::Null;
        }

        // Clone all heatmap data for storage
        let heatmaps_clone: Vec<(Vec<Vec<f64>>, Option<f64>, Option<f64>, String)> = heatmap_data.iter().map(|(d, min, max, p)| {
            (d.clone(), *min, *max, p.clone())
        }).collect();
        let xlabel_clone = xlabel.clone();
        let ylabel_clone = ylabel.clone();

        // Create window with default size for charts
        let window_width = 800;
        let window_height = 600;
        let window_title = title.unwrap_or_else(|| "Heatmap".to_string());

        let window_args = vec![
            Value::Number(window_width as f64),
            Value::Number(window_height as f64),
            Value::String(window_title),
        ];
        let window_result = native_plot_window(&window_args);
        let window = match window_result {
            Value::Window(w) => w,
            _ => return Value::Null,
        };

        // Send command to GUI thread to update chart data
        let plot_system = get_plot_system();
        let proxy = plot_system.proxy();
        
        let chart_data = CommandChartData {
            chart_type: CommandChartType::Heatmap,
                lines: Vec::new(),
                bars: Vec::new(),
                pies: Vec::new(),
                heatmaps: heatmaps_clone,
                xlabel: xlabel_clone,
                ylabel: ylabel_clone,
                pie_rotation: 0.0,
        };
        
        let cmd = GuiCommand::UpdateChart {
            window_id: window.id(),
            chart_data,
        };
        
        if let Err(_e) = proxy.send_event(cmd) {
            return Value::Null;
        }

        // Wait for window to close
        let wait_args = vec![Value::Window(window)];
        native_plot_wait(&wait_args);

        return Value::Null;
    }

    // If we have line data and no image arguments, render line chart
    if has_line_data {
        // Check if first argument is an object with title key (named arguments)
        let title = if !args.is_empty() {
            match &args[0] {
                Value::Object(map) => {
                    if let Some(Value::String(s)) = map.get("title") {
                        Some(s.clone())
                    } else {
                        None
                    }
                }
                Value::String(s) => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        };

        // Get plot state data
        let (xlabel, ylabel, line_data) = PLOT_STATE.with(|state| {
            let plot_state = state.borrow();
            (
                plot_state.xlabel.clone(),
                plot_state.ylabel.clone(),
                plot_state.line_data.clone(),
            )
        });

        // Clear plot state after reading
        PLOT_STATE.with(|state| {
            let mut plot_state = state.borrow_mut();
            plot_state.xlabel = None;
            plot_state.ylabel = None;
            plot_state.line_data.clear();
        });

        // Check if we have line data
        if line_data.is_empty() {
            return Value::Null;
        }

        // Clone all line data for storage
        let lines_clone: Vec<(Vec<f64>, Vec<f64>, bool, usize, usize, u32)> = line_data.iter().map(|(x, y, sp, ps, lw, c)| {
            (x.clone(), y.clone(), *sp, *ps, *lw, *c)
        }).collect();
        let xlabel_clone = xlabel.clone();
        let ylabel_clone = ylabel.clone();

        // Create window with default size for charts
        let window_width = 800;
        let window_height = 600;
        let window_title = title.unwrap_or_else(|| "Line Chart".to_string());

        let window_args = vec![
            Value::Number(window_width as f64),
            Value::Number(window_height as f64),
            Value::String(window_title),
        ];
        let window_result = native_plot_window(&window_args);
        let window = match window_result {
            Value::Window(w) => w,
            _ => return Value::Null,
        };

        // Send command to GUI thread to update chart data
        let plot_system = get_plot_system();
        let proxy = plot_system.proxy();
        
        let chart_data = CommandChartData {
            chart_type: CommandChartType::Line,
            lines: lines_clone.clone(),
                bars: Vec::new(),
                pies: Vec::new(),
                heatmaps: Vec::new(),
            xlabel: xlabel_clone.clone(),
            ylabel: ylabel_clone.clone(),
                pie_rotation: 0.0,
        };
        
        let cmd = GuiCommand::UpdateChart {
            window_id: window.id(),
            chart_data,
        };
        
        if let Err(_e) = proxy.send_event(cmd) {
            return Value::Null;
        }

        // Wait for window to close
        let wait_args = vec![Value::Window(window)];
        native_plot_wait(&wait_args);

        return Value::Null;
    }

    // Original image handling logic
    if args.is_empty() {
        return Value::Null;
    }

    if args.len() > 2 {
        return Value::Null;
    }

    // Extract image (from path or tensor)
    let image = match &args[0] {
        Value::String(_) | Value::Path(_) => {
            // Load from path
            let image_result = native_plot_image(&args[0..1]);
            match image_result {
                Value::Image(img) => img,
                _ => return Value::Null,
            }
        }
        Value::Tensor(tensor_ref) => {
            // Convert tensor to image
            let tensor = tensor_ref.borrow();
            match Image::from_tensor(&tensor) {
                Ok(img) => Rc::new(RefCell::new(img)),
                Err(_) => return Value::Null,
            }
        }
        _ => return Value::Null,
    };

    // Extract title (optional)
    let title = if args.len() == 2 {
        match &args[1] {
            Value::String(s) => s.clone(),
            Value::Number(n) => {
                // Convert number to string for title
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            _ => "Image Viewer".to_string(),
        }
    } else {
        "Image Viewer".to_string()
    };

    let img_ref = image.borrow();
    
    // Calculate scale factor (like matplotlib - scale small images)
    // For images <= 100px, use scale 10, otherwise scale 4
    let scale = if img_ref.width <= 100 && img_ref.height <= 100 {
        10
    } else if img_ref.width <= 200 && img_ref.height <= 200 {
        5
    } else {
        4
    };
    
    // Create window with scaled dimensions (like matplotlib imshow)
    let width = img_ref.width * scale;
    let height = img_ref.height * scale;
    drop(img_ref);

    let window_args = vec![
        Value::Number(width as f64),
        Value::Number(height as f64),
        Value::String(title),
    ];
    let window_result = native_plot_window(&window_args);
    let window = match window_result {
        Value::Window(w) => w,
        _ => return Value::Null,
    };

    // Draw image
    let draw_args = vec![
        Value::Window(window.clone()),
        Value::Image(image),
    ];
    native_window_draw(&draw_args);

    // Wait for window to close
    let wait_args = vec![Value::Window(window)];
    native_plot_wait(&wait_args);

    Value::Null
}

/// Show multiple images in a grid layout
/// plot.show_grid(images, rows, cols) -> Null
/// plot.show_grid(images) -> Null (auto-calculate grid)
pub fn native_plot_show_grid(args: &[Value]) -> Value {
    if args.is_empty() || args.len() > 3 {
        return Value::Null;
    }

    // Extract images array
    let images_array = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            arr_ref.clone()
        }
        _ => return Value::Null,
    };

    if images_array.is_empty() {
        return Value::Null;
    }

    // Calculate or extract rows and cols
    let (rows, cols) = if args.len() == 3 {
        // Both rows and cols provided
        let rows_val = match &args[1] {
            Value::Number(n) => *n as usize,
            _ => return Value::Null,
        };
        let cols_val = match &args[2] {
            Value::Number(n) => *n as usize,
            _ => return Value::Null,
        };
        (rows_val, cols_val)
    } else if args.len() == 2 {
        // Only one dimension provided - assume it's total, calculate square grid
        let total = match &args[1] {
            Value::Number(n) => *n as usize,
            _ => return Value::Null,
        };
        let dim = (total as f64).sqrt().ceil() as usize;
        (dim, dim)
    } else {
        // Auto-calculate grid: find smallest square that fits all images
        let num_images = images_array.len();
        let dim = (num_images as f64).sqrt().ceil() as usize;
        (dim, dim)
    };

    // Convert all images/tensors to Image objects
    let mut images = Vec::new();
    let mut titles = Vec::new();
    
    for (idx, val) in images_array.iter().enumerate() {
        let image = match val {
            Value::String(_) | Value::Path(_) => {
                // Load from path
                let image_result = native_plot_image(&[val.clone()]);
                match image_result {
                    Value::Image(img) => img,
                    _ => continue, // Skip invalid images
                }
            }
            Value::Tensor(tensor_ref) => {
                // Convert tensor to image
                let tensor = tensor_ref.borrow();
                match Image::from_tensor(&tensor) {
                    Ok(img) => Rc::new(RefCell::new(img)),
                    Err(_) => continue, // Skip invalid tensors
                }
            }
            Value::Image(img) => img.clone(),
            _ => continue, // Skip invalid values
        };
        
        images.push(image);
        titles.push(format!("{}", idx));
    }

    if images.is_empty() {
        return Value::Null;
    }

    // Calculate window size based on grid
    // Estimate: each cell should be at least 150x150 pixels
    let cell_size = 150;
    let padding = 10;
    let window_width = cols * cell_size + (cols + 1) * padding;
    let window_height = rows * cell_size + (rows + 1) * padding;

    // Create window
    let window_args = vec![
        Value::Number(window_width as f64),
        Value::Number(window_height as f64),
        Value::String("Image Grid".to_string()),
    ];
    let window_result = native_plot_window(&window_args);
    let window = match window_result {
        Value::Window(w) => w,
        _ => return Value::Null,
    };

    // Convert images to Arc<Mutex<Image>> for Send
    let images_arc: Vec<Arc<Mutex<Image>>> = images.iter().map(|img_rc| {
        let img_ref = img_rc.borrow();
        let cloned_image = Image {
            width: img_ref.width,
            height: img_ref.height,
            data: img_ref.data.clone(),
        };
        Arc::new(Mutex::new(cloned_image))
    }).collect();
    
    // Send command to GUI thread to update image grid
    let plot_system = get_plot_system();
    let proxy = plot_system.proxy();
    
    let cmd = GuiCommand::UpdateImageGrid {
        window_id: window.id(),
        images: images_arc,
        rows,
        cols,
        titles,
    };
    
    if let Err(_e) = proxy.send_event(cmd) {
        return Value::Null;
    }

    // Wait for window to close
    let wait_args = vec![Value::Window(window)];
    native_plot_wait(&wait_args);

    Value::Null
}

/// Create a figure with subplots
/// plot.subplots(rows, cols) -> figure
/// plot.subplots(rows, cols, figsize=(width, height)) -> figure
pub fn native_plot_subplots(args: &[Value]) -> Value {
    if args.len() < 2 || args.len() > 3 {
        return Value::Null;
    }

    let rows = match &args[0] {
        Value::Number(n) => *n as usize,
        _ => return Value::Null,
    };

    let cols = match &args[1] {
        Value::Number(n) => *n as usize,
        _ => return Value::Null,
    };

    // Parse figsize if provided (as named argument or tuple)
    let figsize = if args.len() == 3 {
        match &args[2] {
            Value::Object(map) => {
                // Named arguments: figsize=(10, 10) becomes an object
                if let Some(Value::Array(arr)) = map.get("figsize") {
                    let arr_ref = arr.borrow();
                    if arr_ref.len() >= 2 {
                        let width = match &arr_ref[0] {
                            Value::Number(n) => *n,
                            _ => 10.0,
                        };
                        let height = match &arr_ref[1] {
                            Value::Number(n) => *n,
                            _ => 10.0,
                        };
                        (width, height)
                    } else {
                        (10.0, 10.0)
                    }
                } else {
                    (10.0, 10.0)
                }
            }
            Value::Array(arr) => {
                // Tuple: (10, 10)
                let arr_ref = arr.borrow();
                if arr_ref.len() >= 2 {
                    let width = match &arr_ref[0] {
                        Value::Number(n) => *n,
                        _ => 10.0,
                    };
                    let height = match &arr_ref[1] {
                        Value::Number(n) => *n,
                        _ => 10.0,
                    };
                    (width, height)
                } else {
                    (10.0, 10.0)
                }
            }
            _ => (10.0, 10.0),
        }
    } else {
        (10.0, 10.0)
    };

    // Create figure
    let figure = Figure::new(rows, cols, figsize);
    let figure_rc = Rc::new(RefCell::new(figure));
    
    // Return only the figure (axes accessible via fig.axes)
    Value::Figure(figure_rc)
}

/// Display image in axis
/// axis.imshow(image) -> Null
/// axis.imshow(image, cmap='gray') -> Null
pub fn native_axis_imshow(args: &[Value]) -> Value {
    for (_i, _arg) in args.iter().enumerate() {
    }
    if args.is_empty() || args.len() > 3 {
        return Value::Null;
    }

    // Try to find Axis in args - it might be in a different position due to argument ordering
    let axis = args.iter().find_map(|arg| {
        if let Value::Axis(a) = arg {
            Some(a.clone())
        } else {
            None
        }
    });
    
    let axis = match axis {
        Some(a) => {
            a
        },
        None => {
            return Value::Null;
        },
    };

    // Extract image (from tensor or Image)
    // Find tensor/image in args (skip the axis we already found)
    let image = args.iter().find_map(|arg| {
        match arg {
            Value::Tensor(tensor_ref) => {
                let tensor = tensor_ref.borrow();
                match Image::from_tensor(&tensor) {
                    Ok(img) => {
                        Some(Rc::new(RefCell::new(img)))
                    },
                    Err(_e) => {
                        None
                    },
                }
            }
            Value::Image(img) => {
                Some(img.clone())
            }
            Value::String(_) | Value::Path(_) => {
                let image_result = native_plot_image(&[arg.clone()]);
                match image_result {
                    Value::Image(img) => Some(img),
                    _ => {
                        None
                    },
                }
            }
            _ => None,
        }
    });
    
    let image = match image {
        Some(img) => img,
        None => {
            return Value::Null;
        },
    };

    // Extract cmap if provided - find String in args (skip axis and image)
    let cmap = args.iter().find_map(|arg| {
        match arg {
            Value::String(s) if s != "off" && s != "on" => {
                // This is likely cmap, not axis('off') or axis('on')
                Some(s.clone())
            }
            Value::Object(map) => {
                // Named arguments: cmap='gray' might be passed as object
                if let Some(Value::String(s)) = map.get("cmap") {
                    Some(s.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }).unwrap_or_else(|| {
        "gray".to_string()
    });

    // Update axis
    let mut axis_ref = axis.borrow_mut();
    axis_ref.image = Some(image.clone());
    axis_ref.cmap = cmap;
    
    // Debug: verify image was set
    let _img_ref = image.borrow();

    Value::Null
}

/// Set title for axis
/// axis.set_title(title) -> Null
pub fn native_axis_set_title(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    // Try to find Axis in args - it might be in a different position due to argument ordering
    let axis = args.iter().find_map(|arg| {
        if let Value::Axis(a) = arg {
            Some(a.clone())
        } else {
            None
        }
    });
    
    let axis = match axis {
        Some(a) => {
            a
        },
        None => {
            return Value::Null;
        },
    };

    // Find title (String or Number) in args
    let title = args.iter().find_map(|arg| {
        match arg {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    Some(format!("{}", *n as i64))
                } else {
                    Some(format!("{}", n))
                }
            }
            _ => None,
        }
    });
    
    let title = match title {
        Some(t) => {
            t
        },
        None => {
            return Value::Null;
        },
    };

    let mut axis_ref = axis.borrow_mut();
    axis_ref.title = Some(title);

    Value::Null
}

/// Control axis visibility
/// axis.axis('off') -> Null
/// axis.axis('on') -> Null
pub fn native_axis_axis(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    // Try to find Axis in args - it might be in a different position due to argument ordering
    let axis = args.iter().find_map(|arg| {
        if let Value::Axis(a) = arg {
            Some(a.clone())
        } else {
            None
        }
    });
    
    let axis = match axis {
        Some(a) => a,
        None => return Value::Null,
    };

    // Find mode string ('on' or 'off') in args
    let mode = args.iter().find_map(|arg| {
        if let Value::String(s) = arg {
            Some(s.clone())
        } else {
            None
        }
    });
    
    let mode = match mode {
        Some(m) => m,
        None => return Value::Null,
    };

    let mut axis_ref = axis.borrow_mut();
    axis_ref.axis_visible = mode == "on";

    Value::Null
}

/// Apply tight layout to figure
/// plot.tight_layout(figure) -> Null
pub fn native_plot_tight_layout(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let figure = match &args[0] {
        Value::Figure(f) => f.clone(),
        _ => return Value::Null,
    };

    let mut figure_ref = figure.borrow_mut();
    figure_ref.tight_layout = true;

    Value::Null
}

/// Show figure with all axes
/// plot.show(figure) -> Null
/// plot.show(figure, title) -> Null
pub fn native_plot_show_figure(args: &[Value]) -> Value {
    if args.len() < 1 || args.len() > 2 {
        return Value::Null;
    }

    let figure = match &args[0] {
        Value::Figure(f) => f.clone(),
        _ => return Value::Null,
    };

    // Extract title (optional)
    let title = if args.len() == 2 {
        match &args[1] {
            Value::String(s) => s.clone(),
            Value::Number(n) => {
                // Convert number to string for title
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            _ => "Figure".to_string(),
        }
    } else {
        "Figure".to_string()
    };

    let figure_ref = figure.borrow();
    let rows = figure_ref.axes.len();
    let _cols = if rows > 0 { figure_ref.axes[0].len() } else { 0 };

    // Convert figsize to pixels (figsize is in "figure units", convert to pixels)
    // Default: 1 unit = 100 pixels, so (10, 10) = 1000x1000 pixels
    let scale_factor = 100.0;
    let window_width = (figure_ref.figsize.0 * scale_factor) as u32;
    let window_height = (figure_ref.figsize.1 * scale_factor) as u32;

    // Create window
    let window_args = vec![
        Value::Number(window_width as f64),
        Value::Number(window_height as f64),
        Value::String(title),
    ];
    let window_result = native_plot_window(&window_args);
    let window = match window_result {
        Value::Window(w) => w,
        _ => return Value::Null,
    };

    // Get window ID
    let window_id = window.id();

    // Extract figure data for sending to GUI thread
    let figure_data = {
        let figure_ref = figure.borrow();
        let mut axes_data = Vec::new();
        
        for row in &figure_ref.axes {
            let mut row_data = Vec::new();
            for axis_rc in row {
                let axis_ref = axis_rc.borrow();
                let axis_data = AxisData {
                    image: axis_ref.image.as_ref().map(|img_rc| {
                        // Convert Rc<RefCell<Image>> to Arc<Mutex<Image>>
                        let img = img_rc.borrow();
                        Arc::new(Mutex::new(Image {
                            width: img.width,
                            height: img.height,
                            data: img.data.clone(),
                        }))
                    }),
                    title: axis_ref.title.clone(),
                    axis_visible: axis_ref.axis_visible,
                    cmap: axis_ref.cmap.clone(),
                };
                row_data.push(axis_data);
            }
            axes_data.push(row_data);
        }
        
        FigureData {
            axes: axes_data,
            tight_layout: figure_ref.tight_layout,
        }
    };
    
    // Send command to GUI thread to update figure
    let plot_system = get_plot_system();
    let proxy = plot_system.proxy();
    
    let command = GuiCommand::UpdateFigure {
        window_id,
        figure_data,
    };
    
    if let Err(_e) = proxy.send_event(command) {
        return Value::Null;
    }

    // Wait for window to close
    let wait_args = vec![Value::Window(window)];
    native_plot_wait(&wait_args);

    Value::Null
}

/// Set x-axis label
/// plot.xlabel(label) -> Null
pub fn native_plot_xlabel(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let label = match &args[0] {
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        _ => return Value::Null,
    };

    PLOT_STATE.with(|state| {
        let mut plot_state = state.borrow_mut();
        plot_state.xlabel = Some(label);
    });

    Value::Null
}

/// Set y-axis label
/// plot.ylabel(label) -> Null
pub fn native_plot_ylabel(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let label = match &args[0] {
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        _ => return Value::Null,
    };

    PLOT_STATE.with(|state| {
        let mut plot_state = state.borrow_mut();
        plot_state.ylabel = Some(label);
    });

    Value::Null
}

/// Parse color string to u32 (BGRA format: 0xAABBGGRR)
/// Supports named colors: blue, green, red, black, white
/// Supports hex format: #RRGGBB or RRGGBB
/// Returns default blue (0xFF00BFFF) on error
fn parse_color(color_str: &str) -> u32 {
    let color_lower = color_str.to_lowercase();
    
    // Named colors
    match color_lower.as_str() {
        "blue" => 0xFF00BFFF,   // Deep sky blue
        "green" => 0xFF00FF00, // Lime green
        "red" => 0xFFFF0000,   // Red
        "black" => 0xFF000000,  // Black
        "white" => 0xFFFFFFFF,  // White
        _ => {
            // Try to parse as hex color
            let hex_str = if color_str.starts_with('#') {
                &color_str[1..]
            } else {
                color_str
            };
            
            if hex_str.len() == 6 {
                if let Ok(rgb) = u32::from_str_radix(hex_str, 16) {
                    let r = ((rgb >> 16) & 0xFF) as u8;
                    let g = ((rgb >> 8) & 0xFF) as u8;
                    let b = (rgb & 0xFF) as u8;
                    // Convert RGB to BGRA format: 0xAABBGGRR
                    return (0xFF << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32);
                }
            }
            
            // Default to blue on error
            0xFF00BFFF
        }
    }
}

/// Add line data to plot
/// plot.line(x, y) -> Null
/// plot.line(x, y, show_points) -> Null
/// plot.line(x, y, point_size=5, line_width=2) -> Null
/// plot.line(x, y, color="blue") -> Null
/// plot.line(x, y, color="#a434eb") -> Null
pub fn native_plot_line(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    // Extract x array
    let x_array = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut x_data = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => x_data.push(*n),
                    _ => return Value::Null, // Invalid data type in x array
                }
            }
            x_data
        }
        _ => return Value::Null,
    };

    // Extract y array
    let y_array = match &args[1] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut y_data = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => y_data.push(*n),
                    _ => return Value::Null, // Invalid data type in y array
                }
            }
            y_data
        }
        _ => return Value::Null,
    };

    // Check that arrays have the same length
    if x_array.len() != y_array.len() {
        return Value::Null;
    }

    // Check that arrays are not empty
    if x_array.is_empty() {
        return Value::Null;
    }

    // Default values
    let mut show_points = false;
    let mut point_size = 5usize; // Default: 5 pixels (increased from 3)
    let mut line_width = 2usize; // Default: 2 pixels (increased from 1)
    let mut color = 0xFF00BFFF; // Default: blue (Deep sky blue)

    // Handle positional arguments first (for backward compatibility)
    if args.len() >= 3 {
        match &args[2] {
            Value::Bool(b) => {
                // Positional boolean argument: show_points
                show_points = *b;
            }
            Value::Number(n) => {
                // Positional number argument: treat as point_size for backward compatibility
                point_size = (*n as usize).max(1).min(50);
            }
            _ => {}
        }
    }

    // Extract named parameters from all Object arguments
    // Named arguments can appear anywhere after the first two positional arguments
    // They can be in separate Object arguments or combined in one Object
    // Also check all arguments (not just skip(2)) in case named args come before positional
    for arg in args.iter() {
        if let Value::Object(map) = arg {
            // Extract all named parameters from this Object
            if let Some(Value::Bool(b)) = map.get("marker") {
                show_points = *b;
            }
            if let Some(Value::Bool(b)) = map.get("show_points") {
                show_points = *b;
            }
            
            if let Some(Value::Number(n)) = map.get("point_size") {
                point_size = (*n as usize).max(1).min(50); // Clamp between 1 and 50
            }
            
            if let Some(Value::Number(n)) = map.get("line_width") {
                line_width = (*n as usize).max(1).min(20); // Clamp between 1 and 20
            }
            
            if let Some(Value::String(s)) = map.get("color") {
                color = parse_color(s);
            }
        }
    }
    
    // Handle case where color is passed as a positional string argument after positional args
    // This happens when color="blue" is compiled as a positional argument
    // Check if there's a string argument after positional args that looks like a color
    if args.len() >= 4 {
        // Check arguments after the first 3 (x, y, show_points/point_size)
        for arg in args.iter().skip(3) {
            if let Value::String(s) = arg {
                // Check if this string looks like a color (named color or hex)
                let s_lower = s.to_lowercase();
                let is_named_color = matches!(s_lower.as_str(), "blue" | "green" | "red" | "black" | "white");
                let is_hex_color = s.starts_with('#') && s.len() == 7 && s[1..].chars().all(|c| c.is_ascii_hexdigit())
                    || !s.starts_with('#') && s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit());
                
                if is_named_color || is_hex_color {
                    color = parse_color(s);
                    break; // Use first valid color string found
                }
            }
        }
    }

    // Debug: print color information
    for (_i, arg) in args.iter().enumerate() {
        match arg {
            Value::Object(map) => {
                if let Some(Value::String(_s)) = map.get("color") {
                    // Color found in object
                }
            }
            _ => {
                // Other value types don't need special handling here
            }
        }
    }
    // Add line data to plot state
    PLOT_STATE.with(|state| {
        let mut plot_state = state.borrow_mut();
        plot_state.line_data.push((x_array, y_array, show_points, point_size, line_width, color));
    });

    Value::Null
}

/// Add bar data to plot
/// plot.bar(x, y) -> Null
/// plot.bar(x, y, color="blue") -> Null
/// plot.bar(x, y, color="#a434eb") -> Null
/// x can be array of strings (categories) or numbers (will be converted to strings)
pub fn native_plot_bar(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    // Extract x array (can be strings or numbers)
    let x_labels = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut labels = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::String(s) => labels.push(s.clone()),
                    Value::Number(n) => {
                        // Convert number to string
                        if n.fract() == 0.0 {
                            labels.push(format!("{}", *n as i64));
                        } else {
                            labels.push(format!("{}", n));
                        }
                    }
                    _ => return Value::Null, // Invalid data type in x array
                }
            }
            labels
        }
        _ => return Value::Null,
    };

    // Extract y array (must be numbers)
    let y_array = match &args[1] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut y_data = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => y_data.push(*n),
                    _ => return Value::Null, // Invalid data type in y array
                }
            }
            y_data
        }
        _ => return Value::Null,
    };

    // Check that arrays have the same length
    if x_labels.len() != y_array.len() {
        return Value::Null;
    }

    // Check that arrays are not empty
    if x_labels.is_empty() {
        return Value::Null;
    }

    // Default color
    let mut color = 0xFF00BFFF; // Default: blue (Deep sky blue)

    // Extract named parameters from all Object arguments
    for arg in args.iter() {
        if let Value::Object(map) = arg {
            if let Some(Value::String(s)) = map.get("color") {
                color = parse_color(s);
            }
        }
    }
    
    // Handle case where color is passed as a positional string argument
    if args.len() >= 3 {
        for arg in args.iter().skip(2) {
            if let Value::String(s) = arg {
                let s_lower = s.to_lowercase();
                let is_named_color = matches!(s_lower.as_str(), "blue" | "green" | "red" | "black" | "white");
                let is_hex_color = s.starts_with('#') && s.len() == 7 && s[1..].chars().all(|c| c.is_ascii_hexdigit())
                    || !s.starts_with('#') && s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit());
                
                if is_named_color || is_hex_color {
                    color = parse_color(s);
                    break;
                }
            }
        }
    }

    // Add bar data to plot state
    PLOT_STATE.with(|state| {
        let mut plot_state = state.borrow_mut();
        plot_state.bar_data.push((x_labels, y_array, color));
    });

    Value::Null
}

/// Add pie data to plot
/// plot.pie(x, y) -> Null
/// plot.pie(x, y, color="blue") -> Null
/// plot.pie(x, y, color="#a434eb") -> Null
/// x can be array of strings (categories) or numbers (will be converted to strings)
pub fn native_plot_pie(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    // Extract x array (can be strings or numbers)
    let x_labels = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut labels = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::String(s) => labels.push(s.clone()),
                    Value::Number(n) => {
                        // Convert number to string
                        if n.fract() == 0.0 {
                            labels.push(format!("{}", *n as i64));
                        } else {
                            labels.push(format!("{}", n));
                        }
                    }
                    _ => return Value::Null, // Invalid data type in x array
                }
            }
            labels
        }
        _ => return Value::Null,
    };

    // Extract y array (must be numbers)
    let y_array = match &args[1] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut y_data = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => y_data.push(*n),
                    _ => return Value::Null, // Invalid data type in y array
                }
            }
            y_data
        }
        _ => return Value::Null,
    };

    // Check that arrays have the same length
    if x_labels.len() != y_array.len() {
        return Value::Null;
    }

    // Check that arrays are not empty
    if x_labels.is_empty() {
        return Value::Null;
    }

    // Default color (will be overridden by palette in renderer if not specified)
    let mut color = 0xFF00BFFF; // Default: blue (Deep sky blue)

    // Extract named parameters from all Object arguments
    for arg in args.iter() {
        if let Value::Object(map) = arg {
            if let Some(Value::String(s)) = map.get("color") {
                color = parse_color(s);
            }
        }
    }
    
    // Handle case where color is passed as a positional string argument
    if args.len() >= 3 {
        for arg in args.iter().skip(2) {
            if let Value::String(s) = arg {
                let s_lower = s.to_lowercase();
                let is_named_color = matches!(s_lower.as_str(), "blue" | "green" | "red" | "black" | "white");
                let is_hex_color = s.starts_with('#') && s.len() == 7 && s[1..].chars().all(|c| c.is_ascii_hexdigit())
                    || !s.starts_with('#') && s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit());
                
                if is_named_color || is_hex_color {
                    color = parse_color(s);
                    break;
                }
            }
        }
    }

    // Add pie data to plot state
    PLOT_STATE.with(|state| {
        let mut plot_state = state.borrow_mut();
        plot_state.pie_data.push((x_labels, y_array, color));
    });

    Value::Null
}

/// Add heatmap data to plot
/// plot.heatmap(data) -> Null
/// plot.heatmap(data, min=0, max=100) -> Null
/// plot.heatmap(data, palette="red") -> Null
/// data must be a 2D array (array of arrays of numbers)
pub fn native_plot_heatmap(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    // Extract data array (must be 2D array)
    let heatmap_data = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut data = Vec::new();
            
            // Check if array is empty
            if arr_ref.is_empty() {
                return Value::Null;
            }
            
            // Get first row to determine column count
            let first_row = match &arr_ref[0] {
                Value::Array(row) => {
                    let row_ref = row.borrow();
                    let mut row_data = Vec::new();
                    for val in row_ref.iter() {
                        match val {
                            Value::Number(n) => row_data.push(*n),
                            _ => return Value::Null, // Invalid data type in row
                        }
                    }
                    if row_data.is_empty() {
                        return Value::Null;
                    }
                    row_data
                }
                _ => return Value::Null, // First element must be an array
            };
            
            let col_count = first_row.len();
            data.push(first_row);
            
            // Process remaining rows
            for val in arr_ref.iter().skip(1) {
                match val {
                    Value::Array(row) => {
                        let row_ref = row.borrow();
                        let mut row_data = Vec::new();
                        for val in row_ref.iter() {
                            match val {
                                Value::Number(n) => row_data.push(*n),
                                _ => return Value::Null, // Invalid data type in row
                            }
                        }
                        // Check that all rows have the same length
                        if row_data.len() != col_count {
                            return Value::Null;
                        }
                        data.push(row_data);
                    }
                    _ => return Value::Null, // All elements must be arrays
                }
            }
            
            data
        }
        _ => return Value::Null,
    };

    // Check that data is not empty
    if heatmap_data.is_empty() || heatmap_data[0].is_empty() {
        return Value::Null;
    }

    // Default values
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;
    let mut palette = "green".to_string();

    // Extract named parameters from all Object arguments
    for arg in args.iter() {
        if let Value::Object(map) = arg {
            if let Some(Value::Number(n)) = map.get("min") {
                min_val = Some(*n);
            }
            if let Some(Value::Number(n)) = map.get("max") {
                max_val = Some(*n);
            }
            if let Some(Value::String(s)) = map.get("palette") {
                let s_lower = s.to_lowercase();
                // Validate palette name
                if matches!(s_lower.as_str(), "green" | "red" | "blue" | "bw") {
                    palette = s_lower;
                }
            }
        }
    }
    
    // Handle case where parameters are passed as positional arguments
    if args.len() >= 2 {
        for arg in args.iter().skip(1) {
            if let Value::Number(n) = arg {
                // If min is not set, use this as min; otherwise use as max
                if min_val.is_none() {
                    min_val = Some(*n);
                } else if max_val.is_none() {
                    max_val = Some(*n);
                }
            } else if let Value::String(s) = arg {
                let s_lower = s.to_lowercase();
                if matches!(s_lower.as_str(), "green" | "red" | "blue" | "bw") {
                    palette = s_lower;
                }
            }
        }
    }

    // Add heatmap data to plot state
    PLOT_STATE.with(|state| {
        let mut plot_state = state.borrow_mut();
        plot_state.heatmap_data.push((heatmap_data, min_val, max_val, palette));
    });

    Value::Null
}

