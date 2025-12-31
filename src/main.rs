// Main entry point для DataCode интерпретатора

use data_code::{run, run_with_vm};
use data_code::sqlite_export;
use data_code::plot::{GuiCommand, system, natives, WindowState, RenderContent};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use winit::event_loop::EventLoopBuilder;
use winit::event::{Event, WindowEvent};
use winit::window::WindowBuilder;
use winit::event_loop::ControlFlow;
use data_code::plot::window::Window as PlotWindow;
use data_code::plot::renderer::Renderer;
use data_code::plot::window::ImageViewState;
use data_code::plot::command::ChartType;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

const VERSION: &str = env!("CARGO_PKG_VERSION");


fn print_help() {
    println!("🧠 DataCode - Interactive Programming Language");
    println!();
    println!("Usage:");
    println!("  datacode                   # Start interactive REPL (default)");
    println!("  datacode main.dc           # Execute DataCode file");
    println!("  datacode main.dc --build_model  # Execute and export tables to SQLite");
    println!("  datacode main.dc --build_model output.db  # Export to specific file");
    println!("  datacode --websocket       # Start WebSocket server for remote code execution");
    println!("  datacode --help            # Show this help");
    println!();
    println!("File Execution:");
    println!("  • Create files with .dc extension");
    println!("  • Write DataCode programs in files");
    println!("  • Execute with: datacode filename.dc");

    println!();
    println!("SQLite Export (--build_model):");
    println!("  • Exports all tables from global variables to SQLite database");
    println!("  • Automatically detects foreign key relationships");
    println!("  • Creates metadata table _datacode_variables with all variable info");
    println!("  • Default output: <script_name>.db");
    println!("  • Custom output: --build_model output.db");
    println!("  • Environment variable: DATACODE_SQLITE_OUTPUT=path.db");
    println!();
    println!("WebSocket Server:");
    println!("  • Start server: datacode --websocket");
    println!("  • Default address: ws://127.0.0.1:8080");
    println!("  • Custom host/port: datacode --websocket --host 0.0.0.0 --port 8899");
    println!("  • Or use env var: DATACODE_WS_ADDRESS=0.0.0.0:3000 datacode --websocket");
    println!("  • Virtual environment mode: datacode --websocket --use-ve");
    println!("    - Creates isolated session folders in src/temp_sessions");
    println!("    - getcwd() returns empty string");
    println!("    - Supports file uploads via upload_file request");
    println!("    - Session folder is deleted on disconnect");
    println!("  • Send JSON: {{\"code\": \"print('Hello World')\"}}");
    println!("  • Receive JSON: {{\"success\": true, \"output\": \"Hello World\\n\", \"error\": null}}");
    println!("  • Upload file: {{\"type\": \"upload_file\", \"filename\": \"test.txt\", \"content\": \"...\"}}");
    println!();
    println!("Features:");
    println!("  • Interactive REPL with multiline support");
    println!("  • User-defined functions with local scope");
    println!("  • Arithmetic and logical operations");
    println!("  • File system operations");
    println!("  • For loops and control structures");
    println!("  • Improved error messages with line numbers");
    println!("  • Path manipulation");
    println!("  • Functional programming methods (map, filter, reduce)");
    println!("  • WebSocket server for remote code execution");
    println!();
    println!("Example DataCode file (example.dc):");
    println!("  # Simple DataCode program");
    println!("  fn greet(name) {{");
    println!("      return 'Hello, ' + name + '!'");
    println!("  }}");
    println!("  ");
    println!("  global message = greet('DataCode')");
    println!("  print(message)");
    println!();
    println!("Run with: datacode example.dc");
    println!("Debug run: datacode example.dc --debug");
}


fn print_version() {
    println!("DataCode v{}", VERSION);
}

fn start_websocket_server(host: String, port: u16, use_ve: bool) {
    let address = format!("{}:{}", host, port);
    
    println!("🚀 Запуск WebSocket сервера DataCode...");
    println!("📡 Адрес: ws://{}", address);
    if use_ve {
        println!("📁 Режим виртуальной среды: включен (--use-ve)");
    }
    println!("💡 Используйте --host и --port для изменения адреса");
    println!("💡 Или переменную окружения DATACODE_WS_ADDRESS");
    println!();
    
    // Создаем tokio runtime для асинхронного выполнения
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    if let Err(e) = rt.block_on(data_code::websocket::start_server(&address, use_ve)) {
        eprintln!("❌ Ошибка запуска WebSocket сервера: {}", e);
        std::process::exit(1);
    }
}

/// Global window storage in GUI thread
/// Windows NEVER leave this thread - runtime communicates via commands
static WINDOWS: LazyLock<Mutex<HashMap<winit::window::WindowId, WindowState>>> = 
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Run DataCode code with EventLoop (for plot support)
/// Creates EventLoop in main thread, spawns runtime in separate thread
fn run_with_event_loop<F>(code_runner: F) -> Result<(), String>
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
    let runtime_handle = thread::spawn(move || {
        let result = code_runner();
        // Mark runtime as finished when done
        *runtime_finished_clone.lock().unwrap() = true;
        result
    });
    
    // Run event loop (blocking)
    // All windows are stored in WINDOWS HashMap in GUI thread
    let runtime_finished_event_loop = runtime_finished.clone();
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Wait);
        
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
            Event::WindowEvent { event: WindowEvent::CloseRequested, window_id, .. } => {
                
                // Remove window from storage and notify waiter
                let should_exit = {
                    let mut windows = WINDOWS.lock().unwrap();
                    if let Some(state) = windows.remove(&window_id) {
                        // Notify waiter if exists
                        if let Some(wait) = state.wait {
                            let (lock, cvar) = &*wait;
                            let mut done = lock.lock().unwrap();
                            *done = true;
                            cvar.notify_all();
                        }
                    }
                    // Check if all windows are closed
                    windows.is_empty()
                };
                
                // Also notify via PlotSystem (for backward compatibility)
                system::PlotSystem::notify_window_closed(window_id);
                
                // If all windows are closed and runtime has finished, exit event loop
                if should_exit {
                    let runtime_done = *runtime_finished_event_loop.lock().unwrap();
                    if runtime_done {
                        elwt.exit();
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, window_id, .. } => {
                // Handle cursor movement - update cursor position and find nearest point for line charts
                if let Ok(mut windows) = WINDOWS.lock() {
                    if let Some(state) = windows.get_mut(&window_id) {
                        // Update cursor position in screen coordinates
                        let cursor_pos = (position.x as f32, position.y as f32);
                        state.cursor_pos = Some(cursor_pos);
                        
                        // Find nearest point if this is a line chart
                        if let RenderContent::Chart(chart_data) = &state.content {
                            if let ChartType::Line = chart_data.chart_type {
                                // Get physical window size (same as buffer size in draw_line_chart)
                                // Use window's physical_size to get physical pixels, accounting for DPI scaling
                                let (buffer_width, buffer_height) = state.window.physical_size();
                                
                                // Calculate plot bounds (same as in draw_line_chart)
                                let left_margin = 200;
                                let right_margin = 40;
                                let top_margin = 60;
                                let bottom_margin = 80;
                                let plot_width = buffer_width.saturating_sub(left_margin + right_margin);
                                let plot_height = buffer_height.saturating_sub(top_margin + bottom_margin);
                                let plot_x = left_margin;
                                let plot_y = top_margin;
                                
                                // Calculate data bounds
                                let mut x_min = f64::INFINITY;
                                let mut x_max = f64::NEG_INFINITY;
                                let mut y_min = f64::INFINITY;
                                let mut y_max = f64::NEG_INFINITY;
                                
                                for (x_data, y_data, _, _, _, _) in &chart_data.lines {
                                    for &x_val in x_data {
                                        x_min = x_min.min(x_val);
                                        x_max = x_max.max(x_val);
                                    }
                                    for &y_val in y_data {
                                        y_min = y_min.min(y_val);
                                        y_max = y_max.max(y_val);
                                    }
                                }
                                
                                let x_range = if x_max > x_min { x_max - x_min } else { 1.0 };
                                let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };
                                let x_padding = x_range * 0.05;
                                let y_padding = y_range * 0.05;
                                let x_min_plot = x_min - x_padding;
                                let x_max_plot = x_max + x_padding;
                                let y_min_plot = y_min - y_padding;
                                let y_max_plot = y_max + y_padding;
                                
                                // Find nearest point and store in hovered_point
                                state.hovered_point = Renderer::find_nearest_point_across_lines(
                                    &chart_data.lines,
                                    cursor_pos.0,
                                    cursor_pos.1,
                                    plot_x as usize,
                                    plot_y as usize,
                                    plot_width as usize,
                                    plot_height as usize,
                                    x_min_plot,
                                    x_max_plot,
                                    y_min_plot,
                                    y_max_plot,
                                );
                            } else {
                                // Not a line chart, clear hovered point
                                state.hovered_point = None;
                            }
                        } else {
                            // Not a chart, clear hovered point
                            state.hovered_point = None;
                        }
                        
                        // Request redraw to update coordinate display and hover effects
                        state.window.request_redraw();
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: button_state, button: winit::event::MouseButton::Left, .. }, window_id, .. } => {
                // Handle mouse click - select point on line chart
                if button_state == winit::event::ElementState::Pressed {
                    if let Ok(mut windows) = WINDOWS.lock() {
                        if let Some(state) = windows.get_mut(&window_id) {
                                    // Find nearest point if cursor is over a line chart
                                    if let RenderContent::Chart(chart_data) = &state.content {
                                        if let ChartType::Line = chart_data.chart_type {
                                            if let Some(cursor_pos) = state.cursor_pos {
                                                // Get physical window size (same as buffer size in draw_line_chart)
                                                let (buffer_width, buffer_height) = state.window.physical_size();
                                                
                                                // Calculate plot bounds (same as in draw_line_chart)
                                                let left_margin = 200;
                                                let right_margin = 40;
                                                let top_margin = 60;
                                                let bottom_margin = 80;
                                                let plot_width = buffer_width.saturating_sub(left_margin + right_margin);
                                                let plot_height = buffer_height.saturating_sub(top_margin + bottom_margin);
                                                let plot_x = left_margin;
                                                let plot_y = top_margin;
                                        
                                        // Calculate data bounds
                                        let mut x_min = f64::INFINITY;
                                        let mut x_max = f64::NEG_INFINITY;
                                        let mut y_min = f64::INFINITY;
                                        let mut y_max = f64::NEG_INFINITY;
                                        
                                        for (x_data, y_data, _, _, _, _) in &chart_data.lines {
                                            for &x_val in x_data {
                                                x_min = x_min.min(x_val);
                                                x_max = x_max.max(x_val);
                                            }
                                            for &y_val in y_data {
                                                y_min = y_min.min(y_val);
                                                y_max = y_max.max(y_val);
                                            }
                                        }
                                        
                                        let x_range = if x_max > x_min { x_max - x_min } else { 1.0 };
                                        let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };
                                        let x_padding = x_range * 0.05;
                                        let y_padding = y_range * 0.05;
                                        let x_min_plot = x_min - x_padding;
                                        let x_max_plot = x_max + x_padding;
                                        let y_min_plot = y_min - y_padding;
                                        let y_max_plot = y_max + y_padding;
                                        
                                        // Find nearest point and select it
                                        // Use associated function syntax
                                        if let Some(nearest_point) = Renderer::find_nearest_point_across_lines(
                                            &chart_data.lines,
                                            cursor_pos.0,
                                            cursor_pos.1,
                                            plot_x as usize,
                                            plot_y as usize,
                                            plot_width as usize,
                                            plot_height as usize,
                                            x_min_plot,
                                            x_max_plot,
                                            y_min_plot,
                                            y_max_plot,
                                        ) {
                                            state.selected_point = Some(nearest_point);
                                            state.window.request_redraw();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, window_id, .. } => {
                // Handle redraw - render window content
                if let Ok(mut windows) = WINDOWS.lock() {
                    if let Some(state) = windows.get_mut(&window_id) {
                        // Create renderer if needed
                        if state.renderer.is_none() {
                            match Renderer::new(&state.window) {
                                Ok(renderer) => state.renderer = Some(renderer),
                                Err(_e) => {
                                    return;
                                }
                            }
                        }
                        
                        // Render based on content type from WindowState
                        if let Some(ref mut renderer) = state.renderer {
                            match &state.content {
                                RenderContent::Image(image, view_state) => {
                                    let img = image.lock().unwrap();
                                    let _ = renderer.draw_image_with_transform(
                                        &*img,
                                        &state.window,
                                        view_state
                                    );
                                }
                                RenderContent::Chart(chart_data) => {
                                    match chart_data.chart_type {
                                        ChartType::Line => {
                                            // Use stored hovered_point from WindowState (calculated on CursorMoved)
                                            let hovered_point = state.hovered_point;
                                            
                                            let _ = renderer.draw_line_chart(
                                                &chart_data.lines,
                                                chart_data.xlabel.as_deref(),
                                                chart_data.ylabel.as_deref(),
                                                &state.window,
                                                state.cursor_pos, // cursor_pos from state
                                                hovered_point, // hovered_point from state
                                                state.selected_point, // selected_point from state
                                            );
                                        }
                                        ChartType::Bar => {
                                            let _ = renderer.draw_bar_chart(
                                                &chart_data.bars,
                                                chart_data.xlabel.as_deref(),
                                                chart_data.ylabel.as_deref(),
                                                &state.window,
                                            );
                                        }
                                        ChartType::Pie => {
                                            let _ = renderer.draw_pie_chart(
                                                &chart_data.pies,
                                                chart_data.xlabel.as_deref(),
                                                chart_data.ylabel.as_deref(),
                                                &state.window,
                                                chart_data.pie_rotation,
                                            );
                                        }
                                        ChartType::Heatmap => {
                                            let _ = renderer.draw_heatmap_chart(
                                                &chart_data.heatmaps,
                                                chart_data.xlabel.as_deref(),
                                                chart_data.ylabel.as_deref(),
                                                &state.window,
                                            );
                                        }
                                    }
                                }
                                RenderContent::ImageGrid { images, rows, cols, titles } => {
                                    let _ = renderer.draw_image_grid(
                                        images,
                                        *rows,
                                        *cols,
                                        Some(titles),
                                    );
                                }
                                RenderContent::Figure(figure_data) => {
                                    // Extract titles from figure data
                                    let mut titles = Vec::new();
                                    for row in &figure_data.axes {
                                        for axis_data in row {
                                            if let Some(ref title) = axis_data.title {
                                                titles.push(title.clone());
                                            } else {
                                                titles.push(String::new());
                                            }
                                        }
                                    }
                                    
                                    let rows = figure_data.axes.len();
                                    let cols = if rows > 0 { figure_data.axes[0].len() } else { 0 };
                                    
                                    let _ = renderer.draw_figure_from_data(
                                        figure_data.clone(),
                                        rows,
                                        cols,
                                        &titles,
                                    );
                                }
                                RenderContent::None => {
                                    // Empty window - nothing to render
                                }
                            }
                        }
                    }
                }
            }
            Event::AboutToWait => {
                // Check if we should exit: all windows closed and runtime finished
                let all_windows_closed = WINDOWS.lock().unwrap().is_empty();
                let runtime_done = *runtime_finished_event_loop.lock().unwrap();
                if all_windows_closed && runtime_done {
                    elwt.exit();
                }
            }
            _ => {}
        }
    }).map_err(|e| format!("Event loop error: {:?}", e))?;
    
    // Wait for runtime to finish
    runtime_handle.join().map_err(|_| "Runtime thread panicked".to_string())??;
    
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Обработка аргументов командной строки
    if args.len() > 1 {
        let arg = &args[1];
        
        // Проверка на опции
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return;
            }
            "-v" | "--version" => {
                print_version();
                return;
            }
            "--websocket" => {
                // Парсим аргументы для WebSocket сервера
                let mut host = "127.0.0.1".to_string();
                let mut port = 8080u16;
                let mut use_ve = false;
                
                // Проверяем переменную окружения
                if let Ok(ws_address) = env::var("DATACODE_WS_ADDRESS") {
                    if let Some(colon_pos) = ws_address.find(':') {
                        host = ws_address[..colon_pos].to_string();
                        if let Ok(p) = ws_address[colon_pos + 1..].parse::<u16>() {
                            port = p;
                        }
                    } else {
                        host = ws_address;
                    }
                }
                
                // Парсим аргументы командной строки
                let mut i = 2;
                while i < args.len() {
                    match args[i].as_str() {
                        "--host" => {
                            if i + 1 < args.len() {
                                host = args[i + 1].clone();
                                i += 2;
                            } else {
                                eprintln!("Ошибка: --host требует значение");
                                std::process::exit(1);
                            }
                        }
                        "--port" => {
                            if i + 1 < args.len() {
                                if let Ok(p) = args[i + 1].parse::<u16>() {
                                    port = p;
                                    i += 2;
                                } else {
                                    eprintln!("Ошибка: неверный номер порта");
                                    std::process::exit(1);
                                }
                            } else {
                                eprintln!("Ошибка: --port требует значение");
                                std::process::exit(1);
                            }
                        }
                        "--use-ve" => {
                            use_ve = true;
                            i += 1;
                        }
                        _ => {
                            eprintln!("Неизвестный аргумент: {}", args[i]);
                            std::process::exit(1);
                        }
                    }
                }
                
                start_websocket_server(host, port, use_ve);
                return;
            }
            _ => {
                // Проверка, что это не опция (начинается с -)
                if arg.starts_with('-') {
                    eprintln!("Неизвестная опция: {}", arg);
                    eprintln!("Используйте --help для справки");
                    std::process::exit(1);
                }
            }
        }
        
        // Запуск файла
        let filename = arg;
        
        // Проверка существования файла
        if !Path::new(filename).exists() {
            eprintln!("Ошибка: файл '{}' не найден", filename);
            std::process::exit(1);
        }
        
        // Проверка расширения файла (опционально, но полезно)
        if !filename.ends_with(".dc") {
            eprintln!("Предупреждение: файл '{}' не имеет расширения .dc", filename);
        }
        
        // Проверяем наличие флага --build_model
        let mut build_model = false;
        let mut output_db: Option<String> = None;
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--build_model" | "--build-model" => {
                    build_model = true;
                    // Проверяем следующий аргумент - может быть имя файла
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        output_db = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }
        
        // Определяем имя выходного файла для SQLite
        if build_model {
            let db_filename = if let Some(db) = output_db {
                db
            } else if let Ok(env_db) = env::var("DATACODE_SQLITE_OUTPUT") {
                env_db
            } else {
                // По умолчанию: имя скрипта с расширением .db
                let path = PathBuf::from(filename);
                let stem = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                format!("{}.db", stem)
            };
            
            // Чтение и выполнение файла с экспортом
            match fs::read_to_string(filename) {
                Ok(source) => {
                    match run_with_vm(&source) {
                        Ok((_, vm)) => {
                            // Экспортируем таблицы в SQLite
                            match sqlite_export::export_to_sqlite(&vm, &db_filename) {
                                Ok(_) => {
                                    println!("✅ База данных создана: {}", db_filename);
                                }
                                Err(e) => {
                                    eprintln!("❌ Ошибка экспорта в SQLite: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Ошибка выполнения: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Ошибка чтения файла '{}': {}", filename, e);
                    std::process::exit(1);
                }
            }
        } else {
            // Обычное выполнение без экспорта
            // Use run_with_event_loop to support plot functionality
            match fs::read_to_string(filename) {
                Ok(source) => {
                    let source_clone = source.clone();
                    match run_with_event_loop(move || {
                        run(&source_clone)
                            .map(|_| ()) // Ignore return value
                            .map_err(|e| e.to_string())
                    }) {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("Ошибка выполнения: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Ошибка чтения файла '{}': {}", filename, e);
                    std::process::exit(1);
                }
            }
        }
    } else {
        // REPL режим (интерактивный)
        // Run with EventLoop for plot support
        if let Err(e) = run_with_event_loop(|| {
            println!("ДатаКод v{} - Bytecode VM", VERSION);
            println!("Введите код (Ctrl+D или 'exit' для выхода):");
            println!();
            
            let mut input = String::new();
            loop {
                use std::io::{self, Write};
                
                // Показываем приглашение
                print!("datacode> ");
                io::stdout().flush().unwrap();
                
                match io::stdin().read_line(&mut input) {
                    Ok(0) => {
                        // EOF (Ctrl+D)
                        println!("\nДо свидания!");
                        break;
                    }
                    Ok(_) => {
                        let trimmed = input.trim();
                        
                        // Проверка на команду выхода
                        if trimmed == "exit" || trimmed == "quit" {
                            println!("До свидания!");
                            break;
                        }
                        
                        if trimmed.is_empty() {
                            input.clear();
                            continue;
                        }
                        
                        // Выполнение кода
                        // Note: run() is called inside run_with_event_loop, so PlotSystem is already initialized
                        match run(trimmed) {
                            Ok(value) => {
                                // Если есть результат, показываем его
                                if !matches!(value, data_code::Value::Null) {
                                    println!("=> {:?}", value);
                                }
                            }
                            Err(e) => {
                                eprintln!("Ошибка: {}", e);
                            }
                        }
                        input.clear();
                    }
                    Err(e) => {
                        eprintln!("Ошибка чтения: {}", e);
                        break;
                    }
                }
            }
            Ok(())
        }) {
            eprintln!("Ошибка: {}", e);
        }
    }
}
