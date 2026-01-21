// Window event handlers

use crate::plot::{WindowState, RenderContent};
use crate::plot::command::{ChartType, ChartData};
use crate::plot::renderer::Renderer;
use std::sync::Mutex;

/// Calculate chart bounds for line charts
/// Returns (x_min_plot, x_max_plot, y_min_plot, y_max_plot)
fn calculate_chart_bounds(chart_data: &ChartData) -> (f64, f64, f64, f64) {
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
    
    (x_min_plot, x_max_plot, y_min_plot, y_max_plot)
}

/// Handle window resize event
pub fn handle_window_resize(
    windows: &Mutex<std::collections::HashMap<winit::window::WindowId, WindowState>>,
    window_id: winit::window::WindowId,
) {
    if let Ok(mut windows) = windows.lock() {
        if let Some(state) = windows.get_mut(&window_id) {
            // Update window size with the new size from event
            state.window.update_size();
            
            // Update scale factor and ensure buffer is resized
            if let Some(ref mut renderer) = state.renderer {
                renderer.update_scale_factor(&state.window);
                
                // Force buffer resize - this ensures softbuffer updates buffer size
                // before the next redraw
                let _ = renderer.ensure_buffer_resized();
            }
            
            // Request redraw - this will trigger RedrawRequested event
            // During redraw, buffer will already have the correct size
            state.window.request_redraw();
        }
    }
}

/// Handle window close event
/// Returns true if all windows are closed
pub fn handle_window_close(
    windows: &Mutex<std::collections::HashMap<winit::window::WindowId, WindowState>>,
    window_id: winit::window::WindowId,
) -> bool {
    let should_exit = {
        let mut windows = windows.lock().unwrap();
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
    crate::plot::system::PlotSystem::notify_window_closed(window_id);
    
    should_exit
}

/// Handle cursor moved event
pub fn handle_cursor_moved(
    windows: &Mutex<std::collections::HashMap<winit::window::WindowId, WindowState>>,
    window_id: winit::window::WindowId,
    position: winit::dpi::PhysicalPosition<f64>,
) {
    if let Ok(mut windows) = windows.lock() {
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
                    let (x_min_plot, x_max_plot, y_min_plot, y_max_plot) = calculate_chart_bounds(chart_data);
                    
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

/// Handle mouse click event
pub fn handle_mouse_click(
    windows: &Mutex<std::collections::HashMap<winit::window::WindowId, WindowState>>,
    window_id: winit::window::WindowId,
) {
    if let Ok(mut windows) = windows.lock() {
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
                        let (x_min_plot, x_max_plot, y_min_plot, y_max_plot) = calculate_chart_bounds(chart_data);
                        
                        // Find nearest point and select it
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

/// Handle redraw event
pub fn handle_redraw(
    windows: &Mutex<std::collections::HashMap<winit::window::WindowId, WindowState>>,
    window_id: winit::window::WindowId,
) {
    if let Ok(mut windows) = windows.lock() {
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
            
            // Ensure buffer is resized to match current window size before rendering
            // This is critical - softbuffer needs buffer_mut() to be called to update size
            if let Some(ref mut renderer) = state.renderer {
                // Update scale factor in case it changed
                renderer.update_scale_factor(&state.window);
                
                // Force buffer resize - ensures buffer size matches window size
                // This will recreate the surface if buffer size doesn't match window size
                let _ = renderer.ensure_buffer_resized();
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

