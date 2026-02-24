// Plot execution context (replaces many global RefCells in natives).
// VM owns PlotContext and sets it thread-local at run() so plot natives use it without globals.

use winit::window::WindowId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use crate::plot::{Image, Window};
use crate::plot::window::ImageViewState;
use crate::plot::renderer::Renderer;
use crate::plot::command::ChartData;

#[derive(Default)]
pub struct WindowData {
    pub renderer: Option<Renderer>,
}

#[derive(Debug, Clone, Default)]
pub struct PlotStateData {
    pub xlabel: Option<String>,
    pub ylabel: Option<String>,
    pub line_data: Vec<(Vec<f64>, Vec<f64>, bool, usize, usize, u32)>,
    pub bar_data: Vec<(Vec<String>, Vec<f64>, u32)>,
    pub pie_data: Vec<(Vec<String>, Vec<f64>, u32)>,
    pub heatmap_data: Vec<(Vec<Vec<f64>>, Option<f64>, Option<f64>, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct PointStateData {
    pub hovered: Option<(usize, usize)>,
    pub selected: Option<(usize, usize)>,
}

/// All plot state that was previously in global thread_locals; owned by VM during run().
#[derive(Default)]
pub struct PlotContext {
    pub window_data: HashMap<WindowId, WindowData>,
    pub renderers: HashMap<WindowId, Renderer>,
    pub image_view_states: HashMap<WindowId, ImageViewState>,
    pub window_images: HashMap<WindowId, Rc<RefCell<Image>>>,
    pub cursor_positions: HashMap<WindowId, (f32, f32)>,
    pub plot_state: PlotStateData,
    pub window_chart_data: HashMap<WindowId, ChartData>,
    pub window_point_states: HashMap<WindowId, PointStateData>,
    pub active_windows: HashMap<WindowId, Rc<RefCell<Window>>>,
    pub closed_windows: HashSet<WindowId>,
}

thread_local! {
    static PLOT_CONTEXT: RefCell<Option<PlotContext>> = RefCell::new(None);
}

impl PlotContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_current(ctx: PlotContext) {
        PLOT_CONTEXT.with(|cell| *cell.borrow_mut() = Some(ctx));
    }

    pub fn take_current() -> Option<PlotContext> {
        PLOT_CONTEXT.with(|cell| cell.borrow_mut().take())
    }

    pub fn with_current<F, R>(f: F) -> R
    where
        F: FnOnce(&mut PlotContext) -> R,
    {
        PLOT_CONTEXT.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let ctx = borrow.as_mut().expect("PlotContext not set. VM must set context before run().");
            f(ctx)
        })
    }

    pub fn is_set() -> bool {
        PLOT_CONTEXT.with(|cell| cell.borrow().is_some())
    }
}
