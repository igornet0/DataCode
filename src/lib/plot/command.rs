// GUI commands for communication between runtime and main thread

use winit::window::{WindowId, Icon};
use std::sync::mpsc;
use crate::plot::Image;
use std::sync::Arc;
use std::sync::Mutex;

/// Commands sent from runtime thread to main thread via EventLoopProxy
/// All window operations go through commands - runtime NEVER owns Window
#[derive(Debug)]
pub enum GuiCommand {
    /// Request to create a new window
    CreateWindow {
        width: u32,
        height: u32,
        title: String,
        icon: Option<Icon>,
        response: mpsc::Sender<Result<WindowId, String>>,
    },
    
    /// Draw image to window
    DrawImage {
        window_id: WindowId,
        image: Arc<Mutex<Image>>, // Arc<Mutex<>> is Send + Sync
    },
    
    /// Update chart data for window
    UpdateChart {
        window_id: WindowId,
        chart_data: ChartData,
    },
    
    /// Update figure for window
    /// Figure data extracted from Figure (which contains Rc<RefCell<Axis>> that is not Send)
    UpdateFigure {
        window_id: WindowId,
        figure_data: FigureData,
    },
    
    /// Update image grid for window
    UpdateImageGrid {
        window_id: WindowId,
        images: Vec<Arc<Mutex<Image>>>, // Arc<Mutex<>> is Send + Sync
        rows: usize,
        cols: usize,
        titles: Vec<String>,
    },
    
    /// Request redraw of window
    Redraw {
        window_id: WindowId,
    },
}

/// Chart data that can be sent to GUI thread
#[derive(Debug, Clone)]
pub struct ChartData {
    pub chart_type: ChartType,
    pub lines: Vec<(Vec<f64>, Vec<f64>, bool, usize, usize, u32)>,
    pub bars: Vec<(Vec<String>, Vec<f64>, u32)>,
    pub pies: Vec<(Vec<String>, Vec<f64>, u32)>,
    pub heatmaps: Vec<(Vec<Vec<f64>>, Option<f64>, Option<f64>, String)>,
    pub xlabel: Option<String>,
    pub ylabel: Option<String>,
    pub pie_rotation: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Heatmap,
}

/// Axis data that can be sent to GUI thread (extracted from Axis which contains Rc<RefCell<Image>>)
#[derive(Debug, Clone)]
pub struct AxisData {
    pub image: Option<Arc<Mutex<Image>>>,
    pub title: Option<String>,
    pub axis_visible: bool,
    pub cmap: String,
}

/// Figure data that can be sent to GUI thread (extracted from Figure which contains Rc<RefCell<Axis>>)
#[derive(Debug, Clone)]
pub struct FigureData {
    pub axes: Vec<Vec<AxisData>>, // 2D array of axes data
    pub tight_layout: bool,
}

