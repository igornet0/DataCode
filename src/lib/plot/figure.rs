// Figure structure for plot module

use crate::plot::Axis;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub struct Figure {
    pub axes: Vec<Vec<Rc<RefCell<Axis>>>>, // 2D array of axes
    pub figsize: (f64, f64), // (width, height) in figure units
    pub tight_layout: bool,
    // Note: window is no longer stored here - it's in WindowState in GUI thread
}

impl Figure {
    pub fn new(rows: usize, cols: usize, figsize: (f64, f64)) -> Self {
        let mut axes = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::new();
            for _ in 0..cols {
                row.push(Rc::new(RefCell::new(Axis::new())));
            }
            axes.push(row);
        }
        
        Self {
            axes,
            figsize,
            tight_layout: false,
        }
    }
}

