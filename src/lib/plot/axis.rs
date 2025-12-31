// Axis structure for plot module

use crate::plot::Image;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub struct Axis {
    pub image: Option<Rc<RefCell<Image>>>,
    pub title: Option<String>,
    pub axis_visible: bool, // false for 'off', true for 'on'
    pub cmap: String, // 'gray', 'viridis', etc.
}

impl Axis {
    pub fn new() -> Self {
        Self {
            image: None,
            title: None,
            axis_visible: true,
            cmap: "gray".to_string(),
        }
    }
}

