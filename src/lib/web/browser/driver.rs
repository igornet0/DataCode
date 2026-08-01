//! Browser driver abstraction.

use crate::common::value::Value;
use crate::web::args::BrowserOptions;
use crate::web::error::WebError;

#[derive(Debug, Clone)]
pub struct ElementHandle {
    pub selector: String,
    pub index: usize,
}

pub trait BrowserDriver: Send {
    fn goto(&mut self, url: &str) -> Result<(), WebError>;
    fn click(&mut self, selector: &str) -> Result<(), WebError>;
    fn type_text(&mut self, selector: &str, text: &str) -> Result<(), WebError>;
    fn fill(&mut self, selector: &str, text: &str) -> Result<(), WebError>;
    fn clear(&mut self, selector: &str) -> Result<(), WebError>;
    fn select(&mut self, selector: &str, value: &str) -> Result<(), WebError>;
    fn text(&mut self, selector: Option<&str>) -> Result<String, WebError>;
    fn html(&mut self, selector: Option<&str>) -> Result<String, WebError>;
    fn screenshot(&mut self, path: &str, selector: Option<&str>) -> Result<(), WebError>;
    fn wait(&mut self, secs: f64) -> Result<(), WebError>;
    fn wait_for(&mut self, selector: &str, timeout: Option<f64>) -> Result<(), WebError>;
    fn wait_for_navigation(&mut self, timeout: Option<f64>) -> Result<(), WebError>;
    fn find(&mut self, selector: &str) -> Result<ElementHandle, WebError>;
    fn find_all(&mut self, selector: &str) -> Result<Vec<ElementHandle>, WebError>;
    fn cookies(&mut self) -> Result<Value, WebError>;
    fn set_cookie(&mut self, name: &str, value: &str) -> Result<(), WebError>;
    fn delete_cookie(&mut self, name: &str) -> Result<(), WebError>;
    fn attr(&mut self, selector: &str, name: &str) -> Result<String, WebError>;
    fn attributes(&mut self, selector: &str) -> Result<Value, WebError>;
    fn close(&mut self) -> Result<(), WebError>;

    /// Run a JS snippet in the page context (optional backend capability).
    fn evaluate(&mut self, _script: &str) -> Result<Value, WebError> {
        Err(WebError::runtime(
            "evaluate() is not supported by this browser driver",
        ))
    }

    /// Persist a script for every new document (CDP Page.addScriptToEvaluateOnNewDocument).
    fn evaluate_on_new_document(&mut self, _script: &str) -> Result<(), WebError> {
        // Default: best-effort no-op so custom drivers remain implementable.
        Ok(())
    }
}

/// Launch Standard or Stealth backend based on [`BrowserOptions::stealth`].
pub fn launch(url: &str, opts: &BrowserOptions) -> Result<Box<dyn BrowserDriver>, WebError> {
    if opts.stealth {
        crate::web::browser::stealth::StealthBrowserDriver::launch(url, opts)
            .map(|d| Box::new(d) as Box<dyn BrowserDriver>)
    } else {
        crate::web::browser::chromium::ChromiumDriver::launch(url, opts)
            .map(|d| Box::new(d) as Box<dyn BrowserDriver>)
    }
}
