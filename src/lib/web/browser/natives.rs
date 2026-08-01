//! Natives for `web.browser` and page/element methods.

use crate::common::value::Value;
use crate::web::args::{
    optional_f64, optional_string, parse_browser_options, require_string,
};
use crate::web::browser::driver;
use crate::web::browser::registry;
use crate::web::error::{raise, WebError};
use crate::web::security::check_net_browser;
use crate::web::values::{next_page_id, WebElement, WebPage};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

fn page_from_args(args: &[Value]) -> Result<Rc<RefCell<WebPage>>, WebError> {
    match args.first() {
        Some(Value::WebPage(rc)) => Ok(Rc::clone(rc)),
        _ => Err(WebError::type_err("expected browser page as receiver")),
    }
}

fn element_from_args(args: &[Value]) -> Result<Rc<RefCell<WebElement>>, WebError> {
    match args.first() {
        Some(Value::WebElement(rc)) => Ok(Rc::clone(rc)),
        _ => Err(WebError::type_err("expected web element as receiver")),
    }
}

fn with_page_driver<F, T>(page: &Rc<RefCell<WebPage>>, f: F) -> Result<T, WebError>
where
    F: FnOnce(&mut dyn driver::BrowserDriver) -> Result<T, WebError>,
{
    let page = page.borrow();
    if page.closed {
        return Err(WebError::runtime("browser page is closed"));
    }
    let mut guard = page
        .driver
        .lock()
        .map_err(|_| WebError::runtime("browser driver lock poisoned"))?;
    f(guard.as_mut())
}

/// `browser.open(url, …)`
pub fn native_browser_open(args: &[Value]) -> Value {
    if let Err(e) = check_net_browser() {
        raise(e);
        return Value::Null;
    }
    let url = match require_string(args, 0, "url") {
        Ok(u) => u,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let opts = match parse_browser_options(if args.len() > 1 { &args[1..] } else { &[] }) {
        Ok(o) => o,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    match driver::launch(&url, &opts) {
        Ok(d) => {
            let id = next_page_id();
            let page = Rc::new(RefCell::new(WebPage {
                id,
                closed: false,
                driver: Arc::new(Mutex::new(d)),
            }));
            registry::register(Rc::clone(&page));
            Value::WebPage(page)
        }
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

pub fn native_page_goto(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let url = match require_string(args, 1, "url") {
        Ok(u) => u,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.goto(&url)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_close(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let mut p = page.borrow_mut();
    if !p.closed {
        let _ = p.driver.lock().map(|mut d| d.close());
        p.closed = true;
        registry::unregister(p.id);
    }
    Value::Null
}

pub fn native_page_click(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.click(&sel)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_type(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let text = match require_string(args, 2, "text") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.type_text(&sel, &text)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_fill(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let text = match require_string(args, 2, "text") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.fill(&sel, &text)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_clear(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.clear(&sel)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_select(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let value = match require_string(args, 2, "value") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.select(&sel, &value)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_text(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = optional_string(args, 1);
    match with_page_driver(&page, |d| d.text(sel.as_deref())) {
        Ok(s) => Value::String(s),
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

pub fn native_page_html(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = optional_string(args, 1);
    match with_page_driver(&page, |d| d.html(sel.as_deref())) {
        Ok(s) => Value::String(s),
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

pub fn native_page_screenshot(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let path = match require_string(args, 1, "path") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = optional_string(args, 2);
    if let Err(e) = with_page_driver(&page, |d| d.screenshot(&path, sel.as_deref())) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_wait(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let secs = optional_f64(args, 1).unwrap_or(0.0);
    if let Err(e) = with_page_driver(&page, |d| d.wait(secs)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_wait_for(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let timeout = optional_f64(args, 2);
    if let Err(e) = with_page_driver(&page, |d| d.wait_for(&sel, timeout)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_wait_for_navigation(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let timeout = optional_f64(args, 1);
    if let Err(e) = with_page_driver(&page, |d| d.wait_for_navigation(timeout)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_find(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    match with_page_driver(&page, |d| d.find(&sel)) {
        Ok(handle) => {
            let driver = page.borrow().driver.clone();
            Value::WebElement(Rc::new(RefCell::new(WebElement {
                page_id: page.borrow().id,
                selector: handle.selector,
                index: handle.index,
                driver,
            })))
        }
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

pub fn native_page_find_all(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = match require_string(args, 1, "selector") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    match with_page_driver(&page, |d| d.find_all(&sel)) {
        Ok(handles) => {
            let driver = page.borrow().driver.clone();
            let page_id = page.borrow().id;
            let els: Vec<Value> = handles
                .into_iter()
                .map(|h| {
                    Value::WebElement(Rc::new(RefCell::new(WebElement {
                        page_id,
                        selector: h.selector,
                        index: h.index,
                        driver: Arc::clone(&driver),
                    })))
                })
                .collect();
            Value::Array(Rc::new(RefCell::new(els)))
        }
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

pub fn native_page_set_cookie(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let name = match require_string(args, 1, "name") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let value = match require_string(args, 2, "value") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.set_cookie(&name, &value)) {
        raise(e);
    }
    Value::Null
}

pub fn native_page_delete_cookie(args: &[Value]) -> Value {
    let page = match page_from_args(args) {
        Ok(p) => p,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let name = match require_string(args, 1, "name") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    if let Err(e) = with_page_driver(&page, |d| d.delete_cookie(&name)) {
        raise(e);
    }
    Value::Null
}

// --- WebElement methods ---

pub fn native_element_click(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    match driver.lock() {
        Ok(mut d) => {
            if let Err(e) = d.click(&sel) {
                raise(e);
            }
        }
        Err(_) => raise(WebError::runtime("browser driver lock poisoned")),
    }
    Value::Null
}

pub fn native_element_text(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    let out = match driver.lock() {
        Ok(mut d) => match d.text(Some(&sel)) {
            Ok(s) => Value::String(s),
            Err(e) => {
                raise(e);
                Value::Null
            }
        },
        Err(_) => {
            raise(WebError::runtime("browser driver lock poisoned"));
            Value::Null
        }
    };
    out
}

pub fn native_element_html(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    let out = match driver.lock() {
        Ok(mut d) => match d.html(Some(&sel)) {
            Ok(s) => Value::String(s),
            Err(e) => {
                raise(e);
                Value::Null
            }
        },
        Err(_) => {
            raise(WebError::runtime("browser driver lock poisoned"));
            Value::Null
        }
    };
    out
}

pub fn native_element_type(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let text = match require_string(args, 1, "text") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    match driver.lock() {
        Ok(mut d) => {
            if let Err(e) = d.type_text(&sel, &text) {
                raise(e);
            }
        }
        Err(_) => raise(WebError::runtime("browser driver lock poisoned")),
    }
    Value::Null
}

pub fn native_element_fill(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let text = match require_string(args, 1, "text") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    match driver.lock() {
        Ok(mut d) => {
            if let Err(e) = d.fill(&sel, &text) {
                raise(e);
            }
        }
        Err(_) => raise(WebError::runtime("browser driver lock poisoned")),
    }
    Value::Null
}

pub fn native_element_clear(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    match driver.lock() {
        Ok(mut d) => {
            if let Err(e) = d.clear(&sel) {
                raise(e);
            }
        }
        Err(_) => raise(WebError::runtime("browser driver lock poisoned")),
    }
    Value::Null
}

pub fn native_element_attr(args: &[Value]) -> Value {
    let el = match element_from_args(args) {
        Ok(e) => e,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let name = match require_string(args, 1, "name") {
        Ok(s) => s,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let sel = el.borrow().selector.clone();
    let driver = el.borrow().driver.clone();
    let out = match driver.lock() {
        Ok(mut d) => match d.attr(&sel, &name) {
            Ok(s) => Value::String(s),
            Err(e) => {
                raise(e);
                Value::Null
            }
        },
        Err(_) => {
            raise(WebError::runtime("browser driver lock poisoned"));
            Value::Null
        }
    };
    out
}
