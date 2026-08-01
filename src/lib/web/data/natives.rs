//! Natives for `web.data`.

use crate::common::value::Value;
use crate::datasource::get_table::table_from_value;
use crate::web::data::extract::{
    extract_many, extract_one, parse_field_schema, table_from_html,
};
use crate::web::error::{raise, WebError};
use crate::web::http::natives::ensure_json;
use std::cell::RefCell;
use std::rc::Rc;

fn html_from_source(source: &Value) -> Result<String, WebError> {
    match source {
        Value::String(s) => Ok(s.clone()),
        Value::WebPage(page) => {
            let page = page.borrow();
            if page.closed {
                return Err(WebError::runtime("browser page is closed"));
            }
            let mut driver = page
                .driver
                .lock()
                .map_err(|_| WebError::runtime("browser driver lock poisoned"))?;
            driver.html(None)
        }
        Value::WebElement(el) => {
            let el = el.borrow();
            let mut driver = el
                .driver
                .lock()
                .map_err(|_| WebError::runtime("browser driver lock poisoned"))?;
            driver.html(Some(&el.selector))
        }
        Value::HttpResponse(rc) => {
            let resp = rc.borrow();
            resp.body_string().map_err(WebError::value)
        }
        _ => Err(WebError::type_err(
            "data.* expects HTML string, WebPage, WebElement, HttpResponse, or tabular data",
        )),
    }
}

/// `data.table(source, selector?)`
pub fn native_data_table(args: &[Value]) -> Value {
    if args.is_empty() {
        raise(WebError::type_err("data.table() requires a source"));
        return Value::Null;
    }
    let source = &args[0];
    let selector = match args.get(1) {
        Some(Value::String(s)) => Some(s.as_str()),
        Some(Value::Null) | None => None,
        Some(_) => {
            raise(WebError::type_err("data.table selector must be a string"));
            return Value::Null;
        }
    };

    // Direct conversion paths (JSON / arrays / tables)
    if selector.is_none() {
        match source {
            Value::Table(rc) => return Value::Table(Rc::clone(rc)),
            Value::Array(_) | Value::Object(_) => {
                return match table_from_value(source) {
                    Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
                    Err(e) => {
                        raise(WebError::value(e.display()));
                        Value::Null
                    }
                };
            }
            Value::HttpResponse(rc) => {
                let mut resp = rc.borrow_mut();
                match ensure_json(&mut resp).and_then(|j| {
                    table_from_value(&j).map_err(|e| WebError::value(e.display()))
                }) {
                    Ok(t) => return Value::Table(Rc::new(RefCell::new(t))),
                    Err(e) => {
                        // Fall through to HTML parse of body
                        let _ = e;
                    }
                }
            }
            _ => {}
        }
    }

    let html = match html_from_source(source) {
        Ok(h) => h,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    match table_from_html(&html, selector) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

/// `data.extract(source, schema)` or `data.extract(source, item_selector, schema)`
pub fn native_data_extract(args: &[Value]) -> Value {
    if args.len() < 2 {
        raise(WebError::type_err(
            "data.extract(source, schema) or data.extract(source, item_selector, schema)",
        ));
        return Value::Null;
    }
    let html = match html_from_source(&args[0]) {
        Ok(h) => h,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };

    if args.len() >= 3 {
        let item_sel = match &args[1] {
            Value::String(s) => s.clone(),
            _ => {
                raise(WebError::type_err("item_selector must be a string"));
                return Value::Null;
            }
        };
        let schema = match parse_field_schema(&args[2]) {
            Ok(s) => s,
            Err(e) => {
                raise(e);
                return Value::Null;
            }
        };
        match extract_many(&html, &item_sel, &schema) {
            Ok(v) => v,
            Err(e) => {
                raise(e);
                Value::Null
            }
        }
    } else {
        let schema = match parse_field_schema(&args[1]) {
            Ok(s) => s,
            Err(e) => {
                raise(e);
                return Value::Null;
            }
        };
        // Single-record extract → wrap as one-element array for table conversion convenience
        match extract_one(&html, &schema) {
            Ok(v) => Value::Array(Rc::new(RefCell::new(vec![v]))),
            Err(e) => {
                raise(e);
                Value::Null
            }
        }
    }
}
