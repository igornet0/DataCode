//! Helpers to extract numeric arrays from [`Value`] and build results.

use crate::common::value::Value;
use std::cell::RefCell;
use std::rc::Rc;

pub fn extract_f64_array(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Array(a) => {
            let mut out = Vec::with_capacity(a.borrow().len());
            for item in a.borrow().iter() {
                out.push(item.as_ieee_f64()?);
            }
            Some(out)
        }
        _ => None,
    }
}

pub fn f64_array_to_value(data: Vec<f64>) -> Value {
    let items: Vec<Value> = data.into_iter().map(Value::Number).collect();
    Value::Array(Rc::new(RefCell::new(items)))
}

pub fn all_finite_f64(v: &Value) -> bool {
    extract_f64_array(v).is_some()
}
