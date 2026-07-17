//! Native `pathfind` API: grid A* in Rust.

use crate::common::astar_grid_core::{bitmap_bytes_for_cells, bitmap_set};
use crate::common::set_map::SetMap;
use crate::common::value::Value;
use crate::pathfind::astar_grid_bitmap;
use crate::vm::store_convert::load_value;
use crate::vm::vm::current_vm_ptr;
use crate::websocket::set_native_error;
use std::cell::RefCell;
use std::rc::Rc;

fn arg_u32(v: &Value, name: &str) -> Option<u32> {
    let Some(n) = v.as_finite_f64() else {
        set_native_error(format!("TypeError: {} must be a non-negative integer", name));
        return None;
    };
    if n >= 0.0 && n.fract() == 0.0 && n <= u32::MAX as f64 {
        Some(n as u32)
    } else {
        set_native_error(format!("TypeError: {} must be a non-negative integer", name));
        None
    }
}

fn arg_coord(v: &Value, name: &str) -> Option<(u32, u32)> {
    match v {
        Value::Tuple(t) => {
            let tup = t.borrow();
            if tup.len() != 2 {
                set_native_error(format!("TypeError: {} must be (row, col)", name));
                return None;
            }
            let r = arg_u32(&tup[0], &format!("{name}[0]"))?;
            let c = arg_u32(&tup[1], &format!("{name}[1]"))?;
            Some((r, c))
        }
        Value::Array(a) => {
            let arr = a.borrow();
            if arr.len() != 2 {
                set_native_error(format!("TypeError: {} must be [row, col]", name));
                return None;
            }
            let r = arg_u32(&arr[0], &format!("{name}[0]"))?;
            let c = arg_u32(&arr[1], &format!("{name}[1]"))?;
            Some((r, c))
        }
        _ => {
            set_native_error(format!("TypeError: {} must be (row, col)", name));
            None
        }
    }
}

fn linear_id_from_value(v: &Value, cols: u32) -> Option<u32> {
    match v {
        Value::Tuple(_) | Value::Array(_) => {
            arg_coord(v, "blocked cell").map(|(r, c)| r * cols + c)
        }
        _ => arg_u32(v, "blocked id"),
    }
}

fn parse_blocked_bitmap(
    set: &SetMap,
    cols: u32,
    n_cells: usize,
) -> Option<Vec<u8>> {
    let vm_ptr = current_vm_ptr()?;
    let mut bits = vec![0u8; bitmap_bytes_for_cells(n_cells)];
    let mut ok = true;
    unsafe {
        (*vm_ptr).with_stores(|store, heap| {
            for key_id in set.iter_key_ids() {
                let v = load_value(key_id, store, heap);
                if let Some(id) = linear_id_from_value(&v, cols) {
                    bitmap_set(&mut bits, id);
                } else {
                    set_native_error(
                        "TypeError: blocked set elements must be (row,col) or linear id".to_string(),
                    );
                    ok = false;
                    break;
                }
            }
        });
    }
    ok.then_some(bits)
}

fn path_to_value(path: Vec<(u32, u32)>) -> Value {
    let tuples: Vec<Value> = path
        .into_iter()
        .map(|(r, c)| {
            Value::Tuple(Rc::new(RefCell::new(vec![
                Value::Number(r as f64),
                Value::Number(c as f64),
            ])))
        })
        .collect();
    Value::Array(Rc::new(RefCell::new(tuples)))
}

/// `astar_grid(rows, cols, start, goal, blocked)` → path array or null.
pub fn native_pathfind_astar_grid(args: &[Value]) -> Value {
    if args.len() < 5 {
        set_native_error("TypeError: astar_grid(rows, cols, start, goal, blocked)".to_string());
        return Value::Null;
    }
    let rows = match arg_u32(&args[0], "rows") {
        Some(v) => v,
        None => return Value::Null,
    };
    let cols = match arg_u32(&args[1], "cols") {
        Some(v) => v,
        None => return Value::Null,
    };
    let start = match arg_coord(&args[2], "start") {
        Some(v) => v,
        None => return Value::Null,
    };
    let goal = match arg_coord(&args[3], "goal") {
        Some(v) => v,
        None => return Value::Null,
    };
    let n_cells = rows as usize * cols as usize;
    let blocked_bits = match &args[4] {
        Value::Set(s) => match parse_blocked_bitmap(&s.borrow(), cols, n_cells) {
            Some(b) => b,
            None => return Value::Null,
        },
        Value::Null => vec![0u8; bitmap_bytes_for_cells(n_cells)],
        _ => {
            set_native_error("TypeError: blocked must be a set".to_string());
            return Value::Null;
        }
    };

    match astar_grid_bitmap(rows, cols, start, goal, &blocked_bits) {
        Some(path) => path_to_value(path),
        None => Value::Null,
    }
}
