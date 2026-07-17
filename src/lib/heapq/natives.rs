//! Native `heapq` API: min-heap on [`Value::Array`] using [`crate::common::value_ord::value_partial_cmp`].

use crate::common::value::Value;
use crate::common::value_ord::value_partial_cmp;
use crate::websocket::set_native_error;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

fn arg_array<'a>(args: &'a [Value], fn_name: &str, arg_idx: usize) -> Option<&'a Rc<RefCell<Vec<Value>>>> {
    match args.first() {
        Some(Value::Array(a)) => Some(a),
        _ => {
            set_native_error(format!(
                "TypeError: {}() argument {} must be an array",
                fn_name, arg_idx
            ));
            None
        }
    }
}

#[inline]
fn swap(heap: &mut Vec<Value>, i: usize, j: usize) {
    heap.swap(i, j);
}

fn sift_up(heap: &mut Vec<Value>, mut index: usize) -> Result<(), String> {
    while index > 0 {
        let parent = (index - 1) / 2;
        match value_partial_cmp(&heap[parent], &heap[index]) {
            Ok(ord) if ord != Ordering::Greater => break,
            Ok(_) => {
                swap(heap, parent, index);
                index = parent;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn sift_down(heap: &mut Vec<Value>, mut index: usize) -> Result<(), String> {
    let n = heap.len();
    loop {
        let left = 2 * index + 1;
        let right = 2 * index + 2;
        let mut smallest = index;

        if left < n {
            match value_partial_cmp(&heap[left], &heap[smallest]) {
                Ok(Ordering::Less) => smallest = left,
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        if right < n {
            match value_partial_cmp(&heap[right], &heap[smallest]) {
                Ok(Ordering::Less) => smallest = right,
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }

        if smallest == index {
            break;
        }
        swap(heap, index, smallest);
        index = smallest;
    }
    Ok(())
}

/// `heap_clear(heap)` — O(1) clear + shrink for flat A* open heaps (avoid heappop-clear alloc storm).
pub fn native_heapq_heap_clear(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("TypeError: heap_clear() requires heap array".to_string());
        return Value::Null;
    }
    let Some(arr_rc) = arg_array(args, "heap_clear", 1) else {
        return Value::Null;
    };
    arr_rc.borrow_mut().clear();
    arr_rc.borrow_mut().shrink_to_fit();
    Value::Null
}

/// `heappush(heap, item)`
pub fn native_heapq_heappush(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: heappush() requires heap and item".to_string());
        return Value::Null;
    }
    let Some(arr_rc) = arg_array(args, "heappush", 1) else {
        return Value::Null;
    };
    let item = args[1].clone();
    let mut heap = arr_rc.borrow_mut();
    heap.push(item);
    let idx = heap.len() - 1;
    if let Err(msg) = sift_up(&mut heap, idx) {
        heap.pop();
        set_native_error(msg);
        return Value::Null;
    }
    Value::Null
}

/// `heappop(heap)`
pub fn native_heapq_heappop(args: &[Value]) -> Value {
    let Some(arr_rc) = arg_array(args, "heappop", 1) else {
        return Value::Null;
    };
    let mut heap = arr_rc.borrow_mut();
    let n = heap.len();
    if n == 0 {
        set_native_error("IndexError: heappop from empty heap".to_string());
        return Value::Null;
    }
    if n == 1 {
        return heap.pop().unwrap_or(Value::Null);
    }
    let root = heap[0].clone();
    heap[0] = heap.pop().unwrap();
    if let Err(msg) = sift_down(&mut heap, 0) {
        set_native_error(msg);
        return Value::Null;
    }
    root
}

/// `heapify(heap)`
pub fn native_heapq_heapify(args: &[Value]) -> Value {
    let Some(arr_rc) = arg_array(args, "heapify", 1) else {
        return Value::Null;
    };
    let mut heap = arr_rc.borrow_mut();
    let n = heap.len();
    if n <= 1 {
        return Value::Null;
    }
    let start = n / 2 - 1;
    for i in (0..=start).rev() {
        if let Err(msg) = sift_down(&mut heap, i) {
            set_native_error(msg);
            return Value::Null;
        }
    }
    Value::Null
}

/// `heappeek(heap)`
pub fn native_heapq_heappeek(args: &[Value]) -> Value {
    let Some(arr_rc) = arg_array(args, "heappeek", 1) else {
        return Value::Null;
    };
    let heap = arr_rc.borrow();
    if heap.is_empty() {
        set_native_error("IndexError: peek from empty heap".to_string());
        return Value::Null;
    }
    heap[0].clone()
}

/// `heapreplace(heap, item)`
pub fn native_heapq_heapreplace(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: heapreplace() requires heap and item".to_string());
        return Value::Null;
    }
    let Some(arr_rc) = arg_array(args, "heapreplace", 1) else {
        return Value::Null;
    };
    let item = args[1].clone();
    let mut heap = arr_rc.borrow_mut();
    if heap.is_empty() {
        set_native_error("IndexError: heapreplace on empty heap".to_string());
        return Value::Null;
    }
    let old = std::mem::replace(&mut heap[0], item);
    if let Err(msg) = sift_down(&mut heap, 0) {
        set_native_error(msg);
        return Value::Null;
    }
    old
}
