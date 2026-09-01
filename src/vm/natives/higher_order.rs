//! map / filter / reduce: lazy iterators + terminal reduce (single pass, no intermediate arrays).

use std::cell::RefCell;
use std::rc::Rc;

use crate::common::error::LangError;
use crate::common::value::{IterableInner, Value};
use crate::vm::host::HostFunction;
use crate::vm::iterable::{iterable_from_value, iterable_next, value_to_callable_slot};
use crate::vm::natives::utils::call_user_function;
use crate::vm::vm::current_vm_ptr;

fn runtime(line: usize, msg: impl Into<String>) -> LangError {
    LangError::runtime_error(msg.into(), line)
}

/// `map(collection | iterable, fn | native)` → lazy [`Value::Iterable`].
pub struct MapHostFunction;

impl HostFunction for MapHostFunction {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        if args.len() < 2 {
            return Err(runtime(0, "map() expects (collection, function)"));
        }
        let vm_ptr =
            current_vm_ptr().ok_or_else(|| runtime(0, "map: VM context not available"))?;
        let f = &args[1];
        let coll = &args[0];
        unsafe {
            let vm = &mut *vm_ptr;
            let (slot, arity) = value_to_callable_slot(f, vm)?;
            if let Value::ColumnsReference {
                table,
                column_names,
            } = coll
            {
                let n = column_names.len();
                if arity as usize != n {
                    return Err(runtime(
                        0,
                        format!(
                            "map callback must have {} parameters (one per column), got arity {}",
                            n, arity
                        ),
                    ));
                }
                {
                    let t = table.borrow();
                    for name in column_names {
                        if !t.has_column(name) {
                            return Err(runtime(
                                0,
                                format!("KeyError: column '{}' not found in table", name),
                            ));
                        }
                    }
                }
                let wrapped = IterableInner::ColumnsMap {
                    table: Rc::clone(table),
                    column_names: column_names.clone(),
                    func: slot,
                    index: 0,
                };
                return Ok(Value::Iterable(Rc::new(RefCell::new(wrapped))));
            }
            if arity != 1 && arity != 2 {
                return Err(runtime(
                    0,
                    format!(
                        "map callback must have 1 or 2 parameters, got arity {}",
                        arity
                    ),
                ));
            }
            let base = iterable_from_value(coll)?;
            let wrapped = IterableInner::Map {
                source: base,
                func: slot,
                fn_arity: arity,
                index: 0,
            };
            Ok(Value::Iterable(Rc::new(RefCell::new(wrapped))))
        }
    }
}

/// `filter(collection | iterable, predicate)` → lazy [`Value::Iterable`].
pub struct FilterHostFunction;

impl HostFunction for FilterHostFunction {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        if args.len() < 2 {
            return Err(runtime(0, "filter() expects (collection, predicate)"));
        }
        let vm_ptr =
            current_vm_ptr().ok_or_else(|| runtime(0, "filter: VM context not available"))?;
        let coll = &args[0];
        let pred = &args[1];
        unsafe {
            let vm = &mut *vm_ptr;
            let (slot, arity) = value_to_callable_slot(pred, vm)?;
            if arity != 1 && arity != 2 {
                return Err(runtime(
                    0,
                    format!(
                        "filter predicate must have 1 or 2 parameters, got arity {}",
                        arity
                    ),
                ));
            }
            let base = iterable_from_value(coll)?;
            let wrapped = IterableInner::Filter {
                source: base,
                pred: slot,
                fn_arity: arity,
                index: 0,
            };
            Ok(Value::Iterable(Rc::new(RefCell::new(wrapped))))
        }
    }
}

/// `reduce(collection | iterable, fn(acc, x), initial)` — terminal fold (one pass).
pub struct ReduceHostFunction;

impl HostFunction for ReduceHostFunction {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        if args.len() < 3 {
            return Err(runtime(
                0,
                "reduce() expects (collection, function, initial) — initial is required",
            ));
        }
        let coll = &args[0];
        let f = &args[1];
        let initial = args[2].clone();

        let Value::Function(fn_idx) = f else {
            return Err(runtime(
                0,
                "reduce: second argument must be a user function",
            ));
        };

        let vm_ptr =
            current_vm_ptr().ok_or_else(|| runtime(0, "reduce: VM context not available"))?;
        unsafe {
            let vm = &mut *vm_ptr;
            let arity = vm
                .get_functions()
                .get(*fn_idx)
                .map(|fun| fun.arity)
                .unwrap_or(0);
            if arity != 2 {
                return Err(runtime(
                    0,
                    format!(
                        "reduce callback must have arity 2 (acc, item), got {}",
                        arity
                    ),
                ));
            }
            let rc = iterable_from_value(coll)?;
            let mut inner = rc.borrow_mut();
            let mut acc = initial;
            while let Some(el) = iterable_next(&mut *inner, vm)? {
                acc = call_user_function(*fn_idx, &[acc, el])?;
            }
            Ok(acc)
        }
    }
}
