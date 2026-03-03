//! GetArrayElement for Value::Table (column by name, row by index).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::common::table::Table;
use crate::common::{error::LangError, value::Value, value_store::ValueStore};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::store_value;
use crate::vm::table_ops;
use crate::vm::types::VMStatus;

/// Get element from Table. Column by string, row by number.
#[allow(clippy::too_many_arguments)]
pub fn get_table(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    table: Rc<RefCell<Table>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(property) => {
            if property == "rows" {
                let t = table.borrow();
                let rows: Vec<Value> = if t.is_view() {
                    (0..t.len())
                        .map(|i| {
                            let row = table_ops::get_row(&*t, i, value_store, heavy_store)
                                .unwrap_or_default();
                            Value::Array(Rc::new(RefCell::new(row)))
                        })
                        .collect()
                } else {
                    t.rows_ref()
                        .unwrap()
                        .iter()
                        .map(|row| Value::Array(Rc::new(RefCell::new(row.to_vec()))))
                        .collect()
                };
                drop(t);
                stack::push_id(
                    stack,
                    store_value(
                        Value::Array(Rc::new(RefCell::new(rows))),
                        value_store,
                        heavy_store,
                    ),
                );
            } else if property == "columns" {
                let table_ref = table.borrow();
                let columns: Vec<Value> = table_ref
                    .headers()
                    .iter()
                    .map(|header| Value::String(header.clone()))
                    .collect();
                stack::push_id(
                    stack,
                    store_value(
                        Value::Array(Rc::new(RefCell::new(columns))),
                        value_store,
                        heavy_store,
                    ),
                );
            } else {
                let mut table_ref = table.borrow_mut();
                let has_col = if table_ref.is_view() {
                    table_ref.has_column(&property)
                } else {
                    table_ref.get_column(&property).is_some()
                };
                if has_col {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::ColumnReference {
                                table: table.clone(),
                                column_name: property,
                            },
                            value_store,
                            heavy_store,
                        ),
                    );
                } else {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!("Column '{}' not found in table", property),
                        line,
                        crate::common::error::ErrorType::KeyError,
                    );
                    match ExceptionHandler::handle_exception(
                        stack,
                        frames,
                        exception_handlers,
                        error,
                        value_store,
                        heavy_store,
                    ) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Table row index must be non-negative".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            let table_ref = table.borrow();
            let len = table_ref.len();
            if idx as usize >= len {
                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                    format!("Row index {} out of bounds (length: {})", idx, len),
                    line,
                    crate::common::error::ErrorType::IndexError,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            let row = if table_ref.is_view() {
                table_ops::get_row(&*table_ref, idx as usize, value_store, heavy_store)
            } else {
                table_ref.get_row(idx as usize).map(|r| r.to_vec())
            };
            if let Some(row) = row {
                let mut row_dict = HashMap::new();
                for (i, header) in table_ref.headers().iter().enumerate() {
                    if i < row.len() {
                        row_dict.insert(header.clone(), row[i].clone());
                    }
                }
                stack::push_id(
                    stack,
                    store_value(
                        Value::Object(Rc::new(RefCell::new(row_dict))),
                        value_store,
                        heavy_store,
                    ),
                );
            } else {
                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                    format!("Row index {} out of bounds", idx),
                    line,
                    crate::common::error::ErrorType::IndexError,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Table index must be a string (column name) or number (row index)".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
    Ok(VMStatus::Continue)
}
