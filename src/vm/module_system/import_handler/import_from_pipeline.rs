//! Pipeline phases for [`super::handle_import_from`]: decode operands, finalize chunk patches / argv.
//!
//! Keeps behavior identical to inlined code; larger load/import logic remains in `import_ops.rs`
//! until further decomposition.

use crate::common::{
    error::LangError,
    value::Value,
    value_store::ValueStore,
    TaggedValue,
};
use crate::vm::exceptions::ExceptionHandler;
use crate::common::value_store::ValueId;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;
use crate::vm::types::VMStatus;

/// Decoded `ImportFrom` opcode operands.
pub(crate) struct ImportFromOperands {
    pub module_name: String,
    pub items_array: Vec<Value>,
}

/// Phase 1: read constant pool module name and items array.
pub(crate) fn decode_import_from_operands(
    module_const_id: ValueId,
    items_const_id: ValueId,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<ImportFromOperands, Result<VMStatus, LangError>> {
    let module_name = match load_value(module_const_id, value_store, heavy_store) {
        Value::String(name) => name,
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "ImportFrom expects module name as string".to_string(),
                line,
            );
            return Err(ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ));
        }
    };
    let items_array = match load_value(items_const_id, value_store, heavy_store) {
        Value::Array(arr) => arr.borrow().clone(),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "ImportFrom expects items array".to_string(),
                line,
            );
            return Err(ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ));
        }
    };
    Ok(ImportFromOperands {
        module_name,
        items_array,
    })
}
