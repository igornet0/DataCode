//! Host layer: abstraction for native (host) functions.
//! VM invokes natives through this layer and does not depend on concrete implementations.

use std::sync::Arc;
use crate::common::{error::LangError, value::Value};

/// Type alias for legacy function-pointer natives (used by Extended entry and modules).
pub type NativeFn = fn(&[Value]) -> Value;

/// A single host-callable native. VM calls through this trait.
pub trait HostFunction: Send + Sync {
    /// Invoke the native with the given arguments.
    fn call(&self, args: &[Value]) -> Result<Value, LangError>;
}

/// Wrapper that implements HostFunction for a raw function pointer.
#[derive(Clone, Copy)]
pub struct FnWrapper(pub NativeFn);

impl HostFunction for FnWrapper {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        Ok((self.0)(args))
    }
}

/// One entry in the VM's native table: either a trait-based builtin or a legacy Extended (fn pointer).
#[derive(Clone)]
pub enum HostEntry {
    Builtin(Arc<dyn HostFunction>),
    Extended(NativeFn),
}

impl HostEntry {
    /// Invoke this native. Returns error if the native signals failure (e.g. ABI throw_error).
    pub fn invoke(&self, args: &[Value]) -> Result<Value, LangError> {
        match self {
            HostEntry::Builtin(b) => b.as_ref().call(args),
            HostEntry::Extended(f) => Ok(f(args)),
        }
    }

    /// Raw function pointer, if this entry is Extended. Used for ptr::eq in linker and call_engine.
    pub fn as_fn_ptr(&self) -> Option<*const ()> {
        match self {
            HostEntry::Builtin(_) => None,
            HostEntry::Extended(f) => Some(*f as *const ()),
        }
    }
}
