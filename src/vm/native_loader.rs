//! Загрузка нативных модулей (.so / .dylib) по контракту datacode-abi.

use std::ffi::CStr;
use std::path::Path;

use libloading::Library;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::abi::{
    abi_compatible, DATACODE_ABI_VERSION, DatacodeError, DatacodeModuleFn,
    NativeAbiFn, VmContext, AbiValue, DATACODE_MODULE_SYMBOL,
};
use crate::vm::abi_bridge::{AbiBridgeContext, BridgeError};

thread_local! {
    /// Ошибка, установленная ABI-нативом через throw_error. Исполнитель проверяет после вызова.
    pub(crate) static LAST_ABI_ERROR: std::cell::RefCell<Option<LangError>> = std::cell::RefCell::new(None);
}

/// Очистить последнюю ABI-ошибку (вызывать перед вызовом ABI-натива).
pub(crate) fn clear_last_abi_error() {
    LAST_ABI_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Установить ABI-ошибку (вызывается из VmContext.throw_error).
pub(crate) fn set_last_abi_error(err: LangError) {
    LAST_ABI_ERROR.with(|e| *e.borrow_mut() = Some(err));
}

/// Проверить, установлена ли ABI-ошибка после вызова натива.
pub(crate) fn take_last_abi_error() -> Option<LangError> {
    LAST_ABI_ERROR.with(|e| e.borrow_mut().take())
}

/// Аллокатор для ABI: использует глобальный аллокатор.
extern "C" fn abi_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    match std::alloc::Layout::from_size_align(size, 1) {
        Ok(layout) => unsafe { std::alloc::alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Вызвать ABI-натив: конвертация аргументов/возврата через мост.
/// Исполнитель вызывает это, когда native_index >= natives.len().
pub fn call_abi_native(abi_fn: NativeAbiFn, args: &[Value]) -> Value {
    clear_last_abi_error();
    let mut bridge = AbiBridgeContext::new();
    let abi_args: Vec<AbiValue> = match args
        .iter()
        .map(|v| bridge.value_to_abi(v))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(a) => a,
        Err(BridgeError::Unrepresentable(msg)) => {
            set_last_abi_error(LangError::runtime_error(
                format!("ABI bridge: {}", msg),
                0,
            ));
            return Value::Null;
        }
        Err(BridgeError::InvalidUtf8) => {
            set_last_abi_error(LangError::runtime_error(
                "ABI bridge: invalid UTF-8 in string".to_string(),
                0,
            ));
            return Value::Null;
        }
        Err(BridgeError::InvalidHandle) => {
            set_last_abi_error(LangError::runtime_error(
                "ABI bridge: invalid handle".to_string(),
                0,
            ));
            return Value::Null;
        }
    };
    let argc = abi_args.len();
    let args_ptr = if abi_args.is_empty() {
        std::ptr::null()
    } else {
        abi_args.as_ptr()
    };
    let mut ctx = VmContext {
        alloc: abi_alloc,
        throw_error: abi_throw_error,
        register_native: abi_register_native_noop,
    };
    let result_abi = abi_fn(&mut ctx as *mut VmContext, args_ptr, argc);
    match bridge.abi_to_value(result_abi) {
        Ok(v) => v,
        Err(BridgeError::Unrepresentable(msg)) => {
            set_last_abi_error(LangError::runtime_error(
                format!("ABI bridge (return): {}", msg),
                0,
            ));
            Value::Null
        }
        Err(BridgeError::InvalidUtf8) => {
            set_last_abi_error(LangError::runtime_error(
                "ABI bridge (return): invalid UTF-8".to_string(),
                0,
            ));
            Value::Null
        }
        Err(BridgeError::InvalidHandle) => {
            set_last_abi_error(LangError::runtime_error(
                "ABI bridge (return): invalid handle".to_string(),
                0,
            ));
            Value::Null
        }
    }
}

extern "C" fn abi_register_native_noop(_ctx: *mut VmContext, _name: *const std::ffi::c_char, _func: NativeAbiFn) {
    // Используется только при вызове ABI-натива (модуль не вызывает register_native из натива).
}

extern "C" fn abi_throw_error(code: DatacodeError, msg: *const std::ffi::c_char) {
    let message = if msg.is_null() {
        "ABI error".to_string()
    } else {
        unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
    };
    let error_type = match code {
        DatacodeError::Ok => crate::common::error::ErrorType::RuntimeError,
        DatacodeError::TypeError => crate::common::error::ErrorType::TypeError,
        DatacodeError::RuntimeError => crate::common::error::ErrorType::RuntimeError,
        DatacodeError::Panic => crate::common::error::ErrorType::RuntimeError,
    };
    set_last_abi_error(LangError::runtime_error_with_type(message, 0, error_type));
}

/// Попытаться загрузить нативный модуль по имени.
/// Ищет `lib<name>.so` (Unix) или `lib<name>.dylib` (macOS) в base_path и текущей директории.
/// При успехе добавляет ABI-нативы в `abi_natives`, библиотеку в `loaded_libs`;
/// возвращает объект модуля (имя -> Value::NativeFunction(индекс)), индексы = builtin_natives_count + смещение в abi_natives.
pub fn try_load_native_module(
    name: &str,
    base_path: Option<&Path>,
    builtin_natives_count: usize,
    abi_natives: &mut Vec<NativeAbiFn>,
    loaded_libs: &mut Vec<Library>,
) -> Result<std::collections::HashMap<String, Value>, LangError> {
    let lib_name = if cfg!(target_os = "macos") {
        format!("lib{}.dylib", name)
    } else if cfg!(target_os = "windows") {
        format!("{}.dll", name)
    } else {
        format!("lib{}.so", name)
    };

    let mut candidates = Vec::new();
    if let Some(p) = base_path {
        candidates.push(p.join(&lib_name));
    }
    candidates.push(std::env::current_dir().unwrap_or_default().join(&lib_name));

    let path = candidates
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| {
            LangError::runtime_error(
                format!("Native module '{}' not found (looked for {})", name, lib_name),
                0,
            )
        })?;

    let lib = unsafe { Library::new(&path) }.map_err(|e| {
        LangError::runtime_error(
            format!("Failed to load native module '{}': {}", path.display(), e),
            0,
        )
    })?;

    let get_module = unsafe {
        lib.get::<DatacodeModuleFn>(DATACODE_MODULE_SYMBOL.as_bytes())
            .map_err(|e| {
                LangError::runtime_error(
                    format!(
                        "Native module '{}' has no symbol '{}': {}",
                        name, DATACODE_MODULE_SYMBOL, e
                    ),
                    0,
                )
            })?
    };

    let module_ptr = (*get_module)();
    if module_ptr.is_null() {
        return Err(LangError::runtime_error(
            format!("Native module '{}' returned null from datacode_module()", name),
            0,
        ));
    }

    let module = unsafe { &*module_ptr };
    if !abi_compatible(&module.abi_version, &DATACODE_ABI_VERSION) {
        return Err(LangError::runtime_error(
            format!(
                "Native module '{}' ABI version {}.{} is not compatible with VM ABI version {}.{}",
                name,
                module.abi_version.major,
                module.abi_version.minor,
                DATACODE_ABI_VERSION.major,
                DATACODE_ABI_VERSION.minor
            ),
            0,
        ));
    }

    /// Состояние регистрации: первые три поля совпадают с VmContext (модуль видит только их).
    /// Поля alloc/throw_error/register_native не читаются в Rust, но обязательны для layout — их читает .so/.dylib.
    #[allow(dead_code)]
    struct RegisterState {
        alloc: extern "C" fn(usize) -> *mut u8,
        throw_error: extern "C" fn(DatacodeError, *const std::ffi::c_char),
        register_native: extern "C" fn(*mut VmContext, *const std::ffi::c_char, NativeAbiFn),
        entries: Vec<(String, NativeAbiFn)>,
    }

    extern "C" fn capture_register_native(
        ctx: *mut VmContext,
        name: *const std::ffi::c_char,
        func: NativeAbiFn,
    ) {
        if ctx.is_null() || name.is_null() {
            return;
        }
        let state = unsafe { &mut *(ctx as *mut RegisterState) };
        let name_str = unsafe { CStr::from_ptr(name).to_string_lossy().into_owned() };
        state.entries.push((name_str, func));
    }

    let mut state = RegisterState {
        alloc: abi_alloc,
        throw_error: abi_throw_error,
        register_native: capture_register_native,
        entries: Vec::new(),
    };

    (module.register)(&mut state as *mut RegisterState as *mut VmContext);

    let registered = state.entries;
    let abi_start = abi_natives.len();
    for (_export_name, abi_fn) in &registered {
        abi_natives.push(*abi_fn);
    }

    let mut module_object = std::collections::HashMap::new();
    for (idx, (export_name, _)) in registered.into_iter().enumerate() {
        module_object.insert(
            export_name,
            Value::NativeFunction(builtin_natives_count + abi_start + idx),
        );
    }

    loaded_libs.push(lib);

    Ok(module_object)
}
