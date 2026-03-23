//! Загрузка нативных модулей (.so / .dylib) по контракту datacode-abi.

use std::cell::RefCell;
use std::ffi::CStr;
use std::path::Path;
use std::rc::Rc;

use libloading::Library;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::abi::{
    abi_compatible, AbiExport, AbiModuleDescriptor, DatacodeError, DatacodeModuleEntryFn,
    DatacodeModuleFn, DatacodeModuleLegacy, DATACODE_ABI_VERSION, DATACODE_MODULE_ENTRY_SYMBOL,
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

fn push_export(
    ex: &AbiExport,
    out: &mut Vec<(String, NativeAbiFn)>,
    module_name: &str,
) -> Result<(), LangError> {
    if ex.name.is_null() {
        return Err(LangError::runtime_error(
            format!("Native module '{}' has export with null name", module_name),
            0,
        ));
    }
    let name_str = unsafe { CStr::from_ptr(ex.name).to_string_lossy().into_owned() };
    out.push((name_str, ex.func));
    Ok(())
}

/// Collect `(name, fn)` from root [`AbiModuleDescriptor`] (production entry).
fn collect_from_root_descriptor(
    desc: &AbiModuleDescriptor,
    module_name: &str,
) -> Result<Vec<(String, NativeAbiFn)>, LangError> {
    if !abi_compatible(&desc.abi_version, &DATACODE_ABI_VERSION) {
        return Err(LangError::runtime_error(
            format!(
                "Native module '{}' descriptor ABI {}.{} is not compatible with VM {}.{}",
                module_name,
                desc.abi_version.major,
                desc.abi_version.minor,
                DATACODE_ABI_VERSION.major,
                DATACODE_ABI_VERSION.minor
            ),
            0,
        ));
    }

    let mut out: Vec<(String, NativeAbiFn)> = Vec::new();

    if !desc.functions.is_null() && desc.functions_len > 0 {
        let exports = unsafe { std::slice::from_raw_parts(desc.functions, desc.functions_len) };
        for ex in exports {
            push_export(ex, &mut out, module_name)?;
        }
    }

    if !desc.classes.is_null() && desc.classes_len > 0 {
        let classes = unsafe { std::slice::from_raw_parts(desc.classes, desc.classes_len) };
        for class in classes {
            if class.name.is_null() {
                return Err(LangError::runtime_error(
                    format!("Native module '{}' has class with null name", module_name),
                    0,
                ));
            }
            let class_name = unsafe { CStr::from_ptr(class.name).to_string_lossy().into_owned() };
            if !class.methods.is_null() && class.methods_len > 0 {
                let methods =
                    unsafe { std::slice::from_raw_parts(class.methods, class.methods_len) };
                for m in methods {
                    if m.name.is_null() {
                        return Err(LangError::runtime_error(
                            format!("Native module '{}' has method with null name", module_name),
                            0,
                        ));
                    }
                    let method_name =
                        unsafe { CStr::from_ptr(m.name).to_string_lossy().into_owned() };
                    let full = format!("{class_name}.{method_name}");
                    out.push((full, m.func));
                }
            }
        }
    }

    if !desc.globals.is_null() && desc.globals_len > 0 {
        let globals =
            unsafe { std::slice::from_raw_parts(desc.globals, desc.globals_len) };
        for g in globals {
            if g.name.is_null() {
                return Err(LangError::runtime_error(
                    format!("Native module '{}' has global with null name", module_name),
                    0,
                ));
            }
            let name_str = unsafe { CStr::from_ptr(g.name).to_string_lossy().into_owned() };
            out.push((name_str, g.getter));
        }
    }

    if out.is_empty() {
        return Err(LangError::runtime_error(
            format!(
                "Native module '{}' descriptor has no exports (functions/classes/globals)",
                module_name
            ),
            0,
        ));
    }

    Ok(out)
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

    let cache_root = crate::dcmodule::dcmodule_cache_root();
    let mut dcmodule_candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = base_path {
        dcmodule_candidates.push(p.join(format!("{}.dcmodule", name)));
    }
    for pkg_root in crate::vm::file_import::get_dpm_package_paths() {
        dcmodule_candidates.push(crate::dpm::path_in_packages_directory(&pkg_root, name));
    }
    dcmodule_candidates.push(
        std::env::current_dir()
            .unwrap_or_default()
            .join(format!("{}.dcmodule", name)),
    );

    let path: std::path::PathBuf =
        if let Some(zip_path) = dcmodule_candidates.into_iter().find(|p| p.exists()) {
            crate::dcmodule::resolve_dylib_from_archive(&zip_path, &cache_root).map_err(|e| {
                LangError::runtime_error(
                    format!("Native module '{}' (.dcmodule): {}", name, e),
                    0,
                )
            })?
        } else {
            let mut candidates = Vec::new();
            if let Some(p) = base_path {
                candidates.push(p.join(&lib_name));
            }
            for pkg_root in crate::vm::file_import::get_dpm_package_paths() {
                candidates.push(pkg_root.join(name).join(&lib_name));
            }
            candidates.push(std::env::current_dir().unwrap_or_default().join(&lib_name));

            candidates.into_iter().find(|p| p.exists()).ok_or_else(|| {
                LangError::runtime_error(
                    format!(
                        "Native module '{}' not found (looked for {} or {}.dcmodule)",
                        name, lib_name, name
                    ),
                    0,
                )
            })?
        };

    let lib = unsafe { Library::new(&path) }.map_err(|e| {
        LangError::runtime_error(
            format!("Failed to load native module '{}': {}", path.display(), e),
            0,
        )
    })?;

    /// Состояние регистрации: первые три поля совпадают с [`VmContext`] (модуль видит только их).
    /// Поля alloc/throw_error/register_native не читаются в Rust, но обязательны для layout — их читает .so/.dylib.
    /// `#[repr(C)]` обязателен: без него порядок/выравнивание полей не гарантированы, и нативный модуль
    /// может вызывать неверный `register_native` → SIGSEGV.
    #[allow(dead_code)]
    #[repr(C)]
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

    // 1) Preferred: `datacode_module_entry` → root [`AbiModuleDescriptor`].
    let registered: Vec<(String, NativeAbiFn)> =
        if let Ok(get_entry) = unsafe { lib.get::<DatacodeModuleEntryFn>(DATACODE_MODULE_ENTRY_SYMBOL.as_bytes()) }
        {
            let desc_ptr = (*get_entry)();
            if !desc_ptr.is_null() {
                let desc = unsafe { &*desc_ptr };
                collect_from_root_descriptor(desc, name)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

    let registered: Vec<(String, NativeAbiFn)> = if !registered.is_empty() {
        registered
    } else {
        // 2) Transitional: `datacode_module` → [`DatacodeModule`] / legacy.
        let get_module = unsafe {
            lib.get::<DatacodeModuleFn>(DATACODE_MODULE_SYMBOL.as_bytes())
                .map_err(|e| {
                    LangError::runtime_error(
                        format!(
                            "Native module '{}' has neither '{}' nor '{}': {}",
                            name,
                            DATACODE_MODULE_ENTRY_SYMBOL,
                            DATACODE_MODULE_SYMBOL,
                            e
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

        let abi_ver = unsafe { (*module_ptr).abi_version };
        if !abi_compatible(&abi_ver, &DATACODE_ABI_VERSION) {
            return Err(LangError::runtime_error(
                format!(
                    "Native module '{}' ABI version {}.{} is not compatible with VM ABI version {}.{}",
                    name,
                    abi_ver.major,
                    abi_ver.minor,
                    DATACODE_ABI_VERSION.major,
                    DATACODE_ABI_VERSION.minor
                ),
                0,
            ));
        }

        if abi_ver.minor == 0 {
            let legacy = unsafe { &*(module_ptr as *const DatacodeModuleLegacy) };
            let mut state = RegisterState {
                alloc: abi_alloc,
                throw_error: abi_throw_error,
                register_native: capture_register_native,
                entries: Vec::new(),
            };
            (legacy.register)(&mut state as *mut RegisterState as *mut VmContext);
            state.entries
        } else {
            let m = unsafe { &*module_ptr };
            if !m.export_table.is_null() {
                let table = unsafe { &*m.export_table };
                if table.exports.is_null() || table.exports_len == 0 {
                    return Err(LangError::runtime_error(
                        format!(
                            "Native module '{}' has empty or invalid export_table.exports",
                            name
                        ),
                        0,
                    ));
                }
                let exports =
                    unsafe { std::slice::from_raw_parts(table.exports, table.exports_len) };
                let mut out = Vec::with_capacity(exports.len());
                for ex in exports {
                    push_export(ex, &mut out, name)?;
                }
                out
            } else if let Some(reg) = m.register {
                let mut state = RegisterState {
                    alloc: abi_alloc,
                    throw_error: abi_throw_error,
                    register_native: capture_register_native,
                    entries: Vec::new(),
                };
                (reg)(&mut state as *mut RegisterState as *mut VmContext);
                state.entries
            } else {
                return Err(LangError::runtime_error(
                    format!(
                        "Native module '{}' has neither export_table nor register callback",
                        name
                    ),
                    0,
                ));
            }
        }
    };
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

    if name == "ml" {
        inject_ml_layer_namespace(&mut module_object);
    }

    loaded_libs.push(lib);

    Ok(module_object)
}

/// Exposes `ml.layer.linear` / `relu` / … as aliases of the flat `*_layer` exports (tests & scripts).
fn inject_ml_layer_namespace(module_object: &mut std::collections::HashMap<String, Value>) {
    let mut layer_ns = std::collections::HashMap::new();
    let aliases: &[(&str, &str)] = &[
        ("linear", "linear_layer"),
        ("relu", "relu_layer"),
        ("softmax", "softmax_layer"),
        ("flatten", "flatten_layer"),
    ];
    for (sub, flat) in aliases {
        if let Some(v) = module_object.get(*flat).cloned() {
            layer_ns.insert((*sub).to_string(), v);
        }
    }
    module_object.insert(
        "layer".to_string(),
        Value::Object(Rc::new(RefCell::new(layer_ns))),
    );
}
