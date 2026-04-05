//! Загрузка нативных модулей (.so / .dylib) по контракту datacode-abi.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use libloading::Library;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::abi::{
    abi_compatible, AbiExport, AbiModuleDescriptor, AbiModuleDescriptorV4, AbiNativeParamMeta,
    AbiPluginHooksDescriptor, DatacodeError, DatacodeModuleEntryFn, DatacodeModuleFn,
    DatacodeModuleLegacy, DATACODE_ABI_VERSION, DATACODE_MODULE_ENTRY_SYMBOL, NativeAbiFn,
    VmContext, AbiValue, DATACODE_MODULE_SYMBOL,
};
use crate::common::value_store::ValueStore;
use crate::vm::abi_bridge::{AbiBridgeContext, BridgeError, materialize_value_for_abi};
use crate::vm::heavy_store::HeavyStore;

thread_local! {
    /// Ошибка, установленная ABI-нативом через throw_error. Исполнитель проверяет после вызова.
    pub(crate) static LAST_ABI_ERROR: std::cell::RefCell<Option<LangError>> = std::cell::RefCell::new(None);
}

/// Resolved [`AbiNativeParamMeta`] for storage in [`crate::vm::Vm`].
#[derive(Clone, Debug)]
pub struct ResolvedNativeParamMeta {
    pub param_names: Vec<String>,
    pub flags: u32,
}

/// Optional export-name overrides + param metadata from ABI 1.5+ root descriptor.
#[derive(Default)]
pub struct AbiModuleLoadSidecar {
    pub export_param_meta: std::collections::HashMap<String, ResolvedNativeParamMeta>,
    pub plugin_hooks: Option<PluginHookNames>,
}

/// Resolved [`AbiPluginHooksDescriptor`] (UTF-8 export names for VM wiring).
#[derive(Clone, Debug)]
pub struct PluginHookNames {
    pub native_plugin_call: Option<String>,
    pub opaque_type_name: Option<String>,
    pub opaque_display: Option<String>,
    pub dataset_len: Option<String>,
    pub opaque_binop: Option<String>,
}

fn parse_abi_native_param_meta(m: &AbiNativeParamMeta) -> Result<ResolvedNativeParamMeta, LangError> {
    let names = if m.param_names.is_null() || m.param_names_len == 0 {
        Vec::new()
    } else {
        let sl = unsafe { std::slice::from_raw_parts(m.param_names, m.param_names_len) };
        sl.iter()
            .map(|p| {
                if p.is_null() {
                    Err(LangError::runtime_error(
                        "native param meta: null param name pointer".to_string(),
                        0,
                    ))
                } else {
                    Ok(unsafe { CStr::from_ptr(*p).to_string_lossy().into_owned() })
                }
            })
            .collect::<Result<_, _>>()?
    };
    Ok(ResolvedNativeParamMeta {
        param_names: names,
        flags: m.flags,
    })
}

fn parse_plugin_hooks(ph: &AbiPluginHooksDescriptor) -> Result<PluginHookNames, LangError> {
    fn opt(p: *const std::ffi::c_char) -> Option<String> {
        if p.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
        }
    }
    Ok(PluginHookNames {
        native_plugin_call: opt(ph.native_plugin_call),
        opaque_type_name: opt(ph.opaque_type_name),
        opaque_display: opt(ph.opaque_display),
        dataset_len: opt(ph.dataset_len),
        opaque_binop: opt(ph.opaque_binop),
    })
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
/// Если `materialize` задан, срезы [`Value::ArrayView`] и вложенные представления материализуются
/// перед сериализацией в [`AbiValue`] (нужно для вызовов вроде `dataset(nested_arrays)`).
pub fn call_abi_native(
    abi_fn: NativeAbiFn,
    args: &[Value],
    materialize: Option<(&ValueStore, &HeavyStore)>,
) -> Value {
    clear_last_abi_error();
    let mut bridge = AbiBridgeContext::new();
    let abi_args_result: Result<Vec<AbiValue>, BridgeError> = match materialize {
        Some((store, heap)) => args
            .iter()
            .map(|v| bridge.value_to_abi(&materialize_value_for_abi(v, store, heap)))
            .collect(),
        None => args.iter().map(|v| bridge.value_to_abi(v)).collect(),
    };
    let abi_args: Vec<AbiValue> = match abi_args_result {
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

/// Collect `(name, fn)` + ABI 1.5 sidecar from root [`AbiModuleDescriptor`]. `desc_ptr` must point to a
/// valid descriptor; only read extended fields when `abi_version.minor >= 5`.
fn collect_from_root_descriptor(
    desc_ptr: *const AbiModuleDescriptorV4,
    module_name: &str,
) -> Result<(Vec<(String, NativeAbiFn)>, AbiModuleLoadSidecar), LangError> {
    let desc = unsafe { &*desc_ptr };
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

    let mut sidecar = AbiModuleLoadSidecar::default();
    if desc.abi_version.minor >= 5 {
        let ext = unsafe { &*(desc_ptr as *const AbiModuleDescriptor) };
        if !ext.plugin_hooks.is_null() {
            let ph = unsafe { &*ext.plugin_hooks };
            sidecar.plugin_hooks = Some(parse_plugin_hooks(ph)?);
        }
    }

    let mut out_native: Vec<(String, NativeAbiFn)> = Vec::new();

    if !desc.functions.is_null() && desc.functions_len > 0 {
        let exports = unsafe { std::slice::from_raw_parts(desc.functions, desc.functions_len) };
        for ex in exports {
            push_export(ex, &mut out_native, module_name)?;
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
                    out_native.push((full, m.func));
                }
            }
        }
    }

    if desc.abi_version.minor >= 5 {
        let ext = unsafe { &*(desc_ptr as *const AbiModuleDescriptor) };
        if !ext.native_param_metas.is_null() && ext.native_param_metas_len > 0 {
            if ext.native_param_metas_len != out_native.len() {
                return Err(LangError::runtime_error(
                    format!(
                        "Native module '{}' native_param_metas_len ({}) must match native export count ({})",
                        module_name,
                        ext.native_param_metas_len,
                        out_native.len()
                    ),
                    0,
                ));
            }
            let metas =
                unsafe { std::slice::from_raw_parts(ext.native_param_metas, ext.native_param_metas_len) };
            for (i, (name, _)) in out_native.iter().enumerate() {
                let r = parse_abi_native_param_meta(&metas[i])?;
                sidecar.export_param_meta.insert(name.clone(), r);
            }
        }
    }

    let mut out = out_native;

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

    Ok((out, sidecar))
}

/// Ключ для `typeof()` / `isinstance` по пространству имён нативного модуля (корень: `"module"`, `ml.layer`: `"layer"`).
pub(crate) const NATIVE_MODULE_TYPEOF_NAMESPACE: &str = "__plugin_namespace";

/// `layer.linear` → вложенный объект `ml.layer` с ключом `linear` (вызов `ml.layer.linear(...)`).
fn nest_dotted_module_exports(map: &mut HashMap<String, Value>) {
    // If both a flat `prefix` (NativeFunction) and one or more `prefix.suffix` keys exist, the
    // flat entry blocks nesting (the dotted branch re-inserts the dotted key). Drop the flat
    // binding so `prefix` becomes a single namespace Object built from `prefix.*` only.
    let prefixes_with_dotted: HashSet<String> = map
        .keys()
        .filter(|k| k.contains('.'))
        .filter_map(|k| k.find('.').map(|dot| k[..dot].to_string()))
        .collect();
    for prefix in prefixes_with_dotted {
        if matches!(map.get(&prefix), Some(Value::NativeFunction(_))) {
            map.remove(&prefix);
        }
    }

    let dotted: Vec<(String, Value)> = map
        .iter()
        .filter(|(k, _)| k.contains('.'))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (key, val) in dotted {
        map.remove(&key);
        let Some(dot) = key.find('.') else {
            map.insert(key, val);
            continue;
        };
        // Segment after the first `.` must preserve leading underscores (`dataset.__call__` → `__call__`).
        let prefix = key[..dot].to_string();
        let suffix = key[dot + 1..].to_string();
        if prefix.is_empty() || suffix.is_empty() {
            map.insert(key, val);
            continue;
        }
        match map.get_mut(&prefix) {
            Some(Value::Object(obj_rc)) => {
                let mut inner = obj_rc.borrow_mut();
                inner.insert(suffix, val);
                inner
                    .entry(NATIVE_MODULE_TYPEOF_NAMESPACE.to_string())
                    .or_insert_with(|| Value::String(prefix.clone()));
            }
            None => {
                let mut inner = HashMap::new();
                inner.insert(suffix, val);
                inner.insert(
                    NATIVE_MODULE_TYPEOF_NAMESPACE.to_string(),
                    Value::String(prefix.clone()),
                );
                map.insert(prefix, Value::Object(Rc::new(RefCell::new(inner))));
            }
            Some(_) => {
                map.insert(key, val);
            }
        }
    }
}

/// `DATACODE_NATIVE_MODULE_<NAME>_ROOT` — родительский каталог загруженного `lib<name>.so`/`.dylib`
/// (корень пакета DPM: `.../packages/<name>/`). Нативные модули могут искать данные относительно этого пути.
pub(crate) fn native_module_root_env_key(module_name: &str) -> String {
    let upper: String = module_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("DATACODE_NATIVE_MODULE_{}_ROOT", upper)
}

fn set_native_module_package_root_env(module_name: &str, dylib_path: &Path) {
    let Some(parent) = dylib_path.parent() else {
        return;
    };
    let root: PathBuf = parent.canonicalize().unwrap_or_else(|_| parent.to_path_buf());
    let key = native_module_root_env_key(module_name);
    if let Some(s) = root.to_str() {
        std::env::set_var(&key, s);
    }
}

/// Имя модуля импорта (`ml`) из имени файла библиотеки (`libml.dylib`, `libml.so`, `ml.dll`).
pub fn infer_module_name_from_dylib_path(lib: &Path) -> Option<String> {
    let fname = lib.file_name()?.to_str()?;
    let base = if cfg!(target_os = "windows") {
        fname.strip_suffix(".dll")?
    } else if fname.ends_with(".dylib") {
        fname.strip_suffix(".dylib")?
    } else if fname.ends_with(".so") {
        fname.strip_suffix(".so")?
    } else {
        return None;
    };
    Some(base.strip_prefix("lib").unwrap_or(base).to_string())
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
/// Если `abi_native_export_names` — `Some`, для каждого добавленного ABI-натива пушится то же имя экспорта (параллель `abi_natives`).
pub fn try_load_native_module(
    name: &str,
    base_path: Option<&Path>,
    builtin_natives_count: usize,
    abi_natives: &mut Vec<NativeAbiFn>,
    loaded_libs: &mut Vec<Library>,
    mut abi_native_export_names: Option<&mut Vec<String>>,
) -> Result<(std::collections::HashMap<String, Value>, AbiModuleLoadSidecar), LangError> {
    let lib_name = if cfg!(target_os = "macos") {
        format!("lib{}.dylib", name)
    } else if cfg!(target_os = "windows") {
        format!("{}.dll", name)
    } else {
        format!("lib{}.so", name)
    };

    let path_from_cli_override = crate::vm::file_import::get_native_lib_override().and_then(|ov| {
        if infer_module_name_from_dylib_path(&ov).as_deref() == Some(name) && ov.exists() {
            Some(ov)
        } else {
            None
        }
    });

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

    let path: std::path::PathBuf = if let Some(p) = path_from_cli_override {
        p
    } else if let Some(zip_path) = dcmodule_candidates.into_iter().find(|p| p.exists()) {
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

    set_native_module_package_root_env(name, path.as_path());

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
    let mut entry_sidecar = AbiModuleLoadSidecar::default();
    let registered_from_entry: Vec<(String, NativeAbiFn)> =
        if let Ok(get_entry) = unsafe { lib.get::<DatacodeModuleEntryFn>(DATACODE_MODULE_ENTRY_SYMBOL.as_bytes()) }
        {
            let desc_ptr = (*get_entry)();
            if !desc_ptr.is_null() {
                let (v, sc) =
                    collect_from_root_descriptor(desc_ptr as *const AbiModuleDescriptorV4, name)?;
                entry_sidecar = sc;
                v
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

    // Root descriptor may list only a static subset; `define_module!` also registers names in
    // `register_ml_exports` (e.g. `operator_descriptor`) that must be merged for preload.
    //
    // If both `datacode_module_entry` and `register` list the same name, **prefer `register`**:
    // static descriptors can lag or omit callback-only exports; entry-first would skip the real
    // trampoline when the name already appears in the root descriptor.
    let registered: Vec<(String, NativeAbiFn)> = if !registered_from_entry.is_empty() {
        let mut from_register: Vec<(String, NativeAbiFn)> = Vec::new();
        if let Ok(get_module) = unsafe { lib.get::<DatacodeModuleFn>(DATACODE_MODULE_SYMBOL.as_bytes()) } {
            let module_ptr = (*get_module)();
            if !module_ptr.is_null() {
                let abi_ver = unsafe { (*module_ptr).abi_version };
                if abi_compatible(&abi_ver, &DATACODE_ABI_VERSION) {
                    from_register = if abi_ver.minor == 0 {
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
                        if let Some(cb) = m.register {
                            let mut state = RegisterState {
                                alloc: abi_alloc,
                                throw_error: abi_throw_error,
                                register_native: capture_register_native,
                                entries: Vec::new(),
                            };
                            (cb)(&mut state as *mut RegisterState as *mut VmContext);
                            state.entries
                        } else {
                            Vec::new()
                        }
                    };
                }
            }
        }
        let mut reg = from_register;
        let mut seen: HashSet<String> = reg.iter().map(|(n, _)| n.clone()).collect();
        for (ename, f) in registered_from_entry {
            if seen.insert(ename.clone()) {
                reg.push((ename, f));
            }
        }
        reg
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
    let mut module_object = std::collections::HashMap::new();
    for (idx, (export_name, abi_fn)) in registered.into_iter().enumerate() {
        abi_natives.push(abi_fn);
        if let Some(names) = abi_native_export_names.as_mut() {
            names.push(export_name.clone());
        }
        module_object.insert(
            export_name,
            Value::NativeFunction(builtin_natives_count + abi_start + idx),
        );
    }
    nest_dotted_module_exports(&mut module_object);
    module_object.insert(
        NATIVE_MODULE_TYPEOF_NAMESPACE.to_string(),
        Value::String("module".to_string()),
    );

    loaded_libs.push(lib);

    Ok((module_object, entry_sidecar))
}

/// Merge `native_call_descriptor` export (if present) into the parse-time param registry.
pub fn merge_native_call_descriptor_from_native_module_object(
    module_object: &std::collections::HashMap<String, Value>,
    builtin_natives_count: usize,
    abi_natives: &[NativeAbiFn],
    registry: &mut crate::vm::native_call_registry::NativeCallParamRegistry,
    source_module: &str,
) -> Result<(), LangError> {
    let Some(Value::NativeFunction(i)) = module_object.get("native_call_descriptor") else {
        return Ok(());
    };
    if *i < builtin_natives_count || *i - builtin_natives_count >= abi_natives.len() {
        return Err(LangError::runtime_error(
            "native_call_descriptor native index out of range".to_string(),
            0,
        ));
    }
    let f = abi_natives[*i - builtin_natives_count];
    let v = call_abi_native(f, &[], None);
    if let Some(e) = take_last_abi_error() {
        return Err(e);
    }
    registry.merge_from_descriptor_value(&v, source_module)
}

/// Merge `operator_descriptor` ABI export into the parse-time registry.
/// `source_module` is stored on each [`crate::vm::operator_registry::OperatorInfo`] and used on conflict.
pub fn merge_operator_descriptor_from_native_module_object(
    module_object: &std::collections::HashMap<String, Value>,
    builtin_natives_count: usize,
    abi_natives: &[NativeAbiFn],
    registry: &mut crate::vm::operator_registry::OperatorRegistry,
    source_module: &str,
) -> Result<(), LangError> {
    let Some(Value::NativeFunction(i)) = module_object.get("operator_descriptor") else {
        return Ok(());
    };
    if *i < builtin_natives_count || *i - builtin_natives_count >= abi_natives.len() {
        return Err(LangError::runtime_error(
            "operator_descriptor native index out of range".to_string(),
            0,
        ));
    }
    let f = abi_natives[*i - builtin_natives_count];
    let v = call_abi_native(f, &[], None);
    if let Some(e) = take_last_abi_error() {
        return Err(e);
    }
    registry.merge_from_descriptor_value(&v, source_module)
}

#[cfg(test)]
mod tests {
    use super::{infer_module_name_from_dylib_path, nest_dotted_module_exports, try_load_native_module, NATIVE_MODULE_TYPEOF_NAMESPACE};
    use crate::abi::{DatacodeModuleFn, DATACODE_MODULE_SYMBOL};
    use crate::common::value::Value;
    use crate::vm::module_object::BUILTIN_END;
    use libloading::Library;
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn nest_dotted_dataset_double_underscore_suffix_is_call() {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("dataset.from_table".to_string(), Value::NativeFunction(0));
        m.insert("dataset.__call__".to_string(), Value::NativeFunction(1));
        nest_dotted_module_exports(&mut m);
        let Value::Object(ds) = m.get("dataset").expect("dataset namespace") else {
            panic!("dataset must be Object");
        };
        assert!(
            ds.borrow().contains_key("__call__"),
            "keys: {:?}",
            ds.borrow().keys().collect::<Vec<_>>()
        );
        assert!(!ds.borrow().contains_key("_call__"));
    }

    #[test]
    fn nest_dotted_removes_flat_prefix_when_dotted_exports_exist() {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("dataset".to_string(), Value::NativeFunction(1));
        m.insert("dataset.from_table".to_string(), Value::NativeFunction(2));
        nest_dotted_module_exports(&mut m);
        assert!(
            matches!(m.get("dataset"), Some(Value::Object(_))),
            "flat `dataset` must not block nesting when `dataset.*` keys exist; got {:?}",
            m.get("dataset")
        );
    }

    #[test]
    fn nest_dotted_sets_typeof_namespace_for_layer() {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("layer.linear".to_string(), Value::NativeFunction(0));
        nest_dotted_module_exports(&mut m);
        m.insert(
            NATIVE_MODULE_TYPEOF_NAMESPACE.to_string(),
            Value::String("module".to_string()),
        );
        assert_eq!(
            m.get("__plugin_namespace"),
            Some(&Value::String("module".to_string()))
        );
        let Value::Object(layer_rc) = m.get("layer").expect("layer") else {
            panic!("layer must be Object");
        };
        assert_eq!(
            layer_rc.borrow().get("__plugin_namespace"),
            Some(&Value::String("layer".to_string()))
        );
    }

    #[test]
    fn ml_datacode_module_uses_register_when_export_table_null() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
        let lib_path = ["target/debug/libml.dylib", "target/release/libml.dylib"]
            .into_iter()
            .map(|p| root.join(p))
            .find(|p| p.is_file());
        let Some(lib_path) = lib_path else {
            eprintln!("skip: build ml cdylib");
            return;
        };
        let lib = unsafe { Library::new(&lib_path) }.expect("Library::new ml");
        let get_module = unsafe { lib.get::<DatacodeModuleFn>(DATACODE_MODULE_SYMBOL.as_bytes()) }
            .expect("datacode_module symbol");
        let module_ptr = (*get_module)();
        assert!(!module_ptr.is_null());
        let m = unsafe { &*module_ptr };
        assert!(
            m.export_table.is_null(),
            "define_module! ml must leave export_table null so register_ml_exports runs; got {:?}",
            m.export_table
        );
        assert!(m.register.is_some(), "ml must expose register callback");
    }

    #[test]
    /// Requires `datacode_lib/ML-Datacode-lib/target/{debug,release}/libml.dylib` built from current
    /// sources (`register_ml_exports` must register `opaque_binop` / `operator_descriptor`). If
    /// `CARGO_TARGET_DIR` redirects builds away from that tree, rebuild with
    /// `cd datacode_lib/ML-Datacode-lib && cargo build` so this path updates.
    fn try_load_ml_with_lib_override_includes_operator_descriptor() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
        // Prefer debug artifact: release may be stale if only `cargo build` (debug) was run.
        let debug_lib = root.join("target/debug/libml.dylib");
        let release_lib = root.join("target/release/libml.dylib");
        let lib = if debug_lib.is_file() {
            debug_lib
        } else if release_lib.is_file() {
            release_lib
        } else {
            eprintln!("skip: build ml cdylib (expected {} or {})", debug_lib.display(), release_lib.display());
            return;
        };
        let _g = crate::vm::file_import::push_native_lib_override(Some(lib.clone()));
        let mut abi = Vec::new();
        let mut libs = Vec::new();
        let (m, _) = try_load_native_module("ml", Some(lib.parent().unwrap().as_ref()), BUILTIN_END, &mut abi, &mut libs, None)
            .expect("try_load_native_module ml with override");
        assert!(
            m.contains_key("operator_descriptor"),
            "expected operator_descriptor export, keys: {:?}",
            m.keys().collect::<Vec<_>>()
        );
        assert!(
            m.contains_key("native_call_descriptor"),
            "expected native_call_descriptor export, keys: {:?}",
            m.keys().collect::<Vec<_>>()
        );
    }

    /// `ml.dataset` must be a nested namespace Object when the plugin registers `dataset.*` exports.
    /// If this fails with `NativeFunction` for `dataset`, the loaded `libml.dylib` is not the one
    /// built from `datacode_lib/ML-Datacode-lib` (stale path or different `dlopen` resolution).
    #[test]
    #[ignore = "requires matching libml.dylib from datacode_lib/ML-Datacode-lib (see module_entry register_ml_exports)"]
    fn try_load_ml_dataset_namespace_is_object_when_lib_matches_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
        let debug_lib = root.join("target/debug/libml.dylib");
        let release_lib = root.join("target/release/libml.dylib");
        let lib = if debug_lib.is_file() {
            debug_lib
        } else if release_lib.is_file() {
            release_lib
        } else {
            eprintln!("skip: build ml cdylib");
            return;
        };
        let _g = crate::vm::file_import::push_native_lib_override(Some(lib.clone()));
        let mut abi = Vec::new();
        let mut libs = Vec::new();
        let (m, _) = try_load_native_module("ml", Some(lib.parent().unwrap().as_ref()), BUILTIN_END, &mut abi, &mut libs, None)
            .expect("try_load_native_module ml with override");
        assert!(
            matches!(m.get("dataset"), Some(Value::Object(_))),
            "`from ml import dataset` expects ml.dataset to be a namespace Object; got {:?}",
            m.get("dataset")
        );
    }

    #[test]
    fn try_load_ml_includes_module_typeof_namespace() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
        let debug_lib = root.join("target/debug/libml.dylib");
        let release_lib = root.join("target/release/libml.dylib");
        let base = if debug_lib.is_file() {
            debug_lib.parent().unwrap()
        } else if release_lib.is_file() {
            release_lib.parent().unwrap()
        } else {
            eprintln!("skip: build ml cdylib under datacode_lib/ML-Datacode-lib first");
            return;
        };
        let mut abi = Vec::new();
        let mut libs = Vec::new();
        let (m, _) = try_load_native_module("ml", Some(base), BUILTIN_END, &mut abi, &mut libs, None)
            .expect("try_load_native_module ml");
        assert_eq!(
            m.get("__plugin_namespace"),
            Some(&Value::String("module".to_string())),
            "keys: {:?}",
            m.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn infer_module_name_from_dylib_path_macos_linux() {
        assert_eq!(
            infer_module_name_from_dylib_path(Path::new("/p/libml.dylib")).as_deref(),
            Some("ml")
        );
        assert_eq!(
            infer_module_name_from_dylib_path(Path::new("/p/libfoo.so")).as_deref(),
            Some("foo")
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn infer_module_name_from_dll() {
        assert_eq!(
            infer_module_name_from_dylib_path(Path::new(r"C:\p\ml.dll")).as_deref(),
            Some("ml")
        );
        assert_eq!(
            infer_module_name_from_dylib_path(Path::new(r"C:\p\libml.dll")).as_deref(),
            Some("ml")
        );
    }
}

