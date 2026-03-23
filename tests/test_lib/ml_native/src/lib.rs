//! Minimal native `ml` module for DataCode ABI integration tests (`tests/test_lib_native_ml.rs`).

use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once};
use std::sync::OnceLock;

use datacode_sdk::abi::{AbiExport, AbiModuleDescriptor, AbiValue, AbiVersion};

const TAG_NN: u8 = 1;
const TAG_LAYER: u8 = 2;
const TAG_LAYERS: u8 = 3;
const TAG_METHOD: u8 = 4;

#[allow(dead_code)]
enum Kind {
    Nn {
        device: String,
        num_layers: usize,
    },
    Layer {
        nn_id: u64,
        idx: usize,
        frozen: bool,
    },
    Method {
        target: u64,
        target_tag: u8,
        name: String,
    },
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static STATE: OnceLock<Mutex<HashMap<u64, Kind>>> = OnceLock::new();

fn map() -> &'static Mutex<HashMap<u64, Kind>> {
    STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn alloc_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn leak_cstr(s: &str) -> AbiValue {
    let c = CString::new(s).expect("cstr");
    let p = c.into_raw();
    AbiValue::Str(p)
}

fn abi_str(a: &AbiValue) -> Option<&str> {
    match a {
        AbiValue::Str(p) if !p.is_null() => Some(unsafe { CStr::from_ptr(*p) }.to_str().ok()?),
        _ => None,
    }
}

fn abi_to_index(a: &AbiValue) -> Option<usize> {
    match a {
        AbiValue::Int(i) if *i >= 0 => Some(*i as usize),
        AbiValue::Float(f) if *f >= 0.0 && f.fract() == 0.0 => Some(*f as usize),
        AbiValue::Str(p) if !p.is_null() => unsafe { CStr::from_ptr(*p) }
            .to_str()
            .ok()?
            .parse()
            .ok(),
        _ => None,
    }
}

fn neural_network(args: &[AbiValue]) -> AbiValue {
    let n = args
        .first()
        .map(|a| match a {
            AbiValue::Int(i) => *i,
            AbiValue::Float(f) => *f as i64,
            _ => 3,
        })
        .unwrap_or(3);
    let num_layers = (n.max(1).min(32)) as usize;
    let id = alloc_id();
    let mut m = map().lock().expect("lock");
    m.insert(
        id,
        Kind::Nn {
            device: "cpu".into(),
            num_layers,
        },
    );
    AbiValue::PluginOpaque {
        tag: TAG_NN,
        id,
    }
}

fn method_handle(target: u64, target_tag: u8, name: &str) -> AbiValue {
    let mid = alloc_id();
    let mut m = map().lock().expect("lock");
    m.insert(
        mid,
        Kind::Method {
            target,
            target_tag,
            name: name.into(),
        },
    );
    AbiValue::PluginOpaque {
        tag: TAG_METHOD,
        id: mid,
    }
}

fn dispatch_method(
    m: &mut HashMap<u64, Kind>,
    target: u64,
    target_tag: u8,
    name: &str,
    call_args: &[AbiValue],
) -> AbiValue {
    match (target_tag, name) {
        (TAG_NN, "device") => {
            if let Some(Kind::Nn { device, .. }) = m.get_mut(&target) {
                if let Some(s) = call_args.first().and_then(abi_str) {
                    *device = s.to_string();
                }
            }
            AbiValue::Null
        }
        (TAG_NN, "get_device") => {
            let dev = m
                .get(&target)
                .and_then(|k| match k {
                    Kind::Nn { device, .. } => Some(device.as_str()),
                    _ => None,
                })
                .unwrap_or("cpu");
            leak_cstr(dev)
        }
        (TAG_NN, "save") => AbiValue::Null,
        (TAG_NN, "train") | (TAG_NN, "train_sh") => {
            let mut v = vec![AbiValue::Float(0.42)];
            let len = v.len();
            let ptr = v.as_mut_ptr();
            std::mem::forget(v);
            AbiValue::Array(ptr, len)
        }
        (TAG_NN, "freeze") | (TAG_NN, "unfreeze") => AbiValue::Null,
        (TAG_LAYER, "freeze") => {
            if let Some(Kind::Layer { frozen, .. }) = m.get_mut(&target) {
                *frozen = true;
            }
            AbiValue::Null
        }
        (TAG_LAYER, "unfreeze") => {
            if let Some(Kind::Layer { frozen, .. }) = m.get_mut(&target) {
                *frozen = false;
            }
            AbiValue::Null
        }
        _ => AbiValue::Null,
    }
}

fn native_plugin_call_impl(args: &[AbiValue]) -> AbiValue {
    if args.is_empty() {
        return AbiValue::Null;
    }
    let (tag, id) = match &args[0] {
        AbiValue::PluginOpaque { tag, id } => (*tag, *id),
        _ => return AbiValue::Null,
    };

    if tag == TAG_METHOD {
        let (target, target_tag, name) = {
            let m = map().lock().expect("lock");
            match m.get(&id) {
                Some(Kind::Method {
                    target,
                    target_tag,
                    name,
                }) => (*target, *target_tag, name.clone()),
                _ => return AbiValue::Null,
            }
        };
        let tail = if args.len() > 1 { &args[1..] } else { &[] };
        let mut m = map().lock().expect("lock");
        return dispatch_method(&mut m, target, target_tag, &name, tail);
    }

    if tag == TAG_NN && args.len() == 2 {
        let key = &args[1];
        if let Some(s) = abi_str(key) {
            match s {
                "layers" => {
                    return AbiValue::PluginOpaque {
                        tag: TAG_LAYERS,
                        id,
                    };
                }
                "device" | "get_device" | "save" | "train" | "train_sh" | "freeze" | "unfreeze" => {
                    return method_handle(id, TAG_NN, s);
                }
                _ => return AbiValue::Null,
            }
        }
        return AbiValue::Null;
    }

    if tag == TAG_LAYERS && args.len() == 2 {
        if let Some(idx) = abi_to_index(&args[1]) {
            let num_layers = {
                let m = map().lock().expect("lock");
                m.get(&id)
                    .and_then(|k| match k {
                        Kind::Nn { num_layers, .. } => Some(*num_layers),
                        _ => None,
                    })
                    .unwrap_or(1)
            };
            if idx >= num_layers {
                return AbiValue::Null;
            }
            let layer_id = alloc_id();
            let mut m = map().lock().expect("lock");
            m.insert(
                layer_id,
                Kind::Layer {
                    nn_id: id,
                    idx,
                    frozen: false,
                },
            );
            return AbiValue::PluginOpaque {
                tag: TAG_LAYER,
                id: layer_id,
            };
        }
    }

    if tag == TAG_LAYER && args.len() == 2 {
        if let Some(s) = abi_str(&args[1]) {
            if matches!(s, "freeze" | "unfreeze") {
                return method_handle(id, TAG_LAYER, s);
            }
        }
    }

    AbiValue::Null
}

extern "C" fn trampoline_neural_network(
    _ctx: *mut datacode_sdk::abi::VmContext,
    args_ptr: *const AbiValue,
    argc: usize,
) -> AbiValue {
    let args: &[AbiValue] = if args_ptr.is_null() || argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args_ptr, argc) }
    };
    neural_network(args)
}

extern "C" fn trampoline_native_plugin_call(
    _ctx: *mut datacode_sdk::abi::VmContext,
    args_ptr: *const AbiValue,
    argc: usize,
) -> AbiValue {
    let args: &[AbiValue] = if args_ptr.is_null() || argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args_ptr, argc) }
    };
    native_plugin_call_impl(args)
}

static INIT: Once = Once::new();
static mut MODULE_PTR: *const AbiModuleDescriptor = std::ptr::null();

#[no_mangle]
pub extern "C" fn datacode_module_entry() -> *const AbiModuleDescriptor {
    unsafe {
        INIT.call_once(|| {
            let name = CString::new("ml").expect("module name");
            let name_ptr = name.as_ptr();
            let exports: Vec<AbiExport> = vec![
                AbiExport {
                    name: concat!("neural_network", "\0").as_ptr() as *const c_char,
                    func: trampoline_neural_network,
                    arity: 0,
                    flags: 0,
                },
                AbiExport {
                    name: concat!("native_plugin_call", "\0").as_ptr() as *const c_char,
                    func: trampoline_native_plugin_call,
                    arity: 0,
                    flags: 0,
                },
            ];
            let exports_box = exports.into_boxed_slice();
            let exports_ptr = exports_box.as_ptr();
            let exports_len = exports_box.len();
            std::mem::forget(exports_box);
            std::mem::forget(name);

            let desc = Box::new(AbiModuleDescriptor {
                abi_version: AbiVersion { major: 1, minor: 3 },
                name: name_ptr,
                functions: exports_ptr,
                functions_len: exports_len,
                classes: std::ptr::null(),
                classes_len: 0,
                globals: std::ptr::null(),
                globals_len: 0,
            });
            MODULE_PTR = Box::into_raw(desc);
        });
        MODULE_PTR
    }
}
