//! Disk bytecode cache (.dcb): format, serialization, fingerprint, and cache path.
//! Header: magic, compiler_version, source_mtime, source_hash (SHA-256).
//! Body: serialized Chunk + Vec<Function> using DcbConstant for constants.

use crate::bytecode::chunk::ExceptionHandlerInfo;
use crate::bytecode::function::CapturedVar;
use crate::bytecode::{Chunk, Function, OpCode};
use crate::common::value::Value;
use crate::parser::ast::TypePart;
use crate::vm::module_cache::CachedModule;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DCB_MAGIC: [u8; 4] = [0x44, 0x43, 0x42, 0x01]; // "DCB" + format version 1
pub const COMPILER_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Bump when compiler/bytecode semantics change so old .dcb are rejected (e.g. constructor names in chunk.global_names for merge).
pub const DCB_FORMAT_VERSION: &str = "9";

/// Metadata stored at the start of a .dcb file for freshness checks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DcbHeader {
    pub magic: [u8; 4],
    pub compiler_version: String,
    pub source_mtime: Option<u64>,
    pub source_hash: [u8; 32],
    /// Format version; old .dcb without this field deserialize as "" and are rejected.
    #[serde(default)]
    pub format_version: String,
}

/// Serializable constant for chunk.constants (subset of Value).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DcbConstant {
    Number(f64),
    Bool(bool),
    String(String),
    Null,
    FunctionIndex(usize),
    Array(Vec<DcbConstant>),
}

impl DcbConstant {
    pub fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Number(n) => Some(DcbConstant::Number(*n)),
            Value::Bool(b) => Some(DcbConstant::Bool(*b)),
            Value::String(s) => Some(DcbConstant::String(s.clone())),
            Value::Null => Some(DcbConstant::Null),
            Value::Function(i) => Some(DcbConstant::FunctionIndex(*i)),
            Value::Array(rc) => {
                let arr = rc.borrow();
                let mut out = Vec::with_capacity(arr.len());
                for x in arr.iter() {
                    out.push(DcbConstant::from_value(x)?);
                }
                Some(DcbConstant::Array(out))
            }
            _ => None,
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            DcbConstant::Number(n) => Value::Number(*n),
            DcbConstant::Bool(b) => Value::Bool(*b),
            DcbConstant::String(s) => Value::String(s.clone()),
            DcbConstant::Null => Value::Null,
            DcbConstant::FunctionIndex(i) => Value::Function(*i),
            DcbConstant::Array(arr) => {
                let vals: Vec<Value> = arr.iter().map(DcbConstant::to_value).collect();
                Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vals)))
            }
        }
    }
}

/// Serializable OpCode for .dcb (same variants, serde).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SerOpCode {
    Constant(usize),
    LoadLocal(usize),
    StoreLocal(usize),
    LoadGlobal(usize),
    StoreGlobal(usize),
    Add,
    Sub,
    Mul,
    Div,
    IntDiv,
    Mod,
    Pow,
    Negate,
    Not,
    Or,
    And,
    Equal,
    Greater,
    Less,
    NotEqual,
    GreaterEqual,
    LessEqual,
    In,
    JumpLabel(usize),
    JumpIfFalseLabel(usize),
    ForRange(usize, usize, usize, usize, i32),
    ForRangeNext(i32),
    PopForRange,
    Jump8(i8),
    Jump16(i16),
    Jump32(i32),
    JumpIfFalse8(i8),
    JumpIfFalse16(i16),
    JumpIfFalse32(i32),
    Call(usize),
    CallWithUnpack(usize),
    Return,
    MakeArray(usize),
    MakeArrayDynamic,
    GetArrayLength,
    GetArrayElement,
    SetArrayElement,
    TableFilter,
    Clone,
    MakeTuple(usize),
    MakeObject(usize),
    UnpackObject(usize),
    MakeObjectDynamic,
    BeginTry(usize),
    EndTry,
    Catch(Option<usize>),
    EndCatch,
    Throw(Option<usize>),
    PopExceptionHandler,
    Pop,
    Dup,
    Import(usize),
    ImportFrom(usize, usize),
    RegAdd(u8, u8, u8),
}

impl From<&OpCode> for SerOpCode {
    fn from(op: &OpCode) -> Self {
        match op {
            OpCode::Constant(a) => SerOpCode::Constant(*a),
            OpCode::LoadLocal(a) => SerOpCode::LoadLocal(*a),
            OpCode::StoreLocal(a) => SerOpCode::StoreLocal(*a),
            OpCode::LoadGlobal(a) => SerOpCode::LoadGlobal(*a),
            OpCode::StoreGlobal(a) => SerOpCode::StoreGlobal(*a),
            OpCode::Add => SerOpCode::Add,
            OpCode::Sub => SerOpCode::Sub,
            OpCode::Mul => SerOpCode::Mul,
            OpCode::Div => SerOpCode::Div,
            OpCode::IntDiv => SerOpCode::IntDiv,
            OpCode::Mod => SerOpCode::Mod,
            OpCode::Pow => SerOpCode::Pow,
            OpCode::Negate => SerOpCode::Negate,
            OpCode::Not => SerOpCode::Not,
            OpCode::Or => SerOpCode::Or,
            OpCode::And => SerOpCode::And,
            OpCode::Equal => SerOpCode::Equal,
            OpCode::Greater => SerOpCode::Greater,
            OpCode::Less => SerOpCode::Less,
            OpCode::NotEqual => SerOpCode::NotEqual,
            OpCode::GreaterEqual => SerOpCode::GreaterEqual,
            OpCode::LessEqual => SerOpCode::LessEqual,
            OpCode::In => SerOpCode::In,
            OpCode::JumpLabel(a) => SerOpCode::JumpLabel(*a),
            OpCode::JumpIfFalseLabel(a) => SerOpCode::JumpIfFalseLabel(*a),
            OpCode::ForRange(a, b, c, d, e) => SerOpCode::ForRange(*a, *b, *c, *d, *e),
            OpCode::ForRangeNext(a) => SerOpCode::ForRangeNext(*a),
            OpCode::PopForRange => SerOpCode::PopForRange,
            OpCode::Jump8(a) => SerOpCode::Jump8(*a),
            OpCode::Jump16(a) => SerOpCode::Jump16(*a),
            OpCode::Jump32(a) => SerOpCode::Jump32(*a),
            OpCode::JumpIfFalse8(a) => SerOpCode::JumpIfFalse8(*a),
            OpCode::JumpIfFalse16(a) => SerOpCode::JumpIfFalse16(*a),
            OpCode::JumpIfFalse32(a) => SerOpCode::JumpIfFalse32(*a),
            OpCode::Call(a) => SerOpCode::Call(*a),
            OpCode::CallWithUnpack(a) => SerOpCode::CallWithUnpack(*a),
            OpCode::Return => SerOpCode::Return,
            OpCode::MakeArray(a) => SerOpCode::MakeArray(*a),
            OpCode::MakeArrayDynamic => SerOpCode::MakeArrayDynamic,
            OpCode::GetArrayLength => SerOpCode::GetArrayLength,
            OpCode::GetArrayElement => SerOpCode::GetArrayElement,
            OpCode::SetArrayElement => SerOpCode::SetArrayElement,
            OpCode::TableFilter => SerOpCode::TableFilter,
            OpCode::Clone => SerOpCode::Clone,
            OpCode::MakeTuple(a) => SerOpCode::MakeTuple(*a),
            OpCode::MakeObject(a) => SerOpCode::MakeObject(*a),
            OpCode::UnpackObject(a) => SerOpCode::UnpackObject(*a),
            OpCode::MakeObjectDynamic => SerOpCode::MakeObjectDynamic,
            OpCode::BeginTry(a) => SerOpCode::BeginTry(*a),
            OpCode::EndTry => SerOpCode::EndTry,
            OpCode::Catch(a) => SerOpCode::Catch(*a),
            OpCode::EndCatch => SerOpCode::EndCatch,
            OpCode::Throw(a) => SerOpCode::Throw(*a),
            OpCode::PopExceptionHandler => SerOpCode::PopExceptionHandler,
            OpCode::Pop => SerOpCode::Pop,
            OpCode::Dup => SerOpCode::Dup,
            OpCode::Import(a) => SerOpCode::Import(*a),
            OpCode::ImportFrom(a, b) => SerOpCode::ImportFrom(*a, *b),
            OpCode::RegAdd(a, b, c) => SerOpCode::RegAdd(*a, *b, *c),
        }
    }
}

impl From<SerOpCode> for OpCode {
    fn from(s: SerOpCode) -> Self {
        match s {
            SerOpCode::Constant(a) => OpCode::Constant(a),
            SerOpCode::LoadLocal(a) => OpCode::LoadLocal(a),
            SerOpCode::StoreLocal(a) => OpCode::StoreLocal(a),
            SerOpCode::LoadGlobal(a) => OpCode::LoadGlobal(a),
            SerOpCode::StoreGlobal(a) => OpCode::StoreGlobal(a),
            SerOpCode::Add => OpCode::Add,
            SerOpCode::Sub => OpCode::Sub,
            SerOpCode::Mul => OpCode::Mul,
            SerOpCode::Div => OpCode::Div,
            SerOpCode::IntDiv => OpCode::IntDiv,
            SerOpCode::Mod => OpCode::Mod,
            SerOpCode::Pow => OpCode::Pow,
            SerOpCode::Negate => OpCode::Negate,
            SerOpCode::Not => OpCode::Not,
            SerOpCode::Or => OpCode::Or,
            SerOpCode::And => OpCode::And,
            SerOpCode::Equal => OpCode::Equal,
            SerOpCode::Greater => OpCode::Greater,
            SerOpCode::Less => OpCode::Less,
            SerOpCode::NotEqual => OpCode::NotEqual,
            SerOpCode::GreaterEqual => OpCode::GreaterEqual,
            SerOpCode::LessEqual => OpCode::LessEqual,
            SerOpCode::In => OpCode::In,
            SerOpCode::JumpLabel(a) => OpCode::JumpLabel(a),
            SerOpCode::JumpIfFalseLabel(a) => OpCode::JumpIfFalseLabel(a),
            SerOpCode::ForRange(a, b, c, d, e) => OpCode::ForRange(a, b, c, d, e),
            SerOpCode::ForRangeNext(a) => OpCode::ForRangeNext(a),
            SerOpCode::PopForRange => OpCode::PopForRange,
            SerOpCode::Jump8(a) => OpCode::Jump8(a),
            SerOpCode::Jump16(a) => OpCode::Jump16(a),
            SerOpCode::Jump32(a) => OpCode::Jump32(a),
            SerOpCode::JumpIfFalse8(a) => OpCode::JumpIfFalse8(a),
            SerOpCode::JumpIfFalse16(a) => OpCode::JumpIfFalse16(a),
            SerOpCode::JumpIfFalse32(a) => OpCode::JumpIfFalse32(a),
            SerOpCode::Call(a) => OpCode::Call(a),
            SerOpCode::CallWithUnpack(a) => OpCode::CallWithUnpack(a),
            SerOpCode::Return => OpCode::Return,
            SerOpCode::MakeArray(a) => OpCode::MakeArray(a),
            SerOpCode::MakeArrayDynamic => OpCode::MakeArrayDynamic,
            SerOpCode::GetArrayLength => OpCode::GetArrayLength,
            SerOpCode::GetArrayElement => OpCode::GetArrayElement,
            SerOpCode::SetArrayElement => OpCode::SetArrayElement,
            SerOpCode::TableFilter => OpCode::TableFilter,
            SerOpCode::Clone => OpCode::Clone,
            SerOpCode::MakeTuple(a) => OpCode::MakeTuple(a),
            SerOpCode::MakeObject(a) => OpCode::MakeObject(a),
            SerOpCode::UnpackObject(a) => OpCode::UnpackObject(a),
            SerOpCode::MakeObjectDynamic => OpCode::MakeObjectDynamic,
            SerOpCode::BeginTry(a) => OpCode::BeginTry(a),
            SerOpCode::EndTry => OpCode::EndTry,
            SerOpCode::Catch(a) => OpCode::Catch(a),
            SerOpCode::EndCatch => OpCode::EndCatch,
            SerOpCode::Throw(a) => OpCode::Throw(a),
            SerOpCode::PopExceptionHandler => OpCode::PopExceptionHandler,
            SerOpCode::Pop => OpCode::Pop,
            SerOpCode::Dup => OpCode::Dup,
            SerOpCode::Import(a) => OpCode::Import(a),
            SerOpCode::ImportFrom(a, b) => OpCode::ImportFrom(a, b),
            SerOpCode::RegAdd(a, b, c) => OpCode::RegAdd(a, b, c),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerExceptionHandlerInfo {
    pub catch_ips: Vec<usize>,
    pub error_types: Vec<Option<usize>>,
    pub error_var_slots: Vec<Option<usize>>,
    pub else_ip: Option<usize>,
    pub finally_ip: Option<usize>,
    pub stack_height: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerChunk {
    pub code: Vec<SerOpCode>,
    pub constants: Vec<DcbConstant>,
    pub lines: Vec<usize>,
    pub exception_handlers: Vec<SerExceptionHandlerInfo>,
    pub error_type_table: Vec<String>,
    pub global_names: BTreeMap<usize, String>,
    pub explicit_global_names: BTreeMap<usize, String>,
    #[serde(default)]
    pub source_name: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerCapturedVar {
    pub name: String,
    pub parent_slot_index: usize,
    pub local_slot_index: usize,
    pub ancestor_depth: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerFunction {
    pub name: String,
    pub chunk: SerChunk,
    pub arity: usize,
    pub param_names: Vec<String>,
    pub param_types: Vec<Option<Vec<TypePart>>>,
    pub return_type: Option<Vec<TypePart>>,
    pub default_values: Vec<Option<DcbConstant>>,
    pub captured_vars: Vec<SerCapturedVar>,
    pub is_cached: bool,
    pub route_method: Option<String>,
    pub route_path: Option<String>,
}

fn chunk_to_ser(chunk: &Chunk) -> Result<SerChunk, String> {
    let constants: Vec<DcbConstant> = chunk
        .constants
        .iter()
        .map(|v| DcbConstant::from_value(v).ok_or_else(|| format!("Unsupported constant for .dcb: {:?}", v)))
        .collect::<Result<_, _>>()?;
    Ok(SerChunk {
        code: chunk.code.iter().map(SerOpCode::from).collect(),
        constants,
        lines: chunk.lines.clone(),
        exception_handlers: chunk
            .exception_handlers
            .iter()
            .map(|e| SerExceptionHandlerInfo {
                catch_ips: e.catch_ips.clone(),
                error_types: e.error_types.clone(),
                error_var_slots: e.error_var_slots.clone(),
                else_ip: e.else_ip,
                finally_ip: e.finally_ip,
                stack_height: e.stack_height,
            })
            .collect(),
        error_type_table: chunk.error_type_table.clone(),
        global_names: chunk.global_names.clone(),
        explicit_global_names: chunk.explicit_global_names.clone(),
        source_name: chunk.source_name.clone(),
    })
}

fn ser_to_chunk(ser: &SerChunk) -> Chunk {
    Chunk {
        code: ser.code.iter().cloned().map(OpCode::from).collect(),
        constants: ser.constants.iter().map(DcbConstant::to_value).collect(),
        lines: ser.lines.clone(),
        exception_handlers: ser
            .exception_handlers
            .iter()
            .map(|e| ExceptionHandlerInfo {
                catch_ips: e.catch_ips.clone(),
                error_types: e.error_types.clone(),
                error_var_slots: e.error_var_slots.clone(),
                else_ip: e.else_ip,
                finally_ip: e.finally_ip,
                stack_height: e.stack_height,
            })
            .collect(),
        error_type_table: ser.error_type_table.clone(),
        global_names: ser.global_names.clone(),
        explicit_global_names: ser.explicit_global_names.clone(),
        source_name: ser.source_name.clone(),
    }
}

fn function_to_ser(f: &Function) -> Result<SerFunction, String> {
    let default_values: Vec<Option<DcbConstant>> = f
        .default_values
        .iter()
        .map(|v| v.as_ref().and_then(|v| DcbConstant::from_value(v)))
        .collect();
    Ok(SerFunction {
        name: f.name.clone(),
        chunk: chunk_to_ser(&f.chunk)?,
        arity: f.arity,
        param_names: f.param_names.clone(),
        param_types: f.param_types.clone(),
        return_type: f.return_type.clone(),
        default_values,
        captured_vars: f
            .captured_vars
            .iter()
            .map(|c| SerCapturedVar {
                name: c.name.clone(),
                parent_slot_index: c.parent_slot_index,
                local_slot_index: c.local_slot_index,
                ancestor_depth: c.ancestor_depth,
            })
            .collect(),
        is_cached: f.is_cached,
        route_method: f.route_method.clone(),
        route_path: f.route_path.clone(),
    })
}

fn ser_to_function(ser: &SerFunction) -> Function {
    use crate::bytecode::function::Function as FnDef;
    let default_values: Vec<Option<Value>> = ser
        .default_values
        .iter()
        .map(|v| v.as_ref().map(DcbConstant::to_value))
        .collect();
    let mut f = FnDef::new(ser.name.clone(), ser.arity);
    f.chunk = ser_to_chunk(&ser.chunk);
    f.param_names = ser.param_names.clone();
    f.param_types = ser.param_types.clone();
    f.return_type = ser.return_type.clone();
    f.default_values = default_values;
    f.captured_vars = ser
        .captured_vars
        .iter()
        .map(|c| CapturedVar {
            name: c.name.clone(),
            parent_slot_index: c.parent_slot_index,
            local_slot_index: c.local_slot_index,
            ancestor_depth: c.ancestor_depth,
        })
        .collect();
    f.is_cached = ser.is_cached;
    f.route_method = ser.route_method.clone();
    f.route_path = ser.route_path.clone();
    f.cache = None;
    f
}

/// Compute SHA-256 of source + compiler_version + format version for fingerprint.
pub fn source_fingerprint(source: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(COMPILER_VERSION.as_bytes());
    hasher.update(DCB_FORMAT_VERSION.as_bytes());
    hasher.finalize().into()
}

/// Path to .dcb file for a given canonical source path. Uses cache dir + hash of path.
pub fn dcb_cache_path(canonical_source_path: &Path) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(canonical_source_path.to_string_lossy().as_bytes());
    let hash = hasher.finalize();
    let name = format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}.dcb",
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7]
    );
    dcb_cache_dir().join(name)
}

/// Cache directory for .dcb files. Linux/macOS: ~/.cache/datacode/bytecode, Windows: %LOCALAPPDATA%\datacode\bytecode
fn dcb_cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir();
    #[cfg(not(target_os = "windows"))]
    let base = dirs::cache_dir();
    base.map(|p| p.join("datacode").join("bytecode"))
        .unwrap_or_else(|| PathBuf::from(".cache").join("datacode").join("bytecode"))
}

/// Read only the header from a .dcb file. Returns None if file missing or invalid.
pub fn read_dcb_header(path: &Path) -> Option<DcbHeader> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 4 + 4 || data[..4] != DCB_MAGIC {
        return None;
    }
    let header_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if data.len() < 4 + 4 + header_len {
        return None;
    }
    bincode::deserialize(&data[8..8 + header_len]).ok()
}

/// Serialize CachedModule to bytes (header + body). Returns Err if any constant is not serializable.
/// Format: magic[4] + header_len(u32 le) + header_bytes + body_bytes
pub fn serialize_cached_module(
    module: &CachedModule,
    source: &str,
    source_mtime: Option<u64>,
) -> Result<Vec<u8>, String> {
    let header = DcbHeader {
        magic: DCB_MAGIC,
        compiler_version: COMPILER_VERSION.to_string(),
        source_mtime,
        source_hash: source_fingerprint(source),
        format_version: DCB_FORMAT_VERSION.to_string(),
    };
    let ser_chunk = chunk_to_ser(&module.chunk)?;
    let ser_functions: Vec<SerFunction> = module
        .functions
        .iter()
        .map(|f| function_to_ser(f))
        .collect::<Result<_, _>>()?;
    let body = (ser_chunk, ser_functions);
    let header_bytes = bincode::serialize(&header).map_err(|e| e.to_string())?;
    let body_bytes = bincode::serialize(&body).map_err(|e| e.to_string())?;
    let header_len = header_bytes.len() as u32;
    let mut out = Vec::with_capacity(4 + 4 + header_bytes.len() + body_bytes.len());
    out.extend_from_slice(&DCB_MAGIC);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Deserialize CachedModule from bytes. Verifies magic only; caller checks version and fingerprint.
pub fn deserialize_cached_module(data: &[u8]) -> Result<CachedModule, String> {
    if data.len() < 4 + 4 || data[..4] != DCB_MAGIC {
        return Err("Invalid .dcb magic".to_string());
    }
    let header_len_u32 = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let header_len = header_len_u32 as usize;
    if data.len() < 4 + 4 + header_len {
        return Err("Truncated .dcb header".to_string());
    }
    let _header: DcbHeader =
        bincode::deserialize(&data[8..8 + header_len]).map_err(|e| e.to_string())?;
    let body: (SerChunk, Vec<SerFunction>) =
        bincode::deserialize_from(&mut std::io::Cursor::new(&data[8 + header_len..]))
            .map_err(|e| e.to_string())?;
    let chunk = ser_to_chunk(&body.0);
    let functions: Vec<Function> = body.1.iter().map(ser_to_function).collect();
    Ok(CachedModule {
        chunk,
        functions: Arc::new(functions),
    })
}

/// Load header and body from file. Format: magic[4] + header_len(u32) + header + body.
fn read_dcb_file(path: &Path) -> Option<(DcbHeader, Vec<u8>)> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 4 + 4 || data[..4] != DCB_MAGIC {
        return None;
    }
    let header_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if data.len() < 8 + header_len {
        return None;
    }
    let header: DcbHeader = bincode::deserialize(&data[8..8 + header_len]).ok()?;
    let body = data[8 + header_len..].to_vec();
    Some((header, body))
}

/// Check freshness: format_version, compiler_version and source_hash must match. Optionally source_mtime.
pub fn is_dcb_fresh(
    header: &DcbHeader,
    source: &str,
    source_mtime: Option<u64>,
) -> bool {
    if header.format_version != DCB_FORMAT_VERSION {
        return false;
    }
    if header.compiler_version != COMPILER_VERSION {
        return false;
    }
    if header.source_hash != source_fingerprint(source) {
        return false;
    }
    if let (Some(saved_mtime), Some(current_mtime)) = (header.source_mtime, source_mtime) {
        if current_mtime > saved_mtime {
            return false;
        }
    }
    true
}

/// Load .dcb if file exists and is fresh. Returns None on miss or stale.
pub fn load_dcb_if_fresh(
    dcb_path: &Path,
    source: &str,
    source_mtime: Option<u64>,
) -> Option<CachedModule> {
    let (header, body) = read_dcb_file(dcb_path)?;
    if !is_dcb_fresh(&header, source, source_mtime) {
        return None;
    }
    let (ser_chunk, ser_functions): (SerChunk, Vec<SerFunction>) =
        bincode::deserialize_from(&mut std::io::Cursor::new(&body)).ok()?;
    let chunk = ser_to_chunk(&ser_chunk);
    let functions: Vec<Function> = ser_functions.iter().map(ser_to_function).collect();
    Some(CachedModule {
        chunk,
        functions: Arc::new(functions),
    })
}

/// Write CachedModule to .dcb file. Creates cache directory if needed.
pub fn save_dcb(
    dcb_path: &Path,
    module: &CachedModule,
    source: &str,
    source_mtime: Option<u64>,
) -> Result<(), String> {
    let bytes = serialize_cached_module(module, source, source_mtime)?;
    if let Some(parent) = dcb_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(dcb_path, bytes).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_fingerprint() {
        let h1 = source_fingerprint("x = 1");
        let h2 = source_fingerprint("x = 1");
        let h3 = source_fingerprint("x = 2");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_dcb_constant_roundtrip() {
        let v = Value::Number(3.14);
        let d = DcbConstant::from_value(&v).unwrap();
        assert!(matches!(d, DcbConstant::Number(n) if (n - 3.14).abs() < 1e-9));
        assert!(matches!(d.to_value(), Value::Number(n) if (n - 3.14).abs() < 1e-9));
    }
}
