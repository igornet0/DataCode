// Function object для VM

use super::chunk::Chunk;
use crate::common::value::Value;
use crate::parser::ast::TypePart;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct CapturedVar {
    pub name: String,           // Variable name
    pub parent_slot_index: usize, // Slot index in ancestor function's frame
    pub local_slot_index: usize,  // Slot index in this function's frame
    pub ancestor_depth: usize,    // Depth of ancestor (0 = immediate parent, 1 = grandparent, etc.)
}

/// Ключ для кэша функции - оптимизирован для разных случаев
#[derive(Debug, Clone)]
pub enum CacheKey {
    /// Оптимизация для двух чисел (частый случай для Ackermann)
    TwoNumbers(i64, i64),
    /// Универсальный ключ для произвольных аргументов
    Args(Vec<Value>),
}

impl CacheKey {
    /// Создает CacheKey из аргументов, если все аргументы hashable
    /// Возвращает None, если хотя бы один аргумент не hashable
    /// Оптимизирует для частых случаев (1-2 числа)
    pub fn new(args: &[Value]) -> Option<Self> {
        // Проверяем, что все аргументы hashable
        if !args.iter().all(|v| v.is_hashable()) {
            return None;
        }
        
        // Оптимизация для двух чисел (частый случай для Ackermann)
        if args.len() == 2 {
            if let (Value::Number(m), Value::Number(n)) = (&args[0], &args[1]) {
                // Преобразуем f64 в i64, если возможно (для целых чисел)
                let m_int = *m as i64;
                let n_int = *n as i64;
                // Проверяем, что преобразование было точным (целые числа)
                if (m_int as f64) == *m && (n_int as f64) == *n {
                    return Some(CacheKey::TwoNumbers(m_int, n_int));
                }
            }
        }
        
        // Универсальный случай
        Some(CacheKey::Args(args.to_vec()))
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            CacheKey::TwoNumbers(m, n) => {
                // Быстрое хеширование для двух чисел
                state.write_u8(0); // Тег для TwoNumbers
                m.hash(state);
                n.hash(state);
            }
            CacheKey::Args(args) => {
                // Хешируем количество аргументов
                state.write_u8(1); // Тег для Args
                args.len().hash(state);
                // Хешируем каждый аргумент
                for arg in args {
                    arg.hash(state);
                }
            }
        }
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CacheKey::TwoNumbers(m1, n1), CacheKey::TwoNumbers(m2, n2)) => {
                m1 == m2 && n1 == n2
            }
            (CacheKey::Args(a1), CacheKey::Args(a2)) => {
                a1 == a2
            }
            _ => false,
        }
    }
}

impl Eq for CacheKey {}

/// Кэш для функции - хранит результаты вызовов
#[derive(Debug)]
pub struct FnCache {
    pub map: HashMap<CacheKey, Value>,
}

impl FnCache {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        // Предварительно резервируем память для ~1500 записей (оптимизация для Ackermann)
        map.reserve(1500);
        Self {
            map,
        }
    }
}

#[derive(Debug)]
pub struct Function {
    pub name: String,  // Имя функции для трассировки
    pub chunk: Chunk,
    pub arity: usize, // Количество параметров
    pub param_names: Vec<String>, // Имена параметров для разрешения именованных аргументов
    pub param_types: Vec<Option<Vec<TypePart>>>, // Типы параметров (None если не указан, Vec для union: TypeName + LiteralStr)
    pub return_type: Option<Vec<TypePart>>, // Тип возвращаемого значения (union)
    pub default_values: Vec<Option<Value>>, // Значения по умолчанию для каждого параметра (None если нет default)
    pub captured_vars: Vec<CapturedVar>, // Information about captured variables
    pub is_cached: bool, // Флаг, указывающий, что функция должна кэшироваться
    pub cache: Option<Rc<RefCell<FnCache>>>, // Кэш для мемоизации (если is_cached = true)
    /// Web route: (method, path) from @route("METHOD", "/path")
    pub route_method: Option<String>,
    pub route_path: Option<String>,
    /// Module this function belongs to (e.g. "core.config", "__main__"). None = legacy single global space.
    pub module_name: Option<String>,
}

impl Function {
    pub fn new(name: String, arity: usize) -> Self {
        Self {
            name,
            chunk: Chunk::new(),
            arity,
            param_names: Vec::new(),
            param_types: Vec::new(),
            return_type: None,
            default_values: Vec::new(),
            captured_vars: Vec::new(),
            is_cached: false,
            cache: None,
            route_method: None,
            route_path: None,
            module_name: None,
        }
    }

    pub fn with_cache(name: String, arity: usize) -> Self {
        Self {
            name,
            chunk: Chunk::new(),
            arity,
            param_names: Vec::new(),
            param_types: Vec::new(),
            return_type: None,
            default_values: Vec::new(),
            captured_vars: Vec::new(),
            is_cached: true,
            cache: Some(Rc::new(RefCell::new(FnCache::new()))),
            route_method: None,
            route_path: None,
            module_name: None,
        }
    }
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            chunk: self.chunk.clone(),
            arity: self.arity,
            param_names: self.param_names.clone(),
            param_types: self.param_types.clone(),
            return_type: self.return_type.clone(),
            default_values: self.default_values.clone(),
            captured_vars: self.captured_vars.clone(),
            is_cached: self.is_cached,
            cache: self.cache.clone(),
            route_method: self.route_method.clone(),
            route_path: self.route_path.clone(),
            module_name: self.module_name.clone(),
        }
    }
}

