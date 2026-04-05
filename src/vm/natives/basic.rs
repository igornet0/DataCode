// Basic native functions: print, len, range, type conversions, typeof, isinstance

use crate::common::error::LangError;
use crate::common::value::{IterableInner, Value};
use crate::vm::host::HostFunction;
use crate::vm::iterable::{chunk_source_count, iterable_materialize_capacity_hint, iterable_next};
use crate::vm::vm::VM_CALL_CONTEXT;
use std::rc::Rc;
use std::cell::RefCell;
use std::io::Write;

pub fn native_print(args: &[Value]) -> Value {
    use crate::websocket::output_capture::OutputCapture;
    use crate::common::debug;
    if args.is_empty() {
        if OutputCapture::is_capturing() {
            OutputCapture::write_output("");
        } else {
            println!();
            let _ = std::io::stdout().flush();
            if debug::is_debug_enabled() {
                eprintln!("[OUTPUT]");
                let _ = std::io::stderr().flush();
            }
        }
    } else {
        let mut output = String::new();
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                output.push(' ');
            }
            let piece = if matches!(arg, Value::PluginOpaque { .. }) {
                plugin_opaque_display_via_abi(arg).unwrap_or_else(|| arg.to_string())
            } else {
                arg.to_string()
            };
            output.push_str(&piece);
        }
        if OutputCapture::is_capturing() {
            OutputCapture::write_output(&output);
        } else {
            println!("{}", output);
            let _ = std::io::stdout().flush();
            // When debug is on, also emit to stderr so script output appears in log when using > log.log 2>&1
            if debug::is_debug_enabled() {
                eprintln!("[OUTPUT] {}", output);
                let _ = std::io::stderr().flush();
            }
        }
    }
    Value::Null
}

pub fn native_len(args: &[Value]) -> Value {
    if let Some(arg) = args.first() {
        match arg {
            Value::String(s) => Value::Number(s.len() as f64),
            Value::Array(arr) => Value::Number(arr.borrow().len() as f64),
            Value::ArrayView(av) => Value::Number(av.length as f64),
            Value::ByteBuffer(b) => Value::Number(b.len as f64),
            Value::Table(table) => Value::Number(table.borrow().len() as f64),
            Value::Object(map_rc) => Value::Number(map_rc.borrow().len() as f64),
            Value::ColumnReference { table, column_name } => {
                crate::vm::vm::with_current_stores(|_store, _heap| {
                    let t = table.borrow();
                    crate::vm::table_ops::column_len(&*t, column_name)
                        .map(|len| Value::Number(len as f64))
                        .unwrap_or(Value::Null)
                })
            },
            Value::PluginOpaque { .. } => crate::vm::interpreter::object::plugin_opaque_len_via_plugin_call(arg)
                .unwrap_or(Value::Null),
            Value::Enumerate { data, .. } => Value::Number(data.borrow().len() as f64),
            Value::Iterable(rc) => match &*rc.borrow() {
                IterableInner::Chunks {
                    source,
                    chunk_size,
                    ..
                } => Value::Number(chunk_source_count(source, *chunk_size) as f64),
                _ => Value::Null,
            },
            _ => Value::Null,
        }
    } else {
        Value::Null
    }
}

pub fn native_range(args: &[Value]) -> Value {
    // Определяем параметры в зависимости от количества аргументов
    let (start, end, step) = match args.len() {
        1 => {
            // range(10) → range(0, 10, 1)
            let end = match &args[0] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            (0, end, 1)
        }
        2 => {
            // range(1, 10) → range(1, 10, 1)
            let start = match &args[0] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            let end = match &args[1] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            (start, end, 1)
        }
        3 => {
            // range(1, 10, 2) → range(1, 10, 2)
            let start = match &args[0] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            let end = match &args[1] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            let step = match &args[2] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            if step == 0 {
                return Value::Null; // Ошибка: шаг не может быть 0
            }
            (start, end, step)
        }
        _ => {
            // Ошибка будет обработана в VM при вызове
            return Value::Null;
        }
    };
    
    // Генерация массива с учетом шага
    let mut result = Vec::new();
    if step > 0 {
        let mut current = start;
        while current < end {
            result.push(Value::Number(current as f64));
            current += step;
        }
    } else {
        // Отрицательный шаг: идем в обратном направлении
        let mut current = start;
        while current > end {
            result.push(Value::Number(current as f64));
            current += step; // step уже отрицательный
        }
    }
    
    Value::Array(Rc::new(RefCell::new(result)))
}

/// enum(iterable): returns lazy (idx, element) wrapper; for (i, x) in enum(arr) yields pairs.
pub fn native_enum(args: &[Value]) -> Value {
    let iterable = match args.first() {
        Some(v) => v,
        None => return Value::Null,
    };
    match iterable {
        Value::Array(rc) => Value::Enumerate { data: Rc::clone(rc), start: 0 },
        Value::Tuple(rc) => Value::Enumerate { data: Rc::clone(rc), start: 0 },
        Value::String(s) => {
            let elements: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
            Value::Enumerate { data: Rc::new(RefCell::new(elements)), start: 0 }
        }
        _ => Value::Null,
    }
}

// Функции преобразования типов

pub fn native_int(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    match &args[0] {
        Value::Number(n) => Value::Number(n.trunc()), // Округление вниз до целого
        Value::String(s) => {
            // Парсинг строки в число
            match s.parse::<f64>() {
                Ok(n) => Value::Number(n.trunc()),
                Err(_) => Value::Number(0.0), // При ошибке парсинга возвращаем 0
            }
        }
        Value::Bool(b) => Value::Number(if *b { 1.0 } else { 0.0 }),
        Value::Null => Value::Number(0.0),
        _ => Value::Number(0.0),
    }
}

pub fn native_float(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    match &args[0] {
        Value::Number(n) => Value::Number(*n), // Уже число
        Value::String(s) => {
            // Парсинг строки в число
            match s.parse::<f64>() {
                Ok(n) => Value::Number(n),
                Err(_) => Value::Number(0.0), // При ошибке парсинга возвращаем 0.0
            }
        }
        Value::Bool(b) => Value::Number(if *b { 1.0 } else { 0.0 }),
        Value::Null => Value::Number(0.0),
        _ => Value::Number(0.0),
    }
}

pub fn native_bool(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    Value::Bool(args[0].is_truthy())
}

pub fn native_str(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    if let Value::PluginOpaque { tag, .. } = &args[0] {
        if *tag == 0 {
            if let Some(s) = plugin_tensor_repr_via_abi(&args[0]) {
                return Value::String(s);
            }
        }
    }
    Value::String(args[0].to_string())
}

/// `str(tensor)` → вложенные скобки по `shape` через `ml.native_plugin_call(_, "repr")`.
fn plugin_tensor_repr_via_abi(arg: &Value) -> Option<String> {
    let Value::PluginOpaque { tag, .. } = arg else {
        return None;
    };
    if *tag != 0 {
        return None;
    }
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow())?;
    unsafe {
        let vm = &*vm_ptr;
        let native_idx = vm.plugin_call_native?;
        let builtin_count = vm.builtin_natives_count();
        let abi_natives = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi_natives.len() {
            return None;
        }
        let call_args = [arg.clone(), Value::String("repr".to_string())];
        let v = crate::vm::native_loader::call_abi_native(
            abi_natives[native_idx - builtin_count],
            &call_args,
            Some((vm.value_store(), vm.heavy_store())),
        );
        if crate::vm::native_loader::take_last_abi_error().is_some() {
            return None;
        }
        match v {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// `array(a, b, …)` → `[a, b, …]`. `array(iterable)` materializes lazy [`Value::Iterable`] (e.g. `map` / `filter`).
pub struct ArrayHostFunction;

impl HostFunction for ArrayHostFunction {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        if args.len() == 1 {
            if let Value::Iterable(rc) = &args[0] {
                let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow()).ok_or_else(|| {
                    LangError::runtime_error(
                        "array(iterable): VM context not available".to_string(),
                        0,
                    )
                })?;
                unsafe {
                    let vm = &mut *vm_ptr;
                    let mut inner = rc.borrow().clone();
                    let cap = iterable_materialize_capacity_hint(&inner);
                    let mut out = cap.map_or_else(Vec::new, Vec::with_capacity);
                    loop {
                        match iterable_next(&mut inner, vm)? {
                            None => break,
                            Some(v) => out.push(v),
                        }
                    }
                    return Ok(Value::Array(Rc::new(RefCell::new(out))));
                }
            }
        }
        let result: Vec<Value> = args.iter().cloned().collect();
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    }
}

pub fn native_date(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => {
            // Парсим строку даты и нормализуем в ISO формат
            // Поддерживаем форматы: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SSZ, и другие ISO форматы
            let date_str = s.trim();
            
            // Если уже в формате ISO (YYYY-MM-DD или YYYY-MM-DDTHH:MM:SSZ), возвращаем как есть
            if date_str.len() >= 10 && date_str.chars().nth(4) == Some('-') && date_str.chars().nth(7) == Some('-') {
                Value::String(date_str.to_string())
            } else {
                // Для других форматов пока возвращаем как есть
                // В будущем можно добавить парсинг других форматов
                Value::String(date_str.to_string())
            }
        }
        Value::Number(n) => {
            // Если передано число (timestamp), конвертируем в ISO формат
            // Для простоты пока возвращаем как строку числа
            Value::String(format!("{}", n))
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_money(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String("0".to_string());
    }
    
    if args.len() < 2 {
        // Если формат не указан, просто возвращаем число как строку
        return Value::String(args[0].to_string());
    }
    
    let amount = match &args[0] {
        Value::Number(n) => *n,
        Value::String(s) => {
            // Пытаемся распарсить строку как число
            s.parse::<f64>().unwrap_or(0.0)
        }
        _ => 0.0,
    };
    
    let format_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => String::new(),
    };
    
    // Простое форматирование денег
    // Поддерживаем базовые паттерны: "$0.00", "0,0 $", "0 EUR"
    // Сначала проверяем паттерн "0,0" (запятая как десятичный разделитель)
    let formatted = if format_str.contains("0,0") {
        // Формат с запятой как десятичным разделителем
        let formatted_amount = format!("{:.2}", amount).replace('.', ",");
        if format_str.contains("$") {
            // "$0,0" или "0,0 $" - доллар в начале или конце
            if format_str.starts_with("$") {
                format!("${}", formatted_amount)
            } else {
                format!("{} $", formatted_amount)
            }
        } else if format_str.contains("EUR") || format_str.contains("€") {
            // "0,0 EUR" - евро в конце
            format!("{} EUR", formatted_amount)
        } else {
            // "0,0" без валюты - просто число с запятой
            let parts: Vec<&str> = format_str.split_whitespace().collect();
            if parts.len() > 1 {
                format!("{} {}", formatted_amount, parts[parts.len() - 1])
            } else {
                formatted_amount
            }
        }
    } else if format_str.contains("$") {
        // Формат с долларом (точка как десятичный разделитель)
        if format_str.contains("0.00") {
            format!("${:.2}", amount)
        } else {
            format!("${}", amount)
        }
    } else if format_str.contains("EUR") || format_str.contains("€") {
        // Формат с евро (точка как десятичный разделитель)
        if format_str.contains("0.0") {
            format!("{:.2} EUR", amount)
        } else {
            format!("{} EUR", amount)
        }
    } else {
        // Простое форматирование с двумя знаками после запятой (точка)
        format!("{:.2}", amount)
    };
    
    Value::String(formatted)
}

// Функции работы с типами

/// Имя типа для `PluginOpaque` из нативного модуля (`ml.opaque_type_name`), если загружен.
fn plugin_opaque_type_name_via_abi(arg: &Value) -> Option<String> {
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow());
    let vm_ptr = vm_ptr?;
    unsafe {
        let vm = &mut *vm_ptr;
        let native_idx = vm.plugin_typeof_opaque?;
        let builtin_count = vm.builtin_natives_count();
        let abi_natives = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi_natives.len() {
            return None;
        }
        let v = crate::vm::native_loader::call_abi_native(
            abi_natives[native_idx - builtin_count],
            std::slice::from_ref(arg),
            Some((vm.value_store(), vm.heavy_store())),
        );
        match v {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Короткая строка из `ml.opaque_display` (`<tensor tag=0 id=4>`), если загружен libml.
fn plugin_opaque_display_via_abi(arg: &Value) -> Option<String> {
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow());
    let vm_ptr = vm_ptr?;
    unsafe {
        let vm = &mut *vm_ptr;
        let native_idx = vm.plugin_opaque_display?;
        let builtin_count = vm.builtin_natives_count();
        let abi_natives = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi_natives.len() {
            return None;
        }
        let v = crate::vm::native_loader::call_abi_native(
            abi_natives[native_idx - builtin_count],
            std::slice::from_ref(arg),
            Some((vm.value_store(), vm.heavy_store())),
        );
        if crate::vm::native_loader::take_last_abi_error().is_some() {
            return None;
        }
        match v {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

pub fn native_typeof(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String("null".to_string());
    }
    if let Value::Object(map_rc) = &args[0] {
        let map = map_rc.borrow();
        if let Some(Value::String(ns)) = map.get("__plugin_namespace") {
            return Value::String(ns.clone());
        }
    }
    let plugin_opaque_ty = if matches!(&args[0], Value::PluginOpaque { .. }) {
        plugin_opaque_type_name_via_abi(&args[0])
    } else {
        None
    };
    let type_name = match &args[0] {
        Value::Number(n) => {
            // Различаем int и float по дробной части
            if n.fract() == 0.0 {
                "int"
            } else {
                "float"
            }
        }
        Value::Bool(_) => "bool",
        Value::String(s) => {
            // Проверяем, является ли строка датой или деньгами
            let s_trimmed = s.trim();
            // Проверка на дату: формат YYYY-MM-DD или ISO формат
            if s_trimmed.len() >= 10 && s_trimmed.chars().nth(4) == Some('-') && s_trimmed.chars().nth(7) == Some('-') {
                "date"
            } else if s_trimmed.starts_with('$') || s_trimmed.contains("EUR") || s_trimmed.contains("€") {
                // Проверка на деньги: содержит валютные символы
                "money"
            } else {
                "string"
            }
        }
        Value::Array(_) | Value::ArrayView(_) | Value::ByteBuffer(_) => "array",
        Value::Iterable(_) => "iterable",
        Value::Tuple(_) => "tuple",
        Value::Path(_) => "path",
        Value::Uuid(_, _) => "uuid",
        Value::Table(_) => "table",
        Value::Object(_) => "object",
        Value::ColumnReference { .. } => "column",
        Value::Null => "null",
        Value::Function(_) | Value::ModuleFunction { .. } => "function",
        Value::NativeFunction(_) => "function",
        Value::PluginOpaque { .. } => plugin_opaque_ty.as_deref().unwrap_or("plugin_opaque"),
        Value::Window(_) => "window",
        Value::Image(_) => "image",
        Value::Figure(_) => "figure",
        Value::Axis(_) => "axis",
        Value::DatabaseEngine(_) => "database_engine",
        Value::DatabaseCluster(_) => "database_cluster",
        Value::Enumerate { .. } => "enumerate",
        Value::Generator(_) => "generator",
        Value::Ellipsis => "ellipsis",
    };
    Value::String(type_name.to_string())
}

pub fn native_isinstance(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Bool(false);
    }
    native_isinstance_impl(args)
}

/// Check if an Object's __class_name or __superclass chain includes target_class.
fn object_class_chain_contains(obj: &Value, target_class: &str) -> bool {
    let Value::Object(map_rc) = obj else { return false };
    let map = map_rc.borrow();
    if let Some(Value::String(ref cn)) = map.get("__class_name") {
        if cn == target_class {
            return true;
        }
        if let Some(Value::String(ref super_name)) = map.get("__superclass") {
            if super_name == target_class {
                return true;
            }
        }
        // For Table: check __extends_table (set by compiler for classes inheriting from Table)
        if target_class == "Table" {
            if let Some(Value::Bool(true)) = map.get("__extends_table") {
                return true;
            }
        }
    }
    false
}

fn native_isinstance_impl(args: &[Value]) -> Value {
    let value = &args[0];
    // Если второй аргумент — класс (Object с __class_name), проверяем наследование
    if let Value::Object(class_map_rc) = &args[1] {
        let class_map = class_map_rc.borrow();
        if let Some(Value::String(ref target_class)) = class_map.get("__class_name") {
            let target = target_class.as_str();
            return Value::Bool(match value {
                Value::Table(_) => target == "Table",
                Value::Object(_) => object_class_chain_contains(value, target),
                _ => false,
            });
        }
    }

    // `isinstance(plugin_handle, ctor)` where `ctor` is an ABI export (e.g. flat `tensor`) or a nested
    // namespace object (`from ml import dataset` → `__plugin_namespace` == `"dataset"`).
    // Compare with `opaque_type_name` (plugin hook) — no compiler hardcoding.
    if matches!(value, Value::PluginOpaque { .. }) {
        if let Value::NativeFunction(idx) = &args[1] {
            let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow());
            if let Some(vm_ptr) = vm_ptr {
                let vm = unsafe { &*vm_ptr };
                if let Some(export_name) = vm.abi_export_name_for_native_index(*idx) {
                    if let Some(tn) = plugin_opaque_type_name_via_abi(value) {
                        return Value::Bool(tn.eq_ignore_ascii_case(export_name));
                    }
                    return Value::Bool(false);
                }
            }
        }
        if let Value::Object(map_rc) = &args[1] {
            let map = map_rc.borrow();
            if let Some(Value::String(ns)) =
                map.get(crate::vm::native_loader::NATIVE_MODULE_TYPEOF_NAMESPACE)
            {
                if let Some(tn) = plugin_opaque_type_name_via_abi(value) {
                    return Value::Bool(tn.eq_ignore_ascii_case(ns));
                }
                return Value::Bool(false);
            }
        }
    }

    // Извлекаем имя типа из второго аргумента
    // Поддерживаем как строки, так и другие типы (для констант типов, которые являются строками)
    let type_name_str = match &args[1] {
        Value::String(s) => s.clone(),
        Value::NativeFunction(index) => {
            // Если передан NativeFunction, извлекаем имя типа по индексу
            // Индексы: 0=print, 1=len, 2=range, 3=int, 4=float, 5=bool, 6=str, 7=array, 8=typeof, 9=isinstance, 10=date, 11=money, 12=path
            match *index {
                3 => "int".to_string(),
                4 => "float".to_string(),
                5 => "bool".to_string(),
                6 => "string".to_string(),
                7 => "array".to_string(),
                10 => "date".to_string(),
                11 => "money".to_string(),
                12 => "path".to_string(),
                73 => "table".to_string(),
                _ => "unknown".to_string(),
            }
        }
        // Для обратной совместимости: если это не строка, пытаемся преобразовать в строку
        // Это позволит работать с константами типов, которые уже являются строками
        other => other.to_string(),
    };
    
    // Нормализуем имя типа (приводим к нижнему регистру)
    let type_name_lower = type_name_str.to_lowercase();
    
    let matches = match value {
        Value::Number(n) => {
            // Для чисел проверяем int, float и money
            if type_name_lower == "int" || type_name_lower == "integer" || type_name_lower == "num" || type_name_lower == "number" {
                true
            } else if type_name_lower == "float" {
                n.fract() != 0.0
            } else if type_name_lower == "money" {
                // Числа могут быть деньгами
                true
            } else {
                false
            }
        }
        Value::Bool(_) => type_name_lower == "bool" || type_name_lower == "boolean",
        Value::String(s) => {
            let s_trimmed = s.trim();
            if type_name_lower == "string" || type_name_lower == "str" {
                true
            } else if type_name_lower == "date" {
                // Проверка на формат даты: YYYY-MM-DD или ISO формат
                s_trimmed.len() >= 10 && s_trimmed.chars().nth(4) == Some('-') && s_trimmed.chars().nth(7) == Some('-')
            } else if type_name_lower == "money" {
                // Проверка на деньги: содержит валютные символы
                s_trimmed.starts_with('$') || s_trimmed.contains("EUR") || s_trimmed.contains("€")
            } else {
                false
            }
        }
        Value::Path(_) => type_name_lower == "path",
        Value::Uuid(_, _) => type_name_lower == "uuid",
        Value::Array(_) | Value::ArrayView(_) | Value::ByteBuffer(_) => {
            type_name_lower == "array" || type_name_lower == "list"
        }
        Value::Tuple(_) => type_name_lower == "tuple",
        Value::Table(_) => type_name_lower == "table",
        Value::Object(map_rc) => {
            let map = map_rc.borrow();
            if let Some(Value::String(ns)) = map.get("__plugin_namespace") {
                type_name_lower == ns.to_lowercase()
            } else if type_name_lower == "object" || type_name_lower == "dict" || type_name_lower == "dictionary" {
                true
            } else if type_name_lower == "table" {
                map.get("__extends_table") == Some(&Value::Bool(true))
            } else {
                false
            }
        }
        Value::ColumnReference { .. } => type_name_lower == "column",
        Value::Null => type_name_lower == "null" || type_name_lower == "none",
        Value::Function(_) | Value::ModuleFunction { .. } | Value::NativeFunction(_) => type_name_lower == "function",
        Value::PluginOpaque { .. } => {
            if let Some(tn) = plugin_opaque_type_name_via_abi(value) {
                type_name_lower == tn.to_lowercase()
            } else {
                type_name_lower == "plugin_opaque"
            }
        }
        Value::Window(_) => type_name_lower == "window",
        Value::Image(_) => type_name_lower == "image",
        Value::Figure(_) => type_name_lower == "figure",
        Value::Axis(_) => type_name_lower == "axis",
        Value::DatabaseEngine(_) => type_name_lower == "database_engine",
        Value::DatabaseCluster(_) => type_name_lower == "database_cluster",
        Value::Enumerate { .. } => type_name_lower == "enumerate",
        Value::Iterable(_) => type_name_lower == "iterable",
        Value::Generator(_) => type_name_lower == "generator",
        Value::Ellipsis => type_name_lower == "ellipsis",
    };
    
    Value::Bool(matches)
}

/// `gen.final()` — финальное значение после `ereturn expr` (не из потока yield).
/// Если генератор ждёт ввод после `ireturn` / `x = return` (yield-await), дожимает с подстановкой RHS yield в слот.
pub fn native_generator_final(args: &[Value]) -> Value {
    // Clone Rc before any nested native runs: `execute_native_call` clears `native_args_buffer` at entry,
    // which drops the Value in the buffer while this function still holds `args` pointing into it (UAF / SIGSEGV).
    let rc = match args.first() {
        Some(Value::Generator(rc)) => rc.clone(),
        _ => return Value::Null,
    };
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow());
    if let Some(vm_ptr) = vm_ptr {
        unsafe {
            let vm = &mut *vm_ptr;
            loop {
                let g = rc.borrow_mut();
                if g.finished {
                    break;
                }
                if !g.waiting_for_input {
                    break;
                }
                drop(g);
                let res = {
                    let mut g = rc.borrow_mut();
                    crate::vm::generator::run_generator_resume(
                        vm,
                        &mut *g,
                        crate::vm::generator::GeneratorResumeMode::NextFinalDrain,
                        false,
                    )
                };
                match res {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => return Value::Null,
                }
            }
        }
    }
    let g = rc.borrow();
    if !g.finished {
        return Value::Null;
    }
    g.final_value.clone().unwrap_or(Value::Null)
}

/// `gen.next()` — следующий yield (без двустороннего канала).
pub struct NativeGeneratorNext;
impl HostFunction for NativeGeneratorNext {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow()).ok_or_else(|| {
            LangError::runtime_error("generator.next() requires an active VM".to_string(), 0)
        })?;
        let rc = match args.first() {
            Some(Value::Generator(g)) => g.clone(),
            _ => {
                return Err(LangError::runtime_error(
                    "generator.next() expects a generator".to_string(),
                    0,
                ));
            }
        };
        let mut gen = rc.borrow_mut();
        unsafe {
            let vm = &mut *vm_ptr;
            match crate::vm::generator::run_generator_resume(
                vm,
                &mut *gen,
                crate::vm::generator::GeneratorResumeMode::Next,
                false,
            ) {
                Ok(Some(v)) => Ok(v),
                Ok(None) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// `gen.send(v)` — значение в `x = return expr`.
pub struct NativeGeneratorSend;
impl HostFunction for NativeGeneratorSend {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow()).ok_or_else(|| {
            LangError::runtime_error("generator.send() requires an active VM".to_string(), 0)
        })?;
        let rc = match args.first() {
            Some(Value::Generator(g)) => g.clone(),
            _ => {
                return Err(LangError::runtime_error(
                    "generator.send() expects a generator as first argument".to_string(),
                    0,
                ));
            }
        };
        let v = args.get(1).cloned().unwrap_or(Value::Null);
        let mut gen = rc.borrow_mut();
        unsafe {
            let vm = &mut *vm_ptr;
            match crate::vm::generator::run_generator_resume(
                vm,
                &mut *gen,
                crate::vm::generator::GeneratorResumeMode::Send(v),
                false,
            ) {
                Ok(Some(v)) => Ok(v),
                Ok(None) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Built-in Table class constructor. Called as Table() or Table(path) when used as superclass.
/// Returns an Object with __class_name="Table" for inheritance (e.g. cls Base(Table) { ... }).
pub fn native_table_class(args: &[Value]) -> Value {
    use std::collections::HashMap;
    let mut obj = HashMap::new();
    obj.insert("__class_name".to_string(), Value::String("Table".to_string()));
    obj.insert("__builtin_table".to_string(), Value::Bool(true));
    obj.insert("__extends_table".to_string(), Value::Bool(true));
    if let Some(arg) = args.first() {
        obj.insert("__path".to_string(), arg.clone());
    }
    Value::Object(Rc::new(RefCell::new(obj)))
}

/// Constructor for raise ValueError("message"). Called as ValueError("..."); returns a Value whose to_string() is used for the exception message.
pub fn native_value_error_new(args: &[Value]) -> Value {
    let msg = args.get(0).map(|v| v.to_string()).unwrap_or_default();
    Value::String(msg)
}