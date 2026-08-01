// Basic native functions: print, len, range, type conversions, typeof, isinstance

use crate::common::error::LangError;
use crate::common::numeric::{coerce_to_float_value, coerce_to_int_value, float_is_int_surface, number_is_int_surface, FloatValue, IntValue};
use crate::common::value::{IterableInner, Value};
use crate::vm::host::HostFunction;
use crate::vm::iterable::{chunk_source_count, iterable_materialize_capacity_hint, iterable_next};
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

pub fn native_print(args: &[Value]) -> Value {
    use crate::common::debug;
    use crate::websocket::output_capture::OutputCapture;
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
            } else if let Some(s) = crate::vm::special_methods::try_instance_string(arg) {
                s
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
            Value::Object(map_rc) => {
                if crate::vm::special_methods::class_has_special(arg, "@len") {
                    if let Ok(Some(v)) =
                        crate::vm::special_methods::dispatch_special(arg, "@len", &[])
                    {
                        return v;
                    }
                }
                Value::Number(map_rc.borrow().len() as f64)
            }
            Value::Set(s) => Value::Number(s.borrow().len() as f64),
            Value::Tuple(t) => Value::Number(t.borrow().len() as f64),
            Value::ColumnReference { table, column_name } => {
                crate::vm::vm::with_current_stores(|_store, _heap| {
                    let t = table.borrow();
                    crate::vm::table_ops::column_len(&*t, column_name)
                        .map(|len| Value::Number(len as f64))
                        .unwrap_or(Value::Null)
                })
            }
            Value::PluginOpaque { .. } => {
                crate::vm::interpreter::object::plugin_opaque_len_via_plugin_call(arg)
                    .unwrap_or(Value::Null)
            }
            Value::Enumerate { data, .. } => Value::Number(data.borrow().len() as f64),
            Value::ObjectFieldList { element_ids, .. } => {
                Value::Number(element_ids.len() as f64)
            }
            Value::Iterable(rc) => match &*rc.borrow() {
                IterableInner::Range {
                    current,
                    end,
                    step,
                } => Value::Number(
                    crate::common::range_args::range_len(*current, *end, *step) as f64,
                ),
                IterableInner::Chunks {
                    source, chunk_size, ..
                } => Value::Number(chunk_source_count(source, *chunk_size) as f64),
                other => {
                    if let Some(n) = crate::vm::iterable::iterable_materialize_capacity_hint(other)
                    {
                        Value::Number(n as f64)
                    } else {
                        Value::Null
                    }
                }
            },
            _ => Value::Null,
        }
    } else {
        Value::Null
    }
}

pub fn native_range(args: &[Value]) -> Value {
    match crate::common::range_args::range_spec_from_values(args) {
        Ok(spec) => crate::common::range_args::value_from_range_spec(spec),
        Err(_) => Value::Null,
    }
}

/// enum(iterable): returns lazy (idx, element) wrapper; for (i, x) in enum(arr) yields pairs.
pub fn native_enum(args: &[Value]) -> Value {
    let iterable = match args.first() {
        Some(v) => v,
        None => return Value::Null,
    };
    match iterable {
        Value::Array(rc) => Value::Enumerate {
            data: Rc::clone(rc),
            start: 0,
        },
        Value::Tuple(rc) => Value::Enumerate {
            data: Rc::clone(rc),
            start: 0,
        },
        Value::String(s) => {
            let elements: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
            Value::Enumerate {
                data: Rc::new(RefCell::new(elements)),
                start: 0,
            }
        }
        Value::Table(t) => Value::Iterable(Rc::new(RefCell::new(IterableInner::EnumerateIter {
            source: Rc::new(RefCell::new(IterableInner::TableRows {
                table: Rc::clone(t),
                index: 0,
            })),
            start: 0,
            next_index: 0,
        }))),
        _ => Value::Null,
    }
}

// Функции преобразования типов

pub fn native_int(args: &[Value]) -> Value {
    let iv = if args.is_empty() {
        IntValue::Finite(0)
    } else {
        coerce_to_int_value(&args[0])
    };
    Value::Int(iv)
}

pub fn native_float(args: &[Value]) -> Value {
    let fv = if args.is_empty() {
        FloatValue::Finite(0.0)
    } else {
        coerce_to_float_value(&args[0])
    };
    Value::Float(fv)
}

/// `isinf(x)` — true for ±∞ in `int`, `float`, or legacy `number` (IEEE).
pub fn native_isinf(args: &[Value]) -> Value {
    let inf = match args.first() {
        Some(Value::Float(f)) => f.is_infinity(),
        Some(Value::Int(i)) => i.is_infinity(),
        Some(Value::Number(n)) => n.is_infinite(),
        _ => false,
    };
    Value::Bool(inf)
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
    if let Value::Set(s) = &args[0] {
        if let Some(vm_ptr) = current_vm_ptr() {
            return unsafe {
                (*vm_ptr).with_stores_mut(|store, heap| {
                    Value::String(crate::vm::set_ops::set_to_repr_string(
                        &s.borrow(),
                        store,
                        heap,
                    ))
                })
            };
        }
    }
    if let Some(s) = crate::vm::special_methods::try_instance_string(&args[0]) {
        return Value::String(s);
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
    let vm_ptr = current_vm_ptr()?;
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
                let vm_ptr = current_vm_ptr().ok_or_else(|| {
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
            if let Value::ArrayView(av) = &args[0] {
                let vm_ptr = current_vm_ptr().ok_or_else(|| {
                    LangError::runtime_error(
                        "array(array_view): VM context not available".to_string(),
                        0,
                    )
                })?;
                unsafe {
                    let vm = &mut *vm_ptr;
                    return Ok(vm.with_stores_mut(|store, heap| {
                        crate::vm::array_view::materialize_array_view(av, store, heap)
                    }));
                }
            }
            if let Value::Set(set_rc) = &args[0] {
                let vm_ptr = current_vm_ptr().ok_or_else(|| {
                    LangError::runtime_error(
                        "array(set): VM context not available".to_string(),
                        0,
                    )
                })?;
                unsafe {
                    let vm = &mut *vm_ptr;
                    return Ok(vm.with_stores_mut(|store, heap| {
                        let map = set_rc.borrow();
                        let ids = crate::vm::set_ops::set_member_key_ids(&map, store);
                        let mut out = Vec::with_capacity(ids.len());
                        for id in ids {
                            out.push(crate::vm::store_convert::load_value(id, store, heap));
                        }
                        Value::Array(Rc::new(RefCell::new(out)))
                    }));
                }
            }
        }
        let result: Vec<Value> = args.to_vec();
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    }
}

pub fn native_date(args: &[Value]) -> Value {
    use chrono::{DateTime, FixedOffset};
    match args.first() {
        Some(Value::Date(d)) => Value::Date(*d),
        Some(Value::Number(n)) => {
            let secs = n.trunc() as i64;
            let nanos = ((n.fract().abs()) * 1e9) as u32;
            DateTime::from_timestamp(secs, nanos)
                .map(|utc| {
                    Value::Date(utc.with_timezone(&FixedOffset::east_opt(0).expect("offset")))
                })
                .unwrap_or(Value::Null)
        }
        Some(Value::String(s)) => crate::vm::natives::date_format::try_parse_date(s)
            .map(Value::Date)
            .unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

/// Unix seconds as float (fractional sub-second nanoseconds).
pub fn native_date_to_unix(args: &[Value]) -> Value {
    fn unix_float(d: &chrono::DateTime<chrono::FixedOffset>) -> f64 {
        d.timestamp() as f64 + (d.timestamp_subsec_nanos() as f64) * 1e-9
    }
    match args.first() {
        Some(Value::Date(d)) => Value::Number(unix_float(d)),
        Some(Value::String(s)) => crate::vm::natives::date_format::try_parse_date(s)
            .map(|d| Value::Number(unix_float(&d)))
            .unwrap_or(Value::Null),
        Some(Value::Number(n)) => Value::Number(*n),
        _ => Value::Null,
    }
}

fn first_arg_datetime(args: &[Value]) -> Option<chrono::DateTime<chrono::FixedOffset>> {
    match args.first() {
        Some(Value::Date(d)) => Some(*d),
        Some(Value::String(s)) => crate::vm::natives::date_format::try_parse_date(s),
        _ => None,
    }
}

/// Calendar/time fields in the date's own offset; `to_utc` / `utc` — same instant with offset 0.
pub(crate) fn date_property_value(
    dt: chrono::DateTime<chrono::FixedOffset>,
    key: &str,
) -> Option<Value> {
    use chrono::{Datelike, FixedOffset, Timelike};
    match key {
        "year" => Some(Value::Number(dt.year() as f64)),
        "month" => Some(Value::Number(dt.month() as f64)),
        "quarter" => Some(Value::Number((((dt.month() - 1) / 3) + 1) as f64)),
        "day" => Some(Value::Number(dt.day() as f64)),
        "hour" => Some(Value::Number(dt.hour() as f64)),
        "minute" => Some(Value::Number(dt.minute() as f64)),
        "second" => Some(Value::Number(dt.second() as f64)),
        "weekday" => Some(Value::Number(dt.weekday().number_from_monday() as f64)),
        "to_utc" | "utc" => {
            let zulu = FixedOffset::east_opt(0)?;
            Some(Value::Date(dt.with_timezone(&zulu)))
        }
        _ => None,
    }
}

pub fn native_date_year(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "year"))
        .unwrap_or(Value::Null)
}

pub fn native_date_month(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "month"))
        .unwrap_or(Value::Null)
}

pub fn native_date_day(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "day"))
        .unwrap_or(Value::Null)
}

pub fn native_date_hour(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "hour"))
        .unwrap_or(Value::Null)
}

pub fn native_date_minute(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "minute"))
        .unwrap_or(Value::Null)
}

pub fn native_date_second(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "second"))
        .unwrap_or(Value::Null)
}

pub fn native_date_to_utc(args: &[Value]) -> Value {
    first_arg_datetime(args)
        .and_then(|dt| date_property_value(dt, "to_utc"))
        .unwrap_or(Value::Null)
}

/// Wall-clock span: `seconds + minutes*60 + hours*3600 + days*86400 + milliseconds*0.001` (all args optional, default 0).
pub fn native_duration(args: &[Value]) -> Value {
    fn n(v: Option<&Value>) -> f64 {
        v.and_then(|x| {
            if let Value::Number(n) = x {
                Some(*n)
            } else {
                None
            }
        })
        .unwrap_or(0.0)
    }
    let total_secs = n(args.first())
        + n(args.get(1)) * 60.0
        + n(args.get(2)) * 3600.0
        + n(args.get(3)) * 86_400.0
        + n(args.get(4)) * 0.001;
    let secs_i = total_secs.trunc() as i64;
    let nanos_i = (total_secs.fract() * 1e9) as i64;
    let d = chrono::Duration::seconds(secs_i)
        .checked_add(&chrono::Duration::nanoseconds(nanos_i))
        .unwrap_or_else(chrono::Duration::zero);
    Value::Duration(d)
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
    let vm_ptr = current_vm_ptr();
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
    let vm_ptr = current_vm_ptr();
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
    // Imported `SQLEnum` marker is a module object, not a user enum class instance.
    if crate::database_engine::sqenum::is_sqenum_marker_object(&args[0]) {
        return Value::String("object".to_string());
    }
    if let Some(class_name) = crate::vm::type_compat::instance_class_name(&args[0]) {
        return Value::String(class_name);
    }
    if let Value::Object(map_rc) = &args[0] {
        let map = map_rc.borrow();
        if let Some(Value::String(ns)) = map.str_key_get("__plugin_namespace") {
            return Value::String(ns.clone());
        }
    }
    let plugin_opaque_ty = if matches!(&args[0], Value::PluginOpaque { .. }) {
        plugin_opaque_type_name_via_abi(&args[0])
    } else {
        None
    };
    let type_name = match &args[0] {
        Value::Int(_) => "int",
        Value::Float(f) => {
            if float_is_int_surface(*f) {
                "int"
            } else {
                "float"
            }
        }
        Value::Number(n) => {
            if number_is_int_surface(*n) {
                "int"
            } else {
                "float"
            }
        }
        Value::Bool(_) => "bool",
        Value::Date(_) => "date",
        Value::Duration(_) => "duration",
        Value::String(s) => {
            // Проверяем, является ли строка датой или деньгами
            let s_trimmed = s.trim();
            // Проверка на дату: формат YYYY-MM-DD или ISO формат
            if s_trimmed.len() >= 10
                && s_trimmed.chars().nth(4) == Some('-')
                && s_trimmed.chars().nth(7) == Some('-')
            {
                "date"
            } else if s_trimmed.starts_with('$')
                || s_trimmed.contains("EUR")
                || s_trimmed.contains("€")
            {
                // Проверка на деньги: содержит валютные символы
                "money"
            } else {
                "string"
            }
        }
        Value::Array(_) | Value::ArrayView(_) | Value::ObjectFieldList { .. } | Value::ByteBuffer(_) => {
            "array"
        }
        Value::Iterable(_) => "iterable",
        Value::Tuple(_) => "tuple",
        Value::Path(_) => "path",
        Value::Uuid(_, _) => "uuid",
        Value::Table(_) => "table",
        Value::Object(_) => crate::vm::type_compat::primitive_display_value_type(&args[0]),
        Value::Set(_) => "set",
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
        Value::Archive(_) => "archive",
        Value::DataSource(_) => "datasource",
        Value::DataSourceResponse(_) => "response",
        Value::HttpResponse(_) => "http_response",
        Value::WebPage(_) => "web_page",
        Value::WebElement(_) => "web_element",
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

fn native_isinstance_impl(args: &[Value]) -> Value {
    let value = &args[0];
    // Если второй аргумент — класс (Object с __class_name), проверяем наследование
    if let Value::Object(class_map_rc) = &args[1] {
        let class_map = class_map_rc.borrow();
        if let Some(Value::String(ref target_class)) = class_map.str_key_get("__class_name") {
            let target = target_class.as_str();
            let class_chain = current_vm_ptr()
                .and_then(|vm_ptr| unsafe { (*vm_ptr).superclass_chain_for_instance_value(value) });
            return Value::Bool(match value {
                Value::Table(_) => target == "Table",
                Value::Object(_) => crate::vm::type_compat::value_matches_user_class_name(
                    value,
                    target,
                    class_chain.as_deref(),
                ),
                _ => false,
            });
        }
    }

    // `isinstance(plugin_handle, ctor)` where `ctor` is an ABI export (e.g. flat `tensor`) or a nested
    // namespace object (`from ml import dataset` → `__plugin_namespace` == `"dataset"`).
    // Compare with `opaque_type_name` (plugin hook) — no compiler hardcoding.
    if matches!(value, Value::PluginOpaque { .. }) {
        if let Value::NativeFunction(idx) = &args[1] {
            let vm_ptr = current_vm_ptr();
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
                map.str_key_get(crate::vm::native_loader::NATIVE_MODULE_TYPEOF_NAMESPACE)
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
            // Второй аргумент — встроенный тип/функция из глобалов (int, array, set, Table, …).
            crate::vm::globals::builtin_global_name(*index)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        }
        // Для обратной совместимости: если это не строка, пытаемся преобразовать в строку
        // Это позволит работать с константами типов, которые уже являются строками
        other => other.to_string(),
    };

    // Нормализуем имя типа (приводим к нижнему регистру)
    let type_name_lower = type_name_str.to_lowercase();

    let matches = match value {
        Value::Int(_) => {
            type_name_lower == "int"
                || type_name_lower == "integer"
                || type_name_lower == "num"
                || type_name_lower == "number"
                || type_name_lower == "money"
        }
        Value::Float(_) => {
            type_name_lower == "float"
                || type_name_lower == "num"
                || type_name_lower == "number"
                || type_name_lower == "money"
        }
        Value::Number(n) => {
            // Для чисел проверяем int, float и money
            if type_name_lower == "int"
                || type_name_lower == "integer"
                || type_name_lower == "num"
                || type_name_lower == "number"
            {
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
        Value::Date(_) => type_name_lower == "date",
        Value::Duration(_) => type_name_lower == "duration",
        Value::String(s) => {
            let s_trimmed = s.trim();
            if type_name_lower == "string" || type_name_lower == "str" {
                true
            } else if type_name_lower == "date" {
                // Проверка на формат даты: YYYY-MM-DD или ISO формат
                s_trimmed.len() >= 10
                    && s_trimmed.chars().nth(4) == Some('-')
                    && s_trimmed.chars().nth(7) == Some('-')
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
        Value::ObjectFieldList { .. } => {
            type_name_lower == "array" || type_name_lower == "list"
        }
        Value::Tuple(_) => type_name_lower == "tuple",
        Value::Table(_) => type_name_lower == "table",
        Value::Object(map_rc) => {
            let map = map_rc.borrow();
            if let Some(Value::String(ns)) = map.str_key_get("__plugin_namespace") {
                type_name_lower == ns.to_lowercase()
            } else if type_name_lower == "object"
                || type_name_lower == "dict"
                || type_name_lower == "dictionary"
            {
                true
            } else if type_name_lower == "table" {
                matches!(map.str_key_get("__extends_table"), Some(Value::Bool(true)))
            } else if !crate::vm::type_compat::is_language_primitive_type(type_name_str.as_str()) {
                let class_chain = current_vm_ptr()
                    .and_then(|vm_ptr| unsafe { (*vm_ptr).superclass_chain_for_instance_value(value) });
                crate::vm::type_compat::value_matches_user_class_name(
                    value,
                    type_name_str.as_str(),
                    class_chain.as_deref(),
                )
            } else {
                false
            }
        }
        Value::Set(_) => type_name_lower == "set",
        Value::ColumnReference { .. } => type_name_lower == "column",
        Value::Null => type_name_lower == "null" || type_name_lower == "none",
        Value::Function(_) | Value::ModuleFunction { .. } | Value::NativeFunction(_) => {
            type_name_lower == "function"
        }
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
        Value::Archive(_) => type_name_lower == "archive",
        Value::DataSource(_) => type_name_lower == "datasource",
        Value::DataSourceResponse(_) => type_name_lower == "response",
        Value::HttpResponse(_) => type_name_lower == "http_response",
        Value::WebPage(_) => type_name_lower == "web_page",
        Value::WebElement(_) => type_name_lower == "web_element",
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
    let vm_ptr = current_vm_ptr();
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
        let vm_ptr = current_vm_ptr().ok_or_else(|| {
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
        let vm_ptr = current_vm_ptr().ok_or_else(|| {
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
    obj.insert(
        "__class_name".to_string(),
        Value::String("Table".to_string()),
    );
    obj.insert("__builtin_table".to_string(), Value::Bool(true));
    obj.insert("__extends_table".to_string(), Value::Bool(true));
    if let Some(arg) = args.first() {
        obj.insert("__path".to_string(), arg.clone());
    }
    Value::legacy_object(obj)
}

/// Constructor for raise ValueError("message"). Called as ValueError("..."); returns a Value whose to_string() is used for the exception message.
pub fn native_value_error_new(args: &[Value]) -> Value {
    let msg = args.get(0).map(|v| v.to_string()).unwrap_or_default();
    Value::String(msg)
}
