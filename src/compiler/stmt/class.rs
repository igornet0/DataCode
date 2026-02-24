/// Компиляция class statements

/// Reserved global index used only for LoadGlobal(current_class) when loading model_config in Settings subclasses.
/// Using a sentinel avoids chunk.global_names[class_global_index] being overwritten by default_factory insert
/// (factory_global_index == class_global_index in some layouts), which would make update_chunk_indices patch to the wrong class.
const MODEL_CONFIG_CLASS_LOAD_INDEX: usize = 0x0FFF_FFFF;

use crate::debug_println;
use crate::parser::ast::{Arg, Expr, Stmt, Param};
use crate::bytecode::{OpCode, Function};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::stmt;
use crate::compiler::expr;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Emit bytecode to set this.__extends_table when the class extends Table (for isinstance(x, Table)).
fn emit_instance_extends_table(ctx: &mut CompilationContext, line: usize, this_slot: usize, class_name: &str) {
    if !ctx.class_extends_table.get(class_name).copied().unwrap_or(false) {
        return;
    }
    let extends_const = ctx.chunk.add_constant(Value::Bool(true));
    ctx.chunk.write_with_line(OpCode::Constant(extends_const), line);
    let key_const = ctx.chunk.add_constant(Value::String("__extends_table".to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(key_const), line);
    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), line);
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), line);
}

/// Emit bytecode to set this.__private_fields, this.__private_field_defining_class, and this.__class_name on the instance (for VM privacy checks).
/// private_field_defining_class maps field name -> class that defined it (for subclass private access check).
fn emit_instance_private_metadata(
    ctx: &mut CompilationContext,
    line: usize,
    this_slot: usize,
    class_name: &str,
    merged_private_field_names: &[String],
    private_field_defining_class: &HashMap<String, String>,
) {
    // Set this.__private_fields = array of field name strings
    let private_names_value = Value::Array(Rc::new(RefCell::new(
        merged_private_field_names
            .iter()
            .map(|n| Value::String(n.clone()))
            .collect::<Vec<_>>(),
    )));
    let private_fields_const = ctx.chunk.add_constant(private_names_value);
    ctx.chunk.write_with_line(OpCode::Constant(private_fields_const), line);
    let key_const = ctx.chunk.add_constant(Value::String("__private_fields".to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(key_const), line);
    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), line);
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), line);
    // Set this.__private_field_defining_class = map field -> defining class (for subclass private access check)
    let defining_class_map: HashMap<String, Value> = private_field_defining_class
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    let defining_class_value = Value::Object(Rc::new(RefCell::new(defining_class_map)));
    let defining_class_const = ctx.chunk.add_constant(defining_class_value);
    ctx.chunk.write_with_line(OpCode::Constant(defining_class_const), line);
    let defining_key_const = ctx.chunk.add_constant(Value::String("__private_field_defining_class".to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(defining_key_const), line);
    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), line);
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), line);
    // Set this.__class_name = class name
    let class_name_const = ctx.chunk.add_constant(Value::String(class_name.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(class_name_const), line);
    let class_key_const = ctx.chunk.add_constant(Value::String("__class_name".to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(class_key_const), line);
    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), line);
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), line);
}

/// Extract default_factory identifier from a field's default_value if it is Field(default_factory=Name).
fn extract_default_factory_name(default_value: &Option<Expr>) -> Option<String> {
    let expr = default_value.as_ref()?;
    let args = match expr {
        Expr::Call { name, args, .. } if name == "Field" => args,
        _ => return None,
    };
    for arg in args {
        if let Arg::Named { name: n, value } = arg {
            if n == "default_factory" {
                if let Expr::Variable { name, .. } = value {
                    return Some(name.clone());
                }
                return None;
            }
        }
    }
    None
}

/// Extract env_prefix from class variable model_config = Settings.config(env_prefix="APP__", ...).
fn extract_env_prefix_from_class_vars(
    public_variables: &[crate::parser::ast::ClassVariable],
    private_variables: &[crate::parser::ast::ClassVariable],
) -> Option<String> {
    for var in public_variables.iter().chain(private_variables.iter()) {
        if var.name == "model_config" {
            let args = match &var.value {
                Expr::Call { args, .. } => args,
                Expr::MethodCall { args, .. } => args,
                _ => { break; }
            };
            for arg in args {
                if let Arg::Named { name, value } = arg {
                    if name == "env_prefix" {
                        if let Expr::Literal { value: v, .. } = value {
                            if let Value::String(s) = v {
                                return Some(s.clone());
                            }
                        }
                        break;
                    }
                }
            }
            break;
        }
    }
    None
}

/// Emit bytecode to set this.__protected_fields on the instance (for VM protected checks).
fn emit_instance_protected_metadata(
    ctx: &mut CompilationContext,
    line: usize,
    this_slot: usize,
    merged_protected_field_names: &[String],
) {
    let protected_names_value = Value::Array(Rc::new(RefCell::new(
        merged_protected_field_names
            .iter()
            .map(|n| Value::String(n.clone()))
            .collect::<Vec<_>>(),
    )));
    let protected_fields_const = ctx.chunk.add_constant(protected_names_value);
    ctx.chunk.write_with_line(OpCode::Constant(protected_fields_const), line);
    let key_const = ctx.chunk.add_constant(Value::String("__protected_fields".to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(key_const), line);
    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), line);
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), line);
}

/// Emit bytecode to set this.__class on the instance (for @class parameter injection in methods).
fn emit_instance_class_reference(
    ctx: &mut CompilationContext,
    line: usize,
    this_slot: usize,
    class_global_index: usize,
) {
    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), line);
    let key_const = ctx.chunk.add_constant(Value::String("__class".to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(key_const), line);
    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), line);
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), line);
}

pub fn compile_class(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Class { name, superclass, is_abstract, private_fields, protected_fields, public_fields, private_variables, protected_variables, public_variables, constructors, methods, line } = stmt {
        *ctx.current_line = *line;
        if *is_abstract {
            ctx.abstract_classes.insert(name.clone());
        }
        // Compute class_extends_table: true if superclass is Table or extends Table
        let extends_table = superclass.as_ref().map(|s| {
            s == "Table" || ctx.class_extends_table.get(s).copied().unwrap_or(false)
        }).unwrap_or(false);
        ctx.class_extends_table.insert(name.clone(), extends_table);

        // Создаем класс-объект с метаданными
        let mut class_metadata = HashMap::new();
        
        // Сохраняем информацию о полях
        let private_field_names: Vec<String> = private_fields.iter().map(|f| f.name.clone()).collect();
        let protected_field_names: Vec<String> = protected_fields.iter().map(|f| f.name.clone()).collect();
        let public_field_names: Vec<String> = public_fields.iter().map(|f| f.name.clone()).collect();
        let class_private_var_names: Vec<String> = private_variables.iter().map(|v| v.name.clone()).collect();
        let class_protected_var_names: Vec<String> = protected_variables.iter().map(|v| v.name.clone()).collect();
        
        class_metadata.insert("__class_name".to_string(), Value::String(name.clone()));
        if *is_abstract {
            class_metadata.insert("__abstract".to_string(), Value::Bool(true));
        }
        if let Some(ref super_name) = superclass {
            class_metadata.insert("__superclass".to_string(), Value::String(super_name.clone()));
        }
        class_metadata.insert("__private_fields".to_string(), Value::Array(
            std::rc::Rc::new(std::cell::RefCell::new(
                private_field_names.iter().map(|n| Value::String(n.clone())).collect()
            ))
        ));
        class_metadata.insert("__protected_fields".to_string(), Value::Array(
            std::rc::Rc::new(std::cell::RefCell::new(
                protected_field_names.iter().map(|n| Value::String(n.clone())).collect()
            ))
        ));
        class_metadata.insert("__public_fields".to_string(), Value::Array(
            std::rc::Rc::new(std::cell::RefCell::new(
                public_field_names.iter().map(|n| Value::String(n.clone())).collect()
            ))
        ));
        class_metadata.insert("__class_private_vars".to_string(), Value::Array(
            std::rc::Rc::new(std::cell::RefCell::new(
                class_private_var_names.iter().map(|n| Value::String(n.clone())).collect()
            ))
        ));
        class_metadata.insert("__class_protected_vars".to_string(), Value::Array(
            std::rc::Rc::new(std::cell::RefCell::new(
                class_protected_var_names.iter().map(|n| Value::String(n.clone())).collect()
            ))
        ));
        // ORM: column names in declaration order (for CREATE TABLE column order)
        if extends_table {
            let col_names: Vec<Value> = private_fields.iter().chain(protected_fields.iter()).chain(public_fields.iter())
                .filter(|f| f.default_value.is_some())
                .map(|f| Value::String(f.name.clone()))
                .collect();
            class_metadata.insert("__col_names".to_string(), Value::Array(Rc::new(RefCell::new(col_names))));
        }
        // API for @class parameter: identity, hierarchy, structure
        class_metadata.insert("name".to_string(), Value::String(name.clone()));
        class_metadata.insert("full_name".to_string(), Value::String(name.clone()));
        class_metadata.insert(
            "parent".to_string(),
            superclass
                .as_ref()
                .map(|s| Value::String(s.clone()))
                .unwrap_or(Value::Null),
        );
        if *is_abstract {
            class_metadata.insert("is_abstract".to_string(), Value::Bool(true));
        }
        let method_names_value = Value::Array(Rc::new(RefCell::new(
            methods.iter().map(|m| Value::String(m.name.clone())).collect(),
        )));
        class_metadata.insert("method_names".to_string(), method_names_value);
        
        // ВАЖНО: Сначала объявляем все методы (forward declaration), чтобы их индексы были известны
        // при компиляции конструкторов (конструкторы должны добавлять методы в объект)
        let mut method_indices = std::collections::HashMap::new();
        for method in methods.iter() {
            let method_name = format!("{}::method_{}", name, method.name);
            
            // Check if method has @class as first parameter (injected by VM at call time)
            let has_at_class = method.params.first().map(|p| p.name == "@class").unwrap_or(false);
            let user_params: Vec<_> = if has_at_class {
                method.params.iter().skip(1).collect()
            } else {
                method.params.iter().collect()
            };
            let arity = 1 + (if has_at_class { 1 } else { 0 }) + user_params.len();
            
            let mut method_function = Function::new(method_name.clone(), arity);
            
            // First parameter - this (implicit); second - @class if present (injected by VM)
            let mut param_names = vec!["this".to_string()];
            let mut param_types = vec![None];
            if has_at_class {
                param_names.push("@class".to_string());
                param_types.push(None);
            }
            for param in &user_params {
                param_names.push(param.name.clone());
                param_types.push(param.type_annotation.clone());
            }
            
            method_function.param_names = param_names;
            method_function.param_types = param_types;
            method_function.return_type = method.return_type.clone();
            
            // Сохраняем функцию (forward declaration)
            let function_index = ctx.functions.len();
            ctx.functions.push(method_function.clone());
            ctx.function_names.push(method_name.clone());
            
            // Сохраняем индекс метода для использования в конструкторах
            method_indices.insert(method.name.clone(), function_index);
            
            // Регистрируем метод в глобальной области видимости
            let global_index = ctx.scope.globals.len();
            ctx.scope.globals.insert(method_name.clone(), global_index);
        }
        
        // Резервируем глобальный индекс для класса, чтобы конструктор мог загрузить класс (model_config)
        let class_global_index = ctx.scope.globals.len();
        ctx.scope.globals.insert(name.clone(), class_global_index);
        
        // Неявный конструктор при наследовании без явных конструкторов
        let constructors_to_compile: Vec<&crate::parser::ast::Constructor> = if superclass.is_some() && constructors.is_empty() {
            // Синтетический конструктор с одним параметром (path) — генерируем вручную ниже
            vec![]
        } else {
            constructors.iter().collect()
        };

        if superclass.is_some() && constructors.is_empty() {
            // For extends_table: generate implicit constructor that takes one param per instance field (ORM-style).
            // Use all instance fields (private + protected + public) so classes without "public:" still get the constructor.
            let all_instance_fields: Vec<&crate::parser::ast::ClassField> = private_fields
                .iter()
                .chain(protected_fields.iter())
                .chain(public_fields.iter())
                .collect();
            if extends_table && !all_instance_fields.is_empty() {
                let params: Vec<Param> = all_instance_fields
                    .iter()
                    .map(|f| Param {
                        name: f.name.clone(),
                        type_annotation: f.type_annotation.clone(),
                        default_value: f.default_value.clone(),
                    })
                    .collect();
                let n = params.len();
                let constructor_name = format!("{}::new_{}", name, n);
                let mut constructor_function = Function::new(constructor_name.clone(), n);
                constructor_function.param_names = params.iter().map(|p| p.name.clone()).collect();
                constructor_function.param_types = params.iter().map(|p| p.type_annotation.clone()).collect();
                // ORM constructor: params without constant default get Null when omitted (per plan); use false for bool columns
                let mut default_values = Vec::with_capacity(n);
                for param in &params {
                    let default = match param.default_value.as_ref().and_then(|e| crate::compiler::constant_fold::evaluate_constant_expr(e).ok().flatten()) {
                        Some(v) => Some(v),
                        None => {
                            let ty_lower = param.type_annotation.as_ref().map(|tys| tys.iter().map(|t| t.to_lowercase()).collect::<Vec<_>>());
                            let default = if param.name == "id" && param.type_annotation.as_ref().map(|tys| tys.iter().any(|t| t.to_lowercase().contains("int"))).unwrap_or(false) {
                                Value::Null
                            } else if ty_lower.as_ref().map(|tys| tys.iter().any(|t| t.contains("bool"))).unwrap_or(false) {
                                Value::Bool(false)
                            } else if ty_lower.as_ref().map(|tys| tys.iter().any(|t| t.contains("int") || t.contains("float") || t.contains("number") || t.contains("num"))).unwrap_or(false) {
                                Value::Number(0.0)
                            } else if ty_lower.as_ref().map(|tys| tys.iter().any(|t| t.contains("str") || t.contains("string"))).unwrap_or(false) {
                                Value::String(String::new())
                            } else {
                                Value::Null
                            };
                            Some(default)
                        }
                    };
                    default_values.push(default);
                }
                constructor_function.default_values = default_values;

                // Allow null for params that have default Null (e.g. id with autoincrement)
                for (i, default_val) in constructor_function.default_values.iter().enumerate() {
                    if default_val.as_ref() == Some(&Value::Null) {
                        if let Some(ref mut types) = constructor_function.param_types[i] {
                            let null_s = "null".to_string();
                            if !types.contains(&null_s) {
                                types.push(null_s);
                            }
                        } else {
                            constructor_function.param_types[i] = Some(vec!["null".to_string()]);
                        }
                    }
                }

                // ORM constructor receives row values (str, int, etc.), not Column refs; skip type check
                constructor_function.param_types = vec![None; n];

                let function_index = ctx.functions.len();
                ctx.functions.push(constructor_function.clone());
                ctx.function_names.push(constructor_name.clone());

                let global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(constructor_name.clone(), global_index);
                ctx.chunk.global_names.insert(global_index, constructor_name.clone());

                let saved_chunk = std::mem::replace(&mut *ctx.chunk, constructor_function.chunk.clone());
                let saved_function = ctx.current_function;
                let saved_local_count = ctx.scope.local_count;

                ctx.current_function = Some(function_index);
                ctx.scope.local_count = 0;
                ctx.scope.begin_scope();

                for param in &params {
                    ctx.scope.declare_local(&param.name);
                }
                let this_slot = ctx.scope.declare_local("this");
                ctx.chunk.global_names.insert(class_global_index, name.clone());

                let table_global_index = *ctx.scope.globals.get("Table").ok_or_else(|| LangError::ParseError {
                    message: "Table not found in scope (required for extends_table class with implicit field constructor)".to_string(),
                    line: *line,
                })?;
                ctx.chunk.write_with_line(OpCode::LoadGlobal(table_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(0), *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);

                let class_name_const = ctx.chunk.add_constant(Value::String(name.clone()));
                let class_key_const = ctx.chunk.add_constant(Value::String("__class_name".to_string()));
                ctx.chunk.write_with_line(OpCode::Constant(class_name_const), *line);
                ctx.chunk.write_with_line(OpCode::Constant(class_key_const), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);

                let extends_true_const = ctx.chunk.add_constant(Value::Bool(true));
                let extends_key_const = ctx.chunk.add_constant(Value::String("__extends_table".to_string()));
                ctx.chunk.write_with_line(OpCode::Constant(extends_true_const), *line);
                ctx.chunk.write_with_line(OpCode::Constant(extends_key_const), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);

                for (i, field) in all_instance_fields.iter().enumerate() {
                    ctx.chunk.write_with_line(OpCode::LoadLocal(i), *line);
                    let field_name_const = ctx.chunk.add_constant(Value::String(field.name.clone()));
                    ctx.chunk.write_with_line(OpCode::Constant(field_name_const), *line);
                    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                    ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                }

                emit_instance_class_reference(ctx, *line, this_slot, class_global_index);

                for (_, method) in methods.iter().enumerate() {
                    if let Some(&method_function_index) = method_indices.get(&method.name) {
                        let method_function_const = ctx.chunk.add_constant(Value::Function(method_function_index));
                        ctx.chunk.write_with_line(OpCode::Constant(method_function_const), *line);
                        let method_name_const = ctx.chunk.add_constant(Value::String(method.name.clone()));
                        ctx.chunk.write_with_line(OpCode::Constant(method_name_const), *line);
                        ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                        ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                        ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                    }
                }

                ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                ctx.chunk.write_with_line(OpCode::Return, *line);

                ctx.functions[function_index].chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
                ctx.scope.end_scope();
                ctx.current_function = saved_function;
                ctx.scope.local_count = saved_local_count;

                ctx.chunk.global_names.insert(global_index, constructor_name.clone());
                let constructor_constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                ctx.chunk.write_with_line(OpCode::Constant(constructor_constant_index), *line);
                ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);

                ctx.class_constructor.insert(name.clone(), (constructor_name.clone(), function_index));
            } else {
            let superclass_name = superclass.as_ref().unwrap();
            // Если суперкласс — класс, вызываем его конструктор (Parent::new_1); иначе — функцию (Base)
            // Если суперкласс не имеет конструктора с 1 аргументом — пропускаем генерацию неявного конструктора
            // и помечаем в class_superclass; ошибка будет выдана при компиляции вызова (строка вызова).
            let super_callable = {
                let constructor_name = format!("{}::new_1", superclass_name);
                if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                    Some((constructor_name, idx))
                } else {
                    let has_constructor = (1..=10).any(|n| {
                        ctx.scope.globals.contains_key(&format!("{}::new_{}", superclass_name, n))
                    });
                    if has_constructor {
                        ctx.class_superclass.insert(name.clone(), superclass_name.clone());
                        None
                    } else {
                        let idx = *ctx.scope.globals.get(superclass_name).ok_or_else(|| LangError::ParseError {
                            message: format!("Superclass '{}' not found in scope", superclass_name),
                            line: *line,
                        })?;
                        Some((superclass_name.clone(), idx))
                    }
                }
            };
            if let Some((super_callable_name, supercall_global_index)) = super_callable {
            let constructor_name = format!("{}::new_1", name);
            let mut constructor_function = Function::new(constructor_name.clone(), 1);
            constructor_function.param_names = vec!["path".to_string()];
            constructor_function.param_types = vec![None]; // без типа, чтобы родитель мог принимать любой аргумент
            let function_index = ctx.functions.len();
            ctx.functions.push(constructor_function.clone());
            ctx.function_names.push(constructor_name.clone());
            let global_index = ctx.scope.globals.len();
            ctx.scope.globals.insert(constructor_name.clone(), global_index);
            ctx.chunk.global_names.insert(global_index, constructor_name.clone());
            let saved_chunk = std::mem::replace(&mut *ctx.chunk, constructor_function.chunk.clone());
            let saved_function = ctx.current_function;
            let saved_local_count = ctx.scope.local_count;
            ctx.current_function = Some(function_index);
            ctx.scope.begin_scope();
            ctx.scope.declare_local("path");
            let this_slot = ctx.scope.declare_local("this");
            // Ensure class name is in chunk.global_names first so update_chunk_indices_from_names can patch LoadGlobal(class_global_index) correctly when the module is imported.
            ctx.chunk.global_names.insert(class_global_index, name.clone());
            ctx.chunk.global_names.insert(supercall_global_index, super_callable_name.clone());
            // When superclass is Settings: pass (path, required_keys, model_config) for validation and config
            let is_settings_subclass = superclass_name == "Settings";
            if is_settings_subclass {
                let env_prefix = extract_env_prefix_from_class_vars(public_variables, private_variables);
                let required_env_keys: Vec<String> = public_fields
                    .iter()
                    .filter(|f| f.default_value.is_none())
                    .map(|f| {
                        if env_prefix.is_some() {
                            f.name.to_lowercase()
                        } else {
                            format!("{}{}", env_prefix.as_ref().map(|p| p.as_str()).unwrap_or("").to_lowercase(), f.name.to_lowercase())
                        }
                    })
                    .collect();
                let required_keys_value = Value::Array(Rc::new(RefCell::new(
                    required_env_keys
                        .iter()
                        .map(|k| Value::String(k.clone()))
                        .collect(),
                )));
                let required_keys_const_index = ctx.chunk.add_constant(required_keys_value);
                let model_config_name_const = ctx.chunk.add_constant(Value::String("model_config".to_string()));
                ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line);
                ctx.chunk.write_with_line(OpCode::Constant(required_keys_const_index), *line);
                // Use a dedicated sentinel index so this LoadGlobal is never overwritten by default_factory (factory_global_index can equal class_global_index in some layouts).
                ctx.chunk.global_names.insert(MODEL_CONFIG_CLASS_LOAD_INDEX, name.clone());
                ctx.chunk.write_with_line(OpCode::LoadGlobal(MODEL_CONFIG_CLASS_LOAD_INDEX), *line);
                ctx.chunk.write_with_line(OpCode::Constant(model_config_name_const), *line);
                ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                ctx.chunk.write_with_line(OpCode::LoadGlobal(supercall_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(3), *line);
            } else {
                // VM Call pops callee first (top), then args: stack must be [arg, callee]
                ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line);
                ctx.chunk.write_with_line(OpCode::LoadGlobal(supercall_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(1), *line);
            }
            ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
            // Update __class_name and __superclass when subclassing (e.g. Base(Table)); parent may return object with its own __class_name
            if superclass_name != "Settings" {
                let class_name_const = ctx.chunk.add_constant(Value::String(name.clone()));
                ctx.chunk.write_with_line(OpCode::Constant(class_name_const), *line);
                let class_key_const = ctx.chunk.add_constant(Value::String("__class_name".to_string()));
                ctx.chunk.write_with_line(OpCode::Constant(class_key_const), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                let super_const = ctx.chunk.add_constant(Value::String(superclass_name.clone()));
                ctx.chunk.write_with_line(OpCode::Constant(super_const), *line);
                let super_key_const = ctx.chunk.add_constant(Value::String("__superclass".to_string()));
                ctx.chunk.write_with_line(OpCode::Constant(super_key_const), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                emit_instance_extends_table(ctx, *line, this_slot, name);
            }
            emit_instance_class_reference(ctx, *line, this_slot, class_global_index);
            // For Settings subclasses: fill missing fields that have default_factory (e.g. db: DatabaseConfig = Field(default_factory=DatabaseConfig))
            let pending_jumps_before_default_factory = ctx.labels.pending_jumps.len();
            if is_settings_subclass {
                for f in public_fields.iter() {
                    if let Some(factory_name) = extract_default_factory_name(&f.default_value) {
                        let factory_global_index = *ctx.scope.globals.get(&factory_name).ok_or_else(|| LangError::ParseError {
                            message: format!("default_factory '{}' not found in scope for field '{}'", factory_name, f.name),
                            line: *line,
                        })?;
                        ctx.chunk.global_names.insert(factory_global_index, factory_name.clone());
                        let skip_label = ctx.labels.create_label();
                        let field_name_const = ctx.chunk.add_constant(Value::String(f.name.clone()));
                        // if result[field] is null, set result[field] = factory()
                        ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                        ctx.chunk.write_with_line(OpCode::Constant(field_name_const), *line);
                        ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                        ctx.chunk.write_with_line(OpCode::Clone, *line);
                        let null_const = ctx.chunk.add_constant(Value::Null);
                        ctx.chunk.write_with_line(OpCode::Constant(null_const), *line);
                        ctx.chunk.write_with_line(OpCode::Equal, *line);
                        ctx.labels.emit_jump(ctx.chunk, *line, true, skip_label)?;
                        ctx.chunk.write_with_line(OpCode::Pop, *line);
                        ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line);
                        ctx.chunk.write_with_line(OpCode::LoadGlobal(factory_global_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(1), *line);
                        ctx.chunk.write_with_line(OpCode::Constant(field_name_const), *line);
                        ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                        ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                        ctx.labels.mark_label(skip_label, ctx.chunk.code.len());
                    }
                }
            }
            // Re-insert sentinel and class name after default_factory loop so they are not overwritten by factory_global_index (when factory_global_index == class_global_index in some layouts).
            ctx.chunk.global_names.insert(MODEL_CONFIG_CLASS_LOAD_INDEX, name.clone());
            ctx.chunk.global_names.insert(class_global_index, name.clone());
            for (_, method) in methods.iter().enumerate() {
                if let Some(&method_function_index) = method_indices.get(&method.name) {
                    let method_function_const = ctx.chunk.add_constant(Value::Function(method_function_index));
                    ctx.chunk.write_with_line(OpCode::Constant(method_function_const), *line);
                    let method_name_const = ctx.chunk.add_constant(Value::String(method.name.clone()));
                    ctx.chunk.write_with_line(OpCode::Constant(method_name_const), *line);
                    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                    ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                }
            }
            ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
            ctx.chunk.write_with_line(OpCode::Return, *line);
            if ctx.labels.pending_jumps.len() > pending_jumps_before_default_factory {
                ctx.labels.finalize_jumps_from(ctx.chunk, pending_jumps_before_default_factory, *line)?;
            }
            // Final guarantee: class name in chunk.global_names so update_chunk_indices_from_names patches LoadGlobal(class_global_index) correctly.
            ctx.chunk.global_names.insert(class_global_index, name.clone());
            ctx.functions[function_index].chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
            ctx.scope.end_scope();
            ctx.current_function = saved_function;
            ctx.scope.local_count = saved_local_count;
            ctx.chunk.global_names.insert(global_index, constructor_name.clone());
            let constructor_constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            ctx.chunk.write_with_line(OpCode::Constant(constructor_constant_index), *line);
            ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
            } // end if let Some(super_callable)
            } // end else (path-based implicit constructor)
        }

        for constructor in constructors_to_compile {
            let constructor_name = format!("{}::new_{}", name, constructor.params.len());
            
            // Создаем функцию для конструктора
            let mut constructor_function = Function::new(constructor_name.clone(), constructor.params.len());
            constructor_function.param_names = constructor.params.iter().map(|p| p.name.clone()).collect();
            constructor_function.param_types = constructor.params.iter().map(|p| p.type_annotation.clone()).collect();
            
            // Сохраняем функцию
            let function_index = ctx.functions.len();
            ctx.functions.push(constructor_function.clone());
            ctx.function_names.push(constructor_name.clone());
            
            // Регистрируем конструктор в глобальной области видимости
            let global_index = ctx.scope.globals.len();
            ctx.scope.globals.insert(constructor_name.clone(), global_index);
            // Также регистрируем в chunk.global_names для экспорта из модулей
            ctx.chunk.global_names.insert(global_index, constructor_name.clone());
            
            // Компилируем тело конструктора
            let saved_chunk = std::mem::replace(&mut *ctx.chunk, constructor_function.chunk.clone());
            let saved_function = ctx.current_function;
            let saved_local_count = ctx.scope.local_count;
            
            ctx.current_function = Some(function_index);
            ctx.scope.local_count = 0;
            ctx.scope.begin_scope();
            
            // Параметры конструктора будут в локальных слотах, начиная с 0
            for param in &constructor.params {
                ctx.scope.declare_local(&param.name);
            }
            
            let this_slot = ctx.scope.declare_local("this");
            assert_eq!(
                this_slot,
                constructor.params.len(),
                "constructor '{}': this_slot ({}) must equal arity ({})",
                constructor_name,
                this_slot,
                constructor.params.len()
            );
            // Ensure class name is in chunk.global_names so update_chunk_indices_from_names can patch LoadGlobal(class_global_index) correctly when the module is imported.
            ctx.chunk.global_names.insert(class_global_index, name.clone());
            
            // Track how many statements to skip when compiling body
            // (1 when explicit super() in body was already compiled)
            let mut body_skip_count: usize = 0;
            
            if let Some(ref super_name) = superclass {
                if let Some(ref delegate_args) = constructor.delegate_args {
                    let arity = delegate_args.len();
                    // Если суперкласс — класс, вызываем его конструктор (Parent::new_N); иначе — функцию (Base)
                    let (super_callable_name, supercall_global_index) = {
                        let parent_constructor = format!("{}::new_{}", super_name, arity);
                        if let Some(&idx) = ctx.scope.globals.get(&parent_constructor) {
                            (parent_constructor, idx)
                        } else {
                            let idx = *ctx.scope.globals.get(super_name).ok_or_else(|| LangError::ParseError {
                                message: format!("Superclass '{}' not found in scope", super_name),
                                line: *line,
                            })?;
                            (super_name.clone(), idx)
                        }
                    };
                    ctx.chunk.global_names.insert(supercall_global_index, super_callable_name);
                    // VM Call pops callee first (top), then args: stack must be [arg1, ..., argN, callee]
                    for arg_expr in delegate_args.iter() {
                        expr::compile_expr(ctx, arg_expr)?;
                    }
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(supercall_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arity), *line);
                    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                    // Instance created by parent; set __class_name and __private_fields (merged with parent)
                    let parent_private = ctx.class_private_fields.get(super_name).cloned().unwrap_or_default();
                    let merged_private: Vec<String> = parent_private
                        .iter()
                        .chain(private_field_names.iter())
                        .cloned()
                        .collect();
                    let mut defining_class: HashMap<String, String> = HashMap::new();
                    for f in &parent_private { defining_class.insert(f.clone(), super_name.clone()); }
                    for f in &private_field_names { defining_class.insert(f.clone(), name.clone()); }
                    emit_instance_private_metadata(ctx, *line, this_slot, name, &merged_private, &defining_class);
                    emit_instance_extends_table(ctx, *line, this_slot, name);
                    let parent_protected = ctx.class_protected_fields.get(super_name).cloned().unwrap_or_default();
                    let merged_protected: Vec<String> = parent_protected
                        .iter()
                        .chain(protected_field_names.iter())
                        .cloned()
                        .collect();
                    emit_instance_protected_metadata(ctx, *line, this_slot, &merged_protected);
                    emit_instance_class_reference(ctx, *line, this_slot, class_global_index);
                } else {
                    // No delegate_args - check for explicit super(args) in body
                    // The first statement MUST be super(args) when extending a class
                    let super_args: Vec<Arg> = if !constructor.body.is_empty() {
                        if let Stmt::Expr { expr: Expr::SuperCall { args, .. }, .. } = &constructor.body[0] {
                            // First statement is super(args) - extract args and skip it in body
                            body_skip_count = 1;
                            args.clone()
                        } else {
                            // First statement is NOT super() - error
                            return Err(LangError::ParseError {
                                message: format!("Class '{}' extends '{}' but does not call super(...) as the first statement in constructor", name, super_name),
                                line: constructor.line,
                            });
                        }
                    } else {
                        // Empty body - no super() call
                        return Err(LangError::ParseError {
                            message: format!("Class '{}' extends '{}' but does not call super(...)", name, super_name),
                            line: constructor.line,
                        });
                    };
                    
                    // Check for duplicate super() calls in the rest of the body
                    for stmt in constructor.body.iter().skip(1) {
                        if let Stmt::Expr { expr: Expr::SuperCall { .. }, line: stmt_line } = stmt {
                            return Err(LangError::ParseError {
                                message: "super() must be called exactly once and as the first statement in constructor".to_string(),
                                line: *stmt_line,
                            });
                        }
                    }
                    
                    let arity = super_args.len();
                    // Resolve parent constructor
                    let (super_callable_name, supercall_global_index) = {
                        let parent_constructor = format!("{}::new_{}", super_name, arity);
                        if let Some(&idx) = ctx.scope.globals.get(&parent_constructor) {
                            (parent_constructor, idx)
                        } else {
                            let idx = *ctx.scope.globals.get(super_name).ok_or_else(|| LangError::ParseError {
                                message: format!("Superclass '{}' not found in scope", super_name),
                                line: *line,
                            })?;
                            (super_name.clone(), idx)
                        }
                    };
                    ctx.chunk.global_names.insert(supercall_global_index, super_callable_name);
                    
                    // Compile super() arguments
                    for arg in super_args.iter() {
                        match arg {
                            Arg::Positional(arg_expr) => expr::compile_expr(ctx, arg_expr)?,
                            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                        }
                    }
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(supercall_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arity), *line);
                    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                    
                    // Emit metadata (merged with parent)
                    let parent_private = ctx.class_private_fields.get(super_name).cloned().unwrap_or_default();
                    let merged_private: Vec<String> = parent_private
                        .iter()
                        .chain(private_field_names.iter())
                        .cloned()
                        .collect();
                    let mut defining_class: HashMap<String, String> = HashMap::new();
                    for f in &parent_private { defining_class.insert(f.clone(), super_name.clone()); }
                    for f in &private_field_names { defining_class.insert(f.clone(), name.clone()); }
                    emit_instance_private_metadata(ctx, *line, this_slot, name, &merged_private, &defining_class);
                    emit_instance_extends_table(ctx, *line, this_slot, name);
                    let parent_protected = ctx.class_protected_fields.get(super_name).cloned().unwrap_or_default();
                    let merged_protected: Vec<String> = parent_protected
                        .iter()
                        .chain(protected_field_names.iter())
                        .cloned()
                        .collect();
                    emit_instance_protected_metadata(ctx, *line, this_slot, &merged_protected);
                    emit_instance_class_reference(ctx, *line, this_slot, class_global_index);
                }
            } else {
                // Создаем пустой объект-экземпляр
                ctx.chunk.write_with_line(OpCode::MakeObject(0), *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                let defining_class: HashMap<String, String> = private_field_names.iter().map(|f| (f.clone(), name.clone())).collect();
                emit_instance_private_metadata(ctx, *line, this_slot, name, &private_field_names, &defining_class);
                emit_instance_extends_table(ctx, *line, this_slot, name);
                emit_instance_protected_metadata(ctx, *line, this_slot, &protected_field_names);
                emit_instance_class_reference(ctx, *line, this_slot, class_global_index);
            }

            // Инициализируем поля значениями по умолчанию
            // Сначала private поля
            for field in private_fields.iter() {
                if let Some(ref default_expr) = field.default_value {
                    // Вычисляем значение по умолчанию во время компиляции
                    match crate::compiler::constant_fold::evaluate_constant_expr(default_expr) {
                        Ok(Some(constant_value)) => {
                            // Загружаем значение по умолчанию
                            let const_index = ctx.chunk.add_constant(constant_value);
                            ctx.chunk.write_with_line(OpCode::Constant(const_index), *line);
                            // Загружаем имя поля
                            let field_name_index = ctx.chunk.add_constant(Value::String(field.name.clone()));
                            ctx.chunk.write_with_line(OpCode::Constant(field_name_index), *line);
                            // Загружаем this
                            ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                            // Устанавливаем поле: SetArrayElement ожидает [value, field_name, object] на стеке
                            ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                            // SetArrayElement возвращает обновленный объект, сохраняем его обратно в this
                            ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                        }
                        Ok(None) => {
                            // Выражение не константное - пропускаем, пусть инициализируется в теле конструктора
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
            }
            // Затем protected поля
            for field in protected_fields.iter() {
                if let Some(ref default_expr) = field.default_value {
                    match crate::compiler::constant_fold::evaluate_constant_expr(default_expr) {
                        Ok(Some(constant_value)) => {
                            let const_index = ctx.chunk.add_constant(constant_value);
                            ctx.chunk.write_with_line(OpCode::Constant(const_index), *line);
                            let field_name_index = ctx.chunk.add_constant(Value::String(field.name.clone()));
                            ctx.chunk.write_with_line(OpCode::Constant(field_name_index), *line);
                            ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                            ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                            ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                        }
                        Ok(None) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            // Затем public поля
            for field in public_fields.iter() {
                if let Some(ref default_expr) = field.default_value {
                    // Вычисляем значение по умолчанию во время компиляции
                    match crate::compiler::constant_fold::evaluate_constant_expr(default_expr) {
                        Ok(Some(constant_value)) => {
                            // Загружаем значение по умолчанию
                            let const_index = ctx.chunk.add_constant(constant_value);
                            ctx.chunk.write_with_line(OpCode::Constant(const_index), *line);
                            // Загружаем имя поля
                            let field_name_index = ctx.chunk.add_constant(Value::String(field.name.clone()));
                            ctx.chunk.write_with_line(OpCode::Constant(field_name_index), *line);
                            // Загружаем this
                            ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                            // Устанавливаем поле: SetArrayElement ожидает [value, field_name, object] на стеке
                            ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                            // SetArrayElement возвращает обновленный объект, сохраняем его обратно в this
                            ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                        }
                        Ok(None) => {
                            // Выражение не константное - пропускаем, пусть инициализируется в теле конструктора
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
            }
            
            // Set context for class compilation (used by super.method() resolution)
            ctx.current_class = Some(name.clone());
            ctx.current_superclass = superclass.clone();
            ctx.in_constructor = true;
            
            // Ensure "this" is in scope when compiling body (otherwise resolve_local("this") would return None and we'd emit LoadLocal(0))
            match ctx.scope.resolve_local("this") {
                Some(slot) if slot == this_slot => {}
                other => {
                    return Err(LangError::ParseError {
                        message: format!(
                            "constructor '{}': 'this' must be in scope for body (resolve_local(\"this\") = {:?}, this_slot = {})",
                            constructor_name, other, this_slot
                        ),
                        line: constructor.line,
                    });
                }
            }
            // So that assign/this compilers always use the correct slot for "this" in constructor body.
            // Use arity explicitly so "this" is always at slot [param_count] (params at 0..arity-1, this at arity).
            ctx.constructor_this_slot = Some(constructor.params.len());
            
            // Компилируем тело конструктора (this доступен как локальная переменная)
            // Skip first statement(s) if they were already compiled (e.g., explicit super())
            for stmt in constructor.body.iter().skip(body_skip_count) {
                stmt::compile_stmt(ctx, stmt, true)?;
            }
            
            // Reset constructor context
            ctx.in_constructor = false;
            ctx.constructor_this_slot = None;
            
            // ВАЖНО: Добавляем методы класса в объект перед возвратом
            // Методы должны быть доступны через GetArrayElement при вызове методов
            debug_println!("[DEBUG compile_class] Добавляем {} методов в объект конструктора '{}'", methods.len(), constructor_name);
            for (idx, method) in methods.iter().enumerate() {
                // Используем сохраненный индекс метода из forward declaration
                if let Some(&method_function_index) = method_indices.get(&method.name) {
                    debug_println!("[DEBUG compile_class] Сохраняем метод #{}: '{}' в объект с индексом функции {} (полное имя: '{}::method_{}')", 
                        idx, method.name, method_function_index, name, method.name);
                    
                    // ВАЖНО: SetArrayElement ожидает порядок на стеке: [value, index/key, container]
                    // То есть: [function, method_name, object]
                    // Загружаем функцию метода (value)
                    let method_function_const = ctx.chunk.add_constant(Value::Function(method_function_index));
                    ctx.chunk.write_with_line(OpCode::Constant(method_function_const), *line);
                    // Загружаем имя метода как строку (key)
                    let method_name_const = ctx.chunk.add_constant(Value::String(method.name.clone()));
                    ctx.chunk.write_with_line(OpCode::Constant(method_name_const), *line);
                    // Загружаем this (container)
                    ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
                    // Устанавливаем метод в объект: SetArrayElement ожидает [function, method_name, object]
                    ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                    // SetArrayElement возвращает обновленный объект, сохраняем его обратно в this
                    ctx.chunk.write_with_line(OpCode::StoreLocal(this_slot), *line);
                } else {
                    debug_println!("[ERROR compile_class] Метод '{}' не найден в method_indices!", method.name);
                }
            }
            
            // В конце конструктора возвращаем this
            ctx.chunk.write_with_line(OpCode::LoadLocal(this_slot), *line);
            ctx.chunk.write_with_line(OpCode::Return, *line);
            
            // Логируем сгенерированный байткод конструктора
            let compiled_chunk = &ctx.chunk;
            debug_println!("[DEBUG compile_class] Сгенерированный байткод для конструктора '{}' ({} инструкций):", constructor_name, compiled_chunk.code.len());
            for (i, opcode) in compiled_chunk.code.iter().enumerate() {
                let line_num = compiled_chunk.get_line(i);
                let is_set_array = matches!(opcode, OpCode::SetArrayElement);
                let is_get_array = matches!(opcode, OpCode::GetArrayElement);
                let marker = if is_set_array { " <-- SetArrayElement" } else if is_get_array { " <-- GetArrayElement" } else { "" };
                debug_println!("[DEBUG compile_class]   IP {} (строка {}): {:?}{}", i, line_num, opcode, marker);
            }
            
            // Подсчитываем количество SetArrayElement в байткоде
            let set_array_count = compiled_chunk.code.iter().filter(|op| matches!(op, OpCode::SetArrayElement)).count();
            debug_println!("[DEBUG compile_class] Всего инструкций SetArrayElement в байткоде конструктора '{}': {}", constructor_name, set_array_count);
            
            // Обновляем функцию с скомпилированным телом
            ctx.functions[function_index].chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
            
            ctx.scope.end_scope();
            ctx.current_function = saved_function;
            ctx.scope.local_count = saved_local_count;
            
            // Сохраняем конструктор в глобальную переменную (аналогично обычным функциям)
            let constructor_global_index = *ctx.scope.globals.get(&constructor_name).unwrap();
            ctx.chunk.global_names.insert(constructor_global_index, constructor_name.clone());
            let constructor_constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            ctx.chunk.write_with_line(OpCode::Constant(constructor_constant_index), *line);
            ctx.chunk.write_with_line(OpCode::StoreGlobal(constructor_global_index), *line);
        }
        
        // Компилируем тела методов (функции уже объявлены в первом проходе, используем их индексы)
        for method in methods.iter() {
            let function_index = *method_indices.get(&method.name).ok_or_else(|| LangError::ParseError {
                message: format!("Method '{}' not found in method_indices", method.name),
                line: *line,
            })?;
            let method_function = &ctx.functions[function_index];
            let has_at_class = method.params.first().map(|p| p.name == "@class").unwrap_or(false);
            let user_params: Vec<_> = if has_at_class {
                method.params.iter().skip(1).collect()
            } else {
                method.params.iter().collect()
            };
            
            // Компилируем тело метода в chunk этой функции (уже объявленной)
            let saved_chunk = std::mem::replace(&mut *ctx.chunk, method_function.chunk.clone());
            let saved_function = ctx.current_function;
            let saved_local_count = ctx.scope.local_count;
            
            ctx.current_function = Some(function_index);
            ctx.scope.local_count = 0;
            ctx.scope.begin_scope();
            
            // First parameter - this; second - @class if method has it (injected by VM at call time)
            ctx.scope.declare_local("this");
            if has_at_class {
                ctx.scope.declare_local("@class");
            }
            for param in &user_params {
                ctx.scope.declare_local(&param.name);
            }
            
            // Set context for method compilation (used by super.method() resolution)
            ctx.current_class = Some(name.clone());
            ctx.current_superclass = superclass.clone();
            ctx.in_constructor = false;
            
            // Компилируем тело метода
            for stmt in &method.body {
                stmt::compile_stmt(ctx, stmt, true)?;
            }
            
            // Если нет return, добавляем return null
            ctx.chunk.write_with_line(OpCode::Return, *line);
            
            // Обновляем функцию с скомпилированным телом
            ctx.functions[function_index].chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
            
            ctx.scope.end_scope();
            ctx.current_function = saved_function;
            ctx.scope.local_count = saved_local_count;
        }
        
        // Reset class context
        ctx.current_class = None;
        ctx.current_superclass = None;
        
        // Сохраняем класс-объект в глобальной области видимости (слот зарезервирован в начале как class_global_index)
        let class_value = Value::Object(std::rc::Rc::new(std::cell::RefCell::new(class_metadata)));
        let class_index = ctx.chunk.add_constant(class_value);
        ctx.chunk.write_with_line(OpCode::Constant(class_index), *line);
        
        ctx.chunk.global_names.insert(class_global_index, name.clone());
        ctx.chunk.write_with_line(OpCode::StoreGlobal(class_global_index), *line);
        
        // Вычисляем и записываем переменные уровня класса в объект класса (стек: value, name, class -> SetArrayElement)
        for var in private_variables.iter().chain(protected_variables.iter()).chain(public_variables.iter()) {
            expr::compile_expr(ctx, &var.value)?;
            let name_const = ctx.chunk.add_constant(Value::String(var.name.clone()));
            ctx.chunk.write_with_line(OpCode::Constant(name_const), *line);
            ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
            ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
        }
        
        // Для extends_table: добавляем Column-дескрипторы в объект класса под ключом __col_<name>
        // (обход проверки приватности; run_create_all ищет __col_* для DDL)
        if extends_table {
            for field in private_fields.iter().chain(protected_fields.iter()).chain(public_fields.iter()) {
                if let Some(ref default_expr) = field.default_value {
                    expr::compile_expr(ctx, default_expr)?;
                    let col_key = format!("__col_{}", field.name);
                    let name_const = ctx.chunk.add_constant(Value::String(col_key));
                    ctx.chunk.write_with_line(OpCode::Constant(name_const), *line);
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                }
            }
        }
        
        // Store this class's private and protected field names for subclass constructors (merge with parent)
        ctx.class_private_fields.insert(name.clone(), private_field_names.clone());
        ctx.class_protected_fields.insert(name.clone(), protected_field_names.clone());
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Class statement".to_string(),
            line: stmt.line(),
        })
    }
}
