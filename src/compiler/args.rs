/// Разрешение аргументов функций: именованные -> позиционные, применение значений по умолчанию

use crate::parser::ast::Arg;
use crate::parser::ast::Expr;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::natives;

/// Разрешает аргументы функции: именованные -> позиционные, применяет значения по умолчанию
pub fn resolve_function_args(
    function_name: &str,
    args: &[Arg],
    function_info: Option<(usize, &crate::bytecode::Function)>,
    line: usize,
    file: Option<&str>,
) -> Result<Vec<Arg>, LangError> {
    let file_owned = file.map(String::from);
    // Если это встроенная функция, проверяем, поддерживает ли она именованные аргументы
    if function_info.is_none() {
        // Проверяем, есть ли именованные аргументы
        let has_named = args.iter().any(|a| matches!(a, Arg::Named { .. } | Arg::UnpackObject(_)));
        
        if has_named {
            // Проверяем, поддерживает ли эта нативная функция именованные аргументы
            if let Some(param_names) = natives::get_native_function_params(function_name) {
                // Нативная функция поддерживает именованные аргументы
                // Разрешаем их аналогично пользовательским функциям
                let mut resolved = vec![None; param_names.len()];
                let mut positional_count = 0;
                
                // Для методов объектов (например, nn_train), первый параметр - это объект,
                // который не передается в args метода, поэтому пропускаем позицию 0
                let start_position = if (function_name == "nn_train" || function_name == "nn_train_sh") && !param_names.is_empty() && param_names[0] == "nn" {
                    1  // Пропускаем первый параметр "nn" (объект метода)
                } else {
                    0
                };
                
                // Обрабатываем аргументы
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            let target_position = start_position + positional_count;
                            if target_position >= param_names.len() {
                                return Err(LangError::ParseError {
                                    message: format!(
                                        "Function '{}' takes at most {} arguments but {} positional arguments were provided",
                                        function_name, param_names.len() - start_position, positional_count + 1
                                    ),
                                    line,
                                    file: file_owned.clone(),
                                });
                            }
                            resolved[target_position] = Some(Arg::Positional(expr.clone()));
                            positional_count += 1;
                        }
                        Arg::UnpackObject(expr) => {
                            let target_position = start_position + positional_count;
                            if target_position >= param_names.len() {
                                return Err(LangError::ParseError {
                                    message: format!(
                                        "Function '{}' takes at most {} arguments but {} arguments were provided",
                                        function_name, param_names.len() - start_position, positional_count + 1
                                    ),
                                    line,
                                    file: file_owned.clone(),
                                });
                            }
                            resolved[target_position] = Some(Arg::Positional(expr.clone()));
                            positional_count += 1;
                        }
                        Arg::Named { name, value } => {
                            // Находим индекс параметра по имени
                            if let Some(param_index) = param_names.iter().position(|n| n == name) {
                                if resolved[param_index].is_some() {
                                    return Err(LangError::ParseError {
                                        message: format!(
                                            "Function '{}' got multiple values for argument '{}'",
                                            function_name, name
                                        ),
                                        line,
                                        file: file_owned.clone(),
                                    });
                                }
                                resolved[param_index] = Some(Arg::Named {
                                    name: name.clone(),
                                    value: value.clone(),
                                });
                            } else {
                                return Err(LangError::ParseError {
                                    message: format!(
                                        "Function '{}' got an unexpected keyword argument '{}'",
                                        function_name, name
                                    ),
                                    line,
                                    file: file_owned.clone(),
                                });
                            }
                        }
                    }
                }
                
                // Собираем итоговый список аргументов в правильном порядке.
                // Для нативных функций с большим числом опциональных параметров (например Field)
                // передаём Null для непереданных параметров, чтобы натива получала args[i] = param i.
                let mut final_args = Vec::new();
                for i in 0..param_names.len() {
                    match resolved[i].take() {
                        Some(arg) => match arg {
                            Arg::Positional(expr) => final_args.push(Arg::Positional(expr)),
                            Arg::Named { value, .. } => final_args.push(Arg::Positional(value)),
                            Arg::UnpackObject(_) => unreachable!(),
                        },
                        None => {
                            final_args.push(Arg::Positional(Expr::Literal {
                                value: Value::Null,
                                line,
                            }));
                        }
                    }
                }
                return Ok(final_args);
            } else {
                // Нативная функция не поддерживает именованные аргументы (например, print, min, max)
                return Err(LangError::ParseError {
                    message: format!(
                        "Named arguments are not supported for built-in function '{}'",
                        function_name
                    ),
                    line,
                    file: file_owned.clone(),
                });
            }
        } else {
            // Нет именованных аргументов, просто возвращаем позиционные
            return Ok(args.iter().map(|a| match a {
                Arg::Positional(e) => Arg::Positional(e.clone()),
                Arg::Named { .. } => unreachable!(),
                Arg::UnpackObject(e) => Arg::Positional(e.clone()),
            }).collect());
        }
    }
    
    let (_, function) = function_info.unwrap();
    let param_names = &function.param_names;
    let default_values = &function.default_values;
    
    // Единственный аргумент **obj — распаковка в kwargs; ключи объекта проверяются в runtime.
    if args.len() == 1 {
        if let Arg::UnpackObject(expr) = &args[0] {
            return Ok(vec![Arg::UnpackObject(expr.clone())]);
        }
    }
    
    // Создаем массив для разрешенных аргументов
    let mut resolved = vec![None; param_names.len()];
    let mut positional_count = 0;
    
    // Обрабатываем аргументы
    for arg in args {
        match arg {
            Arg::Positional(expr) => {
                if positional_count >= param_names.len() {
                    return Err(LangError::ParseError {
                        message: format!(
                            "Function '{}' takes {} arguments but {} positional arguments were provided",
                            function_name, param_names.len(), positional_count + 1
                        ),
                        line,
                        file: file_owned.clone(),
                    });
                }
                resolved[positional_count] = Some(Arg::Positional(expr.clone()));
                positional_count += 1;
            }
            Arg::UnpackObject(expr) => {
                if positional_count >= param_names.len() {
                    return Err(LangError::ParseError {
                        message: format!(
                            "Function '{}' takes {} arguments but {} arguments were provided",
                            function_name, param_names.len(), positional_count + 1
                        ),
                        line,
                        file: file_owned.clone(),
                    });
                }
                resolved[positional_count] = Some(Arg::Positional(expr.clone()));
                positional_count += 1;
            }
            Arg::Named { name, value } => {
                // Находим индекс параметра по имени
                if let Some(param_index) = param_names.iter().position(|n| n == name) {
                    if resolved[param_index].is_some() {
                        return Err(LangError::ParseError {
                            message: format!(
                                "Function '{}' got multiple values for argument '{}'",
                                function_name, name
                            ),
                            line,
                            file: file_owned.clone(),
                        });
                    }
                    resolved[param_index] = Some(Arg::Named {
                        name: name.clone(),
                        value: value.clone(),
                    });
                } else {
                    return Err(LangError::ParseError {
                        message: format!(
                            "Function '{}' got an unexpected keyword argument '{}'",
                            function_name, name
                        ),
                        line,
                        file: file_owned.clone(),
                    });
                }
            }
        }
    }
    
    // Применяем значения по умолчанию для незаполненных параметров
    let mut final_args = Vec::new();
    for (i, param_name) in param_names.iter().enumerate() {
        if let Some(arg) = resolved[i].take() {
            match arg {
                Arg::Positional(expr) => {
                    final_args.push(Arg::Positional(expr));
                }
                Arg::Named { value, .. } => {
                    final_args.push(Arg::Positional(value));
                }
                Arg::UnpackObject(_) => unreachable!(),
            }
        } else {
            // Аргумент не был предоставлен - используем значение по умолчанию
            if let Some(default_value) = default_values.get(i).and_then(|v| v.as_ref()) {
                // Значение по умолчанию было вычислено во время компиляции
                use crate::parser::ast::Expr;
                final_args.push(Arg::Positional(Expr::Literal {
                    value: default_value.clone(),
                    line,
                }));
            } else {
                // Обязательный параметр не был предоставлен
                return Err(LangError::ParseError {
                    message: format!(
                        "Function '{}' missing required argument '{}'",
                        function_name, param_name
                    ),
                    line,
                    file: file_owned.clone(),
                });
            }
        }
    }
    
    Ok(final_args)
}






