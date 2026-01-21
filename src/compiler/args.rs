/// Разрешение аргументов функций: именованные -> позиционные, применение значений по умолчанию

use crate::parser::ast::Arg;
use crate::common::error::LangError;
use crate::compiler::natives;

/// Разрешает аргументы функции: именованные -> позиционные, применяет значения по умолчанию
pub fn resolve_function_args(
    function_name: &str,
    args: &[Arg],
    function_info: Option<(usize, &crate::bytecode::Function)>,
    line: usize,
) -> Result<Vec<Arg>, LangError> {
    // Если это встроенная функция, проверяем, поддерживает ли она именованные аргументы
    if function_info.is_none() {
        // Проверяем, есть ли именованные аргументы
        let has_named = args.iter().any(|a| matches!(a, Arg::Named { .. }));
        
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
                                });
                            }
                        }
                    }
                }
                
                // Собираем итоговый список аргументов в правильном порядке
                // Включаем только предоставленные параметры (нативные функции сами обрабатывают опциональные)
                let mut final_args = Vec::new();
                for i in 0..param_names.len() {
                    if let Some(arg) = resolved[i].take() {
                        match arg {
                            Arg::Positional(expr) => final_args.push(Arg::Positional(expr)),
                            Arg::Named { value, .. } => final_args.push(Arg::Positional(value)),
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
                });
            }
        } else {
            // Нет именованных аргументов, просто возвращаем позиционные
            return Ok(args.iter().map(|a| match a {
                Arg::Positional(e) => Arg::Positional(e.clone()),
                Arg::Named { .. } => unreachable!(),
            }).collect());
        }
    }
    
    let (_, function) = function_info.unwrap();
    let param_names = &function.param_names;
    let default_values = &function.default_values;
    
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
                    });
                }
            }
        }
    }
    
    // Применяем значения по умолчанию для незаполненных параметров
    let mut final_args = Vec::new();
    for (i, param_name) in param_names.iter().enumerate() {
        if let Some(arg) = resolved[i].take() {
            // Аргумент был предоставлен
            match arg {
                Arg::Positional(expr) => {
                    final_args.push(Arg::Positional(expr));
                }
                Arg::Named { value, .. } => {
                    final_args.push(Arg::Positional(value));
                }
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
                });
            }
        }
    }
    
    Ok(final_args)
}






