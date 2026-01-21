/// Работа с распаковкой паттернов (unpack patterns)

use crate::parser::ast::UnpackPattern;
use crate::bytecode::{Chunk, OpCode};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::scope::ScopeManager;
use crate::compiler::labels::LabelManager;

/// Подсчитывает количество фиксированных переменных (не wildcard, не variadic) в паттерне распаковки
/// Для вложенных паттернов считает только переменные текущего уровня (вложенный паттерн = 1 элемент)
pub fn count_unpack_variables(pattern: &[UnpackPattern]) -> usize {
    let mut count = 0;
    for pat in pattern {
        match pat {
            UnpackPattern::Variable(_) => count += 1,
            UnpackPattern::Wildcard => {}, // Wildcard не считается
            UnpackPattern::Variadic(_) | UnpackPattern::VariadicWildcard => {
                // Variadic не считается в фиксированных переменных
            }
            UnpackPattern::Nested(_) => count += 1, // Вложенный паттерн считается как один элемент
        }
    }
    count
}

/// Объявляет переменные из паттерна распаковки и возвращает их локальные индексы
pub fn declare_unpack_pattern_variables(
    pattern: &[UnpackPattern],
    scope: &mut ScopeManager,
    line: usize,
) -> Result<Vec<Option<usize>>, LangError> {
    let mut var_locals = Vec::new();
    for pat in pattern {
        match pat {
            UnpackPattern::Variable(name) => {
                let index = scope.declare_local(name);
                var_locals.push(Some(index));
            }
            UnpackPattern::Wildcard => {
                // Wildcard не создает переменную
                var_locals.push(None);
            }
            UnpackPattern::Variadic(name) => {
                // Variadic переменная создает переменную
                let index = scope.declare_local(name);
                var_locals.push(Some(index));
            }
            UnpackPattern::VariadicWildcard => {
                // Variadic wildcard не создает переменную
                var_locals.push(None);
            }
            UnpackPattern::Nested(nested) => {
                // Рекурсивно обрабатываем вложенные паттерны
                let nested_locals = declare_unpack_pattern_variables(nested, scope, line)?;
                var_locals.extend(nested_locals);
            }
        }
    }
    Ok(var_locals)
}

/// Компилирует код для распаковки значения в переменные
/// Предполагается, что значение находится на вершине стека
pub fn compile_unpack_pattern(
    pattern: &[UnpackPattern],
    var_locals: &[Option<usize>],
    expected_count: usize,
    chunk: &mut Chunk,
    scope: &mut ScopeManager,
    labels: &mut LabelManager,
    current_line: usize,
    line: usize,
) -> Result<(), LangError> {
    // Сохраняем элемент во временную переменную
    let temp_local = scope.declare_local("__unpack_temp");
    chunk.write_with_line(OpCode::StoreLocal(temp_local), line);
    
    // Проверяем минимальную длину элемента (M >= N_fixed)
    // Загружаем элемент
    chunk.write_with_line(OpCode::LoadLocal(temp_local), line);
    // Получаем длину
    chunk.write_with_line(OpCode::GetArrayLength, line);
    // Загружаем минимальную требуемую длину (N_fixed)
    let n_fixed_const = chunk.add_constant(Value::Number(expected_count as f64));
    chunk.write_with_line(OpCode::Constant(n_fixed_const), line);
    // Сравниваем: length < N_fixed?
    chunk.write_with_line(OpCode::Less, line);
    
    // Если length >= N_fixed, пропускаем ошибку
    let skip_error_label = labels.create_label();
    labels.emit_jump(chunk, current_line, true, skip_error_label)?;
    
    // Выбрасываем ошибку: длина меньше минимально требуемой
    let error_msg = format!("Unpack pattern requires at least {} elements, but got array with length less than {}", expected_count, expected_count);
    let error_msg_index = chunk.add_constant(Value::String(error_msg));
    chunk.write_with_line(OpCode::Constant(error_msg_index), line);
    chunk.write_with_line(OpCode::Throw(None), line);
    
    // Метка для пропуска ошибки
    labels.mark_label(skip_error_label, chunk.code.len());
    
    // Распаковываем значения
    // var_locals соответствует структуре pattern, итерируем их вместе
    let mut var_index = 0;
    compile_unpack_pattern_recursive(pattern, var_locals, &mut var_index, temp_local, chunk, scope, labels, current_line, line)?;
    
    Ok(())
}

/// Рекурсивно компилирует распаковку вложенных паттернов
fn compile_unpack_pattern_recursive(
    pattern: &[UnpackPattern],
    var_locals: &[Option<usize>],
    var_index: &mut usize,
    source_local: usize,
    chunk: &mut Chunk,
    scope: &mut ScopeManager,
    labels: &mut LabelManager,
    current_line: usize,
    line: usize,
) -> Result<(), LangError> {
    // Находим позицию variadic в паттерне (если есть)
    let variadic_pos = pattern.iter().position(|p| matches!(p, UnpackPattern::Variadic(_) | UnpackPattern::VariadicWildcard));
    
    // Обрабатываем фиксированные переменные до variadic
    let end_pos = variadic_pos.unwrap_or(pattern.len());
    for (i, pat) in pattern[..end_pos].iter().enumerate() {
        match pat {
            UnpackPattern::Variable(_) => {
                // Получаем значение по индексу
                chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                let index_const = chunk.add_constant(Value::Number(*var_index as f64));
                chunk.write_with_line(OpCode::Constant(index_const), line);
                chunk.write_with_line(OpCode::GetArrayElement, line);
                
                // Сохраняем в переменную
                if let Some(local_index) = var_locals.get(i).and_then(|&x| x) {
                    chunk.write_with_line(OpCode::StoreLocal(local_index), line);
                }
                *var_index += 1;
            }
            UnpackPattern::Wildcard => {
                // Получаем значение, но не сохраняем (просто удаляем со стека)
                chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                let index_const = chunk.add_constant(Value::Number(*var_index as f64));
                chunk.write_with_line(OpCode::Constant(index_const), line);
                chunk.write_with_line(OpCode::GetArrayElement, line);
                chunk.write_with_line(OpCode::Pop, line); // Удаляем значение
                *var_index += 1;
            }
            UnpackPattern::Variadic(_) | UnpackPattern::VariadicWildcard => {
                // Не должно быть здесь, так как мы обрабатываем только до variadic
                unreachable!("Variadic should be handled separately");
            }
            UnpackPattern::Nested(nested) => {
                // Для вложенной распаковки нужно получить элемент и рекурсивно распаковать
                chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                let index_const = chunk.add_constant(Value::Number(*var_index as f64));
                chunk.write_with_line(OpCode::Constant(index_const), line);
                chunk.write_with_line(OpCode::GetArrayElement, line);
                
                // Сохраняем вложенный элемент во временную переменную
                let nested_temp = scope.declare_local("__unpack_nested_temp");
                chunk.write_with_line(OpCode::StoreLocal(nested_temp), line);
                
                // Упрощенный подход: для вложенных паттернов создаем новые локальные переменные
                let nested_var_locals = declare_unpack_pattern_variables(nested, scope, line)?;
                let mut nested_var_index = 0;
                compile_unpack_pattern_recursive(nested, &nested_var_locals, &mut nested_var_index, nested_temp, chunk, scope, labels, current_line, line)?;
                
                *var_index += 1;
            }
        }
    }
    
    // Обрабатываем variadic (если есть)
    if let Some(pos) = variadic_pos {
        let variadic_pattern = &pattern[pos];
        match variadic_pattern {
            UnpackPattern::Variadic(_) => {
                // Создаем массив из оставшихся элементов
                // 1. Вычисляем count = length - var_index
                chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                chunk.write_with_line(OpCode::GetArrayLength, line);
                let var_index_const = chunk.add_constant(Value::Number(*var_index as f64));
                chunk.write_with_line(OpCode::Constant(var_index_const), line);
                chunk.write_with_line(OpCode::Sub, line);
                // Теперь на стеке: count
                
                // Сохраняем count во временную переменную
                let count_temp = scope.declare_local("__variadic_count");
                chunk.write_with_line(OpCode::StoreLocal(count_temp), line);
                
                // 2. Загружаем элементы от length-1 до var_index (в обратном порядке индексов)
                // чтобы они были на стеке в правильном порядке для MakeArrayDynamic
                let loop_start_label = labels.create_label();
                let loop_end_label = labels.create_label();
                let loop_index_temp = scope.declare_local("__variadic_loop_idx");
                
                // Начинаем с length-1
                chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                chunk.write_with_line(OpCode::GetArrayLength, line);
                let one_const = chunk.add_constant(Value::Number(1.0));
                chunk.write_with_line(OpCode::Constant(one_const), line);
                chunk.write_with_line(OpCode::Sub, line);
                chunk.write_with_line(OpCode::StoreLocal(loop_index_temp), line);
                
                // Начало цикла
                labels.mark_label(loop_start_label, chunk.code.len());
                
                // Проверяем условие: loop_index >= var_index
                chunk.write_with_line(OpCode::LoadLocal(loop_index_temp), line);
                chunk.write_with_line(OpCode::Constant(var_index_const), line);
                chunk.write_with_line(OpCode::GreaterEqual, line);
                
                // Если условие false, выходим из цикла
                labels.emit_jump(chunk, current_line, true, loop_end_label)?;
                
                // Загружаем элемент по индексу loop_index
                chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                chunk.write_with_line(OpCode::LoadLocal(loop_index_temp), line);
                chunk.write_with_line(OpCode::GetArrayElement, line);
                
                // Декрементируем loop_index
                chunk.write_with_line(OpCode::LoadLocal(loop_index_temp), line);
                chunk.write_with_line(OpCode::Constant(one_const), line);
                chunk.write_with_line(OpCode::Sub, line);
                chunk.write_with_line(OpCode::StoreLocal(loop_index_temp), line);
                
                // Переход к началу цикла
                labels.emit_loop(chunk, current_line, loop_start_label)?;
                
                // Конец цикла
                labels.mark_label(loop_end_label, chunk.code.len());
                
                // 3. Создаем массив динамически: загружаем count и используем MakeArrayDynamic
                chunk.write_with_line(OpCode::LoadLocal(count_temp), line);
                chunk.write_with_line(OpCode::MakeArrayDynamic, line);
                
                // 4. Сохраняем в variadic переменную
                if let Some(local_index) = var_locals.get(pos).and_then(|&x| x) {
                    chunk.write_with_line(OpCode::StoreLocal(local_index), line);
                }
            }
            UnpackPattern::VariadicWildcard => {
                // Variadic wildcard: пропускаем оставшиеся элементы
                // Ничего не делаем, элементы уже пропущены
            }
            _ => {}
        }
    }
    
    Ok(())
}






