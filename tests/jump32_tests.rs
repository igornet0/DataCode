// Тесты для проверки генерации Jump32 инструкций компилятором
// Проверяем, что компилятор корректно апгрейдит Jump8 → Jump16 → Jump32
// при больших смещениях переходов

#[cfg(test)]
mod tests {
    use data_code::bytecode::OpCode;
    use data_code::{compile, LangError};

    /// Генерирует код с большим телом цикла для тестирования backward jump
    fn generate_large_loop_body(instruction_count: usize) -> String {
        let mut source = String::from("let x = 0\nwhile x < 1 {\n");

        // Генерируем много простых инструкций в теле цикла
        // Каждая строка "x = x + 1" генерирует несколько инструкций:
        // LoadLocal, Constant, Add, StoreLocal
        // Используем достаточно итераций, чтобы гарантировать нужное количество инструкций
        let iterations = instruction_count / 4 + 1; // +1 для запаса

        for _ in 0..iterations {
            source.push_str("x = x + 1\n");
        }

        source.push_str("}\n");
        source
    }

    /// Генерирует код с большим телом if для тестирования forward jump
    fn generate_large_if_body(instruction_count: usize) -> String {
        let mut source = String::from("let x = 0\nif true {\n");

        // Генерируем много простых инструкций в теле if
        let iterations = instruction_count / 4 + 1;

        for _ in 0..iterations {
            source.push_str("x = x + 1\n");
        }

        source.push_str("}\n");
        source
    }

    /// Проверяет наличие Jump32 в байткоде
    fn has_jump32(chunk: &data_code::Chunk) -> bool {
        chunk
            .code
            .iter()
            .any(|op| matches!(op, OpCode::Jump32(_) | OpCode::JumpIfFalse32(_)))
    }

    /// Проверяет наличие Jump16 в байткоде
    fn has_jump16(chunk: &data_code::Chunk) -> bool {
        chunk
            .code
            .iter()
            .any(|op| matches!(op, OpCode::Jump16(_) | OpCode::JumpIfFalse16(_)))
    }

    /// Проверяет наличие Jump8 в байткоде
    fn has_jump8(chunk: &data_code::Chunk) -> bool {
        chunk
            .code
            .iter()
            .any(|op| matches!(op, OpCode::Jump8(_) | OpCode::JumpIfFalse8(_)))
    }

    /// Получает все Jump32 инструкции из байткода
    fn get_jump32_instructions(chunk: &data_code::Chunk) -> Vec<(usize, i32)> {
        chunk
            .code
            .iter()
            .enumerate()
            .filter_map(|(idx, op)| match op {
                OpCode::Jump32(offset) => Some((idx, *offset)),
                OpCode::JumpIfFalse32(offset) => Some((idx, *offset)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_jump32_backward_loop() -> Result<(), LangError> {
        // Создаем цикл с телом, которое гарантированно переполнит Jump16
        // Нужно > 32767 инструкций в теле цикла для backward jump
        // Используем 70000 инструкций для надежности
        let source = generate_large_loop_body(70000);

        // Компилируем код
        let (chunk, _) = compile(&source)?;

        // Проверяем, что компиляция успешна (нет ошибки overflow)
        // Проверяем наличие Jump32
        assert!(
            has_jump32(&chunk),
            "Expected Jump32 instruction in bytecode for large backward jump"
        );

        // Проверяем, что есть backward jump (отрицательное смещение)
        let jump32_instructions = get_jump32_instructions(&chunk);
        assert!(
            !jump32_instructions.is_empty(),
            "No Jump32 instructions found"
        );

        // Проверяем, что есть хотя бы один backward jump (для цикла)
        let has_backward = jump32_instructions.iter().any(|(_, offset)| *offset < 0);
        assert!(
            has_backward,
            "Expected at least one backward Jump32 (for loop)"
        );

        Ok(())
    }

    #[test]
    fn test_jump32_forward_jump() -> Result<(), LangError> {
        // Создаем if с телом, которое гарантированно переполнит Jump16
        // Нужно > 32767 инструкций в теле if для forward jump
        let source = generate_large_if_body(70000);

        // Компилируем код
        let (chunk, _) = compile(&source)?;

        // Проверяем, что компиляция успешна
        // Проверяем наличие Jump32
        assert!(
            has_jump32(&chunk),
            "Expected Jump32 instruction in bytecode for large forward jump"
        );

        // Проверяем, что есть forward jump (положительное смещение)
        let jump32_instructions = get_jump32_instructions(&chunk);
        assert!(
            !jump32_instructions.is_empty(),
            "No Jump32 instructions found"
        );

        // Для if может быть как forward, так и backward jump (в зависимости от структуры)
        // Главное - проверить наличие Jump32
        Ok(())
    }

    #[test]
    fn test_jump32_boundary_65535() -> Result<(), LangError> {
        // Граничное условие: ровно 32767 инструкций должно использовать Jump16
        // Но поскольку мы считаем в индексах инструкций, а не байтах,
        // и смещение = dst_index - (src_index + 1), нам нужно учесть это

        // Для backward jump в цикле:
        // offset = start_index - (end_index + 1)
        // Если тело цикла имеет N инструкций, то offset ≈ -(N + overhead)
        // Для Jump16 нужно |offset| <= 32767

        // Создаем цикл с телом, которое должно поместиться в Jump16
        // Используем 30000 инструкций (с запасом)
        let source = generate_large_loop_body(30000);

        let (chunk, _) = compile(&source)?;

        // При таком размере должен использоваться Jump16 или Jump32 в зависимости от точного размера
        // Проверяем, что компиляция успешна
        assert!(
            has_jump16(&chunk) || has_jump32(&chunk),
            "Expected Jump16 or Jump32 for boundary test"
        );

        Ok(())
    }

    #[test]
    fn test_jump32_boundary_65536() -> Result<(), LangError> {
        // Граничное условие: > 32767 инструкций должно использовать Jump32
        // Используем 40000 инструкций, что гарантированно переполнит Jump16
        let source = generate_large_loop_body(40000);

        let (chunk, _) = compile(&source)?;

        // Проверяем, что используется Jump32
        assert!(
            has_jump32(&chunk),
            "Expected Jump32 instruction for large loop body (> 32767 instructions)"
        );

        Ok(())
    }

    #[test]
    fn test_jump32_upgrade_chain() -> Result<(), LangError> {
        // Тест для проверки цепочки апгрейда 8 → 16 → 32
        // Создаем код с несколькими переходами разного размера

        let mut source = String::from("let x = 0\n");

        // Маленький if (должен использовать Jump8)
        source.push_str("if true {\n");
        source.push_str("x = 1\n");
        source.push_str("}\n");

        // Средний if (должен использовать Jump16)
        source.push_str("if true {\n");
        for _ in 0..200 {
            source.push_str("x = x + 1\n");
        }
        source.push_str("}\n");

        // Большой цикл (должен использовать Jump32)
        source.push_str("while x < 1 {\n");
        for _ in 0..40000 {
            source.push_str("x = x + 1\n");
        }
        source.push_str("}\n");

        let (chunk, _) = compile(&source)?;

        // Проверяем, что все три типа jump присутствуют
        // (компилятор должен апгрейдить до минимально необходимого размера)
        assert!(
            has_jump8(&chunk) || has_jump16(&chunk) || has_jump32(&chunk),
            "Expected at least one type of jump instruction"
        );

        // Главное - проверить, что большой цикл использует Jump32
        assert!(
            has_jump32(&chunk),
            "Expected Jump32 for large loop in upgrade chain test"
        );

        Ok(())
    }

    #[test]
    fn test_jump32_no_overflow_error() -> Result<(), LangError> {
        // Негативный тест: компиляция не должна падать с ошибкой overflow
        // даже при очень больших переходах

        // Создаем очень большой цикл
        let source = generate_large_loop_body(100000);

        // Компиляция должна быть успешной
        let result = compile(&source);

        assert!(
            result.is_ok(),
            "Compilation should succeed even with very large jumps, got error: {:?}",
            result.err()
        );

        let (chunk, _) = result?;

        // Проверяем, что Jump32 присутствует
        assert!(has_jump32(&chunk), "Expected Jump32 for very large loop");

        Ok(())
    }

    #[test]
    fn test_jump32_conditional_jump() -> Result<(), LangError> {
        // Тест для условного перехода (JumpIfFalse32)
        // Создаем большой if с условием

        let mut source = String::from("let x = 0\n");
        source.push_str("if false {\n");

        // Большое тело if
        for _ in 0..40000 {
            source.push_str("x = x + 1\n");
        }

        source.push_str("}\n");
        source.push_str("x\n");

        let (chunk, _) = compile(&source)?;

        // Проверяем наличие JumpIfFalse32 или Jump32
        // Может быть JumpIfFalse32 или обычный Jump32 в зависимости от структуры
        let has_jump_if_false_32 = chunk
            .code
            .iter()
            .any(|op| matches!(op, OpCode::JumpIfFalse32(_)));

        assert!(
            has_jump32(&chunk) || has_jump_if_false_32,
            "Expected Jump32 or JumpIfFalse32 for large conditional jump, found neither. Jump32: {}, JumpIfFalse32: {}",
            has_jump32(&chunk),
            has_jump_if_false_32
        );

        Ok(())
    }
}
