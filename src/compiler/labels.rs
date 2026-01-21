/// Управление метками и jump-инструкциями

use crate::bytecode::{Chunk, OpCode};
use crate::common::error::LangError;

pub struct LabelManager {
    pub label_counter: usize,
    pub labels: std::collections::HashMap<usize, usize>,
    pub pending_jumps: Vec<(usize, usize, bool)>, // (индекс_инструкции, label_id, is_conditional)
}

impl LabelManager {
    pub fn new() -> Self {
        Self {
            label_counter: 0,
            labels: std::collections::HashMap::new(),
            pending_jumps: Vec::new(),
        }
    }

    pub fn create_label(&mut self) -> usize {
        let label_id = self.label_counter;
        self.label_counter += 1;
        label_id
    }

    pub fn mark_label(&mut self, label_id: usize, instruction_index: usize) {
        self.labels.insert(label_id, instruction_index);
    }

    pub fn emit_jump(&mut self, chunk: &mut Chunk, current_line: usize, is_conditional: bool, label_id: usize) -> Result<usize, LangError> {
        let jump_index = chunk.code.len();
        let opcode = if is_conditional {
            OpCode::JumpIfFalseLabel(label_id)
        } else {
            OpCode::JumpLabel(label_id)
        };
        chunk.write_with_line(opcode, current_line);
        self.pending_jumps.push((jump_index, label_id, is_conditional));
        Ok(jump_index)
    }

    pub fn emit_loop(&mut self, chunk: &mut Chunk, current_line: usize, loop_label_id: usize) -> Result<(), LangError> {
        self.emit_jump(chunk, current_line, false, loop_label_id)?;
        Ok(())
    }

    /// Вычисляет размер инструкции в байтах для эталонного алгоритма апгрейда jump-инструкций
    pub fn instruction_size(opcode: &OpCode) -> usize {
        match opcode {
            // Jump инструкции с относительными смещениями
            OpCode::Jump8(_) | OpCode::JumpIfFalse8(_) => 2,  // 1 байт opcode + 1 байт смещение
            OpCode::Jump16(_) | OpCode::JumpIfFalse16(_) => 3, // 1 байт opcode + 2 байта смещение
            OpCode::Jump32(_) | OpCode::JumpIfFalse32(_) => 5, // 1 байт opcode + 4 байта смещение
            OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_) => 2, // Временно считаем как Jump8 до финализации
            
            // Инструкции с параметрами
            OpCode::Constant(_) => 2,  // 1 байт opcode + 1 байт индекс константы (usize может быть больше, но упрощаем)
            OpCode::LoadLocal(_) | OpCode::StoreLocal(_) => 2, // 1 байт opcode + 1 байт индекс
            OpCode::LoadGlobal(_) | OpCode::StoreGlobal(_) => 2, // 1 байт opcode + 1 байт индекс
            OpCode::Call(_) => 2, // 1 байт opcode + 1 байт количество аргументов
            OpCode::MakeArray(_) => 2, // 1 байт opcode + 1 байт размер
            OpCode::MakeTuple(_) => 2, // 1 байт opcode + 1 байт размер
            OpCode::MakeArrayDynamic => 1, // 1 байт opcode (размер на стеке) 1 байт opcode + 1 байт количество элементов
            OpCode::BeginTry(_) => 2, // 1 байт opcode + 1 байт индекс обработчика
            OpCode::Catch(Some(_)) => 2, // 1 байт opcode + 1 байт тип ошибки
            OpCode::Catch(None) => 1, // 1 байт opcode
            OpCode::Throw(Some(_)) => 2, // 1 байт opcode + 1 байт тип ошибки
            OpCode::Throw(None) => 1, // 1 байт opcode
            
            // Все остальные инструкции занимают 1 байт
            _ => 1,
        }
    }

    /// Вычисляет абсолютные адреса всех инструкций с учетом их размеров
    pub fn compute_instruction_addresses(&self, chunk: &Chunk) -> Vec<usize> {
        let mut addresses = Vec::with_capacity(chunk.code.len());
        let mut current_addr = 0;
        
        for opcode in &chunk.code {
            addresses.push(current_addr);
            current_addr += Self::instruction_size(opcode);
        }
        
        addresses
    }

    /// Апгрейдит jump-инструкции до минимально достаточного формата
    /// Возвращает true если были изменения
    pub fn upgrade_jump_instructions(&mut self, chunk: &mut Chunk) -> bool {
        let addresses = self.compute_instruction_addresses(chunk);
        let mut changed = false;
        
        // Обрабатываем все pending jumps (JumpLabel и JumpIfFalseLabel)
        for (jump_index, label_id, is_conditional) in self.pending_jumps.iter() {
            if *jump_index >= chunk.code.len() {
                continue;
            }
            
            // Получаем адрес целевой метки
            let dst_instruction_index = *self.labels.get(label_id).unwrap_or(jump_index);
            if dst_instruction_index >= addresses.len() {
                continue;
            }
            
            // VM использует индексы инструкций, а не байтовые адреса
            // IP инкрементируется на 1 после каждой инструкции
            // Поэтому смещение вычисляется как: offset = dst_index - (src_index + 1)
            let src_index = *jump_index;
            let dst_index = dst_instruction_index;
            
            // Вычисляем относительное смещение в индексах инструкций
            let offset = (dst_index as i64 - (src_index as i64 + 1)) as i32;
            
            // Определяем текущий размер jump-инструкции (для апгрейда)
            let current_opcode = &chunk.code[*jump_index];
            
            // Определяем минимально достаточный формат
            let new_opcode = if offset >= -128 && offset <= 127 {
                if *is_conditional {
                    OpCode::JumpIfFalse8(offset as i8)
                } else {
                    OpCode::Jump8(offset as i8)
                }
            } else if offset >= -32768 && offset <= 32767 {
                if *is_conditional {
                    OpCode::JumpIfFalse16(offset as i16)
                } else {
                    OpCode::Jump16(offset as i16)
                }
            } else {
                if *is_conditional {
                    OpCode::JumpIfFalse32(offset)
                } else {
                    OpCode::Jump32(offset)
                }
            };
            
            // Проверяем, нужно ли апгрейдить
            // НЕ заменяем JumpLabel здесь - это делает finalize_jumps
            if matches!(current_opcode, OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_)) {
                continue;
            }
            
            let new_size = Self::instruction_size(&new_opcode);
            let current_size = Self::instruction_size(current_opcode);
            if new_size != current_size {
                chunk.code[*jump_index] = new_opcode;
                changed = true;
            } else {
                // Если размер не изменился, но смещение могло измениться, обновляем
                match current_opcode {
                    OpCode::Jump8(_) | OpCode::Jump16(_) | OpCode::Jump32(_) |
                    OpCode::JumpIfFalse8(_) | OpCode::JumpIfFalse16(_) | OpCode::JumpIfFalse32(_) => {
                        chunk.code[*jump_index] = new_opcode;
                    }
                    _ => {}
                }
            }
        }
        
        changed
    }

    /// Итеративно стабилизирует layout до полной фиксации размеров
    pub fn stabilize_layout(&mut self, chunk: &mut Chunk, current_line: usize) -> Result<(), LangError> {
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;
        
        loop {
            let changed = self.upgrade_jump_instructions(chunk);
            iterations += 1;
            
            if !changed {
                break;
            }
            
            if iterations >= MAX_ITERATIONS {
                return Err(LangError::ParseError {
                    message: "Layout stabilization failed: too many iterations".to_string(),
                    line: current_line,
                });
            }
        }
        
        Ok(())
    }

    /// Финализирует jump-инструкции: заменяет все JumpLabel на финальные инструкции
    pub fn finalize_jumps(&mut self, chunk: &mut Chunk, current_line: usize) -> Result<(), LangError> {
        let addresses = self.compute_instruction_addresses(chunk);
        
        let mut jumps_to_finalize = self.pending_jumps.clone();
        
        // Также ищем все JumpLabel в коде, которые могут не быть в pending_jumps
        for (jump_index, opcode) in chunk.code.iter().enumerate() {
            match opcode {
                OpCode::JumpLabel(label_id) => {
                    if !jumps_to_finalize.iter().any(|(idx, _, _)| *idx == jump_index) {
                        jumps_to_finalize.push((jump_index, *label_id, false));
                    }
                }
                OpCode::JumpIfFalseLabel(label_id) => {
                    if !jumps_to_finalize.iter().any(|(idx, _, _)| *idx == jump_index) {
                        jumps_to_finalize.push((jump_index, *label_id, true));
                    }
                }
                _ => {}
            }
        }
        
        for (jump_index, label_id, is_conditional) in jumps_to_finalize.iter() {
            if *jump_index >= chunk.code.len() {
                continue;
            }
            
            let current_opcode = &chunk.code[*jump_index];
            
            // Пропускаем, если уже финализировано
            if !matches!(current_opcode, OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_)) {
                continue;
            }
            
            // Получаем адрес целевой метки
            let dst_instruction_index = *self.labels.get(label_id)
                .ok_or_else(|| LangError::ParseError {
                    message: format!("Label {} not found", label_id),
                    line: current_line,
                })?;
            
            let dst_instruction_index = if dst_instruction_index >= chunk.code.len() {
                if chunk.code.is_empty() {
                    return Err(LangError::ParseError {
                        message: format!("Label {} points to empty code", label_id),
                        line: current_line,
                    });
                }
                chunk.code.len() - 1
            } else {
                dst_instruction_index
            };
            
            if dst_instruction_index >= addresses.len() {
                return Err(LangError::ParseError {
                    message: format!("Label {} instruction index {} >= addresses len {} (code len: {})", 
                        label_id, dst_instruction_index, addresses.len(), chunk.code.len()),
                    line: current_line,
                });
            }
            
            let src_index = *jump_index;
            let dst_index = dst_instruction_index;
            let offset = (dst_index as i64 - (src_index as i64 + 1)) as i32;
            
            let final_opcode = if offset >= -128 && offset <= 127 {
                if *is_conditional {
                    OpCode::JumpIfFalse8(offset as i8)
                } else {
                    OpCode::Jump8(offset as i8)
                }
            } else if offset >= -32768 && offset <= 32767 {
                if *is_conditional {
                    OpCode::JumpIfFalse16(offset as i16)
                } else {
                    OpCode::Jump16(offset as i16)
                }
            } else {
                if *is_conditional {
                    OpCode::JumpIfFalse32(offset)
                } else {
                    OpCode::Jump32(offset)
                }
            };
            
            chunk.code[*jump_index] = final_opcode;
        }
        
        self.pending_jumps.clear();
        
        Ok(())
    }

    pub fn clear(&mut self) {
        self.labels.clear();
        self.label_counter = 0;
        self.pending_jumps.clear();
    }
}


