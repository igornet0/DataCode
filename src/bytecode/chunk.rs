// Chunk - контейнер байт-кода и констант

use super::opcode::OpCode;
use crate::common::value::Value;

/// Информация об обработчике исключений для передачи из компилятора в VM
#[derive(Debug, Clone)]
pub struct ExceptionHandlerInfo {
    pub catch_ips: Vec<usize>,           // IP начала каждого catch блока
    pub error_types: Vec<Option<usize>>, // Типы ошибок для каждого catch (None для catch всех)
    pub error_var_slots: Vec<Option<usize>>, // Слоты для переменных ошибок
    pub else_ip: Option<usize>,          // IP начала else блока
    pub stack_height: usize,             // Высота стека при входе в try
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<OpCode>,
    pub constants: Vec<Value>,
    pub lines: Vec<usize>, // Номер строки для каждой инструкции (для отладки и ошибок)
    pub exception_handlers: Vec<ExceptionHandlerInfo>, // Обработчики исключений
    pub error_type_table: Vec<String>, // Таблица типов ошибок для текущей функции
    pub global_names: std::collections::HashMap<usize, String>, // Маппинг индексов глобальных переменных на их имена
    pub explicit_global_names: std::collections::HashMap<usize, String>, // Маппинг индексов переменных, явно объявленных с ключевым словом 'global'
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(256), // Предварительное резервирование памяти
            constants: Vec::with_capacity(64),
            lines: Vec::with_capacity(256),
            exception_handlers: Vec::new(),
            error_type_table: Vec::new(),
            global_names: std::collections::HashMap::new(),
            explicit_global_names: std::collections::HashMap::new(),
        }
    }

    pub fn write(&mut self, opcode: OpCode) {
        self.write_with_line(opcode, 0); // По умолчанию строка 0
    }

    pub fn write_with_line(&mut self, opcode: OpCode, line: usize) {
        self.code.push(opcode);
        self.lines.push(line);
    }

    pub fn get_line(&self, ip: usize) -> usize {
        if ip < self.lines.len() {
            self.lines[ip]
        } else {
            0
        }
    }

    pub fn add_constant(&mut self, value: Value) -> usize {
        // Оптимизация: проверяем, есть ли уже такая константа
        if let Some(index) = self.constants.iter().position(|v| v == &value) {
            return index;
        }
        self.constants.push(value);
        self.constants.len() - 1
    }

    /// Debug mode: дамп байт-кода
    pub fn disassemble(&self, name: &str) -> String {
        let mut result = format!("== {} ==\n", name);
        let mut offset = 0;
        while offset < self.code.len() {
            offset = self.disassemble_instruction(offset, &mut result);
        }
        result
    }

    fn disassemble_instruction(&self, offset: usize, output: &mut String) -> usize {
        let line = if offset < self.lines.len() {
            self.lines[offset]
        } else {
            0
        };
        
        output.push_str(&format!("{:04} ", offset));
        if offset > 0 && line == self.lines[offset - 1] {
            output.push_str("   | ");
        } else {
            output.push_str(&format!("{:4} ", line));
        }

        let instruction = &self.code[offset];
        match instruction {
            OpCode::Constant(index) => {
                let value = &self.constants[*index];
                output.push_str(&format!("CONSTANT {:4} '{}'\n", index, value.to_string()));
                offset + 1
            }
            OpCode::LoadLocal(index) => {
                output.push_str(&format!("LOAD_LOCAL {}\n", index));
                offset + 1
            }
            OpCode::StoreLocal(index) => {
                output.push_str(&format!("STORE_LOCAL {}\n", index));
                offset + 1
            }
            OpCode::LoadGlobal(index) => {
                output.push_str(&format!("LOAD_GLOBAL {}\n", index));
                offset + 1
            }
            OpCode::StoreGlobal(index) => {
                output.push_str(&format!("STORE_GLOBAL {}\n", index));
                offset + 1
            }
            OpCode::Add => {
                output.push_str("ADD\n");
                offset + 1
            }
            OpCode::Sub => {
                output.push_str("SUB\n");
                offset + 1
            }
            OpCode::Mul => {
                output.push_str("MUL\n");
                offset + 1
            }
            OpCode::Div => {
                output.push_str("DIV\n");
                offset + 1
            }
            OpCode::IntDiv => {
                output.push_str("INT_DIV\n");
                offset + 1
            }
            OpCode::Mod => {
                output.push_str("MOD\n");
                offset + 1
            }
            OpCode::Pow => {
                output.push_str("POW\n");
                offset + 1
            }
            OpCode::Negate => {
                output.push_str("NEGATE\n");
                offset + 1
            }
            OpCode::Not => {
                output.push_str("NOT\n");
                offset + 1
            }
            OpCode::Or => {
                output.push_str("OR\n");
                offset + 1
            }
            OpCode::And => {
                output.push_str("AND\n");
                offset + 1
            }
            OpCode::Equal => {
                output.push_str("EQUAL\n");
                offset + 1
            }
            OpCode::NotEqual => {
                output.push_str("NOT_EQUAL\n");
                offset + 1
            }
            OpCode::Greater => {
                output.push_str("GREATER\n");
                offset + 1
            }
            OpCode::Less => {
                output.push_str("LESS\n");
                offset + 1
            }
            OpCode::GreaterEqual => {
                output.push_str("GREATER_EQUAL\n");
                offset + 1
            }
            OpCode::LessEqual => {
                output.push_str("LESS_EQUAL\n");
                offset + 1
            }
            OpCode::In => {
                output.push_str("IN\n");
                offset + 1
            }
            OpCode::Jump8(rel_offset) => {
                output.push_str(&format!("JUMP8 {:+.4}\n", *rel_offset as i32));
                offset + 2  // 1 байт opcode + 1 байт смещение
            }
            OpCode::Jump16(rel_offset) => {
                output.push_str(&format!("JUMP16 {:+.6}\n", *rel_offset as i32));
                offset + 3  // 1 байт opcode + 2 байта смещение
            }
            OpCode::Jump32(rel_offset) => {
                output.push_str(&format!("JUMP32 {:+.10}\n", *rel_offset));
                offset + 5  // 1 байт opcode + 4 байта смещение
            }
            OpCode::JumpIfFalse8(rel_offset) => {
                output.push_str(&format!("JUMP_IF_FALSE8 {:+.4}\n", *rel_offset as i32));
                offset + 2  // 1 байт opcode + 1 байт смещение
            }
            OpCode::JumpIfFalse16(rel_offset) => {
                output.push_str(&format!("JUMP_IF_FALSE16 {:+.6}\n", *rel_offset as i32));
                offset + 3  // 1 байт opcode + 2 байта смещение
            }
            OpCode::JumpIfFalse32(rel_offset) => {
                output.push_str(&format!("JUMP_IF_FALSE32 {:+.10}\n", *rel_offset));
                offset + 5  // 1 байт opcode + 4 байта смещение
            }
            OpCode::JumpLabel(label_id) => {
                output.push_str(&format!("JUMP_LABEL {}\n", label_id));
                offset + 1
            }
            OpCode::JumpIfFalseLabel(label_id) => {
                output.push_str(&format!("JUMP_IF_FALSE_LABEL {}\n", label_id));
                offset + 1
            }
            OpCode::Call(arity) => {
                output.push_str(&format!("CALL {}\n", arity));
                offset + 1
            }
            OpCode::Return => {
                output.push_str("RETURN\n");
                offset + 1
            }
            OpCode::MakeTuple(count) => {
                output.push_str(&format!("MAKE_TUPLE {}\n", count));
                offset + 1
            }
            OpCode::MakeArray(count) => {
                output.push_str(&format!("MAKE_ARRAY {}\n", count));
                offset + 1
            }
            OpCode::MakeArrayDynamic => {
                output.push_str("MAKE_ARRAY_DYNAMIC\n");
                offset + 1
            }
            OpCode::GetArrayLength => {
                output.push_str("GET_ARRAY_LENGTH\n");
                offset + 1
            }
            OpCode::GetArrayElement => {
                output.push_str("GET_ARRAY_ELEMENT\n");
                offset + 1
            }
            OpCode::Clone => {
                output.push_str("CLONE\n");
                offset + 1
            }
            OpCode::Pop => {
                output.push_str("POP\n");
                offset + 1
            }
            OpCode::BeginTry(handler_index) => {
                output.push_str(&format!("BEGIN_TRY handler={}\n", handler_index));
                offset + 1
            }
            OpCode::EndTry => {
                output.push_str("END_TRY\n");
                offset + 1
            }
            OpCode::Catch(error_type) => {
                match error_type {
                    Some(et) => output.push_str(&format!("CATCH type={}\n", et)),
                    None => output.push_str("CATCH all\n"),
                }
                offset + 1
            }
            OpCode::EndCatch => {
                output.push_str("END_CATCH\n");
                offset + 1
            }
            OpCode::Throw(error_type) => {
                match error_type {
                    Some(et) => output.push_str(&format!("THROW type={}\n", et)),
                    None => output.push_str("THROW\n"),
                }
                offset + 1
            }
            OpCode::PopExceptionHandler => {
                output.push_str("POP_EXCEPTION_HANDLER\n");
                offset + 1
            }
            OpCode::Import(module_index) => {
                let module_name = &self.constants[*module_index];
                output.push_str(&format!("IMPORT {:4} '{}'\n", module_index, module_name.to_string()));
                offset + 1
            }
            OpCode::ImportFrom(module_index, items_index) => {
                let module_name = &self.constants[*module_index];
                let items_array = &self.constants[*items_index];
                output.push_str(&format!("IMPORT_FROM {:4} '{}' items={}\n", module_index, module_name.to_string(), items_array.to_string()));
                offset + 1
            }
        }
    }
}

