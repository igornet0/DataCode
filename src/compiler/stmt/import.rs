/// Компиляция import statements

use crate::parser::ast::{Stmt, ImportStmt, ImportItem};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;

pub fn compile_import(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Import { import_stmt, line } = stmt {
        match import_stmt {
            ImportStmt::Modules(modules) => {
                // import ml, plot
                for module in modules {
                    // Import statements are handled at runtime by the VM
                    // We compile them as a special opcode that the VM will handle
                    // Store the module name as a constant and emit Import opcode
                    let module_index = ctx.chunk.add_constant(Value::String(module.clone()));
                    ctx.chunk.write_with_line(OpCode::Import(module_index), *line);
                    
                    // Register the module name in globals so subsequent uses are recognized
                    if !ctx.scope.globals.contains_key(module) {
                        let global_index = ctx.scope.globals.len();
                        ctx.scope.globals.insert(module.clone(), global_index);
                        ctx.chunk.global_names.insert(global_index, module.clone());
                    }
                }
            }
            ImportStmt::From { module, items } => {
                // from ml import load_mnist, *
                // Создаем массив элементов импорта в константах
                use std::rc::Rc;
                use std::cell::RefCell;
                let mut item_strings = Vec::new();
                for item in items {
                    match item {
                        ImportItem::Named(name) => {
                            item_strings.push(Value::String(name.clone()));
                        }
                        ImportItem::Aliased { name, alias } => {
                            // Формат: "name:alias"
                            item_strings.push(Value::String(format!("{}:{}", name, alias)));
                        }
                        ImportItem::All => {
                            item_strings.push(Value::String("*".to_string()));
                        }
                    }
                }
                let items_array = Value::Array(Rc::new(RefCell::new(item_strings)));
                let items_index = ctx.chunk.add_constant(items_array);
                
                // Store the module name as a constant
                let module_index = ctx.chunk.add_constant(Value::String(module.clone()));
                ctx.chunk.write_with_line(OpCode::ImportFrom(module_index, items_index), *line);
                
                // Register the module name in globals
                if !ctx.scope.globals.contains_key(module) {
                    let global_index = ctx.scope.globals.len();
                    ctx.scope.globals.insert(module.clone(), global_index);
                    ctx.chunk.global_names.insert(global_index, module.clone());
                }
                
                // Register imported item names in globals for from-import
                for item in items {
                    match item {
                        ImportItem::Named(name) => {
                            if !ctx.scope.globals.contains_key(name) {
                                let global_index = ctx.scope.globals.len();
                                ctx.scope.globals.insert(name.clone(), global_index);
                                ctx.chunk.global_names.insert(global_index, name.clone());
                            }
                        }
                        ImportItem::Aliased { alias, .. } => {
                            if !ctx.scope.globals.contains_key(alias) {
                                let global_index = ctx.scope.globals.len();
                                ctx.scope.globals.insert(alias.clone(), global_index);
                                ctx.chunk.global_names.insert(global_index, alias.clone());
                            }
                        }
                        ImportItem::All => {
                            // All items will be imported at runtime, we can't register them here
                        }
                    }
                }
            }
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Import statement".to_string(),
            line: stmt.line(),
        })
    }
}

