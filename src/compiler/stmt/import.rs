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
                
                // Register imported item names in globals for from-import, in deterministic order (sort by bound name) so the same set of names always gets the same global indices.
                let mut to_register: Vec<(String, String)> = Vec::new();
                for item in items {
                    match item {
                        ImportItem::Named(name) => {
                            to_register.push((name.clone(), module.clone()));
                        }
                        ImportItem::Aliased { alias, .. } => {
                            to_register.push((alias.clone(), module.clone()));
                        }
                        ImportItem::All => {
                            // All items will be imported at runtime, we can't register them here
                        }
                    }
                }
                to_register.sort_by(|a, b| a.0.cmp(&b.0));
                to_register.dedup_by(|a, b| a.0 == b.0);
                for (bound_name, mod_name) in to_register {
                    ctx.imported_symbols.insert(bound_name.clone(), mod_name);
                    if !ctx.scope.globals.contains_key(&bound_name) {
                        let global_index = ctx.scope.globals.len();
                        ctx.scope.globals.insert(bound_name.clone(), global_index);
                        ctx.chunk.global_names.insert(global_index, bound_name);
                    }
                }
            }
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Import statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}

