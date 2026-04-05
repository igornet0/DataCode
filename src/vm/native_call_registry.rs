//! Parse-time registry of native export keys → ordered parameter names (from `native_call_descriptor`).
//!
//! Rows in the descriptor array:
//! - `[export_key, param1, param2, ...]` — kwargs for the native export `export_key`.
//! - `["__method__", method_name, export_key]` — ambiguous source method name (e.g. `split`) maps to
//!   that export for compile-time plugin vs builtin disambiguation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::error::LangError;
use crate::common::value::Value;

const METHOD_MAP_TAG: &str = "__method__";

/// Keys and param lists merged from loaded native modules' `native_call_descriptor` export.
#[derive(Clone, Default, Debug)]
pub struct NativeCallParamRegistry {
    pub by_key: HashMap<String, Vec<String>>,
    /// Language method name → native export key (e.g. `split` → `native_dataset_split`).
    pub method_to_export: HashMap<String, String>,
}

impl NativeCallParamRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge rows from a descriptor value (see module docs).
    pub fn merge_from_descriptor_value(
        &mut self,
        v: &Value,
        source_module: &str,
    ) -> Result<(), LangError> {
        let arr = match v {
            Value::Array(a) => a.borrow(),
            _ => {
                return Err(LangError::ParseError {
                    message: format!(
                        "native_call_descriptor from '{}' must return an array",
                        source_module
                    ),
                    line: 0,
                    file: None,
                });
            }
        };
        for row_v in arr.iter() {
            let row = match row_v {
                Value::Array(r) => r.borrow(),
                _ => {
                    return Err(LangError::ParseError {
                        message: format!(
                            "native_call_descriptor row from '{}' must be an array",
                            source_module
                        ),
                        line: 0,
                        file: None,
                    });
                }
            };
            if row.is_empty() {
                continue;
            }
            let key = match &row[0] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(LangError::ParseError {
                        message: format!(
                            "native_call_descriptor row key from '{}' must be a string",
                            source_module
                        ),
                        line: 0,
                        file: None,
                    });
                }
            };
            if key == METHOD_MAP_TAG {
                if row.len() < 3 {
                    return Err(LangError::ParseError {
                        message: format!(
                            "native_call_descriptor '__method__' row from '{}' needs [__method__, method, export_key]",
                            source_module
                        ),
                        line: 0,
                        file: None,
                    });
                }
                let method_name = match &row[1] {
                    Value::String(s) => s.clone(),
                    _ => {
                        return Err(LangError::ParseError {
                            message: format!(
                                "native_call_descriptor '__method__' method name from '{}' must be a string",
                                source_module
                            ),
                            line: 0,
                            file: None,
                        });
                    }
                };
                let export_key = match &row[2] {
                    Value::String(s) => s.clone(),
                    _ => {
                        return Err(LangError::ParseError {
                            message: format!(
                                "native_call_descriptor '__method__' export key from '{}' must be a string",
                                source_module
                            ),
                            line: 0,
                            file: None,
                        });
                    }
                };
                if self
                    .method_to_export
                    .insert(method_name.clone(), export_key)
                    .is_some()
                {
                    return Err(LangError::ParseError {
                        message: format!(
                            "native_call_descriptor: duplicate __method__ '{}' (module '{}')",
                            method_name, source_module
                        ),
                        line: 0,
                        file: None,
                    });
                }
                continue;
            }
            let params: Vec<String> = row[1..]
                .iter()
                .map(|c| match c {
                    Value::String(s) => Ok(s.clone()),
                    _ => Err(LangError::ParseError {
                        message: format!(
                            "native_call_descriptor param names from '{}' must be strings",
                            source_module
                        ),
                        line: 0,
                        file: None,
                    }),
                })
                .collect::<Result<_, _>>()?;
            if self.by_key.insert(key.clone(), params).is_some() {
                return Err(LangError::ParseError {
                    message: format!(
                        "native_call_descriptor: duplicate key '{}' (module '{}')",
                        key, source_module
                    ),
                    line: 0,
                    file: None,
                });
            }
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&[String]> {
        self.by_key.get(key).map(|v| v.as_slice())
    }

    /// Native export key for an ambiguous method name (e.g. `split`), if the plugin registered a mapping.
    pub fn export_for_method(&self, method: &str) -> Option<&str> {
        self.method_to_export.get(method).map(|s| s.as_str())
    }
}

pub type SharedNativeCallParamRegistry = Arc<NativeCallParamRegistry>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn merge_parses_param_rows_and_method_map() {
        let v = Value::Array(Rc::new(RefCell::new(vec![
            Value::Array(Rc::new(RefCell::new(vec![
                Value::String("native_dataset_split".to_string()),
                Value::String("test_size".to_string()),
            ]))),
            Value::Array(Rc::new(RefCell::new(vec![
                Value::String(METHOD_MAP_TAG.to_string()),
                Value::String("split".to_string()),
                Value::String("native_dataset_split".to_string()),
            ]))),
        ])));
        let mut r = NativeCallParamRegistry::new();
        r.merge_from_descriptor_value(&v, "ml").expect("merge");
        assert_eq!(
            r.get("native_dataset_split").map(|x| x.to_vec()),
            Some(vec!["test_size".to_string()])
        );
        assert_eq!(r.export_for_method("split"), Some("native_dataset_split"));
    }
}
