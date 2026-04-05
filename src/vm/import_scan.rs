//! Token-level scan for `import` / `from … import` module names (no full parse).
//! Used to preload native `operator_descriptor` before the main parse pass.

use crate::lexer::{Token, TokenKind};
use std::collections::BTreeSet;

/// Collects top-level module paths referenced by import statements (deduplicated, sorted).
///
/// Handles:
/// - `import a`, `import a, b`
/// - `from pkg.sub import x`, `from pkg import *`
///
/// Does not evaluate relative imports (`from . import`); those are skipped.
pub fn collect_imported_module_names_from_tokens(tokens: &[Token]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i].kind {
            TokenKind::Import => {
                i += 1;
                while i < tokens.len() {
                    match tokens[i].kind {
                        TokenKind::Identifier => {
                            let name = tokens[i].lexeme.clone();
                            seen.insert(name);
                            i += 1;
                            if i < tokens.len() && tokens[i].kind == TokenKind::As {
                                // import foo as bar — skip alias tokens
                                i += 1;
                                if i < tokens.len() && tokens[i].kind == TokenKind::Identifier {
                                    i += 1;
                                }
                            }
                            if i < tokens.len() && tokens[i].kind == TokenKind::Comma {
                                i += 1;
                                continue;
                            }
                            break;
                        }
                        _ => break,
                    }
                }
            }
            TokenKind::From => {
                i += 1;
                // Relative import: from . / from .. — skip this statement
                if i < tokens.len() && tokens[i].kind == TokenKind::Dot {
                    while i < tokens.len() && tokens[i].kind == TokenKind::Dot {
                        i += 1;
                    }
                    while i < tokens.len() && tokens[i].kind != TokenKind::Import {
                        i += 1;
                    }
                    if i < tokens.len() && tokens[i].kind == TokenKind::Import {
                        i += 1;
                    }
                    continue;
                }
                let mut parts: Vec<String> = Vec::new();
                while i < tokens.len() && tokens[i].kind == TokenKind::Identifier {
                    parts.push(tokens[i].lexeme.clone());
                    i += 1;
                    if i < tokens.len() && tokens[i].kind == TokenKind::Dot {
                        i += 1;
                        continue;
                    }
                    break;
                }
                if !parts.is_empty() {
                    seen.insert(parts.join("."));
                }
                while i < tokens.len() && tokens[i].kind != TokenKind::Import {
                    i += 1;
                }
                if i < tokens.len() && tokens[i].kind == TokenKind::Import {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    seen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    #[test]
    fn collects_import_and_from() {
        let src = "import ml\nfrom foo.bar import x\n";
        let tokens = Lexer::new(src).tokenize().expect("lex");
        let names = collect_imported_module_names_from_tokens(&tokens);
        assert!(names.contains(&"ml".to_string()));
        assert!(names.contains(&"foo.bar".to_string()));
    }

    #[test]
    fn dedupes_and_sorted() {
        let src = "import ml\nimport ml, a";
        let tokens = Lexer::new(src).tokenize().expect("lex");
        let names = collect_imported_module_names_from_tokens(&tokens);
        assert_eq!(names, vec!["a".to_string(), "ml".to_string()]);
    }
}
