//! Style rules: STYLE001-003 (independent AST walks).

use crate::ast::Ast::{Module, AstArena};
use crate::sema::Sema::SemaResult;
use crate::tooling::common::Diagnostic::Diagnostic;

/// STYLE001: naming convention check.
/// fun/type: lowerCamel / PascalCase; val/var: lowerCamel; CONST: UPPER_SNAKE.
pub fn naming(_module: &Module, _arena: &AstArena, _sema: &SemaResult) -> Vec<Diagnostic> {
    // Phase 1: stub — returns empty (will be implemented with AST walk)
    Vec::new()
}

/// STYLE002: unused import.
pub fn unused_import(_module: &Module, _arena: &AstArena, sema: &SemaResult) -> Vec<Diagnostic> {
    // Phase 1: stub — check if import aliases are referenced in the module
    let _aliases = &sema.import_aliases;
    Vec::new()
}

/// STYLE003: redundant parentheses.
pub fn redundant_paren(_module: &Module, _arena: &AstArena, _sema: &SemaResult) -> Vec<Diagnostic> {
    // Phase 1: stub — walk Expr::Paren and check precedence
    Vec::new()
}
