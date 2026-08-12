//! Idiom rules: IDIOM001-002 (independent AST walks).

use crate::ast::Ast::{Module, AstArena};
use crate::sema::Sema::SemaResult;
use crate::tooling::Common::Diagnostic::Diagnostic;

/// IDIOM001: prefer val over var.
pub fn prefer_val(_module: &Module, _arena: &AstArena, _sema: &SemaResult) -> Vec<Diagnostic> {
    // Phase 1: stub — check if var is never reassigned
    Vec::new()
}

/// IDIOM002: string concatenation could use interpolation.
pub fn string_interpolation(_module: &Module, _arena: &AstArena, _sema: &SemaResult) -> Vec<Diagnostic> {
    // Phase 1: stub — walk Binary(Add, String, String)
    Vec::new()
}
