//! Cache entries and error types for module loading.

use bumpalo::Bump;
use rustc_hash::FxHashSet;

use crate::ast::Ast::Module;

/// A loaded module entry.
///
/// Owns the bump arena and source string that back `module`'s `&'a str` references.
/// The `'static` lifetime on `module` is a soundness fiction maintained by keeping
/// `_arena` and `_source` alive for as long as `module` — Rust drops fields in
/// reverse declaration order, so `module` is dropped before `_arena` and `_source`,
/// and `Module` has no custom `Drop` that would dereference arena data.
pub struct LoadedModule {
    /// Owning arena — all AST nodes and dynamically built strings in `module` are
    /// allocated from this arena. Dropped after `module`.
    pub(crate) _arena: Box<Bump>,
    /// Owning source string — the original source code. Token lexemes in `module`
    /// point into this string. Dropped after `module`.
    pub(crate) _source: Box<str>,
    /// The parsed AST module. References data in `_arena` and `_source`.
    /// Safety: the `'static` lifetime is maintained by `_arena` / `_source` ownership.
    pub module: Module<'static>,
    /// Public symbols exported by the module (names of `pub fun` / `pub type` / `pub val`).
    pub exports: FxHashSet<String>,
}

/// Reasons for module loading failures.
///
/// All loading failures (module not found / parse failed) are recorded structurally
/// in `ModuleLoader::load_errors` and reported by the caller, preventing silent error
/// swallowing that would trigger cascading sema false positives.
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Module path not found (neither the stdlib embed table nor filesystem search paths matched).
    ModuleNotFound { path: String },
    /// Module source parsing failed (fatal parse error, AST unavailable).
    ParseFailed {
        path: String,
        line: u32,
        column: u32,
        message: String,
    },
    /// Circular import detected (A imports B, B imports A).
    CircularImport { path: String },
}
