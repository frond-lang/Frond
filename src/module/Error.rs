//! Cache entries and error types for module loading.

use rustc_hash::FxHashSet;

use crate::ast::Ast::Module;

/// A loaded module entry.
pub struct LoadedModule {
    /// The parsed AST module (`'static` lifetime, safe to cache).
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
