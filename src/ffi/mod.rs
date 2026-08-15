#![allow(non_snake_case)]
//! ffi — Foreign Function Interface modules.
//!
//! FFI form: **only** stdlib `@extern("C") #{ }#` (compiled and linked into the
//! frond binary by build.rs). Runtime symbol resolution goes through [`Symbols`]
//! (dlsym self-lookup + cache); there is no longer a compile-time binding table.
//!
//! User code is not allowed to use FFI directly (`@extern`/`#{ }#` are only
//! available to builtin modules).
//!
//! Aggregates:
//! - [`Abi`]: C ABI dynamic invoker (trampoline table, architecture-independent)
//! - [`ExternC`]: `@extern("C")` AST extractor + C source generation
//! - [`Gen`]: shared type mapping + C code generation (shared by build.rs and ExternC.rs)
//! - [`Marshal`]: Value ↔ C ABI bidirectional conversion
//! - [`Symbols`]: stdlib C symbol address cache (dlsym self-lookup)

pub mod Abi;
pub mod ExternC;
pub mod Gen;
pub mod Marshal;
pub mod Symbols;

/// Attribute name constants (single source of truth).
pub const ATTR_EXTERN: &str = "extern";
pub const ATTR_C_INCLUDE: &str = "c_include";
/// `@internal` is a reserved attribute marking "language-implementation internal
/// modules". Writing it in user code raises an error; stdlib modules under the
/// `builtin/` path are authorized by path and need no explicit annotation.
pub const ATTR_INTERNAL: &str = "internal";
