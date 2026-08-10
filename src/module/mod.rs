#![allow(non_snake_case)]
//! ModuleLoader — the unified module loader.
//!
//! Merges loading logic for stdlib and user modules:
//! - builtin modules are fully preloaded at initialization (visible by default, no import needed)
//! - std/user modules are loaded on demand via `resolve_and_load` (triggered when an `ImportDecl` is encountered)
//! - a module cache avoids redundant parse/check work
//!
//! ## Lifetime strategy
//!
//! stdlib sources are `&'static str` (via `include_str!`), while user module sources are
//! converted to `&'static str` through `Box::leak`. The bump arena is likewise made `&'static`
//! via `Box::leak`, so every `Module<'static>` produced by parsing is safe to cache.
//! Memory is reclaimed by the OS when the compiler process exits, with no leak risk.
//!
//! ## Module path conventions
//!
//! `import std.io.File` → module_path = ["std", "io", "File"]
//! → resolves to the file path "std/io/File.kz"
//! → first checks the stdlib embed table, then the filesystem search paths
//!
//! ## File organization
//!
//! - [`StdlibEmbed`]: stdlib source embed table (BUILTIN_FILES / STD_FILES) and lookups
//! - [`LoadError`]: the module cache entry `LoadedModule` and the load error types
//! - [`Loader`]: the `ModuleLoader` itself (cache, search paths, transitive dependency loading)

pub mod Error;
pub mod Loader;
pub mod StdlibEmbed;

pub use Loader::{collect_imports, ModuleLoader};
pub use StdlibEmbed::{find, BUILTIN_FILES, STD_FILES, StdlibFile};
