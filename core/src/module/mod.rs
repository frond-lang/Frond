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
//! `LoadedModule` owns the bump arena (`Box<Bump>`) and source string (`Box<str>`) that back
//! each `Module<'static>`. The `'static` lifetime on `module` is a soundness fiction maintained
//! by struct field drop order (reverse declaration): `module` is dropped before `_arena` and
//! `_source`, so arena-allocated data is never accessed after free. `Module` has no custom
//! `Drop`, so dropping it only frees `Vec` buffers (on the regular heap) and drops `&str`
//! references (no-ops). Memory is fully reclaimed when a `LoadedModule` is dropped or replaced.
//!
//! ## Module path conventions
//!
//! `import std.io.File` → module_path = ["std", "io", "File"]
//! → resolves to the file path "std/io/File.frond"
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
