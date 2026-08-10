#![allow(non_snake_case)]
//! Type — Kuzo type system module.
//!
//! Hosts the static attributes of all builtin types, the unified semantic-layer enum
//! `Ty` (the single source of types, `Copy`), `TypeArena` (type allocator +
//! unify/occurs/resolve), the `TypeOps` trait + ops lookup table, `DynamicOpsRegistry`
//! (replacing TypeDescriptorPool), and the type-family classification `TypeFamily`.
//!
//! Both the sema and IR layers use `Ty`; there is no longer `ConcreteType` /
//! `TypeDescriptor`.
//!
//! Submodule organization:
//! - `Tag`: type discriminator tags (`ValueTag`) + base structures
//!   (`TypeHandle` / `TypeFamily` / `FieldType` / `TraitMethodSig` / `EnvId`)
//! - `ty` (file `Ty.rs`): the unified type enum `Ty` + type variables +
//!   `BuiltinInfo`/`BUILTIN_TABLE` + `TypeDetail` + `SemKind` + `UnifyError`
//! - `Arena`: the type allocator `TypeArena` + unify/occurs/resolve + snapshots
//! - `Display`: type display formatting (`TypeDisplay`)
//! - `Ops`: `TypeOps` trait + scalar/reference ops implementations + ops lookup table +
//!   `DynamicOpsRegistry`

pub mod Tag;

// Note: the core `Ty` enum in Ty.rs shares its name with the module.
// Using `pub mod Ty;` would make the module name shadow the glob-re-exported `Ty`
// enum, causing `crate::types::Ty` to resolve to the module rather than the enum and
// breaking external references. We therefore use `#[path]` to name the module `ty`
// (private) and expose its pub items only through the glob re-export.
#[path = "Ty.rs"]
mod ty;

pub mod Arena;
pub mod Display;
pub mod Ops;

pub use Tag::*;
pub use ty::*;
pub use Arena::*;
pub use Display::*;
pub use Ops::*;
