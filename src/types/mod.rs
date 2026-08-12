#![allow(non_snake_case)]
//! Type — Kuzo type system module.
//!
//! Hosts the static attributes of all builtin types, the unified semantic-layer enum
//! `Type` (the single source of types, `Copy`), `TypeArena` (type allocator +
//! unify/occurs/resolve), the `TypeOps` trait + ops lookup table, `DynamicOpsRegistry`
//! (replacing TypeDescriptorPool), and the type-family classification `TypeFamily`.
//!
//! Both the sema and IR layers use `Type`; there is no longer `ConcreteType` /
//! `TypeDescriptor`.
//!
//! Submodule organization:
//! - `Tag`: type discriminator tags (`ValueTag`) + base structures
//!   (`TypeHandle` / `TypeFamily` / `FieldType` / `TraitMethodSig` / `EnvId`)
//! - `Ty` (file `Ty.rs`): the unified type enum `Type` + `TypeKind` + type variables +
//!   `BuiltinInfo`/`BUILTIN_TABLE` + `TypeDetail` + `SemKind` + `UnifyError`
//! - `Arena`: the type allocator `TypeArena` + unify/occurs/resolve + snapshots
//! - `Display`: type display formatting (`TypeDisplay`)
//! - `Ops`: `TypeOps` trait + scalar/reference ops implementations + ops lookup table +
//!   `DynamicOpsRegistry`

pub mod Tag;

// Re-export value::Tag contents (ValueTag, TypeFamily, BuiltinInfo, etc.) so that
// `crate::types::ValueTag` etc. remain available. ValueTag lives in the value
// module; types depends on value (unidirectional).
pub use crate::value::{ValueTag, TypeFamily, BuiltinInfo, BUILTIN_TABLE, builtin_info_by_name, builtin_info_by_tag, builtin_info_by_type_id};

// The file `Ty.rs` defines the `Type` enum. The module name `Ty` (matching the file
// name) does not collide with the `Type` enum, so `pub mod Ty;` works directly —
// `crate::types::Type` resolves to the enum via glob re-export, and `crate::types::Ty`
// resolves to the module.
pub mod Ty;

pub mod Arena;
pub mod Display;
pub mod Ops;

pub use Tag::*;
pub use Ty::*;
pub use Arena::*;
pub use Display::*;
pub use Ops::*;
