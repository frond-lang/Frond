#![allow(non_snake_case)]
//! value — Frond unified value system
//!
//! Split into five submodules:
//! - `Value`: value representation layer (scalar primitive types + heap object types)
//! - `Arena`: storage and query layer (Bucket + ValueArena + ValueTrait + equality)
//! - `Ops`: operation layer (Num/BitOps + cast + batch/SIMD + allocator + pure arithmetic core)
//! - `Reflect`: reflection layer (extern "C" primitives + format_value + layout queries)
//! - `Tag`: type metadata (ValueTag + TypeFamily + BuiltinInfo + BUILTIN_TABLE)

// Note: the core `Value` enum in Value.rs shares its name with the module.
// Using `pub mod Value;` would make the module name shadow the glob-re-exported `Value`
// enum, causing `crate::value::Value` to resolve to the module rather than the enum.
// We therefore use `#[path]` to name the module `value` (private) and expose its pub
// items only through the glob re-export.
// (types/Ty.rs avoids this issue by naming the enum `Type` instead of `Ty`.)
#[path = "Value.rs"]
mod value;

#[macro_use]
pub mod Arena;

pub mod Registry;

pub mod Ops;
pub mod Reflect;
pub mod Tag;

pub use value::*;
pub use Arena::*;
pub use Ops::*;
pub use Reflect::*;
pub use Tag::*;
