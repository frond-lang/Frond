#![allow(non_snake_case)]
//! value — Kuzo unified value system
//!
//! Split into four submodules:
//! - `Value`: value representation layer (scalar primitive types + heap object types)
//! - `Arena`: storage and query layer (Bucket + ValueArena + ValueTrait + equality)
//! - `Ops`: operation layer (Num/BitOps + cast + batch/SIMD + allocator + pure arithmetic core)
//! - `Reflect`: reflection layer (extern "C" primitives + format_value + layout queries)

#[path = "Value.rs"]
mod value;

// `#[macro_use]` makes the read_int_as! / write_int_bytes! macros defined in Arena.rs
// visible to the subsequent Ops.rs module.
#[macro_use]
#[path = "Arena.rs"]
mod arena;

#[path = "Ops.rs"]
mod ops;

#[path = "Reflect.rs"]
mod reflect;

pub use value::*;
pub use arena::*;
pub use ops::*;
pub use reflect::*;
