// =========================================================================
// TypeKind — single source of truth for reflect kind numbering.
//
// Every Value maps to a TypeKind via `reflect_kind(&Value)`. The numbering is
// stable across the FFI boundary, compute_fn layer, and the user-facing
// `reflect.Value.kind()` method.
//
// Numbering rule: scalars collapse to one bucket (PRIMITIVE = 2); heap objects
// get one slot each in HeapObj declaration order. This matches what the user
// actually observes at runtime — the historical `KIND_*` constants in
// `value/Reflect.rs` (0-26, keyed by ValueTag) were dead code under the current
// dispatch path and are deleted.
// =========================================================================

/// Reflect type kind constants (0–23). Authoritative numbering shared by:
/// - `ir/Compute.rs::reflect_kind` (the producer)
/// - `ir/Compute.rs::reflect_kind_str` (display name)
/// - `value/Reflect.rs` (legacy `#[no_mangle]` ABI surface, being removed)
/// - Builder type-descriptor synthesis (compile-time `kind` field)
#[allow(non_snake_case)]
pub mod kind {
    pub const NULL: u8 = 0;
    pub const VOID: u8 = 1;
    pub const PRIMITIVE: u8 = 2;
    pub const STR: u8 = 3;
    pub const ARRAY: u8 = 4;
    pub const RECORD: u8 = 5;
    pub const ADT: u8 = 6;
    pub const NEWTYPE: u8 = 7;
    pub const CELL: u8 = 8;
    pub const RANGE: u8 = 9;
    pub const CLOSURE: u8 = 10;
    pub const PARTIAL: u8 = 11;
    pub const BUILTIN: u8 = 12;
    pub const TRAIT: u8 = 13;
    pub const LAZY: u8 = 14;
    pub const ERROR: u8 = 15;
    pub const THROW: u8 = 16;
    pub const ATOMIC: u8 = 17;
    pub const ASYNC: u8 = 18;
    pub const CHANNEL: u8 = 19;
    pub const SENDER: u8 = 20;
    pub const RECEIVER: u8 = 21;
    pub const COROUTINE: u8 = 22;
    pub const PTR: u8 = 23;
}
