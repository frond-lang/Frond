// Frond basic type ↔ C type mapping table (single source of truth).
//
// This file is pure data + lookup functions, with **no `crate::` dependency**, so it
// can be reused from two places:
//  - lib crate: `crate::types::Ctype` (via `pub mod Ctype`)
//  - build.rs: `include!` inline (build-time text extraction path)
//
// Gen.rs's `FROND_TYPE_MAP` / `c_type_to_rust` and other FFI mappings derive from
// this table, avoiding the Frond→C type correspondence being scattered across
// multiple places.

// ============ Frond type → C type ============

/// Frond basic type → C type mapping table.
///
/// Covers:
///  - scalar integers: i8..i128, u8..u128, isize, usize
///  - floating-point: f16, f32, f64, f128
///  - boolean/character: bool, char
///  - unit type: void
///  - string: str (passed on the C side via data/len, mapped here to `const char*`)
///  - byte array: u8[] (passed on the C side via data/len, mapped here to `uint8_t*`)
///  - raw pointers: *u8, *i8, ..., *void
///
/// Note: i128/u128/f128 are passed via an out-param pattern in C return position
/// (MSVC does not support `__int128` return values), but still use LoHi splitting in
/// argument position; this table gives "the corresponding C type name for that type",
/// and the concrete passing strategy at the FFI layer is decided by Gen.rs's
/// `CParamKind`.
pub const TO_C_TYPE: &[(&str, &str)] = &[
    // ── scalar integers ──
    ("i8",    "int8_t"),
    ("i16",   "int16_t"),
    ("i32",   "int32_t"),
    ("i64",   "int64_t"),
    ("i128",  "__int128"),
    ("u8",    "uint8_t"),
    ("u16",   "uint16_t"),
    ("u32",   "uint32_t"),
    ("u64",   "uint64_t"),
    ("u128",  "unsigned __int128"),
    // isize → int64_t (NOT ssize_t — MSVC has no ssize_t; same width/layout
    // on every LP64 POSIX target, and the Marshal slot is a plain 8-byte int).
    ("isize", "int64_t"),
    ("usize", "size_t"),
    // ── floating-point ──
    ("f16",   "uint16_t"),    // IEEE 754 binary16 passed via uint16
    ("f32",   "float"),
    ("f64",   "double"),
    ("f128",  "unsigned __int128"), // IEEE 754 binary128 bit pattern passed via __int128
    // ── boolean/character/unit ──
    ("bool",  "int"),
    ("char",  "uint32_t"),
    ("void",  "void"),
    // ── string/byte array (actually passed via a data/len two-argument pair; the element pointer type is given here) ──
    ("str",   "const char*"),
    ("u8[]",  "uint8_t*"),
    // ── raw pointers ──
    ("*u8",   "uint8_t*"),
    ("*i8",   "int8_t*"),
    ("*u16",  "uint16_t*"),
    ("*i16",  "int16_t*"),
    ("*u32",  "uint32_t*"),
    ("*i32",  "int32_t*"),
    ("*u64",  "uint64_t*"),
    ("*i64",  "int64_t*"),
    ("*void", "void*"),
];

/// Frond type name → C type name lookup.
///
/// Returns `None` to indicate that the type has no direct C counterpart (e.g. a
/// user-defined type).
#[inline]
pub fn to_c_type(name: &str) -> Option<&'static str> {
    TO_C_TYPE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, c)| *c)
}
