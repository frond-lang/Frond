//! Reflect.rs — reflection primitive `extern "C" fn` implementations.
//!
//! All primitives receive a `ValueHandle` (`u32`); internally they look up the `HeapObj` from the global
//! `ValueArena` and directly match on the type information it already carries. No `type_table`,
//! no `type_id` injection, no table lookup.
//!
//! Responsibility boundaries:
//! - Value.rs: the Value system is self-describing (fields like `RecordValue.type_name` carry their own type info)
//! - Reflect.rs: `extern "C"` primitives that match on `HeapObj` to return reflection information
//! - Raw.kz: `@extern("C")` declarations, C ABI calling convention
//! - Reflect.kz: `Reflect` built-in type + wrapper functions

use std::ffi::CString;

use super::arena::ValueArena;
use super::value::{F16, F128, HeapObj, RefKind, ValueTag, Value, ValueHandle};

// =========================================================================
// TypeKind enum (values match the Kuzo-side kind constants; lets users classify types)
// =========================================================================

pub const KIND_NULL: u8 = 0;
pub const KIND_VOID: u8 = 1;
pub const KIND_BOOL: u8 = 2;
pub const KIND_CHAR: u8 = 3;
pub const KIND_I8: u8 = 4;
pub const KIND_I16: u8 = 5;
pub const KIND_I32: u8 = 6;
pub const KIND_I64: u8 = 7;
pub const KIND_U8: u8 = 8;
pub const KIND_U16: u8 = 9;
pub const KIND_U32: u8 = 10;
pub const KIND_U64: u8 = 11;
pub const KIND_ISIZE: u8 = 12;
pub const KIND_USIZE: u8 = 13;
pub const KIND_I128: u8 = 14;
pub const KIND_U128: u8 = 15;
pub const KIND_F16: u8 = 16;
pub const KIND_F32: u8 = 17;
pub const KIND_F64: u8 = 18;
pub const KIND_F128: u8 = 19;
pub const KIND_STR: u8 = 20;
pub const KIND_REF: u8 = 21;
pub const KIND_RECORD: u8 = 22;
pub const KIND_ADT: u8 = 23;
pub const KIND_NEWTYPE: u8 = 24;
pub const KIND_ARRAY: u8 = 25;

/// ValueTag → TypeKind mapping (scalars map directly; heap objects go through ref_kind).
fn tag_to_kind(tag: ValueTag) -> u8 {
    match tag {
        ValueTag::Null => KIND_NULL,
        ValueTag::Void => KIND_VOID,
        ValueTag::Bool => KIND_BOOL,
        ValueTag::Char => KIND_CHAR,
        ValueTag::I8 => KIND_I8,
        ValueTag::I16 => KIND_I16,
        ValueTag::I32 => KIND_I32,
        ValueTag::I64 => KIND_I64,
        ValueTag::U8 => KIND_U8,
        ValueTag::U16 => KIND_U16,
        ValueTag::U32 => KIND_U32,
        ValueTag::U64 => KIND_U64,
        ValueTag::Isize => KIND_ISIZE,
        ValueTag::Usize => KIND_USIZE,
        ValueTag::I128 => KIND_I128,
        ValueTag::U128 => KIND_U128,
        ValueTag::F16 => KIND_F16,
        ValueTag::F32 => KIND_F32,
        ValueTag::F64 => KIND_F64,
        ValueTag::F128 => KIND_F128,
        ValueTag::Ref => KIND_REF,
    }
}

/// ref_kind → TypeKind.
fn ref_kind_to_kind(rk: RefKind) -> u8 {
    match rk {
        RefKind::Str => KIND_STR,
        RefKind::Array => KIND_ARRAY,
        RefKind::Record => KIND_RECORD,
        RefKind::Adt => KIND_ADT,
        RefKind::Newtype => KIND_NEWTYPE,
        _ => KIND_REF,
    }
}

// =========================================================================
// str return helpers: thread_local buffers avoid dangling pointers.
//
// [R-1 contract] The pointer written to *out_data by write_str_out / write_slice_out is only valid
// until "the next call to any reflect primitive on the same thread". The C side must consume it
// immediately (memcpy) and must not retain it across reflect calls. Reason: the buffer is a
// thread_local single-slot; the next call replaces it.
// =========================================================================

thread_local! {
    static NAME_BUF: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::new("").unwrap());
}

fn write_str_out(s: &str, out_data: *mut *const u8, out_len: *mut usize) {
    NAME_BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        // [R-4] Strings containing NUL are treated as illegal (type names must not contain NUL); mark explicitly rather than silently producing an empty string
        *b = CString::new(s).unwrap_or_else(|_| CString::new("<invalid-name>").unwrap());
        unsafe {
            *out_data = b.as_ptr() as *const u8;
            *out_len = b.to_bytes().len();
        }
    });
}

/// Writes a static/borrowed byte-slice pointer to the out parameters (no CString involved; zero-copy).
/// The caller must ensure `data` outlives consumption of the pointer (static slice or thread_local buffer).
fn write_slice_out(data: &[u8], out_data: *mut *const u8, out_len: *mut usize) {
    unsafe {
        *out_data = data.as_ptr();
        *out_len = data.len();
    }
}

thread_local! {
    static FORMAT_BUF: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

// =========================================================================
// Reflection primitives — all `#[no_mangle] extern "C" fn`; receive `u32` (ValueHandle raw).
// =========================================================================

/// Returns the value's TypeKind (scalars map directly; heap objects look up ref_kind via the arena).
#[no_mangle]
pub extern "C" fn __reflect_kind(handle: u32) -> u8 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    if tag != ValueTag::Ref {
        return tag_to_kind(tag);
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        ref_kind_to_kind(obj.ref_kind())
    } else {
        KIND_NULL
    }
}

/// Returns the type name (scalars return a static string; heap objects read the type_name field via the arena).
///
/// The scalar branch is derived from `Type::BUILTIN_TABLE` (single source of truth): it looks up
/// the `&'static str` type name via `builtin_info_by_tag`, eliminating the original 21 hardcoded
/// `b"..."` branches.
/// Invariant: the outer `tag != ValueTag::Ref` guard ensures the tag is always in BUILTIN_TABLE (all
/// 20 non-Ref tags are registered); the `.expect` is a fail-fast on invariant violation, not a fallback.
#[no_mangle]
pub extern "C" fn __reflect_type_name(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    if tag != ValueTag::Ref {
        let info = crate::types::builtin_info_by_tag(tag)
            .expect("non-Ref ValueTag must be in BUILTIN_TABLE");
        // info.name is &'static str; the pointer is 'static and has no dangling risk
        write_slice_out(info.name.as_bytes(), out_data, out_len);
        return;
    }
    // Heap object: look up the user type name via the arena
    let name = if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            HeapObj::Record(r) => r.type_name.clone(),
            HeapObj::Adt(a) => a.type_name.clone(),
            HeapObj::Newtype(n) => n.type_name.clone(),
            HeapObj::Str(_) => "str".to_string(),
            HeapObj::Array(_) => "array".to_string(),
            _ => obj.ref_kind().as_str().to_string(),
        }
    } else {
        "<unknown>".to_string()
    };
    write_str_out(&name, out_data, out_len);
}

/// Returns the value's byte size (scalars delegate to `ValueTag::byte_width`; heap objects are estimated by ref_kind).
#[no_mangle]
pub extern "C" fn __reflect_size(handle: u32) -> u8 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    if tag != ValueTag::Ref {
        // Scalars/Null/Void uniformly delegate to byte_width (single source of truth in Value.rs)
        return tag.byte_width() as u8;
    }
    // Heap object: str/array estimated as 16 (data+len); others have no fixed size
    if let Some(obj) = ValueArena::get_global_obj(h) {
        match obj.ref_kind() {
            RefKind::Str => 16,
            RefKind::Array => 16,
            RefKind::Record | RefKind::Adt | RefKind::Newtype => 0,
            _ => 0,
        }
    } else { 0 }
}

/// Returns the field count (number of fields/elements of Record/Adt/Newtype/Array).
#[no_mangle]
pub extern "C" fn __reflect_field_count(handle: u32) -> u16 {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        return 0;
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            // [R-2] Clamp to u16::MAX instead of `as` truncation wraparound, preventing >65535 fields from silently becoming a small wrong value
            HeapObj::Record(r) => r.fields.len().min(u16::MAX as usize) as u16,
            HeapObj::Adt(a) => a.fields.len().min(u16::MAX as usize) as u16,
            HeapObj::Newtype(_) => 1,
            HeapObj::Array(a) => a.elements.len().min(u16::MAX as usize) as u16,
            _ => 0,
        }
    } else {
        0
    }
}

/// Returns the field name (field names of Record/Adt; arrays/tuples return an empty string).
#[no_mangle]
pub extern "C" fn __reflect_field_name(handle: u32, index: u16, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        write_str_out("", out_data, out_len);
        return;
    }
    let name = if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            HeapObj::Record(r) => r.field_names
                .get(index as usize)
                .and_then(|n| n.as_ref())
                .cloned()
                .unwrap_or_default(),
            HeapObj::Adt(a) => a.fields
                .get(index as usize)
                .and_then(|f| f.name.as_ref())
                .cloned()
                .unwrap_or_default(),
            _ => String::new(),
        }
    } else {
        String::new()
    };
    write_str_out(&name, out_data, out_len);
}

/// Returns the field value (a child ValueHandle for recursive reflection).
/// HeapObj fields have been migrated to Value; `alloc_value` converts them back to ValueHandle for FFI return.
#[no_mangle]
pub extern "C" fn __reflect_field_value(handle: u32, index: u16) -> u32 {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        return ValueHandle::NULL.to_raw();
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        match &*obj {
            // Record/Adt/Array fields are Values: alloc_value converts them back to ValueHandle
            HeapObj::Record(r) => r.fields.get(index as usize)
                .map(|f| ValueArena::with_global_mut(|a| a.alloc_value(f)).to_raw())
                .unwrap_or(ValueHandle::NULL.to_raw()),
            HeapObj::Adt(a) => a.fields.get(index as usize)
                .map(|f| ValueArena::with_global_mut(|a| a.alloc_value(&f.value)).to_raw())
                .unwrap_or(ValueHandle::NULL.to_raw()),
            // Newtype.inner is still a ValueHandle; return directly
            HeapObj::Newtype(n) => if index == 0 { n.inner.to_raw() } else { ValueHandle::NULL.to_raw() },
            HeapObj::Array(a) => a.elements.get(index as usize)
                .map(|f| ValueArena::with_global_mut(|arena| arena.alloc_value(f)).to_raw())
                .unwrap_or(ValueHandle::NULL.to_raw()),
            _ => ValueHandle::NULL.to_raw(),
        }
    } else {
        ValueHandle::NULL.to_raw()
    }
}

/// Returns the array length.
#[no_mangle]
pub extern "C" fn __reflect_array_len(handle: u32) -> usize {
    let h = ValueHandle::from_raw(handle);
    if h.tag() != ValueTag::Ref {
        return 0;
    }
    if let Some(obj) = ValueArena::get_global_obj(h) {
        if let HeapObj::Array(a) = &*obj {
            return a.elements.len();
        }
    }
    0
}

/// Returns the ADT constructor name.
#[no_mangle]
pub extern "C" fn __reflect_adt_constructor(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let ctor = if h.tag() == ValueTag::Ref {
        if let Some(obj) = ValueArena::get_global_obj(h) {
            if let HeapObj::Adt(a) = &*obj {
                a.constructor.clone()
            } else {
                String::new()
            }
        } else { String::new() }
    } else { String::new() };
    write_str_out(&ctor, out_data, out_len);
}

/// Converts a scalar to string (formats by ValueTag dispatch).
#[no_mangle]
pub extern "C" fn __reflect_scalar_to_str(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    FORMAT_BUF.with(|buf| {
        let mut s = buf.borrow_mut();
        s.clear();
        if tag == ValueTag::Ref {
            if let Some(obj) = ValueArena::get_global_obj(h) {
                match &*obj {
                    HeapObj::Str(kuzo_str) => s.push_str(kuzo_str.bytes()),
                    _ => s.push_str("<non-scalar>"),
                }
            } else {
                s.push_str("<unknown>");
            }
        } else {
            ValueArena::with_global(|a| {
                // [V-2] FFI boundary defense: a dirty handle's index may be out of bounds; validate before reading
                if !a.is_valid_handle_inner(h) {
                    s.push_str("<invalid>");
                    return;
                }
                match tag {
                    ValueTag::Null => s.push_str("null"),
                    ValueTag::Void => s.push_str("void"),
                    ValueTag::Bool => s.push_str(if h == ValueHandle::TRUE { "true" } else { "false" }),
                    ValueTag::Char => {
                        let c = a.get_char(h);
                        // Code point → Unicode scalar value → char (covers all valid code points, including non-ASCII)
                        if let Some(ch) = char::from_u32(c) {
                            s.push(ch);
                        } else {
                            s.push_str(&format!("U+{:04X}", c));
                        }
                    }
                    ValueTag::I8 => s.push_str(&a.get_i8(h).to_string()),
                    ValueTag::I16 => s.push_str(&a.get_i16(h).to_string()),
                    ValueTag::I32 => s.push_str(&a.get_i32(h).to_string()),
                    ValueTag::I64 => s.push_str(&a.get_i64(h).to_string()),
                    ValueTag::U8 => s.push_str(&a.get_u8(h).to_string()),
                    ValueTag::U16 => s.push_str(&a.get_u16(h).to_string()),
                    ValueTag::U32 => s.push_str(&a.get_u32(h).to_string()),
                    ValueTag::U64 => s.push_str(&a.get_u64(h).to_string()),
                    ValueTag::Isize => s.push_str(&a.get_isize(h).to_string()),
                    ValueTag::Usize => s.push_str(&a.get_usize(h).to_string()),
                    ValueTag::F32 => s.push_str(&a.get_f32(h).to_string()),
                    ValueTag::F64 => s.push_str(&a.get_f64(h).to_string()),
                    ValueTag::I128 => s.push_str(&a.get_i128(h).to_string()),
                    ValueTag::U128 => s.push_str(&a.get_u128(h).to_string()),
                    ValueTag::F16 => s.push_str(&F16(a.get_f16(h)).to_f32().to_string()),
                    ValueTag::F128 => s.push_str(&a.get_f128(h).to_f64().to_string()),
                    _ => s.push_str("<scalar>"),
                }
            });
        }
        // s borrows FORMAT_BUF; the pointer is valid until the next reflect call (see [R-1 contract])
        write_slice_out(s.as_bytes(), out_data, out_len);
    });
}

/// Top-level formatting entry point: recursively matches on HeapObj to produce a string.
#[no_mangle]
pub extern "C" fn __reflect_format(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    // Entry handle → Value; subsequent recursion takes the Value path throughout.
    let h = ValueHandle::from_raw(handle);
    let v = ValueArena::with_global(|arena| arena.get_value(h));
    let result = format_value(&v, 0);
    write_str_out(&result, out_data, out_len);
}

/// Recursively formats a Value into a String (internal function, not `extern "C"`).
/// [R-3] `depth` limits recursion depth, preventing stack overflow from reference cycles or deeply nested structures.
const FORMAT_MAX_DEPTH: u32 = 64;
pub fn format_value(v: &Value, depth: u32) -> String {
    // Depth exceeded: truncate to ellipsis to avoid stack overflow (defense against cycles/deep nesting)
    if depth > FORMAT_MAX_DEPTH {
        return "...".to_string();
    }
    match v {
        Value::Null => "null".to_string(),
        Value::Void => "void".to_string(),
        Value::Scalar(sv, tag) => {
            // Scalar formatting: read directly from ScalarValue, without going through ValueArena
            unsafe {
                match tag {
                    ValueTag::Bool => (if sv.bool_val { "true" } else { "false" }).to_string(),
                    ValueTag::Char => {
                        let c = sv.char_val;
                        // Code point → Unicode scalar value → char (covers all valid code points, including non-ASCII)
                        char::from_u32(c).map(|ch| ch.to_string()).unwrap_or_else(|| format!("U+{:04X}", c))
                    }
                    ValueTag::I8 => sv.i8_val.to_string(),
                    ValueTag::I16 => sv.i16_val.to_string(),
                    ValueTag::I32 => sv.i32_val.to_string(),
                    ValueTag::I64 => sv.i64_val.to_string(),
                    ValueTag::U8 => sv.u8_val.to_string(),
                    ValueTag::U16 => sv.u16_val.to_string(),
                    ValueTag::U32 => sv.u32_val.to_string(),
                    ValueTag::U64 => sv.u64_val.to_string(),
                    ValueTag::Isize => sv.isize_val.to_string(),
                    ValueTag::Usize => sv.usize_val.to_string(),
                    ValueTag::I128 => i128::from_ne_bytes(std::mem::transmute(sv.i128_val)).to_string(),
                    ValueTag::U128 => u128::from_ne_bytes(std::mem::transmute(sv.u128_val)).to_string(),
                    ValueTag::F16 => format!("{:?}", F16(sv.f16_val)),
                    ValueTag::F32 => sv.f32_val.to_string(),
                    ValueTag::F64 => sv.f64_val.to_string(),
                    ValueTag::F128 => format!("{:?}", F128(std::mem::transmute(sv.f128_val))),
                    _ => unreachable!("non-scalar tag in ScalarValue"),
                }
            }
        }
        Value::Ref(r) => {
            // Heap object: match on HeapObj
            match &**r {
                HeapObj::Record(rec) => {
                    let mut out = format!("{}(", rec.type_name);
                    for (i, f) in rec.fields.iter().enumerate() {
                        if i > 0 { out.push_str(", "); }
                        if let Some(name) = rec.field_names.get(i).and_then(|n| n.as_ref()) {
                            out.push_str(name);
                            out.push_str(": ");
                        }
                        out.push_str(&format_value(f, depth + 1));
                    }
                    out.push(')');
                    out
                }
                HeapObj::Adt(a) => {
                    if a.fields.is_empty() {
                        a.constructor.clone()
                    } else {
                        let mut out = format!("{}(", a.constructor);
                        for (i, f) in a.fields.iter().enumerate() {
                            if i > 0 { out.push_str(", "); }
                            if let Some(name) = &f.name {
                                out.push_str(name);
                                out.push_str(": ");
                            }
                            out.push_str(&format_value(&f.value, depth + 1));
                        }
                        out.push(')');
                        out
                    }
                }
                HeapObj::Newtype(n) => {
                    // Newtype.inner is still a ValueHandle: convert to Value then recurse
                    let inner_val = ValueArena::with_global(|arena| arena.get_value(n.inner));
                    format!("{}({})", n.type_name, format_value(&inner_val, depth + 1))
                }
                HeapObj::Array(a) => {
                    let mut out = String::from("[");
                    for (i, e) in a.elements.iter().enumerate() {
                        if i > 0 { out.push_str(", "); }
                        out.push_str(&format_value(e, depth + 1));
                    }
                    out.push(']');
                    out
                }
                HeapObj::Str(kuzo_str) => kuzo_str.bytes().to_string(),
                HeapObj::LazyVal(lazy) => {
                    // Forced LazyValue: format the cached value
                    // Unforced LazyValue: normally pre-processed by Engine's force_lazy_value_sync;
                    // here we only handle residual unforced LazyValues in nested structures (defensive fallback)
                    if lazy.forced.load(std::sync::atomic::Ordering::Relaxed) {
                        match &*lazy.cached.lock().unwrap() {
                            Some(v) => format_value(v, depth + 1),
                            None => "<lazy:empty>".to_string(),
                        }
                    } else {
                        "<lazy>".to_string()
                    }
                }
                HeapObj::ThrowVal(t) => {
                    // Throw value formatting: Ok(v) → "Ok(v)", Err(e) → "Err(e)"
                    match &t.payload {
                        crate::value::ThrowPayload::Ok(v) => {
                            format!("Ok({})", format_value(v, depth + 1))
                        }
                        crate::value::ThrowPayload::Err(e) => {
                            format!("Err({})", format_value(e, depth + 1))
                        }
                    }
                }
                HeapObj::ErrorVal(e) => {
                    // Error value formatting: display type name and message
                    format!("{}({})", e.type_name, e.message)
                }
                _ => {
                    // Other heap objects: fall back to the ref_kind name
                    "<non-scalar>".to_string()
                }
            }
        }
    }
}

/// Returns the type kind string ("Primitive"/"Record"/"Adt"/"Newtype"/"Str"/"Array"/"Nullable"/"Ref").
#[no_mangle]
pub extern "C" fn __reflect_kind_str(handle: u32, out_data: *mut *const u8, out_len: *mut usize) {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    let kind: &[u8] = if tag != ValueTag::Ref {
        match tag {
            ValueTag::Null => b"Null",
            ValueTag::Void => b"Void",
            _ => b"Primitive",
        }
    } else if let Some(obj) = ValueArena::get_global_obj(h) {
        match obj.ref_kind() {
            RefKind::Str => b"Str",
            RefKind::Array => b"Array",
            RefKind::Record => b"Record",
            RefKind::Adt => b"Adt",
            RefKind::Newtype => b"Newtype",
            RefKind::Closure => b"Closure",
            RefKind::TraitVal => b"Trait",
            _ => b"Ref",
        }
    } else {
        b"Null"
    };
    write_slice_out(kind, out_data, out_len);
}

/// Returns the value's layout size (scalars by tag; heap objects estimate total field size by ref_kind).
#[no_mangle]
pub extern "C" fn __reflect_layout_size(handle: u32) -> u32 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    match tag {
        ValueTag::Null | ValueTag::Void => 0,
        ValueTag::Bool => 1,
        ValueTag::Char => 4,
        ValueTag::I8 | ValueTag::U8 => 1,
        ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
        ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
        ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
        ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
        ValueTag::Ref => {
            if let Some(obj) = ValueArena::get_global_obj(h) {
                match &*obj {
                    HeapObj::Str(_) => 16,
                    HeapObj::Array(_) => {
                        // 16B fat pointer + element_size * len (estimated)
                        16
                    }
                    HeapObj::Record(r) => {
                        // Sum of field sizes (rough estimate, excluding padding)
                        r.fields.iter().map(value_size).sum::<u32>()
                    }
                    HeapObj::Adt(a) => {
                        // Sum of field sizes (current constructor's fields, excluding tag)
                        a.fields.iter().map(|f| value_size(&f.value)).sum::<u32>()
                    }
                    HeapObj::Newtype(n) => {
                        // inner value size
                        ValueArena::with_global(|arena| value_size(&arena.get_value(n.inner)))
                    }
                    _ => 0,
                }
            } else { 0 }
        }
    }
}

/// Returns the value's alignment (scalars by size; heap objects by maximum field alignment).
#[no_mangle]
pub extern "C" fn __reflect_layout_alignment(handle: u32) -> u32 {
    let h = ValueHandle::from_raw(handle);
    let tag = h.tag();
    match tag {
        ValueTag::Null | ValueTag::Void | ValueTag::Bool => 1,
        ValueTag::Char => 4,
        ValueTag::I8 | ValueTag::U8 => 1,
        ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
        ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
        ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
        ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
        ValueTag::Ref => {
            if let Some(obj) = ValueArena::get_global_obj(h) {
                match &*obj {
                    HeapObj::Str(_) => 8,
                    HeapObj::Array(_) => 8,
                    HeapObj::Record(r) => {
                        // Maximum field alignment
                        r.fields.iter().map(value_alignment).max().unwrap_or(1)
                    }
                    HeapObj::Adt(a) => {
                        // Maximum of tag(1) and maximum field alignment
                        a.fields.iter().map(|f| value_alignment(&f.value)).max().unwrap_or(1).max(1)
                    }
                    HeapObj::Newtype(n) => {
                        ValueArena::with_global(|arena| value_alignment(&arena.get_value(n.inner)))
                    }
                    _ => 8,
                }
            } else { 8 }
        }
    }
}

/// Estimates the byte size of a Value (used for Record/Adt/Newtype layout estimation).
fn value_size(v: &Value) -> u32 {
    match v {
        Value::Null | Value::Void => 0,
        Value::Scalar(_, tag) => {
            match tag {
                ValueTag::Bool => 1,
                ValueTag::Char => 4,
                ValueTag::I8 | ValueTag::U8 => 1,
                ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
                ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
                ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
                ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
                _ => unreachable!("non-scalar tag in ScalarValue"),
            }
        }
        Value::Ref(r) => {
            match &**r {
                HeapObj::Str(_) => 16,
                HeapObj::Array(_) => 16,
                HeapObj::Record(rec) => rec.fields.iter().map(value_size).sum(),
                // ADT: sum of field sizes (current constructor's fields, excluding tag)
                HeapObj::Adt(a) => a.fields.iter().map(|f| value_size(&f.value)).sum(),
                // Newtype: look up the inner value's size from the global arena
                HeapObj::Newtype(n) => {
                    ValueArena::with_global(|arena| value_size(&arena.get_value(n.inner)))
                }
                _ => 8,
            }
        }
    }
}

/// Estimates the alignment of a Value.
fn value_alignment(v: &Value) -> u32 {
    match v {
        Value::Null | Value::Void => 1,
        Value::Scalar(_, tag) => {
            match tag {
                ValueTag::Bool => 1,
                ValueTag::Char => 4,
                ValueTag::I8 | ValueTag::U8 => 1,
                ValueTag::I16 | ValueTag::U16 | ValueTag::F16 => 2,
                ValueTag::I32 | ValueTag::U32 | ValueTag::F32 => 4,
                ValueTag::I64 | ValueTag::U64 | ValueTag::F64 | ValueTag::Isize | ValueTag::Usize => 8,
                ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
                _ => unreachable!("non-scalar tag in ScalarValue"),
            }
        }
        Value::Ref(r) => {
            match &**r {
                HeapObj::Str(_) | HeapObj::Array(_) => 8,
                HeapObj::Record(rec) => rec.fields.iter().map(value_alignment).max().unwrap_or(1),
                HeapObj::Adt(a) => a.fields.iter().map(|f| value_alignment(&f.value)).max().unwrap_or(1).max(1),
                // Newtype: look up the inner value's alignment from the global arena
                HeapObj::Newtype(n) => {
                    ValueArena::with_global(|arena| value_alignment(&arena.get_value(n.inner)))
                }
                _ => 8,
            }
        }
    }
}

/// Public API: estimates layout size from a &Value (for Engine FFI calls).
pub fn reflect_layout_size(v: &Value) -> u32 {
    value_size(v)
}

/// Public API: estimates alignment from a &Value (for Engine FFI calls).
pub fn reflect_layout_alignment(v: &Value) -> u32 {
    value_alignment(v)
}

// =========================================================================
// RefKind::as_str helper (used as a type_name fallback)
// =========================================================================

impl RefKind {
    fn as_str(&self) -> &'static str {
        match self {
            RefKind::Str => "str",
            RefKind::Array => "array",
            RefKind::Record => "record",
            RefKind::Adt => "adt",
            RefKind::Newtype => "newtype",
            RefKind::Cell => "cell",
            RefKind::Range => "range",
            RefKind::Closure => "closure",
            RefKind::Partial => "partial",
            RefKind::Builtin => "builtin",
            RefKind::TraitVal => "trait",
            RefKind::LazyVal => "lazy",
            RefKind::ErrorVal => "error",
            RefKind::ThrowVal => "throw",
            RefKind::AtomicVal => "atomic",
            RefKind::AsyncVal => "async",
            RefKind::ChannelVal => "channel",
            RefKind::SenderVal => "sender",
            RefKind::ReceiverVal => "receiver",
            RefKind::CoroutineFrame => "coroutine",
        }
    }
}
