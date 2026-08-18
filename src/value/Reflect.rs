//! Reflect.rs — reflection helper implementations (pure Rust).
//!
//! Reflect operations are standalone `CF_REFLECT_*` compute_fns (see
//! `ir/Compute.rs::compute_reflect_kind` etc.) that call directly into the
//! pure-Rust helpers kept here. There is no FFI/@builtin layer — the compute_fns
//! are the sole entry point.
//!
//! What remains:
//! - `format_value` — recursive Value → String formatting (the engine behind
//!   `format(x)` / `x.format()` / string interpolation).
//! - `reflect_layout_size` / `reflect_layout_alignment` — aggregate layout
//!   estimates used by `x.size()` / `x.alignment()`.
//! - `value_size` / `value_alignment` — internal helpers for the above.

use super::value::{F16, F128, HeapObj, Value};

// =========================================================================
// format_value — recursive Value → String formatting.
// [R-3] `depth` limits recursion depth, preventing stack overflow from
// reference cycles or deeply nested structures.
// =========================================================================

const FORMAT_MAX_DEPTH: u32 = 64;

/// Recursively formats a Value into a String (internal helper, not `extern "C"`).
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
                    super::ValueTag::Bool => (if sv.bool_val { "true" } else { "false" }).to_string(),
                    super::ValueTag::Char => {
                        let c = sv.char_val;
                        // Code point → Unicode scalar value → char (covers all valid code points, including non-ASCII)
                        char::from_u32(c).map(|ch| ch.to_string()).unwrap_or_else(|| format!("U+{:04X}", c))
                    }
                    super::ValueTag::I8 => sv.i8_val.to_string(),
                    super::ValueTag::I16 => sv.i16_val.to_string(),
                    super::ValueTag::I32 => sv.i32_val.to_string(),
                    super::ValueTag::I64 => sv.i64_val.to_string(),
                    super::ValueTag::U8 => sv.u8_val.to_string(),
                    super::ValueTag::U16 => sv.u16_val.to_string(),
                    super::ValueTag::U32 => sv.u32_val.to_string(),
                    super::ValueTag::U64 => sv.u64_val.to_string(),
                    super::ValueTag::Isize => sv.isize_val.to_string(),
                    super::ValueTag::Usize => sv.usize_val.to_string(),
                    super::ValueTag::I128 => i128::from_ne_bytes(std::mem::transmute(sv.i128_val)).to_string(),
                    super::ValueTag::U128 => u128::from_ne_bytes(std::mem::transmute(sv.u128_val)).to_string(),
                    super::ValueTag::F16 => format!("{:?}", F16(sv.f16_val)),
                    super::ValueTag::F32 => sv.f32_val.to_string(),
                    super::ValueTag::F64 => sv.f64_val.to_string(),
                    super::ValueTag::F128 => format!("{:?}", F128(std::mem::transmute(sv.f128_val))),
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
                    let inner_val = super::Arena::ValueArena::with_global(|arena| arena.get_value(n.inner));
                    format!("{}({})", n.type_name, format_value(&inner_val, depth + 1))
                }
                HeapObj::Array(a) => {
                    let mut out = String::from("[");
                    // SoA-first: elements can be an empty shell (single-source
                    // clones keep data only in the contiguous storage).
                    let n = a.len();
                    for i in 0..n {
                        if i > 0 { out.push_str(", "); }
                        let e = a.get(i).unwrap_or(crate::value::Value::VOID);
                        out.push_str(&format_value(&e, depth + 1));
                    }
                    out.push(']');
                    out
                }
                HeapObj::Str(frond_str) => frond_str.bytes().to_string(),
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

// =========================================================================
// Layout estimation — used by x.size() / x.alignment() reflect methods.
// =========================================================================

/// Public API: estimates layout size from a &Value (for compute_reflect_layout_size).
pub fn reflect_layout_size(v: &Value) -> u32 {
    value_size(v)
}

/// Public API: estimates alignment from a &Value (for compute_reflect_layout_align).
pub fn reflect_layout_alignment(v: &Value) -> u32 {
    value_alignment(v)
}

/// Estimates the byte size of a Value (used for Record/Adt/Newtype layout estimation).
fn value_size(v: &Value) -> u32 {
    match v {
        Value::Null | Value::Void => 0,
        Value::Scalar(_, tag) => {
            match tag {
                super::ValueTag::Bool => 1,
                super::ValueTag::Char => 4,
                super::ValueTag::I8 | super::ValueTag::U8 => 1,
                super::ValueTag::I16 | super::ValueTag::U16 | super::ValueTag::F16 => 2,
                super::ValueTag::I32 | super::ValueTag::U32 | super::ValueTag::F32 => 4,
                super::ValueTag::I64 | super::ValueTag::U64 | super::ValueTag::F64 | super::ValueTag::Isize | super::ValueTag::Usize => 8,
                super::ValueTag::I128 | super::ValueTag::U128 | super::ValueTag::F128 => 16,
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
                    super::Arena::ValueArena::with_global(|arena| value_size(&arena.get_value(n.inner)))
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
                super::ValueTag::Bool => 1,
                super::ValueTag::Char => 4,
                super::ValueTag::I8 | super::ValueTag::U8 => 1,
                super::ValueTag::I16 | super::ValueTag::U16 | super::ValueTag::F16 => 2,
                super::ValueTag::I32 | super::ValueTag::U32 | super::ValueTag::F32 => 4,
                super::ValueTag::I64 | super::ValueTag::U64 | super::ValueTag::F64 | super::ValueTag::Isize | super::ValueTag::Usize => 8,
                super::ValueTag::I128 | super::ValueTag::U128 | super::ValueTag::F128 => 16,
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
                    super::Arena::ValueArena::with_global(|arena| value_alignment(&arena.get_value(n.inner)))
                }
                _ => 8,
            }
        }
    }
}
