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
/// Formats a single-block record via its shared shape (cast_to_str path).
pub fn format_record_value(r: &crate::value::RecordRef) -> String {
    use crate::value::ShapeKind;
    let shape = r.shape();
    match shape.kind {
        ShapeKind::Adt => {
            if r.field_count() == 0 {
                shape.constructor.to_string()
            } else {
                let mut out = format!("{}(", shape.constructor);
                for i in 0..r.field_count() {
                    if i > 0 { out.push_str(", "); }
                    if let Some(name) = shape.field_names.get(i).and_then(|n| n.as_ref()) {
                        out.push_str(name);
                        out.push_str(": ");
                    }
                    out.push_str(&format_value(&r.field(i), 0));
                }
                out.push(')');
                out
            }
        }
        ShapeKind::Newtype => format!(
            "{}({})",
            crate::sema::Sema::display_type_name(&shape.type_name),
            format_value(&if r.field_count() > 0 { r.field(0) } else { crate::value::Value::VOID }, 0)
        ),
        ShapeKind::Record => {
            let mut out = format!("{}(", crate::sema::Sema::display_type_name(&shape.type_name));
            for i in 0..r.field_count() {
                if i > 0 { out.push_str(", "); }
                if let Some(name) = shape.field_names.get(i).and_then(|n| n.as_ref()) {
                    out.push_str(name);
                    out.push_str(": ");
                }
                out.push_str(&format_value(&r.field(i), 0));
            }
            out.push(')');
            out
        }
    }
}

pub fn format_value(v: &Value, depth: u32) -> String {
    if let Value::Str(s) = v {
        return s.to_string();
    }

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
        Value::Str(_) => unreachable!("handled by early return"),
        Value::Record(rec) => {
            let shape = rec.shape();
            match shape.kind {
                crate::value::ShapeKind::Adt => {
                    if rec.field_count() == 0 {
                        shape.constructor.to_string()
                    } else {
                        let mut out = format!("{}(", shape.constructor);
                        for i in 0..rec.field_count() {
                            if i > 0 { out.push_str(", "); }
                            if let Some(name) = shape.field_names.get(i).and_then(|n| n.as_ref()) {
                                out.push_str(name);
                                out.push_str(": ");
                            }
                            out.push_str(&format_value(&rec.field(i), depth + 1));
                        }
                        out.push(')');
                        out
                    }
                }
                crate::value::ShapeKind::Newtype => format!(
                    "{}({})",
                    crate::sema::Sema::display_type_name(&shape.type_name),
                    format_value(&if rec.field_count() > 0 { rec.field(0) } else { Value::VOID }, depth + 1)
                ),
                crate::value::ShapeKind::Record => {
                    let mut out = format!("{}(", crate::sema::Sema::display_type_name(&shape.type_name));
                    for i in 0..rec.field_count() {
                        if i > 0 { out.push_str(", "); }
                        if let Some(name) = shape.field_names.get(i).and_then(|n| n.as_ref()) {
                            out.push_str(name);
                            out.push_str(": ");
                        }
                        out.push_str(&format_value(&rec.field(i), depth + 1));
                    }
                    out.push(')');
                    out
                }
            }
        }
        Value::Ref(r) => {
            // Heap object: match on HeapObj
            match &**r {
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
        Value::Str(_) => 16,
        Value::Record(rec) => (0..rec.field_count()).map(|i| value_size(&rec.field(i))).sum(),
        Value::Ref(r) => {
            match &**r {
                HeapObj::Array(_) => 16,
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
        Value::Str(_) => 8,
        Value::Record(rec) => (0..rec.field_count()).map(|i| value_alignment(&rec.field(i))).max().unwrap_or(1),
        Value::Ref(r) => {
            match &**r {
                HeapObj::Array(_) => 8,
                _ => 8,
            }
        }
    }
}
