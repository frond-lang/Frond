//! Compute.rs — compute_fn table module.
//!
//! Split out of Engine.rs, this module centralizes every `compute_fn` (a
//! build-time bound node computation function), including:
//! - Sentinel constants (THUNK_FRAME_ID / IO / CTOR / TYPE_NAME, etc.)
//! - reflect helpers + UTF-8 decode utilities
//! - compute_fn generation macros (read_node_inputs / impl_cmp_compute / impl_int_ops / impl_float_ops)
//! - All compute_fns (arithmetic / comparison / record / array / string / channel / async / closure, etc.)
//! - Synchronous execution helpers: force_lazy_value_sync / run_frame_sync / run_defers_sync / unwrap_cell
//!
//! The scheduler (Engine.rs) calls these indirectly via `graph.compute_fns[idx]`,
//! and `ir/Ir.rs`'s `build_compute_fn_table` references them through `super::Compute::`.

use super::Ir::*;
use crate::value::Value;
use crate::engine::{prepare_frame_nodes, switch_subgraph, notify_downstream};
use std::sync::OnceLock;

/// Caches environment-variable boolean flags so hot paths do not call `std::env::var`
/// (a `getenv` syscall plus a `String` allocation) on every invocation. The first call
/// reads the env var; subsequent calls return the cached `bool`.
#[inline]
fn env_flag(name: &str) -> bool {
    static FLAG_CALL: OnceLock<bool> = OnceLock::new();
    static FLAG_GATE: OnceLock<bool> = OnceLock::new();
    static FLAG_STALL: OnceLock<bool> = OnceLock::new();
    static FLAG_WB: OnceLock<bool> = OnceLock::new();
    match name {
        "KUZO_DEBUG_CALL" => *FLAG_CALL.get_or_init(|| std::env::var("KUZO_DEBUG_CALL").is_ok()),
        "KUZO_DEBUG_GATE" => *FLAG_GATE.get_or_init(|| std::env::var("KUZO_DEBUG_GATE").is_ok()),
        "KUZO_DEBUG_STALL" => *FLAG_STALL.get_or_init(|| std::env::var("KUZO_DEBUG_STALL").is_ok()),
        "KUZO_DEBUG_WB" => *FLAG_WB.get_or_init(|| std::env::var("KUZO_DEBUG_WB").is_ok()),
        _ => std::env::var(name).is_ok(),
    }
}

// =========================================================================
// Sentinel constants — centralized to avoid scattered magic numbers.
// =========================================================================

/// Sentinel `FrameId` used for thunk frames (does not participate in normal
/// allocation, avoiding conflicts with `alloc_frame_id`).
const THUNK_FRAME_ID: FrameId = FrameId(u32::MAX);
/// Sentinel `FrameId` used for the LoopBody fallback subframe (does not
/// participate in normal allocation).
const LOOPBODY_FALLBACK_FRAME_ID: FrameId = FrameId(u32::MAX - 1);

/// Return value (i32) for a successful IO write. Only used in the
/// `#[cfg(not(has_extern_c))]` fallback path.
#[cfg(not(has_extern_c))]
const IO_OK: i32 = 0;
/// Return value (i32) for a failed IO write. Only used in the
/// `#[cfg(not(has_extern_c))]` fallback path.
#[cfg(not(has_extern_c))]
const IO_ERR: i32 = -1;
/// Return value (i64) for UTF-8 decode failure / out-of-bounds access.
const UTF8_DECODE_ERR: i64 = -1;

/// `Result` variant constructor names (kept in sync with the stdlib `Result` type definition).
pub(crate) const CTOR_OK: &str = "Ok";
pub(crate) const CTOR_ERR: &str = "Error";
pub(crate) const CTOR_ERR_ALT: &str = "Err";

/// reflect type-name constants (single source of truth, shared by
/// `__reflect_type_name` / `compute_cast_to_str`).
const TYPE_NAME_NULL: &str = "null";
const TYPE_NAME_VOID: &str = "void";
const TYPE_NAME_STR: &str = "str";
const TYPE_NAME_ARRAY: &str = "array";
const TYPE_NAME_UNKNOWN: &str = "unknown";

// =========================================================================
// Runtime error construction — uniformly uses ErrorVal (isomorphic to Arena::alloc_error_val).
// =========================================================================

/// Constructs a runtime error value by wrapping an `ErrorValue` (the dedicated
/// error type) inside a `ThrowVal::Err`.
///
/// Uses the same `HeapObj::ErrorVal` representation as `ValueArena::alloc_error_val`,
/// eliminating the repeated hand-rolled `RecordValue` pattern across compute_fns.
/// compute_fns have no Arena access, so they construct `Value::ref_val` directly.
fn make_error_throw(type_name: &str, msg: &str) -> Value {
    use crate::value::{HeapObj, ErrorValue, ThrowValue, ThrowPayload};
    let err_val = Value::ref_val(HeapObj::ErrorVal(ErrorValue {
        type_name: type_name.to_string(),
        message: msg.to_string(),
        is_error_subtype: true,
    }));
    Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(err_val) }))
}

// =========================================================================
// reflect helpers — eliminate duplication between the FFI and fallback paths.
// =========================================================================

/// Returns the reflect kind number of a `Value` (ABI protocol: 0–12).
/// Single authoritative source shared by FFI and fallback paths to ensure consistency.
fn reflect_kind(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Void => 1,
        Value::Scalar(_, _) => 2,
        Value::Ref(r) => match &**r {
            crate::value::HeapObj::Str(_) => 3,
            crate::value::HeapObj::Array(_) => 4,
            crate::value::HeapObj::Record(_) => 5,
            crate::value::HeapObj::Adt(_) => 6,
            crate::value::HeapObj::Closure(_) => 7,
            crate::value::HeapObj::TraitVal(_) => 8,
            crate::value::HeapObj::ThrowVal(_) => 9,
            crate::value::HeapObj::ChannelVal(_) => 10,
            crate::value::HeapObj::AsyncVal(_) => 11,
            _ => 12,
        },
    }
}

/// Returns the reflect kind display name of a `Value` (single source of truth,
/// shared by FFI and fallback paths).
fn reflect_kind_str(v: &Value) -> &'static str {
    match v {
        Value::Null => "Null",
        Value::Void => "Void",
        Value::Scalar(_, _) => "Primitive",
        Value::Ref(r) => match &**r {
            crate::value::HeapObj::Str(_) => "Str",
            crate::value::HeapObj::Array(_) => "Array",
            crate::value::HeapObj::Record(_) => "Record",
            crate::value::HeapObj::Adt(_) => "Adt",
            crate::value::HeapObj::Newtype(_) => "Newtype",
            crate::value::HeapObj::Closure(_) => "Closure",
            crate::value::HeapObj::TraitVal(_) => "Trait",
            _ => "Ref",
        },
    }
}

/// Returns the type name of a `Value` (single source of truth, shared by FFI,
/// fallback, and `cast_to_str`).
fn reflect_type_name(v: &Value) -> String {
    match v {
        Value::Null => TYPE_NAME_NULL.to_string(),
        Value::Void => TYPE_NAME_VOID.to_string(),
        Value::Scalar(_, tag) => tag.type_name().to_string(),
        Value::Ref(r) => match &**r {
            crate::value::HeapObj::Str(_) => TYPE_NAME_STR.to_string(),
            crate::value::HeapObj::Array(_) => TYPE_NAME_ARRAY.to_string(),
            crate::value::HeapObj::Record(rec) => rec.type_name.clone(),
            crate::value::HeapObj::Adt(a) => a.type_name.clone(),
            crate::value::HeapObj::Newtype(n) => n.type_name.clone(),
            _ => TYPE_NAME_UNKNOWN.to_string(),
        },
    }
}

/// UTF-8 decode: decodes one codepoint starting at `bytes[offset]`.
/// Returns `Some((codepoint, consumed_bytes))` on success, or `None` on
/// failure (out of bounds or illegal leading byte).
/// Single implementation, eliminating duplication between FFI and fallback paths.
fn utf8_decode_at(bytes: &[u8], offset: usize) -> Option<(u32, usize)> {
    if offset >= bytes.len() {
        return None;
    }
    let c = bytes[offset];
    // ASCII (1 byte)
    if c < 0x80 {
        return Some((c as u32, 1));
    }
    // 2-byte sequence: 110xxxxx 10xxxxxx
    if (c & 0xE0) == 0xC0 {
        if offset + 1 >= bytes.len() {
            return None;
        }
        let cp = ((c as u32 & 0x1F) << 6) | (bytes[offset + 1] as u32 & 0x3F);
        return Some((cp, 2));
    }
    // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
    if (c & 0xF0) == 0xE0 {
        if offset + 2 >= bytes.len() {
            return None;
        }
        let cp = ((c as u32 & 0x0F) << 12)
            | ((bytes[offset + 1] as u32 & 0x3F) << 6)
            | (bytes[offset + 2] as u32 & 0x3F);
        return Some((cp, 3));
    }
    // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    if (c & 0xF8) == 0xF0 {
        if offset + 3 >= bytes.len() {
            return None;
        }
        let cp = ((c as u32 & 0x07) << 18)
            | ((bytes[offset + 1] as u32 & 0x3F) << 12)
            | ((bytes[offset + 2] as u32 & 0x3F) << 6)
            | (bytes[offset + 3] as u32 & 0x3F);
        return Some((cp, 4));
    }
    // Illegal leading byte
    None
}

/// Converts a `u32` codepoint to a `char`, falling back to U+0000 for invalid codepoints.
/// Single unified helper that eliminates three duplicate
/// `char::from_u32(x).unwrap_or('\0')` call sites.
#[inline]
pub fn char_from_u32_or_nul(u: u32) -> char {
    char::from_u32(u).unwrap_or('\0')
}

// =========================================================================
// compute_fn generation macros — batch-generate type-specialized compute functions.
// =========================================================================

/// Boilerplate macro for reading a node's inputs.
///
/// Every compute_fn starts by pulling the node and its inputs slice out of
/// `frame.graph`. These three lines are fully duplicated across 100+ compute_fns.
/// This macro eliminates that duplication.
///
/// Usage (inside a compute_fn body):
/// ```ignore
/// pub fn compute_foo(frame: &mut Frame, node: NodeId) -> Value {
///     read_node_inputs!(frame, node, graph, n, inputs);
///     let a = frame.get_value_by_global(inputs[0]).as_i32();
///     ...
/// }
/// ```
/// After expansion, the three bindings `graph`, `n`, and `inputs` are in scope.
/// `inputs` is tied to the lifetime of `graph` (an `Arc` clone of `frame.graph`).
macro_rules! read_node_inputs {
    ($frame:ident, $node:ident, $graph:ident, $n:ident, $inputs:ident) => {
        let $graph = $frame.graph.clone();
        let $n = $graph.node($node.0 as usize);
        let $inputs = $graph.inputs($n.inputs_offset, $n.input_count);
    };
}

/// Batch-generates comparison compute_fns (returns `bool`).
macro_rules! impl_cmp_compute {
    ($($name:ident: $op:tt for $acc:ident);* $(;)?) => {
        $(
            pub fn $name(frame: &mut Frame, node: NodeId) -> Value {
                read_node_inputs!(frame, node, graph, n, inputs);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::bool_val(a $op b)
            }
        )*
    };
}

// =========================================================================
// SIMD batch processing — batch evaluation inside compute_fns (decided
// autonomously by the EvalContext).
// =========================================================================

/// Batch-extracts binary-op inputs → SIMD/rayon batch eval → returns a list of
/// `(local NodeId, Value)`. Does not write `frame.value_table` and does not
/// notify downstream — the engine hot loop handles these via `NodeResult::Batch`.
macro_rules! compute_bin_batch_results {
    ($frame:expr, $graph:expr, $locals:expr, $ns:expr, $rust:ty, $ctor:ident, $acc:ident, $batch_fn:ident, $op:expr) => {{
        let n = $locals.len();
        let mut a: Vec<$rust> = Vec::with_capacity(n);
        let mut b: Vec<$rust> = Vec::with_capacity(n);
        for &lid in $locals.iter() {
            let gid = NodeId(lid.0 + $ns);
            let node = $graph.node(gid.0 as usize);
            let inp = $graph.inputs(node.inputs_offset, node.input_count);
            a.push($frame.get_value_by_global(inp[0]).$acc());
            b.push($frame.get_value_by_global(inp[1]).$acc());
        }
        let mut dst = vec![0 as $rust; n];
        crate::value::$batch_fn(&mut dst, &a, &b, $op);
        $locals.iter().zip(dst.iter())
            .map(|(&lid, &v)| (lid, Value::$ctor(v)))
            .collect::<Vec<_>>()
    }};
}

/// Batch-extracts comparison-op inputs → SIMD/rayon batch eval → returns a list
/// of `(local NodeId, bool Value)`. Does not write `frame.value_table` and does
/// not notify downstream — the engine hot loop handles these via `NodeResult::Batch`.
macro_rules! compute_cmp_batch_results {
    ($frame:expr, $graph:expr, $locals:expr, $ns:expr, $rust:ty, $acc:ident, $batch_fn:ident, $op:expr) => {{
        let n = $locals.len();
        let mut a: Vec<$rust> = Vec::with_capacity(n);
        let mut b: Vec<$rust> = Vec::with_capacity(n);
        for &lid in $locals.iter() {
            let gid = NodeId(lid.0 + $ns);
            let node = $graph.node(gid.0 as usize);
            let inp = $graph.inputs(node.inputs_offset, node.input_count);
            a.push($frame.get_value_by_global(inp[0]).$acc());
            b.push($frame.get_value_by_global(inp[1]).$acc());
        }
        let mut mask = vec![0u8; n];
        crate::value::$batch_fn(&mut mask, &a, &b, $op);
        $locals.iter().zip(mask.iter())
            .map(|(&lid, &m)| (lid, Value::bool_val(m != 0)))
            .collect::<Vec<_>>()
    }};
}

/// Batch-extracts unary-op inputs → SIMD/rayon batch eval → returns a list of
/// `(local NodeId, Value)`. Does not write `frame.value_table` and does not
/// notify downstream — the engine hot loop handles these via `NodeResult::Batch`.
macro_rules! compute_unary_batch_results {
    ($frame:expr, $graph:expr, $locals:expr, $ns:expr, $rust:ty, $ctor:ident, $acc:ident, $op:expr) => {{
        let n = $locals.len();
        let mut a: Vec<$rust> = Vec::with_capacity(n);
        for &lid in $locals.iter() {
            let gid = NodeId(lid.0 + $ns);
            let node = $graph.node(gid.0 as usize);
            let inp = $graph.inputs(node.inputs_offset, node.input_count);
            a.push($frame.get_value_by_global(inp[0]).$acc());
        }
        let mut dst = vec![0 as $rust; n];
        crate::value::batch_unaryop(&mut dst, &a, $op);
        $locals.iter().zip(dst.iter())
            .map(|(&lid, &v)| (lid, Value::$ctor(v)))
            .collect::<Vec<_>>()
    }};
}

/// SIMD batch processing: evaluates a group of same-type, same-op nodes in bulk.
///
/// Reads inputs from the frame, calls the SIMD batch-eval functions in `Value.rs`,
/// and returns a list of `(local NodeId, Value)`. Returns `None` for unsupported
/// types — the caller (the `wrap_fn!` macro) then falls back to per-node evaluation.
/// Does not write `frame.value_table` and does not notify downstream — the engine
/// hot loop handles these via `NodeResult::Batch`.
pub fn do_simd_batch(
    frame: &Frame,
    locals: &[NodeId],
    info: BatchInfo,
    node_start: u32,
) -> Option<Vec<(NodeId, Value)>> {
    use crate::value::{ValueTag, BinOp, CmpOp, UnaryOp};
    let _ = (BinOp::Add, CmpOp::Eq, UnaryOp::Neg); // suppress unused imports
    let graph = &frame.graph;

    if locals.is_empty() { return None; }

    match info {
        BatchInfo { tag, op: BatchOp::Bin(op) } => {
            match tag {
                ValueTag::I32 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i32, i32, as_i32, batch_binop_i32, op)),
                ValueTag::I64 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i64, i64, as_i64, batch_binop_i64, op)),
                ValueTag::F32 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, f32, f32, as_f32, batch_binop_f32, op)),
                ValueTag::F64 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, f64, f64, as_f64, batch_binop_f64, op)),
                ValueTag::I8 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i8, i8, as_i8, batch_binop, op)),
                ValueTag::I16 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i16, i16, as_i16, batch_binop, op)),
                ValueTag::U8 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u8, u8, as_u8, batch_binop, op)),
                ValueTag::U16 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u16, u16, as_u16, batch_binop, op)),
                ValueTag::U32 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u32, u32, as_u32, batch_binop, op)),
                ValueTag::U64 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u64, u64, as_u64, batch_binop, op)),
                ValueTag::I128 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i128, i128, as_i128, batch_binop, op)),
                ValueTag::U128 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u128, u128, as_u128, batch_binop, op)),
                ValueTag::Isize => Some(compute_bin_batch_results!(frame, graph, locals, node_start, isize, isize_val, as_isize, batch_binop, op)),
                ValueTag::Usize => Some(compute_bin_batch_results!(frame, graph, locals, node_start, usize, usize_val, as_usize, batch_binop, op)),
                _ => None, // F16/F128/Bool/Char → unsupported, fall back to single-node path
            }
        }
        BatchInfo { tag, op: BatchOp::Cmp(op) } => {
            match tag {
                ValueTag::F32 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, f32, as_f32, batch_cmp_f32, op)),
                ValueTag::F64 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, f64, as_f64, batch_cmp_f64, op)),
                ValueTag::I32 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i32, as_i32, batch_cmp, op)),
                ValueTag::I64 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i64, as_i64, batch_cmp, op)),
                ValueTag::I8 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i8, as_i8, batch_cmp, op)),
                ValueTag::I16 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i16, as_i16, batch_cmp, op)),
                ValueTag::U8 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u8, as_u8, batch_cmp, op)),
                ValueTag::U16 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u16, as_u16, batch_cmp, op)),
                ValueTag::U32 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u32, as_u32, batch_cmp, op)),
                ValueTag::U64 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u64, as_u64, batch_cmp, op)),
                ValueTag::I128 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i128, as_i128, batch_cmp, op)),
                ValueTag::U128 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u128, as_u128, batch_cmp, op)),
                ValueTag::Isize => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, isize, as_isize, batch_cmp, op)),
                ValueTag::Usize => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, usize, as_usize, batch_cmp, op)),
                _ => None, // F16/F128/Bool/Char → unsupported
            }
        }
        BatchInfo { tag, op: BatchOp::Unary(op) } => {
            match tag {
                ValueTag::I32 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i32, i32, as_i32, op)),
                ValueTag::I64 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i64, i64, as_i64, op)),
                ValueTag::I8 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i8, i8, as_i8, op)),
                ValueTag::I16 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i16, i16, as_i16, op)),
                ValueTag::U8 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u8, u8, as_u8, op)),
                ValueTag::U16 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u16, u16, as_u16, op)),
                ValueTag::U32 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u32, u32, as_u32, op)),
                ValueTag::U64 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u64, u64, as_u64, op)),
                ValueTag::I128 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i128, i128, as_i128, op)),
                ValueTag::U128 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u128, u128, as_u128, op)),
                ValueTag::Isize => Some(compute_unary_batch_results!(frame, graph, locals, node_start, isize, isize_val, as_isize, op)),
                ValueTag::Usize => Some(compute_unary_batch_results!(frame, graph, locals, node_start, usize, usize_val, as_usize, op)),
                _ => None, // F16/F128/F32/F64/Bool/Char → unsupported
            }
        }
    }
}

// =========================================================================
// compute_fns — actual compute functions (build-time bound function indices).
// =========================================================================

/// compute_fn: i32 less-than-or-equal comparison (`<=`).
pub fn compute_le_i32(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_i32();
    let b = frame.get_value_by_global(inputs[1]).as_i32();
    Value::bool_val(a <= b)
}

// ---- i32 comparisons (indices 8–12, 25; arithmetic/bitwise/unary generated by macro) ----

impl_cmp_compute! {
    compute_eq_i32: == for as_i32;
    compute_ne_i32: != for as_i32;
    compute_lt_i32: < for as_i32;
    compute_gt_i32: > for as_i32;
    compute_ge_i32: >= for as_i32;
}

// ---- i64 comparisons (indices 55–60; arithmetic/bitwise/unary generated by macro) ----

impl_cmp_compute! {
    compute_eq_i64: == for as_i64;
    compute_ne_i64: != for as_i64;
    compute_lt_i64: < for as_i64;
    compute_gt_i64: > for as_i64;
    compute_le_i64: <= for as_i64;
    compute_ge_i64: >= for as_i64;
}

// ---- i128 comparisons (indices 69–74; arithmetic/bitwise/unary generated by macro) ----
// The i128 path covers i128/u128 types and supports all integer-typed inputs via `as_int_i128`.

impl_cmp_compute! {
    compute_eq_i128: == for as_int_i128;
    compute_ne_i128: != for as_int_i128;
    compute_lt_i128: < for as_int_i128;
    compute_gt_i128: > for as_int_i128;
    compute_le_i128: <= for as_int_i128;
    compute_ge_i128: >= for as_int_i128;
}

// ---- Integer bitwise operations (indices 78–92) ----
// BitAnd/BitOr/BitXor for i32/i64/i128 families, Shl/Shr for i32/i64/i128 families.
// Read uniformly via `as_int_i128`; results are constructed with the target type.
// Note: the concrete bitwise compute_fns are generated by the `impl_int_ops` macro below.

// =========================================================================
// All primitive-type compute_fns (indices 92+): generated in full per type via the `paste` macro.
// =========================================================================
// Integers: 12 types × 12 ops = 144; floats: 4 types × 6 ops = 24; total = 168.
// Comparisons reuse the per-family shared versions (results are `bool`, inputs are
// read cross-type via `as_int_i128`/`as_float_f64`).
// Arithmetic/bitwise/unary are generated per concrete type, so results carry the
// correct tag and are truncated/wrapped at the type's width.
//
// Type spec table: (type name, Rust type, Value ctor, accessor, is_integer)
// Indices start at 92.

/// Generates the full set of compute_fns for a given integer type
/// (add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot).
///
/// Arithmetic logic reuses the pure arithmetic core in `Value.rs` (the `arith_*`
/// functions), shared by both the runtime and compile-time const-fold.
/// The compute_fn only handles Frame value fetches and Value wrapping; the
/// arithmetic itself has no Frame dependency.
macro_rules! impl_int_ops {
    ($ty:ident, $rust:ty, $ctor:ident, $acc:ident) => {
        pastey::paste! {
            pub fn [<compute_add_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_add_$ty>](a, b))
            }
            pub fn [<compute_sub_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_sub_$ty>](a, b))
            }
            pub fn [<compute_mul_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mul_$ty>](a, b))
            }
            pub fn [<compute_div_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                // Integer divide-by-zero returns 0 (checked semantics, implemented by `arith_div_$ty`).
                Value::$ctor(crate::value::[<arith_div_$ty>](a, b))
            }
            pub fn [<compute_mod_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mod_$ty>](a, b))
            }
            pub fn [<compute_bitand_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_bitand_$ty>](a, b))
            }
            pub fn [<compute_bitor_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_bitor_$ty>](a, b))
            }
            pub fn [<compute_bitxor_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_bitxor_$ty>](a, b))
            }
            pub fn [<compute_shl_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                // Shift amount is read as i32 (matching the original semantics); the pure function internally casts to u32.
                let shift = frame.get_value_by_global(inputs[1]).as_i32();
                Value::$ctor(crate::value::[<arith_shl_$ty>](a, shift))
            }
            pub fn [<compute_shr_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let shift = frame.get_value_by_global(inputs[1]).as_i32();
                Value::$ctor(crate::value::[<arith_shr_$ty>](a, shift))
            }
            pub fn [<compute_neg_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                Value::$ctor(crate::value::[<arith_neg_$ty>](a))
            }
            pub fn [<compute_bitnot_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                Value::$ctor(crate::value::[<arith_bitnot_$ty>](a))
            }
        }
    };
}

/// Generates the full set of compute_fns for a given float type (add/sub/mul/div/mod/neg).
///
/// Arithmetic logic reuses the pure arithmetic core in `Value.rs` (the `arith_*` functions).
macro_rules! impl_float_ops {
    ($ty:ident, $rust:ty, $ctor:ident, $acc:ident) => {
        pastey::paste! {
            pub fn [<compute_add_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_add_$ty>](a, b))
            }
            pub fn [<compute_sub_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_sub_$ty>](a, b))
            }
            pub fn [<compute_mul_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mul_$ty>](a, b))
            }
            pub fn [<compute_div_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_div_$ty>](a, b))
            }
            pub fn [<compute_mod_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mod_$ty>](a, b))
            }
            pub fn [<compute_neg_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                Value::$ctor(crate::value::[<arith_neg_$ty>](a))
            }
        }
    };
}

// Integer type expansion (12 types × 12 ops = 144 functions)
impl_int_ops!(i8,    i8,    i8,    as_i8);
impl_int_ops!(i16,   i16,   i16,   as_i16);
impl_int_ops!(i32,   i32,   i32,   as_i32);
impl_int_ops!(i64,   i64,   i64,   as_i64);
impl_int_ops!(i128,  i128,  i128,  as_i128);
impl_int_ops!(u8,    u8,    u8,    as_u8);
impl_int_ops!(u16,   u16,   u16,   as_u16);
impl_int_ops!(u32,   u32,   u32,   as_u32);
impl_int_ops!(u64,   u64,   u64,   as_u64);
impl_int_ops!(u128,  u128,  u128,  as_u128);
impl_int_ops!(isize, isize, isize_val, as_isize);
impl_int_ops!(usize, usize, usize_val, as_usize);

// Float type expansion (4 types × 6 ops = 24 functions)
impl_float_ops!(f16, F16, f16, as_f16);
impl_float_ops!(f32, f32, f32, as_f32);
impl_float_ops!(f64, f64, f64, as_f64);
impl_float_ops!(f128, F128, f128, as_f128);

// ---- f64 comparisons (indices 16–21; arithmetic/unary generated by macro) ----

impl_cmp_compute! {
    compute_eq_f64: == for as_f64;
    compute_ne_f64: != for as_f64;
    compute_lt_f64: < for as_f64;
    compute_gt_f64: > for as_f64;
    compute_le_f64: <= for as_f64;
    compute_ge_f64: >= for as_f64;
}

// ---- f128 comparisons (indices 302–307): IEEE 754 semantics, no precision loss via to_f64 ----
// F128's derived `PartialEq` is a bit-pattern comparison (so `NaN == NaN` is `true`),
// which cannot be used directly for IEEE semantics. Implemented manually here:
//   - NaN compared with anything: eq/lt/gt/le/ge → false, ne → true
//   - -0 == +0 (treated as equal when only the sign bit differs)
//   - Otherwise uses the `totalOrder` sort key (sign-aware bit-pattern)

/// F128 NaN test.
#[inline]
fn f128_is_nan(bits: u128) -> bool {
    (bits >> 112) & 0x7FFF == 0x7FFF && (bits & ((1u128 << 112) - 1)) != 0
}

/// F128 `totalOrder` sort key (for non-NaN values).
#[inline]
fn f128_sort_key(bits: u128) -> u128 {
    // Negative (sign=1): flip all bits → maps to [0, 0x7FFF...FFF].
    // Positive (sign=0): set the sign bit → maps to [0x8000...000, 0xFFFF...FFF].
    // This makes -0 < +0 (totalOrder semantics), -Inf < +Inf, etc.
    if (bits >> 127) != 0 { !bits } else { bits | (1u128 << 127) }
}

pub fn compute_eq_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let eq = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else {
        ab == bb || (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
    };
    Value::bool_val(eq)
}

pub fn compute_ne_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let ne = if f128_is_nan(ab) || f128_is_nan(bb) {
        true
    } else {
        ab != bb && ((ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF != 0)
    };
    Value::bool_val(ne)
}

pub fn compute_lt_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let lt = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else if (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0 {
        false // -0 == +0, not less than
    } else {
        f128_sort_key(ab) < f128_sort_key(bb)
    };
    Value::bool_val(lt)
}

pub fn compute_gt_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let gt = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else if (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0 {
        false // -0 == +0, not greater than
    } else {
        f128_sort_key(ab) > f128_sort_key(bb)
    };
    Value::bool_val(gt)
}

pub fn compute_le_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let le = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else {
        // -0 == +0 → le=true; otherwise totalOrder less-or-equal
        (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
            || f128_sort_key(ab) < f128_sort_key(bb)
    };
    Value::bool_val(le)
}

pub fn compute_ge_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let ge = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else {
        (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
            || f128_sort_key(ab) > f128_sort_key(bb)
    };
    Value::bool_val(ge)
}

// ---- bool logic (indices 22–24, 27) ----

/// compute_fn: bool AND (reuses the pure arithmetic core).
pub fn compute_and_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(crate::value::arith_and_bool(a, b))
}

/// compute_fn: bool OR (reuses the pure arithmetic core).
pub fn compute_or_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(crate::value::arith_or_bool(a, b))
}

/// compute_fn: bool NOT (unary, reuses the pure arithmetic core).
pub fn compute_not_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    Value::bool_val(crate::value::arith_not_bool(a))
}

/// compute_fn: bool equality.
pub fn compute_eq_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(a == b)
}

/// compute_fn: bool inequality (symmetric with `eq_bool`).
pub fn compute_ne_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(a != b)
}

// ---- throw wrapping (index 28, no try-catch) ----

/// compute_fn: wraps a value as `ThrowVal(Err)` (used by `throw` statements).
///
/// Kuzo has no try-catch; `throw` produces a `ThrowVal(Err)` plus a `Return`
/// signal that propagates up to the top level. The Err payload holds the thrown
/// value itself (before Bug #27 was fixed, the original type was wrapped as an
/// `Error(value:v)` record, requiring `Error(Error(v))` nested destructuring).
/// - Input is a `ThrowVal` (already a thrown value) → returned directly (idempotent).
/// - Any other value (scalar/Str/Record/Adt/Array) → wrapped directly as `ThrowVal(Err(v))`.
pub fn compute_throw_wrap_err(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    use crate::value::{HeapObj, ThrowValue, ThrowPayload};
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    // Already a ThrowVal → re-throw directly (idempotent, supports re-throw).
    if let Some(HeapObj::ThrowVal(_)) = v.heap_obj() {
        return NodeResult::Return(v);
    }
    // Any value becomes the Err payload directly (the original type is no longer wrapped as an Error record).
    let throw_val = Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(v) }));
    NodeResult::Return(throw_val)
}

/// compute_fn: wraps a value as `ThrowVal(Ok(val))` (used by the `Ok` constructor).
pub fn compute_throw_ok(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ThrowValue, ThrowPayload};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Ok(val) }))
}

/// compute_fn: wraps a value as `ThrowVal(Err(v))` (used by the `Err` constructor).
///
/// The input is typically the result of a `record_construct` node (Record/Adt),
/// but the `Err` constructor treats any value type uniformly: it wraps it
/// directly as `ThrowVal(Err(v))`. Consistent with `compute_throw_wrap_err`, it
/// no longer wraps the original type as `Error(value:v)` (Bug #27).
pub fn compute_throw_err(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ThrowValue, ThrowPayload};
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(v) }))
}

/// compute_fn (idx 47): the `?` operator (Propagate).
///
/// Input is a `ThrowVal`:
/// - `Ok(val)` → returns `NodeResult::Value(val)` (unwrapped).
/// - `Err(err)` → returns `NodeResult::Return(ThrowVal(Err))`, causing the
///   function to return early with the error.
///
/// Input is a Nullable value:
/// - `null` → returns `NodeResult::Return(null)`, causing the function to
///   return early with null.
/// - non-null → returns `NodeResult::Value(v)` (nullable and non-null values
///   share the same representation, so the value is passed through directly).
pub fn compute_propagate(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);

    if let Some(crate::value::HeapObj::ThrowVal(tv)) = v.heap_obj() {
        match &tv.payload {
            crate::value::ThrowPayload::Ok(val) => NodeResult::Value(val.clone()),
            crate::value::ThrowPayload::Err(_) => {
                // Error propagation: return Return carrying the original ThrowVal(Err), which propagates up the call stack.
                NodeResult::Return(v.clone())
            }
        }
    } else if v.is_null() {
        // Nullable propagation: when the value is null, return Return carrying null to exit early.
        NodeResult::Return(v.clone())
    } else {
        // Non-null Nullable value: pass through directly.
        NodeResult::Value(v)
    }
}

/// compute_fn (idx 46): `@extern("C")` FFI call.
///
/// Looks up the function name from the node's `ffi_call_names` metadata,
/// collects argument values from the inputs, dispatches to the corresponding
/// `Ffi::wrapper` function, and returns the resulting `Value`.
/// FFI calls are synchronous: they set no `pending_call` and do not suspend the frame.
#[cfg(has_extern_c)]
pub fn compute_ffi_call(frame: &mut Frame, node: NodeId) -> Value {
    use crate::ffi::Ffi::wrapper;
    read_node_inputs!(frame, node, graph, n, inputs);
    let fn_name = graph.ffi_call_name(node.0 as usize)
        .expect("compute_ffi_call: no ffi_call_name");

    // Extract a str argument from a Value (HeapObj::Str → owned String, avoiding temporary Value lifetime issues).
    fn extract_str(v: &Value) -> String {
        match v.heap_obj() {
            Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
            _ => panic!("FFI str arg expected, got non-str value"),
        }
    }

    // Extract a u8[] argument from a Value (unified through ArrayValue::collect_u8_bytes).
    fn extract_u8_buf(v: &Value) -> Vec<u8> {
        match v.heap_obj() {
            Some(crate::value::HeapObj::Array(arr)) => arr.collect_u8_bytes(),
            _ => panic!("FFI u8[] arg expected"),
        }
    }

    // Writes data read by FFI back into a Kuzo u8[] array (in-place mutation of
    // the underlying HeapObj, with `&self` semantics).
    //
    // `extract_u8_buf` returns a cloned `Vec`; after FFI writes into the local
    // Vec, it must be written back to the original array, otherwise the Kuzo
    // side reads zeros (H-2 fix).
    // Only `buf[0..n]` is written back, where `n` is the FFI return value
    // (provided by the caller).
    fn writeback_u8_buf(buf_val: &Value, data: &[u8], n: usize) {
        if let Value::Ref(arc) = buf_val {
            // Safety: the engine executes on a single thread; the caller frame is
            // Suspended while the callee runs, so there is no concurrent access to
            // the same HeapObj (consistent with `compute_record_field_set`).
            let ptr = std::sync::Arc::as_ptr(arc) as *mut crate::value::HeapObj;
            unsafe {
                if let crate::value::HeapObj::Array(arr) = &mut *ptr {
                    let len = n.min(data.len()).min(arr.elements.len());
                    // SOA fast path: U8 contiguous storage, direct memcpy.
                    if let Some(crate::value::ScalarSoA::U8(ref mut soa_data)) = arr.scalar_soa {
                        let len = len.min(soa_data.len());
                        soa_data[..len].copy_from_slice(&data[..len]);
                    } else {
                        for i in 0..len {
                            arr.elements[i] = Value::u8(data[i]);
                        }
                    }
                }
            }
        }
    }


    match fn_name {
        // ── IO: stdout/stderr ──
        "__stdout_write_raw" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let rc = unsafe { wrapper::__stdout_write_raw(&s) };
            Value::i32(rc)
        }
        "__stderr_write_raw" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let rc = unsafe { wrapper::__stderr_write_raw(&s) };
            Value::i32(rc)
        }

        // ── IO: file ops ──
        "__file_open_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let flags = frame.get_value_by_global(inputs[1]).as_i32();
            let mode = frame.get_value_by_global(inputs[2]).as_i32();
            let fd = unsafe { wrapper::__file_open_raw(&path, flags, mode) };
            Value::i64(fd)
        }
        "__file_close_raw" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let rc = unsafe { wrapper::__file_close_raw(fd) };
            Value::i32(rc)
        }
        "__file_seek_raw" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let offset = frame.get_value_by_global(inputs[1]).as_i64();
            let whence = frame.get_value_by_global(inputs[2]).as_i32();
            let pos = unsafe { wrapper::__file_seek_raw(fd, offset, whence) };
            Value::i64(pos)
        }
        "__file_remove_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let rc = unsafe { wrapper::__file_remove_raw(&path) };
            Value::i32(rc)
        }
        "__file_rename_raw" => {
            let old = extract_str(&frame.get_value_by_global(inputs[0]));
            let new = extract_str(&frame.get_value_by_global(inputs[1]));
            let rc = unsafe { wrapper::__file_rename_raw(&old, &new) };
            Value::i32(rc)
        }
        "__file_chmod_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let mode = frame.get_value_by_global(inputs[1]).as_i32();
            let rc = unsafe { wrapper::__file_chmod_raw(&path, mode) };
            Value::i32(rc)
        }

        // ── IO: file read/write (u8[] + len) ──
        "__file_read_into" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__file_read_into(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }
        "__file_write" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[1]));
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__file_write(fd, &buf, len) };
            Value::i64(n)
        }

        // ── IO: stdin ──
        "__stdin_readln_into" => {
            let buf_val = frame.get_value_by_global(inputs[0]);
            let mut buf = extract_u8_buf(&buf_val);
            let n = unsafe { wrapper::__stdin_readln_into(&mut buf) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }

        // ── IO: stat/fstat ──
        "__file_stat_into" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let rc = unsafe { wrapper::__file_stat_into(&path, &mut buf) };
            if rc == 0 { writeback_u8_buf(&buf_val, &buf, buf.len()); }
            Value::i32(rc)
        }
        "__file_fstat_into" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let rc = unsafe { wrapper::__file_fstat_into(fd, &mut buf) };
            if rc == 0 { writeback_u8_buf(&buf_val, &buf, buf.len()); }
            Value::i32(rc)
        }

        // ── IO: dir ops ──
        "__dir_create_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let recursive = frame.get_value_by_global(inputs[1]).as_bool();
            let rc = unsafe { wrapper::__dir_create_raw(&path, recursive) };
            Value::i32(rc)
        }
        "__dir_remove_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let recursive = frame.get_value_by_global(inputs[1]).as_bool();
            let rc = unsafe { wrapper::__dir_remove_raw(&path, recursive) };
            Value::i32(rc)
        }

        // ── IO: dir list ──
        "__dir_list_into" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let names_val = frame.get_value_by_global(inputs[1]);
            let offsets_val = frame.get_value_by_global(inputs[2]);
            let kinds_val = frame.get_value_by_global(inputs[3]);
            let mut names_buf = extract_u8_buf(&names_val);
            let mut name_offsets = extract_u8_buf(&offsets_val);
            let mut kinds_buf = extract_u8_buf(&kinds_val);
            let max_count = frame.get_value_by_global(inputs[4]).as_usize();
            let count = unsafe {
                wrapper::__dir_list_into(&path, &mut names_buf, &mut name_offsets, &mut kinds_buf, max_count)
            };
            if count > 0 {
                writeback_u8_buf(&names_val, &names_buf, names_buf.len());
                writeback_u8_buf(&offsets_val, &name_offsets, name_offsets.len());
                writeback_u8_buf(&kinds_val, &kinds_buf, kinds_buf.len());
            }
            Value::i64(count)
        }

        // ── net: tcp ──
        "__net_tcp_connect_v4" => {
            let ip_bits = frame.get_value_by_global(inputs[0]).as_u32();
            let port = frame.get_value_by_global(inputs[1]).as_u16();
            let timeout_ns = frame.get_value_by_global(inputs[2]).as_i64();
            let fd = unsafe { wrapper::__net_tcp_connect_v4(ip_bits, port, timeout_ns) };
            Value::i64(fd)
        }
        "__net_tcp_listen_v4" => {
            let ip_bits = frame.get_value_by_global(inputs[0]).as_u32();
            let port = frame.get_value_by_global(inputs[1]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[2]).as_bool();
            let fd = unsafe {
                wrapper::__net_tcp_listen_v4(ip_bits, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_tcp_accept" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let conn_fd = unsafe { wrapper::__net_tcp_accept(fd) };
            Value::i64(conn_fd)
        }
        "__net_tcp_read" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_tcp_read(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }
        "__net_tcp_write" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[1]));
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_tcp_write(fd, &buf, len) };
            Value::i64(n)
        }
        "__net_tcp_close_raw" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let rc = unsafe { wrapper::__net_tcp_close_raw(fd) };
            Value::i32(rc)
        }

        // ── net: udp v4 ──
        "__net_udp_bind_v4" => {
            let ip_bits = frame.get_value_by_global(inputs[0]).as_u32();
            let port = frame.get_value_by_global(inputs[1]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[2]).as_bool();
            let fd = unsafe {
                wrapper::__net_udp_bind_v4(ip_bits, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_udp_send_to_v4" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let ip_bits = frame.get_value_by_global(inputs[1]).as_u32();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[3]));
            let len = frame.get_value_by_global(inputs[4]).as_usize();
            let n = unsafe { wrapper::__net_udp_send_to_v4(fd, ip_bits, port, &buf, len) };
            Value::i64(n)
        }
        "__net_udp_recv_from_v4" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_udp_recv_from_v4(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }

        // ── net: resolve ──
        "__net_resolve_into" => {
            let host = extract_str(&frame.get_value_by_global(inputs[0]));
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut out_buf = extract_u8_buf(&buf_val);
            let max_count = frame.get_value_by_global(inputs[2]).as_usize();
            let count = unsafe { wrapper::__net_resolve_into(&host, &mut out_buf, max_count) };
            if count > 0 { writeback_u8_buf(&buf_val, &out_buf, out_buf.len()); }
            Value::i64(count)
        }

        // ── net: tcp/udp v6 ──
        "__net_tcp_connect_v6" => {
            let ip_hi = frame.get_value_by_global(inputs[0]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[1]).as_u64();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let timeout_ns = frame.get_value_by_global(inputs[3]).as_i64();
            let fd = unsafe { wrapper::__net_tcp_connect_v6(ip_hi, ip_lo, port, timeout_ns) };
            Value::i64(fd)
        }
        "__net_tcp_listen_v6" => {
            let ip_hi = frame.get_value_by_global(inputs[0]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[1]).as_u64();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[3]).as_bool();
            let fd = unsafe {
                wrapper::__net_tcp_listen_v6(ip_hi, ip_lo, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_udp_bind_v6" => {
            let ip_hi = frame.get_value_by_global(inputs[0]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[1]).as_u64();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[3]).as_bool();
            let fd = unsafe {
                wrapper::__net_udp_bind_v6(ip_hi, ip_lo, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_udp_send_to_v6" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let ip_hi = frame.get_value_by_global(inputs[1]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[2]).as_u64();
            let port = frame.get_value_by_global(inputs[3]).as_u16();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[4]));
            let len = frame.get_value_by_global(inputs[5]).as_usize();
            let n = unsafe { wrapper::__net_udp_send_to_v6(fd, ip_hi, ip_lo, port, &buf, len) };
            Value::i64(n)
        }
        "__net_udp_recv_from_v6" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_udp_recv_from_v6(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }

        // ── time ──
        "__instant_now_ns" => {
            let ns = unsafe { wrapper::__instant_now_ns() };
            Value::i64(ns)
        }
        "__systemtime_now_ns" => {
            let ns = unsafe { wrapper::__systemtime_now_ns() };
            Value::i64(ns)
        }
        "__sleep_ns" => {
            let ns = frame.get_value_by_global(inputs[0]).as_i64();
            unsafe { wrapper::__sleep_ns(ns) };
            Value::VOID
        }
        "__localtime_offset_minutes" => {
            let minutes = unsafe { wrapper::__localtime_offset_minutes() };
            Value::i32(minutes)
        }

        // ── cast: widening to i128 ──
        "__cast_i8_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i8();
            let r = unsafe { wrapper::__cast_i8_to_i128(x) };
            Value::i128(r)
        }
        "__cast_i16_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i16();
            let r = unsafe { wrapper::__cast_i16_to_i128(x) };
            Value::i128(r)
        }
        "__cast_i32_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i32();
            let r = unsafe { wrapper::__cast_i32_to_i128(x) };
            Value::i128(r)
        }
        "__cast_i64_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i64();
            let r = unsafe { wrapper::__cast_i64_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u8_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u8();
            let r = unsafe { wrapper::__cast_u8_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u16_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u16();
            let r = unsafe { wrapper::__cast_u16_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u32_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u32();
            let r = unsafe { wrapper::__cast_u32_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u64_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u64();
            let r = unsafe { wrapper::__cast_u64_to_i128(x) };
            Value::i128(r)
        }
        "__cast_usize_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_usize();
            let r = unsafe { wrapper::__cast_usize_to_i128(x) };
            Value::i128(r)
        }

        // ── cast: narrowing from i128 ──
        "__cast_i128_to_i8" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i8(v) };
            Value::i8(r)
        }
        "__cast_i128_to_i16" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i16(v) };
            Value::i16(r)
        }
        "__cast_i128_to_i32" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i32(v) };
            Value::i32(r)
        }
        "__cast_i128_to_i64" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i64(v) };
            Value::i64(r)
        }
        "__cast_i128_to_u8" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u8(v) };
            Value::u8(r)
        }
        "__cast_i128_to_u16" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u16(v) };
            Value::u16(r)
        }
        "__cast_i128_to_u32" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u32(v) };
            Value::u32(r)
        }
        "__cast_i128_to_u64" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u64(v) };
            Value::u64(r)
        }
        "__cast_i128_to_usize" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_usize(v) };
            Value::usize_val(r)
        }

        // ── cast: char ──
        "__cast_char_to_u8" => {
            let x = frame.get_value_by_global(inputs[0]).as_u32();
            let r = unsafe { wrapper::__cast_char_to_u8(char_from_u32_or_nul(x)) };
            Value::u8(r)
        }

        // ── reflect: __reflect_format/__reflect_scalar_to_str have been split into
        // standalone compute_fns (CF_REFLECT_FORMAT/CF_REFLECT_SCALAR_TO_STR, idx 290/291),
        // and no longer go through the FFI dispatch path ──
        "__reflect_kind" => {
            let v = frame.get_value_by_global(inputs[0]);
            Value::u8(reflect_kind(&v))
        }
        "__reflect_type_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = reflect_type_name(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_array_len" => {
            let v = frame.get_value_by_global(inputs[0]);
            match v.heap_obj() {
                Some(crate::value::HeapObj::Array(arr)) => Value::usize_val(arr.elements.len()),
                _ => Value::usize_val(0),
            }
        }
        "__reflect_field_count" => {
            let v = frame.get_value_by_global(inputs[0]);
            let count: u16 = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => rec.fields.len() as u16,
                Some(crate::value::HeapObj::Adt(a)) => a.fields.len() as u16,
                _ => 0,
            };
            Value::u16(count)
        }
        "__reflect_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u8 = match &v {
                Value::Scalar(_, tag) => tag.byte_width() as u8,
                _ => 0,
            };
            Value::u8(size)
        }
        "__reflect_field_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => {
                    rec.field_names.get(i as usize)
                        .and_then(|n| n.as_ref())
                        .cloned()
                        .unwrap_or_default()
                }
                Some(crate::value::HeapObj::Adt(a)) => {
                    a.fields.get(i as usize)
                        .and_then(|f| f.name.as_ref().cloned())
                        .unwrap_or_default()
                }
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_field_value" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => {
                    rec.fields.get(i as usize).cloned().unwrap_or(Value::NULL)
                }
                Some(crate::value::HeapObj::Adt(a)) => {
                    a.fields.get(i as usize).map(|f| f.value.clone()).unwrap_or(Value::NULL)
                }
                _ => Value::NULL,
            }
        }
        "__reflect_adt_constructor" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Adt(a)) => a.constructor.clone(),
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_kind_str" => {
            let v = frame.get_value_by_global(inputs[0]);
            let kind = reflect_kind_str(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(kind)))
        }
        "__reflect_layout_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u32 = crate::value::reflect_layout_size(&v);
            Value::u32(size)
        }
        "__reflect_layout_alignment" => {
            let v = frame.get_value_by_global(inputs[0]);
            let align: u32 = crate::value::reflect_layout_alignment(&v);
            Value::u32(align)
        }

        // ── str: UTF-8 per-character decode (pure Rust bit operations, matching the C implementation's semantics) ──
        "__str_utf8_decode_at" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            match utf8_decode_at(bytes, offset) {
                Some((cp, _)) => Value::i64(cp as i64),
                None => Value::i64(UTF8_DECODE_ERR),
            }
        }
        "__str_utf8_char_len_at" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            if offset >= bytes.len() {
                Value::usize_val(0)
            } else {
                let c = bytes[offset];
                let len = if c < 0x80 { 1 }
                    else if (c & 0xE0) == 0xC0 { 2 }
                    else if (c & 0xF0) == 0xE0 { 3 }
                    else if (c & 0xF8) == 0xF0 { 4 }
                    else { 1 };
                Value::usize_val(len)
            }
        }

        // ── Unimplemented FFI functions ──
        other => panic!("compute_ffi_call: unimplemented FFI function '{}'", other),
    }
}

/// compute_fn (idx 46) fallback: when `has_extern_c` is not set (no C compiler),
/// computes the FFI functions implementable in pure Rust (casts) directly in
/// Rust; all others return a default value.
#[cfg(not(has_extern_c))]
pub fn compute_ffi_call(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let fn_name = graph.ffi_call_name(node.0 as usize)
        .expect("compute_ffi_call: no ffi_call_name");

    match fn_name {
        // ── IO: implemented directly via Rust std ──
        "__stdout_write_raw" => {
            if let Some(crate::value::HeapObj::Str(s)) = frame.get_value_by_global(inputs[0]).heap_obj() {
                use std::io::Write;
                let _ = std::io::stdout().write_all(s.bytes().as_bytes());
                let _ = std::io::stdout().flush();
                Value::i32(IO_OK)
            } else {
                Value::i32(IO_ERR)
            }
        }
        "__stderr_write_raw" => {
            if let Some(crate::value::HeapObj::Str(s)) = frame.get_value_by_global(inputs[0]).heap_obj() {
                use std::io::Write;
                let _ = std::io::stderr().write_all(s.bytes().as_bytes());
                Value::i32(IO_OK)
            } else {
                Value::i32(IO_ERR)
            }
        }

        // ── time: implemented directly via Rust std ──
        "__instant_now_ns" => {
            let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
            Value::i64(ns as i64)
        }
        "__systemtime_now_ns" => {
            let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
            Value::i64(ns as i64)
        }
        "__sleep_ns" => {
            let ns = frame.get_value_by_global(inputs[0]).as_i64();
            std::thread::sleep(std::time::Duration::from_nanos(ns as u64));
            Value::VOID
        }
        "__localtime_offset_minutes" => {
            Value::i32(0)
        }

        // ── cast: widening to i128 (pure Rust computation) ──
        "__cast_i8_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i8() as i128),
        "__cast_i16_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i16() as i128),
        "__cast_i32_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i32() as i128),
        "__cast_i64_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i64() as i128),
        "__cast_u8_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u8() as i128),
        "__cast_u16_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u16() as i128),
        "__cast_u32_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u32() as i128),
        "__cast_u64_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u64() as i128),
        "__cast_usize_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_usize() as i128),

        // ── cast: narrowing from i128 (pure Rust computation) ──
        "__cast_i128_to_i8" => Value::i8(frame.get_value_by_global(inputs[0]).as_i128() as i8),
        "__cast_i128_to_i16" => Value::i16(frame.get_value_by_global(inputs[0]).as_i128() as i16),
        "__cast_i128_to_i32" => Value::i32(frame.get_value_by_global(inputs[0]).as_i128() as i32),
        "__cast_i128_to_i64" => Value::i64(frame.get_value_by_global(inputs[0]).as_i128() as i64),
        "__cast_i128_to_u8" => Value::u8(frame.get_value_by_global(inputs[0]).as_i128() as u8),
        "__cast_i128_to_u16" => Value::u16(frame.get_value_by_global(inputs[0]).as_i128() as u16),
        "__cast_i128_to_u32" => Value::u32(frame.get_value_by_global(inputs[0]).as_i128() as u32),
        "__cast_i128_to_u64" => Value::u64(frame.get_value_by_global(inputs[0]).as_i128() as u64),
        "__cast_i128_to_usize" => Value::usize_val(frame.get_value_by_global(inputs[0]).as_i128() as usize),

        // ── cast: char ──
        "__cast_char_to_u8" => Value::u8(frame.get_value_by_global(inputs[0]).as_u32() as u8),

        // ── reflect: __reflect_format/__reflect_scalar_to_str have been split into
        // standalone compute_fns (CF_REFLECT_FORMAT/CF_REFLECT_SCALAR_TO_STR, idx 290/291),
        // and no longer go through the FFI dispatch path ──
        "__reflect_kind" => {
            let v = frame.get_value_by_global(inputs[0]);
            Value::u8(reflect_kind(&v))
        }
        "__reflect_type_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = reflect_type_name(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_array_len" => {
            let v = frame.get_value_by_global(inputs[0]);
            match v.heap_obj() {
                Some(crate::value::HeapObj::Array(arr)) => Value::usize_val(arr.elements.len()),
                _ => Value::usize_val(0),
            }
        }
        "__reflect_field_count" => {
            let v = frame.get_value_by_global(inputs[0]);
            let count: u16 = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => rec.fields.len() as u16,
                Some(crate::value::HeapObj::Adt(a)) => a.fields.len() as u16,
                _ => 0,
            };
            Value::u16(count)
        }
        "__reflect_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u8 = match &v {
                Value::Scalar(_, tag) => tag.byte_width() as u8,
                _ => 0,
            };
            Value::u8(size)
        }
        "__reflect_field_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => {
                    rec.field_names.get(i as usize).and_then(|n| n.as_ref()).cloned().unwrap_or_default()
                }
                Some(crate::value::HeapObj::Adt(a)) => {
                    a.fields.get(i as usize).and_then(|f| f.name.as_ref().cloned()).unwrap_or_default()
                }
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_field_value" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => rec.fields.get(i as usize).cloned().unwrap_or(Value::NULL),
                Some(crate::value::HeapObj::Adt(a)) => a.fields.get(i as usize).map(|f| f.value.clone()).unwrap_or(Value::NULL),
                _ => Value::NULL,
            }
        }
        "__reflect_adt_constructor" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Adt(a)) => a.constructor.clone(),
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_kind_str" => {
            let v = frame.get_value_by_global(inputs[0]);
            let kind = reflect_kind_str(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(kind)))
        }
        "__reflect_layout_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u32 = crate::value::reflect_layout_size(&v);
            Value::u32(size)
        }
        "__reflect_layout_alignment" => {
            let v = frame.get_value_by_global(inputs[0]);
            let align: u32 = crate::value::reflect_layout_alignment(&v);
            Value::u32(align)
        }

        // ── str: UTF-8 per-character decode (pure Rust bit operations, matching the C implementation's semantics) ──
        "__str_utf8_decode_at" => {
            let s = match frame.get_value_by_global(inputs[0]).heap_obj() {
                Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
                _ => String::new(),
            };
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            match utf8_decode_at(bytes, offset) {
                Some((cp, _)) => Value::i64(cp as i64),
                None => Value::i64(UTF8_DECODE_ERR),
            }
        }
        "__str_utf8_char_len_at" => {
            let s = match frame.get_value_by_global(inputs[0]).heap_obj() {
                Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
                _ => String::new(),
            };
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            if offset >= bytes.len() {
                Value::usize_val(0)
            } else {
                let c = bytes[offset];
                let len = if c < 0x80 { 1 }
                    else if (c & 0xE0) == 0xC0 { 2 }
                    else if (c & 0xF0) == 0xE0 { 3 }
                    else if (c & 0xF8) == 0xF0 { 4 }
                    else { 1 };
                Value::usize_val(len)
            }
        }

        // ── Unimplemented FFI functions (return a default value when no C compiler is available) ──
        _ => Value::i32(0),
    }
}

// =========================================================================
// Standalone reflect compute_fns (290–291)
//
// Split out of `compute_ffi_call` to avoid coupling lazy-force logic with FFI
// dispatch. These are the only reflect operations that involve forcing a
// LazyValue; once separated:
//   - they no longer depend on `ffi_call_name` metadata,
//   - they do not go through the FFI dispatch path, and
//   - the lazy-force logic is colocated with the reflect formatting logic.
// =========================================================================

/// compute_fn (idx 290): `__reflect_format` — any value → str.
///
/// Before formatting, forces evaluation of the LazyValue (if the input is
/// lazy), then calls `Reflect::format_value`. Does not depend on
/// `ffi_call_name`; reads `inputs[0]` directly.
pub fn compute_reflect_format(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, _n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    let v = force_lazy_value_sync(frame, &v);
    let s = crate::value::format_value(&v, 0);
    Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&s)))
}

/// compute_fn (idx 291): `__reflect_scalar_to_str` — scalar value → str.
///
/// Semantically identical to `compute_reflect_format` (both go through
/// `format_value`); kept separate to match the two distinct `@extern("C")`
/// primitive declarations in `Raw.kz`.
pub fn compute_reflect_scalar_to_str(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, _n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    let v = force_lazy_value_sync(frame, &v);
    let s = crate::value::format_value(&v, 0);
    Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&s)))
}

/// compute_fn: type construction (collects field values from inputs and builds
/// a Record/Adt/Newtype HeapObj based on `kind`).
pub fn compute_record_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::ir::Ir::{RecordLitKind, RecordLitInfo};
    use crate::value::{AdtField, AdtValue, HeapObj, NewtypeValue, RecordValue, ValueArena};
    read_node_inputs!(frame, node, graph, n, inputs);
    let fields: Vec<Value> = inputs
        .iter()
        .map(|&input_node| frame.get_value_by_global(input_node))
        .collect();
    let info = graph.record_lit_info_at(node.0 as usize);
    let info: &RecordLitInfo = info
        .as_ref()
        .expect("record construct node has no RecordLitInfo");
    match info.kind {
        RecordLitKind::Record => {
            Value::ref_val(HeapObj::Record(RecordValue {
                type_name: info.type_name.clone(),
                fields,
                field_names: info.field_names.clone(),
                field_ref_bits: 0,
            }))
        }
        RecordLitKind::Adt => {
            let adt_fields: Vec<AdtField> = fields
                .into_iter()
                .enumerate()
                .map(|(i, v)| AdtField {
                    name: info.field_names.get(i).and_then(|n| n.clone()),
                    value: v,
                })
                .collect();
            Value::ref_val(HeapObj::Adt(AdtValue {
                type_name: info.type_name.clone(),
                constructor: info.constructor.clone(),
                fields: adt_fields,
                field_ref_bits: 0,
            }))
        }
        RecordLitKind::Newtype => {
            // Newtype: single field; store the inner Value in the global arena to obtain a ValueHandle.
            let inner_val = fields.into_iter().next().unwrap_or(Value::VOID);
            let inner = ValueArena::with_global_mut(|a| a.alloc_value(&inner_val));
            Value::ref_val(HeapObj::Newtype(NewtypeValue {
                type_name: info.type_name.clone(),
                inner,
            }))
        }
    }
}

/// compute_fn: record field access (fetches a field value from a Record/Adt by field name).
///
/// Unified mechanism: both Record and Adt use `find_field(name)` for name-based
/// lookup, independent of the compile-time `field_idx`. This eliminates the idx
/// fallback and any Record/Adt path divergence.
pub fn compute_record_field_get(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let record_val = frame.get_value_by_global(inputs[0]);
    let name = graph.field_set_name(node.0 as usize);
    let make_err = |msg: &str| make_error_throw("FieldError", msg);
    let Some(h) = record_val.heap_obj() else {
        return make_err("field access on non-record value");
    };
    let Some(name) = name else {
        return make_err("field_get node has no field name");
    };
    h.field_get(name).unwrap_or_else(|| {
        make_err(&format!("no such field '{}' on record", name))
    })
}

/// compute_fn: array construction (collects elements from inputs and builds an ArrayValue).
pub fn compute_array_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let elements: Vec<Value> = inputs
        .iter()
        .map(|&input_node| frame.get_value_by_global(input_node))
        .collect();
    Value::ref_val(HeapObj::Array(ArrayValue::new(elements)))
}

/// compute_fn: stack-allocated record construction (288).
///
/// Used at allocation sites the analyzer marks as non-escaping.
/// The current implementation is identical to `compute_record_construct`
/// (under the Value model, `Arc` is the only reference mechanism), and is kept
/// as a separation point: once the Value model supports frame-local
/// allocation, this function can switch to genuine stack allocation.
pub fn compute_record_construct_stack(frame: &mut Frame, node: NodeId) -> Value {
    compute_record_construct(frame, node)
}

/// compute_fn: stack-allocated array construction (289).
///
/// Used at allocation sites the analyzer marks as non-escaping.
/// The current implementation is identical to `compute_array_construct`, kept
/// as a separation point.
pub fn compute_array_construct_stack(frame: &mut Frame, node: NodeId) -> Value {
    compute_array_construct(frame, node)
}

/// compute_fn: array indexing (fetches an element from an ArrayValue by i32 index).
/// Returns a `ThrowVal(Err)` error value on out-of-bounds access, which
/// propagates up to the top level.
pub fn compute_array_index(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let recv_val = frame.get_value_by_global(inputs[0]);
    let idx = frame.get_value_by_global(inputs[1]).as_i32() as usize;
    let make_err = |msg: &str| make_error_throw("IndexError", msg);
    match recv_val.heap_obj() {
        Some(crate::value::HeapObj::Array(arr)) => {
            arr.get(idx).cloned().unwrap_or_else(|| {
                make_err(&format!("index {} out of bounds (len {})", idx, arr.len()))
            })
        }
        Some(crate::value::HeapObj::Str(s)) => {
            s.char_at(idx).map(|c| Value::char_val(c)).unwrap_or_else(|| {
                make_err(&format!("index {} out of bounds (len {})", idx, s.codepoint_count()))
            })
        }
        _ => make_err("index on non-indexable type"),
    }
}

/// compute_fn: slicing `recv[start..end]` / `recv[start..=end]`.
///
/// Three inputs: recv, start, end. The `inclusive` flag is read from
/// `graph.slice_inclusive[node]`.
/// - str: sliced by codepoint index, returns a new str.
/// - array: sliced by element index, returns a new array.
/// Out-of-bounds indices are clamped to `[0, len]`, matching Rust slice
/// semantics (no panic).
pub fn compute_slice(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue, KuzoStr};
    read_node_inputs!(frame, node, graph, n, inputs);
    let recv_val = frame.get_value_by_global(inputs[0]);
    let start = frame.get_value_by_global(inputs[1]).as_usize();
    let mut end = frame.get_value_by_global(inputs[2]).as_usize();
    let inclusive = graph.slice_inclusive(node.0 as usize);
    if inclusive {
        end = end.saturating_add(1);
    }
    let make_err = |msg: &str| make_error_throw("SliceError", msg);
    match recv_val.heap_obj() {
        Some(crate::value::HeapObj::Array(arr)) => {
            let len = arr.len();
            let s = start.min(len);
            let e = end.min(len);
            if s > e {
                return make_err(&format!("slice start {} > end {}", s, e));
            }
            let sliced: Vec<Value> = arr.elements[s..e].to_vec();
            Value::ref_val(HeapObj::Array(ArrayValue {
                elements: sliced,
                fixed_size: None,
                elem_is_ref: arr.elem_is_ref,
                scalar_soa: None,
            }))
        }
        Some(crate::value::HeapObj::Str(s)) => {
            // Slice by codepoint index: collect chars in [start, end) and reassemble into a str.
            let chars: Vec<char> = s.bytes().chars().collect();
            let len = chars.len();
            let st = start.min(len);
            let en = end.min(len);
            if st > en {
                return make_err(&format!("slice start {} > end {}", st, en));
            }
            let mut buf = String::with_capacity(en - st);
            for c in &chars[st..en] {
                buf.push(*c);
            }
            Value::ref_val(HeapObj::Str(KuzoStr::new(buf)))
        }
        _ => make_err("slice on non-sliceable type"),
    }
}

/// compute_fn: string concatenation `lhs + rhs` (both sides must be str).
///
/// Two inputs: lhs, rhs. Returns an error value if either side is not a str.
pub fn compute_str_concat(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::HeapObj;
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let make_err = |msg: &str| make_error_throw("TypeError", msg);
    match (lhs.heap_obj(), rhs.heap_obj()) {
        (Some(HeapObj::Str(a)), Some(HeapObj::Str(b))) => {
            Value::ref_val(HeapObj::Str(a.concat(b)))
        }
        _ => make_err("str concat on non-str operand"),
    }
}

/// compute_fn (idx 270): global variable read.
///
/// No inputs; reads the value from `graph.global_var_storage[slot]`.
/// The slot index is obtained from `graph.global_load_slots[node]`.
/// Global variables do not depend on the frame chain, so any function can read them correctly.
pub fn compute_global_load(frame: &mut Frame, node: NodeId) -> Value {
    let slot = frame.graph.global_load_slot(node.0 as usize)
        .expect("global_load node has no slot");
    let storage = &frame.graph.global_var_storage;
    let guard = storage[slot as usize].lock().unwrap();
    let val = guard.clone().unwrap_or(Value::NULL);
    val
}

/// compute_fn (idx 271): global variable write.
///
/// `inputs[0]` is the value-source node; the value is written to
/// `graph.global_var_storage[slot]`. The slot index is obtained from
/// `graph.global_store_slots[node]`. Returns the written value (for downstream
/// chained use).
pub fn compute_global_store(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let slot = graph.global_store_slot(node.0 as usize)
        .expect("global_store node has no slot");
    let storage = &frame.graph.global_var_storage;
    *storage[slot as usize].lock().unwrap() = Some(val.clone());
    val
}

/// compute_fn (idx 308): memoization cache lookup.
///
/// `inputs[0..param_count]` are the parameter values (used as the cache key).
/// `MemoInfo.table_index` indexes into `graph.memo_tables`'s hash table.
/// Returns a Record `{hit: bool, value: Value}`:
/// - hit: `hit=true`, `value=cached value`
/// - miss: `hit=false`, `value=Void`
pub fn compute_memo_check(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, RecordValue};
    use std::hash::{Hash, Hasher};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.memo_info(node.0 as usize)
        .expect("memo_check node has no MemoInfo");
    let param_count = info.param_count as usize;
    // Build the cache key: hash the parameter values into a u64.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let param_vals: Vec<Value> = inputs[..param_count].iter()
        .map(|&inp| frame.get_value_by_global(inp))
        .collect();
    if std::env::var("KUZO_DEBUG_MEMO").is_ok() {
        eprintln!("[MEMO_CHECK] table={} params={:?}", info.table_index, param_vals);
    }
    for val in &param_vals {
        val.hash(&mut hasher);
    }
    let key = hasher.finish();
    // Look up the cache table.
    let table = &frame.graph.memo_tables;
    let hit_val = {
        let guard = table[info.table_index as usize].lock().unwrap();
        guard.get(&key).cloned()
    };
    if std::env::var("KUZO_DEBUG_MEMO").is_ok() {
        eprintln!("[MEMO_CHECK] key={} hit={}", key, hit_val.is_some());
    }
    match hit_val {
        Some(cached) => {
            // Hit: return record(hit=true, value=cached).
            Value::ref_val(HeapObj::Record(RecordValue {
                type_name: String::new(),
                fields: vec![Value::bool_val(true), cached],
                field_names: vec![Some("hit".into()), Some("value".into())],
                field_ref_bits: 0,
            }))
        }
        None => {
            // Miss: return record(hit=false, value=void).
            Value::ref_val(HeapObj::Record(RecordValue {
                type_name: String::new(),
                fields: vec![Value::bool_val(false), Value::VOID],
                field_names: vec![Some("hit".into()), Some("value".into())],
                field_ref_bits: 0,
            }))
        }
    }
}

/// compute_fn (idx 309): memoization cache write.
///
/// `inputs[0..param_count]` are the parameter values (used as the cache key),
/// and `inputs[param_count]` is the result value. Writes the result into the
/// cache table and then forwards it (for downstream use).
pub fn compute_memo_store(frame: &mut Frame, node: NodeId) -> Value {
    use std::hash::{Hash, Hasher};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.memo_info(node.0 as usize)
        .expect("memo_store node has no MemoInfo");
    let param_count = info.param_count as usize;
    let result_val = frame.get_value_by_global(inputs[param_count]);
    // Build the cache key.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let param_vals: Vec<Value> = inputs[..param_count].iter()
        .map(|&inp| frame.get_value_by_global(inp))
        .collect();
    for val in &param_vals {
        val.hash(&mut hasher);
    }
    let key = hasher.finish();
    if std::env::var("KUZO_DEBUG_MEMO").is_ok() {
        eprintln!("[MEMO_STORE] table={} key={} params={:?} result={:?}",
            info.table_index, key, param_vals, result_val);
    }
    // Write to the cache table.
    let table = &frame.graph.memo_tables;
    {
        let mut guard = table[info.table_index as usize].lock().unwrap();
        guard.insert(key, result_val.clone());
    }
    result_val
}

/// compute_fn (idx 272): record extension.
///
/// `inputs[0]` is the base RecordValue; `inputs[1..]` are the updated field
/// values. `RecordExtendInfo.update_names` gives the field names corresponding
/// to `inputs[1..]`. Clones the base's fields and field names, then either
/// replaces same-named fields or appends new ones per `update_names`, building a
/// new RecordValue (preserving the base's `type_name`).
pub fn compute_record_extend(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, RecordValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.record_extend_info_at(node.0 as usize);
    let info = info
        .as_ref()
        .expect("record extend node has no RecordExtendInfo");

    // Take the base RecordValue.
    let base_val = frame.get_value_by_global(inputs[0]);
    let base_record: RecordValue = match base_val.heap_obj() {
        Some(HeapObj::Record(r)) => r.clone(),
        _ => {
            // base not a record: degrade to an empty record; all update fields are appended as new fields.
            RecordValue::new(String::new(), Vec::new(), Vec::new())
        }
    };

    // Collect the update values (inputs[1..], in `update_names` order).
    let update_values: Vec<Value> = inputs[1..]
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();

    // Clone the base fields and field names, then replace/append per `update_names`.
    let mut fields: Vec<Value> = base_record.fields.clone();
    let mut field_names: Vec<Option<String>> = base_record.field_names.clone();
    for (i, update_name) in info.update_names.iter().enumerate() {
        let update_val = update_values[i].clone();
        // Find the position of a same-named field.
        let pos = field_names.iter().position(|n| n.as_deref() == Some(update_name));
        match pos {
            Some(idx) => {
                // Replace the existing field value.
                fields[idx] = update_val;
            }
            None => {
                // Append a new field.
                fields.push(update_val);
                field_names.push(Some(update_name.clone()));
            }
        }
    }

    Value::ref_val(HeapObj::Record(RecordValue {
        type_name: base_record.type_name.clone(),
        fields,
        field_names,
        field_ref_bits: 0,
    }))
}

/// compute_fn (idx 273): atomic construction.
///
/// `inputs[0]` is the initial-value node; it is wrapped in an `AtomicValue`
/// (an atomic container sharing the underlying memory). `AtomicValue.data` is a
/// Value, so this compute_fn can construct it without an arena.
pub fn compute_atomic_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, AtomicValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    Value::ref_val(HeapObj::AtomicVal(AtomicValue::new(val)))
}

/// compute_fn: pattern match — constructor name discrimination (idx 274).
///
/// Input: scrutinee. Metadata: constructor name (`graph.pattern_ctor_names`).
/// Checks whether the scrutinee is an ADT whose `constructor` matches, or a
/// Record whose `type_name` matches, or a ThrowVal whose constructor name is
/// "Ok"/"Error" matching the corresponding payload variant.
/// Returns `bool`.
pub fn compute_pattern_ctor_match(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let ctor_name = graph.pattern_ctor_name(node.0 as usize)
        .expect("pattern ctor match node has no ctor name");
    let matched = match val.heap_obj() {
        Some(crate::value::HeapObj::Adt(a)) => a.constructor == ctor_name,
        Some(crate::value::HeapObj::Record(r)) => r.type_name == ctor_name,
        // Newtype: constructor name == type name; match `NewtypeValue.type_name`.
        Some(crate::value::HeapObj::Newtype(n)) => n.type_name == ctor_name,
        Some(crate::value::HeapObj::ThrowVal(tv)) => match &tv.payload {
            crate::value::ThrowPayload::Ok(_) => ctor_name == CTOR_OK,
            crate::value::ThrowPayload::Err(_) => ctor_name == CTOR_ERR || ctor_name == CTOR_ERR_ALT,
        },
        _ => false,
    };
    Value::bool_val(matched)
}

/// compute_fn: pattern match — positional field extraction from ADT/Record/ThrowVal (idx 275).
///
/// Input: scrutinee. Metadata: field index (`graph.pattern_field_indices`).
/// Fetches a field value from an ADT by position, or from a Record by position,
/// or extracts the inner value from a ThrowVal (index 0: Ok's `val` or Err's
/// `record`). Returns the field value (out-of-bounds returns Void).
pub fn compute_pattern_adt_field_get(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let idx = graph.pattern_field_index(node.0 as usize)
        .expect("pattern adt field get node has no field index")
        as usize;
    match val.heap_obj() {
        Some(crate::value::HeapObj::Adt(a)) => {
            a.fields.get(idx).map(|f| f.value.clone()).unwrap_or(Value::VOID)
        }
        Some(crate::value::HeapObj::Record(r)) => {
            r.fields.get(idx).cloned().unwrap_or(Value::VOID)
        }
        // Newtype: single field; idx 0 fetches the inner value (via the ValueArena global handle dereference).
        Some(crate::value::HeapObj::Newtype(n)) => {
            if idx == 0 {
                crate::value::ValueArena::with_global(|a| a.get_value(n.inner))
            } else {
                Value::VOID
            }
        }
        Some(crate::value::HeapObj::ThrowVal(tv)) => {
            if idx == 0 {
                match &tv.payload {
                    crate::value::ThrowPayload::Ok(v) => v.clone(),
                    // Err holds the thrown value itself (Bug #27); the `v` in
                    // the `Error(v)` match pattern binds directly to the thrown
                    // value, no `Error(Error(v))` nested destructuring needed.
                    crate::value::ThrowPayload::Err(v) => v.clone(),
                }
            } else {
                Value::VOID
            }
        }
        _ => Value::VOID,
    }
}

/// compute_fn: pattern match — string equality discrimination (idx 276).
///
/// Inputs: scrutinee, str_const. Compares whether the two values are equal strings.
/// Returns `bool`.
pub fn compute_pattern_str_eq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let lhs_str = match lhs.heap_obj() {
        Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
        _ => return Value::bool_val(false),
    };
    let rhs_str = match rhs.heap_obj() {
        Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
        _ => return Value::bool_val(false),
    };
    Value::bool_val(lhs_str == rhs_str)
}

/// compute_fn: str comparisons (292–297).
///
/// Compared lexicographically by Unicode codepoint sequence (Rust `str`'s `Ord`
/// semantics; UTF-8 byte order matches codepoint order).
/// Returns `false` when an operand is not a str (for Eq/Le/Ge), or returns
/// `false` without panicking under `Ord` semantics.
/// Uses `KuzoStr.compare` (`Ordering`) to avoid redundant allocations.
fn str_compare_operands(frame: &mut Frame, node: NodeId) -> Option<std::cmp::Ordering> {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    match (lhs.heap_obj(), rhs.heap_obj()) {
        (Some(crate::value::HeapObj::Str(a)), Some(crate::value::HeapObj::Str(b))) => {
            Some(a.compare(b))
        }
        _ => None,
    }
}

pub fn compute_eq_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) == Some(std::cmp::Ordering::Equal))
}

pub fn compute_ne_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) != Some(std::cmp::Ordering::Equal))
}

pub fn compute_lt_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) == Some(std::cmp::Ordering::Less))
}

pub fn compute_gt_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) == Some(std::cmp::Ordering::Greater))
}

pub fn compute_le_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(matches!(str_compare_operands(frame, node), Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)))
}

pub fn compute_ge_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(matches!(str_compare_operands(frame, node), Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)))
}

/// compute_fn: generic type conversion — any value → str (idx 277).
///
/// Input: source-value node. Dispatches formatting per `Value` variant to
/// produce a KuzoStr:
///   - scalar integer → `as_int_i128().to_string()`
///   - scalar float → `as_float_f64().to_string()`
///   - bool → "true"/"false"
///   - char → `String::from(char)`
///   - Str → clone (identity)
///   - Null → "null"
///   - Void → "void"
///   - other Ref → "<non-scalar>"
pub fn compute_cast_to_str(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, KuzoStr, ValueTag};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);

    let s: String = match &val {
        Value::Null => TYPE_NAME_NULL.to_string(),
        Value::Void => TYPE_NAME_VOID.to_string(),
        Value::Scalar(_, tag) => {
            match tag {
                ValueTag::Bool => val.as_bool().to_string(),
                ValueTag::Char => {
                    let c = val.as_char();
                    let mut buf = [0u8; 4];
                    c.encode_utf8(&mut buf).to_string()
                }
                ValueTag::F16 | ValueTag::F32 | ValueTag::F64 | ValueTag::F128 => {
                    val.as_float_f64().to_string()
                }
                // All integer types.
                _ => val.as_int_i128().to_string(),
            }
        }
        Value::Ref(r) => match r.as_ref() {
            HeapObj::Str(kuzo_str) => kuzo_str.bytes().to_string(),
            _ => "<non-scalar>".to_string(),
        },
    };
    Value::ref_val(HeapObj::Str(KuzoStr::new(s)))
}

/// compute_fn: generic type conversion — scalar → scalar (idx 278).
///
/// Input: source-value node. Metadata: target type name
/// (`graph.cast_target_types`). Covers all scalar-to-scalar conversions:
/// int↔int (truncate/extend), int↔float, float↔float, bool→int, char→int.
/// The target type is read from the `cast_target_types` metadata and the
/// corresponding Value is constructed by dispatching on `ValueTag`.
pub fn compute_cast_scalar(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::ValueTag;
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let target_ty = graph.cast_target_type(node.0 as usize)
        .expect("cast_scalar node has no target type");

    let target_tag = match ValueTag::from_name(target_ty) {
        Some(tag) => tag,
        // Unknown target type: safe cast returns Null, otherwise returns Void.
        None => {
            return if graph.safe_op_flag(node.0 as usize) {
                Value::Null
            } else {
                Value::VOID
            };
        }
    };

    // Whether the source value is a float.
    let src_is_float = matches!(
        &val,
        Value::Scalar(_, ValueTag::F16 | ValueTag::F32 | ValueTag::F64 | ValueTag::F128)
    );
    // Read the source value uniformly as f64: floats use `as_float_f64`, integers use `as_int_i128 as f64`.
    let src_f64 = if src_is_float { val.as_float_f64() } else { val.as_int_i128() as f64 };

    match target_tag {
        ValueTag::I8 => Value::i8(if src_is_float { src_f64 as i8 } else { val.as_i8() }),
        ValueTag::I16 => Value::i16(if src_is_float { src_f64 as i16 } else { val.as_i16() }),
        ValueTag::I32 => Value::i32(if src_is_float { src_f64 as i32 } else { val.as_i32() }),
        ValueTag::I64 => Value::i64(if src_is_float { src_f64 as i64 } else { val.as_i64() }),
        ValueTag::I128 => Value::i128(if src_is_float { src_f64 as i128 } else { val.as_i128() }),
        ValueTag::U8 => Value::u8(if src_is_float { src_f64 as u8 } else { val.as_u8() }),
        ValueTag::U16 => Value::u16(if src_is_float { src_f64 as u16 } else { val.as_u16() }),
        ValueTag::U32 => Value::u32(if src_is_float { src_f64 as u32 } else { val.as_u32() }),
        ValueTag::U64 => Value::u64(if src_is_float { src_f64 as u64 } else { val.as_u64() }),
        ValueTag::U128 => Value::u128(if src_is_float { src_f64 as u128 } else { val.as_u128() }),
        ValueTag::Isize => Value::isize_val(if src_is_float { src_f64 as isize } else { val.as_isize() }),
        ValueTag::Usize => Value::usize_val(if src_is_float { src_f64 as usize } else { val.as_usize() }),
        ValueTag::F16 => Value::f16(crate::value::F16::from_f64(src_f64)),
        ValueTag::F32 => Value::f32(src_f64 as f32),
        ValueTag::F64 => Value::f64(src_f64),
        // Use the precise `as_f128()` accessor: integer sources go through from_i128/from_u128, float sources go through to_f64 (already precisely rounded).
        ValueTag::F128 => Value::f128(val.as_f128()),
        ValueTag::Bool => Value::bool_val(if src_is_float { src_f64 != 0.0 } else { val.as_int_i128() != 0 }),
        ValueTag::Char => Value::char_val(char_from_u32_or_nul(if src_is_float { src_f64 as u32 } else { val.as_int_i128() as u32 })),
        _ => unreachable!("non-scalar target_tag {:?} in cast", target_tag),
    }
}

/// compute_fn (idx 279): non-null assertion `expr!`.
///
/// Input is a nullable value: `Null` → panic (a programming error, not a
/// recoverable flow); non-Null → returned as-is (Scalar/Ref pass-through, i.e.
/// unwrapping the nullable).
pub fn compute_non_null_assert(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    if v.is_null() {
        panic!("non-null assertion failed: value is null");
    }
    v
}

/// compute_fn (idx 280): take a reference `&expr` (RefOf).
///
/// Wraps the input value in an `Arc<HeapObj::Cell>` and returns
/// `Value::Ref(arc)`. Multiple references share the same Cell (via Arc clone),
/// so writes are visible to all of them. For values that are already a Ref
/// (records, etc.), the same Arc is shared directly (no second wrapping needed).
pub fn compute_ref_of(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    match &v {
        // Scalar/Null/Void → wrap in a Cell.
        Value::Scalar(_, _) | Value::Null | Value::Void => {
            let cell = crate::value::Cell::new(v.clone());
            Value::ref_val(crate::value::HeapObj::Cell(cell))
        }
        // Already a heap reference: share the Arc directly (reference semantics, no deep copy).
        Value::Ref(_) => v,
    }
}

/// compute_fn (idx 281): dereference read `*ref` (Deref).
///
/// Input is an `Arc<HeapObj::Cell>`: returns the value inside the Cell.
/// Input is any other Ref (record/array, etc.): returned as-is (`&rec` shares
/// the Arc, so `*r` is just `rec` itself).
pub fn compute_deref_read(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    match v.heap_obj() {
        Some(crate::value::HeapObj::Cell(c)) => c.get(),
        _ => v,
    }
}

/// compute_fn (idx 282): dereference write `*ref = value` (DerefAssign).
///
/// `inputs[0]` is the reference (Cell); `inputs[1]` is the new value. Writes
/// the new value into the Cell and returns the written value (for chained use).
/// Non-Cell references (a record's shared Arc) are left untouched (record field
/// writes go through `record_field_set`).
pub fn compute_deref_write(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let ref_val = frame.get_value_by_global(inputs[0]);
    let new_val = frame.get_value_by_global(inputs[1]);
    if let Some(crate::value::HeapObj::Cell(c)) = ref_val.heap_obj() {
        c.set(new_val.clone());
    }
    new_val
}


/// compute_fn: record field assignment (in-place mutation of a RecordValue's
/// field; returns void).
///
/// `inputs[0]` is the record-value node; `inputs[1]` is the new value. The
/// field name is obtained from `graph.field_set_names[node]` and mutated in
/// place via `Arc::make_mut`. After mutation the value is written back to the
/// value-table slot so the change is visible to other nodes.
pub fn compute_record_field_set(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let new_value = frame.get_value_by_global(inputs[1]);
    let field_name = graph.field_set_name(node.0 as usize)
        .expect("field set node has no field name");
    let record_node_local = NodeId(inputs[0].0.wrapping_sub(frame.node_offset));
    // &self semantics: mutate the Arc's underlying HeapObj directly, so the
    // change is visible to all owners. This is critical for iterator-style
    // patterns (next() mutating self.pos): the for loop passes the iterator
    // reference via tail recursion; if COW kicked in, pos would never update →
    // infinite loop.
    //
    // Arc::make_mut would COW when refcount > 1, breaking &self reference
    // semantics. Here we obtain a mutable pointer via Arc::as_ptr and mutate
    // directly, bypassing Rust's aliasing rules.
    //
    // Safety: the engine executes single-threaded (LockStrategy::Single is
    // lock-free; Multi is mutually exclusive at the frame level). The caller
    // frame is Suspended while the callee runs, so there is no concurrent
    // access to the same HeapObj. The Arc's refcount is unchanged (no clone or
    // drop), only the heap data is mutated.
    if let Some(val) = frame.value_table.get_value_mut(record_node_local.0 as usize) {
        if let Value::Ref(arc) = val {
            let ptr = std::sync::Arc::as_ptr(arc) as *mut crate::value::HeapObj;
            unsafe {
                match &mut *ptr {
                    crate::value::HeapObj::Record(r) => {
                        if let Some(idx) = r.field_names.iter().position(|n| n.as_deref() == Some(field_name)) {
                            if idx < r.fields.len() {
                                r.fields[idx] = new_value.clone();
                            }
                        }
                    }
                    crate::value::HeapObj::Adt(a) => {
                        if let Some(idx) = a.fields.iter().position(|f| f.name.as_deref() == Some(field_name)) {
                            a.fields[idx].value = new_value.clone();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Value::VOID
}

/// compute_fn (idx 301): array index store `arr[i] = x`.
///
/// Three inputs: arr, index, value. Mutates the Array heap object's `elements`
/// vector in place. Same semantics as `record_field_set`: mutates the heap data
/// directly via `Arc::as_ptr` to preserve `&self` reference semantics (the
/// change is visible to all owners).
///
/// Safety: the engine executes single-threaded; the caller frame is Suspended
/// while the callee runs, so there is no concurrent access. Out-of-bounds
/// indices grow the array to `idx + 1` (padding with Void), matching dynamic-array semantics.
pub fn compute_array_store(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let idx = frame.get_value_by_global(inputs[1]).as_usize();
    let new_value = frame.get_value_by_global(inputs[2]);

    let arr_node_local = NodeId(inputs[0].0.wrapping_sub(frame.node_offset));
    if let Some(val) = frame.value_table.get_value_mut(arr_node_local.0 as usize) {
        if let Value::Ref(arc) = val {
            let ptr = std::sync::Arc::as_ptr(arc) as *mut crate::value::HeapObj;
            unsafe {
                if let crate::value::HeapObj::Array(arr) = &mut *ptr {
                    if idx >= arr.elements.len() {
                        arr.elements.resize(idx + 1, Value::VOID);
                        // SOA layout must be rebuilt after resize (new elements are padded with Void; SOA cannot simply extend).
                        arr.scalar_soa = None;
                    }
                    arr.elements[idx] = new_value.clone();
                    // Sync the SOA: if the type matches, write in place; otherwise invalidate the SOA cache.
                    if let Some(ref mut soa) = arr.scalar_soa {
                        if !soa.try_store(idx, &new_value) {
                            arr.scalar_soa = None;
                        }
                    }
                }
            }
        }
    }
    Value::VOID
}

/// compute_fn: null check (checks whether a value is null; returns `bool`).
pub fn compute_is_null(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let is_null = val.is_null();
    Value::bool_val(is_null)
}

/// compute_fn: length (returns i32, matching the default integer-arithmetic type).
/// - Array: element count.
/// - Str: Unicode codepoint count (consistent with `str[i]` indexing, both
///   counted by codepoint).
pub fn compute_array_len(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let len = match val.heap_obj() {
        Some(crate::value::HeapObj::Array(arr)) => arr.len() as i32,
        Some(crate::value::HeapObj::Str(s)) => s.codepoint_count() as i32,
        _ => 0,
    };
    Value::i32(len)
}

/// compute_fn: reference equality comparison (`===`), checks whether two Refs'
/// Arc pointers refer to the same object. Returns `bool`. Uses `Arc::ptr_eq`
/// when both sides are Refs; otherwise returns `false`.
pub fn compute_ref_eq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let eq = match (&lhs, &rhs) {
        (Value::Ref(a), Value::Ref(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    };
    Value::bool_val(eq)
}

/// compute_fn: reference inequality comparison (`!==`), the negation of RefEq.
pub fn compute_ref_neq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let neq = match (&lhs, &rhs) {
        (Value::Ref(a), Value::Ref(b)) => !std::sync::Arc::ptr_eq(a, b),
        _ => true,
    };
    Value::bool_val(neq)
}

/// compute_fn: semantic equality for composite types (record/adt/newtype/array/closure/throw, etc.).
/// Refs go through `heap_equals` for deep comparison; scalars/Null/Void fall
/// back to `value_equals`.
pub fn compute_eq_obj(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let eq = crate::value::ValueArena::with_global(|arena| {
        crate::value::value_equals_with_arena(&lhs, &rhs, arena)
    });
    Value::bool_val(eq)
}

/// compute_fn: semantic inequality for composite types; the negation of `compute_eq_obj`.
pub fn compute_ne_obj(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let neq = crate::value::ValueArena::with_global(|arena| {
        !crate::value::value_equals_with_arena(&lhs, &rhs, arena)
    });
    Value::bool_val(neq)
}

/// compute_fn: list concatenation (ConcatList) — concatenates two Arrays into a new Array.
pub fn compute_concat_list(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    match (lhs.heap_obj(), rhs.heap_obj()) {
        (Some(HeapObj::Array(a)), Some(HeapObj::Array(b))) => {
            let mut elements = Vec::with_capacity(a.len() + b.len());
            elements.extend(a.elements.iter().cloned());
            elements.extend(b.elements.iter().cloned());
            Value::ref_val(HeapObj::Array(ArrayValue::new(elements)))
        }
        _ => Value::VOID,
    }
}

/// compute_fn: range generation (Range, `a..b`, half-open).
pub fn compute_range(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Range};
    read_node_inputs!(frame, node, graph, n, inputs);
    let start = frame.get_value_by_global(inputs[0]).as_i64();
    let end = frame.get_value_by_global(inputs[1]).as_i64();
    Value::ref_val(HeapObj::Range(Range::new(start, end, false)))
}

/// compute_fn: range generation (RangeInclusive, `a..=b`, closed).
pub fn compute_range_inclusive(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Range};
    read_node_inputs!(frame, node, graph, n, inputs);
    let start = frame.get_value_by_global(inputs[0]).as_i64();
    let end = frame.get_value_by_global(inputs[1]).as_i64();
    Value::ref_val(HeapObj::Range(Range::new(start, end, true)))
}

/// compute_fn: Elvis operation (`lhs ?: rhs`).
///
/// Uniformly handles Nullable and Throw, the two "potentially-absent value"
/// types (Bug #28):
/// - `ThrowVal(Ok(v))` → returns `v` (unwraps the success value).
/// - `ThrowVal(Err(_))` → returns `rhs` (default value on error).
/// - `null` (Nullable) → returns `rhs`.
/// - any other non-null value → returns `lhs`.
pub fn compute_elvis(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    // Throw type: Ok unwraps, Err uses the default value.
    if let Some(crate::value::HeapObj::ThrowVal(tv)) = lhs.heap_obj() {
        return match &tv.payload {
            crate::value::ThrowPayload::Ok(v) => v.clone(),
            crate::value::ThrowPayload::Err(_) => frame.get_value_by_global(inputs[1]),
        };
    }
    // Nullable type: null uses the default value; non-null returns lhs.
    if lhs.is_null() {
        frame.get_value_by_global(inputs[1])
    } else {
        lhs
    }
}

/// compute_fn: Call node launches a subgraph (collects arguments + sets
/// `frame.pending_call`).
///
/// Unifies the sync/async call path: `is_async` is derived from
/// `target_sg.has_suspend`, and the core loop decides whether to spawn a
/// subframe + suspend the current frame after observing `pending_call`.
/// Does not call `start_subgraph` directly (compute_fns have no Engine reference).
pub fn compute_call_launch(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    let call_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));

    // safe_op short-circuit: `?.method(args)` returns Null when the receiver is null, without invoking the call.
    if graph.safe_op_flag(node.0 as usize) {
        let n = graph.node(node.0 as usize);
        if n.input_count > 0 {
            let inputs = graph.inputs(n.inputs_offset, n.input_count);
            let recv = frame.get_value_by_global(inputs[0]);
            if recv.is_null() {
                return NodeResult::Value(Value::Null);
            }
        }
    }

    // Static binding: has call_target → collect args + return NodeResult::Call.
    if let Some(target_sg) = graph.call_target(node.0 as usize) {
        if env_flag("KUZO_DEBUG_CALL") {
            eprintln!("[CALL] node={:?} target_sg={} frame.sg={} frame.offset={}",
                node, target_sg.0, frame.subgraph_id.0, frame.node_offset);
        }
        let is_async = graph.subgraphs[target_sg.0 as usize].has_suspend;
        let param_count = graph.subgraphs[target_sg.0 as usize].param_count as usize;
        let n = graph.node(node.0 as usize);
        let inputs = graph.inputs(n.inputs_offset, n.input_count);
        let args: Vec<Value> = inputs
            .iter()
            .take(param_count)
            .map(|&in_node| frame.get_value_by_global(in_node))
            .collect();

        return NodeResult::Call(PendingCall {
            target_sg,
            args,
            call_node_local,
            is_async,
            closure_val: None,
        });
    }

    // Dynamic dispatch: vtable_call_methods (queries the method subgraph at runtime from a TraitVal).
    if let Some(method_idx) = graph.vtable_call_method(node.0 as usize) {
        let n = graph.node(node.0 as usize);
        let inputs = graph.inputs(n.inputs_offset, n.input_count);
        let recv_val = frame.get_value_by_global(inputs[0]);

        let (target_sg, upvalues): (SubGraphId, Vec<Value>) = match recv_val.heap_obj() {
            Some(crate::value::HeapObj::TraitVal(tv)) => {
                let idx = method_idx as usize;
                match tv.method_values.get(idx).and_then(|v| v.heap_obj()) {
                    Some(crate::value::HeapObj::Closure(c)) => {
                        (SubGraphId(c.func_id), c.upvalues.clone())
                    }
                    _ => panic!("vtable method_idx {} is not a Closure", method_idx),
                }
            }
            _ => panic!("vtable call on non-trait value"),
        };

        let is_async = graph.subgraphs[target_sg.0 as usize].has_suspend;
        let arity = (graph.subgraphs[target_sg.0 as usize].param_count as usize)
            .saturating_sub(upvalues.len());
        let mut args: Vec<Value> = Vec::with_capacity(arity + upvalues.len());
        for &in_node in inputs.iter().skip(1).take(arity) {
            args.push(frame.get_value_by_global(in_node));
        }
        args.extend(upvalues);

        return NodeResult::Call(PendingCall {
            target_sg,
            args,
            call_node_local,
            is_async,
            closure_val: None,
        });
    }

    // Neither present: the compiler guarantees a Call node has at least one; no panic here, kept fault-tolerant.
    NodeResult::Value(Value::VOID)
}

/// compute_fn: Gate node selects a branch + returns `NodeResult::Call`.
pub fn compute_gate_launch(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    let branches = graph.gate_branches_at(node.0 as usize);
    let branches = branches
        .as_ref()
        .expect("Gate node has no branches");

    // Read the condition value.
    let cond_raw = frame.get_value_by_global(branches.condition_input);
    let cond = cond_raw.as_bool();

    if env_flag("KUZO_DEBUG_GATE") {
        let sg = &graph.subgraphs[frame.subgraph_id.0 as usize];
        eprintln!("[GATE] node={:?} cond_raw={:?} cond={} frame.sg={} frame.offset={} sg.range=[{},{}) branches={:?}",
            node, cond_raw, cond, frame.subgraph_id.0, frame.node_offset,
            sg.node_range.0 .0, sg.node_range.1 .0,
            branches.branches.iter().map(|(c, sg, _)| (*c, sg.0)).collect::<Vec<_>>());
    }

    // Select a branch.
    let (target_sg, branch_inputs) = branches
        .branches
        .iter()
        .find(|(c, _, _)| *c == cond)
        .map(|(_, sg, inputs)| (*sg, inputs.clone()))
        .expect("no matching gate branch");

    // Collect arguments.
    let param_count = graph.subgraphs[target_sg.0 as usize].param_count as usize;
    let args: Vec<Value> = branch_inputs
        .iter()
        .take(param_count)
        .map(|&n| frame.get_value_by_global(n))
        .collect();

    if env_flag("KUZO_DEBUG_STALL") {
        let (ns, ne) = graph.subgraphs[target_sg.0 as usize].node_range;
        eprintln!("[GATE] node={} cond={} target_sg={} sg_range=[{},{}) params={} branch_inputs={:?} args={}",
            node.0, cond, target_sg.0, ns.0, ne.0, param_count, branch_inputs, args.len());
        for gid in ns.0..ne.0 {
            let n = graph.node(gid as usize);
            let cv = graph.const_value(gid as usize);
            eprintln!("  [GATE-NODE] gid={} kind={:?} cf={} const_values={:?} inputs_count={}",
                gid, n.kind, n.compute_fn.0, cv.is_some(), n.input_count);
        }
    }

    let gate_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));

    NodeResult::Call(PendingCall {
        target_sg,
        args,
        call_node_local: gate_node_local,
        is_async: false,
        closure_val: None,
    })
}

/// compute_await (idx 38): the await node returns `NodeResult::Await`.
///
/// Spec 4.4: event source not ready → await not ready → frame has no more ready
/// nodes → suspend. When the core loop receives `NodeResult::Await`, it resolves
/// the event source → checks readiness → if ready, injects the value and
/// continues; if not ready, suspends.
pub fn compute_await(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    use crate::ir::Ir::PendingAwait;

    read_node_inputs!(frame, node, graph, n, inputs);
    // inputs[0] = event-object node (AsyncHandle/Channel/Timer).
    let event_obj = frame.get_value_by_global(inputs[0]);
    let await_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));

    // The EventSource node is read from the await_event_sources table (a metadata reference, not a data dependency).
    let es_node = graph.await_event_source(node.0 as usize);
    let event_kind = match es_node {
        Some(es) => graph
            .subgraphs
            .get(frame.subgraph_id.0 as usize)
            .and_then(|sg| {
                sg.event_source_decls
                    .iter()
                    .find(|d| d.node == es)
                    .map(|d| d.kind)
            })
            .unwrap_or(crate::ir::Ir::EventSourceKind::AsyncJoin),
        None => crate::ir::Ir::EventSourceKind::AsyncJoin,
    };

    NodeResult::Await(PendingAwait {
        await_node_local,
        event_obj,
        event_kind,
    })
}

/// compute_channel_create (idx 283): creates a ChannelValue heap object.
///
/// Input: `inputs[0] = capacity` (usize).
/// Output: `Value::ref_val(HeapObj::ChannelVal(Arc<ChannelValue>))`.
pub fn compute_channel_create(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let capacity = frame.get_value_by_global(inputs[0]).as_usize();
    Value::ref_val(crate::value::HeapObj::ChannelVal(
        std::sync::Arc::new(crate::value::ChannelValue::new(capacity)),
    ))
}

/// compute_channel_send (idx 284): non-blocking send + returns
/// `NodeResult::ChannelNotify`.
///
/// Inputs: `inputs[0] = channel ref`, `inputs[1] = value`.
/// After sending, returns `NodeResult::ChannelNotify`; when the core loop
/// consumes it, it triggers a `ChannelReady` event that wakes the suspended
/// frames waiting on that channel (inline trigger, zero latency).
pub fn compute_channel_send(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    // safe_op short-circuit: `?.send(v)` returns Null when the receiver is null.
    if graph.safe_op_flag(node.0 as usize) {
        let ch_val = frame.get_value_by_global(inputs[0]);
        if ch_val.is_null() {
            return NodeResult::Value(Value::Null);
        }
    }
    let ch_val = frame.get_value_by_global(inputs[0]);
    let val = frame.get_value_by_global(inputs[1]);
    let make_err = |msg: &str| make_error_throw("ChannelError", msg);
    let ch = match ch_val.heap_obj().and_then(|h| h.channel()) {
        Some(ch) => ch,
        None => return NodeResult::Value(make_err("send on non-channel value")),
    };
    match ch.send(val) {
        Ok(()) => {
            let ch_id = crate::ir::Ir::ChannelId(ch.id());
            NodeResult::ChannelNotify(ch_id)
        }
        Err(e) => NodeResult::Value(make_err(e.message())),
    }
}

/// compute_channel_close (idx 285): closes the channel.
///
/// Input: `inputs[0] = channel ref`.
pub fn compute_channel_close(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let ch_val = frame.get_value_by_global(inputs[0]);
    let ch = ch_val.heap_obj().and_then(|h| h.channel())
        .expect("close on non-channel value");
    ch.close();
    Value::VOID
}

/// compute_fn: closure construction (idx 40).
///
/// Reads the subgraph id + arity from `graph.closure_infos`, merges the inputs
/// (captured values) to construct a Closure heap object. The node's inputs are
/// the captured upvalues (in the order of `captured` in `compile_lambda`).
pub fn compute_closure_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Closure, Cell};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.closure_info(node.0 as usize)
        .expect("closure construct node has no ClosureInfo");
    // Wrap each upvalue in a Cell so that escaping closures (cross-function
    // calls) can persist upvalue mutations via the Cell's interior mutability.
    // same_function calls do not use the Cell (they read the latest value
    // directly from the parent frame).
    let upvalues: Vec<Value> = inputs
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .map(|v| Value::ref_val(HeapObj::Cell(Cell::new(v))))
        .collect();
    let cell_bits = if upvalues.len() >= 8 { 0xFF } else { (1u8 << upvalues.len()) - 1 };
    Value::ref_val(HeapObj::Closure(Closure {
        func_id: info.subgraph_id.0,
        arity: info.arity,
        upvalues,
        bound_args: Vec::new(),
        self_upvalue_idx: info.self_upvalue_idx,
        upvalue_ref_bits: 0,
        cell_upvalues: cell_bits,
    }))
}

/// compute_fn: inline_trait construction (idx 266).
///
/// Reads the trait name + method list from `graph.trait_construct_infos`,
/// merges the node's inputs (each method's upvalues concatenated in order) to
/// build multiple Closures, and packs them into a TraitValue heap object.
pub fn compute_trait_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Closure, TraitValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.trait_construct_info_at(node.0 as usize);
    let info = info
        .as_ref()
        .expect("trait construct node has no TraitConstructInfo");

    // Slice `inputs` per each method's `upvalue_count` in order, and build each method's Closure.
    let mut method_values: Vec<Value> = Vec::with_capacity(info.methods.len());
    let mut input_cursor = 0usize;
    for m in &info.methods {
        let upvalue_count = m.upvalue_count as usize;
        let upvalues: Vec<Value> = inputs[input_cursor..input_cursor + upvalue_count]
            .iter()
            .map(|&in_node| frame.get_value_by_global(in_node))
            .collect();
        input_cursor += upvalue_count;
        method_values.push(Value::ref_val(HeapObj::Closure(Closure {
            func_id: m.subgraph_id.0,
            arity: m.arity,
            upvalues,
            bound_args: Vec::new(),
            self_upvalue_idx: -1,
            upvalue_ref_bits: 0,
            cell_upvalues: 0,
        })));
    }

    Value::ref_val(HeapObj::TraitVal(TraitValue {
        trait_name: info.trait_name.clone(),
        method_names: info.method_names.clone(),
        method_values,
        data: None,
        owned: true,
    }))
}

/// compute_fn: lazy construction (idx 267).
///
/// Reads the thunk subgraph id from `graph.lazy_construct_infos`, merges the
/// node's inputs (upvalues) to construct a LazyValue heap object. The thunk is
/// unevaluated; on the first force it starts subgraph computation and caches
/// the result.
pub fn compute_lazy_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, LazyValue, Closure};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.lazy_construct_info(node.0 as usize)
        .expect("lazy construct node has no LazyConstructInfo");

    // Collect upvalues from `inputs` into a Closure (used when the thunk is first forced).
    let upvalues: Vec<Value> = inputs
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();

    // Wrap the thunk subgraph in a Closure (func_id = thunk_sg) and store it as
    // LazyValue.data. On force, take the Closure from `data`, start subgraph
    // computation, and cache the result in `cached`.
    let thunk_closure = Value::ref_val(HeapObj::Closure(Closure {
        func_id: info.thunk_sg.0,
        arity: 0,
        upvalues,
        bound_args: Vec::new(),
        self_upvalue_idx: -1,
        upvalue_ref_bits: 0,
        cell_upvalues: 0,
    }));

    Value::ref_val(HeapObj::LazyVal(LazyValue {
        cached: std::sync::Mutex::new(None),
        forced: std::sync::atomic::AtomicBool::new(false),
        data: Some(thunk_closure),
    }))
}

// =========================================================================
// LazyValue force mechanism: synchronously executes the thunk subgraph and caches the result.
// =========================================================================

/// Forces evaluation of a LazyValue: synchronously executes the thunk subgraph
/// and returns the result.
///
/// If already forced, returns the cached value directly; otherwise creates a
/// thunk frame, runs it synchronously to completion, caches the result in the
/// LazyValue (updated in place via `Arc::make_mut`), and returns the result.
///
/// Called by `compute_reflect_format` / `compute_reflect_scalar_to_str` to
/// force lazy values before formatting.
pub fn force_lazy_value_sync(caller_frame: &mut Frame, lazy_val: &Value) -> Value {
    use crate::value::HeapObj;

    // Extract the LazyValue reference.
    let arc = match lazy_val {
        Value::Ref(r) => r,
        _ => return lazy_val.clone(), // not a LazyValue, return directly
    };

    // Check whether already forced.
    {
        if let HeapObj::LazyVal(lazy) = &**arc {
            if lazy.forced.load(std::sync::atomic::Ordering::Relaxed) {
                return lazy.cached.lock().unwrap().clone().unwrap_or(Value::NULL);
            }
        } else {
            return lazy_val.clone(); // not a LazyVal, return directly
        }
    }

    // Take the thunk Closure.
    let closure = {
        let HeapObj::LazyVal(lazy) = &**arc else { return lazy_val.clone() };
        match &lazy.data {
            Some(v) => match v.heap_obj() {
                Some(HeapObj::Closure(c)) => c.clone(),
                _ => return Value::NULL,
            },
            None => return Value::NULL,
        }
    };

    let graph = caller_frame.graph.clone();
    let thunk_sg = SubGraphId(closure.func_id);

    // Create the thunk frame.
    let (node_start, node_end) = graph.subgraphs[thunk_sg.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;
    let mut thunk_frame = Frame::new(THUNK_FRAME_ID, thunk_sg, node_count, graph.clone());
    prepare_frame_nodes(&mut thunk_frame, &graph);

    // Inject upvalues as arguments.
    let offset = node_start.0 as usize;
    let param_count = graph.subgraphs[thunk_sg.0 as usize].param_count as usize;
    for (i, arg) in closure.upvalues.iter().enumerate().take(param_count) {
        let local_id = NodeId(i as u32);
        let consumer_count = graph.downstream_slice(offset + i).len() as u16;
        thunk_frame.set_value(local_id, arg.clone(), consumer_count);
        thunk_frame.push_ready(local_id);
    }

    // The thunk frame's upvalues have been injected as arguments (loop above);
    // outer variables are not accessed via `parent_frame_ptr`. Set to null to
    // avoid the caller_frame's `&mut` borrow forming aliased UB with the raw
    // pointer deref.
    thunk_frame.parent_frame_ptr = std::ptr::null_mut();

    // Synchronously execute the thunk frame.
    let result = run_frame_sync(&mut thunk_frame, &graph);

    // Cache the result in the LazyValue (updated via the Mutex/AtomicBool's interior mutability).
    if let HeapObj::LazyVal(lazy) = &**arc {
        lazy.forced.store(true, std::sync::atomic::Ordering::Relaxed);
        *lazy.cached.lock().unwrap() = Some(result.clone());
    }

    result
}

/// Synchronous-path loop-iteration reset: after a LoopBody finishes a
/// Continue/None, resets the loop frame's cond/gate/iter_next so it re-enters
/// the next iteration.
///
/// Mirrors `Engine::reset_loop_iteration` but does not handle body_frame reuse
/// (the sync path creates a new child_frame for each iteration, never reusing).
/// The sync path is not driven by a frame queue; instead, the
/// `run_frame_sync_inner` main loop pops nodes directly from the `ready_queue`,
/// so after reset, cond/iter_next are pushed and picked up again by the main loop.
fn reset_loop_frame_for_next_iteration(frame: &mut Frame, graph: &DataFlowGraph) {
    let loop_sg_id = frame.subgraph_id;
    let (loop_kind, cond_node, return_node, iter_next_node) = {
        let sg = &graph.subgraphs[loop_sg_id.0 as usize];
        (sg.loop_kind, sg.cond_node, sg.return_node, sg.iter_next_node)
    };
    let loop_offset = frame.node_offset;

    // 0. Clear the ready_queue (must precede pushing cond/iter_next).
    // If not cleared, stale ready entries would execute before cond/iter_next,
    // referencing outdated values.
    frame.ready_queue.clear();

    // 1. For loop: reset iter_next_node.
    if loop_kind == LoopKind::For {
        if let Some(next_node) = iter_next_node {
            let next_local = NodeId(next_node.0.wrapping_sub(loop_offset));
            let i = next_local.0 as usize;
            if i < frame.pending_inputs.len() {
                frame.pending_inputs[i] = 0;
            }
            if i < frame.value_table.len() {
                frame.value_table.reset_slot(i);
            }
            frame.push_ready(next_local);
        }
    }

    // 2. Reset cond_node.
    if let Some(cond_node) = cond_node {
        let cond_local = NodeId(cond_node.0.wrapping_sub(loop_offset));
        let i = cond_local.0 as usize;
        if loop_kind == LoopKind::For {
            // For loop cond depends on iter_next, pending=1.
            if i < frame.pending_inputs.len() {
                frame.pending_inputs[i] = 1;
            }
            if i < frame.value_table.len() {
                frame.value_table.reset_slot(i);
            }
        } else {
            // While/Loop cond has no input dependency, pending=0.
            if i < frame.pending_inputs.len() {
                frame.pending_inputs[i] = 0;
            }
            if i < frame.value_table.len() {
                frame.value_table.reset_slot(i);
            }
            // Re-pre-fill a Const cond_node.
            if graph.node(cond_node.0 as usize).kind == NodeKind::Const {
                if let Some(cv) = graph.const_value(cond_node.0 as usize) {
                    let handle = cv.to_value(graph.string_pool_slice());
                    let consumer_count =
                        graph.downstream_slice(cond_node.0 as usize).len() as u16;
                    frame.set_value(cond_local, handle, consumer_count);
                }
            }
            frame.push_ready(cond_local);
        }
    }

    // 3. Reset the Gate node (= return_node; pending=1, waits for cond notify).
    let gate_local = NodeId(return_node.0.wrapping_sub(loop_offset));
    let gi = gate_local.0 as usize;
    if gi < frame.pending_inputs.len() {
        frame.pending_inputs[gi] = 1;
    }
    if gi < frame.value_table.len() {
        frame.value_table.reset_slot(gi);
    }

    // 4. Reset loop-frame state.
    frame.control_signal = ControlSignal::None;
    frame.state = FrameState::Ready;
}

/// Synchronously runs a frame to completion, handling nested function calls,
/// control signals, and vtable dispatch.
///
/// This is the synchronous simplified version of Engine's async execution model:
/// - Nodes within a frame are scheduled by the ready queue.
/// - On a Call node, recursively calls `run_frame_sync` to execute the subframe.
/// - Control signals (return/break/continue) terminate the loop.
///
/// Defer execution: after the frame finishes (any termination path), runs the
/// defer bodies in `defer_table` in LIFO order. Defer bodies are run via
/// recursive `run_frame_sync` (supports nested defers).
fn run_frame_sync(frame: &mut Frame, graph: &DataFlowGraph) -> Value {
    let result = run_frame_sync_inner(frame, graph);
    // Execute defers (LIFO): any termination path runs the defers.
    run_defers_sync(frame, graph);
    result
}

/// Executes the defer bodies in the frame's `defer_table` (LIFO order).
/// A defer body is an independent subgraph; a new frame is created and run
/// synchronously via `run_frame_sync`.
fn run_defers_sync(frame: &mut Frame, graph: &DataFlowGraph) {
    let sg_id = frame.subgraph_id;
    let defer_entries: Vec<crate::ir::Ir::DeferEntry> =
        graph.subgraphs[sg_id.0 as usize].defer_table.clone();
    for entry in defer_entries.iter().rev() {
        let (dn_start, dn_end) = graph.subgraphs[entry.body_subgraph.0 as usize].node_range;
        let dn_count = (dn_end.0 - dn_start.0) as usize;
        let mut defer_frame = Frame::new(
            FrameId(u32::MAX),
            entry.body_subgraph,
            dn_count,
            frame.graph.clone(),
        );
        prepare_frame_nodes(&mut defer_frame, graph);
        let _ = run_frame_sync(&mut defer_frame, graph);
    }
}

/// Inner implementation of `run_frame_sync` (does not execute defers).
///
/// Unified hot loop: pop → compute_fn → match NodeResult.
/// - Call: recursively create a subframe + run synchronously + inject the return value.
/// - Return/Break/Continue: set `control_signal` to terminate the loop.
/// - Await/ChannelNotify/Cancel/SelectWait: not supported on the sync path, returns NULL.
///
/// Not supported: async/await, channel/timer events, select, loop-body reuse.
/// Suitable for thunk subgraphs (pure computation + synchronous function calls).
fn run_frame_sync_inner(frame: &mut Frame, graph: &DataFlowGraph) -> Value {
    use crate::ir::Ir::{ControlSignal, LoopKind, NodeKind};

    let mut iter_guard: u64 = 0;
    loop {
        iter_guard += 1;
        if iter_guard > 100000 {
            return Value::NULL;
        }
        // 1. Check the control signal.
        let cs = frame.control_signal.clone();
        match cs {
            ControlSignal::Return(v) => return v,
            ControlSignal::Break | ControlSignal::Continue => return Value::VOID,
            ControlSignal::None => {}
        }

        // 2. POP.
        let local_id = match frame.pop_ready() {
            Some(n) => n,
            None => {
                let sg = &graph.subgraphs[frame.subgraph_id.0 as usize];
                let return_local = sg.return_node.0.wrapping_sub(frame.node_offset);
                if (return_local as usize) < frame.value_table.len()
                    && !frame.value_table.is_ready(return_local as usize)
                {
                    return Value::NULL;
                }
                return frame.get_value_by_global(sg.return_node);
            }
        };

        let node_start = frame.node_offset;
        let graph_node_id = NodeId(local_id.0 + node_start);
        let node = graph.node(graph_node_id.0 as usize);
        let ctx = EvalContext { node_start };

        // 3. COMPUTE.
        let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, graph_node_id, &ctx);

        // 4. MATCH NodeResult
        match result {
            NodeResult::Value(v) => {
                let cc = graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
                frame.set_value(local_id, v, cc);
                notify_downstream(frame, graph, local_id, graph_node_id, NodeId(node_start));
            }
            NodeResult::Batch(results) => {
                for &(lid, ref v) in &results {
                    let gid = NodeId(lid.0 + node_start);
                    let cc = graph.downstream_slice(gid.0 as usize).len() as u16;
                    frame.set_value(lid, v.clone(), cc);
                }
                for &(lid, _) in &results {
                    frame.ready_queue.retain(|n| *n != lid);
                }
                for &(lid, _) in &results {
                    let gid = NodeId(lid.0 + node_start);
                    notify_downstream(frame, graph, lid, gid, NodeId(node_start));
                }
            }
            NodeResult::Call(pending) => {
                // Tail call: reuse the current frame.
                if graph.tail_call_flag(graph_node_id.0 as usize) {
                    switch_subgraph(frame, graph, pending.target_sg, &pending.args);
                    continue;
                }

                let target_loop_kind = graph.subgraphs[pending.target_sg.0 as usize].loop_kind;

                // LoopBody: loop-body reuse is not supported (thunks should not contain loops); fall back to a normal call.
                let (child_start, child_end) = graph.subgraphs[pending.target_sg.0 as usize].node_range;
                let child_count = (child_end.0 - child_start.0) as usize;
                let mut child_frame = Frame::new(
                    LOOPBODY_FALLBACK_FRAME_ID,
                    pending.target_sg,
                    child_count,
                    frame.graph.clone(),
                );
                prepare_frame_nodes(&mut child_frame, graph);

                // Inject arguments.
                let child_offset = child_start.0 as usize;
                let child_param_count = graph.subgraphs[pending.target_sg.0 as usize].param_count as usize;
                for (i, arg) in pending.args.iter().enumerate().take(child_param_count) {
                    let lid = NodeId(i as u32);
                    let cc = graph.downstream_slice(child_offset + i).len() as u16;
                    child_frame.set_value(lid, arg.clone(), cc);
                    child_frame.push_ready(lid);
                }

                // Set up the frame-chain pointers.
                let same_function = graph.subgraphs[frame.subgraph_id.0 as usize].function_id
                    == graph.subgraphs[pending.target_sg.0 as usize].function_id;
                child_frame.parent_frame_ptr = if same_function {
                    frame as *mut Frame
                } else {
                    std::ptr::null_mut()
                };
                child_frame.root_frame_ptr = if same_function {
                    if frame.root_frame_ptr.is_null() {
                        frame as *mut Frame
                    } else {
                        frame.root_frame_ptr
                    }
                } else {
                    std::ptr::null_mut()
                };
                child_frame.closure_val = pending.closure_val.clone();

                // Synchronously execute the child frame.
                let child_result = run_frame_sync(&mut child_frame, graph);
                let child_signal = child_frame.control_signal.clone();

                // Inject the return value into the current frame.
                let consumer_count = graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
                frame.set_value(pending.call_node_local, child_result.clone(), consumer_count);

                // Propagate throw.
                let is_throw_err = matches!(
                    child_result.heap_obj(),
                    Some(crate::value::HeapObj::ThrowVal(t)) if matches!(t.payload, crate::value::ThrowPayload::Err(_))
                );
                if is_throw_err {
                    frame.control_signal = ControlSignal::Return(child_result);
                    continue;
                }

                // Propagate the Gate branch control signal.
                let is_gate = graph.node(graph_node_id.0 as usize).kind == NodeKind::Gate;
                if is_gate && !matches!(child_signal, ControlSignal::None) {
                    frame.control_signal = child_signal;
                    continue;
                }

                // LoopBody completion handling.
                if target_loop_kind == LoopKind::LoopBody {
                    match child_signal {
                        ControlSignal::Break | ControlSignal::Return(_) => {
                            frame.control_signal = child_signal;
                            continue;
                        }
                        ControlSignal::Continue => {
                            reset_loop_frame_for_next_iteration(frame, graph);
                            continue;
                        }
                        ControlSignal::None => {
                            let loop_kind = graph.subgraphs[frame.subgraph_id.0 as usize].loop_kind;
                            if loop_kind == LoopKind::TailRec {
                                frame.control_signal = ControlSignal::Return(child_result);
                                continue;
                            } else {
                                reset_loop_frame_for_next_iteration(frame, graph);
                                continue;
                            }
                        }
                    }
                }

                notify_downstream(frame, graph, pending.call_node_local, graph_node_id, NodeId(node_start));
            }
            NodeResult::Return(v) => {
                frame.control_signal = ControlSignal::Return(v);
                continue;
            }
            NodeResult::Break => {
                frame.control_signal = ControlSignal::Break;
                continue;
            }
            NodeResult::Continue => {
                frame.control_signal = ControlSignal::Continue;
                continue;
            }
            // Not supported on the sync path: async/await, channel/timer, select.
            NodeResult::Await(_)
            | NodeResult::ChannelNotify(_)
            | NodeResult::Cancel(_)
            | NodeResult::SelectWait(_) => {
                return Value::NULL;
            }
        }
    }
}

/// compute_fn: partial application construction (idx 286).
///
/// Reads the subgraph id + bound_count from `partial_infos`, merges the inputs
/// (already-bound argument values) to construct `HeapObj::Partial`.
/// `remaining_arity = subgraph.param_count - bound_count`.
/// For top-level function partial application, upvalues are empty and `self_upvalue_idx = -1`.
pub fn compute_partial_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, PartialApplication};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.partial_info(node.0 as usize)
        .expect("partial construct node has no PartialInfo");
    let bound_args: Vec<Value> = inputs
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();
    let param_count = graph.subgraphs[info.subgraph_id.0 as usize].param_count as usize;
    let remaining_arity = param_count.saturating_sub(bound_args.len()) as u8;
    Value::ref_val(HeapObj::Partial(PartialApplication {
        func_id: info.subgraph_id.0,
        upvalues: Vec::new(),
        bound_args,
        remaining_arity,
        self_upvalue_idx: -1,
    }))
}

/// compute_str_bytes (idx 287): `str.bytes()` -> `u8[]`.
/// Constructs a `u8` array from the UTF-8 byte sequence of a `KuzoStr`.
pub fn compute_str_bytes(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let bytes: Vec<Value> = match val.heap_obj() {
        Some(HeapObj::Str(s)) => s.bytes().as_bytes()
            .iter()
            .map(|&b| Value::u8(b))
            .collect(),
        _ => Vec::new(),
    };
    Value::ref_val(HeapObj::Array(ArrayValue::new(bytes)))
}

/// compute_fn: callable value invocation (idx 41) — uniformly handles `Closure | Partial`.
///
/// `inputs[0]` = callable value node, `inputs[1..1+arg_count]` = call argument nodes
/// (`arg_count` is read from the `closure_call_arg_counts` metadata, excluding the
/// closure value and effect dependencies).
///
/// Unified call semantics:
/// - Closure: `needed_arity = subgraph.param_count - upvalues.len()`
/// - Partial: `needed_arity = remaining_arity`
///
/// When the new argument count < `needed_arity` -> produce a new `Partial` (chained partial application);
/// When the new argument count >= `needed_arity` -> merge `bound_args` + new args + upvalues, set `pending_call`.
/// Unwraps a `Cell`-wrapped upvalue: if the value is a `Cell`, returns a clone of the inner value;
/// otherwise returns a clone as-is.
/// Used by `compute_closure_call` to convert `Cell` upvalues into raw values injected into the child frame parameters.
fn unwrap_cell(v: &Value) -> Value {
    match v.heap_obj() {
        Some(crate::value::HeapObj::Cell(cell)) => cell.get(),
        _ => v.clone(),
    }
}

pub fn compute_closure_call(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    use crate::value::{HeapObj, PartialApplication};
    read_node_inputs!(frame, node, graph, n, inputs);
    let callable_val = frame.get_value_by_global(inputs[0]);
    // safe_op short-circuit: `?.method(args)` returns Null when the receiver is null.
    if graph.safe_op_flag(node.0 as usize) && callable_val.is_null() {
        return NodeResult::Value(Value::Null);
    }

    // Read the actual argument count from metadata (excluding closure value and effect dependencies).
    let arg_count = graph.closure_call_arg_count(node.0 as usize)
        .expect("closure_call node has no arg_count") as usize;
    let new_args: Vec<Value> = inputs
        .iter()
        .skip(1)
        .take(arg_count)
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();

    // Uniformly extract the launch info of the callable value.
    let (func_id, upvalues, bound_args, needed_arity, self_upvalue_idx) = match callable_val.heap_obj() {
        Some(HeapObj::Closure(c)) => {
            let total_params = graph.subgraphs[c.func_id as usize].param_count as usize;
            let needed = total_params.saturating_sub(c.upvalues.len());
            // Unwrap Cell upvalues into raw values for parameter injection (Cell is used for escaped closure persistent writeback).
            let upvalues: Vec<Value> = c.upvalues.iter().map(|v| unwrap_cell(v)).collect();
            (c.func_id, upvalues, Vec::new(), needed, c.self_upvalue_idx)
        }
        Some(HeapObj::Partial(p)) => {
            let upvalues: Vec<Value> = p.upvalues.iter().map(|v| unwrap_cell(v)).collect();
            (p.func_id, upvalues, p.bound_args.clone(), p.remaining_arity as usize, p.self_upvalue_idx)
        }
        _ => panic!("compute_closure_call: input is not callable (Closure or Partial)"),
    };

    // Chained partial application: insufficient new args -> produce a new Partial.
    if new_args.len() < needed_arity {
        let provided = new_args.len();
        let mut extended = bound_args;
        extended.extend(new_args);
        let new_remaining = needed_arity - provided;
        return NodeResult::Value(Value::ref_val(HeapObj::Partial(PartialApplication {
            func_id,
            upvalues,
            bound_args: extended,
            remaining_arity: new_remaining as u8,
            self_upvalue_idx,
        })));
    }

    // Full arity: merge `bound_args` + `new_args[..needed]` + upvalues, return `NodeResult::Call`.
    let target_sg = SubGraphId(func_id);
    let call_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));
    let upvalues_len = upvalues.len();
    let mut args: Vec<Value> = Vec::with_capacity(bound_args.len() + needed_arity + upvalues_len);
    args.extend(bound_args);
    args.extend(new_args.iter().take(needed_arity).cloned());
    args.extend(upvalues);

    // Recursive closure: inject the self reference into the upvalue slot at `self_upvalue_idx`.
    // Bounds check: prevent usize underflow and array out-of-bounds (`self_upvalue_idx` must fall within the upvalues range).
    if self_upvalue_idx >= 0 {
        assert!(upvalues_len <= args.len(), "upvalues_len exceeds args.len()");
        let self_upvalue_idx = self_upvalue_idx as usize;
        assert!(self_upvalue_idx < upvalues_len, "self_upvalue_idx out of bounds");
        let upvalues_start = args.len() - upvalues_len;
        let self_idx = upvalues_start + self_upvalue_idx;
        assert!(self_idx < args.len(), "self_idx out of bounds");
        args[self_idx] = callable_val.clone();
    }

    NodeResult::Call(PendingCall {
        target_sg,
        args,
        call_node_local,
        is_async: false,
        closure_val: Some(callable_val.clone()),
    })
}

/// compute_fn: cancels the child frame corresponding to an async handle.
///
/// `inputs[0]` = async handle value (an `i32` scalar whose value is `async_id`).
/// Returns `NodeResult::Cancel`; the core loop looks up `async_id` -> `child_fid` in
/// `AsyncJoinRuntime` to perform the cancellation.
pub fn compute_cancel_async_handle(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    let handle_val = frame.get_value_by_global(inputs[0]);
    // safe_op short-circuit: `?.cancel()` returns Null when the receiver is null.
    if graph.safe_op_flag(node.0 as usize) && handle_val.is_null() {
        return NodeResult::Value(Value::Null);
    }
    // The async handle is an i32 scalar; its value is the async_id.
    let async_id = crate::ir::Ir::AsyncHandleId(handle_val.as_i32() as u32);
    NodeResult::Cancel(async_id)
}

/// compute_fn: select gate node (idx 43) — returns `NodeResult::SelectWait`.
///
/// Upon receiving this, the core loop checks the ready state of all branch event
/// sources (it has access to the full Engine state).
pub fn compute_select_gate(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    // Verify that the gate node actually has a bound SelectInfo.
    let info = graph.select_info_at(node.0 as usize);
    let _ = info
        .as_ref()
        .expect("select gate node has no SelectInfo");
    let gate_local = NodeId(node.0.wrapping_sub(frame.node_offset));
    NodeResult::SelectWait(gate_local)
}


/// noop compute_fn (matches the real signature).
pub fn noop_compute_real(_frame: &mut Frame, _node: NodeId) -> Value {
    Value::VOID
}

/// compute_fn for Const nodes (new signature, not wrapped via `wrap_fn!`).
/// Materializes a value from the `const_values` table and returns it.
/// Non-Const nodes (which also use `CF_NOOP`) return `Value::VOID` (compatible with `noop_compute_real`).
pub fn compute_const(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    if let Some(cv) = frame.graph.const_value(node.0 as usize) {
        NodeResult::Value(crate::engine::alloc_const_value(cv, frame.graph.string_pool_slice()))
    } else {
        NodeResult::Value(Value::VOID)
    }
}

/// compute_return (idx 311): extracts the input value and returns `NodeResult::Return`.
///
/// `inputs[0]` = return value. Optional `inputs[1]` = prior side-effect dependency
/// (used only for readiness checks; its value is ignored).
/// Replaces the old `control_signal_nodes[SignalKind::Return]` table lookup.
pub fn compute_return(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    NodeResult::Return(v)
}

/// compute_break (idx 312): returns `NodeResult::Break`.
///
/// Optional `inputs[0]` = prior side-effect dependency (used only for readiness
/// checks; its value is ignored).
/// Replaces the old `control_signal_nodes[SignalKind::Break]` table lookup.
pub fn compute_break(_frame: &mut Frame, _node: NodeId, _ctx: &EvalContext) -> NodeResult {
    NodeResult::Break
}

/// compute_continue (idx 313): returns `NodeResult::Continue`.
///
/// Optional `inputs[0]` = prior side-effect dependency (used only for readiness
/// checks; its value is ignored).
/// Replaces the old `control_signal_nodes[SignalKind::Continue]` table lookup.
pub fn compute_continue(_frame: &mut Frame, _node: NodeId, _ctx: &EvalContext) -> NodeResult {
    NodeResult::Continue
}

/// compute_fn (idx 48): sequence node — waits for all inputs to be ready, then returns
/// the value of the last input.
///
/// Used for statement sequencing: `inputs = [prev_effect, current_value]`, returns `current_value`.
/// `prev_effect` acts only as a data-dependency edge (ordering constraint) to ensure the previous
/// statement completes before the current one executes.
pub fn compute_seq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    if n.input_count == 0 {
        return Value::VOID;
    }
    let last_input = inputs[n.input_count as usize - 1];
    frame.get_value_by_global(last_input)
}

/// compute_writeback (idx 49): assigns an outer variable, writing back to the function
/// root frame via `root_frame_ptr`.
///
/// `inputs[0]` = value source (a node in the current frame),
/// `writeback_targets[node]` = the outer global `NodeId`.
/// Non-blocking: the write completes directly inside the `compute_fn` — no pending state,
/// no Engine-layer consumption.
///
/// Three writeback paths (in priority order):
/// 1. `parent_frame_ptr` chain: same-function closure call, writes to the nearest parent
///    frame containing the target.
/// 2. `root_frame_ptr`: same-function closure call, writes to the function root frame
///    (so other `same_function` calls can observe the latest value).
/// 3. `closure_val` Cell: escaped closure (cross-function call, frame chain is null),
///    updates the closure upvalues via the `Cell`'s interior mutability so the next call
///    reads the latest value.
pub fn compute_writeback(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    let n = graph.node(node.0 as usize);
    if n.input_count == 0 {
        return NodeResult::Value(Value::VOID);
    }
    let val_node = graph.inputs(n.inputs_offset, n.input_count)[0];
    let val = frame.get_value_by_global(val_node);
    let target = graph.writeback_target(node.0 as usize)
        .expect("WriteBack node missing target");
    let consumer_count = graph.downstream_slice(target.0 as usize).len() as u16;

    if env_flag("KUZO_DEBUG_WB") {
        let sg = &graph.subgraphs[frame.subgraph_id.0 as usize];
        eprintln!("[WB] node={:?} target={:?} val={:?} val_node={:?} frame.sg={} frame.offset={} sg.range=[{},{}) sg.func_id={} vt_len={}",
            node, target, val, val_node, frame.subgraph_id.0, frame.node_offset,
            sg.node_range.0 .0, sg.node_range.1 .0, sg.function_id, frame.value_table.len());
    }

    // Path 0: write to the current frame (same_function closure call scenario).
    // The value table of a same_function frame is extended to the parent frame size,
    // so the target may fall within the current frame's range.
    // If the current frame is not written: after a() modifies `log`, WriteBack only
    // writes the parent frame chain (the main frame); the `log` in a's own child frame
    // remains stale. A subsequent b() reading the upvalue from a's child frame
    // (parent_frame) gets a stale value, breaking mutable capture sharing across the
    // closure chain (Bug #31).
    let cur_local = target.0.wrapping_sub(frame.node_offset);
    if (cur_local as usize) < frame.value_table.len() {
        frame.set_value(NodeId(cur_local), val.clone(), consumer_count);
    }

    // Path 1: walk the `parent_frame_ptr` chain, writing to every ancestor frame that
    // contains the target.
    // We cannot break after writing only the nearest parent: in nested same_function
    // subgraphs (e.g. if branch -> loop body -> loop frame -> main), intermediate frames
    // (the loop frame) also need updating; otherwise the next iteration's body reads a
    // stale value when copying from the loop frame.
    // SAFETY: `parent_frame_ptr` points to a same-function frame (set by `setup_frame_chain`);
    // the caller frame is in the Suspended state while the callee executes, so there is no
    // concurrent access.
    let mut written_parent = false;
    let mut ptr = frame.parent_frame_ptr;
    while !ptr.is_null() {
        let f = unsafe { &mut *ptr };
        let local = target.0.wrapping_sub(f.node_offset);
        if (local as usize) < f.value_table.len() {
            f.set_value(NodeId(local), val.clone(), consumer_count);
            written_parent = true;
        }
        ptr = f.parent_frame_ptr;
    }

    // Path 2: write to `root_frame_ptr` (the function root frame) so that same-function
    // closure calls can read the latest value from the root frame.
    if !frame.root_frame_ptr.is_null() {
        let root = unsafe { &mut *frame.root_frame_ptr };
        let local = target.0.wrapping_sub(root.node_offset);
        if (local as usize) < root.value_table.len() {
            root.set_value(NodeId(local), val.clone(), consumer_count);
        } else {
            return NodeResult::Return(make_error_throw("InternalError",
                &format!("writeback target {:?} out of root frame range", target)));
        }
    } else if !written_parent {
        // Path 3: escaped closure (frame chain is null) — write back the upvalue via the
        // `closure_val`'s Cell.
        // When an escaped closure is called across functions, both `parent` and `root`
        // are null, so writeback via the frame chain is impossible.
        // The upvalues inside `closure_val` are wrapped in Cells (see `compute_closure_construct`);
        // we persist the mutation via `Cell::set` so the next call reads the latest value.
        let mut written_cell = false;
        if let Some(ref closure_val) = frame.closure_val {
            if let Value::Ref(arc) = closure_val {
                let upvalues: &[Value] = match arc.as_ref() {
                    crate::value::HeapObj::Closure(c) => &c.upvalues,
                    crate::value::HeapObj::Partial(p) => &p.upvalues,
                    _ => &[],
                };
                if !upvalues.is_empty() {
                    let sg_idx = frame.subgraph_id.0 as usize;
                    for (i, &outer_node) in frame.graph.sg_upvalue_outer_nodes(sg_idx).iter().enumerate() {
                        if outer_node == target && i < upvalues.len() {
                            if let Some(crate::value::HeapObj::Cell(cell)) = upvalues[i].heap_obj() {
                                cell.set(val.clone());
                                written_cell = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
        // Path 4: non-escaped closure root-frame scenario (assignment inside a top-level
        // function) — write to the current frame.
        if !written_cell {
            let local = target.0.wrapping_sub(frame.node_offset);
            if (local as usize) < frame.value_table.len() {
                frame.set_value(NodeId(local), val.clone(), consumer_count);
            } else {
                return NodeResult::Return(make_error_throw("InternalError",
                    &format!("writeback target {:?} out of current frame range", target)));
            }
        }
    }
    NodeResult::Value(val)
}

/// compute_tailrec_writeback (idx 310): a WriteBack specialized for tail-recursion-to-iteration.
///
/// Performs the same writeback logic as `compute_writeback`, but additionally returns
/// `NodeResult::Continue`.
/// In a TailRec loop, when `body_sg` completes:
/// - `Continue` (returned by the rec arm's WriteBack) -> `reset_loop_iteration` (loop continues)
/// - `None` (base arm has no WriteBack) -> loop exits, returning `body_sg`'s return value
pub fn compute_tailrec_writeback(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    // Normal writeback -> Continue (loop continues); out-of-bounds and other errors
    // (NodeResult::Return) -> propagate upward (not silent).
    match compute_writeback(frame, node, _ctx) {
        NodeResult::Value(_) => NodeResult::Continue,
        other => other,
    }
}
