//! Marshal — Value ↔ C ABI bidirectional conversion (only serves stdlib
//! `@extern("C") #{ }#` dynamic FFI).
//!
//! Encodes a Kuzo `Value` into a list of `AbiSlot`s according to an `AbiSig` for
//! use by the [`crate::ffi::Abi`] invoker; after the call returns, decodes the
//! `RetSlot` back into a `Value`.
//!
//! Supported types (v1):
//! - Integers (i8-i64/u8-u64/bool/char/isize/usize) → `AbiSlot::Int`
//! - Floating-point (f32/f64) → `AbiSlot::Float`
//! - Pointers (`HeapObj::OpaquePtr` or scalar pointers) → `AbiSlot::Ptr`
//! - str (`HeapObj::Str`) → split into two slots `(Ptr, Int)` (already expanded
//!   at sig construction time)
//!
//! Supported since the u8[] expansion fix: `u8[]` → `(Ptr, Int)` (data + length),
//! with post-call writeback of C-side mutations (`apply_writebacks`).
//! Not supported (later stages): str out-param returns.

use crate::ffi::Abi::{AbiSig, AbiSlot, AbiType, RetSlot};
use crate::value::{HeapObj, OpaquePointer, PtrKind, ScalarSoA, Value};

/// Encode `args` into a list of `AbiSlot`s according to `sig.params`.
///
/// The caller must ensure `args.len()` matches `sig.params.len()` (str is already
/// expanded into two `AbiType`s at sig construction time, corresponding to a
/// single str argument in the original `Value` list).
///
/// Returns `MarshalArgs` (carrying the slots + NULL-buffer keepalive). The caller
/// must keep `MarshalArgs` alive until `Abi::call_dynamic` returns, otherwise str
/// pointers may dangle.
///
/// Returns `Err(msg)` on a type mismatch.
pub fn encode_args(sig: &AbiSig, args: &[Value]) -> Result<MarshalArgs, String> {
    // NULL-ended buffers for str args (and plain byte buffers for u8[] args); must
    // outlive the call (str: C functions like strlen/printf require NULL-terminated
    // strings, but Str's Arc<str> has no trailing NULL; u8[]: the buffer doubles as
    // the writeback target for C-side mutations).
    let mut str_buffers: Vec<Vec<u8>> = Vec::new();
    let mut writebacks: Vec<(Value, usize)> = Vec::new();
    let mut slots = Vec::with_capacity(sig.params.len());
    let mut arg_idx = 0usize;
    let mut param_idx = 0usize;

    while param_idx < sig.params.len() {
        if arg_idx >= args.len() {
            return Err(format!(
                "encode_args: not enough arguments (param_idx={param_idx}, arg_idx={arg_idx})"
            ));
        }
        let param = &sig.params[param_idx];
        let arg = &args[arg_idx];

        // Detect u8[] expansion: the current param is Ptr, the next is Int, and the
        // arg is HeapObj::Array. Bytes are copied into a keepalive buffer whose
        // pointer is handed to C; `apply_writebacks` copies C-side mutations back
        // into the array after the call (out-parameter pattern, e.g. read_into).
        if matches!(param, AbiType::Ptr)
            && param_idx + 1 < sig.params.len()
            && matches!(sig.params[param_idx + 1], AbiType::Int { .. })
        {
            if let Some(HeapObj::Array(arr)) = arg.heap_obj() {
                let bytes = arr.collect_u8_bytes();
                let len = bytes.len();
                let ptr = bytes.as_ptr();
                str_buffers.push(bytes);
                let buf_idx = str_buffers.len() - 1;
                slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
                slots.push(AbiSlot::Int(len as u64));
                writebacks.push((arg.clone(), buf_idx));
                param_idx += 2;
                arg_idx += 1;
                continue;
            }
        }

        // Detect str expansion: the current param is Ptr, the next is Int, and the arg is
        // HeapObj::Str.
        if matches!(param, AbiType::Ptr)
            && param_idx + 1 < sig.params.len()
            && matches!(sig.params[param_idx + 1], AbiType::Int { .. })
            && matches!(arg.heap_obj(), Some(HeapObj::Str(_)))
        {
            // str → (NULL-ended data_ptr, len)
            let s = match arg.heap_obj() {
                Some(HeapObj::Str(s)) => s,
                _ => unreachable!(),
            };
            let bytes = s.bytes();
            // Build a NULL-terminated copy so C functions (strlen/printf/etc.) see a valid C string.
            let mut buf = bytes.as_bytes().to_vec();
            buf.push(0);
            let len = bytes.len();
            let ptr = buf.as_ptr();
            str_buffers.push(buf);
            slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
            slots.push(AbiSlot::Int(len as u64));
            param_idx += 2;
            arg_idx += 1;
            continue;
        }

        // Single-slot parameter.
        match param {
            AbiType::Int { bits: _, signed: _ } => {
                slots.push(AbiSlot::Int(arg.as_i64() as u64));
            }
            AbiType::Float32 => {
                slots.push(AbiSlot::Float(arg.as_f32() as f64));
            }
            AbiType::Float64 => {
                slots.push(AbiSlot::Float(arg.as_f64()));
            }
            AbiType::Ptr => {
                slots.push(value_to_ptr_slot(arg));
            }
            AbiType::Void => {
                return Err("encode_args: Void is not a valid parameter type".to_string());
            }
        }
        param_idx += 1;
        arg_idx += 1;
    }

    Ok(MarshalArgs { slots, _keepalive: str_buffers, writebacks })
}

/// Return value of `encode_args`: the slots plus the NULL-buffer keepalive.
/// The caller must keep this value alive until `Abi::call_dynamic` returns.
pub struct MarshalArgs {
    pub slots: Vec<AbiSlot>,
    /// NULL-ended str buffers / u8[] byte buffers. Prevents C functions from
    /// reading out of bounds and keeps u8[] writeback targets alive for the
    /// duration of the call.
    pub _keepalive: Vec<Vec<u8>>,
    /// (array Value, keepalive buffer index) pairs for u8[] out-parameter
    /// writeback; consumed by `apply_writebacks` after the call returns.
    pub writebacks: Vec<(Value, usize)>,
}

/// Copy C-side mutations from the keepalive buffers back into the u8[] heap
/// objects recorded during `encode_args`. Must run after `Abi::call_dynamic`
/// returns (and only if the call actually happened — on dispatch error the C
/// function never ran, so there is nothing to write back).
pub fn apply_writebacks(m: &mut MarshalArgs) {
    if m.writebacks.is_empty() {
        return;
    }
    let buffers = std::mem::take(&mut m._keepalive);
    let wbs = std::mem::take(&mut m.writebacks);
    for (val, idx) in wbs {
        if let Some(bytes) = buffers.get(idx) {
            write_bytes_back(&val, bytes);
        }
    }
}

/// Write bytes into a `u8[]` heap object in place via `Arc::as_ptr` (the same
/// shared-mutation pattern as `compute_array_store`: the engine is
/// single-threaded, and the change is visible to every owner of the Arc).
/// SOA U8 storage is the source of truth when present (mirrors
/// `ArrayValue::collect_u8_bytes`' read preference); otherwise the per-element
/// `elements` vector is updated.
fn write_bytes_back(val: &Value, bytes: &[u8]) {
    if let Value::Ref(arc) = val {
        let ptr = std::sync::Arc::as_ptr(arc) as *mut HeapObj;
        unsafe {
            if let HeapObj::Array(arr) = &mut *ptr {
                if let Some(ScalarSoA::U8(ref mut data)) = arr.scalar_soa {
                    let n = bytes.len().min(data.len());
                    data[..n].copy_from_slice(&bytes[..n]);
                    return;
                }
                let n = bytes.len().min(arr.elements.len());
                for i in 0..n {
                    arr.elements[i] = Value::u8(bytes[i]);
                }
            }
        }
    }
}

/// Decode `RetSlot` back into a `Value` according to `ret_type`.
pub fn decode_ret(ret_type: &AbiType, slot: RetSlot) -> Value {
    match (ret_type, slot) {
        (AbiType::Void, _) => Value::VOID,
        (AbiType::Int { bits, signed }, RetSlot::Int(v)) => {
            int_u64_to_value(v, *bits, *signed)
        }
        (AbiType::Float32, RetSlot::Float(f)) => Value::f32(f as f32),
        (AbiType::Float64, RetSlot::Float(f)) => Value::f64(f),
        (AbiType::Ptr, RetSlot::Int(v)) => {
            // Pointer return (the Abi invoker also places ptr returns into RetSlot::Int).
            u64_to_opaque_ptr_value(v)
        }
        (AbiType::Ptr, RetSlot::Ptr(p)) => {
            opaque_ptr_to_value(p)
        }
        _ => {
            // Mismatched type combination: return Void as a safe fallback (the caller
            // is expected to provide a correct signature).
            Value::VOID
        }
    }
}

/// Extract a pointer slot from a `Value`. Supports `HeapObj::OpaquePtr` and
/// scalar pointers (pointers stored as integers).
fn value_to_ptr_slot(arg: &Value) -> AbiSlot {
    match arg.heap_obj() {
        Some(HeapObj::OpaquePtr(op)) => AbiSlot::Ptr(op.ptr),
        _ => {
            // Scalar pointer (e.g. `*u8` stored as an integer) — read as an integer
            // and turn it into a pointer slot.
            AbiSlot::Ptr(arg.as_u64() as *mut core::ffi::c_void)
        }
    }
}

/// Build the corresponding scalar `Value` from a `u64` return value, selecting
/// the width and signedness.
fn int_u64_to_value(v: u64, bits: u8, signed: bool) -> Value {
    match (bits, signed) {
        (8, true) => Value::i8(v as i8),
        (8, false) => Value::u8(v as u8),
        (16, true) => Value::i16(v as i16),
        (16, false) => Value::u16(v as u16),
        (32, true) => Value::i32(v as i32),
        (32, false) => Value::u32(v as u32),
        (64, true) => Value::i64(v as i64),
        (64, false) => Value::u64(v as u64),
        _ => {
            // Unknown bit width: default to i64.
            Value::i64(v as i64)
        }
    }
}

/// Wrap a `u64` (a pointer address) into a `HeapObj::OpaquePtr` `Value`
/// (`Borrowed`, no destructor).
fn u64_to_opaque_ptr_value(addr: u64) -> Value {
    opaque_ptr_to_value(addr as *mut core::ffi::c_void)
}

/// Wrap a raw pointer into a `HeapObj::OpaquePtr` `Value` (`Borrowed`, no
/// destructor).
fn opaque_ptr_to_value(ptr: *mut core::ffi::c_void) -> Value {
    Value::ref_val(HeapObj::OpaquePtr(OpaquePointer {
        ptr,
        kind: PtrKind::Borrowed,
        type_name: "ptr",
        destructor: None,
    }))
}
