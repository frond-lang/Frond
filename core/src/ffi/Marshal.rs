//! Marshal — Value ↔ C ABI bidirectional conversion (only serves stdlib
//! `@extern("C") #{ }#` dynamic FFI).
//!
//! Encodes a Frond `Value` into a list of `AbiSlot`s according to an `AbiSig` for
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
//! Scalar arrays (i32[], f64[], …) use the same `(Ptr, Int)` shape since the
//! mem scalar-intrinsic batch: the SoA column / element vector is serialized to
//! native-endian bytes, the Int slot carries the ELEMENT count (for u8[] that
//! is identical to its byte count), and mutations are decoded back by the
//! array's own element tag. Not supported (later stages): str out-param returns.
//!
//! Runtime-only single-slot pointer atoms (Lib.lookup): `cstr` maps a str
//! arg to a lone NUL-terminated `Ptr` slot; `cbuf` maps a scalar array arg to
//! a lone `Ptr` slot over a serialized keepalive buffer with post-call
//! writeback. Neither takes a length slot, so system-C signatures with a
//! pointer before other parameters keep their slot positions.

use crate::ffi::Abi::{AbiSig, AbiSlot, AbiType, RetSlot};
use crate::value::{HeapObj, OpaquePointer, PtrKind, Value, ValueTag};

/// Keepalive buffer backing one marshaled str / array argument.
pub enum ArgBuf {
    /// NUL-ended byte buffer for a str argument (`len` excludes the NUL).
    Str { buf: Vec<u8>, len: usize },
    /// Native-endian scalar-array bytes stored as u64 words so the data
    /// pointer handed to C is at least 8-byte aligned — the C side
    /// dereferences TYPED pointers (`int32_t*` etc.) into this buffer.
    /// `nbytes` is the semantic byte length (elements × width), never
    /// including the word padding.
    Array { words: Vec<u64>, nbytes: usize },
}

/// Element width of an FFI-marshalable scalar tag (matches the C pointer
/// types in ffi/Gen.rs `TYPE_MAP` and `ArrayValue::collect_scalar_bytes`).
fn scalar_tag_esize(tag: ValueTag) -> usize {
    match tag {
        ValueTag::I8 | ValueTag::U8 | ValueTag::Bool => 1,
        ValueTag::I16 | ValueTag::U16 => 2,
        ValueTag::I32 | ValueTag::U32 | ValueTag::F32 | ValueTag::Char => 4,
        ValueTag::I64 | ValueTag::U64 | ValueTag::Isize | ValueTag::Usize | ValueTag::F64 => 8,
        ValueTag::F16 => 2,
        ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => 16,
        _ => 0,
    }
}

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
    // Keepalive buffers for str / array args; must outlive the call (str: C
    // functions like strlen/printf require NUL-terminated strings, but Str's
    // Arc<str> has no trailing NULL; arrays: the buffer doubles as the
    // writeback target for C-side mutations).
    let mut buffers: Vec<ArgBuf> = Vec::new();
    // Heap address of each buffer's source array (null for str-arg buffers):
    // alias detection so the SAME array passed as two array params shares ONE
    // buffer instead of round-tripping stale bytes over each other's writeback.
    let mut buf_owners: Vec<*const core::ffi::c_void> = Vec::new();
    // (array Value, keepalive buffer index, element tag) for out-parameter
    // writeback; consumed by `apply_writebacks` after the call returns.
    let mut writebacks: Vec<(Value, usize, ValueTag)> = Vec::new();
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

        // Single-slot pointer atoms (`cstr` / `cbuf`, indices recorded in
        // sig.raw_ptr_params): a lone C pointer slot in the system-C ABI
        // shape. Checked BEFORE the (Ptr, Int) expansion branches so shapes
        // like `(cstr, u64)` or `(u64, cbuf, u32, u8)` are not mistaken for
        // the stdlib `f(data, len)` convention — that mismatch shifts every
        // trailing argument one slot and crashes the callee.
        //   cstr + str arg  → NUL-terminated keepalive copy, no length slot.
        //   cbuf + scalar array arg → serialized keepalive buffer with C-side
        //     mutations written back (out-parameter pattern, e.g.
        //     LLVMGetTargetFromTriple's LLVMTargetRef*); no length slot.
        // Anything else falls through to the generic pointer slot (raw handle
        // passthrough — integers/OpaquePtr, `0u64` yields a NULL pointer).
        if matches!(param, AbiType::Ptr) && sig.raw_ptr_params.contains(&param_idx) {
            match arg.heap_obj() {
                Some(HeapObj::Str(s)) => {
                    let bytes = s.bytes();
                    let mut buf = bytes.as_bytes().to_vec();
                    buf.push(0);
                    let ptr = buf.as_ptr();
                    buffers.push(ArgBuf::Str { buf, len: bytes.len() });
                    buf_owners.push(core::ptr::null());
                    slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
                    param_idx += 1;
                    arg_idx += 1;
                    continue;
                }
                Some(HeapObj::Array(arr)) => {
                    let tag = arr.scalar_marshal_tag().ok_or_else(|| {
                        "encode_args: cbuf argument is not a marshalable scalar array \
                         (non-scalar or mixed elements)"
                            .to_string()
                    })?;
                    if arr.len() == 0 && arr.scalar_soa.is_none() {
                        // Empty cbuf: a per-call keepalive word, NOT a read-only
                        // static — out-parameter APIs write through the pointer,
                        // and C mutating a `*const` static is UB (an access
                        // violation on .rodata pages). Nothing to serialize,
                        // nothing to write back.
                        buffers.push(ArgBuf::Array { words: vec![0u64], nbytes: 0 });
                        let ptr = match buffers.last() {
                            Some(ArgBuf::Array { words, .. }) => words.as_ptr(),
                            _ => unreachable!(),
                        };
                        buf_owners.push(core::ptr::null());
                        slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
                        param_idx += 1;
                        arg_idx += 1;
                        continue;
                    }
                    let (bytes, ser_tag) = arr.collect_scalar_bytes().ok_or_else(|| {
                        "encode_args: cbuf argument is not a marshalable scalar array \
                         (non-scalar or mixed elements)"
                            .to_string()
                    })?;
                    debug_assert_eq!(ser_tag, tag);
                    let words = bytes_to_aligned_words(&bytes);
                    let ptr = words.as_ptr();
                    buffers.push(ArgBuf::Array { words, nbytes: bytes.len() });
                    buf_owners.push(core::ptr::null());
                    slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
                    writebacks.push((arg.clone(), buffers.len() - 1, tag));
                    param_idx += 1;
                    arg_idx += 1;
                    continue;
                }
                _ => {}
            }
        }

        // Detect array expansion: the current param is Ptr, the next is Int, and
        // the arg is HeapObj::Array. The SoA column (or element vector) is
        // serialized to native-endian bytes in an 8-aligned keepalive buffer;
        // the Int slot carries the ELEMENT count — the C side reads elements
        // through its typed `{p}_data` pointer (for u8[] the element count is
        // the byte count, unchanged from the original contract).
        // `apply_writebacks` decodes C-side mutations back into the array
        // (out-parameter pattern, e.g. read_into, Mem.fill).
        if matches!(param, AbiType::Ptr)
            && param_idx + 1 < sig.params.len()
            && matches!(sig.params[param_idx + 1], AbiType::Int { .. })
        {
            if let Some(HeapObj::Array(arr)) = arg.heap_obj() {
                // Aliased array args (the SAME array as two array params, e.g.
                // __mem_copy(dst, .., src, ..) with dst === src) must share ONE
                // buffer: per-arg copies would have C mutate dst's copy only,
                // then apply_writebacks writes src's stale copy back over it —
                // the whole call silently becomes a no-op. Sharing the buffer
                // lets C see the true overlapping windows (memmove semantics
                // hold) and the duplicate writebacks are idempotent.
                let owner = match arg {
                    Value::Ref(arc) => std::sync::Arc::as_ptr(arc) as *const core::ffi::c_void,
                    _ => core::ptr::null(),
                };
                // Empty array (no SoA, no elements — element type invisible):
                // zero-length calls are legal C shapes (e.g. Mem.compare with
                // an empty operand), so hand C a non-null aligned pointer with
                // element count 0. Nothing to serialize, nothing to write back.
                if arr.len() == 0 && arr.scalar_soa.is_none() {
                    // Same per-call empty-buffer rule as the cbuf arm: writable,
                    // 8-aligned, non-null — a legal C shape for zero-length
                    // calls (never a read-only static handed out as *mut).
                    buffers.push(ArgBuf::Array { words: vec![0u64], nbytes: 0 });
                    let ptr = match buffers.last() {
                        Some(ArgBuf::Array { words, .. }) => words.as_ptr(),
                        _ => unreachable!(),
                    };
                    buf_owners.push(owner);
                    slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
                    slots.push(AbiSlot::Int(0));
                    param_idx += 2;
                    arg_idx += 1;
                    continue;
                }
                let tag = arr.scalar_marshal_tag().ok_or_else(|| {
                    "encode_args: array argument is not a marshalable scalar array \
                     (non-scalar or mixed elements)"
                        .to_string()
                })?;
                let existing = if owner.is_null() {
                    None
                } else {
                    buf_owners.iter().position(|p| !p.is_null() && *p == owner)
                };
                let buf_idx = match existing {
                    Some(i) => i,
                    None => {
                        let (bytes, ser_tag) = arr.collect_scalar_bytes().ok_or_else(|| {
                            "encode_args: array argument is not a marshalable scalar array \
                             (non-scalar or mixed elements)"
                                .to_string()
                        })?;
                        debug_assert_eq!(ser_tag, tag);
                        let words = bytes_to_aligned_words(&bytes);
                        buffers.push(ArgBuf::Array { words, nbytes: bytes.len() });
                        buf_owners.push(owner);
                        buffers.len() - 1
                    }
                };
                let (ptr, nbytes) = match &buffers[buf_idx] {
                    ArgBuf::Array { words, nbytes } => (words.as_ptr(), *nbytes),
                    // Same array aliased across a str param is not a real call
                    // shape (the str branch fires on Str args first); reaching
                    // this arm means a type-confused signature.
                    ArgBuf::Str { .. } => {
                        return Err("encode_args: array argument aliases a str buffer".to_string())
                    }
                };
                let esize = scalar_tag_esize(tag);
                let elems = (nbytes / esize) as u64;
                slots.push(AbiSlot::Ptr(ptr as *mut core::ffi::c_void));
                slots.push(AbiSlot::Int(elems));
                writebacks.push((arg.clone(), buf_idx, tag));
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
            buffers.push(ArgBuf::Str { buf, len });
            // str buffers never join the array writeback path; keep the owners
            // index aligned by pushing a null entry.
            buf_owners.push(core::ptr::null());
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
                // f32 params cross in the LOW 32 BITS of the SSE register:
                // every x64/aarch64 C ABI has the callee read `float` from the
                // low half. Widening numerically to f64 leaves the DOUBLE
                // encoding's low half there (1.5f32 → low32 = 0 → the callee
                // reads 0.0f — verified against an MSVC-compiled callee). Pack
                // the f32 BITS into the f64 transport word instead; the high
                // half is a denormal garbage tail the callee never reads.
                let bits = arg.as_f32().to_bits() as u64;
                slots.push(AbiSlot::Float(f64::from_bits(bits)));
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

    Ok(MarshalArgs { slots, _keepalive: buffers, writebacks })
}

/// Return value of `encode_args`: the slots plus the keepalive buffers.
/// The caller must keep this value alive until `Abi::call_dynamic` returns.
pub struct MarshalArgs {
    pub slots: Vec<AbiSlot>,
    /// NUL-ended str buffers / 8-aligned array byte buffers. Prevents C
    /// functions from reading out of bounds and keeps writeback targets alive
    /// for the duration of the call.
    pub _keepalive: Vec<ArgBuf>,
    /// (array Value, keepalive buffer index, element tag) pairs for
    /// out-parameter writeback; consumed by `apply_writebacks` after the call
    /// returns.
    pub writebacks: Vec<(Value, usize, ValueTag)>,
}

/// Copy C-side mutations from the keepalive buffers back into the array heap
/// objects recorded during `encode_args`. Must run after `Abi::call_dynamic`
/// returns (and only if the call actually happened — on dispatch error the C
/// function never ran, so there is nothing to write back).
pub fn apply_writebacks(m: &mut MarshalArgs) {
    if m.writebacks.is_empty() {
        return;
    }
    let buffers = std::mem::take(&mut m._keepalive);
    let wbs = std::mem::take(&mut m.writebacks);
    for (val, idx, tag) in wbs {
        let bytes: &[u8] = match buffers.get(idx) {
            Some(ArgBuf::Array { words, nbytes }) => {
                // SAFETY: `words` is owned by `buffers` and outlives this
                // slice use; `nbytes` is within the word storage by
                // construction (elements × width, padded up to words).
                unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, *nbytes) }
            }
            _ => continue,
        };
        if let Value::Ref(arc) = &val {
            // Shared in-place mutation via `Arc::as_ptr` (same pattern as
            // `compute_array_store`: the engine is single-threaded, and the
            // change is visible to every owner of the Arc).
            let ptr = std::sync::Arc::as_ptr(arc) as *mut HeapObj;
            unsafe {
                if let HeapObj::Array(arr) = &mut *ptr {
                    arr.write_scalar_bytes(bytes, tag);
                }
            }
        }
    }
}

/// Pack bytes into u64-word storage (little copying, 8-byte-aligned pointer).
/// Trailing bytes occupy the final (zero-padded) word.
fn bytes_to_aligned_words(bytes: &[u8]) -> Vec<u64> {
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: copying exactly `bytes.len()` into storage of
    // `words.len() * 8 >= bytes.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr() as *mut u8, bytes.len());
    }
    words
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
