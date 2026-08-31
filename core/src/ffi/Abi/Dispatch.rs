//! Dispatch — dispatches to the corresponding extern "C" trampoline
//! (architecture-independent).
//!
//! Core idea: for each `(integer argument count, floating-point argument count)`
//! combination, generate a fully-typed `extern "C" fn` call wrapper, letting rustc
//! handle the calling-convention details automatically (register allocation, stack
//! alignment, shadow space, stack arguments). At call time the target function
//! pointer is `transmute`d into the matching signature and called directly.
//!
//! Key design: **no arch-based dispatch**. The trampoline table is defined in
//! [`crate::platform::AbiTable`] using fully-typed `extern "C" fn` signatures, so
//! rustc emits ABI-correct code for each target platform. Differences in register
//! capacity across SysV/Win64/AAPCS do not affect correctness — arguments beyond
//! the register file spill onto the stack automatically, which is valid on every
//! platform. Accordingly this file carries no `#[cfg(target_arch)]` at all.
//!
//! Return values come in two wrapper forms: `_ret_int` (returns `u64`, covering
//! integer / pointer / void returns) and `_ret_float` (returns `f64`, covering
//! floating-point returns). The caller picks one based on the signature.
//!
//! Limitations:
//! - The argument count is bounded by the trampoline table's coverage (see
//!   `AbiTable::MAX_INT/MAX_FLOAT`); exceeding it returns `Err`. This bound is
//!   simply "the size the table is currently generated to" and is platform-
//!   independent — expanding the table supports more arguments.
//! - Stack arguments are handled by rustc following extern "C", consistently
//!   across platforms.

use super::Sig::{AbiSig, AbiSlot, AbiType, RetSlot};
use crate::platform::AbiTable;

/// Entry point: split `args` into integer slots / floating-point slots according
/// to the signature and dispatch to the corresponding trampoline.
pub fn dispatch(
    sig: &AbiSig,
    fn_ptr: *mut core::ffi::c_void,
    args: &[AbiSlot],
) -> Result<RetSlot, &'static str> {
    debug_assert_eq!(
        args.len(),
        sig.params.len(),
        "dispatch: args.len must match sig.params.len"
    );

    // 1. Split AbiSlots into integer slots / floating-point slots by signature
    //    (preserving order).
    let mut int_slots: Vec<u64> = Vec::with_capacity(AbiTable::MAX_INT);
    let mut float_slots: Vec<f64> = Vec::with_capacity(AbiTable::MAX_FLOAT);
    for (param, slot) in sig.params.iter().zip(args.iter()) {
        match (param, *slot) {
            (AbiType::Int { .. }, AbiSlot::Int(v)) => int_slots.push(v),
            (AbiType::Int { .. }, AbiSlot::Ptr(p)) => int_slots.push(p as u64),
            (AbiType::Ptr, AbiSlot::Ptr(p)) => int_slots.push(p as u64),
            (AbiType::Ptr, AbiSlot::Int(v)) => int_slots.push(v),
            (AbiType::Float32 | AbiType::Float64, AbiSlot::Float(f)) => float_slots.push(f),
            _ => return Err("dispatch: AbiSlot kind does not match AbiType"),
        }
    }

    // 2. Check that the counts stay within the trampoline table's coverage
    //    (arguments beyond the register file are spilled by rustc automatically;
    //    this only limits to the combinations the table currently generates).
    if int_slots.len() > AbiTable::MAX_INT {
        return Err("dispatch: integer argument count exceeds trampoline table coverage");
    }
    if float_slots.len() > AbiTable::MAX_FLOAT {
        return Err("dispatch: float argument count exceeds trampoline table coverage");
    }

    // 3. Dispatch to the trampoline by (int_count, float_count, ret_kind).
    let int_count = int_slots.len();
    let float_count = float_slots.len();
    let ret_is_float = matches!(sig.ret, AbiType::Float32 | AbiType::Float64);

    // SAFETY: `fn_ptr` is a valid function pointer guaranteed by the upper
    //         Marshal/Loader layer; `int_slots`/`float_slots` are already
    //         correctly classified per the signature.
    let ret_raw = unsafe {
        AbiTable::dispatch(
            int_count,
            float_count,
            ret_is_float,
            fn_ptr,
            &int_slots,
            &float_slots,
        )
    };

    // 4. Assemble RetSlot according to `sig.ret`.
    Ok(match sig.ret {
        AbiType::Void => RetSlot::Void,
        // The trampoline reads the raw 64-bit xmm0 return. A C `float` result
        // lives in the LOW 32 bits (upper bits undefined), so reinterpret those
        // as f32 before widening; a `double` result uses all 64 bits.
        AbiType::Float32 => RetSlot::Float(f64::from(f32::from_bits(ret_raw as u32))),
        AbiType::Float64 => RetSlot::Float(f64::from_bits(ret_raw)),
        AbiType::Int { .. } | AbiType::Ptr => {
            // Integer / pointer return; the low 64 bits of `ret_raw` are the value.
            RetSlot::Int(ret_raw)
        }
    })
}
