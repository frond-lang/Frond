//! CallDynamic — C ABI dynamic call entry point (thin forwarding to Dispatch).
//!
//! Moved out of `Abi/mod.rs` to honor the mod.rs zero-function rule.

use super::Sig::{AbiSig, AbiSlot, RetSlot};

/// Dynamically call a function pointer following the C ABI.
///
/// - `sig`: signature description (parameter types + return type)
/// - `fn_ptr`: target function pointer (type-erased)
/// - `args`: already-marshalled argument slots (one-to-one with `sig.params`)
///
/// Returns `Result`: on success yields `RetSlot`; returns `Err` when the argument
/// count falls outside the trampoline table's coverage.
pub fn call_dynamic(
    sig: &AbiSig,
    fn_ptr: *mut core::ffi::c_void,
    args: &[AbiSlot],
) -> Result<RetSlot, &'static str> {
    super::Dispatch::dispatch(sig, fn_ptr, args)
}
