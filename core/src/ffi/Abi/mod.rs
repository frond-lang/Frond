#![allow(non_snake_case)]
//! Abi — C ABI dynamic invoker (architecture-independent).
//!
//! Given a type-erased function pointer + a list of argument values + a signature
//! description, issues a call following the C ABI and retrieves the return value.
//! It does not depend on libffi — instead, for common combinations of
//! (integer argument count × floating-point argument count), a type-erased
//! `extern "C" fn` trampoline is generated, letting rustc handle shadow space /
//! stack alignment / register allocation / stack arguments automatically.
//!
//! **Architecture-independent**: the trampoline table is defined in
//! [`crate::platform::Invoke`] (generated, order-preserving) using fully-typed `extern "C" fn` signatures, so
//! rustc emits the ABI-correct code for each target platform. Differences in
//! register capacity across SysV/Win64/AAPCS do not affect correctness (arguments
//! beyond the register file spill onto the stack automatically), so neither this
//! module nor `platform::Invoke` carries any `#[cfg(target_arch)]`.
//!
//! Aggregate submodules:
//! - [`Sig`]: ABI type and signature descriptions (AbiType / AbiSig / AbiSlot / RetSlot)
//! - [`Dispatch`]: dispatches to the corresponding extern "C" trampoline (architecture-independent)
//! - [`CallDynamic`]: dynamic call entry point (thin forwarding to Dispatch)

pub mod CallDynamic;
pub mod Dispatch;
pub mod Sig;

pub use Sig::{abi_type_from_name, parse_arg_sig, push_abi_types_for_name, AbiSlot, AbiSig, AbiType, RetSlot};
