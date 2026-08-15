//! AbiTable — the `extern "C"` trampoline table for the C ABI dynamic caller.
//!
//! **Architecture-agnostic**: for every combination of `(integer-arg count
//! 0..=MAX_INT) × (float-arg count 0..=MAX_FLOAT)`, this instantiates a
//! concretely-typed `extern "C" fn(u64×I, f64×F) -> u64|f64` signature, letting
//! rustc emit the ABI-correct code (register allocation, shadow space, stack
//! alignment, stack arguments) for **each target platform**. Differences in
//! register capacity between SysV/Win64/AAPCS do not affect correctness —
//! arguments that exceed the register file are spilled to the stack by rustc
//! according to the target ABI, so every platform stays valid.
//!
//! Accordingly, this file contains **no `#[cfg(target_arch)]`**. The same source
//! compiles and passes on x86_64, aarch64, and any other C ABI platform that
//! rustc supports.
//!
//! # Why 117 match arms
//!
//! Rust's `extern "C" fn(...)` parameter type list must be written out verbatim
//! in source — macros cannot splice types dynamically in "type position" (types
//! and expressions occupy distinct syntactic positions). So each `(I, F)`
//! combination needs its own explicit signature line. The two `match (I, F)`
//! blocks below each enumerate 117 arms (`0..=12 × 0..=8`); every arm is minimal
//! (transmute + call), rustc dead-code-eliminates the untaken arms, and the
//! runtime cost is zero.

#![allow(non_snake_case)]

/// Maximum number of integer arguments covered by the trampoline table.
///
/// Takes the upper bound across architectures: aarch64 has 8 general-purpose
/// registers (x0..x7), x86_64 SysV has 6 (rdi/rsi/rdx/rcx/r8/r9), and Win64 has
/// 4. Choosing 8 lets a single source path cover every arch; arguments beyond a
/// platform's register count are spilled to the stack automatically by rustc, so
/// all platforms remain correct.
///
/// 12 (not 8) so multi-buffer FFI shapes fit: e.g. `__dir_list_into(path, names,
/// offsets, kinds, max)` expands to 9 integer ABI slots (str→2, three u8[]→6,
/// usize→1).
pub const MAX_INT: usize = 12;

/// Maximum number of floating-point arguments covered by the trampoline table.
///
/// x86_64 SysV: 8 xmm registers; aarch64: 8 v0..v7 registers. Both platforms
/// have 8.
pub const MAX_FLOAT: usize = 8;

/// Dispatches to the concrete `(int_count, float_count)` trampoline.
///
/// Returns u64: an integer return value when `ret_is_float=false`, or
/// `f64::to_bits` when `true`.
///
/// # Safety
///
/// The caller guarantees: `fn_ptr` is a valid C function pointer;
/// `int_slots`/`float_slots` have been correctly classified per the signature,
/// with lengths `>= int_count` / `>= float_count` respectively; and
/// `(int_count, float_count)` falls within `(0..=MAX_INT, 0..=MAX_FLOAT)`.
pub unsafe fn dispatch(
    int_count: usize,
    float_count: usize,
    ret_is_float: bool,
    fn_ptr: *mut core::ffi::c_void,
    int_slots: &[u64],
    float_slots: &[f64],
) -> u64 {
    macro_rules! route_int {
        ($i:literal) => {{ match float_count {
            0 => call::<$i, 0>(ret_is_float, fn_ptr, int_slots, float_slots),
            1 => call::<$i, 1>(ret_is_float, fn_ptr, int_slots, float_slots),
            2 => call::<$i, 2>(ret_is_float, fn_ptr, int_slots, float_slots),
            3 => call::<$i, 3>(ret_is_float, fn_ptr, int_slots, float_slots),
            4 => call::<$i, 4>(ret_is_float, fn_ptr, int_slots, float_slots),
            5 => call::<$i, 5>(ret_is_float, fn_ptr, int_slots, float_slots),
            6 => call::<$i, 6>(ret_is_float, fn_ptr, int_slots, float_slots),
            7 => call::<$i, 7>(ret_is_float, fn_ptr, int_slots, float_slots),
            8 => call::<$i, 8>(ret_is_float, fn_ptr, int_slots, float_slots),
            _ => unreachable!("float_count > MAX_FLOAT ({MAX_FLOAT})"),
        }
    }};
    }
    match int_count {
        0 => route_int!(0),
        1 => route_int!(1),
        2 => route_int!(2),
        3 => route_int!(3),
        4 => route_int!(4),
        5 => route_int!(5),
        6 => route_int!(6),
        7 => route_int!(7),
        8 => route_int!(8),
        9 => route_int!(9),
        10 => route_int!(10),
        11 => route_int!(11),
        12 => route_int!(12),
        _ => unreachable!("int_count > MAX_INT ({MAX_INT})"),
    }
}

/// Trampoline selecting between integer and floating-point return per
/// `ret_is_float`.
///
/// Returns u64 (when `ret_float`, returns `f64::to_bits`).
///
/// # Safety
///
/// Forwards to the concrete trampoline, inheriting its safety contract.
#[inline(always)]
unsafe fn call<const I: usize, const F: usize>(
    ret_is_float: bool,
    fn_ptr: *mut core::ffi::c_void,
    int_slots: &[u64],
    float_slots: &[f64],
) -> u64 {
    if ret_is_float {
        let r: f64 = call_ret_float::<I, F>(fn_ptr, int_slots, float_slots);
        r.to_bits()
    } else {
        call_ret_int::<I, F>(fn_ptr, int_slots, float_slots)
    }
}

// ─── 117-arm trampoline table (0..=12 int × 0..=8 float) ───────────────
//
// Integer arguments come first, floats after: Marshal splits arguments into
// int_slots/float_slots by type, and each C ABI likewise allocates the
// "integer register sequence" and "float register sequence" independently;
// rustc assigns them to the corresponding registers/stack slots in the
// (u64×I, f64×F) order of this signature.

/// Trampoline for integer returns. Returns u64.
///
/// # Safety
///
/// `fn_ptr` is a valid C function pointer for the corresponding signature;
/// `int_slots.len() >= I` and `float_slots.len() >= F`.
unsafe fn call_ret_int<const I: usize, const F: usize>(
    fn_ptr: *mut core::ffi::c_void,
    int_slots: &[u64],
    float_slots: &[f64],
) -> u64 {
    match (I, F) {
        (0, 0) => { let f: extern "C" fn() -> u64 = core::mem::transmute(fn_ptr); f() }
        (0, 1) => { let f: extern "C" fn(f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0]) }
        (0, 2) => { let f: extern "C" fn(f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1]) }
        (0, 3) => { let f: extern "C" fn(f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2]) }
        (0, 4) => { let f: extern "C" fn(f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (0, 5) => { let f: extern "C" fn(f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (0, 6) => { let f: extern "C" fn(f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (0, 7) => { let f: extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (0, 8) => { let f: extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (1, 0) => { let f: extern "C" fn(u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0]) }
        (1, 1) => { let f: extern "C" fn(u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0]) }
        (1, 2) => { let f: extern "C" fn(u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1]) }
        (1, 3) => { let f: extern "C" fn(u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2]) }
        (1, 4) => { let f: extern "C" fn(u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (1, 5) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (1, 6) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (1, 7) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (1, 8) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (2, 0) => { let f: extern "C" fn(u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1]) }
        (2, 1) => { let f: extern "C" fn(u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0]) }
        (2, 2) => { let f: extern "C" fn(u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1]) }
        (2, 3) => { let f: extern "C" fn(u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2]) }
        (2, 4) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (2, 5) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (2, 6) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (2, 7) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (2, 8) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (3, 0) => { let f: extern "C" fn(u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2]) }
        (3, 1) => { let f: extern "C" fn(u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0]) }
        (3, 2) => { let f: extern "C" fn(u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1]) }
        (3, 3) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2]) }
        (3, 4) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (3, 5) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (3, 6) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (3, 7) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (3, 8) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (4, 0) => { let f: extern "C" fn(u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3]) }
        (4, 1) => { let f: extern "C" fn(u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0]) }
        (4, 2) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1]) }
        (4, 3) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2]) }
        (4, 4) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (4, 5) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (4, 6) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (4, 7) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (4, 8) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (5, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4]) }
        (5, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0]) }
        (5, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1]) }
        (5, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2]) }
        (5, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (5, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (5, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (5, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (5, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (6, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5]) }
        (6, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0]) }
        (6, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1]) }
        (6, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2]) }
        (6, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (6, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (6, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (6, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (6, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (7, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6]) }
        (7, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0]) }
        (7, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1]) }
        (7, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2]) }
        (7, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (7, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (7, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (7, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (7, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (8, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7]) }
        (8, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0]) }
        (8, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1]) }
        (8, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2]) }
        (8, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (8, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (8, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (8, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (8, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (9, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8]) }
        (9, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0]) }
        (9, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1]) }
        (9, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2]) }
        (9, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (9, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (9, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (9, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (9, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (10, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9]) }
        (10, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0]) }
        (10, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1]) }
        (10, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2]) }
        (10, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (10, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (10, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (10, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (10, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (11, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10]) }
        (11, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0]) }
        (11, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1]) }
        (11, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2]) }
        (11, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (11, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (11, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (11, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (11, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (12, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11]) }
        (12, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0]) }
        (12, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1]) }
        (12, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2]) }
        (12, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (12, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (12, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (12, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (12, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> u64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        _ => unreachable!("call_ret_int: (I={I}, F={F}) out of table range"),
    }
}

/// Trampoline for floating-point returns. Returns f64 (the caller then applies
/// `to_bits`).
///
/// # Safety
///
/// Same as [`call_ret_int`].
unsafe fn call_ret_float<const I: usize, const F: usize>(
    fn_ptr: *mut core::ffi::c_void,
    int_slots: &[u64],
    float_slots: &[f64],
) -> f64 {
    match (I, F) {
        (0, 0) => { let f: extern "C" fn() -> f64 = core::mem::transmute(fn_ptr); f() }
        (0, 1) => { let f: extern "C" fn(f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0]) }
        (0, 2) => { let f: extern "C" fn(f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1]) }
        (0, 3) => { let f: extern "C" fn(f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2]) }
        (0, 4) => { let f: extern "C" fn(f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (0, 5) => { let f: extern "C" fn(f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (0, 6) => { let f: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (0, 7) => { let f: extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (0, 8) => { let f: extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (1, 0) => { let f: extern "C" fn(u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0]) }
        (1, 1) => { let f: extern "C" fn(u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0]) }
        (1, 2) => { let f: extern "C" fn(u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1]) }
        (1, 3) => { let f: extern "C" fn(u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2]) }
        (1, 4) => { let f: extern "C" fn(u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (1, 5) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (1, 6) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (1, 7) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (1, 8) => { let f: extern "C" fn(u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (2, 0) => { let f: extern "C" fn(u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1]) }
        (2, 1) => { let f: extern "C" fn(u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0]) }
        (2, 2) => { let f: extern "C" fn(u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1]) }
        (2, 3) => { let f: extern "C" fn(u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2]) }
        (2, 4) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (2, 5) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (2, 6) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (2, 7) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (2, 8) => { let f: extern "C" fn(u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (3, 0) => { let f: extern "C" fn(u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2]) }
        (3, 1) => { let f: extern "C" fn(u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0]) }
        (3, 2) => { let f: extern "C" fn(u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1]) }
        (3, 3) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2]) }
        (3, 4) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (3, 5) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (3, 6) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (3, 7) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (3, 8) => { let f: extern "C" fn(u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (4, 0) => { let f: extern "C" fn(u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3]) }
        (4, 1) => { let f: extern "C" fn(u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0]) }
        (4, 2) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1]) }
        (4, 3) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2]) }
        (4, 4) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (4, 5) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (4, 6) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (4, 7) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (4, 8) => { let f: extern "C" fn(u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (5, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4]) }
        (5, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0]) }
        (5, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1]) }
        (5, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2]) }
        (5, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (5, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (5, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (5, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (5, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (6, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5]) }
        (6, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0]) }
        (6, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1]) }
        (6, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2]) }
        (6, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (6, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (6, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (6, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (6, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (7, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6]) }
        (7, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0]) }
        (7, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1]) }
        (7, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2]) }
        (7, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (7, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (7, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (7, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (7, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (8, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7]) }
        (8, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0]) }
        (8, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1]) }
        (8, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2]) }
        (8, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (8, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (8, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (8, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (8, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (9, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8]) }
        (9, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0]) }
        (9, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1]) }
        (9, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2]) }
        (9, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (9, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (9, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (9, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (9, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (10, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9]) }
        (10, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0]) }
        (10, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1]) }
        (10, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2]) }
        (10, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (10, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (10, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (10, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (10, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (11, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10]) }
        (11, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0]) }
        (11, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1]) }
        (11, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2]) }
        (11, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (11, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (11, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (11, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (11, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        (12, 0) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11]) }
        (12, 1) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0]) }
        (12, 2) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1]) }
        (12, 3) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2]) }
        (12, 4) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3]) }
        (12, 5) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4]) }
        (12, 6) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5]) }
        (12, 7) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6]) }
        (12, 8) => { let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 = core::mem::transmute(fn_ptr); f(int_slots[0], int_slots[1], int_slots[2], int_slots[3], int_slots[4], int_slots[5], int_slots[6], int_slots[7], int_slots[8], int_slots[9], int_slots[10], int_slots[11], float_slots[0], float_slots[1], float_slots[2], float_slots[3], float_slots[4], float_slots[5], float_slots[6], float_slots[7]) }
        _ => unreachable!("call_ret_float: (I={I}, F={F}) out of table range"),
    }
}

// Note: every `match` arm in `call_ret_int` / `call_ret_float` uses
// `core::mem::transmute(fn_ptr)` to re-interpret the type-erased pointer as a
// concrete `extern "C" fn`. This cannot be factored into a generic helper — in
// that form `T` has no fixed size and transmute would fail to compile. By
// inlining `core::mem::transmute` in each arm, `T` is pinned to a concrete
// function-pointer type (pointer-sized, like `*mut c_void`), so the conversion
// is safe.
