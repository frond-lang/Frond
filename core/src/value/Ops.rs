// =========================================================================
// Ops — Num/BitOps trait + cast + batch/SIMD + allocator + pure arithmetic core
// =========================================================================

use std::hash::Hash;

use rayon::prelude::*;
use pastey::paste;
use wide::{f32x4, f64x4, i8x16, i16x8, i32x4, i64x4, u8x16, u16x8, u32x4, u64x4};

pub use super::Tag::ValueTag;

use super::value::*;

// =========================================================================
// Part 11: ops.rs (Num trait + BitOps trait + impl)
// =========================================================================

/// Numeric operation trait: arithmetic operations with overflow detection.
pub trait Num: Sized + Copy {
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;
    fn checked_mul(self, other: Self) -> Option<Self>;
    fn checked_div(self, other: Self) -> Option<Self>;
    fn checked_rem(self, other: Self) -> Option<Self>;
    fn neg(self) -> Option<Self>;
    fn zero() -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_neg(self) -> Self;
    fn abs(self) -> Self;
    fn to_u32(self) -> u32;
}

macro_rules! impl_num_signed {
    ($($t:ty),*) => {
        $(
            impl Num for $t {
                fn checked_add(self, other: Self) -> Option<Self> { self.checked_add(other) }
                fn checked_sub(self, other: Self) -> Option<Self> { self.checked_sub(other) }
                fn checked_mul(self, other: Self) -> Option<Self> { self.checked_mul(other) }
                fn checked_div(self, other: Self) -> Option<Self> { self.checked_div(other) }
                fn checked_rem(self, other: Self) -> Option<Self> { self.checked_rem(other) }
                fn neg(self) -> Option<Self> { self.checked_neg() }
                fn zero() -> Self { 0 }
                fn wrapping_add(self, other: Self) -> Self { self.wrapping_add(other) }
                fn wrapping_sub(self, other: Self) -> Self { self.wrapping_sub(other) }
                fn wrapping_mul(self, other: Self) -> Self { self.wrapping_mul(other) }
                fn wrapping_neg(self) -> Self { self.wrapping_neg() }
                fn abs(self) -> Self { self.wrapping_abs() }
                fn to_u32(self) -> u32 { self as u32 }
            }
        )*
    };
}

macro_rules! impl_num_unsigned {
    ($($t:ty),*) => {
        $(
            impl Num for $t {
                fn checked_add(self, other: Self) -> Option<Self> { self.checked_add(other) }
                fn checked_sub(self, other: Self) -> Option<Self> { self.checked_sub(other) }
                fn checked_mul(self, other: Self) -> Option<Self> { self.checked_mul(other) }
                fn checked_div(self, other: Self) -> Option<Self> { self.checked_div(other) }
                fn checked_rem(self, other: Self) -> Option<Self> { self.checked_rem(other) }
                fn neg(self) -> Option<Self> { self.checked_neg() }
                fn zero() -> Self { 0 }
                fn wrapping_add(self, other: Self) -> Self { self.wrapping_add(other) }
                fn wrapping_sub(self, other: Self) -> Self { self.wrapping_sub(other) }
                fn wrapping_mul(self, other: Self) -> Self { self.wrapping_mul(other) }
                fn wrapping_neg(self) -> Self { self.wrapping_neg() }
                fn abs(self) -> Self { self }
                fn to_u32(self) -> u32 { self as u32 }
            }
        )*
    };
}

impl_num_signed!(i8, i16, i32, i64, i128, isize);
impl_num_unsigned!(u8, u16, u32, u64, u128, usize);

impl Num for f32 {
    fn checked_add(self, other: Self) -> Option<Self> { Some(self + other) }
    fn checked_sub(self, other: Self) -> Option<Self> { Some(self - other) }
    fn checked_mul(self, other: Self) -> Option<Self> { Some(self * other) }
    fn checked_div(self, other: Self) -> Option<Self> { Some(self / other) }
    fn checked_rem(self, other: Self) -> Option<Self> { Some(self % other) }
    fn neg(self) -> Option<Self> { Some(-self) }
    fn zero() -> Self { 0.0 }
    fn wrapping_add(self, other: Self) -> Self { self + other }
    fn wrapping_sub(self, other: Self) -> Self { self - other }
    fn wrapping_mul(self, other: Self) -> Self { self * other }
    fn wrapping_neg(self) -> Self { -self }
    fn abs(self) -> Self { self.abs() }
    fn to_u32(self) -> u32 { self as u32 }
}

impl Num for f64 {
    fn checked_add(self, other: Self) -> Option<Self> { Some(self + other) }
    fn checked_sub(self, other: Self) -> Option<Self> { Some(self - other) }
    fn checked_mul(self, other: Self) -> Option<Self> { Some(self * other) }
    fn checked_div(self, other: Self) -> Option<Self> { Some(self / other) }
    fn checked_rem(self, other: Self) -> Option<Self> { Some(self % other) }
    fn neg(self) -> Option<Self> { Some(-self) }
    fn zero() -> Self { 0.0 }
    fn wrapping_add(self, other: Self) -> Self { self + other }
    fn wrapping_sub(self, other: Self) -> Self { self - other }
    fn wrapping_mul(self, other: Self) -> Self { self * other }
    fn wrapping_neg(self) -> Self { -self }
    fn abs(self) -> Self { self.abs() }
    fn to_u32(self) -> u32 { self as u32 }
}

// F16 Num impl: delegates to exact IEEE 754 binary16 arithmetic (no f64 intermediate)
impl Num for F16 {
    fn checked_add(self, other: Self) -> Option<Self> { Some(self + other) }
    fn checked_sub(self, other: Self) -> Option<Self> { Some(self - other) }
    fn checked_mul(self, other: Self) -> Option<Self> { Some(self * other) }
    fn checked_div(self, other: Self) -> Option<Self> { Some(self / other) }
    fn checked_rem(self, other: Self) -> Option<Self> { Some(self % other) }
    fn neg(self) -> Option<Self> { Some(-self) }
    fn zero() -> Self { F16(0) }
    fn wrapping_add(self, other: Self) -> Self { self + other }
    fn wrapping_sub(self, other: Self) -> Self { self - other }
    fn wrapping_mul(self, other: Self) -> Self { self * other }
    fn wrapping_neg(self) -> Self { -self }
    fn abs(self) -> Self {
        // Clear the sign bit
        F16(self.0 & 0x7FFF)
    }
    fn to_u32(self) -> u32 { self.to_f32() as u32 }
}

// F128 Num impl: delegates to exact IEEE 754 binary128 arithmetic (no f64 intermediate)
impl Num for F128 {
    fn checked_add(self, other: Self) -> Option<Self> { Some(self + other) }
    fn checked_sub(self, other: Self) -> Option<Self> { Some(self - other) }
    fn checked_mul(self, other: Self) -> Option<Self> { Some(self * other) }
    fn checked_div(self, other: Self) -> Option<Self> { Some(self / other) }
    fn checked_rem(self, other: Self) -> Option<Self> { Some(self % other) }
    fn neg(self) -> Option<Self> { Some(-self) }
    fn zero() -> Self { F128::from_f64(0.0) }
    fn wrapping_add(self, other: Self) -> Self { self + other }
    fn wrapping_sub(self, other: Self) -> Self { self - other }
    fn wrapping_mul(self, other: Self) -> Self { self * other }
    fn wrapping_neg(self) -> Self { -self }
    fn abs(self) -> Self {
        // Clear the sign bit (bit 127)
        let bits = u128::from_le_bytes(self.0) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;
        F128(bits.to_le_bytes())
    }
    fn to_u32(self) -> u32 { self.to_f64() as u32 }
}

/// Bitwise operation trait.
pub trait BitOps: Sized + Copy {
    fn bit_and(self, other: Self) -> Self;
    fn bit_or(self, other: Self) -> Self;
    fn bit_xor(self, other: Self) -> Self;
    fn bit_not(self) -> Self;
    fn shl(self, amount: u32) -> Self;
    fn shr(self, amount: u32) -> Self;
}

macro_rules! impl_bitops {
    ($($t:ty),*) => {
        $(
            impl BitOps for $t {
                fn bit_and(self, other: Self) -> Self { self & other }
                fn bit_or(self, other: Self) -> Self { self | other }
                fn bit_xor(self, other: Self) -> Self { self ^ other }
                fn bit_not(self) -> Self { !self }
                fn shl(self, amount: u32) -> Self {
                    // SIMD batch path: shift out of bounds returns the original value (wrapping semantics); single-node path returns Throw
                    if amount >= Self::BITS { self } else { self.wrapping_shl(amount) }
                }
                fn shr(self, amount: u32) -> Self {
                    if amount >= Self::BITS { self } else { self.wrapping_shr(amount) }
                }
            }
        )*
    };
}

impl_bitops!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

// =========================================================================
// Part 13: batch.rs (slimmed)
// =========================================================================

/// Binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum BinOp {
    Add = 0, Sub = 1, Mul = 2, Div = 3, Mod = 4, Band = 5, Bor = 6, Bxor = 7, Shl = 8, Shr = 9,
}

/// Unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum UnaryOp {
    Neg = 0, Abs = 1, Bnot = 2,
}

/// Comparison operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum CmpOp {
    Lt = 0, Gt = 1, Eq = 2, Ne = 3, Le = 4, Ge = 5,
}

/// Reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add, Mul, Band, Bor, Bxor,
}

/// Large-array parallel threshold: arrays longer than this use rayon parallel chunking.
pub const PARALLEL_THRESHOLD: usize = 4096;

/// Computes the parallel chunk size: slices the array into roughly (thread_count × 4) chunks,
/// aligned to 4 lanes so each SIMD kernel chunk can fill whole lanes.
#[inline]
pub fn par_chunk_size(n: usize) -> usize {
    let pieces = rayon::current_num_threads().max(1) * 4;
    let chunk = n.div_ceil(pieces);
    // Round up to a multiple of 4
    let chunk = (chunk + 3) & !3;
    chunk.max(4)
}

/// Generic binary operation dispatch (scalar path). Large arrays (> PARALLEL_THRESHOLD) use rayon parallelism;
/// small arrays use single-threaded scalar code to avoid thread-scheduling overhead.
pub fn batch_binop<T>(dst: &mut [T], a: &[T], b: &[T], op: BinOp)
where
    T: Num + BitOps + Send + Sync,
{
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| {
                let m = d.len();
                for i in 0..m {
                    d[i] = binop_scalar_t(av[i], bv[i], op);
                }
            });
    } else {
        for i in 0..n {
            dst[i] = binop_scalar_t(a[i], b[i], op);
        }
    }
}

/// Scalar binary operation (generic fallback; semantically identical to the original for loop).
#[inline]
fn binop_scalar_t<T: Num + BitOps>(a: T, b: T, op: BinOp) -> T {
    match op {
        // Bug #75: unified wrapping semantics (consistent with Bug #22).
        // Integer wrapping_add/sub/mul wraps; floating point wrapping_* is equivalent to native + - * (Num trait implementation).
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        // SIMD batch path: divide-by-zero returns 0 (wrapping semantics); single-node path returns Throw.
        // Floating point checked_div/checked_rem always returns Some (native / produces inf/nan), does not trigger panic.
        BinOp::Div => a.checked_div(b).unwrap_or_else(T::zero),
        BinOp::Mod => a.checked_rem(b).unwrap_or_else(T::zero),
        BinOp::Band => a.bit_and(b),
        BinOp::Bor => a.bit_or(b),
        BinOp::Bxor => a.bit_xor(b),
        BinOp::Shl => a.shl(b.to_u32()),
        BinOp::Shr => a.shr(b.to_u32()),
    }
}

/// Generic unary operation dispatch.
pub fn batch_unaryop<T>(dst: &mut [T], a: &[T], op: UnaryOp)
where T: Num + BitOps {
    let n = dst.len().min(a.len());
    for i in 0..n {
        dst[i] = match op {
            // Bug #75: unified wrapping semantics (consistent with Bug #22).
            UnaryOp::Neg => a[i].wrapping_neg(),
            UnaryOp::Abs => a[i].abs(),
            UnaryOp::Bnot => a[i].bit_not(),
        };
    }
}

/// Batch comparison operation: outputs a `u8` mask (0/1). Large arrays use rayon parallelism; small arrays use scalar code.
pub fn batch_cmp<T>(dst: &mut [u8], a: &[T], b: &[T], op: CmpOp)
where
    T: PartialOrd + Sync,
{
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| {
                let m = d.len();
                for i in 0..m {
                    d[i] = cmp_scalar_t(&av[i], &bv[i], op) as u8;
                }
            });
    } else {
        for i in 0..n {
            dst[i] = cmp_scalar_t(&a[i], &b[i], op) as u8;
        }
    }
}

/// Scalar comparison (generic fallback; compares by reference, so it does not require `T: Copy`).
#[inline]
fn cmp_scalar_t<T: PartialOrd + ?Sized>(a: &T, b: &T, op: CmpOp) -> bool {
    match op {
        CmpOp::Lt => a < b,
        CmpOp::Gt => a > b,
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Le => a <= b,
        CmpOp::Ge => a >= b,
    }
}

// =========================================================================
// Part 13 supplement: SIMD acceleration specializations (wide crate + rayon)
//
// Provides standalone SIMD-specialized functions (4-wide) for f32/f64/i32/i64.
// - Arithmetic/bitwise ops run on SIMD lanes; operations that cannot be vectorized
//   (integer Div/Mod/Shl/Shr, float Mod) fall back to scalar;
// - Large arrays (> PARALLEL_THRESHOLD) use rayon parallel chunking; each chunk is handled by a SIMD kernel;
// - These are *additional* pub fns; the generic versions (batch_binop, etc.) remain unchanged.
// =========================================================================

/// SIMD lane width.
pub const SIMD_LANES: usize = 4;

// -------------------- f32 / f64 binop (SIMD) --------------------
// f32/f64 use a dedicated float macro: no bitwise/shift ops; Div is native float division.
// i32/i64 binop is generated by impl_simd_int_binop! (defined below) — calls are placed
// alongside the other integer macro invocations.

/// Float SIMD binop kernel generation macro.
/// Accelerates add/sub/mul/div via SIMD; mod and bitwise/shift ops fall back to scalar.
/// f32/f64 do not support bitwise/shift ops — the scalar fn keeps the original value for those.
macro_rules! impl_simd_float_binop {
    ($ty:ty, $vec:ty, $lanes:expr, $scalar_fn:ident) => {
        #[inline]
        fn $scalar_fn(a: $ty, b: $ty, op: BinOp) -> $ty {
            match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Mod => a % b,
                // f32/f64 does not support bitwise/shift ops; keep the original value
                _ => a,
            }
        }

        paste! {
            #[inline]
            fn [<binop_ $ty _kernel>](dst: &mut [$ty], a: &[$ty], b: &[$ty], op: BinOp) {
                let n = dst.len().min(a.len()).min(b.len());
                let blocks = n / $lanes;
                let use_simd = matches!(
                    op,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div
                );
                if use_simd {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        let va = <$vec>::new(a[i..i + $lanes].try_into().unwrap());
                        let vb = <$vec>::new(b[i..i + $lanes].try_into().unwrap());
                        let r = match op {
                            BinOp::Add => va + vb,
                            BinOp::Sub => va - vb,
                            BinOp::Mul => va * vb,
                            BinOp::Div => va / vb,
                            _ => unreachable!(),
                        };
                        dst[i..i + $lanes].copy_from_slice(&r.to_array());
                    }
                } else {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        for j in 0..$lanes {
                            dst[i + j] = $scalar_fn(a[i + j], b[i + j], op);
                        }
                    }
                }
                let tail = blocks * $lanes;
                for i in tail..n {
                    dst[i] = $scalar_fn(a[i], b[i], op);
                }
            }

            pub fn [<batch_binop_ $ty>](dst: &mut [$ty], a: &[$ty], b: &[$ty], op: BinOp) {
                let n = dst.len().min(a.len()).min(b.len());
                if n == 0 { return; }
                if n > PARALLEL_THRESHOLD {
                    let chunk = par_chunk_size(n);
                    dst[..n]
                        .par_chunks_mut(chunk)
                        .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
                        .for_each(|(d, (av, bv))| [<binop_ $ty _kernel>](d, av, bv, op));
                } else {
                    [<binop_ $ty _kernel>](&mut dst[..n], &a[..n], &b[..n], op);
                }
            }
        }
    };
}

impl_simd_float_binop!(f32, f32x4, 4, binop_f32_scalar);
impl_simd_float_binop!(f64, f64x4, 4, binop_f64_scalar);

// -------------------- Compare f32 / f64 (SIMD) --------------------

/// Float SIMD cmp kernel generation macro.
/// wide provides all six comparison methods (cmp_lt/cmp_gt/cmp_eq/cmp_ne/cmp_le/cmp_ge) as
/// inherent methods on f32x4/f64x4. The returned mask is all-1 bits (appears as NaN) for true
/// and 0.0 for false; use to_bits() != 0 to convert to bool.
/// NaN semantics follow IEEE 754: == returns false for NaN, != returns true, ordered comparisons
/// (lt/gt/le/ge) return false when either operand is NaN — consistent with scalar cmp_scalar_t.
macro_rules! impl_simd_float_cmp {
    ($ty:ty, $vec:ty, $lanes:expr) => {
        paste! {
            #[inline]
            fn [<cmp_ $ty _kernel>](dst: &mut [u8], a: &[$ty], b: &[$ty], op: CmpOp) {
                let n = dst.len().min(a.len()).min(b.len());
                let blocks = n / $lanes;
                for blk in 0..blocks {
                    let i = blk * $lanes;
                    let va = <$vec>::new(a[i..i + $lanes].try_into().unwrap());
                    let vb = <$vec>::new(b[i..i + $lanes].try_into().unwrap());
                    // wide float comparison returns a mask: true is all-1 bits (f32 appears as NaN),
                    // false is 0.0. Use to_bits() != 0 to test.
                    let m = match op {
                        CmpOp::Lt => va.simd_lt(vb),
                        CmpOp::Gt => va.simd_gt(vb),
                        CmpOp::Eq => va.simd_eq(vb),
                        CmpOp::Ne => va.simd_ne(vb),
                        CmpOp::Le => va.simd_le(vb),
                        CmpOp::Ge => va.simd_ge(vb),
                    };
                    let arr = m.to_array();
                    for j in 0..$lanes {
                        dst[i + j] = (arr[j].to_bits() != 0) as u8;
                    }
                }
                let tail = blocks * $lanes;
                for i in tail..n {
                    dst[i] = cmp_scalar_t(&a[i], &b[i], op) as u8;
                }
            }

            pub fn [<batch_cmp_ $ty>](dst: &mut [u8], a: &[$ty], b: &[$ty], op: CmpOp) {
                let n = dst.len().min(a.len()).min(b.len());
                if n == 0 { return; }
                if n > PARALLEL_THRESHOLD {
                    let chunk = par_chunk_size(n);
                    dst[..n]
                        .par_chunks_mut(chunk)
                        .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
                        .for_each(|(d, (av, bv))| [<cmp_ $ty _kernel>](d, av, bv, op));
                } else {
                    [<cmp_ $ty _kernel>](&mut dst[..n], &a[..n], &b[..n], op);
                }
            }
        }
    };
}

impl_simd_float_cmp!(f32, f32x4, 4);
impl_simd_float_cmp!(f64, f64x4, 4);

// =========================================================================
// SIMD completion: i8/i16/u8/u16/u32/u64 binop + cmp, plus i32/i64 cmp
// Each type uses the wide crate's native lane count (i8x16=16, i16x8=8, i32x4=4, i64x4=4,
// u8x16=16, u16x8=8, u32x4=4, u64x4=4) to maximize SIMD utilization.
// =========================================================================

/// Generic integer SIMD binop kernel generation macro (including multiplication).
/// Accelerates add/sub/mul/and/or/xor via SIMD; div/mod/shl/shr fall back to scalar.
macro_rules! impl_simd_int_binop {
    ($ty:ty, $vec:ty, $lanes:expr, $scalar_fn:ident) => {
        #[inline]
        fn $scalar_fn(a: $ty, b: $ty, op: BinOp) -> $ty {
            match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                // SIMD batch path: divide-by-zero returns 0 (wrapping semantics); single-node path returns Throw
                BinOp::Div => if b == 0 { 0 } else { a.wrapping_div(b) },
                BinOp::Mod => if b == 0 { 0 } else { a.wrapping_rem(b) },
                BinOp::Band => a & b,
                BinOp::Bor => a | b,
                BinOp::Bxor => a ^ b,
                BinOp::Shl => if (b as i64) < 0 || b as u32 >= <$ty>::BITS { a } else { a.wrapping_shl(b as u32) },
                BinOp::Shr => if (b as i64) < 0 || b as u32 >= <$ty>::BITS { a } else { a.wrapping_shr(b as u32) },
            }
        }

        paste! {
            #[inline]
            fn [<binop_ $ty _kernel>](dst: &mut [$ty], a: &[$ty], b: &[$ty], op: BinOp) {
                let n = dst.len().min(a.len()).min(b.len());
                let blocks = n / $lanes;
                let use_simd = matches!(
                    op,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Band | BinOp::Bor | BinOp::Bxor
                );
                if use_simd {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        let va = <$vec>::new(a[i..i + $lanes].try_into().unwrap());
                        let vb = <$vec>::new(b[i..i + $lanes].try_into().unwrap());
                        let r = match op {
                            BinOp::Add => va + vb,
                            BinOp::Sub => va - vb,
                            BinOp::Mul => va * vb,
                            BinOp::Band => va & vb,
                            BinOp::Bor => va | vb,
                            BinOp::Bxor => va ^ vb,
                            _ => unreachable!(),
                        };
                        dst[i..i + $lanes].copy_from_slice(&r.to_array());
                    }
                } else {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        for j in 0..$lanes {
                            dst[i + j] = $scalar_fn(a[i + j], b[i + j], op);
                        }
                    }
                }
                let tail = blocks * $lanes;
                for i in tail..n {
                    dst[i] = $scalar_fn(a[i], b[i], op);
                }
            }

            pub fn [<batch_binop_ $ty>](dst: &mut [$ty], a: &[$ty], b: &[$ty], op: BinOp) {
                let n = dst.len().min(a.len()).min(b.len());
                if n == 0 { return; }
                if n > PARALLEL_THRESHOLD {
                    let chunk = par_chunk_size(n);
                    dst[..n]
                        .par_chunks_mut(chunk)
                        .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
                        .for_each(|(d, (av, bv))| [<binop_ $ty _kernel>](d, av, bv, op));
                } else {
                    [<binop_ $ty _kernel>](&mut dst[..n], &a[..n], &b[..n], op);
                }
            }
        }
    };
}

/// i8/u8-specific macro: no SIMD multiplication (8-bit multiplication has no hardware support); other operations as above.
macro_rules! impl_simd_int_binop_no_mul {
    ($ty:ty, $vec:ty, $lanes:expr, $scalar_fn:ident) => {
        #[inline]
        fn $scalar_fn(a: $ty, b: $ty, op: BinOp) -> $ty {
            match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                // SIMD batch path: divide-by-zero returns 0 (wrapping semantics); single-node path returns Throw
                BinOp::Div => if b == 0 { 0 } else { a.wrapping_div(b) },
                BinOp::Mod => if b == 0 { 0 } else { a.wrapping_rem(b) },
                BinOp::Band => a & b,
                BinOp::Bor => a | b,
                BinOp::Bxor => a ^ b,
                BinOp::Shl => if (b as i64) < 0 || b as u32 >= <$ty>::BITS { a } else { a.wrapping_shl(b as u32) },
                BinOp::Shr => if (b as i64) < 0 || b as u32 >= <$ty>::BITS { a } else { a.wrapping_shr(b as u32) },
            }
        }

        paste! {
            #[inline]
            fn [<binop_ $ty _kernel>](dst: &mut [$ty], a: &[$ty], b: &[$ty], op: BinOp) {
                let n = dst.len().min(a.len()).min(b.len());
                let blocks = n / $lanes;
                let use_simd = matches!(
                    op,
                    BinOp::Add | BinOp::Sub | BinOp::Band | BinOp::Bor | BinOp::Bxor
                );
                if use_simd {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        let va = <$vec>::new(a[i..i + $lanes].try_into().unwrap());
                        let vb = <$vec>::new(b[i..i + $lanes].try_into().unwrap());
                        let r = match op {
                            BinOp::Add => va + vb,
                            BinOp::Sub => va - vb,
                            BinOp::Band => va & vb,
                            BinOp::Bor => va | vb,
                            BinOp::Bxor => va ^ vb,
                            _ => unreachable!(),
                        };
                        dst[i..i + $lanes].copy_from_slice(&r.to_array());
                    }
                } else {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        for j in 0..$lanes {
                            dst[i + j] = $scalar_fn(a[i + j], b[i + j], op);
                        }
                    }
                }
                let tail = blocks * $lanes;
                for i in tail..n {
                    dst[i] = $scalar_fn(a[i], b[i], op);
                }
            }

            pub fn [<batch_binop_ $ty>](dst: &mut [$ty], a: &[$ty], b: &[$ty], op: BinOp) {
                let n = dst.len().min(a.len()).min(b.len());
                if n == 0 { return; }
                if n > PARALLEL_THRESHOLD {
                    let chunk = par_chunk_size(n);
                    dst[..n]
                        .par_chunks_mut(chunk)
                        .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
                        .for_each(|(d, (av, bv))| [<binop_ $ty _kernel>](d, av, bv, op));
                } else {
                    [<binop_ $ty _kernel>](&mut dst[..n], &a[..n], &b[..n], op);
                }
            }
        }
    };
}

// i8/u8 have no SIMD multiplication (8-bit multiplication has no hardware support); other integer types do
impl_simd_int_binop_no_mul!(i8, i8x16, 16, binop_i8_scalar);
impl_simd_int_binop!(i16, i16x8, 8, binop_i16_scalar);
impl_simd_int_binop_no_mul!(u8, u8x16, 16, binop_u8_scalar);
impl_simd_int_binop!(u16, u16x8, 8, binop_u16_scalar);
impl_simd_int_binop!(u32, u32x4, 4, binop_u32_scalar);
impl_simd_int_binop!(u64, u64x4, 4, binop_u64_scalar);
// i32/i64 binop was previously handwritten; now generated by the same macro (structurally identical).
impl_simd_int_binop!(i32, i32x4, 4, binop_i32_scalar);
impl_simd_int_binop!(i64, i64x4, 4, binop_i64_scalar);

// -------------------- Signed integer SIMD cmp (i8/i16/i32/i64) --------------------
// wide provides CmpEq/CmpLt/CmpGt for signed integers; other combinations: Ne=!Eq, Le=Lt|Eq, Ge=Gt|Eq

/// Signed integer SIMD cmp kernel generation macro.
macro_rules! impl_simd_signed_cmp {
    ($ty:ty, $vec:ty, $lanes:expr) => {
        paste! {
            #[inline]
            fn [<cmp_ $ty _kernel>](dst: &mut [u8], a: &[$ty], b: &[$ty], op: CmpOp) {
                let n = dst.len().min(a.len()).min(b.len());
                let blocks = n / $lanes;
                for blk in 0..blocks {
                    let i = blk * $lanes;
                    let va = <$vec>::new(a[i..i + $lanes].try_into().unwrap());
                    let vb = <$vec>::new(b[i..i + $lanes].try_into().unwrap());
                    // wide signed integer comparison returns a same-type mask (all 1/0); convert to bool
                    let arr = match op {
                        CmpOp::Eq => va.simd_eq(vb).to_array(),
                        CmpOp::Ne => {
                            let m = va.simd_eq(vb);
                            (!m).to_array()
                        }
                        CmpOp::Lt => va.simd_lt(vb).to_array(),
                        CmpOp::Gt => va.simd_gt(vb).to_array(),
                        CmpOp::Le => {
                            let lt = va.simd_lt(vb);
                            let eq = va.simd_eq(vb);
                            (lt | eq).to_array()
                        }
                        CmpOp::Ge => {
                            let gt = va.simd_gt(vb);
                            let eq = va.simd_eq(vb);
                            (gt | eq).to_array()
                        }
                    };
                    for j in 0..$lanes {
                        dst[i + j] = (arr[j] != 0) as u8;
                    }
                }
                let tail = blocks * $lanes;
                for i in tail..n {
                    dst[i] = cmp_scalar_t(&a[i], &b[i], op) as u8;
                }
            }

            pub fn [<batch_cmp_ $ty>](dst: &mut [u8], a: &[$ty], b: &[$ty], op: CmpOp) {
                let n = dst.len().min(a.len()).min(b.len());
                if n == 0 { return; }
                if n > PARALLEL_THRESHOLD {
                    let chunk = par_chunk_size(n);
                    dst[..n]
                        .par_chunks_mut(chunk)
                        .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
                        .for_each(|(d, (av, bv))| [<cmp_ $ty _kernel>](d, av, bv, op));
                } else {
                    [<cmp_ $ty _kernel>](&mut dst[..n], &a[..n], &b[..n], op);
                }
            }
        }
    };
}

impl_simd_signed_cmp!(i8, i8x16, 16);
impl_simd_signed_cmp!(i16, i16x8, 8);
impl_simd_signed_cmp!(i32, i32x4, 4);
impl_simd_signed_cmp!(i64, i64x4, 4);

// -------------------- Unsigned integer SIMD cmp (u8/u16/u32/u64) --------------------
// wide provides only CmpEq for unsigned integers; other comparisons fall back to scalar (no SIMD unsigned comparison instructions)

/// Unsigned integer SIMD cmp kernel generation macro: only Eq/Ne use SIMD; others use scalar.
macro_rules! impl_simd_unsigned_cmp {
    ($ty:ty, $vec:ty, $lanes:expr) => {
        paste! {
            #[inline]
            fn [<cmp_ $ty _kernel>](dst: &mut [u8], a: &[$ty], b: &[$ty], op: CmpOp) {
                let n = dst.len().min(a.len()).min(b.len());
                let blocks = n / $lanes;
                let use_simd = matches!(op, CmpOp::Eq | CmpOp::Ne);
                if use_simd {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        let va = <$vec>::new(a[i..i + $lanes].try_into().unwrap());
                        let vb = <$vec>::new(b[i..i + $lanes].try_into().unwrap());
                        let arr = match op {
                            CmpOp::Eq => va.simd_eq(vb).to_array(),
                            CmpOp::Ne => {
                                let m = va.simd_eq(vb);
                                (!m).to_array()
                            }
                            _ => unreachable!(),
                        };
                        for j in 0..$lanes {
                            dst[i + j] = (arr[j] != 0) as u8;
                        }
                    }
                } else {
                    for blk in 0..blocks {
                        let i = blk * $lanes;
                        for j in 0..$lanes {
                            dst[i + j] = cmp_scalar_t(&a[i + j], &b[i + j], op) as u8;
                        }
                    }
                }
                let tail = blocks * $lanes;
                for i in tail..n {
                    dst[i] = cmp_scalar_t(&a[i], &b[i], op) as u8;
                }
            }

            pub fn [<batch_cmp_ $ty>](dst: &mut [u8], a: &[$ty], b: &[$ty], op: CmpOp) {
                let n = dst.len().min(a.len()).min(b.len());
                if n == 0 { return; }
                if n > PARALLEL_THRESHOLD {
                    let chunk = par_chunk_size(n);
                    dst[..n]
                        .par_chunks_mut(chunk)
                        .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
                        .for_each(|(d, (av, bv))| [<cmp_ $ty _kernel>](d, av, bv, op));
                } else {
                    [<cmp_ $ty _kernel>](&mut dst[..n], &a[..n], &b[..n], op);
                }
            }
        }
    };
}

impl_simd_unsigned_cmp!(u8, u8x16, 16);
impl_simd_unsigned_cmp!(u16, u16x8, 8);
impl_simd_unsigned_cmp!(u32, u32x4, 4);
impl_simd_unsigned_cmp!(u64, u64x4, 4);

// =========================================================================
// Part 15: Pure arithmetic core — no Frame dependency; shared by runtime compute_fn and compile-time ConstFold
// =========================================================================
//
// Generates pure arithmetic functions for all integer/float types. Semantics strictly match the compute_fn macro in Engine.rs:
//   - Integer add/sub/mul/neg: wrapping semantics (consistent with Bug #22; Bug #75 unified to wrapping, no debug/release branch)
//   - Integer div/mod: divide-by-zero panics (no silent fallback)
//   - Integer shl/shr: shift amount is i32 (matching Engine.rs reading as_i32), cast to u32 then wrapping
//   - Float div: native division (divide-by-zero yields inf/nan)
// runtime compute_fn calls these pure functions (reuse); compile-time ConstFold also calls the same arithmetic (decoupled from Frame).

/// Generates a full set of pure arithmetic functions for the specified integer type (add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot).
/// The shl/shr shift amount parameter is i32 (matching Engine.rs compute_shl_*/compute_shr_* reading as_i32).
macro_rules! impl_arith_int {
    ($ty:ident, $rust:ty) => {
        paste! {
            #[inline] pub fn [<arith_add_$ty>](a: $rust, b: $rust) -> $rust { a.wrapping_add(b) }
            #[inline] pub fn [<arith_sub_$ty>](a: $rust, b: $rust) -> $rust { a.wrapping_sub(b) }
            #[inline] pub fn [<arith_mul_$ty>](a: $rust, b: $rust) -> $rust { a.wrapping_mul(b) }
            /// Divide-by-zero returns None; the caller converts it to a Throw error value for propagation (consistent with compute_str_concat).
            #[inline] pub fn [<arith_div_$ty>](a: $rust, b: $rust) -> Option<$rust> { if b == 0 { None } else { Some(a.wrapping_div(b)) } }
            #[inline] pub fn [<arith_mod_$ty>](a: $rust, b: $rust) -> Option<$rust> { if b == 0 { None } else { Some(a.wrapping_rem(b)) } }
            #[inline] pub fn [<arith_bitand_$ty>](a: $rust, b: $rust) -> $rust { a & b }
            #[inline] pub fn [<arith_bitor_$ty>](a: $rust, b: $rust) -> $rust { a | b }
            #[inline] pub fn [<arith_bitxor_$ty>](a: $rust, b: $rust) -> $rust { a ^ b }
            /// Shift out of bounds (negative or >= type bit width) returns None; the caller converts it to a Throw error value.
            #[inline] pub fn [<arith_shl_$ty>](a: $rust, shift: i32) -> Option<$rust> {
                if shift < 0 || shift as u32 >= <$rust>::BITS { None } else { Some(a.wrapping_shl(shift as u32)) }
            }
            #[inline] pub fn [<arith_shr_$ty>](a: $rust, shift: i32) -> Option<$rust> {
                if shift < 0 || shift as u32 >= <$rust>::BITS { None } else { Some(a.wrapping_shr(shift as u32)) }
            }
            #[inline] pub fn [<arith_neg_$ty>](a: $rust) -> $rust { a.wrapping_neg() }
            #[inline] pub fn [<arith_bitnot_$ty>](a: $rust) -> $rust { !a }
        }
    };
}

/// Generates a full set of pure arithmetic functions for the specified float type (add/sub/mul/div/mod/neg).
macro_rules! impl_arith_float {
    ($ty:ident, $rust:ty) => {
        paste! {
            #[inline] pub fn [<arith_add_$ty>](a: $rust, b: $rust) -> $rust { a + b }
            #[inline] pub fn [<arith_sub_$ty>](a: $rust, b: $rust) -> $rust { a - b }
            #[inline] pub fn [<arith_mul_$ty>](a: $rust, b: $rust) -> $rust { a * b }
            #[inline] pub fn [<arith_div_$ty>](a: $rust, b: $rust) -> $rust { a / b }
            #[inline] pub fn [<arith_mod_$ty>](a: $rust, b: $rust) -> $rust { a % b }
            #[inline] pub fn [<arith_neg_$ty>](a: $rust) -> $rust { -a }
        }
    };
}

// Integer type expansion (12 types × 12 operations)
impl_arith_int!(i8,    i8);
impl_arith_int!(i16,   i16);
impl_arith_int!(i32,   i32);
impl_arith_int!(i64,   i64);
impl_arith_int!(i128,  i128);
impl_arith_int!(u8,    u8);
impl_arith_int!(u16,   u16);
impl_arith_int!(u32,   u32);
impl_arith_int!(u64,   u64);
impl_arith_int!(u128,  u128);
impl_arith_int!(isize, isize);
impl_arith_int!(usize, usize);

// Float type expansion (4 types × 6 operations)
impl_arith_float!(f16, F16);
impl_arith_float!(f32, f32);
impl_arith_float!(f64, f64);
impl_arith_float!(f128, F128);

// =========================================================================
// Boolean pure arithmetic — semantically consistent with Engine.rs compute_and_bool/or/not
// =========================================================================

#[inline] pub fn arith_and_bool(a: bool, b: bool) -> bool { a && b }
#[inline] pub fn arith_or_bool(a: bool, b: bool) -> bool { a || b }
#[inline] pub fn arith_not_bool(a: bool) -> bool { !a }