// =========================================================================
// Ops — Num/BitOps trait + cast + batch/SIMD + allocator + pure arithmetic core
// =========================================================================

use std::hash::Hash;

use rayon::prelude::*;
use pastey::paste;
use wide::{f32x4, f64x4, i8x16, i16x8, i32x4, i64x4, u8x16, u16x8, u32x4, u64x4, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe};

pub use crate::types::ValueTag;

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
                fn shl(self, amount: u32) -> Self { self.wrapping_shl(amount) }
                fn shr(self, amount: u32) -> Self { self.wrapping_shr(amount) }
            }
        )*
    };
}

impl_bitops!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

// =========================================================================
// Part 12: cast.rs (cast_value)
// =========================================================================

pub fn cast_value(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag) -> Vec<u8> {
    let dst_width = dst_tag.byte_width();
    let mut result = vec![0u8; dst_width];

    if src_tag == dst_tag {
        let copy_len = src_bytes.len().min(dst_width);
        result[..copy_len].copy_from_slice(&src_bytes[..copy_len]);
        return result;
    }

    match (src_tag, dst_tag) {
        (ValueTag::Bool, _) => {
            let b = read_bool(src_bytes);
            cast_from_bool(b, dst_tag, &mut result);
        }
        (ValueTag::Char, _) => {
            let cp = read_u32_le(src_bytes);
            cast_from_u32(cp, dst_tag, &mut result);
        }
        (_, ValueTag::Bool) => {
            let b = cast_to_bool(src_tag, src_bytes);
            write_bool(b, &mut result);
        }
        (_, ValueTag::Char) => {
            let cp = cast_to_u32(src_tag, src_bytes);
            write_u32_le(cp, &mut result);
        }
        (s, d) if s.is_int() && d.is_int() => {
            cast_int_to_int(src_tag, src_bytes, dst_tag, &mut result);
        }
        (s, d) if s.is_int() && d.is_float() => {
            cast_int_to_float(src_tag, src_bytes, dst_tag, &mut result);
        }
        (s, d) if s.is_float() && d.is_int() => {
            cast_float_to_int(src_tag, src_bytes, dst_tag, &mut result);
        }
        (s, d) if s.is_float() && d.is_float() => {
            cast_float_to_float(src_tag, src_bytes, dst_tag, &mut result);
        }
        _ => {}
    }

    result
}

// ---- cast internal helpers ----

fn read_bool(bytes: &[u8]) -> bool {
    bytes.first().copied().unwrap_or(0) != 0
}

fn read_i16_le(bytes: &[u8]) -> i16 {
    let mut buf = [0u8; 2];
    let len = bytes.len().min(2);
    buf[..len].copy_from_slice(&bytes[..len]);
    i16::from_le_bytes(buf)
}

fn read_u16_le(bytes: &[u8]) -> u16 {
    let mut buf = [0u8; 2];
    let len = bytes.len().min(2);
    buf[..len].copy_from_slice(&bytes[..len]);
    u16::from_le_bytes(buf)
}

fn read_i32_le(bytes: &[u8]) -> i32 {
    let mut buf = [0u8; 4];
    let len = bytes.len().min(4);
    buf[..len].copy_from_slice(&bytes[..len]);
    i32::from_le_bytes(buf)
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let len = bytes.len().min(4);
    buf[..len].copy_from_slice(&bytes[..len]);
    u32::from_le_bytes(buf)
}

fn read_i64_le(bytes: &[u8]) -> i64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    i64::from_le_bytes(buf)
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(buf)
}

fn read_i128_le(bytes: &[u8]) -> i128 {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    i128::from_le_bytes(buf)
}

fn read_u128_le(bytes: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    u128::from_le_bytes(buf)
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    let mut buf = [0u8; 4];
    let len = bytes.len().min(4);
    buf[..len].copy_from_slice(&bytes[..len]);
    f32::from_le_bytes(buf)
}

fn read_f64_le(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    f64::from_le_bytes(buf)
}

fn read_f16(bits: &[u8]) -> F16 {
    let mut buf = [0u8; 2];
    let len = bits.len().min(2);
    buf[..len].copy_from_slice(&bits[..len]);
    F16(u16::from_le_bytes(buf))
}

fn read_f128(bytes: &[u8]) -> F128 {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    F128(buf)
}

fn read_int_as_i128(tag: ValueTag, bytes: &[u8]) -> i128 {
    read_int_as!(tag, bytes, i128)
}

fn read_int_as_u128(tag: ValueTag, bytes: &[u8]) -> u128 {
    read_int_as!(tag, bytes, u128)
}

fn write_bool(b: bool, dst: &mut [u8]) {
    if dst.is_empty() { return; }
    dst[0] = if b { 1 } else { 0 };
}

fn write_u8(v: u8, dst: &mut [u8]) {
    if !dst.is_empty() { dst[0] = v; }
}

fn write_i8(v: i8, dst: &mut [u8]) {
    write_u8(v as u8, dst);
}

fn write_u16_le(v: u16, dst: &mut [u8]) {
    let bytes = v.to_le_bytes();
    let len = dst.len().min(2);
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_i16_le(v: i16, dst: &mut [u8]) {
    write_u16_le(v as u16, dst);
}

fn write_u32_le(v: u32, dst: &mut [u8]) {
    let bytes = v.to_le_bytes();
    let len = dst.len().min(4);
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_i32_le(v: i32, dst: &mut [u8]) {
    write_u32_le(v as u32, dst);
}

fn write_u64_le(v: u64, dst: &mut [u8]) {
    let bytes = v.to_le_bytes();
    let len = dst.len().min(8);
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_i64_le(v: i64, dst: &mut [u8]) {
    write_u64_le(v as u64, dst);
}

fn write_u128_le(v: u128, dst: &mut [u8]) {
    let bytes = v.to_le_bytes();
    let len = dst.len().min(16);
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_i128_le(v: i128, dst: &mut [u8]) {
    write_u128_le(v as u128, dst);
}

fn write_f32_le(v: f32, dst: &mut [u8]) {
    let bytes = v.to_le_bytes();
    let len = dst.len().min(4);
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_f64_le(v: f64, dst: &mut [u8]) {
    let bytes = v.to_le_bytes();
    let len = dst.len().min(8);
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_f16(f: F16, dst: &mut [u8]) {
    write_u16_le(f.0, dst);
}

fn write_f128(f: F128, dst: &mut [u8]) {
    let len = dst.len().min(16);
    dst[..len].copy_from_slice(&f.0[..len]);
}

fn cast_from_bool(b: bool, dst_tag: ValueTag, dst: &mut [u8]) {
    let val: i128 = if b { 1 } else { 0 };
    if dst_tag.is_int() {
        cast_from_i128(val, dst_tag, dst);
    } else if dst_tag.is_float() {
        let f = if b { 1.0 } else { 0.0 };
        cast_from_f64(f, dst_tag, dst);
    } else if dst_tag == ValueTag::Char {
        write_u32_le(val as u32, dst);
    }
}

fn cast_from_u32(cp: u32, dst_tag: ValueTag, dst: &mut [u8]) {
    if dst_tag.is_int() {
        cast_from_u128(cp as u128, dst_tag, dst);
    } else if dst_tag.is_float() {
        cast_from_f64(cp as f64, dst_tag, dst);
    } else if dst_tag == ValueTag::Bool {
        write_bool(cp != 0, dst);
    }
}

fn cast_from_i128(val: i128, dst_tag: ValueTag, dst: &mut [u8]) {
    write_int_bytes!(val, dst_tag, dst)
}

fn cast_from_u128(val: u128, dst_tag: ValueTag, dst: &mut [u8]) {
    write_int_bytes!(val, dst_tag, dst)
}

fn cast_from_f64(val: f64, dst_tag: ValueTag, dst: &mut [u8]) {
    match dst_tag {
        ValueTag::F16 => write_f16(F16::from_f32(val as f32), dst),
        ValueTag::F32 => write_f32_le(val as f32, dst),
        ValueTag::F64 => write_f64_le(val, dst),
        ValueTag::F128 => write_f128(F128::from_f64(val), dst),
        _ => {}
    }
}

fn cast_to_bool(src_tag: ValueTag, src_bytes: &[u8]) -> bool {
    match src_tag {
        ValueTag::Bool => read_bool(src_bytes),
        ValueTag::Char => read_u32_le(src_bytes) != 0,
        ValueTag::I8 => src_bytes.first().copied().unwrap_or(0) as i8 != 0,
        ValueTag::U8 => src_bytes.first().copied().unwrap_or(0) != 0,
        ValueTag::I16 | ValueTag::U16 => {
            let mut b = [0u8; 2];
            let l = src_bytes.len().min(2);
            b[..l].copy_from_slice(&src_bytes[..l]);
            u16::from_le_bytes(b) != 0
        }
        ValueTag::I32 | ValueTag::U32 => {
            let v = read_u32_le(src_bytes);
            v != 0
        }
        ValueTag::I64 | ValueTag::U64 | ValueTag::Isize | ValueTag::Usize => {
            read_u64_le(src_bytes) != 0
        }
        ValueTag::I128 | ValueTag::U128 => {
            read_u128_le(src_bytes) != 0
        }
        ValueTag::F32 => read_f32_le(src_bytes) != 0.0,
        ValueTag::F64 => read_f64_le(src_bytes) != 0.0,
        ValueTag::F16 => read_f16(src_bytes).to_f32() != 0.0,
        ValueTag::F128 => read_f128(src_bytes).to_f64() != 0.0,
        _ => false,
    }
}

fn cast_to_u32(src_tag: ValueTag, src_bytes: &[u8]) -> u32 {
    if src_tag.is_int() {
        read_int_as_u128(src_tag, src_bytes) as u32
    } else if src_tag.is_float() {
        let f = read_float_as_f64(src_tag, src_bytes);
        if f.is_nan() {
            0
        } else if f >= u32::MAX as f64 {
            u32::MAX
        } else if f <= 0.0 {
            0
        } else {
            f as u32
        }
    } else if src_tag == ValueTag::Bool {
        if read_bool(src_bytes) { 1 } else { 0 }
    } else {
        0
    }
}

fn read_float_as_f64(tag: ValueTag, bytes: &[u8]) -> f64 {
    match tag {
        ValueTag::F16 => read_f16(bytes).to_f64(),
        ValueTag::F32 => read_f32_le(bytes) as f64,
        ValueTag::F64 => read_f64_le(bytes),
        ValueTag::F128 => read_f128(bytes).to_f64(),
        _ => 0.0,
    }
}

fn cast_int_to_int(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag, dst: &mut [u8]) {
    if dst_tag.is_signed() {
        let val = read_int_as_i128(src_tag, src_bytes);
        cast_from_i128(val, dst_tag, dst);
    } else {
        let val = read_int_as_u128(src_tag, src_bytes);
        cast_from_u128(val, dst_tag, dst);
    }
}

fn cast_int_to_float(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag, dst: &mut [u8]) {
    // F128 target: integers via `as f64` lose precision for values >2^53, so use from_i128/from_u128 for exact construction
    if dst_tag == ValueTag::F128 {
        let f = if src_tag.is_signed() {
            F128::from_i128(read_int_as_i128(src_tag, src_bytes))
        } else {
            F128::from_u128(read_int_as_u128(src_tag, src_bytes))
        };
        write_f128(f, dst);
        return;
    }
    let val = if src_tag.is_signed() {
        read_int_as_i128(src_tag, src_bytes) as f64
    } else {
        read_int_as_u128(src_tag, src_bytes) as f64
    };
    cast_from_f64(val, dst_tag, dst);
}

fn cast_float_to_int(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag, dst: &mut [u8]) {
    let f = read_float_as_f64(src_tag, src_bytes);
    match dst_tag {
        ValueTag::I8 => write_i8(f as i8, dst),
        ValueTag::I16 => write_i16_le(f as i16, dst),
        ValueTag::I32 => write_i32_le(f as i32, dst),
        ValueTag::I64 => write_i64_le(f as i64, dst),
        ValueTag::I128 => write_i128_le(f as i128, dst),
        ValueTag::Isize => write_i64_le(f as isize as i64, dst),
        ValueTag::U8 => write_u8(f as u8, dst),
        ValueTag::U16 => write_u16_le(f as u16, dst),
        ValueTag::U32 => write_u32_le(f as u32, dst),
        ValueTag::U64 => write_u64_le(f as u64, dst),
        ValueTag::U128 => write_u128_le(f as u128, dst),
        ValueTag::Usize => write_u64_le(f as usize as u64, dst),
        _ => {}
    }
}

fn cast_float_to_float(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag, dst: &mut [u8]) {
    let f = read_float_as_f64(src_tag, src_bytes);
    cast_from_f64(f, dst_tag, dst);
}

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
        // Bug #75: debug mode panics on integer overflow; release mode wraps.
        // Float checked_* always returns Some, so expect never panics for floats.
        BinOp::Add => {
            if cfg!(debug_assertions) {
                a.checked_add(b).expect("integer overflow in addition")
            } else {
                a.wrapping_add(b)
            }
        }
        BinOp::Sub => {
            if cfg!(debug_assertions) {
                a.checked_sub(b).expect("integer overflow in subtraction")
            } else {
                a.wrapping_sub(b)
            }
        }
        BinOp::Mul => {
            if cfg!(debug_assertions) {
                a.checked_mul(b).expect("integer overflow in multiplication")
            } else {
                a.wrapping_mul(b)
            }
        }
        // Division-by-zero semantics match scalar arith_div/arith_mod:
        //   - Integer checked_div/checked_rem return None on divide-by-zero → expect panics
        //   - Float checked_div/checked_rem always return Some (Kuzo Num impl delegates to native /, yielding inf/nan) → expect never panics
        BinOp::Div => a.checked_div(b).expect("integer divide by zero"),
        BinOp::Mod => a.checked_rem(b).expect("integer modulo by zero"),
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
            UnaryOp::Neg => {
                if cfg!(debug_assertions) {
                    a[i].neg().expect("integer overflow in negation")
                } else {
                    a[i].wrapping_neg()
                }
            }
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

// -------------------- f32 --------------------

#[inline]
fn binop_f32_scalar(a: f32, b: f32, op: BinOp) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => a % b,
        // f32 does not support bitwise/shift ops; keep the original value
        _ => a,
    }
}

fn binop_f32_kernel(dst: &mut [f32], a: &[f32], b: &[f32], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    let use_simd = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div);
    if use_simd {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            let va = f32x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
            let vb = f32x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
            let r = match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Div => va / vb,
                _ => unreachable!(),
            };
            dst[i..i + SIMD_LANES].copy_from_slice(&r.to_array());
        }
    } else {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            for j in 0..SIMD_LANES {
                dst[i + j] = binop_f32_scalar(a[i + j], b[i + j], op);
            }
        }
    }
    let tail = blocks * SIMD_LANES;
    for i in tail..n {
        dst[i] = binop_f32_scalar(a[i], b[i], op);
    }
}

/// f32 SIMD + rayon parallel binary operation.
pub fn batch_binop_f32(dst: &mut [f32], a: &[f32], b: &[f32], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| binop_f32_kernel(d, av, bv, op));
    } else {
        binop_f32_kernel(&mut dst[..n], &a[..n], &b[..n], op);
    }
}

// -------------------- f64 --------------------

#[inline]
fn binop_f64_scalar(a: f64, b: f64, op: BinOp) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => a % b,
        _ => a,
    }
}

fn binop_f64_kernel(dst: &mut [f64], a: &[f64], b: &[f64], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    let use_simd = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div);
    if use_simd {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            let va = f64x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
            let vb = f64x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
            let r = match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Div => va / vb,
                _ => unreachable!(),
            };
            dst[i..i + SIMD_LANES].copy_from_slice(&r.to_array());
        }
    } else {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            for j in 0..SIMD_LANES {
                dst[i + j] = binop_f64_scalar(a[i + j], b[i + j], op);
            }
        }
    }
    let tail = blocks * SIMD_LANES;
    for i in tail..n {
        dst[i] = binop_f64_scalar(a[i], b[i], op);
    }
}

/// f64 SIMD + rayon parallel binary operation.
pub fn batch_binop_f64(dst: &mut [f64], a: &[f64], b: &[f64], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| binop_f64_kernel(d, av, bv, op));
    } else {
        binop_f64_kernel(&mut dst[..n], &a[..n], &b[..n], op);
    }
}

// -------------------- i32 --------------------

#[inline]
fn binop_i32_scalar(a: i32, b: i32, op: BinOp) -> i32 {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        // Integer divide-by-zero panics directly (no fallback)
        BinOp::Div => a / b,
        BinOp::Mod => a % b,
        BinOp::Band => a & b,
        BinOp::Bor => a | b,
        BinOp::Bxor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
    }
}

fn binop_i32_kernel(dst: &mut [i32], a: &[i32], b: &[i32], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    // i32x4 supports arithmetic + bitwise ops (all wrapping, consistent with the generic semantics);
    // Div/Mod (no SIMD integer division, and divide-by-zero protection needed) and Shl/Shr
    // (per-lane variable-length shifts are unsupported) fall back to scalar.
    let use_simd = matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Band | BinOp::Bor | BinOp::Bxor
    );
    if use_simd {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            let va = i32x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
            let vb = i32x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
            let r = match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Band => va & vb,
                BinOp::Bor => va | vb,
                BinOp::Bxor => va ^ vb,
                _ => unreachable!(),
            };
            dst[i..i + SIMD_LANES].copy_from_slice(&r.to_array());
        }
    } else {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            for j in 0..SIMD_LANES {
                dst[i + j] = binop_i32_scalar(a[i + j], b[i + j], op);
            }
        }
    }
    let tail = blocks * SIMD_LANES;
    for i in tail..n {
        dst[i] = binop_i32_scalar(a[i], b[i], op);
    }
}

/// i32 SIMD + rayon parallel binary operation.
pub fn batch_binop_i32(dst: &mut [i32], a: &[i32], b: &[i32], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| binop_i32_kernel(d, av, bv, op));
    } else {
        binop_i32_kernel(&mut dst[..n], &a[..n], &b[..n], op);
    }
}

// -------------------- i64 --------------------

#[inline]
fn binop_i64_scalar(a: i64, b: i64, op: BinOp) -> i64 {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => a / b,
        BinOp::Mod => a % b,
        BinOp::Band => a & b,
        BinOp::Bor => a | b,
        BinOp::Bxor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
    }
}

fn binop_i64_kernel(dst: &mut [i64], a: &[i64], b: &[i64], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    let use_simd = matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Band | BinOp::Bor | BinOp::Bxor
    );
    if use_simd {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            let va = i64x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
            let vb = i64x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
            let r = match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Band => va & vb,
                BinOp::Bor => va | vb,
                BinOp::Bxor => va ^ vb,
                _ => unreachable!(),
            };
            dst[i..i + SIMD_LANES].copy_from_slice(&r.to_array());
        }
    } else {
        for blk in 0..blocks {
            let i = blk * SIMD_LANES;
            for j in 0..SIMD_LANES {
                dst[i + j] = binop_i64_scalar(a[i + j], b[i + j], op);
            }
        }
    }
    let tail = blocks * SIMD_LANES;
    for i in tail..n {
        dst[i] = binop_i64_scalar(a[i], b[i], op);
    }
}

/// i64 SIMD + rayon parallel binary operation.
pub fn batch_binop_i64(dst: &mut [i64], a: &[i64], b: &[i64], op: BinOp) {
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| binop_i64_kernel(d, av, bv, op));
    } else {
        binop_i64_kernel(&mut dst[..n], &a[..n], &b[..n], op);
    }
}

// -------------------- Compare f32 / f64 --------------------

fn cmp_f32_kernel(dst: &mut [u8], a: &[f32], b: &[f32], op: CmpOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    for blk in 0..blocks {
        let i = blk * SIMD_LANES;
        let va = f32x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
        let vb = f32x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
        // wide float comparison returns a mask: true is all-1 bits (f32 appears as NaN),
        // false is 0.0. Use to_bits() != 0 to test.
        let m = match op {
            CmpOp::Lt => va.cmp_lt(vb),
            CmpOp::Gt => va.cmp_gt(vb),
            CmpOp::Eq => va.cmp_eq(vb),
            CmpOp::Ne => va.cmp_ne(vb),
            CmpOp::Le => va.cmp_le(vb),
            CmpOp::Ge => va.cmp_ge(vb),
        };
        let arr = m.to_array();
        for j in 0..SIMD_LANES {
            dst[i + j] = (arr[j].to_bits() != 0) as u8;
        }
    }
    let tail = blocks * SIMD_LANES;
    for i in tail..n {
        dst[i] = cmp_scalar_t(&a[i], &b[i], op) as u8;
    }
}

/// f32 SIMD + rayon parallel comparison (outputs a u8 mask).
pub fn batch_cmp_f32(dst: &mut [u8], a: &[f32], b: &[f32], op: CmpOp) {
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| cmp_f32_kernel(d, av, bv, op));
    } else {
        cmp_f32_kernel(&mut dst[..n], &a[..n], &b[..n], op);
    }
}

fn cmp_f64_kernel(dst: &mut [u8], a: &[f64], b: &[f64], op: CmpOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    for blk in 0..blocks {
        let i = blk * SIMD_LANES;
        let va = f64x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
        let vb = f64x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
        let m = match op {
            CmpOp::Lt => va.cmp_lt(vb),
            CmpOp::Gt => va.cmp_gt(vb),
            CmpOp::Eq => va.cmp_eq(vb),
            CmpOp::Ne => va.cmp_ne(vb),
            CmpOp::Le => va.cmp_le(vb),
            CmpOp::Ge => va.cmp_ge(vb),
        };
        let arr = m.to_array();
        for j in 0..SIMD_LANES {
            dst[i + j] = (arr[j].to_bits() != 0) as u8;
        }
    }
    let tail = blocks * SIMD_LANES;
    for i in tail..n {
        dst[i] = cmp_scalar_t(&a[i], &b[i], op) as u8;
    }
}

/// f64 SIMD + rayon parallel comparison (outputs a u8 mask).
pub fn batch_cmp_f64(dst: &mut [u8], a: &[f64], b: &[f64], op: CmpOp) {
    let n = dst.len().min(a.len()).min(b.len());
    if n == 0 {
        return;
    }
    if n > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(n);
        dst[..n]
            .par_chunks_mut(chunk)
            .zip(a[..n].par_chunks(chunk).zip(b[..n].par_chunks(chunk)))
            .for_each(|(d, (av, bv))| cmp_f64_kernel(d, av, bv, op));
    } else {
        cmp_f64_kernel(&mut dst[..n], &a[..n], &b[..n], op);
    }
}

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
                BinOp::Div => a / b,
                BinOp::Mod => a % b,
                BinOp::Band => a & b,
                BinOp::Bor => a | b,
                BinOp::Bxor => a ^ b,
                BinOp::Shl => a.wrapping_shl(b as u32),
                BinOp::Shr => a.wrapping_shr(b as u32),
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
                BinOp::Div => a / b,
                BinOp::Mod => a % b,
                BinOp::Band => a & b,
                BinOp::Bor => a | b,
                BinOp::Bxor => a ^ b,
                BinOp::Shl => a.wrapping_shl(b as u32),
                BinOp::Shr => a.wrapping_shr(b as u32),
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
                        CmpOp::Eq => CmpEq::cmp_eq(va, vb).to_array(),
                        CmpOp::Ne => {
                            let m = CmpEq::cmp_eq(va, vb);
                            (!m).to_array()
                        }
                        CmpOp::Lt => CmpLt::cmp_lt(va, vb).to_array(),
                        CmpOp::Gt => CmpGt::cmp_gt(va, vb).to_array(),
                        CmpOp::Le => {
                            let lt = CmpLt::cmp_lt(va, vb);
                            let eq = CmpEq::cmp_eq(va, vb);
                            (lt | eq).to_array()
                        }
                        CmpOp::Ge => {
                            let gt = CmpGt::cmp_gt(va, vb);
                            let eq = CmpEq::cmp_eq(va, vb);
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
                            CmpOp::Eq => CmpEq::cmp_eq(va, vb).to_array(),
                            CmpOp::Ne => {
                                let m = CmpEq::cmp_eq(va, vb);
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
//   - Integer add/sub/mul/neg: debug mode panics on overflow; release mode wraps (Bug #75)
//   - Integer div/mod: divide-by-zero panics (no silent fallback)
//   - Integer shl/shr: shift amount is i32 (matching Engine.rs reading as_i32), cast to u32 then wrapping
//   - Float div: native division (divide-by-zero yields inf/nan)
// runtime compute_fn calls these pure functions (reuse); compile-time ConstFold also calls the same arithmetic (decoupled from Frame).

/// Generates a full set of pure arithmetic functions for the specified integer type (add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot).
/// The shl/shr shift amount parameter is i32 (matching Engine.rs compute_shl_*/compute_shr_* reading as_i32).
macro_rules! impl_arith_int {
    ($ty:ident, $rust:ty) => {
        paste! {
            #[inline] pub fn [<arith_add_$ty>](a: $rust, b: $rust) -> $rust {
                if cfg!(debug_assertions) { a.checked_add(b).expect("integer overflow in addition") } else { a.wrapping_add(b) }
            }
            #[inline] pub fn [<arith_sub_$ty>](a: $rust, b: $rust) -> $rust {
                if cfg!(debug_assertions) { a.checked_sub(b).expect("integer overflow in subtraction") } else { a.wrapping_sub(b) }
            }
            #[inline] pub fn [<arith_mul_$ty>](a: $rust, b: $rust) -> $rust {
                if cfg!(debug_assertions) { a.checked_mul(b).expect("integer overflow in multiplication") } else { a.wrapping_mul(b) }
            }
            #[inline] pub fn [<arith_div_$ty>](a: $rust, b: $rust) -> $rust { if b == 0 { panic!("integer divide by zero") } else { a.wrapping_div(b) } }
            #[inline] pub fn [<arith_mod_$ty>](a: $rust, b: $rust) -> $rust { if b == 0 { panic!("integer modulo by zero") } else { a.wrapping_rem(b) } }
            #[inline] pub fn [<arith_bitand_$ty>](a: $rust, b: $rust) -> $rust { a & b }
            #[inline] pub fn [<arith_bitor_$ty>](a: $rust, b: $rust) -> $rust { a | b }
            #[inline] pub fn [<arith_bitxor_$ty>](a: $rust, b: $rust) -> $rust { a ^ b }
            #[inline] pub fn [<arith_shl_$ty>](a: $rust, shift: i32) -> $rust { a.wrapping_shl(shift as u32) }
            #[inline] pub fn [<arith_shr_$ty>](a: $rust, shift: i32) -> $rust { a.wrapping_shr(shift as u32) }
            #[inline] pub fn [<arith_neg_$ty>](a: $rust) -> $rust {
                if cfg!(debug_assertions) { a.checked_neg().expect("integer overflow in negation") } else { a.wrapping_neg() }
            }
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