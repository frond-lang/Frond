// =========================================================================
// Ops — Num/BitOps trait + cast + batch/SIMD + allocator + 纯算术核心
// =========================================================================

use std::hash::Hash;
use std::sync::Arc;

use rayon::prelude::*;
use pastey::paste;
use wide::{f32x4, f64x4, i8x16, i16x8, i32x4, i64x4, u8x16, u16x8, u32x4, u64x4, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe};

pub use crate::types::ValueTag;

use super::value::*;

// =========================================================================
// 第十一部分：ops.rs（Num trait + BitOps trait + impl）
// =========================================================================

/// 数值运算 trait：支持溢出检测的算术运算
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

// F16 实现 Num：委托到精确 IEEE 754 binary16 运算（不经 f64 中转）
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
        // 清除符号位
        F16(self.0 & 0x7FFF)
    }
    fn to_u32(self) -> u32 { self.to_f32() as u32 }
}

// F128 实现 Num：委托到精确 IEEE 754 binary128 运算（不经 f64 中转）
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
        // 清除符号位（bit 127）
        let bits = u128::from_le_bytes(self.0) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;
        F128(bits.to_le_bytes())
    }
    fn to_u32(self) -> u32 { self.to_f64() as u32 }
}

/// 位运算 trait
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
// 第十二部分：cast.rs（CastError, ParseError, cast_value, try_cast_value, parse_str）
// =========================================================================

/// 转换错误
#[derive(Debug, Clone, PartialEq)]
pub enum CastError {
    Overflow,
    InvalidCodepoint,
}

/// 解析错误
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    ParseFailed(String),
}

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

pub fn try_cast_value(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag) -> Result<Vec<u8>, CastError> {
    if src_tag == dst_tag {
        return Ok(cast_value(src_tag, src_bytes, dst_tag));
    }

    if src_tag.is_int() && dst_tag == ValueTag::Char {
        let cp = cast_to_u32(src_tag, src_bytes);
        if cp > 0x10FFFF || (0xD800..=0xDFFF).contains(&cp) {
            return Err(CastError::InvalidCodepoint);
        }
        let mut result = vec![0u8; 4];
        write_u32_le(cp, &mut result);
        return Ok(result);
    }

    if src_tag.is_int() && dst_tag.is_int() && src_tag.byte_width() > dst_tag.byte_width() {
        return try_cast_int_narrow(src_tag, src_bytes, dst_tag);
    }

    if src_tag.is_float() && dst_tag.is_int() {
        return try_cast_float_to_int(src_tag, src_bytes, dst_tag);
    }

    Ok(cast_value(src_tag, src_bytes, dst_tag))
}

pub fn parse_str(s: &str, dst_tag: ValueTag) -> Result<Vec<u8>, ParseError> {
    let trimmed = s.trim();
    let result = vec![0u8; dst_tag.byte_width()];

    if trimmed.is_empty() {
        return Err(ParseError::ParseFailed("empty string".to_string()));
    }

    let mut result = result;

    match dst_tag {
        ValueTag::Bool => {
            let b = match trimmed.to_ascii_lowercase().as_str() {
                "true" => true,
                "false" => false,
                _ => return Err(ParseError::ParseFailed(format!("invalid bool: {}", s))),
            };
            write_bool(b, &mut result);
        }
        ValueTag::Char => {
            let mut chars = trimmed.chars();
            let c = chars.next().ok_or_else(|| ParseError::ParseFailed("empty char".to_string()))?;
            if chars.next().is_some() {
                return Err(ParseError::ParseFailed("char must be single character".to_string()));
            }
            write_u32_le(c as u32, &mut result);
        }
        ValueTag::I8 => {
            let v: i8 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_i8(v, &mut result);
        }
        ValueTag::I16 => {
            let v: i16 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_i16_le(v, &mut result);
        }
        ValueTag::I32 => {
            let v: i32 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_i32_le(v, &mut result);
        }
        ValueTag::I64 => {
            let v: i64 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_i64_le(v, &mut result);
        }
        ValueTag::I128 => {
            let v: i128 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_i128_le(v, &mut result);
        }
        ValueTag::U8 => {
            let v: u8 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_u8(v, &mut result);
        }
        ValueTag::U16 => {
            let v: u16 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_u16_le(v, &mut result);
        }
        ValueTag::U32 => {
            let v: u32 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_u32_le(v, &mut result);
        }
        ValueTag::U64 => {
            let v: u64 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_u64_le(v, &mut result);
        }
        ValueTag::U128 => {
            let v: u128 = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_u128_le(v, &mut result);
        }
        ValueTag::Isize => {
            let v: isize = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_i64_le(v as i64, &mut result);
        }
        ValueTag::Usize => {
            let v: usize = trimmed.parse().map_err(|e: std::num::ParseIntError| ParseError::ParseFailed(e.to_string()))?;
            write_u64_le(v as u64, &mut result);
        }
        ValueTag::F32 => {
            let v: f32 = trimmed.parse().map_err(|e: std::num::ParseFloatError| ParseError::ParseFailed(e.to_string()))?;
            write_f32_le(v, &mut result);
        }
        ValueTag::F64 => {
            let v: f64 = trimmed.parse().map_err(|e: std::num::ParseFloatError| ParseError::ParseFailed(e.to_string()))?;
            write_f64_le(v, &mut result);
        }
        ValueTag::F16 => {
            let v: f32 = trimmed.parse().map_err(|e: std::num::ParseFloatError| ParseError::ParseFailed(e.to_string()))?;
            let f16 = F16::from_f32(v);
            write_u16_le(f16.0, &mut result);
        }
        ValueTag::F128 => {
            // 运行时字符串→f128：不经 f64 中转，直接解析为 binary128（与编译期字面量一致）
            // 优先用精确十进制解析；失败时（如 "inf"/"nan"）回退 f64 路径
            if let Some(bits) = crate::ir::Builder::parse_decimal_f128(trimmed) {
                result.copy_from_slice(&bits);
            } else {
                let v: f64 = trimmed.parse().map_err(|e: std::num::ParseFloatError| ParseError::ParseFailed(e.to_string()))?;
                let f128 = F128::from_f64(v);
                result.copy_from_slice(&f128.0);
            }
        }
        _ => return Err(ParseError::ParseFailed(format!("unsupported tag: {:?}", dst_tag))),
    }

    Ok(result)
}

// ---- cast 内部辅助 ----

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
    // F128 目标：整数经 as f64 对 >2^53 的值会丢精度，改用 from_i128/from_u128 精确构造
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

fn try_cast_int_narrow(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag) -> Result<Vec<u8>, CastError> {
    let sval = read_int_as_i128(src_tag, src_bytes);
    let uval = read_int_as_u128(src_tag, src_bytes);

    let in_range = if dst_tag.is_signed() {
        match dst_tag {
            ValueTag::I8 => (i8::MIN as i128..=i8::MAX as i128).contains(&sval),
            ValueTag::I16 => (i16::MIN as i128..=i16::MAX as i128).contains(&sval),
            ValueTag::I32 => (i32::MIN as i128..=i32::MAX as i128).contains(&sval),
            ValueTag::I64 => (i64::MIN as i128..=i64::MAX as i128).contains(&sval),
            ValueTag::Isize => (isize::MIN as i128..=isize::MAX as i128).contains(&sval),
            _ => true,
        }
    } else {
        if src_tag.is_signed() && sval < 0 {
            false
        } else {
            match dst_tag {
                ValueTag::U8 => uval <= u8::MAX as u128,
                ValueTag::U16 => uval <= u16::MAX as u128,
                ValueTag::U32 => uval <= u32::MAX as u128,
                ValueTag::U64 => uval <= u64::MAX as u128,
                ValueTag::Usize => uval <= usize::MAX as u128,
                _ => true,
            }
        }
    };

    if !in_range {
        return Err(CastError::Overflow);
    }

    Ok(cast_value(src_tag, src_bytes, dst_tag))
}

fn try_cast_float_to_int(src_tag: ValueTag, src_bytes: &[u8], dst_tag: ValueTag) -> Result<Vec<u8>, CastError> {
    let f = read_float_as_f64(src_tag, src_bytes);

    if f.is_nan() || f.is_infinite() {
        return Err(CastError::Overflow);
    }

    let in_range = if dst_tag.is_signed() {
        match dst_tag {
            ValueTag::I8 => f >= i8::MIN as f64 && f <= i8::MAX as f64,
            ValueTag::I16 => f >= i16::MIN as f64 && f <= i16::MAX as f64,
            ValueTag::I32 => f >= i32::MIN as f64 && f <= i32::MAX as f64,
            ValueTag::I64 => f >= i64::MIN as f64 && f <= i64::MAX as f64,
            ValueTag::I128 => f >= i128::MIN as f64 && f <= i128::MAX as f64,
            ValueTag::Isize => f >= isize::MIN as f64 && f <= isize::MAX as f64,
            _ => true,
        }
    } else {
        match dst_tag {
            ValueTag::U8 => f >= 0.0 && f <= u8::MAX as f64,
            ValueTag::U16 => f >= 0.0 && f <= u16::MAX as f64,
            ValueTag::U32 => f >= 0.0 && f <= u32::MAX as f64,
            ValueTag::U64 => f >= 0.0 && f <= u64::MAX as f64,
            ValueTag::U128 => f >= 0.0 && f <= u128::MAX as f64,
            ValueTag::Usize => f >= 0.0 && f <= usize::MAX as f64,
            _ => true,
        }
    };

    if !in_range {
        return Err(CastError::Overflow);
    }

    Ok(cast_value(src_tag, src_bytes, dst_tag))
}

// =========================================================================
// 第十三部分：batch.rs（精简）
// =========================================================================

/// 二元运算
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum BinOp {
    Add = 0, Sub = 1, Mul = 2, Div = 3, Mod = 4, Band = 5, Bor = 6, Bxor = 7, Shl = 8, Shr = 9,
}

/// 一元运算
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum UnaryOp {
    Neg = 0, Abs = 1, Bnot = 2,
}

/// 比较运算
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum CmpOp {
    Lt = 0, Gt = 1, Eq = 2, Ne = 3, Le = 4, Ge = 5,
}

/// 归约运算
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add, Mul, Band, Bor, Bxor,
}

/// 大数组并行阈值：超过该长度时启用 rayon 并行分块。
pub const PARALLEL_THRESHOLD: usize = 4096;

/// 计算并行分块大小：将数组切成约 (线程数 × 4) 块，并对齐到 4 lane
/// 以便 SIMD kernel 每个 chunk 都能尽量走满整 lane。
#[inline]
pub fn par_chunk_size(n: usize) -> usize {
    let pieces = rayon::current_num_threads().max(1) * 4;
    let chunk = n.div_ceil(pieces);
    // 向上对齐到 4 的倍数
    let chunk = (chunk + 3) & !3;
    chunk.max(4)
}

/// 通用二元运算分派（标量路径）。大数组（> PARALLEL_THRESHOLD）走 rayon 并行，
/// 小数组走单线程标量以避免线程调度开销。
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

/// 标量二元运算（泛型后备，与原始 for 循环语义完全一致）。
#[inline]
fn binop_scalar_t<T: Num + BitOps>(a: T, b: T, op: BinOp) -> T {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        // 除零语义与标量 arith_div/arith_mod 一致：
        //   - 整数 checked_div/checked_rem 在除零时返回 None → unwrap_or(zero) 返回 0
        //   - 浮点 checked_div/checked_rem 恒返回 Some（除零产生 inf/nan）→ unwrap_or 不触发
        BinOp::Div => a.checked_div(b).unwrap_or(T::zero()),
        BinOp::Mod => a.checked_rem(b).unwrap_or(T::zero()),
        BinOp::Band => a.bit_and(b),
        BinOp::Bor => a.bit_or(b),
        BinOp::Bxor => a.bit_xor(b),
        BinOp::Shl => a.shl(b.to_u32()),
        BinOp::Shr => a.shr(b.to_u32()),
    }
}

/// 通用一元运算分派
pub fn batch_unaryop<T>(dst: &mut [T], a: &[T], op: UnaryOp)
where T: Num + BitOps {
    let n = dst.len().min(a.len());
    for i in 0..n {
        dst[i] = match op {
            UnaryOp::Neg => a[i].wrapping_neg(),
            UnaryOp::Abs => a[i].abs(),
            UnaryOp::Bnot => a[i].bit_not(),
        };
    }
}

/// 批量比较运算：输出 `u8` 掩码（0/1）。大数组走 rayon 并行，小数组走标量。
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

/// 标量比较（泛型后备，按引用比较，因此不要求 T: Copy）。
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

/// 批量归约运算。大数组走 rayon 并行归约（各分块局部归约后再合并，
/// wrapping add/mul 与位运算均满足结合律，结果与顺序归约一致）。
pub fn batch_reduce<T>(a: &[T], op: ReduceOp) -> T
where
    T: Num + BitOps + Send + Sync,
{
    if a.is_empty() {
        return T::zero();
    }
    if a.len() > PARALLEL_THRESHOLD {
        let chunk = par_chunk_size(a.len());
        let partials: Vec<T> = a.par_chunks(chunk).map(|c| reduce_seq(c, op)).collect();
        let mut acc = partials[0];
        for &p in &partials[1..] {
            acc = reduce_combine(acc, p, op);
        }
        acc
    } else {
        reduce_seq(a, op)
    }
}

/// 顺序归约（从 a[0] 起累加，与原始实现语义一致）。
#[inline]
fn reduce_seq<T: Num + BitOps>(a: &[T], op: ReduceOp) -> T {
    if a.is_empty() {
        return T::zero();
    }
    let mut acc = a[0];
    for &v in &a[1..] {
        acc = match op {
            ReduceOp::Add => acc.wrapping_add(v),
            ReduceOp::Mul => acc.wrapping_mul(v),
            ReduceOp::Band => acc.bit_and(v),
            ReduceOp::Bor => acc.bit_or(v),
            ReduceOp::Bxor => acc.bit_xor(v),
        };
    }
    acc
}

/// 合并两个归约部分结果。
#[inline]
fn reduce_combine<T: Num + BitOps>(a: T, b: T, op: ReduceOp) -> T {
    match op {
        ReduceOp::Add => a.wrapping_add(b),
        ReduceOp::Mul => a.wrapping_mul(b),
        ReduceOp::Band => a.bit_and(b),
        ReduceOp::Bor => a.bit_or(b),
        ReduceOp::Bxor => a.bit_xor(b),
    }
}

/// 掩码选择
pub fn batch_select<T>(dst: &mut [T], mask: &[u8], t: &[T], f: &[T])
where T: Copy {
    let n = dst.len().min(mask.len()).min(t.len()).min(f.len());
    for i in 0..n {
        dst[i] = if mask[i] != 0 { t[i] } else { f[i] };
    }
}

/// 广播
pub fn broadcast<T>(dst: &mut [T], val: T)
where T: Copy {
    for slot in dst.iter_mut() {
        *slot = val;
    }
}

// =========================================================================
// 第十三部分补充：SIMD 加速特化（wide crate + rayon）
//
// 对 f32/f64/i32/i64 提供独立的 SIMD 特化函数（4-wide）。
// - 算术/位运算走 SIMD lane，无法向量化的运算（整数 Div/Mod/Shl/Shr、
//   浮点 Mod）回退到标量；
// - 大数组（> PARALLEL_THRESHOLD）走 rayon 并行分块，每块由 SIMD kernel 处理；
// - 这些是 *额外* 的 pub fn，泛型版本（batch_binop 等）保持不变。
// =========================================================================

/// SIMD lane 宽度。
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
        // f32 不支持位运算/移位，保持原值
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

/// f32 SIMD + rayon 并行二元运算。
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

/// f64 SIMD + rayon 并行二元运算。
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
        // 整数除零直接 panic（不回退）
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
    // i32x4 支持算术 + 位运算（均 wrapping，与泛型语义一致）；
    // Div/Mod（无 SIMD 整数除法、且需除零保护）与 Shl/Shr（逐 lane 变长移位
    // 不支持）回退标量。
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

/// i32 SIMD + rayon 并行二元运算。
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

/// i64 SIMD + rayon 并行二元运算。
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

// -------------------- 比较 f32 / f64 --------------------

fn cmp_f32_kernel(dst: &mut [u8], a: &[f32], b: &[f32], op: CmpOp) {
    let n = dst.len().min(a.len()).min(b.len());
    let blocks = n / SIMD_LANES;
    for blk in 0..blocks {
        let i = blk * SIMD_LANES;
        let va = f32x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
        let vb = f32x4::new(b[i..i + SIMD_LANES].try_into().unwrap());
        // wide 浮点比较返回 mask：true 为全 1 位（f32 表现为 NaN），
        // false 为 0.0。用 to_bits() != 0 判定。
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

/// f32 SIMD + rayon 并行比较（输出 u8 掩码）。
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

/// f64 SIMD + rayon 并行比较（输出 u8 掩码）。
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
// SIMD 补全：i8/i16/u8/u16/u32/u64 binop + cmp，以及 i32/i64 cmp
// 每类型用 wide 原生 lane 数（i8x16=16, i16x8=8, i32x4=4, i64x4=4,
// u8x16=16, u16x8=8, u32x4=4, u64x4=4），最大化 SIMD 利用率。
// =========================================================================

/// 通用整数 SIMD binop kernel 生成宏（含乘法）
/// 支持 add/sub/mul/and/or/xor 的 SIMD 加速；div/mod/shl/shr 回退标量。
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

/// i8/u8 专用宏：无 SIMD 乘法（8 位乘法无硬件支持），其余运算同上
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

// i8/u8 无 SIMD 乘法（8 位乘法无硬件支持），其余整数类型有
impl_simd_int_binop_no_mul!(i8, i8x16, 16, binop_i8_scalar);
impl_simd_int_binop!(i16, i16x8, 8, binop_i16_scalar);
impl_simd_int_binop_no_mul!(u8, u8x16, 16, binop_u8_scalar);
impl_simd_int_binop!(u16, u16x8, 8, binop_u16_scalar);
impl_simd_int_binop!(u32, u32x4, 4, binop_u32_scalar);
impl_simd_int_binop!(u64, u64x4, 4, binop_u64_scalar);

// -------------------- 有符号整数 SIMD cmp（i8/i16/i32/i64）--------------------
// wide 对有符号整数提供 CmpEq/CmpLt/CmpGt，其余组合：Ne=!Eq, Le=Lt|Eq, Ge=Gt|Eq

/// 有符号整数 SIMD cmp kernel 生成宏
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
                    // wide 有符号整数比较返回同类型 mask（全 1/0），转 bool
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

// -------------------- 无符号整数 SIMD cmp（u8/u16/u32/u64）--------------------
// wide 对无符号整数仅提供 CmpEq，其余比较回退标量（无 SIMD 无符号比较指令）

/// 无符号整数 SIMD cmp kernel 生成宏：仅 Eq/Ne 走 SIMD，其余标量
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

// -------------------- 归约 f32 / f64 --------------------

fn reduce_add_f32_seq(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let blocks = n / SIMD_LANES;
    let mut acc = f32x4::splat(0.0);
    for blk in 0..blocks {
        let i = blk * SIMD_LANES;
        acc += f32x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
    }
    let mut sum = acc.reduce_add();
    for &v in a.iter().skip(blocks * SIMD_LANES) {
        sum += v;
    }
    sum
}

/// f32 归约：Add 走 SIMD（+ rayon 并行），Mul 走标量，位运算对 f32 无意义。
pub fn batch_reduce_f32(a: &[f32], op: ReduceOp) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    match op {
        ReduceOp::Add => {
            let n = a.len();
            if n > PARALLEL_THRESHOLD {
                let chunk = par_chunk_size(n);
                let partials: Vec<f32> =
                    a.par_chunks(chunk).map(reduce_add_f32_seq).collect();
                partials.iter().copied().fold(0.0, |x, y| x + y)
            } else {
                reduce_add_f32_seq(a)
            }
        }
        _ => {
            let mut acc = a[0];
            for &v in &a[1..] {
                acc = match op {
                    ReduceOp::Mul => acc * v,
                    _ => acc,
                };
            }
            acc
        }
    }
}

fn reduce_add_f64_seq(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let blocks = n / SIMD_LANES;
    let mut acc = f64x4::splat(0.0);
    for blk in 0..blocks {
        let i = blk * SIMD_LANES;
        acc += f64x4::new(a[i..i + SIMD_LANES].try_into().unwrap());
    }
    let mut sum = acc.reduce_add();
    for &v in a.iter().skip(blocks * SIMD_LANES) {
        sum += v;
    }
    sum
}

/// f64 归约：Add 走 SIMD（+ rayon 并行），Mul 走标量。
pub fn batch_reduce_f64(a: &[f64], op: ReduceOp) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    match op {
        ReduceOp::Add => {
            let n = a.len();
            if n > PARALLEL_THRESHOLD {
                let chunk = par_chunk_size(n);
                let partials: Vec<f64> =
                    a.par_chunks(chunk).map(reduce_add_f64_seq).collect();
                partials.iter().copied().fold(0.0, |x, y| x + y)
            } else {
                reduce_add_f64_seq(a)
            }
        }
        _ => {
            let mut acc = a[0];
            for &v in &a[1..] {
                acc = match op {
                    ReduceOp::Mul => acc * v,
                    _ => acc,
                };
            }
            acc
        }
    }
}

// =========================================================================
// 第十四部分：allocator.rs
// =========================================================================

/// 内存分配器 trait
pub trait Allocator: Clone {
    fn alloc_str(&self, s: &str) -> Arc<str>;
    fn alloc_array(&self, vals: Vec<ValueHandle>) -> Arc<Vec<ValueHandle>>;
    fn alloc_value(&self, val: ValueHandle) -> ValueHandle {
        val
    }
}

/// 默认分配器
#[derive(Debug, Clone, Default)]
pub struct DefaultAllocator;

impl Allocator for DefaultAllocator {
    fn alloc_str(&self, s: &str) -> Arc<str> {
        Arc::from(s)
    }
    fn alloc_array(&self, vals: Vec<ValueHandle>) -> Arc<Vec<ValueHandle>> {
        Arc::new(vals)
    }
}

pub fn default_allocator() -> DefaultAllocator {
    DefaultAllocator
}

// =========================================================================
// 第十五部分：纯算术核心 — 无 Frame 依赖，runtime compute_fn 与编译期 ConstFold 共用
// =========================================================================
//
// 为所有整数/浮点类型生成纯算术函数，语义与 Engine.rs 的 compute_fn 宏严格一致：
//   - 整数 add/sub/mul: wrapping 语义
//   - 整数 div/mod: checked，除零返回 0
//   - 整数 shl/shr: 移位量为 i32（与 Engine.rs 读取 as_i32 一致），cast u32 后 wrapping
//   - 浮点 div: 原生除法（除零产生 inf/nan）
// runtime compute_fn 调用这些纯函数（复用），编译期 ConstFold 也调用同一份算术（解耦 Frame）。

/// 为指定整数类型生成全套纯算术函数（add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot）。
/// shl/shr 的移位量参数为 i32（与 Engine.rs compute_shl_*/compute_shr_* 读取 as_i32 一致）。
macro_rules! impl_arith_int {
    ($ty:ident, $rust:ty) => {
        paste! {
            #[inline] pub fn [<arith_add_$ty>](a: $rust, b: $rust) -> $rust { a.wrapping_add(b) }
            #[inline] pub fn [<arith_sub_$ty>](a: $rust, b: $rust) -> $rust { a.wrapping_sub(b) }
            #[inline] pub fn [<arith_mul_$ty>](a: $rust, b: $rust) -> $rust { a.wrapping_mul(b) }
            #[inline] pub fn [<arith_div_$ty>](a: $rust, b: $rust) -> $rust { if b == 0 { 0 } else { a.wrapping_div(b) } }
            #[inline] pub fn [<arith_mod_$ty>](a: $rust, b: $rust) -> $rust { if b == 0 { 0 } else { a.wrapping_rem(b) } }
            #[inline] pub fn [<arith_bitand_$ty>](a: $rust, b: $rust) -> $rust { a & b }
            #[inline] pub fn [<arith_bitor_$ty>](a: $rust, b: $rust) -> $rust { a | b }
            #[inline] pub fn [<arith_bitxor_$ty>](a: $rust, b: $rust) -> $rust { a ^ b }
            #[inline] pub fn [<arith_shl_$ty>](a: $rust, shift: i32) -> $rust { a.wrapping_shl(shift as u32) }
            #[inline] pub fn [<arith_shr_$ty>](a: $rust, shift: i32) -> $rust { a.wrapping_shr(shift as u32) }
            #[inline] pub fn [<arith_neg_$ty>](a: $rust) -> $rust { a.wrapping_neg() }
            #[inline] pub fn [<arith_bitnot_$ty>](a: $rust) -> $rust { !a }
        }
    };
}

/// 为指定浮点类型生成全套纯算术函数（add/sub/mul/div/mod/neg）。
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

// 整数类型展开（12 类型 × 12 运算）
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

// 浮点类型展开（4 类型 × 6 运算）
impl_arith_float!(f16, F16);
impl_arith_float!(f32, f32);
impl_arith_float!(f64, f64);
impl_arith_float!(f128, F128);

// =========================================================================
// 布尔纯算术 — 与 Engine.rs compute_and_bool/or/not 语义一致
// =========================================================================

#[inline] pub fn arith_and_bool(a: bool, b: bool) -> bool { a && b }
#[inline] pub fn arith_or_bool(a: bool, b: bool) -> bool { a || b }
#[inline] pub fn arith_not_bool(a: bool) -> bool { !a }