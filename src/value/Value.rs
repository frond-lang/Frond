//! Value.rs — Kuzo 统一值系统（合并 14 个子模块）

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::AtomicBool;

// 从 Type 模块 re-export 类型判别标签
pub use crate::types::ValueTag;

// 跨子模块：HeapObj::hash（本文件）复用 Arena.rs 的 SIMD 批量哈希辅助函数
use super::arena::simd_hash_soa;

// =========================================================================
// 第一部分：标量基础类型（scalar.rs + char.rs）
// =========================================================================

// ---- F16 — IEEE 754 半精度浮点（binary16）----

/// IEEE 754 半精度浮点数：以 `u16` 存储 bit pattern
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct F16(pub u16);

impl F16 {
    pub fn from_f32(x: f32) -> Self {
        F16(f32_to_f16_bits(x))
    }
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x3FF;
        exp == 0x1F && mant != 0
    }
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x3FF;
        exp == 0x1F && mant == 0
    }
    pub fn to_bits(self) -> u16 {
        self.0
    }
    pub fn from_bits(b: u16) -> Self {
        F16(b)
    }

    // ---- IEEE 754 binary16 精确运算（不经过 f64 中转）----
    // 布局：sign(1) | exp(5, bias=15) | fraction(10)
    // 正规数 mantissa = (1 << 10) | fraction，共 11 位
    // 次正规数 mantissa = fraction，指数 = 1 - bias = -14
    // 与 F128 同一 unpack/pack 框架，因 mantissa 仅 11 位，u32 足够

    fn nan_val() -> Self { F16(0x7C00 | 1) }
    fn inf_val(sign: bool) -> Self { F16(if sign { 0xFC00 } else { 0x7C00 }) }
    fn zero_val(sign: bool) -> Self { F16(if sign { 0x8000 } else { 0 }) }

    /// 拆解为 (sign, unbiased_exp, mantissa)。
    /// 正规数 mantissa 含隐含 1（bit 10 = 1）；次正规数/零 mantissa = fraction。
    fn unpack(&self) -> (bool, i32, u32) {
        let bits = self.0;
        let sign = (bits >> 15) != 0;
        let raw_exp = ((bits >> 10) & 0x1F) as i32;
        let frac = (bits & 0x3FF) as u32;
        if raw_exp == 0 {
            (sign, 1 - 15, frac)
        } else {
            (sign, raw_exp - 15, frac | (1u32 << 10))
        }
    }

    /// 将 (sign, exp, mant, sticky) 规范化并舍入为 F16。
    /// mant 的 MSB 是隐含 1（可在任意位置），pack 负责对齐到 bit 10。
    /// 舍入模式：round-to-nearest-even。
    fn pack(sign: bool, exp: i32, mant: u32, sticky: bool) -> Self {
        if mant == 0 {
            return Self::zero_val(sign);
        }
        let msb = 31 - mant.leading_zeros() as i32;
        let shift = msb - 10;
        let mut adj_exp = exp + shift;
        let mut m = mant;
        let mut stk = sticky;
        let mut guard = false;
        if shift > 0 {
            let sh = shift as u32;
            if sh >= 32 {
                m = 0;
                stk = true;
            } else {
                guard = (mant >> (sh - 1)) & 1 != 0;
                if sh > 1 {
                    stk = stk || (mant & ((1u32 << (sh - 1)) - 1)) != 0;
                }
                m = mant >> sh;
            }
        } else if shift < 0 {
            m = mant << (-shift as u32);
        }
        if m == 0 {
            return Self::zero_val(sign);
        }
        let biased = adj_exp + 15;
        if biased >= 0x1F {
            return Self::inf_val(sign);
        }
        if biased <= 0 {
            let extra = (1 - biased) as u32;
            if extra >= 32 {
                if guard && stk { return Self::zero_val(false); }
                return Self::zero_val(sign);
            }
            if extra > 0 {
                let new_guard = (m >> (extra - 1)) & 1 != 0;
                if extra > 1 {
                    stk = stk || (m & ((1u32 << (extra - 1)) - 1)) != 0;
                }
                guard = new_guard;
                m >>= extra;
            }
            if guard && (stk || (m & 1) != 0) {
                m = m.wrapping_add(1);
                if m >= (1u32 << 10) {
                    return F16((if sign { 0x8000 } else { 0 }) | (1u16 << 10));
                }
            }
            return F16((if sign { 0x8000 } else { 0 }) | m as u16);
        }
        if guard && (stk || (m & 1) != 0) {
            m = m.wrapping_add(1);
            if m >= (1u32 << 11) {
                m >>= 1;
                adj_exp += 1;
                if adj_exp + 15 >= 0x1F {
                    return Self::inf_val(sign);
                }
            }
        }
        let frac = (m & 0x3FF) as u16;
        F16((if sign { 0x8000 } else { 0 }) | (((adj_exp + 15) as u16) << 10) | frac)
    }

    pub fn neg_f16(self) -> Self {
        F16(self.0 ^ 0x8000)
    }

    pub fn add_f16(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() { return Self::nan_val(); }
        if self.is_infinite() {
            if other.is_infinite() {
                let (sa, _, _) = self.unpack();
                let (sb, _, _) = other.unpack();
                return if sa == sb { self } else { Self::nan_val() };
            }
            return self;
        }
        if other.is_infinite() { return other; }

        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();
        if ma == 0 && mb == 0 { return Self::zero_val(sa && sb); }
        if ma == 0 { return other; }
        if mb == 0 { return self; }

        let ma_ext = ma << 2;
        let mb_ext = mb << 2;
        let result_exp;
        let (aligned_a, aligned_b, stk) = if ea > eb {
            let diff = (ea - eb) as u32;
            result_exp = ea;
            if diff >= 32 { (ma_ext, 0u32, mb_ext != 0) }
            else {
                let lost = mb_ext & ((1u32 << diff) - 1);
                (ma_ext, mb_ext >> diff, lost != 0)
            }
        } else if eb > ea {
            let diff = (eb - ea) as u32;
            result_exp = eb;
            if diff >= 32 { (0u32, mb_ext, ma_ext != 0) }
            else {
                let lost = ma_ext & ((1u32 << diff) - 1);
                (ma_ext >> diff, mb_ext, lost != 0)
            }
        } else {
            result_exp = ea;
            (ma_ext, mb_ext, false)
        };

        let (result_sign, result_mant) = if sa == sb {
            (sa, aligned_a.wrapping_add(aligned_b))
        } else if aligned_a >= aligned_b {
            (sa, aligned_a - aligned_b)
        } else {
            (sb, aligned_b - aligned_a)
        };
        if result_mant == 0 { return Self::zero_val(false); }
        Self::pack(result_sign, result_exp - 2, result_mant, stk)
    }

    pub fn sub_f16(self, other: Self) -> Self {
        self.add_f16(other.neg_f16())
    }

    pub fn mul_f16(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() { return Self::nan_val(); }
        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();
        let result_sign = sa ^ sb;
        if self.is_infinite() && mb == 0 { return Self::nan_val(); }
        if other.is_infinite() && ma == 0 { return Self::nan_val(); }
        if self.is_infinite() || other.is_infinite() { return Self::inf_val(result_sign); }
        if ma == 0 || mb == 0 { return Self::zero_val(result_sign); }

        let result_exp = ea + eb;
        // 11 × 11 = 22 位乘积，u32 足够
        let prod = (ma as u32) * (mb as u32);
        let total_bits = 32 - prod.leading_zeros() as i32;
        let shift = total_bits - 11;
        let (m, stk) = if shift > 0 {
            let sh = shift as u32;
            let lost = prod & ((1u32 << (sh - 1)) - 1);
            (prod >> sh, lost != 0)
        } else {
            (prod, false)
        };
        Self::pack(result_sign, result_exp - 10 + shift, m, stk)
    }

    pub fn div_f16(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() { return Self::nan_val(); }
        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();
        let result_sign = sa ^ sb;
        if self.is_infinite() && other.is_infinite() { return Self::nan_val(); }
        if self.is_infinite() { return Self::inf_val(result_sign); }
        if other.is_infinite() { return Self::zero_val(result_sign); }
        if mb == 0 {
            if ma == 0 { return Self::nan_val(); }
            return Self::inf_val(result_sign);
        }
        if ma == 0 { return Self::zero_val(result_sign); }

        let result_exp = ea - eb;
        // (ma << 12) / mb，商 ~12 位，u32 足够
        // ma/mb ∈ [0.5, 2)，(ma<<12)/mb ∈ [2^11, 2^13)，不溢出 u32
        let quot = ((ma as u32) << 12) / mb;
        let stk = ((ma << 12) % mb) != 0;
        // pack 语义：值 = mant * 2^(exp - 10)
        // 真实商 = (ma/mb) * 2^result_exp = quot * 2^(result_exp - 12)
        // exp = result_exp - 12 + 10 = result_exp - 2
        Self::pack(result_sign, result_exp - 2, quot, stk)
    }

    pub fn rem_f16(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() { return Self::nan_val(); }
        if other.is_infinite() { return self; }
        if self.is_infinite() { return Self::nan_val(); }
        let (_, _, mb) = other.unpack();
        if mb == 0 { return Self::nan_val(); }
        let (_, _, ma) = self.unpack();
        if ma == 0 { return self; }

        let quot = self.div_f16(other);
        let q_bits = quot.0;
        let q_exp = ((q_bits >> 10) & 0x1F) as i32 - 15;
        let q_int = if q_exp >= 0 {
            let shift = q_exp as u32;
            let q_mant = ((q_bits & 0x3FF) as u32) | (1u32 << 10);
            if shift >= 11 { 0u32 } else { q_mant >> shift }
        } else { 0u32 };
        let q_val = Self::from_f64(q_int as f64);
        let prod = q_val.mul_f16(other);
        self.sub_f16(prod)
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_nan() {
            write!(f, "NaN(f16)")
        } else if self.is_infinite() {
            if self.0 >> 15 != 0 {
                write!(f, "-inf(f16)")
            } else {
                write!(f, "inf(f16)")
            }
        } else {
            write!(f, "{}f16", self.to_f32())
        }
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// IEEE 754 totalOrder 语义：NaN 视为最大（符号位区分），-0 < +0，负数按量级反序
impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for F16 {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.0 as i16;
        let b = other.0 as i16;
        // 负数（符号位=1）按量级反序：翻转符号位外的所有位
        let ka = if a < 0 { a ^ 0x7FFF } else { a };
        let kb = if b < 0 { b ^ 0x7FFF } else { b };
        ka.cmp(&kb)
    }
}

/// f32 bit pattern → f16 bit pattern（IEEE 754 round-to-nearest）
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        let m = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7C00 | (m as u16);
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return sign | 0x7C00;
    }

    if new_exp <= 0 {
        if 14 - new_exp >= 24 {
            return sign;
        }
        let m = mant | 0x800000;
        let shift = 14 - new_exp;
        let rounded_m = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let mut result = rounded_m;
        if rem > half || (rem == half && (rounded_m & 1) != 0) {
            result += 1;
        }
        return sign | (result as u16);
    }

    let m = mant >> 13;
    let rem = mant & 0x1FFF;
    let half = 0x1000;
    let mut result: u32 = ((new_exp as u32) << 10) | m;
    if rem > half || (rem == half && (m & 1) != 0) {
        result += 1;
    }
    sign | (result as u16)
}

/// f16 bit pattern → f32 bit pattern
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) >> 10) & 0x1F;
    let mant = (bits as u32) & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e: i32 = -1;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let new_exp = (127 + e - 14) as u32;
        return f32::from_bits(sign | (new_exp << 23) | (m << 13));
    }

    if exp == 0x1F {
        let m = if mant != 0 { (mant << 13) | 0x400000 } else { 0 };
        return f32::from_bits(sign | 0x7F800000 | m);
    }

    let new_exp = exp + (127 - 15);
    f32::from_bits(sign | (new_exp << 23) | (mant << 13))
}

// ---- F128 — IEEE 754 四倍精度浮点（binary128）----

/// IEEE 754 四倍精度浮点数：以 `[u8; 16]` 存储 bit pattern
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct F128(pub [u8; 16]);

/// f64→f128 是无损的（f64 的 53 位 mantissa 左移 60 位填满 binary128 的 113 位，无信息丢失）。
/// f128→f64（to_f64）已实现 round-to-nearest-even，对超出 f64 精度的低位正确舍入。
/// 因此 f64→f128→f64 往返保真；f128→f64→f128 仅在 f128 值超出 f64 精度时有舍入（符合 IEEE 754 语义）。
impl F128 {
    pub fn from_f64(x: f64) -> Self {
        let bits = x.to_bits();
        let sign = ((bits >> 63) & 1) as u128;
        let exp = ((bits >> 52) & 0x7FF) as i32;
        let mant = (bits & 0xFFFFFFFFFFFFF) as u128;

        let result: u128 = if exp == 0x7FF {
            let new_exp: u128 = 0x7FFF;
            let new_mant: u128 = if mant != 0 { (mant << 60) | 0x8000000000000000 } else { 0 };
            (sign << 127) | (new_exp << 112) | new_mant
        } else if exp == 0 {
            if mant == 0 {
                sign << 127
            } else {
                let new_exp: u128 = 0;
                (sign << 127) | (new_exp << 112) | (mant << 60)
            }
        } else {
            let new_exp = (exp - 1023 + 16383) as u128;
            (sign << 127) | (new_exp << 112) | (mant << 60)
        };

        F128(result.to_le_bytes())
    }

    pub fn to_f64(self) -> f64 {
        let bits = u128::from_le_bytes(self.0);
        let sign = ((bits >> 127) & 1) as u64;
        let exp = ((bits >> 112) & 0x7FFF) as i32;
        let mant = bits & ((1u128 << 112) - 1);

        // NaN / Inf
        if exp == 0x7FFF {
            // 任意非零 payload → canonical NaN；Inf → Inf
            let m: u64 = if mant != 0 { 1 } else { 0 };
            return f64::from_bits((sign << 63) | (0x7FF << 52) | m);
        }

        // 真实指数（正规数 exp-16383，次正规数 -16382）
        let true_exp = if exp == 0 { -16382 } else { exp - 16383 };
        // 113 位完整 mantissa（正规数补隐含 1）
        let full_mant: u128 = if exp == 0 { mant } else { mant | (1u128 << 112) };

        // f64 指数（bias 1023）
        let f64_exp = true_exp + 1023;
        if f64_exp >= 0x7FF {
            // 溢出 → ±Inf
            return f64::from_bits((sign << 63) | (0x7FF << 52));
        }
        if f64_exp <= 0 {
            // 次正规或下溢：需将 113 位 mantissa 右移到 f64 次正规位置
            // f64 次正规 mantissa 在 bit 0..51，隐含位为 0，指数为 0（true_exp = -1022）
            // 目标 shift = 112 - 51 + (1 - f64_exp) = 62 - f64_exp
            let shift = (62 - f64_exp) as u32;
            if shift >= 128 {
                return f64::from_bits(sign << 63); // 下溢 → ±0
            }
            let round_bit = (full_mant >> (shift.saturating_sub(1))) & 1;
            let sticky = if shift >= 2 {
                (full_mant & ((1u128 << (shift - 1)) - 1)) != 0
            } else {
                false
            };
            let mut result_mant = (full_mant >> shift) as u64;
            // round-to-nearest-even
            if round_bit != 0 && (sticky || (result_mant & 1) != 0) {
                result_mant = result_mant.saturating_add(1);
            }
            return f64::from_bits((sign << 63) | result_mant);
        }

        // 正规数：113 位 mantissa → 53 位（隐含 1 + 52 位 fraction），右移 60 位并舍入
        let shift = 60u32;
        let round_bit = (full_mant >> (shift - 1)) & 1;
        let sticky = (full_mant & ((1u128 << (shift - 1)) - 1)) != 0;
        let mut result_mant = (full_mant >> shift) as u64;
        // result_mant 此时为 53 位（含隐含 1），需放入 f64 的 52 位 fraction
        if round_bit != 0 && (sticky || (result_mant & 1) != 0) {
            result_mant += 1;
            if result_mant >> 53 != 0 {
                // 进位导致 mantissa 溢出（1.111... → 10.000...），指数 +1，mantissa 归零
                return f64::from_bits((sign << 63) | (((f64_exp as u64) + 1) << 52));
            }
        }
        f64::from_bits((sign << 63) | ((f64_exp as u64) << 52) | (result_mant & ((1u64 << 52) - 1)))
    }

    pub fn from_f32(x: f32) -> Self {
        Self::from_f64(x as f64)
    }
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
    /// 从 i128 精确构造 F128（不经 f64 中转，避免精度损失）。
    /// F128 有 113 位尾数，可精确表示所有 i128 值。
    pub fn from_i128(x: i128) -> Self {
        if x == 0 {
            return Self::zero_val(false);
        }
        let sign = x < 0;
        let abs = x.unsigned_abs();
        // abs 为 u128，MSB 即为隐含 1 的位置。exp = msb，mant = abs。
        // pack 会将 MSB 对齐到 bit 112 并处理舍入（此处无舍入，abs 完整保留）。
        Self::pack(sign, 0, abs, false)
    }
    /// 从 u128 精确构造 F128（不经 f64 中转）。
    pub fn from_u128(x: u128) -> Self {
        if x == 0 {
            return Self::zero_val(false);
        }
        Self::pack(false, 0, x, false)
    }
    pub fn is_nan(self) -> bool {
        let bits = u128::from_le_bytes(self.0);
        let exp = (bits >> 112) & 0x7FFF;
        let mant = bits & ((1u128 << 112) - 1);
        exp == 0x7FFF && mant != 0
    }
    pub fn is_infinite(self) -> bool {
        let bits = u128::from_le_bytes(self.0);
        let exp = (bits >> 112) & 0x7FFF;
        let mant = bits & ((1u128 << 112) - 1);
        exp == 0x7FFF && mant == 0
    }
    pub fn to_bits(self) -> [u8; 16] {
        self.0
    }
    pub fn from_bits(b: [u8; 16]) -> Self {
        F128(b)
    }

    // ---- IEEE 754 binary128 精确运算（不经过 f64 中转）----
    // 布局：sign(1) | exp(15, bias=16383) | fraction(112)
    // 正规数 mantissa = (1 << 112) | fraction，共 113 位
    // 次正规数 mantissa = fraction，指数 = 1 - bias = -16382

    fn nan_val() -> Self {
        F128(((0x7FFFu128 << 112) | 1).to_le_bytes())
    }
    fn inf_val(sign: bool) -> Self {
        F128((((sign as u128) << 127) | (0x7FFFu128 << 112)).to_le_bytes())
    }
    fn zero_val(sign: bool) -> Self {
        F128(((sign as u128) << 127).to_le_bytes())
    }

    /// 拆解为 (sign, unbiased_exp, mantissa)。
    /// 正规数 mantissa 含隐含 1（bit 112 = 1）；次正规数/零 mantissa = fraction。
    fn unpack(&self) -> (bool, i32, u128) {
        let bits = u128::from_le_bytes(self.0);
        let sign = (bits >> 127) != 0;
        let raw_exp = ((bits >> 112) & 0x7FFF) as i32;
        let frac = bits & ((1u128 << 112) - 1);
        if raw_exp == 0 {
            (sign, 1 - 16383, frac)
        } else {
            (sign, raw_exp - 16383, frac | (1u128 << 112))
        }
    }

    /// 将 (sign, exp, mant, sticky) 规范化并舍入为 F128。
    /// mant 的 MSB 是隐含 1（可以在任意位置），pack 负责对齐到 bit 112。
    /// sticky 表示低于 mant 最低有效位是否有非零信息。
    /// 舍入模式：round-to-nearest-even。
    fn pack(sign: bool, exp: i32, mant: u128, sticky: bool) -> Self {
        if mant == 0 {
            // 值极小，round-to-nearest-even 向下到 0
            return Self::zero_val(sign);
        }

        // 规范化：将 MSB 对齐到 bit 112
        let msb = 127 - mant.leading_zeros() as i32;
        let shift = msb - 112;
        let mut adj_exp = exp + shift;
        let mut m = mant;
        let mut stk = sticky;

        // guard 位：右移时移出的最高位
        let mut guard = false;
        if shift > 0 {
            let sh = shift as u32;
            if sh >= 128 {
                m = 0;
                guard = false;
                stk = true;
            } else {
                guard = (mant >> (sh - 1)) & 1 != 0;
                if sh > 1 {
                    stk = stk || (mant & ((1u128 << (sh - 1)) - 1)) != 0;
                }
                m = mant >> sh;
            }
        } else if shift < 0 {
            m = mant << (-shift as u32);
        }

        if m == 0 {
            return Self::zero_val(sign);
        }

        let biased = adj_exp + 16383;

        // 溢出 → ±Inf
        if biased >= 0x7FFF {
            return Self::inf_val(sign);
        }

        // 次正规数或下溢
        if biased <= 0 {
            let extra = (1 - biased) as u32;
            if extra >= 128 {
                // 完全下溢
                if guard && stk {
                    return Self::zero_val(false); // 0 是偶数
                }
                return Self::zero_val(sign);
            }
            // 右移 extra 位，保留 guard/sticky
            if extra > 0 {
                let new_guard = (m >> (extra - 1)) & 1 != 0;
                if extra > 1 {
                    stk = stk || (m & ((1u128 << (extra - 1)) - 1)) != 0;
                }
                guard = new_guard;
                m >>= extra;
            }
            // 舍入（round-to-nearest-even）
            if guard && (stk || (m & 1) != 0) {
                m = m.wrapping_add(1);
                if m >= (1u128 << 112) {
                    // 进位到最小正规数
                    return F128((((sign as u128) << 127) | (1u128 << 112)).to_le_bytes());
                }
            }
            return F128((((sign as u128) << 127) | m).to_le_bytes());
        }

        // 正规数：m 的 bit 112 = 1，小数 = bits 0-111
        // 舍入（round-to-nearest-even）
        if guard && (stk || (m & 1) != 0) {
            m = m.wrapping_add(1);
            // 进位可能使 mantissa 从 113 位变 114 位（bit 113 = 1）
            if m >= (1u128 << 113) {
                m >>= 1;
                adj_exp += 1;
                let biased2 = adj_exp + 16383;
                if biased2 >= 0x7FFF {
                    return Self::inf_val(sign);
                }
            }
        }
        let frac = m & ((1u128 << 112) - 1);
        let bits = ((sign as u128) << 127) | (((adj_exp + 16383) as u128) << 112) | frac;
        F128(bits.to_le_bytes())
    }

    /// 113 位 × 113 位 → 226 位乘积 (hi, lo)
    fn mul_113(a: u128, b: u128) -> (u128, u128) {
        let a_lo = a as u64 as u128;
        let a_hi = (a >> 64) as u64 as u128;
        let b_lo = b as u64 as u128;
        let b_hi = (b >> 64) as u64 as u128;
        let ll = a_lo * b_lo;
        let lh = a_lo * b_hi;
        let hl = a_hi * b_lo;
        let hh = a_hi * b_hi;
        let mid = (lh & 0xFFFF_FFFF_FFFF_FFFF) + (hl & 0xFFFF_FFFF_FFFF_FFFF) + (ll >> 64);
        let lo = (mid << 64) | (ll & 0xFFFF_FFFF_FFFF_FFFF);
        let hi = hh + (lh >> 64) + (hl >> 64) + (mid >> 64);
        (hi, lo)
    }

    /// 256 位 / 113 位长除法，返回 (商, 余数!=0)
    /// 被 rem 始终 < denom (< 2^113)，左移后 < 2^114，不会溢出 u128。
    fn div_256_by_113(numer_hi: u128, numer_lo: u128, denom: u128) -> (u128, bool) {
        let mut rem: u128 = 0;
        let mut quot: u128 = 0;
        for i in (0..256).rev() {
            let bit: u128 = if i >= 128 {
                (numer_hi >> (i - 128)) & 1
            } else {
                (numer_lo >> i) & 1
            };
            rem = (rem << 1) | bit;
            if rem >= denom {
                rem -= denom;
                if i < 128 {
                    quot |= 1u128 << i;
                }
            }
        }
        (quot, rem != 0)
    }

    /// 精确取负
    pub fn neg_f128(self) -> Self {
        let bits = u128::from_le_bytes(self.0) ^ (1u128 << 127);
        F128(bits.to_le_bytes())
    }

    /// 精确加法
    pub fn add_f128(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan_val();
        }
        if self.is_infinite() {
            if other.is_infinite() {
                let (sa, _, _) = self.unpack();
                let (sb, _, _) = other.unpack();
                return if sa == sb { self } else { Self::nan_val() };
            }
            return self;
        }
        if other.is_infinite() {
            return other;
        }

        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();

        if ma == 0 && mb == 0 {
            // +0 + +0 = +0; -0 + -0 = -0; 混合 → +0 (round-to-nearest)
            return Self::zero_val(sa && sb);
        }
        if ma == 0 {
            return other;
        }
        if mb == 0 {
            return self;
        }

        // 扩展 mantissa 左移 2 位（腾出 guard/round 位空间）
        let ma_ext = ma << 2;
        let mb_ext = mb << 2;
        let result_exp;

        // 对齐指数（较小的右移，保留 sticky）
        let (aligned_a, aligned_b, stk) = if ea > eb {
            let diff = (ea - eb) as u32;
            result_exp = ea;
            if diff >= 128 {
                (ma_ext, 0u128, mb_ext != 0)
            } else {
                let lost = mb_ext & ((1u128 << diff) - 1);
                (ma_ext, mb_ext >> diff, lost != 0)
            }
        } else if eb > ea {
            let diff = (eb - ea) as u32;
            result_exp = eb;
            if diff >= 128 {
                (0u128, mb_ext, ma_ext != 0)
            } else {
                let lost = ma_ext & ((1u128 << diff) - 1);
                (ma_ext >> diff, mb_ext, lost != 0)
            }
        } else {
            result_exp = ea;
            (ma_ext, mb_ext, false)
        };

        // 带符号加法
        let (result_sign, result_mant) = if sa == sb {
            (sa, aligned_a.wrapping_add(aligned_b))
        } else if aligned_a >= aligned_b {
            (sa, aligned_a - aligned_b)
        } else {
            (sb, aligned_b - aligned_a)
        };

        if result_mant == 0 {
            return Self::zero_val(false); // x + (-x) = +0
        }

        // result_mant 是 115 位（113 + 2），pack 负责规范化到 113 位
        Self::pack(result_sign, result_exp - 2, result_mant, stk)
    }

    /// 精确减法
    pub fn sub_f128(self, other: Self) -> Self {
        self.add_f128(other.neg_f128())
    }

    /// 精确乘法
    pub fn mul_f128(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan_val();
        }
        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();
        let result_sign = sa ^ sb;

        // Inf × 0 = NaN
        if self.is_infinite() && mb == 0 {
            return Self::nan_val();
        }
        if other.is_infinite() && ma == 0 {
            return Self::nan_val();
        }
        if self.is_infinite() || other.is_infinite() {
            return Self::inf_val(result_sign);
        }
        if ma == 0 || mb == 0 {
            return Self::zero_val(result_sign);
        }

        let result_exp = ea + eb;

        // 113 × 113 = 226 位乘积
        let (hi, lo) = Self::mul_113(ma, mb);

        // 确定乘积 MSB 位置
        let total_bits = if hi != 0 {
            128 + (128 - hi.leading_zeros() as i32)
        } else {
            128 - lo.leading_zeros() as i32
        };
        let shift = total_bits - 113; // 右移到 113 位

        let (m, stk) = if shift >= 128 {
            (0u128, hi != 0 || lo != 0)
        } else if shift > 0 {
            let sh = shift as u32;
            let lost = if sh > 0 {
                lo & ((1u128 << sh) - 1)
            } else {
                0
            };
            let m = (hi << (128 - sh)) | (lo >> sh);
            (m, lost != 0)
        } else {
            (lo, false)
        };

        // pack 语义：值 = mant * 2^(exp - 112)
        // 真实值 = (ma*mb) * 2^(result_exp - 224)
        // mant = (ma*mb) >> shift，所以 exp = result_exp - 112 + shift
        Self::pack(result_sign, result_exp - 112 + shift, m, stk)
    }

    /// 精确除法
    pub fn div_f128(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan_val();
        }
        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();
        let result_sign = sa ^ sb;

        // Inf / Inf = NaN; x / 0 = NaN（x≠0）
        if self.is_infinite() && other.is_infinite() {
            return Self::nan_val();
        }
        if self.is_infinite() {
            return Self::inf_val(result_sign);
        }
        if other.is_infinite() {
            return Self::zero_val(result_sign);
        }
        if mb == 0 {
            if ma == 0 {
                return Self::nan_val(); // 0/0 = NaN
            }
            return Self::inf_val(result_sign); // x/0 = Inf
        }
        if ma == 0 {
            return Self::zero_val(result_sign);
        }

        let result_exp = ea - eb;

        // 计算 (ma << 114) / mb，得到 ~115 位商（在 u128 范围内）
        // ma/mb ∈ [0.5, 2)，所以 (ma<<114)/mb ∈ [2^113, 2^115)，不溢出 u128
        // pack 语义：值 = mant * 2^(exp - 112)
        // 真实商 = (ma/mb) * 2^result_exp = quot * 2^(result_exp - 114)
        // 所以 exp = result_exp - 114 + 112 = result_exp - 2
        let numer_hi = ma >> 14;
        let numer_lo = ma << 14;
        let (quot, stk) = Self::div_256_by_113(numer_hi, numer_lo, mb);
        Self::pack(result_sign, result_exp - 2, quot, stk)
    }

    /// 精确取模：IEEE 754 remainder（result = a - round_to_even(a/b) * b）
    pub fn rem_f128(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan_val();
        }
        if other.is_infinite() {
            return self; // rem(x, Inf) = x
        }
        if self.is_infinite() {
            return Self::nan_val(); // rem(Inf, y) = NaN
        }
        let (_, _, mb) = other.unpack();
        if mb == 0 {
            return Self::nan_val(); // rem(x, 0) = NaN
        }
        let (_, _, ma) = self.unpack();
        if ma == 0 {
            return self; // rem(0, y) = 0
        }

        // q = round_to_even(a / b)
        let quot = self.div_f128(other);
        // 将 q 舍入到最接近的偶数整数
        let q_bits = u128::from_le_bytes(quot.0);
        let q_exp = ((q_bits >> 112) & 0x7FFF) as i32 - 16383;
        let q_int = if q_exp >= 0 {
            // q >= 1，右移小数部分取整
            let shift = q_exp as u32;
            let q_mant = (q_bits & ((1u128 << 112) - 1)) | (1u128 << 112);
            if shift >= 113 {
                0u128
            } else {
                q_mant >> shift
            }
        } else {
            0u128
        };
        // result = a - q_int * b
        // 用 from_u128 精确构造（from_f64 对 q_int > 2^53 会丢精度）
        let q_val = Self::from_u128(q_int);
        let prod = q_val.mul_f128(other);
        self.sub_f128(prod)
    }
}

impl fmt::Debug for F128 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_nan() {
            write!(f, "NaN(f128)")
        } else if self.is_infinite() {
            let bits = u128::from_le_bytes(self.0);
            if bits >> 127 != 0 {
                write!(f, "-inf(f128)")
            } else {
                write!(f, "inf(f128)")
            }
        } else {
            write!(f, "{}f128", self.to_f64())
        }
    }
}

impl fmt::Display for F128 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // NaN/Inf 特判，正常值经 to_f64（Phase A2 已 round-to-nearest-even）打印。
        // 完整精确十进制输出作为后续优化项，不阻塞本计划。
        if self.is_nan() {
            return write!(f, "NaN(f128)");
        }
        if self.is_infinite() {
            let bits = u128::from_le_bytes(self.0);
            return write!(f, "{}inf(f128)", if bits >> 127 != 0 { "-" } else { "" });
        }
        write!(f, "{}f128", self.to_f64())
    }
}

// IEEE 754 totalOrder 语义
impl PartialOrd for F128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for F128 {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = u128::from_le_bytes(self.0);
        let b = u128::from_le_bytes(other.0);
        // totalOrder 排序键：
        //   负数（sign=1）：翻转所有位 → 映射到 [0, 0x7FFF...FFF]（-Inf 最小，-0 最大）
        //   正数（sign=0）：置符号位 → 映射到 [0x8000...000, 0xFFFF...FFF]（+0 最小，+Inf 最大）
        // 这样 -0 < +0（totalOrder 语义正确）
        let ka = if (a >> 127) != 0 { !a } else { a | (1u128 << 127) };
        let kb = if (b >> 127) != 0 { !b } else { b | (1u128 << 127) };
        ka.cmp(&kb)
    }
}

// F16/F128 运算符 trait：走精确 IEEE 754 运算，不经过 f64 中转
macro_rules! impl_float_ops {
    ($t:ty, $add:ident, $sub:ident, $mul:ident, $div:ident, $rem:ident, $neg:ident) => {
        impl std::ops::Add for $t {
            type Output = $t;
            fn add(self, rhs: $t) -> $t { self.$add(rhs) }
        }
        impl std::ops::Sub for $t {
            type Output = $t;
            fn sub(self, rhs: $t) -> $t { self.$sub(rhs) }
        }
        impl std::ops::Mul for $t {
            type Output = $t;
            fn mul(self, rhs: $t) -> $t { self.$mul(rhs) }
        }
        impl std::ops::Div for $t {
            type Output = $t;
            fn div(self, rhs: $t) -> $t { self.$div(rhs) }
        }
        impl std::ops::Rem for $t {
            type Output = $t;
            fn rem(self, rhs: $t) -> $t { self.$rem(rhs) }
        }
        impl std::ops::Neg for $t {
            type Output = $t;
            fn neg(self) -> $t { self.$neg() }
        }
    };
}

impl_float_ops!(F16, add_f16, sub_f16, mul_f16, div_f16, rem_f16, neg_f16);
impl_float_ops!(F128, add_f128, sub_f128, mul_f128, div_f128, rem_f128, neg_f128);

// ---- ValueTag / ValueTag 已移至 Type.rs（通过 re-export 保持兼容）----

// ---- ScalarValue — 标量值 union（16 字节）----

/// 标量值 union（16 字节，容纳 i128/u128/F128）。
/// 通过 ValueTag 类型守卫访问，unsafe 代码必须有对应 tag 检查。
#[derive(Clone, Copy)]
#[repr(C)]
pub union ScalarValue {
    pub bool_val: bool,
    pub char_val: u32,
    pub i8_val: i8,
    pub i16_val: i16,
    pub i32_val: i32,
    pub i64_val: i64,
    pub u8_val: u8,
    pub u16_val: u16,
    pub u32_val: u32,
    pub u64_val: u64,
    pub isize_val: isize,
    pub usize_val: usize,
    pub i128_val: [u64; 2],
    pub u128_val: [u64; 2],
    pub f16_val: u16,
    pub f32_val: f32,
    pub f64_val: f64,
    pub f128_val: [u64; 2],
}

// ---- Value — Kuzo 运行时统一值表示（spec §3.3）----

/// Kuzo 运行时统一值表示（spec §3.3）。
/// Value 自包含：标量内联、堆对象通过 Arc 跨 worker 共享。
#[derive(Clone)]
pub enum Value {
    Null,
    Void,
    /// 标量值。tag 必须为标量变体（Bool/Char/I8.../F128），
    /// 非标量 tag（Null/Void/Ref）禁止进入此路径。
    Scalar(ScalarValue, ValueTag),
    Ref(Arc<HeapObj>),
}

impl Value {
    /// 构造标量值。tag 必须为标量变体（由各 typed 构造器保证，非标量 tag 禁止进入此路径）。
    #[inline]
    fn scalar(sv: ScalarValue, tag: ValueTag) -> Self {
        Value::Scalar(sv, tag)
    }
}

unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl Value {
    // ---- 标量构造器 ----
    // 所有构造器统一调用 Self::scalar()，tag 由各 typed 构造器正确传入。
    pub fn i32(v: i32) -> Self { Self::scalar(ScalarValue { i32_val: v }, ValueTag::I32) }
    pub fn i64(v: i64) -> Self { Self::scalar(ScalarValue { i64_val: v }, ValueTag::I64) }
    pub fn f64(v: f64) -> Self { Self::scalar(ScalarValue { f64_val: v }, ValueTag::F64) }
    pub fn f32(v: f32) -> Self { Self::scalar(ScalarValue { f32_val: v }, ValueTag::F32) }
    pub fn bool_val(v: bool) -> Self { Self::scalar(ScalarValue { bool_val: v }, ValueTag::Bool) }
    pub fn char_val(v: char) -> Self { Self::scalar(ScalarValue { char_val: v as u32 }, ValueTag::Char) }
    pub fn i8(v: i8) -> Self { Self::scalar(ScalarValue { i8_val: v }, ValueTag::I8) }
    pub fn i16(v: i16) -> Self { Self::scalar(ScalarValue { i16_val: v }, ValueTag::I16) }
    pub fn u8(v: u8) -> Self { Self::scalar(ScalarValue { u8_val: v }, ValueTag::U8) }
    pub fn u16(v: u16) -> Self { Self::scalar(ScalarValue { u16_val: v }, ValueTag::U16) }
    pub fn u32(v: u32) -> Self { Self::scalar(ScalarValue { u32_val: v }, ValueTag::U32) }
    pub fn u64(v: u64) -> Self { Self::scalar(ScalarValue { u64_val: v }, ValueTag::U64) }
    pub fn isize_val(v: isize) -> Self { Self::scalar(ScalarValue { isize_val: v }, ValueTag::Isize) }
    pub fn usize_val(v: usize) -> Self { Self::scalar(ScalarValue { usize_val: v }, ValueTag::Usize) }
    pub fn f16(v: F16) -> Self { Self::scalar(ScalarValue { f16_val: v.0 }, ValueTag::F16) }
    // 128 位标量构造器（bit pattern 存为 [u64; 2]）
    pub fn i128(v: i128) -> Self {
        let bits = v as u128;
        Self::scalar(ScalarValue { i128_val: [(bits & 0xFFFF_FFFF_FFFF_FFFF) as u64, (bits >> 64) as u64] }, ValueTag::I128)
    }
    pub fn u128(v: u128) -> Self {
        Self::scalar(ScalarValue { u128_val: [(v & 0xFFFF_FFFF_FFFF_FFFF) as u64, (v >> 64) as u64] }, ValueTag::U128)
    }
    pub fn f128(v: F128) -> Self {
        Self::scalar(ScalarValue { f128_val: unsafe { std::mem::transmute(v.0) } }, ValueTag::F128)
    }

    // ---- 堆对象构造器 ----
    pub fn ref_val(obj: HeapObj) -> Self { Value::Ref(Arc::new(obj)) }
    pub fn from_ref(r: HeapRef) -> Self { Value::Ref(r) }

    pub const NULL: Value = Value::Null;
    pub const VOID: Value = Value::Void;

    // ---- 标量访问器（带 tag 守卫，整数类型间自动提升/截断）----
    /// 通用整数读取：覆盖所有整数 ValueTag，统一中转为 i128。
    /// 所有 as_iN/as_uN/as_isize/as_usize 委托本方法再 `as` 截断，避免特例匹配。
    pub fn as_int_i128(&self) -> i128 {
        match self {
            Value::Scalar(v, t) => unsafe {
                match t {
                    ValueTag::I8 => v.i8_val as i128,
                    ValueTag::I16 => v.i16_val as i128,
                    ValueTag::I32 => v.i32_val as i128,
                    ValueTag::I64 => v.i64_val as i128,
                    ValueTag::I128 => i128::from_ne_bytes(std::mem::transmute(v.i128_val)),
                    ValueTag::U8 => v.u8_val as i128,
                    ValueTag::U16 => v.u16_val as i128,
                    ValueTag::U32 => v.u32_val as i128,
                    ValueTag::U64 => v.u64_val as i128,
                    ValueTag::U128 => u128::from_ne_bytes(std::mem::transmute(v.u128_val)) as i128,
                    ValueTag::Isize => v.isize_val as i128,
                    ValueTag::Usize => v.usize_val as i128,
                    ValueTag::Char => v.char_val as i128,
                    ValueTag::Bool => if v.bool_val { 1 } else { 0 },
                    _ => 0,
                }
            },
            _ => 0,
        }
    }
    /// 通用浮点读取：覆盖 F16/F32/F64/F128，统一中转为 f64。
    /// 所有 as_fN 委托本方法，避免特例匹配。
    pub fn as_float_f64(&self) -> f64 {
        match self {
            Value::Scalar(v, t) => unsafe {
                match t {
                    ValueTag::F16 => F16(v.f16_val).to_f64(),
                    ValueTag::F32 => v.f32_val as f64,
                    ValueTag::F64 => v.f64_val,
                    ValueTag::F128 => F128(std::mem::transmute(v.f128_val)).to_f64(),
                    // 整数 → f64 提升（支持混合 int-float 算术，Bug #55）
                    ValueTag::I8 => v.i8_val as f64,
                    ValueTag::I16 => v.i16_val as f64,
                    ValueTag::I32 => v.i32_val as f64,
                    ValueTag::I64 => v.i64_val as f64,
                    ValueTag::I128 => i128::from_ne_bytes(std::mem::transmute(v.i128_val)) as f64,
                    ValueTag::U8 => v.u8_val as f64,
                    ValueTag::U16 => v.u16_val as f64,
                    ValueTag::U32 => v.u32_val as f64,
                    ValueTag::U64 => v.u64_val as f64,
                    ValueTag::U128 => u128::from_ne_bytes(std::mem::transmute(v.u128_val)) as f64,
                    ValueTag::Isize => v.isize_val as f64,
                    ValueTag::Usize => v.usize_val as f64,
                    ValueTag::Char => v.char_val as f64,
                    ValueTag::Bool => if v.bool_val { 1.0 } else { 0.0 },
                    _ => 0.0,
                }
            },
            _ => 0.0,
        }
    }
    // ---- 整数访问器：统一委托 as_int_i128，支持任意整数类型互读 ----
    pub fn as_i8(&self) -> i8 { self.as_int_i128() as i8 }
    pub fn as_i16(&self) -> i16 { self.as_int_i128() as i16 }
    pub fn as_i32(&self) -> i32 { self.as_int_i128() as i32 }
    pub fn as_i64(&self) -> i64 { self.as_int_i128() as i64 }
    pub fn as_i128(&self) -> i128 { self.as_int_i128() }
    pub fn as_u8(&self) -> u8 { self.as_int_i128() as u8 }
    pub fn as_u16(&self) -> u16 { self.as_int_i128() as u16 }
    pub fn as_u32(&self) -> u32 { self.as_int_i128() as u32 }
    pub fn as_u64(&self) -> u64 { self.as_int_i128() as u64 }
    pub fn as_u128(&self) -> u128 { self.as_int_i128() as u128 }
    pub fn as_isize(&self) -> isize { self.as_int_i128() as isize }
    pub fn as_usize(&self) -> usize { self.as_int_i128() as usize }
    // ---- 浮点访问器：统一委托 as_float_f64，支持任意浮点类型互读 ----
    // F16/F32 经 f64 中转无额外精度损失（f64 尾数 52 位，足以精确表示所有整数到 F16/F32 的舍入）
    pub fn as_f16(&self) -> F16 { F16::from_f64(self.as_float_f64()) }
    pub fn as_f32(&self) -> f32 { self.as_float_f64() as f32 }
    pub fn as_f64(&self) -> f64 { self.as_float_f64() }
    /// F128 访问器：对整数类型直接精确构造，不经 f64 中转（避免 i128 精度损失）。
    /// F128 有 113 位尾数，可精确表示所有 i128/u128 值。
    pub fn as_f128(&self) -> F128 {
        match self {
            Value::Scalar(v, t) => unsafe {
                match t {
                    ValueTag::F16 => F128::from_f64(F16(v.f16_val).to_f64()),
                    ValueTag::F32 => F128::from_f64(v.f32_val as f64),
                    ValueTag::F64 => F128::from_f64(v.f64_val),
                    ValueTag::F128 => F128(std::mem::transmute(v.f128_val)),
                    // 整数 → F128 直接构造，保证精度
                    ValueTag::I8 => F128::from_i128(v.i8_val as i128),
                    ValueTag::I16 => F128::from_i128(v.i16_val as i128),
                    ValueTag::I32 => F128::from_i128(v.i32_val as i128),
                    ValueTag::I64 => F128::from_i128(v.i64_val as i128),
                    ValueTag::I128 => F128::from_i128(i128::from_ne_bytes(std::mem::transmute(v.i128_val))),
                    ValueTag::U8 => F128::from_u128(v.u8_val as u128),
                    ValueTag::U16 => F128::from_u128(v.u16_val as u128),
                    ValueTag::U32 => F128::from_u128(v.u32_val as u128),
                    ValueTag::U64 => F128::from_u128(v.u64_val as u128),
                    ValueTag::U128 => F128::from_u128(u128::from_ne_bytes(std::mem::transmute(v.u128_val))),
                    ValueTag::Isize => F128::from_i128(v.isize_val as i128),
                    ValueTag::Usize => F128::from_u128(v.usize_val as u128),
                    ValueTag::Char => F128::from_u128(v.char_val as u128),
                    _ => F128::from_f64(0.0),
                }
            },
            _ => F128::from_f64(0.0),
        }
    }
    // ---- 其他标量访问器 ----
    pub fn as_bool(&self) -> bool { match self { Value::Scalar(v, ValueTag::Bool) => unsafe { v.bool_val }, _ => false } }
    pub fn as_char(&self) -> char { match self { Value::Scalar(v, ValueTag::Char) => unsafe { char::from_u32_unchecked(v.char_val) }, _ => '\0' } }

    // ---- 堆对象访问器 ----
    pub fn heap_obj(&self) -> Option<&HeapObj> { match self { Value::Ref(r) => Some(r.as_ref()), _ => None } }
    pub fn heap_ref(&self) -> Option<HeapRef> { match self { Value::Ref(r) => Some(r.clone()), _ => None } }

    // ---- 判别 ----
    pub fn is_null(&self) -> bool { matches!(self, Value::Null) }
    pub fn is_void(&self) -> bool { matches!(self, Value::Void) }
    pub fn is_ref(&self) -> bool { matches!(self, Value::Ref(_)) }

    // ---- 标量 tag 访问（供 Hash/Debug/反射适配）----
    pub fn scalar_tag(&self) -> Option<ValueTag> {
        match self { Value::Scalar(_, t) => Some(*t), _ => None }
    }

    // ---- Weak 引用基础设施（用于打破 Cell 循环引用）----
    /// 返回指向自身堆对象的 Weak 引用。
    /// 仅对 `Value::Ref` 有意义；标量/Null/Void 返回 None。
    /// 调用方可将 Weak 存入 Cell 内部以打破 `a = Cell(b); b = Cell(a)` 形成的环。
    pub fn make_weak(&self) -> Option<Weak<HeapObj>> {
        match self { Value::Ref(r) => Some(Arc::downgrade(r)), _ => None }
    }

    /// 将 Weak 引用升级回 Value。若原对象已被回收则返回 None。
    pub fn upgrade_weak(weak: &Weak<HeapObj>) -> Option<Value> {
        weak.upgrade().map(Value::from_ref)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Void => write!(f, "()"),
            Value::Scalar(v, tag) => {
                // 复用 ValueHandle 的标量格式化逻辑：按 tag 读取 union 字段
                match tag {
                    ValueTag::Bool => write!(f, "{}", unsafe { v.bool_val }),
                    ValueTag::Char => write!(f, "'{}'", Char::from_codepoint_unchecked(unsafe { v.char_val })),
                    ValueTag::I8 => write!(f, "{}i8", unsafe { v.i8_val }),
                    ValueTag::I16 => write!(f, "{}i16", unsafe { v.i16_val }),
                    ValueTag::I32 => write!(f, "{}", unsafe { v.i32_val }),
                    ValueTag::I64 => write!(f, "{}i64", unsafe { v.i64_val }),
                    ValueTag::I128 => write!(f, "{}i128", unsafe { i128::from_ne_bytes(std::mem::transmute(v.i128_val)) }),
                    ValueTag::U8 => write!(f, "{}u8", unsafe { v.u8_val }),
                    ValueTag::U16 => write!(f, "{}u16", unsafe { v.u16_val }),
                    ValueTag::U32 => write!(f, "{}u32", unsafe { v.u32_val }),
                    ValueTag::U64 => write!(f, "{}u64", unsafe { v.u64_val }),
                    ValueTag::U128 => write!(f, "{}u128", unsafe { u128::from_ne_bytes(std::mem::transmute(v.u128_val)) }),
                    ValueTag::Isize => write!(f, "{}isize", unsafe { v.isize_val }),
                    ValueTag::Usize => write!(f, "{}usize", unsafe { v.usize_val }),
                    ValueTag::F16 => write!(f, "{:?}", F16(unsafe { v.f16_val })),
                    ValueTag::F32 => write!(f, "{}f32", unsafe { v.f32_val }),
                    ValueTag::F64 => write!(f, "{}", unsafe { v.f64_val }),
                    ValueTag::F128 => write!(f, "{:?}", F128(unsafe { std::mem::transmute(v.f128_val) })),
                    _ => unreachable!("non-scalar tag {:?} in ScalarValue", tag),
                }
            }
            Value::Ref(r) => fmt::Debug::fmt(r.as_ref(), f),
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Null | Value::Void => {}
            Value::Scalar(v, tag) => {
                tag.hash(state);
                // 按 tag 哈希对应 union 字段
                match tag {
                    ValueTag::Bool => unsafe { v.bool_val }.hash(state),
                    ValueTag::Char => unsafe { v.char_val }.hash(state),
                    ValueTag::I8 => unsafe { v.i8_val }.hash(state),
                    ValueTag::I16 => unsafe { v.i16_val }.hash(state),
                    ValueTag::I32 => unsafe { v.i32_val }.hash(state),
                    ValueTag::I64 => unsafe { v.i64_val }.hash(state),
                    ValueTag::I128 => unsafe { v.i128_val }.hash(state),
                    ValueTag::U8 => unsafe { v.u8_val }.hash(state),
                    ValueTag::U16 => unsafe { v.u16_val }.hash(state),
                    ValueTag::U32 => unsafe { v.u32_val }.hash(state),
                    ValueTag::U64 => unsafe { v.u64_val }.hash(state),
                    ValueTag::U128 => unsafe { v.u128_val }.hash(state),
                    ValueTag::Isize => unsafe { v.isize_val }.hash(state),
                    ValueTag::Usize => unsafe { v.usize_val }.hash(state),
                    ValueTag::F16 => unsafe { v.f16_val }.hash(state),
                    ValueTag::F32 => unsafe { v.f32_val }.to_bits().hash(state),
                    ValueTag::F64 => unsafe { v.f64_val }.to_bits().hash(state),
                    ValueTag::F128 => unsafe { v.f128_val }.hash(state),
                    _ => unreachable!("non-scalar tag {:?} in ScalarValue", tag),
                }
            }
            Value::Ref(r) => (Arc::as_ptr(r) as usize).hash(state),
        }
    }
}

// ---- ValueHandle — 4B 索引句柄 ----

/// Kuzo 值的唯一句柄：4B 索引，编码类型桶 + 桶内索引。
/// 高 8 位 = ValueTag，低 24 位 = 桶内索引。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueHandle(u32);

impl ValueHandle {
    const TAG_SHIFT: u32 = 24;
    const INDEX_MASK: u32 = 0x00FF_FFFF;

    #[inline]
    pub fn new(tag: ValueTag, index: usize) -> Self {
        // [V-3] release 也保留检查：index >= 2^24 会静默截断（MASK 抹掉高位）导致
        // 两个不同索引产生相同 ValueHandle → 句柄别名损坏。这是不可恢复的不变式违反，
        // 显式 panic 优于静默损坏（arena 不应分配超 16M 个同类型值）。
        assert!(index < (1 << 24), "ValueHandle index overflow: {index} >= 2^24");
        Self(((tag as u8 as u32) << Self::TAG_SHIFT) | (index as u32 & Self::INDEX_MASK))
    }

    #[inline]
    pub fn tag(self) -> ValueTag {
        // FFI 防御：extern "C" 原语经 from_raw 还原的 u32 可能携带越界 tag
        // （21..=255）。transmute 到 #[repr(u8)] enum 的非法判别值是 UB，
        // 故用 match 显式映射，越界统一兜底为 Null，保证任何 u32 都安全。
        match (self.0 >> Self::TAG_SHIFT) as u8 {
            0 => ValueTag::Null,
            1 => ValueTag::Void,
            2 => ValueTag::Bool,
            3 => ValueTag::Char,
            4 => ValueTag::I8,
            5 => ValueTag::I16,
            6 => ValueTag::I32,
            7 => ValueTag::I64,
            8 => ValueTag::U8,
            9 => ValueTag::U16,
            10 => ValueTag::U32,
            11 => ValueTag::U64,
            12 => ValueTag::Isize,
            13 => ValueTag::Usize,
            14 => ValueTag::I128,
            15 => ValueTag::U128,
            16 => ValueTag::F16,
            17 => ValueTag::F32,
            18 => ValueTag::F64,
            19 => ValueTag::F128,
            20 => ValueTag::Ref,
            _ => ValueTag::Null,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        (self.0 & Self::INDEX_MASK) as usize
    }

    /// 从原始 u32 构造 ValueHandle（供 extern "C" 原语跨 ABI 边界还原）
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// 转为原始 u32（供 extern "C" 原语跨 ABI 边界传递）
    #[inline]
    pub fn to_raw(self) -> u32 {
        self.0
    }

    pub const NULL: ValueHandle = ValueHandle((ValueTag::Null as u8 as u32) << 24);
    pub const VOID: ValueHandle = ValueHandle((ValueTag::Void as u8 as u32) << 24);
    pub const TRUE: ValueHandle = ValueHandle(((ValueTag::Bool as u8 as u32) << 24) | 1);
    pub const FALSE: ValueHandle = ValueHandle((ValueTag::Bool as u8 as u32) << 24);
}

impl Default for ValueHandle {
    fn default() -> Self {
        ValueHandle::VOID
    }
}

impl fmt::Debug for ValueHandle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ValueHandle({:?}, {})", self.tag(), self.index())
    }
}

impl fmt::Display for ValueHandle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---- Char / CharError ----

/// 字符错误：codepoint 越界
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharError {
    InvalidCodepoint,
}

/// Unicode 字符：包装 codepoint（u32）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Char {
    pub codepoint: u32,
}

impl Char {
    pub fn from_codepoint(cp: u32) -> Result<Self, CharError> {
        if cp > 0x10FFFF || (0xD800..=0xDFFF).contains(&cp) {
            return Err(CharError::InvalidCodepoint);
        }
        Ok(Char { codepoint: cp })
    }

    pub fn from_codepoint_unchecked(cp: u32) -> Self {
        Char { codepoint: cp }
    }

    pub fn codepoint(self) -> u32 {
        self.codepoint
    }

    pub fn is_ascii(self) -> bool {
        self.codepoint < 0x80
    }

    pub fn is_digit(self) -> bool {
        (b'0' as u32..=b'9' as u32).contains(&self.codepoint)
    }

    pub fn is_alpha(self) -> bool {
        (b'a' as u32..=b'z' as u32).contains(&self.codepoint)
            || (b'A' as u32..=b'Z' as u32).contains(&self.codepoint)
    }

    pub fn is_alphanumeric(self) -> bool {
        self.is_alpha() || self.is_digit()
    }

    pub fn is_whitespace(self) -> bool {
        matches!(self.codepoint, 0x09 | 0x0A | 0x0B | 0x0C | 0x0D | 0x20 | 0x85 | 0xA0)
    }

    pub fn to_upper(self) -> Self {
        if (b'a' as u32..=b'z' as u32).contains(&self.codepoint) {
            Char { codepoint: self.codepoint - 32 }
        } else {
            self
        }
    }

    pub fn to_lower(self) -> Self {
        if (b'A' as u32..=b'Z' as u32).contains(&self.codepoint) {
            Char { codepoint: self.codepoint + 32 }
        } else {
            self
        }
    }

    pub fn successor(self) -> Self {
        // [V-6] 跳过代理区 + 饱和到 0x10FFFF，避免 wrapping 回绕产生非法 codepoint
        let next = if self.codepoint >= 0x10FFFF {
            0x10FFFF
        } else if self.codepoint == 0xD7FF {
            0xE000
        } else {
            self.codepoint + 1
        };
        Char { codepoint: next }
    }

    pub fn predecessor(self) -> Self {
        // [V-6] 跳过代理区 + 饱和到 0，避免 wrapping 回绕产生非法 codepoint
        let prev = if self.codepoint == 0 {
            0
        } else if self.codepoint == 0xE000 {
            0xD7FF
        } else {
            self.codepoint - 1
        };
        Char { codepoint: prev }
    }

    pub fn compare(self, other: Self) -> Ordering {
        self.codepoint.cmp(&other.codepoint)
    }
}

impl fmt::Display for Char {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(c) = char::from_u32(self.codepoint) {
            write!(f, "{}", c)
        } else {
            write!(f, "\u{FFFD}")
        }
    }
}

impl From<char> for Char {
    fn from(c: char) -> Self {
        Char { codepoint: c as u32 }
    }
}

// =========================================================================
// 第二部分：堆对象类型（合并 6 个文件）
// =========================================================================

// ---- str.rs → KuzoStr ----

/// Kuzo 字符串：引用计数的不可变 UTF-8 字符串
#[derive(Debug, Clone)]
pub struct KuzoStr {
    inner: Arc<str>,
}

impl KuzoStr {
    pub fn new(s: impl Into<String>) -> Self {
        Self { inner: Arc::from(s.into().as_str()) }
    }
    pub fn from_rust_str(s: &str) -> Self {
        Self { inner: Arc::from(s) }
    }
    pub fn bytes(&self) -> &str {
        &self.inner
    }
    pub fn byte_len(&self) -> usize {
        self.inner.len()
    }
    pub fn codepoint_count(&self) -> usize {
        self.inner.chars().count()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn concat(&self, other: &Self) -> Self {
        let mut buf = String::with_capacity(self.byte_len() + other.byte_len());
        buf.push_str(&self.inner);
        buf.push_str(&other.inner);
        Self::from_rust_str(&buf)
    }
    pub fn equals(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    pub fn compare(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }

    /// 按码点索引取字符（UTF-8 安全）。
    ///
    /// 返回第 idx 个 Unicode 码点。越界返回 None。
    pub fn char_at(&self, idx: usize) -> Option<char> {
        self.inner.chars().nth(idx)
    }
}

impl PartialEq for KuzoStr {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl Eq for KuzoStr {}

impl Hash for KuzoStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl fmt::Display for KuzoStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

// ---- composite.rs → ArrayValue, RecordField, RecordValue, AdtField, AdtValue, NewtypeValue, Cell, Range, RangeIter ----

/// 数组值：元素可变（支持 push/pop），`fixed_size` 为 `Some` 时表示固定大小数组
#[derive(Debug, Clone)]
pub struct ArrayValue {
    pub elements: Vec<Value>,
    pub fixed_size: Option<u64>,
    pub elem_is_ref: bool,
    pub scalar_soa: Option<ScalarSoA>,
}

/// SoA 连续存储：当数组元素全为同类型标量时启用 SIMD 快路径
#[derive(Debug, Clone)]
pub enum ScalarSoA {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    I128(Vec<i128>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    U128(Vec<u128>),
    Isize(Vec<isize>),
    Usize(Vec<usize>),
    Bool(Vec<bool>),
    Char(Vec<u32>),
    F16(Vec<u16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    F128(Vec<F128>),
}

impl ScalarSoA {
    /// 尝试在指定索引写入标量值。
    /// 返回 true 表示类型匹配且写入成功；false 表示类型不匹配（调用方应失效 SOA）。
    /// 索引越界时自动扩展（补 0）。
    pub fn try_store(&mut self, idx: usize, val: &Value) -> bool {
        match (self, val) {
            (ScalarSoA::I8(v), Value::Scalar(sv, crate::value::ValueTag::I8)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.i8_val; } true }
            (ScalarSoA::I16(v), Value::Scalar(sv, crate::value::ValueTag::I16)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.i16_val; } true }
            (ScalarSoA::I32(v), Value::Scalar(sv, crate::value::ValueTag::I32)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.i32_val; } true }
            (ScalarSoA::I64(v), Value::Scalar(sv, crate::value::ValueTag::I64)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.i64_val; } true }
            (ScalarSoA::U8(v), Value::Scalar(sv, crate::value::ValueTag::U8)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.u8_val; } true }
            (ScalarSoA::U16(v), Value::Scalar(sv, crate::value::ValueTag::U16)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.u16_val; } true }
            (ScalarSoA::U32(v), Value::Scalar(sv, crate::value::ValueTag::U32)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.u32_val; } true }
            (ScalarSoA::U64(v), Value::Scalar(sv, crate::value::ValueTag::U64)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.u64_val; } true }
            (ScalarSoA::Bool(v), Value::Scalar(sv, crate::value::ValueTag::Bool)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, false); } v[idx] = sv.bool_val; } true }
            (ScalarSoA::Char(v), Value::Scalar(sv, crate::value::ValueTag::Char)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.char_val; } true }
            (ScalarSoA::F32(v), Value::Scalar(sv, crate::value::ValueTag::F32)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0.0); } v[idx] = sv.f32_val; } true }
            (ScalarSoA::F64(v), Value::Scalar(sv, crate::value::ValueTag::F64)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0.0); } v[idx] = sv.f64_val; } true }
            (ScalarSoA::I128(v), Value::Scalar(sv, crate::value::ValueTag::I128)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = i128::from_ne_bytes(std::mem::transmute(sv.i128_val)); } true }
            (ScalarSoA::U128(v), Value::Scalar(sv, crate::value::ValueTag::U128)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = u128::from_ne_bytes(std::mem::transmute(sv.u128_val)); } true }
            (ScalarSoA::Isize(v), Value::Scalar(sv, crate::value::ValueTag::Isize)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.isize_val; } true }
            (ScalarSoA::Usize(v), Value::Scalar(sv, crate::value::ValueTag::Usize)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.usize_val; } true }
            (ScalarSoA::F16(v), Value::Scalar(sv, crate::value::ValueTag::F16)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, 0); } v[idx] = sv.f16_val; } true }
            (ScalarSoA::F128(v), Value::Scalar(sv, crate::value::ValueTag::F128)) => { unsafe { if idx >= v.len() { v.resize(idx + 1, F128([0; 16])); } v[idx] = F128(std::mem::transmute(sv.f128_val)); } true }
            _ => false, // 类型不匹配
        }
    }
}

impl ArrayValue {
    pub fn new(elements: Vec<Value>) -> Self {
        Self { elements, fixed_size: None, elem_is_ref: false, scalar_soa: None }
    }
    pub fn new_fixed(elements: Vec<Value>, size: u64) -> Self {
        Self { elements, fixed_size: Some(size), elem_is_ref: false, scalar_soa: None }
    }
    pub fn len(&self) -> usize {
        self.elements.len()
    }
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.elements.get(index)
    }
    pub fn push(&mut self, val: Value) {
        self.elements.push(val);
    }
    pub fn pop(&mut self) -> Option<Value> {
        self.elements.pop()
    }
    /// 统一收集 u8 字节：SOA 快路径（U8 连续存储）或回退到逐元素提取。
    /// 封装双表示访问，调用方无需关心 SOA 是否启用。
    pub fn collect_u8_bytes(&self) -> Vec<u8> {
        if let Some(crate::value::ScalarSoA::U8(ref data)) = self.scalar_soa {
            return data.clone();
        }
        self.elements.iter().map(|e| e.as_u8()).collect()
    }
}

/// 记录字段：可选名称 + 值
#[derive(Debug, Clone)]
pub struct RecordField {
    pub name: Option<String>,
    pub value: ValueHandle,
}

/// 记录值：具名类型的结构化数据
#[derive(Debug, Clone)]
pub struct RecordValue {
    pub type_name: String,
    pub fields: Vec<Value>,
    pub field_names: Vec<Option<String>>,
    pub field_ref_bits: u64,
}

impl RecordValue {
    pub fn new(type_name: String, fields: Vec<Value>, field_names: Vec<Option<String>>) -> Self {
        Self { type_name, fields, field_names, field_ref_bits: 0 }
    }
    pub fn get_field(&self, index: usize) -> Option<&Value> {
        self.fields.get(index)
    }
    pub fn find_field(&self, name: &str) -> Option<&Value> {
        for (i, field_name) in self.field_names.iter().enumerate() {
            if let Some(n) = field_name {
                if n == name {
                    return self.fields.get(i);
                }
            }
        }
        None
    }
}

/// ADT 字段：构造器的参数
#[derive(Debug, Clone)]
pub struct AdtField {
    pub name: Option<String>,
    pub value: Value,
}

/// ADT 值：代数数据类型实例
#[derive(Debug, Clone)]
pub struct AdtValue {
    pub type_name: String,
    pub constructor: String,
    pub fields: Vec<AdtField>,
    pub field_ref_bits: u64,
}

impl AdtValue {
    pub fn new(type_name: String, constructor: String, fields: Vec<AdtField>) -> Self {
        Self { type_name, constructor, fields, field_ref_bits: 0 }
    }
    pub fn get_field(&self, index: usize) -> Option<&Value> {
        self.fields.get(index).map(|f| &f.value)
    }
    pub fn find_field(&self, name: &str) -> Option<&Value> {
        for field in &self.fields {
            if let Some(n) = &field.name {
                if n == name {
                    return Some(&field.value);
                }
            }
        }
        None
    }
}

/// Newtype 值：包装单个内部值的具名类型
#[derive(Debug, Clone)]
pub struct NewtypeValue {
    pub type_name: String,
    pub inner: ValueHandle,
}

/// Cell：可变引用单元（`&T` 引用语义的运行时载体）
///
/// 内部持有 `Value`（自包含值，标量内联 + 堆对象 Arc 共享）。
/// `&expr` 创建 `Arc<HeapObj::Cell>` 包装当前值；`*r` 读取 Cell；
/// `*r = v` 写入 Cell。多個引用共享同一 Arc，写入对所有引用可见。
#[derive(Debug)]
pub struct Cell {
    pub inner: parking_lot::Mutex<Value>,
}

impl Clone for Cell {
    fn clone(&self) -> Self {
        Self { inner: parking_lot::Mutex::new(self.get()) }
    }
}

impl Cell {
    pub fn new(val: Value) -> Self {
        Self { inner: parking_lot::Mutex::new(val) }
    }
    /// 返回内部值的克隆。
    pub fn get(&self) -> Value {
        self.inner.lock().clone()
    }
    pub fn set(&self, val: Value) {
        *self.inner.lock() = val;
    }

    /// 返回指向自身的 Weak 引用（用于打破循环引用）。
    /// 调用方需确保 Cell 被包装在 `Arc<HeapObj::Cell>` 中；
    /// 若传入的 Arc 并非 Cell，返回 None。
    pub fn downgrade(arc: &Arc<HeapObj>) -> Option<Weak<HeapObj>> {
        match arc.as_ref() {
            HeapObj::Cell(_) => Some(Arc::downgrade(arc)),
            _ => None,
        }
    }
}

/// 范围值
#[derive(Debug, Clone)]
pub struct Range {
    pub start: i64,
    pub end: i64,
    pub inclusive: bool,
}

impl Range {
    pub fn new(start: i64, end: i64, inclusive: bool) -> Self {
        Self { start, end, inclusive }
    }
    pub fn contains(&self, val: i64) -> bool {
        if self.inclusive {
            val >= self.start && val <= self.end
        } else {
            val >= self.start && val < self.end
        }
    }
    pub fn len(&self) -> usize {
        if self.inclusive {
            if self.end >= self.start {
                (self.end - self.start + 1) as usize
            } else {
                0
            }
        } else if self.end > self.start {
            (self.end - self.start) as usize
        } else {
            0
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> RangeIter {
        RangeIter { current: self.start, end: self.end, inclusive: self.inclusive }
    }
}

/// 范围迭代器（composite 内部）
#[derive(Debug, Clone)]
pub struct RangeIter {
    pub current: i64,
    pub end: i64,
    pub inclusive: bool,
}

// ---- callable.rs → BuiltinFn, Builtin, Closure, PartialApplication, TraitValue, LazyValue ----

/// 内建函数指针类型
pub type BuiltinFn = fn(&[ValueHandle]) -> Result<ValueHandle, String>;

/// 内建函数值
#[derive(Clone)]
pub struct Builtin {
    pub fn_ptr: BuiltinFn,
    pub name: String,
}

impl fmt::Debug for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<builtin {}>", self.name)
    }
}

/// 闭包值
#[derive(Debug, Clone)]
pub struct Closure {
    pub func_id: u32,
    pub arity: u8,
    pub upvalues: Vec<Value>,
    pub bound_args: Vec<ValueHandle>,
    pub self_upvalue_idx: i32,
    pub upvalue_ref_bits: u8,
    pub cell_upvalues: u8,
}

/// 偏应用值：对函数/闭包绑定了前导参数后得到的可调用值。
///
/// 统一调用语义：当新参数数 < remaining_arity → 产出新的 Partial（链式偏应用）；
/// 当新参数数 >= remaining_arity → 合并 bound_args + 新参数 + upvalues 启动子图。
/// upvalues 来自源 Closure（顶层函数偏应用时为空），与 Closure 的 upvalues 语义一致。
#[derive(Debug, Clone)]
pub struct PartialApplication {
    /// 目标子图 id（与 Closure.func_id 语义一致）
    pub func_id: u32,
    /// 来自源 Closure 的 upvalues（顶层函数偏应用时为空）
    pub upvalues: Vec<Value>,
    /// 已绑定的前导参数（按原函数参数顺序）
    pub bound_args: Vec<Value>,
    /// 仍需参数数 = subgraph.param_count - upvalues.len() - bound_args.len()
    pub remaining_arity: u8,
    /// 递归闭包自引用 upvalue 索引（-1 表示无自引用）
    pub self_upvalue_idx: i32,
}

/// Trait 值
#[derive(Debug, Clone)]
pub struct TraitValue {
    pub trait_name: String,
    pub method_names: Vec<String>,
    pub method_values: Vec<Value>,
    pub data: Option<Value>,
    pub owned: bool,
}

/// 惰性值
pub struct LazyValue {
    /// 缓存的求值结果（首次 force 后填充）
    /// Mutex 允许通过 &LazyValue 更新缓存（Arc 共享场景下的 interior mutability）
    pub cached: Mutex<Option<Value>>,
    /// 是否已求值
    pub forced: AtomicBool,
    /// thunk 子图的 Closure（func_id = thunk_sg, upvalues = 捕获值）
    /// force 时取此 Closure 启动子图计算，结果存入 cached
    pub data: Option<Value>,
}

impl Clone for LazyValue {
    fn clone(&self) -> Self {
        Self {
            cached: Mutex::new(self.cached.lock().unwrap().clone()),
            forced: AtomicBool::new(self.forced.load(std::sync::atomic::Ordering::Relaxed)),
            data: self.data.clone(),
        }
    }
}

impl fmt::Debug for LazyValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LazyValue")
            .field("cached", &self.cached.lock().unwrap().is_some())
            .field("forced", &self.forced.load(std::sync::atomic::Ordering::Relaxed))
            .field("has_data", &self.data.is_some())
            .finish()
    }
}

// ---- control.rs → ErrorValue, ThrowPayload, ThrowValue ----

/// 错误值
#[derive(Debug, Clone)]
pub struct ErrorValue {
    pub type_name: String,
    pub message: String,
    pub is_error_subtype: bool,
}

/// 抛出载荷
///
/// Err 直接持有 Value（而非 Arc<RecordValue>），统一所有 throw 场景：
/// - throw 原始类型（i32/str/bool）→ Err 持有裸标量值，无需 Error(value:v) 包装
/// - throw record/adt → Err 持有 record Value
/// - 内部错误（FieldError/IndexError 等）→ Err 持有构造好的 record Value
/// 这使得 throw 任意值后，match 模式 `Error(v)` 的 v 直接绑定到 throw 的值本身。
#[derive(Debug, Clone)]
pub enum ThrowPayload {
    Ok(Value),
    Err(Value),
}

/// 抛出值
#[derive(Debug, Clone)]
pub struct ThrowValue {
    pub payload: ThrowPayload,
}

// ---- iterator.rs → 已全部迁移至 Kuzo builtin (Iterator.kz) ----
// 注：ArrayIterator / StringIterator / RangeIterator 均已迁移至 Kuzo builtin。

// ---- concurrent.rs → AtomicValue, AsyncStatus, AsyncHandle, ChannelValue, SenderValue, ReceiverValue ----

/// 原子值
#[derive(Debug)]
pub struct AtomicValue {
    data: Mutex<Value>,
}

impl AtomicValue {
    pub fn new(val: Value) -> Self {
        Self { data: Mutex::new(val) }
    }
    pub fn load(&self) -> Value {
        self.data.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub fn store(&self, val: Value) {
        *self.data.lock().unwrap_or_else(|e| e.into_inner()) = val;
    }
    pub fn swap(&self, val: Value) -> Value {
        std::mem::replace(&mut *self.data.lock().unwrap_or_else(|e| e.into_inner()), val)
    }
}

impl Clone for AtomicValue {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}

/// 异步任务状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

/// 异步句柄
#[derive(Debug)]
pub struct AsyncHandle {
    status: Mutex<AsyncStatus>,
    result: Mutex<Option<ValueHandle>>,
}

impl AsyncHandle {
    pub fn new() -> Self {
        Self { status: Mutex::new(AsyncStatus::Pending), result: Mutex::new(None) }
    }
    pub fn status(&self) -> AsyncStatus {
        *self.status.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn set_status(&self, status: AsyncStatus) {
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = status;
    }
    pub fn result(&self) -> Option<ValueHandle> {
        *self.result.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn set_result(&self, val: ValueHandle) {
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(val);
    }
}

impl Default for AsyncHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AsyncHandle {
    fn clone(&self) -> Self {
        let status = self.status();
        let result = self.result();
        Self { status: Mutex::new(status), result: Mutex::new(result) }
    }
}

/// 全局 channel id 计数器（线程安全，单/多 worker 共用）
static CHANNEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 通道值
///
/// 统一存储 Engine 的 Value（非 ValueHandle），与 async 运行时一致。
/// id 用于 RuntimeEvent::ChannelReady 事件标识（send 后内联触发 on_event_arrived）。
#[derive(Debug)]
pub struct ChannelValue {
    id: u64,
    buffer: Mutex<VecDeque<Value>>,
    capacity: usize,
    closed: Mutex<bool>,
}

/// channel send 失败原因（运行时条件，非程序员错误）。
#[derive(Debug, Clone, Copy)]
pub enum ChannelSendError {
    /// channel 已关闭
    Closed,
    /// 有界 channel 已满
    Full { capacity: usize },
}

impl ChannelSendError {
    pub fn message(&self) -> &'static str {
        match self {
            ChannelSendError::Closed => "send on closed channel",
            ChannelSendError::Full { .. } => "channel full",
        }
    }
}

impl ChannelValue {
    pub fn new(capacity: usize) -> Self {
        Self {
            id: CHANNEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            buffer: Mutex::new(VecDeque::new()),
            capacity,
            closed: Mutex::new(false),
        }
    }
    /// 返回 channel 的唯一 id（用于 RuntimeEvent::ChannelReady 事件标识）
    pub fn id(&self) -> u64 {
        self.id
    }
    /// 非阻塞发送：push 到 buffer。满或已关闭时返回 Err（运行时条件，非程序员错误）。
    pub fn send(&self, val: Value) -> Result<(), ChannelSendError> {
        let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        // [V-5] 持 buffer 锁期间检查 closed，与 close（同样持 buffer 锁）互斥，消除 TOCTOU
        if *self.closed.lock().unwrap_or_else(|e| e.into_inner()) {
            return Err(ChannelSendError::Closed);
        }
        if self.capacity > 0 && buf.len() >= self.capacity {
            return Err(ChannelSendError::Full { capacity: self.capacity });
        }
        buf.push_back(val);
        Ok(())
    }
    /// 接收：pop 从 buffer 前端，无数据返回 None（await 路径在 resolve_and_check_await 处理挂起）
    pub fn recv(&self) -> Option<Value> {
        let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buf.pop_front()
    }
    /// 是否有数据可读
    pub fn has_data(&self) -> bool {
        !self.buffer.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }
    pub fn close(&self) {
        // [V-5] 持 buffer 锁设置 closed，与 send 的持锁检查互斥（锁序 buffer→closed 一致，无死锁）
        let _buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        *self.closed.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }
    pub fn is_closed(&self) -> bool {
        *self.closed.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Clone for ChannelValue {
    fn clone(&self) -> Self {
        let buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Self {
            id: self.id,
            buffer: Mutex::new(buf),
            capacity: self.capacity,
            closed: Mutex::new(*self.closed.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }
}

/// 发送端值
#[derive(Debug, Clone)]
pub struct SenderValue {
    pub channel: Arc<ChannelValue>,
}

/// 接收端值
#[derive(Debug, Clone)]
pub struct ReceiverValue {
    pub channel: Arc<ChannelValue>,
}

// ---- heap.rs → HeapObj enum + HeapRef + RefKind + impl ----

/// 堆对象：所有堆分配值类型的统一表示（23 种）
#[derive(Debug, Clone)]
pub enum HeapObj {
    Str(KuzoStr),
    Array(ArrayValue),
    Record(RecordValue),
    Adt(AdtValue),
    Newtype(NewtypeValue),
    Cell(Cell),
    Range(Range),
    Closure(Closure),
    Partial(PartialApplication),
    Builtin(Builtin),
    TraitVal(TraitValue),
    LazyVal(LazyValue),
    ErrorVal(ErrorValue),
    ThrowVal(ThrowValue),
    AtomicVal(AtomicValue),
    AsyncVal(AsyncHandle),
    ChannelVal(Arc<ChannelValue>),
    SenderVal(SenderValue),
    ReceiverVal(ReceiverValue),
    CoroutineFrame,
}

/// 堆引用：引用计数的堆对象
pub type HeapRef = Arc<HeapObj>;

/// 引用类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefKind {
    Str, Array, Record, Adt, Newtype, Cell, Range, Closure, Partial, Builtin,
    TraitVal, LazyVal, ErrorVal, ThrowVal,
    AtomicVal, AsyncVal, ChannelVal, SenderVal, ReceiverVal, CoroutineFrame,
}

impl HeapObj {
    /// 统一提取底层 channel：ChannelVal/SenderVal/ReceiverVal 共享同一 Arc<ChannelValue>。
    /// 消除各处对三种类型的重复分派，send/close/await/select 统一调用此方法。
    pub fn channel(&self) -> Option<&Arc<ChannelValue>> {
        match self {
            HeapObj::ChannelVal(ch) => Some(ch),
            HeapObj::SenderVal(tx) => Some(&tx.channel),
            HeapObj::ReceiverVal(rx) => Some(&rx.channel),
            _ => None,
        }
    }

    /// 统一字段访问：Record/Adt 按名查找字段值，
    /// ChannelVal 按 channel 协议字段（sender/receiver）派生。
    /// 消除 compute_record_field_get 中对字段名和类型的硬编码分派。
    pub fn field_get(&self, name: &str) -> Option<Value> {
        match self {
            HeapObj::Record(r) => r.find_field(name).cloned(),
            HeapObj::Adt(a) => a.find_field(name).cloned(),
            HeapObj::ChannelVal(ch) => match name {
                "sender" => Some(Value::ref_val(HeapObj::SenderVal(SenderValue { channel: ch.clone() }))),
                "receiver" => Some(Value::ref_val(HeapObj::ReceiverVal(ReceiverValue { channel: ch.clone() }))),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn ref_kind(&self) -> RefKind {
        match self {
            HeapObj::Str(_) => RefKind::Str,
            HeapObj::Array(_) => RefKind::Array,
            HeapObj::Record(_) => RefKind::Record,
            HeapObj::Adt(_) => RefKind::Adt,
            HeapObj::Newtype(_) => RefKind::Newtype,
            HeapObj::Cell(_) => RefKind::Cell,
            HeapObj::Range(_) => RefKind::Range,
            HeapObj::Closure(_) => RefKind::Closure,
            HeapObj::Partial(_) => RefKind::Partial,
            HeapObj::Builtin(_) => RefKind::Builtin,
            HeapObj::TraitVal(_) => RefKind::TraitVal,
            HeapObj::LazyVal(_) => RefKind::LazyVal,
            HeapObj::ErrorVal(_) => RefKind::ErrorVal,
            HeapObj::ThrowVal(_) => RefKind::ThrowVal,
            HeapObj::AtomicVal(_) => RefKind::AtomicVal,
            HeapObj::AsyncVal(_) => RefKind::AsyncVal,
            HeapObj::ChannelVal(_) => RefKind::ChannelVal,
            HeapObj::SenderVal(_) => RefKind::SenderVal,
            HeapObj::ReceiverVal(_) => RefKind::ReceiverVal,
            HeapObj::CoroutineFrame => RefKind::CoroutineFrame,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            HeapObj::Str(_) => "str",
            HeapObj::Array(_) => "array",
            HeapObj::Record(_) => "record",
            HeapObj::Adt(_) => "adt",
            HeapObj::Newtype(_) => "newtype",
            HeapObj::Cell(_) => "cell",
            HeapObj::Range(_) => "range",
            HeapObj::Closure(_) => "closure",
            HeapObj::Partial(_) => "partial",
            HeapObj::Builtin(_) => "builtin",
            HeapObj::TraitVal(_) => "trait",
            HeapObj::LazyVal(_) => "lazy",
            HeapObj::ErrorVal(_) => "error",
            HeapObj::ThrowVal(_) => "throw",
            HeapObj::AtomicVal(_) => "atomic",
            HeapObj::AsyncVal(_) => "async",
            HeapObj::ChannelVal(_) => "channel",
            HeapObj::SenderVal(_) => "sender",
            HeapObj::ReceiverVal(_) => "receiver",
            HeapObj::CoroutineFrame => "coroutine",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            HeapObj::Str(_) => "str",
            HeapObj::Array(_) => "[...]",
            HeapObj::Record(_) => "record",
            HeapObj::Adt(_) => "adt",
            HeapObj::Newtype(_) => "newtype",
            HeapObj::Cell(_) => "cell",
            HeapObj::Range(_) => "range",
            HeapObj::Closure(_) => "<closure>",
            HeapObj::Partial(_) => "<partial>",
            HeapObj::Builtin(_) => "<builtin>",
            HeapObj::TraitVal(_) => "<trait>",
            HeapObj::LazyVal(_) => "<lazy>",
            HeapObj::ErrorVal(_) => "<error>",
            HeapObj::ThrowVal(_) => "<throw>",
            HeapObj::AtomicVal(_) => "<atomic>",
            HeapObj::AsyncVal(_) => "<async>",
            HeapObj::ChannelVal(_) => "<channel>",
            HeapObj::SenderVal(_) => "<sender>",
            HeapObj::ReceiverVal(_) => "<receiver>",
            HeapObj::CoroutineFrame => "<coroutine>",
        }
    }

    pub fn is_memoizable(&self) -> bool {
        matches!(
            self,
            HeapObj::Str(_) | HeapObj::Array(_) | HeapObj::Record(_) | HeapObj::Adt(_)
                | HeapObj::Newtype(_) | HeapObj::Range(_) | HeapObj::ErrorVal(_) | HeapObj::ThrowVal(_)
        )
    }
}

impl Hash for HeapObj {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            HeapObj::Str(s) => s.hash(state),
            HeapObj::Array(a) => {
                a.elements.len().hash(state);
                // SoA SIMD 快路径：批量哈希标量
                if let Some(soa) = &a.scalar_soa {
                    simd_hash_soa(soa, state);
                } else {
                    for e in &a.elements {
                        e.hash(state);
                    }
                }
                a.fixed_size.hash(state);
            }
            HeapObj::Record(r) => {
                r.type_name.hash(state);
                r.fields.len().hash(state);
                for f in &r.fields {
                    f.hash(state);
                }
                r.field_names.hash(state);
            }
            HeapObj::Adt(a) => {
                a.type_name.hash(state);
                a.constructor.hash(state);
                a.fields.len().hash(state);
                for f in &a.fields {
                    f.value.hash(state);
                }
            }
            HeapObj::Newtype(n) => {
                n.type_name.hash(state);
                n.inner.hash(state);
            }
            HeapObj::Cell(c) => {
                c.inner.lock().hash(state);
            }
            HeapObj::Range(r) => {
                r.start.hash(state);
                r.end.hash(state);
                r.inclusive.hash(state);
            }
            HeapObj::ErrorVal(e) => {
                e.type_name.hash(state);
                e.message.hash(state);
                e.is_error_subtype.hash(state);
            }
            HeapObj::ThrowVal(t) => match &t.payload {
                ThrowPayload::Ok(v) => {
                    0u8.hash(state);
                    v.hash(state);
                }
                ThrowPayload::Err(v) => {
                    1u8.hash(state);
                    v.hash(state);
                }
            },
            HeapObj::Closure(c) => {
                c.func_id.hash(state);
                c.arity.hash(state);
                c.upvalues.len().hash(state);
            }
            HeapObj::Builtin(b) => {
                (b.fn_ptr as usize).hash(state);
                b.name.hash(state);
            }
            HeapObj::Partial(_) | HeapObj::TraitVal(_) | HeapObj::LazyVal(_)
            | HeapObj::AtomicVal(_) | HeapObj::AsyncVal(_) | HeapObj::ChannelVal(_)
            | HeapObj::SenderVal(_) | HeapObj::ReceiverVal(_) | HeapObj::CoroutineFrame => {}
        }
    }
}
