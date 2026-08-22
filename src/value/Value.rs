//! Value.rs — Frond unified value system (merges 14 submodules)

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::AtomicBool;

// Re-export the type discriminant tag from the Tag submodule
pub use super::Tag::ValueTag;

// Cross-submodule: HeapObj::hash (in this file) reuses the SIMD batch hash helper from Arena.rs
use super::Arena::simd_hash_soa;

// =========================================================================
// Part 1: scalar primitive types (scalar.rs + char.rs)
// =========================================================================

// ---- F16 — IEEE 754 half-precision float (binary16) ----

/// IEEE 754 half-precision float: stores its bit pattern in a `u16`.
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

    // ---- IEEE 754 binary16 exact arithmetic (no f64 intermediate) ----
    // Layout: sign(1) | exp(5, bias=15) | fraction(10)
    // Normal mantissa = (1 << 10) | fraction, 11 bits total
    // Subnormal mantissa = fraction, exponent = 1 - bias = -14
    // Same unpack/pack framework as F128; since mantissa is only 11 bits, u32 is sufficient

    fn nan_val() -> Self { F16(0x7C00 | 1) }
    fn inf_val(sign: bool) -> Self { F16(if sign { 0xFC00 } else { 0x7C00 }) }
    fn zero_val(sign: bool) -> Self { F16(if sign { 0x8000 } else { 0 }) }

    /// Unpacks into (sign, unbiased_exp, mantissa).
    /// For normal numbers the mantissa includes the implicit 1 (bit 10 = 1); for subnormals/zero the mantissa = fraction.
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

    /// Normalizes and rounds (sign, exp, mant, sticky) into an F16.
    /// The MSB of `mant` is the implicit 1 (may be at any position); `pack` aligns it to bit 10.
    /// Rounding mode: round-to-nearest-even.
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
        // 11 × 11 = 22-bit product, u32 is sufficient
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
        // (ma << 12) / mb, quotient ~12 bits, u32 is sufficient
        // ma/mb ∈ [0.5, 2), (ma<<12)/mb ∈ [2^11, 2^13), no u32 overflow
        let quot = ((ma as u32) << 12) / mb;
        let stk = ((ma << 12) % mb) != 0;
        // pack semantics: value = mant * 2^(exp - 10)
        // true quotient = (ma/mb) * 2^result_exp = quot * 2^(result_exp - 12)
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

// IEEE 754 totalOrder semantics: NaN is largest (sign bit distinguishes), -0 < +0, negatives reverse by magnitude
impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for F16 {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.0 as i16;
        let b = other.0 as i16;
        // Negatives (sign bit = 1) reverse by magnitude: flip all bits except the sign bit
        let ka = if a < 0 { a ^ 0x7FFF } else { a };
        let kb = if b < 0 { b ^ 0x7FFF } else { b };
        ka.cmp(&kb)
    }
}

/// f32 bit pattern → f16 bit pattern (IEEE 754 round-to-nearest)
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

// ---- F128 — IEEE 754 quadruple-precision float (binary128) ----

/// IEEE 754 quadruple-precision float: stores its bit pattern in `[u8; 16]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct F128(pub [u8; 16]);


impl F128 {
    /// IEEE 754 numeric equality (the 2026-08-18 ruling aligning F128 with
    /// F16/F32/F64): +0.0 == -0.0, NaN != NaN. Bit-exact otherwise — no
    /// lossy f64 roundtrip (binary128 has a 113-bit mantissa).
    pub fn ieee_eq(&self, other: &F128) -> bool {
        let a = u128::from_le_bytes(self.0);
        let b = u128::from_le_bytes(other.0);
        let sign = 1u128 << 127;
        let expo = 0x7FFFu128 << 112;
        let mag_a = a & !sign;
        let mag_b = b & !sign;
        let a_nan = (mag_a & expo) == expo && (mag_a & !expo) != 0;
        let b_nan = (mag_b & expo) == expo && (mag_b & !expo) != 0;
        if a_nan || b_nan { return false; }
        if mag_a == 0 && mag_b == 0 { return true; } // ±0.0
        a == b
    }
}
/// f64→f128 is lossless (f64's 53-bit mantissa shifted left by 60 fills binary128's 113 bits with no loss).
/// f128→f64 (to_f64) implements round-to-nearest-even, correctly rounding the low bits beyond f64 precision.
/// Therefore f64→f128→f64 round-trips losslessly; f128→f64→f128 only rounds when the f128 value exceeds f64 precision (per IEEE 754 semantics).
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
            // Any non-zero payload → canonical NaN; Inf → Inf
            let m: u64 = if mant != 0 { 1 } else { 0 };
            return f64::from_bits((sign << 63) | (0x7FF << 52) | m);
        }

        // True exponent (normal: exp-16383, subnormal: -16382)
        let true_exp = if exp == 0 { -16382 } else { exp - 16383 };
        // Full 113-bit mantissa (normal numbers add the implicit 1)
        let full_mant: u128 = if exp == 0 { mant } else { mant | (1u128 << 112) };

        // f64 exponent (bias 1023)
        let f64_exp = true_exp + 1023;
        if f64_exp >= 0x7FF {
            // Overflow → ±Inf
            return f64::from_bits((sign << 63) | (0x7FF << 52));
        }
        if f64_exp <= 0 {
            // Subnormal or underflow: shift the 113-bit mantissa right to the f64 subnormal position
            // f64 subnormal mantissa occupies bits 0..51, implicit bit is 0, exponent is 0 (true_exp = -1022)
            // target shift = 112 - 51 + (1 - f64_exp) = 62 - f64_exp
            let shift = (62 - f64_exp) as u32;
            if shift >= 128 {
                return f64::from_bits(sign << 63); // underflow → ±0
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

        // Normal: 113-bit mantissa → 53 bits (implicit 1 + 52-bit fraction), shift right 60 bits and round
        let shift = 60u32;
        let round_bit = (full_mant >> (shift - 1)) & 1;
        let sticky = (full_mant & ((1u128 << (shift - 1)) - 1)) != 0;
        let mut result_mant = (full_mant >> shift) as u64;
        // result_mant is now 53 bits (with implicit 1); must fit into f64's 52-bit fraction
        if round_bit != 0 && (sticky || (result_mant & 1) != 0) {
            result_mant += 1;
            if result_mant >> 53 != 0 {
                // Carry-out causes mantissa overflow (1.111... → 10.000...), exponent +1, mantissa resets to zero
                return f64::from_bits((sign << 63) | (((f64_exp as u64) + 1) << 52));
            }
        }
        f64::from_bits((sign << 63) | ((f64_exp as u64) << 52) | (result_mant & ((1u64 << 52) - 1)))
    }

    /// Constructs an F128 from an i128 exactly (no f64 intermediate, avoiding precision loss).
    /// F128 has a 113-bit mantissa and can represent all i128 values exactly.
    pub fn from_i128(x: i128) -> Self {
        if x == 0 {
            return Self::zero_val(false);
        }
        let sign = x < 0;
        let abs = x.unsigned_abs();
        // pack's contract is value = mant × 2^(exp - 112), so exp must be 112
        // for the integer value abs × 2^0 (passing 0 scales by 2^-112).
        Self::pack(sign, 112, abs, false)
    }
    /// Constructs an F128 from a u128 exactly (no f64 intermediate).
    pub fn from_u128(x: u128) -> Self {
        if x == 0 {
            return Self::zero_val(false);
        }
        Self::pack(false, 112, x, false)
    }

    /// Extracts the integer part of an F128 value as i128 (lossless for values within i128 range).
    /// Uses the 113-bit mantissa and exponent directly — never goes through f64.
    pub fn to_i128(self) -> i128 {
        let (sign, exp, mant) = self.unpack();
        // NaN / Inf → 0
        if exp > 16383 + 127 {
            return 0;
        }
        let val = f128_mant_to_i128(exp, mant);
        if sign { -val } else { val }
    }

    /// Extracts the integer part of an F128 value as u128 (lossless for non-negative values within u128 range).
    pub fn to_u128(self) -> u128 {
        let (sign, exp, mant) = self.unpack();
        if sign { return 0; }
        if exp > 16383 + 127 {
            return u128::MAX;
        }
        f128_mant_to_u128(exp, mant)
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
    // ---- IEEE 754 binary128 exact arithmetic (no f64 intermediate) ----
    // Layout: sign(1) | exp(15, bias=16383) | fraction(112)
    // Normal mantissa = (1 << 112) | fraction, 113 bits total
    // Subnormal mantissa = fraction, exponent = 1 - bias = -16382

    fn nan_val() -> Self {
        F128(((0x7FFFu128 << 112) | 1).to_le_bytes())
    }
    fn inf_val(sign: bool) -> Self {
        F128((((sign as u128) << 127) | (0x7FFFu128 << 112)).to_le_bytes())
    }
    fn zero_val(sign: bool) -> Self {
        F128(((sign as u128) << 127).to_le_bytes())
    }

    /// Unpacks into (sign, unbiased_exp, mantissa).
    /// For normal numbers the mantissa includes the implicit 1 (bit 112 = 1); for subnormals/zero the mantissa = fraction.
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

    /// Normalizes and rounds (sign, exp, mant, sticky) into an F128.
    /// The MSB of `mant` is the implicit 1 (may be at any position); `pack` aligns it to bit 112.
    /// `sticky` indicates whether non-zero information exists below the LSB of `mant`.
    /// Rounding mode: round-to-nearest-even.
    fn pack(sign: bool, exp: i32, mant: u128, sticky: bool) -> Self {
        if mant == 0 {
            // Value is extremely small; round-to-nearest-even rounds down to 0
            return Self::zero_val(sign);
        }

        // Normalize: align MSB to bit 112
        let msb = 127 - mant.leading_zeros() as i32;
        let shift = msb - 112;
        let mut adj_exp = exp + shift;
        let mut m = mant;
        let mut stk = sticky;

        // guard bit: highest bit shifted out during right shift
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

        // Overflow → ±Inf
        if biased >= 0x7FFF {
            return Self::inf_val(sign);
        }

        // Subnormal or underflow
        if biased <= 0 {
            let extra = (1 - biased) as u32;
            if extra >= 128 {
                // Complete underflow
                if guard && stk {
                    return Self::zero_val(false); // 0 is even
                }
                return Self::zero_val(sign);
            }
            // Right-shift by `extra` bits, preserving guard/sticky
            if extra > 0 {
                let new_guard = (m >> (extra - 1)) & 1 != 0;
                if extra > 1 {
                    stk = stk || (m & ((1u128 << (extra - 1)) - 1)) != 0;
                }
                guard = new_guard;
                m >>= extra;
            }
            // Round (round-to-nearest-even)
            if guard && (stk || (m & 1) != 0) {
                m = m.wrapping_add(1);
                if m >= (1u128 << 112) {
                    // Carry to smallest normal number
                    return F128((((sign as u128) << 127) | (1u128 << 112)).to_le_bytes());
                }
            }
            return F128((((sign as u128) << 127) | m).to_le_bytes());
        }

        // Normal: bit 112 of m is 1, fraction = bits 0-111
        // Round (round-to-nearest-even)
        if guard && (stk || (m & 1) != 0) {
            m = m.wrapping_add(1);
            // Carry may grow mantissa from 113 to 114 bits (bit 113 = 1)
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

    /// 113-bit × 113-bit → 226-bit product (hi, lo)
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

    /// 256-bit / 113-bit long division, returns (quotient, remainder != 0).
    /// The remainder is always < denom (< 2^113); after left-shift it is < 2^114, so no u128 overflow.
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

    /// Exact negation
    pub fn neg_f128(self) -> Self {
        let bits = u128::from_le_bytes(self.0) ^ (1u128 << 127);
        F128(bits.to_le_bytes())
    }

    /// Exact addition
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
            // +0 + +0 = +0; -0 + -0 = -0; mixed → +0 (round-to-nearest)
            return Self::zero_val(sa && sb);
        }
        if ma == 0 {
            return other;
        }
        if mb == 0 {
            return self;
        }

        // Extend mantissa left by 2 bits (make room for guard/round bits)
        let ma_ext = ma << 2;
        let mb_ext = mb << 2;
        let result_exp;

        // Align exponents (right-shift the smaller, preserving sticky)
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

        // Signed addition
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

        // result_mant is 115 bits (113 + 2); pack normalizes it to 113 bits
        Self::pack(result_sign, result_exp - 2, result_mant, stk)
    }

    /// Exact subtraction
    pub fn sub_f128(self, other: Self) -> Self {
        self.add_f128(other.neg_f128())
    }

    /// Exact multiplication
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

        // 113 × 113 = 226-bit product
        let (hi, lo) = Self::mul_113(ma, mb);

        // Determine product MSB position
        let total_bits = if hi != 0 {
            128 + (128 - hi.leading_zeros() as i32)
        } else {
            128 - lo.leading_zeros() as i32
        };
        let shift = total_bits - 113; // right-shift to 113 bits

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

        // pack semantics: value = mant * 2^(exp - 112)
        // true value = (ma*mb) * 2^(result_exp - 224)
        // mant = (ma*mb) >> shift, so exp = result_exp - 112 + shift
        Self::pack(result_sign, result_exp - 112 + shift, m, stk)
    }

    /// Exact division
    pub fn div_f128(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan_val();
        }
        let (sa, ea, ma) = self.unpack();
        let (sb, eb, mb) = other.unpack();
        let result_sign = sa ^ sb;

        // Inf / Inf = NaN; x / 0 = NaN (x≠0)
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

        // Compute (ma << 114) / mb, yielding a ~115-bit quotient (within u128 range)
        // ma/mb ∈ [0.5, 2), so (ma<<114)/mb ∈ [2^113, 2^115), no u128 overflow
        // pack semantics: value = mant * 2^(exp - 112)
        // true quotient = (ma/mb) * 2^result_exp = quot * 2^(result_exp - 114)
        // so exp = result_exp - 114 + 112 = result_exp - 2
        // 256-bit split of ma * 2^114: hi = ma >> 14, lo = (low 14 bits of ma) << 114.
        // (The previous `ma << 14` for lo put ma's middle bits in the wrong positions,
        // leaving garbage in the quotient's low mantissa bits: 6.0/4.0 printed 1.5
        // but differed from the 1.5f128 literal at bit level.)
        let numer_hi = ma >> 14;
        let numer_lo = (ma & 0x3FFF) << 114;
        let (quot, stk) = Self::div_256_by_113(numer_hi, numer_lo, mb);
        Self::pack(result_sign, result_exp - 2, quot, stk)
    }

    /// Exact modulo: IEEE 754 remainder (result = a - round_to_even(a/b) * b)
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
        // Round q to the nearest even integer
        let q_bits = u128::from_le_bytes(quot.0);
        let q_exp = ((q_bits >> 112) & 0x7FFF) as i32 - 16383;
        let q_int = if q_exp >= 0 {
            // q >= 1, right-shift the fraction to take the integer part
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
        // Use from_u128 for exact construction (from_f64 loses precision when q_int > 2^53)
        let q_val = Self::from_u128(q_int);
        let prod = q_val.mul_f128(other);
        self.sub_f128(prod)
    }
}

/// Converts F128 mantissa (113-bit with implicit 1) + unbiased exponent to i128 integer part.
/// Uses the 113-bit mantissa directly without f64 intermediate — lossless for values within i128 range.
fn f128_mant_to_i128(exp: i32, mant: u128) -> i128 {
    if exp < 112 {
        // Fractional only: shift right to truncate
        (mant >> (112 - exp)) as i128
    } else {
        let shift = exp - 112;
        if shift >= 128 {
            return i128::MAX;
        }
        let result = mant as u128;
        if shift >= 113 {
            // mant is at most 2^113 - 1, shift >= 113 → result > 2^126 → overflow
            return i128::MAX;
        }
        let shifted = result.checked_shl(shift as u32);
        match shifted {
            Some(v) if v <= i128::MAX as u128 => v as i128,
            _ => i128::MAX,
        }
    }
}

/// Converts F128 mantissa (113-bit with implicit 1) + unbiased exponent to u128 integer part.
fn f128_mant_to_u128(exp: i32, mant: u128) -> u128 {
    if exp < 112 {
        mant >> (112 - exp)
    } else {
        let shift = exp - 112;
        if shift >= 128 {
            return u128::MAX;
        }
        mant.checked_shl(shift as u32).unwrap_or(u128::MAX)
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
        // Special-case NaN/Inf; normal values are printed via to_f64 (Phase A2 already does round-to-nearest-even).
        // Full exact decimal output is a future optimization and does not block this plan.
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

// IEEE 754 totalOrder semantics
impl PartialOrd for F128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for F128 {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = u128::from_le_bytes(self.0);
        let b = u128::from_le_bytes(other.0);
        // totalOrder sort key:
        //   negatives (sign=1): flip all bits → maps to [0, 0x7FFF...FFF] (-Inf smallest, -0 largest)
        //   positives (sign=0): set sign bit → maps to [0x8000...000, 0xFFFF...FFF] (+0 smallest, +Inf largest)
        // This makes -0 < +0 (correct totalOrder semantics)
        let ka = if (a >> 127) != 0 { !a } else { a | (1u128 << 127) };
        let kb = if (b >> 127) != 0 { !b } else { b | (1u128 << 127) };
        ka.cmp(&kb)
    }
}

// F16/F128 operator traits: use exact IEEE 754 arithmetic without f64 intermediates
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

// ---- ValueTag / ValueTag moved to Type.rs (re-exported for compatibility) ----

// ---- ScalarValue — scalar value union (16 bytes) ----

/// Scalar value union (16 bytes, accommodates i128/u128/F128).
/// Accessed via ValueTag type guards; unsafe code must perform the corresponding tag check.
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

// ---- Value — Frond runtime unified value representation (spec §3.3) ----

/// Frond runtime unified value representation (spec §3.3).
/// `Value` is self-contained: scalars are inline; heap objects are shared across workers via `Arc`.
#[derive(Clone)]
pub enum Value {
    Null,
    Void,
    /// Scalar value. The tag must be a scalar variant (Bool/Char/I8.../F128);
    /// non-scalar tags (Null/Void/Ref) must not enter this path.
    Scalar(ScalarValue, ValueTag),
    Ref(Arc<HeapObj>),
}

impl Value {
    /// Constructs a scalar value. The tag must be a scalar variant (guaranteed by each typed constructor; non-scalar tags must not enter this path).
    #[inline]
    fn scalar(sv: ScalarValue, tag: ValueTag) -> Self {
        Value::Scalar(sv, tag)
    }
}

unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl Value {
    // ---- Scalar constructors ----
    // All constructors uniformly call Self::scalar(); the tag is correctly passed by each typed constructor.
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
    // 128-bit scalar constructors (bit pattern stored as [u64; 2])
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

    // ---- Heap object constructors ----
    pub fn ref_val(obj: HeapObj) -> Self { Value::Ref(Arc::new(obj)) }
    pub fn from_ref(r: HeapRef) -> Self { Value::Ref(r) }

    pub const NULL: Value = Value::Null;
    pub const VOID: Value = Value::Void;

    // ---- Scalar accessors (with tag guards; automatic promotion/truncation between integer types) ----
    /// Generic integer read: covers all integer ValueTags, uniformly converting to i128.
    /// All as_iN/as_uN/as_isize/as_usize delegate to this method and then `as`-truncate, avoiding special-case matching.
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
    /// Generic float read: covers F16/F32/F64/F128, uniformly converting to f64.
    /// All as_fN delegate to this method, avoiding special-case matching.
    pub fn as_float_f64(&self) -> f64 {
        match self {
            Value::Scalar(v, t) => unsafe {
                match t {
                    ValueTag::F16 => F16(v.f16_val).to_f64(),
                    ValueTag::F32 => v.f32_val as f64,
                    ValueTag::F64 => v.f64_val,
                    ValueTag::F128 => F128(std::mem::transmute(v.f128_val)).to_f64(),
                    // integer → f64 promotion (supports mixed int-float arithmetic, Bug #55)
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
    // ---- Integer accessors: uniformly delegate to as_int_i128, supporting cross-reads between any integer types ----
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
    // ---- Float accessors: uniformly delegate to as_float_f64, supporting cross-reads between any float types ----
    // F16/F32 via f64 intermediate incur no extra precision loss (f64 has a 52-bit mantissa, sufficient to exactly represent all integers rounding to F16/F32)
    pub fn as_f16(&self) -> F16 { F16::from_f64(self.as_float_f64()) }
    pub fn as_f32(&self) -> f32 { self.as_float_f64() as f32 }
    pub fn as_f64(&self) -> f64 { self.as_float_f64() }
    /// F128 accessor: for integer types, constructs directly without f64 intermediate (avoiding i128 precision loss).
    /// F128 has a 113-bit mantissa and can represent all i128/u128 values exactly.
    pub fn as_f128(&self) -> F128 {
        match self {
            Value::Scalar(v, t) => unsafe {
                match t {
                    ValueTag::F16 => F128::from_f64(F16(v.f16_val).to_f64()),
                    ValueTag::F32 => F128::from_f64(v.f32_val as f64),
                    ValueTag::F64 => F128::from_f64(v.f64_val),
                    ValueTag::F128 => F128(std::mem::transmute(v.f128_val)),
                    // integer → F128 direct construction, preserving precision
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
                    ValueTag::Bool => F128::from_f64(if v.bool_val { 1.0 } else { 0.0 }),
                    _ => F128::from_f64(0.0),
                }
            },
            _ => F128::from_f64(0.0),
        }
    }
    // ---- Other scalar accessors ----
    pub fn as_bool(&self) -> bool { match self { Value::Scalar(v, ValueTag::Bool) => unsafe { v.bool_val }, _ => false } }
    pub fn as_char(&self) -> char { match self { Value::Scalar(v, ValueTag::Char) => unsafe { char::from_u32_unchecked(v.char_val) }, _ => '\0' } }

    // ---- Heap object accessors ----
    pub fn heap_obj(&self) -> Option<&HeapObj> { match self { Value::Ref(r) => Some(r.as_ref()), _ => None } }
    pub fn heap_ref(&self) -> Option<HeapRef> { match self { Value::Ref(r) => Some(r.clone()), _ => None } }

    // ---- Discriminants ----
    pub fn is_null(&self) -> bool { matches!(self, Value::Null) }
    pub fn is_void(&self) -> bool { matches!(self, Value::Void) }
    pub fn is_ref(&self) -> bool { matches!(self, Value::Ref(_)) }

    // ---- Scalar tag access (for Hash/Debug/reflection adaptation) ----
    pub fn scalar_tag(&self) -> Option<ValueTag> {
        match self { Value::Scalar(_, t) => Some(*t), _ => None }
    }

    // ---- Weak reference infrastructure (used to break Cell reference cycles) ----
    /// Returns a Weak reference to this value's heap object.
    /// Only meaningful for `Value::Ref`; scalars/Null/Void return None.
    /// Callers can store the Weak inside a Cell to break cycles formed by `a = Cell(b); b = Cell(a)`.
    pub fn make_weak(&self) -> Option<Weak<HeapObj>> {
        match self { Value::Ref(r) => Some(Arc::downgrade(r)), _ => None }
    }

    /// Upgrades a Weak reference back into a Value. Returns None if the original object has been reclaimed.
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
                // Reuse ValueHandle's scalar formatting logic: read the union field by tag
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
                // Hash the corresponding union field by tag
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

// ---- ValueHandle — 4B index handle ----

/// Unique handle for a Frond value: a 4B index encoding the type bucket + index within the bucket.
/// High 8 bits = ValueTag, low 24 bits = index within the bucket.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueHandle(u32);

impl ValueHandle {
    const TAG_SHIFT: u32 = 24;
    const INDEX_MASK: u32 = 0x00FF_FFFF;

    #[inline]
    pub fn new(tag: ValueTag, index: usize) -> Self {
        // [V-3] release also keeps the check: index >= 2^24 would silently truncate (MASK strips the high bits),
        // causing two distinct indices to produce the same ValueHandle → handle aliasing corruption. This is an
        // unrecoverable invariant violation; an explicit panic is preferable to silent corruption (an arena should
        // never allocate more than 16M values of the same type).
        assert!(index < (1 << 24), "ValueHandle index overflow: {index} >= 2^24");
        Self(((tag as u8 as u32) << Self::TAG_SHIFT) | (index as u32 & Self::INDEX_MASK))
    }

    #[inline]
    pub fn tag(self) -> ValueTag {
        // FFI defense: a u32 restored via from_raw in an extern "C" primitive may carry an out-of-range tag
        // (21..=255). Transmuting to an invalid discriminant of a #[repr(u8)] enum is UB,
        // so we use an explicit match; out-of-range values fall back to Null, ensuring any u32 is safe.
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

    /// Constructs a ValueHandle from a raw u32 (used by extern "C" primitives to cross ABI boundaries).
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Converts to a raw u32 (used by extern "C" primitives to cross ABI boundaries).
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

/// Character error: codepoint out of range
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharError {
    InvalidCodepoint,
}

/// Unicode character: wraps a codepoint (u32)
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
        // [V-6] skip the surrogate range + saturate at 0x10FFFF, avoiding wrapping that would produce an invalid codepoint
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
        // [V-6] skip the surrogate range + saturate at 0, avoiding wrapping that would produce an invalid codepoint
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
// Part 2: heap object types (merges 6 files)
// =========================================================================

// ---- str.rs → Str ----

/// Frond string: a reference-counted immutable UTF-8 string
#[derive(Debug, Clone)]
pub struct Str {
    inner: Arc<str>,
}

impl Str {
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

    /// Returns the character at the given codepoint index (UTF-8 safe).
    ///
    /// Returns the `idx`-th Unicode codepoint, or None if out of bounds.
    pub fn char_at(&self, idx: usize) -> Option<char> {
        self.inner.chars().nth(idx)
    }
}

impl PartialEq for Str {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl Eq for Str {}

impl Hash for Str {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl fmt::Display for Str {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

// ---- composite.rs → ArrayValue, RecordField, RecordValue, AdtField, AdtValue, NewtypeValue, Cell, Range, RangeIter ----

/// Array value: elements are mutable (supports push/pop); `fixed_size` of `Some` denotes a fixed-size array
#[derive(Debug, Clone)]
pub struct ArrayValue {
    pub elements: Vec<Value>,
    pub fixed_size: Option<u64>,
    pub elem_is_ref: bool,
    pub scalar_soa: Option<ScalarSoA>,
}

/// SoA contiguous storage: enables SIMD fast paths when all array elements are scalars of the same type
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
    /// Attempts to store a scalar value at the given index.
    /// Returns true if the type matched and the store succeeded; false if the type did not match (caller should invalidate SOA).
    /// Automatically grows (zero-filling) when the index is out of bounds.
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
            _ => false, // type mismatch
        }
    }

    /// Element count of the contiguous storage.
    pub fn soa_len(&self) -> usize {
        match self {
            ScalarSoA::I8(v) => v.len(),
            ScalarSoA::I16(v) => v.len(),
            ScalarSoA::I32(v) => v.len(),
            ScalarSoA::I64(v) => v.len(),
            ScalarSoA::U8(v) => v.len(),
            ScalarSoA::U16(v) => v.len(),
            ScalarSoA::U32(v) => v.len(),
            ScalarSoA::U64(v) => v.len(),
            ScalarSoA::I128(v) => v.len(),
            ScalarSoA::U128(v) => v.len(),
            ScalarSoA::Isize(v) => v.len(),
            ScalarSoA::Usize(v) => v.len(),
            ScalarSoA::Bool(v) => v.len(),
            ScalarSoA::Char(v) => v.len(),
            ScalarSoA::F16(v) => v.len(),
            ScalarSoA::F32(v) => v.len(),
            ScalarSoA::F64(v) => v.len(),
            ScalarSoA::F128(v) => v.len(),
        }
    }

    /// The scalar tag this storage holds (mirrors the element type name).
    pub fn tag(&self) -> crate::value::ValueTag {
        match self {
            ScalarSoA::I8(_) => crate::value::ValueTag::I8,
            ScalarSoA::I16(_) => crate::value::ValueTag::I16,
            ScalarSoA::I32(_) => crate::value::ValueTag::I32,
            ScalarSoA::I64(_) => crate::value::ValueTag::I64,
            ScalarSoA::U8(_) => crate::value::ValueTag::U8,
            ScalarSoA::U16(_) => crate::value::ValueTag::U16,
            ScalarSoA::U32(_) => crate::value::ValueTag::U32,
            ScalarSoA::U64(_) => crate::value::ValueTag::U64,
            ScalarSoA::I128(_) => crate::value::ValueTag::I128,
            ScalarSoA::U128(_) => crate::value::ValueTag::U128,
            ScalarSoA::Isize(_) => crate::value::ValueTag::Isize,
            ScalarSoA::Usize(_) => crate::value::ValueTag::Usize,
            ScalarSoA::Bool(_) => crate::value::ValueTag::Bool,
            ScalarSoA::Char(_) => crate::value::ValueTag::Char,
            ScalarSoA::F16(_) => crate::value::ValueTag::F16,
            ScalarSoA::F32(_) => crate::value::ValueTag::F32,
            ScalarSoA::F64(_) => crate::value::ValueTag::F64,
            ScalarSoA::F128(_) => crate::value::ValueTag::F128,
        }
    }

    /// Element type name ("u8", "i32", ...) via the ValueTag mapping.
    pub fn type_name(&self) -> &'static str {
        self.tag().type_name()
    }

    /// Read-side mirror of `try_store`: the Value at idx, or None when out of
    /// bounds. Never resizes. SoA is the source of truth when present, so
    /// scalar reads materialize straight from the contiguous storage.
    pub fn get_value(&self, idx: usize) -> Option<Value> {
        match self {
            ScalarSoA::I8(v) => v.get(idx).map(|&x| Value::i8(x)),
            ScalarSoA::I16(v) => v.get(idx).map(|&x| Value::i16(x)),
            ScalarSoA::I32(v) => v.get(idx).map(|&x| Value::i32(x)),
            ScalarSoA::I64(v) => v.get(idx).map(|&x| Value::i64(x)),
            ScalarSoA::U8(v) => v.get(idx).map(|&x| Value::u8(x)),
            ScalarSoA::U16(v) => v.get(idx).map(|&x| Value::u16(x)),
            ScalarSoA::U32(v) => v.get(idx).map(|&x| Value::u32(x)),
            ScalarSoA::U64(v) => v.get(idx).map(|&x| Value::u64(x)),
            ScalarSoA::I128(v) => v.get(idx).map(|&x| Value::i128(x)),
            ScalarSoA::U128(v) => v.get(idx).map(|&x| Value::u128(x)),
            ScalarSoA::Isize(v) => v.get(idx).map(|&x| Value::isize_val(x)),
            ScalarSoA::Usize(v) => v.get(idx).map(|&x| Value::usize_val(x)),
            ScalarSoA::Bool(v) => v.get(idx).map(|&x| Value::bool_val(x)),
            ScalarSoA::Char(v) => v.get(idx).map(|&x| Value::char_val(char::from_u32(x).unwrap_or(' '))),
            ScalarSoA::F16(v) => v.get(idx).map(|&x| Value::f16(F16(x))),
            ScalarSoA::F32(v) => v.get(idx).map(|&x| Value::f32(x)),
            ScalarSoA::F64(v) => v.get(idx).map(|&x| Value::f64(x)),
            ScalarSoA::F128(v) => v.get(idx).map(|&x| Value::f128(x)),
        }
    }
}

impl ArrayValue {
    pub fn new(elements: Vec<Value>) -> Self {
        let mut s = Self { elements, fixed_size: None, elem_is_ref: false, scalar_soa: None };
        s.optimize_soa();
        s
    }
    pub fn new_fixed(elements: Vec<Value>, size: u64) -> Self {
        let mut s = Self { elements, fixed_size: Some(size), elem_is_ref: false, scalar_soa: None };
        s.optimize_soa();
        s
    }
    /// Length: SoA-first. In the single-source model a cloned SoA array may
    /// carry an EMPTY elements vector (deep clone copies only the contiguous
    /// storage); elements.len() would report 0.
    pub fn len(&self) -> usize {
        if let Some(soa) = &self.scalar_soa {
            return soa.soa_len();
        }
        self.elements.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Element read. SoA is the source of truth when present (marshal
    /// writebacks may leave `elements` stale); scalar construction from the
    /// contiguous storage costs the same as cloning a scalar Value.
    pub fn get(&self, index: usize) -> Option<Value> {
        if let Some(soa) = &self.scalar_soa {
            return soa.get_value(index);
        }
        self.elements.get(index).cloned()
    }
    /// Fill the SoA fast-path storage when every element is a scalar of the
    /// same tag (same criteria as the former ValueArena::optimize_array_soa,
    /// moved here so EVERY construction site gets it automatically).
    pub fn optimize_soa(&mut self) {
        if self.elements.is_empty() { return; }
        let tag = match self.elements[0].scalar_tag() {
            Some(t) => t,
            None => return,
        };
        if !self.elements.iter().all(|h| h.scalar_tag() == Some(tag)) {
            return;
        }
        self.scalar_soa = Some(match tag {
            ValueTag::I8 => ScalarSoA::I8(self.elements.iter().map(|h| h.as_i8()).collect()),
            ValueTag::I16 => ScalarSoA::I16(self.elements.iter().map(|h| h.as_i16()).collect()),
            ValueTag::I32 => ScalarSoA::I32(self.elements.iter().map(|h| h.as_i32()).collect()),
            ValueTag::I64 => ScalarSoA::I64(self.elements.iter().map(|h| h.as_i64()).collect()),
            ValueTag::U8 => ScalarSoA::U8(self.elements.iter().map(|h| h.as_u8()).collect()),
            ValueTag::U16 => ScalarSoA::U16(self.elements.iter().map(|h| h.as_u16()).collect()),
            ValueTag::U32 => ScalarSoA::U32(self.elements.iter().map(|h| h.as_u32()).collect()),
            ValueTag::U64 => ScalarSoA::U64(self.elements.iter().map(|h| h.as_u64()).collect()),
            ValueTag::Bool => ScalarSoA::Bool(self.elements.iter().map(|h| h.as_bool()).collect()),
            ValueTag::Char => ScalarSoA::Char(self.elements.iter().map(|h| h.as_char() as u32).collect()),
            ValueTag::F32 => ScalarSoA::F32(self.elements.iter().map(|h| h.as_f32()).collect()),
            ValueTag::F64 => ScalarSoA::F64(self.elements.iter().map(|h| h.as_f64()).collect()),
            ValueTag::I128 => ScalarSoA::I128(self.elements.iter().map(|h| h.as_i128()).collect()),
            ValueTag::U128 => ScalarSoA::U128(self.elements.iter().map(|h| h.as_u128()).collect()),
            ValueTag::Isize => ScalarSoA::Isize(self.elements.iter().map(|h| h.as_isize()).collect()),
            ValueTag::Usize => ScalarSoA::Usize(self.elements.iter().map(|h| h.as_usize()).collect()),
            ValueTag::F16 => ScalarSoA::F16(self.elements.iter().map(|h| h.as_u16()).collect()),
            ValueTag::F128 => ScalarSoA::F128(self.elements.iter().map(|h| h.as_f128()).collect()),
            // scalar_tag() only yields the 18 scalar tags; Null/Void/Ref are
            // unreachable here but must be covered for exhaustiveness.
            ValueTag::Null | ValueTag::Void | ValueTag::Ref => return,
        });
    }
    pub fn push(&mut self, val: Value) {
        self.elements.push(val);
    }
    pub fn pop(&mut self) -> Option<Value> {
        self.elements.pop()
    }
    /// Uniformly collects u8 bytes: SOA fast path (U8 contiguous storage) or fallback to per-element extraction.
    /// Encapsulates dual-representation access; callers need not care whether SOA is enabled.
    pub fn collect_u8_bytes(&self) -> Vec<u8> {
        if let Some(crate::value::ScalarSoA::U8(ref data)) = self.scalar_soa {
            return data.clone();
        }
        self.elements.iter().map(|e| e.as_u8()).collect()
    }
}

/// Record field: optional name + value
#[derive(Debug, Clone)]
pub struct RecordField {
    pub name: Option<String>,
    pub value: ValueHandle,
}

/// Record value: structured data of a named type
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

/// ADT field: a constructor argument
#[derive(Debug, Clone)]
pub struct AdtField {
    pub name: Option<String>,
    pub value: Value,
}

/// ADT value: an algebraic data type instance
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

/// Newtype value: a named type wrapping a single inner value
#[derive(Debug, Clone)]
pub struct NewtypeValue {
    pub type_name: String,
    pub inner: ValueHandle,
}

/// Cell: a mutable reference cell (runtime carrier of `&T` reference semantics).
///
/// Holds a `Value` internally (self-contained value; scalar inline + heap object Arc sharing).
/// `&expr` creates an `Arc<HeapObj::Cell>` wrapping the current value; `*r` reads the Cell;
/// `*r = v` writes the Cell. Multiple references share the same Arc; writes are visible to all references.
#[derive(Debug)]
pub struct Cell {
    inner: std::cell::UnsafeCell<Value>,
}

// Safety: the engine executes user graphs single-threaded (the same argument
// as the `Arc::as_ptr` in-place mutation in compute_record_field_set /
// compute_array_store — frames are suspended while callees run; async is
// cooperative; rayon only parallelizes arena construction, never graph
// execution). Cell get/set therefore need no lock; the previous
// parking_lot::Mutex cost an uncontended lock per scalar `var` store, which
// the all-vars place model made per-iteration in hot loops.
unsafe impl Send for Cell {}
unsafe impl Sync for Cell {}

impl Clone for Cell {
    fn clone(&self) -> Self {
        Self { inner: std::cell::UnsafeCell::new(self.get()) }
    }
}

impl Cell {
    pub fn new(val: Value) -> Self {
        Self { inner: std::cell::UnsafeCell::new(val) }
    }
    /// Returns a clone of the inner value.
    pub fn get(&self) -> Value {
        unsafe { (*self.inner.get()).clone() }
    }
    pub fn set(&self, val: Value) {
        unsafe { *self.inner.get() = val; }
    }

    /// Returns a Weak reference to itself (used to break reference cycles).
    /// The caller must ensure the Cell is wrapped in an `Arc<HeapObj::Cell>`;
    /// if the supplied Arc is not a Cell, returns None.
    pub fn downgrade(arc: &Arc<HeapObj>) -> Option<Weak<HeapObj>> {
        match arc.as_ref() {
            HeapObj::Cell(_) => Some(Arc::downgrade(arc)),
            _ => None,
        }
    }
}

/// Range value
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

/// Range iterator (internal to composite)
#[derive(Debug, Clone)]
pub struct RangeIter {
    pub current: i64,
    pub end: i64,
    pub inclusive: bool,
}

// ---- callable.rs → BuiltinFn, Builtin, Closure, PartialApplication, TraitValue, LazyValue ----

/// Built-in function pointer type
pub type BuiltinFn = fn(&[ValueHandle]) -> Result<ValueHandle, String>;

/// Built-in function value
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

/// Closure value
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

/// Partial application value: a callable produced by binding leading arguments to a function/closure.
///
/// Unified call semantics: when the new argument count < remaining_arity → produces a new Partial (chained partial application);
/// when the new argument count >= remaining_arity → merges bound_args + new args + upvalues and launches the subgraph.
/// upvalues come from the source Closure (empty when partially applying a top-level function), matching Closure's upvalue semantics.
#[derive(Debug, Clone)]
pub struct PartialApplication {
    /// Target subgraph id (same semantics as Closure.func_id)
    pub func_id: u32,
    /// upvalues from the source Closure (empty when partially applying a top-level function)
    pub upvalues: Vec<Value>,
    /// Already-bound leading arguments (in original function parameter order)
    pub bound_args: Vec<Value>,
    /// Remaining argument count = subgraph.param_count - upvalues.len() - bound_args.len()
    pub remaining_arity: u8,
    /// Recursive closure self-reference upvalue index (-1 means no self-reference)
    pub self_upvalue_idx: i32,
}

/// Trait value
#[derive(Debug, Clone)]
pub struct TraitValue {
    pub trait_name: String,
    pub method_names: Vec<String>,
    pub method_values: Vec<Value>,
    pub data: Option<Value>,
    pub owned: bool,
}

/// Lazy value
pub struct LazyValue {
    /// Cached evaluation result (filled after first force)
    /// Mutex allows updating the cache through &LazyValue (interior mutability under Arc sharing)
    pub cached: Mutex<Option<Value>>,
    /// Whether it has been evaluated
    pub forced: AtomicBool,
    /// Closure of the thunk subgraph (func_id = thunk_sg, upvalues = captured values)
    /// On force, this Closure is taken to launch the subgraph computation; the result is stored in cached
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

/// Error value
#[derive(Debug, Clone)]
pub struct ErrorValue {
    pub type_name: String,
    pub message: String,
    pub is_error_subtype: bool,
}

/// Throw payload
///
/// Err directly holds a Value (rather than Arc<RecordValue>), unifying all throw scenarios:
/// - throw of a primitive (i32/str/bool) → Err holds a bare scalar value, no Error(value:v) wrapping needed
/// - throw of a record/adt → Err holds the record Value
/// - internal errors (FieldError/IndexError, etc.) → Err holds a pre-constructed record Value
/// This means after throwing any value, the match pattern `Error(v)` binds v directly to the thrown value.
#[derive(Debug, Clone)]
pub enum ThrowPayload {
    Ok(Value),
    Err(Value),
}

/// Throw value
#[derive(Debug, Clone)]
pub struct ThrowValue {
    pub payload: ThrowPayload,
}

// ---- iterator.rs → fully migrated to Frond builtin (Iterator.frond) ----
// Note: ArrayIterator / StringIterator / RangeIterator have all been migrated to the Frond builtin.

// ---- concurrent.rs → AtomicValue, AsyncStatus, AsyncHandle, ChannelValue, SenderValue, ReceiverValue ----

/// Atomic value
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
    /// Compare-and-exchange: if the current value equals `expected`, replace it with
    /// `new` and return true; otherwise return false (leaving the value unchanged).
    ///
    /// Uses semantic value equality (`value_equals`) rather than reference equality,
    /// so atomic semantics apply to the logical value, not the heap identity.
    pub fn compare_exchange(&self, expected: &Value, new: Value) -> bool {
        let mut guard = self.data.lock().unwrap_or_else(|e| e.into_inner());
        if crate::value::value_equals(&*guard, expected) {
            *guard = new;
            true
        } else {
            false
        }
    }
}

impl Clone for AtomicValue {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}

/// Async task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

/// Async handle
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

/// Global channel id counter (thread-safe; shared by single/multi-worker)
static CHANNEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Channel value
///
/// Uniformly stores the Engine's Value (not ValueHandle), consistent with the async runtime.
/// The id is used to identify RuntimeEvent::ChannelReady events (inline-triggered on_event_arrived after send).
#[derive(Debug)]
pub struct ChannelValue {
    id: u64,
    buffer: Mutex<VecDeque<Value>>,
    capacity: usize,
    closed: Mutex<bool>,
}

/// channel send failure reason (runtime condition, not a programmer error).
#[derive(Debug, Clone, Copy)]
pub enum ChannelSendError {
    /// channel is closed
    Closed,
    /// bounded channel is full
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
    /// Returns the channel's unique id (used to identify RuntimeEvent::ChannelReady events)
    pub fn id(&self) -> u64 {
        self.id
    }
    /// Non-blocking send: pushes to the buffer. Returns Err when full or closed (runtime condition, not a programmer error).
    pub fn send(&self, val: Value) -> Result<(), ChannelSendError> {
        let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        // [V-5] check closed while holding the buffer lock; this mutexes with close (which also holds the buffer lock), eliminating TOCTOU
        if *self.closed.lock().unwrap_or_else(|e| e.into_inner()) {
            return Err(ChannelSendError::Closed);
        }
        if self.capacity > 0 && buf.len() >= self.capacity {
            return Err(ChannelSendError::Full { capacity: self.capacity });
        }
        buf.push_back(val);
        Ok(())
    }
    /// Receive: pops from the front of the buffer; returns None when empty (the await path handles suspension in resolve_and_check_await)
    pub fn recv(&self) -> Option<Value> {
        let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buf.pop_front()
    }
    /// Whether there is data available to read
    pub fn has_data(&self) -> bool {
        !self.buffer.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }
    pub fn close(&self) {
        // [V-5] set closed while holding the buffer lock; this mutexes with send's locked check (lock order buffer→closed is consistent, no deadlock)
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

/// Sender end value
#[derive(Debug, Clone)]
pub struct SenderValue {
    pub channel: Arc<ChannelValue>,
}

/// Receiver end value
#[derive(Debug, Clone)]
pub struct ReceiverValue {
    pub channel: Arc<ChannelValue>,
}

// ---- heap.rs → HeapObj enum + HeapRef + RefKind + impl ----

/// Ownership kind of an FFI opaque pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtrKind {
    /// C-side owned; Frond does not free it (e.g. a `FILE*` returned by `fopen` that the user
    /// must `fclose` manually). v1: all pointers returned from FFI are Borrowed.
    Borrowed,
    /// C-allocated, Frond holds it; Drop invokes the destructor (e.g. a handle with a cleanup fn).
    /// v1 unused — reserved for future RAII FFI support.
    Owned,
}

/// Wrapper for a raw C pointer returned from or passed to FFI (`@extern("C") #{ }#` calls).
/// Stored as a `HeapObj::OpaquePtr` so it can flow through Frond's `Value::Ref(Arc<HeapObj>)`.
#[derive(Debug)]
pub struct OpaquePointer {
    pub ptr: *mut core::ffi::c_void,
    pub kind: PtrKind,
    /// Diagnostic type name (e.g. "FILE", "CURL"); used by reflect/format. `'static` to avoid
    /// allocation overhead — callers pass string literals or `Box::leak`.
    pub type_name: &'static str,
    /// Optional C destructor (e.g. `curl_easy_cleanup`). Invoked on Drop iff `kind == Owned`.
    pub destructor: Option<unsafe extern "C" fn(*mut core::ffi::c_void)>,
}

impl Clone for OpaquePointer {
    fn clone(&self) -> Self {
        // Cloning a Borrowed pointer is safe (shared). Cloning an Owned pointer would create
        // double-free on Drop — v1 only produces Borrowed, so this is fine. If Owned cloning
        // is needed later, it must semantically be a "reference clone" (no ownership transfer).
        Self {
            ptr: self.ptr,
            kind: self.kind,
            type_name: self.type_name,
            destructor: self.destructor,
        }
    }
}

impl Drop for OpaquePointer {
    fn drop(&mut self) {
        if matches!(self.kind, PtrKind::Owned) {
            if let Some(dtor) = self.destructor {
                // SAFETY: caller of the constructor guaranteed ptr is a valid C handle and dtor
                // is the matching cleanup function.
                unsafe { dtor(self.ptr); }
            }
        }
        // Borrowed: do nothing — C side owns the memory.
    }
}

// SAFETY: OpaquePointer holds a raw C pointer. It is Send/Sync because FFI execution is
// single-threaded (engine runs on one worker; caller frames suspend while callee runs — see
// ffi_writeback_u8_buf safety comment in Compute.rs). The pointer itself has no thread affinity
// for our usage patterns. This matches the existing `unsafe impl Send/Sync for Value`.
unsafe impl Send for OpaquePointer {}
unsafe impl Sync for OpaquePointer {}

/// Shared state of a loaded native library (dlopen / LoadLibrary handle).
/// A `Lib` value and every `ForeignFn` resolved from it hold the same `Arc`,
/// so `close()` flips `closed` once and all derived ForeignFns observe it, and
/// Drop releases the OS handle exactly once.
pub struct LibShared {
    pub handle: *mut core::ffi::c_void,
    pub path: String,
    pub closed: std::sync::atomic::AtomicBool,
}

impl Drop for LibShared {
    fn drop(&mut self) {
        if !self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            crate::platform::Dylib::close(self.handle);
        }
    }
}

// SAFETY: same single-threaded-FFE argument as OpaquePointer above.
unsafe impl Send for LibShared {}
unsafe impl Sync for LibShared {}

impl std::fmt::Debug for LibShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibShared")
            .field("path", &self.path)
            .field("closed", &self.closed.load(std::sync::atomic::Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

/// Heap value of a `Lib`: opaque handle over a dynamically loaded native library.
#[derive(Debug, Clone)]
pub struct LibValue {
    pub shared: Arc<LibShared>,
}

/// Heap value of a `ForeignFn[R]`: resolved symbol address + runtime-built AbiSig.
/// Lifetime is tied to the owning library through `shared` (closed libs reject calls).
#[derive(Debug, Clone)]
pub struct ForeignFnValue {
    pub shared: Arc<LibShared>,
    pub addr: *mut core::ffi::c_void,
    pub sig: crate::ffi::Abi::AbiSig,
    pub name: String,
}

/// Heap object: unified representation of all heap-allocated value types (24 kinds)
#[derive(Debug, Clone)]
pub enum HeapObj {
    Str(Str),
    Array(ArrayValue),
    Record(RecordValue),
    Adt(AdtValue),
    Newtype(NewtypeValue),
    Cell(Cell),
    /// Place reference (place model B-stage): handle to a mutable storage
    /// location. `&arr[i]` creates this; `*r` reads the element LIVE (SoA
    /// -aware) and `*r = v` stores in place (same semantics as `arr[i] = v`).
    ArrayElemRef { arr: Value, idx: Value },
    /// Place reference: `&rec.field` / `&this.field`. Field is by name
    /// (records/ADTs store names); read/write mirror record_field_get/set.
    RecordFieldRef { rec: Value, field: Box<str> },
    /// Place reference: `&global` — indexes `graph.global_var_storage`.
    GlobalSlotRef { slot: u32 },
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
    /// FFI opaque pointer (@extern("C") #{ }# calls). Wraps a raw `*mut c_void`.
    OpaquePtr(OpaquePointer),
    /// `Lib` value: handle over a dlopen/LoadLibrary-loaded native library.
    LibVal(LibValue),
    /// `ForeignFn[R]` value: resolved symbol + runtime AbiSig (from `Lib.lookup`).
    ForeignFnVal(ForeignFnValue),
}

/// Heap reference: a reference-counted heap object
pub type HeapRef = Arc<HeapObj>;

/// Reference kind enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefKind {
    Str, Array, Record, Adt, Newtype, Cell, Range, Closure, Partial, Builtin,
    TraitVal, LazyVal, ErrorVal, ThrowVal, ArrayElemRef, RecordFieldRef, GlobalSlotRef,
    AtomicVal, AsyncVal, ChannelVal, SenderVal, ReceiverVal, CoroutineFrame,
    OpaquePtr, LibVal, ForeignFnVal,
}

impl HeapObj {
    /// Uniformly extracts the underlying channel: ChannelVal/SenderVal/ReceiverVal all share the same Arc<ChannelValue>.
    /// Eliminates repeated dispatch over the three types; send/close/await/select all call this method.
    pub fn channel(&self) -> Option<&Arc<ChannelValue>> {
        match self {
            HeapObj::ChannelVal(ch) => Some(ch),
            HeapObj::SenderVal(tx) => Some(&tx.channel),
            HeapObj::ReceiverVal(rx) => Some(&rx.channel),
            _ => None,
        }
    }

    /// Uniform field access: Record/Adt look up a field value by name;
    /// ChannelVal derives by channel protocol fields (sender/receiver).
    /// Eliminates hardcoded dispatch over field names and types in compute_record_field_get.
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
            HeapObj::ArrayElemRef { .. } => RefKind::ArrayElemRef,
            HeapObj::RecordFieldRef { .. } => RefKind::RecordFieldRef,
            HeapObj::GlobalSlotRef { .. } => RefKind::GlobalSlotRef,
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
            HeapObj::OpaquePtr(_) => RefKind::OpaquePtr,
            HeapObj::LibVal(_) => RefKind::LibVal,
            HeapObj::ForeignFnVal(_) => RefKind::ForeignFnVal,
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
            HeapObj::ArrayElemRef { .. } => "array_elem_ref",
            HeapObj::RecordFieldRef { .. } => "record_field_ref",
            HeapObj::GlobalSlotRef { .. } => "global_slot_ref",
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
            HeapObj::OpaquePtr(op) => op.type_name,
            HeapObj::LibVal(_) => "Lib",
            HeapObj::ForeignFnVal(_) => "ForeignFn",
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
            HeapObj::ArrayElemRef { .. } => "array_elem_ref",
            HeapObj::RecordFieldRef { .. } => "record_field_ref",
            HeapObj::GlobalSlotRef { .. } => "global_slot_ref",
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
            HeapObj::OpaquePtr(op) => op.type_name,
            HeapObj::LibVal(_) => "<lib>",
            HeapObj::ForeignFnVal(_) => "<foreignfn>",
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
                a.len().hash(state);
                // SoA SIMD fast path: batch-hash scalars
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
                c.get().hash(state);
            }
            HeapObj::ArrayElemRef { arr, idx } => {
                arr.hash(state);
                idx.hash(state);
            }
            HeapObj::RecordFieldRef { rec, field } => {
                rec.hash(state);
                field.hash(state);
            }
            HeapObj::GlobalSlotRef { slot } => slot.hash(state),
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
            | HeapObj::SenderVal(_) | HeapObj::ReceiverVal(_) | HeapObj::CoroutineFrame
            | HeapObj::OpaquePtr(_) | HeapObj::LibVal(_) | HeapObj::ForeignFnVal(_) => {}
        }
    }
}
