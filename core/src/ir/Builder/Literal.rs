//! Literal — Free functions: numeric-literal parsing (hex float, bigint, F128). Mechanically split from Builder.rs (no logic changes).

use super::*;

/// Detect the type suffix of a float literal, returning (stripped, suffix).
pub(super) fn detect_float_suffix(s: &str) -> (&str, Option<&str>) {
    for suffix in &["f128", "f64", "f32", "f16"] {
        if s.ends_with(suffix) {
            return (&s[..s.len() - suffix.len()], Some(suffix));
        }
    }
    (s, None)
}

// =========================================================================
// Integer literal parsing + type range checking
// =========================================================================

/// Parse the raw text of an integer literal into i128, supporting 0x/0o/0b prefixes and underscore separators.
/// Returns an error with span info on parse failure (invalid syntax).
pub(super) fn parse_int_to_i128(raw: &str, span: crate::ast::Ast::Span) -> Result<i128, String> {
    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = cleaned
        .strip_prefix("0x").map(|s| (s, 16u32))
        .or_else(|| cleaned.strip_prefix("0o").map(|s| (s, 8)))
        .or_else(|| cleaned.strip_prefix("0b").map(|s| (s, 2)))
        .unwrap_or((cleaned.as_str(), 10));
    i128::from_str_radix(digits, radix).map_err(|_| {
        format!("invalid integer literal '{}' at line {}:{}", raw, span.line, span.column)
    })
}

/// Parse the raw text of an integer literal into u128, supporting 0x/0o/0b prefixes and underscore separators.
/// u128 has unsigned semantics (no leading minus), used for u128-suffix literals to cover the full 0..=2^128-1 range.
/// Returns an error with span info on parse failure (invalid syntax or a leading minus).
pub(super) fn parse_int_to_u128(raw: &str, span: crate::ast::Ast::Span) -> Result<u128, String> {
    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = cleaned
        .strip_prefix("0x").map(|s| (s, 16u32))
        .or_else(|| cleaned.strip_prefix("0o").map(|s| (s, 8)))
        .or_else(|| cleaned.strip_prefix("0b").map(|s| (s, 2)))
        .unwrap_or((cleaned.as_str(), 10));
    u128::from_str_radix(digits, radix).map_err(|_| {
        format!("invalid integer literal '{}' at line {}:{}", raw, span.line, span.column)
    })
}

/// Range-check an i128 value against the target type and convert it to a ConstValue.
/// Returns an error with the type name, valid range, and span info when out of range.
pub(super) fn check_int_range(v: i128, ty_name: &str, raw: &str, span: crate::ast::Ast::Span) -> Result<ConstValue, String> {
    macro_rules! try_int {
        ($ty:ty, $variant:ident) => {
            match <$ty>::try_from(v) {
                Ok(val) => return Ok(ConstValue::$variant(val)),
                Err(_) => return Err(format!(
                    "integer literal '{}' at line {}:{} is out of range for {} (valid range: {}..={})",
                    raw, span.line, span.column, ty_name, <$ty>::MIN, <$ty>::MAX
                )),
            }
        };
    }
    // Single source of truth: derived via ValueTag::from_name, eliminating string special-casing
    let tag = crate::value::ValueTag::from_name(ty_name).unwrap_or(crate::value::ValueTag::I32);
    match tag {
        crate::value::ValueTag::I8 => try_int!(i8, I8),
        crate::value::ValueTag::I16 => try_int!(i16, I16),
        crate::value::ValueTag::I32 => try_int!(i32, I32),
        crate::value::ValueTag::I64 => try_int!(i64, I64),
        crate::value::ValueTag::I128 => Ok(ConstValue::I128(v)),
        crate::value::ValueTag::U8 => try_int!(u8, U8),
        crate::value::ValueTag::U16 => try_int!(u16, U16),
        crate::value::ValueTag::U32 => try_int!(u32, U32),
        crate::value::ValueTag::U64 => try_int!(u64, U64),
        crate::value::ValueTag::U128 => try_int!(u128, U128),
        crate::value::ValueTag::Isize => try_int!(isize, Isize),
        crate::value::ValueTag::Usize => try_int!(usize, Usize),
        _ => try_int!(i32, I32),
    }
}

// =========================================================================
// Hexadecimal float literal parsing (exact IEEE 754 bit patterns)
// =========================================================================
// Format: 0x<integer part>.<fractional part>p<exponent part>
//   0x1.921fb54442d18p+1 = 1.* 16^... * 2^(+1) = PI (f64)
// Supports positive/negative exponents, optional sign, and upper/lower-case 0x/P.

/// Parse a hexadecimal float literal into an f64 bit pattern, returning f64.
pub(super) fn parse_hex_float_f64(s: &str) -> Option<f64> {
    let bits = parse_hex_float_to_u128(s, 11, 52, 1023)?;
    Some(f64::from_bits(bits as u64))
}

/// Parse a hexadecimal float literal into an f32 bit pattern, returning f32.
pub(super) fn parse_hex_float_f32(s: &str) -> Option<f32> {
    let bits = parse_hex_float_to_u128(s, 8, 23, 127)?;
    Some(f32::from_bits(bits as u32))
}

/// Parse a hexadecimal float literal into an f16 bit pattern, returning u16 bits.
pub(super) fn parse_hex_float_f16(s: &str) -> Option<u16> {
    let bits = parse_hex_float_to_u128(s, 5, 10, 15)?;
    Some(bits as u16)
}

/// Parse a hexadecimal float literal into an f128 bit pattern, returning [u8; 16].
pub(super) fn parse_hex_float_f128(s: &str) -> Option<[u8; 16]> {
    let bits = parse_hex_float_to_u128(s, 15, 112, 16383)?;
    Some(bits.to_le_bytes())
}

/// Generic hexadecimal float parser.
/// Params: (literal, exponent bit width, mantissa bit width, exponent bias)
/// Returns: a u128 bit pattern (the caller truncates to the target width)
pub(super) fn parse_hex_float_to_u128(s: &str, exp_bits: u32, mant_bits: u32, exp_bias: i64) -> Option<u128> {
    // Strip the 0x/0X prefix
    let body = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;

    // Split the mantissa part and the exponent part (p or P)
    let p_pos = body.find(|c| c == 'p' || c == 'P')?;
    let mantissa_str = &body[..p_pos];
    let exp_str = &body[p_pos + 1..];

    // Parse the mantissa: may contain a '.'
    let (int_part, frac_part) = match mantissa_str.find('.') {
        Some(dot) => (&mantissa_str[..dot], &mantissa_str[dot + 1..]),
        None => (mantissa_str, ""),
    };

    // Convert the hex mantissa to a numeric value (ignore the decimal point position; collect all hex digits first)
    let mut mantissa: u128 = 0;
    let mut frac_hex_digits: i32 = 0; // number of hex digits after the decimal point

    // Integer part
    for c in int_part.chars() {
        let d = c.to_digit(16)?;
        mantissa = mantissa.checked_mul(16)?.checked_add(d as u128)?;
    }

    // Fractional part
    for c in frac_part.chars() {
        let d = c.to_digit(16)?;
        mantissa = mantissa.checked_mul(16)?.checked_add(d as u128)?;
        frac_hex_digits += 1;
    }

    if mantissa == 0 {
        // Zero: may carry a sign, but the current implementation does not parse a sign prefix (the lexer already handles the minus)
        return Some(0);
    }

    // Parse the binary exponent (the part after p)
    let exp2: i64 = exp_str.parse().ok()?;

    // Actual exponent = exp2 - frac_hex_digits * 4 (because each hex digit = 4 bits)
    let binary_exp = exp2 - (frac_hex_digits as i64) * 4;

    // Normalize the mantissa: find the most significant bit, compute the unbiased exp
    // MSB position of mantissa (0-indexed from LSB)
    let msb = 127 - mantissa.leading_zeros() as i64;

    // We want to normalize the mantissa into the 1.xxx form:
    // The current mantissa represents an integer with its binary point at the end.
    // After normalization: mantissa = 1.fraction * 2^(msb + binary_exp)
    // But the MSB of mantissa is the implicit 1, so unbiased_exp = msb + binary_exp
    let unbiased_exp = msb + binary_exp;

    // Extract the fraction bits (the bits after removing the MSB)
    let fraction_mant = mantissa & ((1u128 << msb) - 1);
    let frac_bits_available = msb as u32;

    // Round the fraction to mant_bits (round-to-nearest-even)
    // Returns (fraction_field, exp_adjust)
    let (fraction, exp_adjust): (u128, i64) = if frac_bits_available > mant_bits {
        let shift = frac_bits_available - mant_bits;
        let kept = fraction_mant >> shift;
        let remainder = fraction_mant & ((1u128 << shift) - 1);
        let halfway = 1u128 << (shift - 1);
        let mut rounded = kept;
        if remainder > halfway {
            rounded += 1;
        } else if remainder == halfway {
            if kept & 1 != 0 {
                rounded += 1;
            }
        }
        if rounded >> mant_bits != 0 {
            (0, 1)
        } else {
            (rounded, 0)
        }
    } else if frac_bits_available < mant_bits {
        (fraction_mant << (mant_bits - frac_bits_available), 0)
    } else {
        (fraction_mant, 0)
    };

    let biased_exp = unbiased_exp + exp_adjust + exp_bias;
    let max_biased = (1i64 << exp_bits) - 1;

    if biased_exp >= max_biased {
        return Some((max_biased as u128) << mant_bits);
    }

    if biased_exp > 0 {
        let exp_field = (biased_exp as u128) << mant_bits;
        let frac_field = fraction & ((1u128 << mant_bits) - 1);
        return Some(exp_field | frac_field);
    }

    // biased_exp <= 0: subnormal or zero
    let shift = (1 - biased_exp) as u32;
    if shift >= 128 {
        return Some(0);
    }
    let full_mant = (1u128 << mant_bits) | (fraction & ((1u128 << mant_bits) - 1));
    let sub_fraction = (full_mant >> shift) & ((1u128 << mant_bits) - 1);
    if sub_fraction == 0 {
        return Some(0);
    }
    Some(sub_fraction)
}

// =========================================================================
// Decimal float literal -> IEEE 754 binary128 exact parsing (no f64 intermediary)
// =========================================================================
// Algorithm: decimal digits * 10^e10 -> big integer M * 2^e2 -> normalize 113-bit mantissa
//            + round-to-nearest-even rounding -> binary128 bit pattern.
// Big integers are represented as Vec<u64> little-endian; only multiply/divide by small
// integers and left/right shifts are needed, avoiding big-integer / big-integer division
// (10^k = 2^k * 5^k, so multiply/divide by 5 step by step suffices).

/// Decimal digit string -> Vec<u64> big integer (little-endian limbs).
pub(super) fn bigint_from_dec(s: &str) -> Vec<u64> {
    let mut limbs = vec![0u64];
    for c in s.chars() {
        let d = (c as u8 - b'0') as u64;
        let mut carry = d;
        for l in limbs.iter_mut() {
            let prod = (*l as u128) * 10 + carry as u128;
            *l = prod as u64;
            carry = (prod >> 64) as u64;
        }
        if carry != 0 {
            limbs.push(carry);
        }
    }
    limbs
}

/// Multiply a big integer by a small integer (in place).
pub(super) fn bigint_mul_small(limbs: &mut Vec<u64>, m: u64) {
    let mut carry = 0u128;
    for l in limbs.iter_mut() {
        let prod = (*l as u128) * (m as u128) + carry;
        *l = prod as u64;
        carry = prod >> 64;
    }
    if carry != 0 {
        limbs.push(carry as u64);
    }
}

/// Divide a big integer by a small integer (in place), returning the remainder.
pub(super) fn bigint_divmod_small(limbs: &mut Vec<u64>, d: u64) -> u64 {
    let mut rem = 0u128;
    for l in limbs.iter_mut().rev() {
        let cur = (rem << 64) | (*l as u128);
        *l = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    while limbs.len() > 1 && *limbs.last().unwrap() == 0 {
        limbs.pop();
    }
    rem as u64
}

/// Left-shift a big integer by n bits (in place).
pub(super) fn bigint_shl(limbs: &mut Vec<u64>, n: u32) {
    let word_shift = (n / 64) as usize;
    let bit_shift = n % 64;
    if bit_shift > 0 {
        let mut carry = 0u64;
        for l in limbs.iter_mut() {
            let new = (*l << bit_shift) | carry;
            carry = *l >> (64 - bit_shift);
            *l = new;
        }
        if carry != 0 {
            limbs.push(carry);
        }
    }
    if word_shift > 0 {
        limbs.splice(0..0, std::iter::repeat(0u64).take(word_shift));
    }
}

/// Big integer bit length (most significant bit position + 1).
pub(super) fn bigint_bit_len(limbs: &[u64]) -> u32 {
    let mut i = limbs.len();
    while i > 0 && limbs[i - 1] == 0 {
        i -= 1;
    }
    if i == 0 {
        return 0;
    }
    ((i - 1) * 64 + (64 - limbs[i - 1].leading_zeros()) as usize) as u32
}

/// Extract bits [start, start+n-1] from a big integer (n <= 128).
pub(super) fn bigint_extract_bits(limbs: &[u64], start: u32, n: u32) -> u128 {
    let mut result: u128 = 0;
    for i in 0..n {
        let pos = (start + i) as usize;
        let word = pos / 64;
        let bit = pos % 64;
        if word < limbs.len() && (limbs[word] >> bit) & 1 != 0 {
            result |= 1u128 << i;
        }
    }
    result
}

/// Whether bit `pos` of a big integer is 1 (pos is i64 to allow negative values to return false).
pub(super) fn bigint_bit(limbs: &[u64], pos: i64) -> bool {
    if pos < 0 {
        return false;
    }
    let pos = pos as usize;
    let word = pos / 64;
    let bit = pos % 64;
    word < limbs.len() && (limbs[word] >> bit) & 1 != 0
}

/// Whether the low n bits of a big integer are non-zero.
pub(super) fn bigint_low_nonzero(limbs: &[u64], n: u32) -> bool {
    if n == 0 {
        return false;
    }
    let words = (n / 64) as usize;
    let bits = n % 64;
    for i in 0..words.min(limbs.len()) {
        if limbs[i] != 0 {
            return true;
        }
    }
    if bits > 0 && words < limbs.len() {
        let mask = (1u64 << bits) - 1;
        if limbs[words] & mask != 0 {
            return true;
        }
    }
    false
}

/// Convert the low 128 bits of a big integer to u128.
pub(super) fn bigint_to_u128(limbs: &[u64]) -> u128 {
    let mut r = 0u128;
    for i in 0..2.min(limbs.len()) {
        r |= (limbs[i] as u128) << (64 * i);
    }
    r
}

/// Decimal float literal -> IEEE 754 binary128 bit pattern ([u8;16] little-endian).
///
/// Performs exact conversion via big-integer arithmetic without an f64 intermediary (round-to-nearest-even).
/// Supports: [+-]digits[.digits][e[+-]digits]
pub(crate) fn parse_decimal_f128(s: &str) -> Option<[u8; 16]> {
    // 1. Parse the decimal format
    let s = s.trim();
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        (false, rest)
    } else {
        (false, s)
    };

    // Split the exponent part e/E
    let (mantissa_str, exp_str) = match body.find(|c| c == 'e' || c == 'E') {
        Some(pos) => (&body[..pos], &body[pos + 1..]),
        None => (body, ""),
    };
    let exp10: i32 = if exp_str.is_empty() { 0 } else { exp_str.parse().ok()? };

    // Split the decimal point
    let (int_part, frac_part) = match mantissa_str.find('.') {
        Some(pos) => (&mantissa_str[..pos], &mantissa_str[pos + 1..]),
        None => (mantissa_str, ""),
    };

    let digits_str: String = format!("{}{}", int_part, frac_part);
    if digits_str.is_empty() || !digits_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let frac_len = frac_part.len() as i32;
    let e10 = exp10 - frac_len;

    // Fast return for zero
    if digits_str.chars().all(|c| c == '0') {
        let bits: u128 = if sign { 1u128 << 127 } else { 0 };
        return Some(bits.to_le_bytes());
    }

    // 2. digits -> big integer M
    let mut m_big = bigint_from_dec(&digits_str);
    let digits_bitlen = bigint_bit_len(&m_big);
    let mut e2: i64 = 0;
    let mut div_sticky = false;

    // 3. Estimate range, fast-path inf/0
    let log2_est = (digits_bitlen as f64 - 1.0) + (e10 as f64) * 3.32193;
    if log2_est > 16384.0 {
        let bits: u128 = (if sign { 1u128 << 127 } else { 0 }) | (0x7FFFu128 << 112);
        return Some(bits.to_le_bytes());
    }
    if log2_est < -16510.0 {
        let bits: u128 = if sign { 1u128 << 127 } else { 0 };
        return Some(bits.to_le_bytes());
    }

    // 4. Handle e10: value = digits * 10^e10 = digits * 5^e10 * 2^e10
    if e10 > 0 {
        for _ in 0..e10 {
            bigint_mul_small(&mut m_big, 5);
        }
        e2 = e10 as i64;
    } else if e10 < 0 {
        // value = digits / 10^k = (digits * 2^P / 5^k) * 2^(-k-P), k = -e10
        let k = (-e10) as u64;
        // P must ensure at least 114 bits of precision after M/5^k: P >= 114 - digits_bitlen + 2.322*k
        let p_needed = (2.4 * (k as f64)) as u32 + 128;
        bigint_shl(&mut m_big, p_needed);
        e2 = -(k as i64) - (p_needed as i64);
        for _ in 0..k {
            let r = bigint_divmod_small(&mut m_big, 5);
            if r != 0 {
                div_sticky = true;
            }
        }
    }

    // 5. Normalize + extract mantissa + guard + sticky
    let msb = bigint_bit_len(&m_big) as i64 - 1;
    if msb < 0 {
        let bits: u128 = if sign { 1u128 << 127 } else { 0 };
        return Some(bits.to_le_bytes());
    }
    let unbiased_exp = e2 + msb;

    let (bits113, guard, sticky, final_exp): (u128, bool, bool, i64) =
        if unbiased_exp >= -16382 {
            // Normal number: mantissa is 113 bits (the msb is the implicit 1)
            let shift = msb - 112;
            if shift >= 0 {
                let s = shift as u32;
                let mant = bigint_extract_bits(&m_big, s, 113);
                let g = bigint_bit(&m_big, (shift - 1) as i64);
                let stk = if s >= 2 {
                    bigint_low_nonzero(&m_big, s - 1)
                } else {
                    false
                };
                (mant, g, stk || div_sticky, unbiased_exp)
            } else {
                // Left-shift to fill; M is represented exactly (no guard)
                let mut m = m_big.clone();
                bigint_shl(&mut m, (-shift) as u32);
                let mant = bigint_to_u128(&m) & ((1u128 << 113) - 1);
                (mant, false, div_sticky, unbiased_exp)
            }
        } else {
            // Subnormal number: fraction is 112 bits, exp is fixed at -16382
            // fraction = M * 2^(e2 + 16494)
            let p = e2 + 16494;
            if p >= 0 {
                let mut m = m_big.clone();
                bigint_shl(&mut m, p as u32);
                let frac = bigint_to_u128(&m) & ((1u128 << 112) - 1);
                (frac, false, div_sticky, -16382)
            } else {
                let s = (-p) as u32;
                let frac = bigint_extract_bits(&m_big, s, 112);
                let g = bigint_bit(&m_big, (-p - 1) as i64);
                let stk = if s >= 2 {
                    bigint_low_nonzero(&m_big, s - 1)
                } else {
                    false
                };
                (frac, g, stk || div_sticky, -16382)
            }
        };

    // 6. Round to nearest even
    let mut mant = bits113;
    let mut exp = final_exp;
    // `final_exp` is clamped to -16382 inside the subnormal branch, so the
    // subnormal test must use the pre-clamp unbiased exponent.
    let was_subnormal = unbiased_exp < -16382;
    if guard && (sticky || (mant & 1) != 0) {
        mant += 1;
    }
    if was_subnormal {
        // Subnormal rounding may carry up to the smallest normal (mant reaches 2^112).
        // Without the carry the exponent field must stay 0 (pure subnormal
        // encoding), so park exp below the normal threshold — otherwise the
        // normal assembly below would emit exponent field 1 and inflate every
        // subnormal literal to the 2^-16382 binade.
        if mant >= (1u128 << 112) {
            exp = -16382;
        } else {
            exp = -16383;
        }
    } else if mant >= (1u128 << 113) {
        // Normal number rounding carry
        mant >>= 1;
        exp += 1;
    }

    // 7. Assemble binary128
    if exp > 16383 {
        let bits: u128 = (if sign { 1u128 << 127 } else { 0 }) | (0x7FFFu128 << 112);
        return Some(bits.to_le_bytes());
    }
    if exp >= -16382 {
        // Normal number
        let frac = mant & ((1u128 << 112) - 1);
        let biased = (exp + 16383) as u128;
        let bits = (if sign { 1u128 << 127 } else { 0 }) | (biased << 112) | frac;
        return Some(bits.to_le_bytes());
    }
    // Subnormal number
    let frac = mant & ((1u128 << 112) - 1);
    let bits = (if sign { 1u128 << 127 } else { 0 }) | frac;
    Some(bits.to_le_bytes())
}

/// Resolve a TypeRef to a type-name string using the given arena.
/// Used by build_abi_sig to resolve types in the correct module's arena.
pub(super) fn type_name_in_arena(
    ty: Option<crate::ast::Ast::TypeRef>,
    arena: &crate::ast::Ast::AstArena<'_>,
) -> String {
    use crate::ast::Ast::TypeNode;
    let ty_ref = match ty {
        Some(t) => t,
        None => return String::new(),
    };
    let node = match arena.types.get(ty_ref.0 as usize) {
        Some(n) => n,
        None => return String::new(),
    };
    match &node.node {
        TypeNode::Named { name } => (*name).to_string(),
        TypeNode::RawPtr { inner } => {
            let inner_name = type_name_in_arena(Some(*inner), arena);
            format!("*{inner_name}")
        }
        TypeNode::Array { element_type, size } => {
            let elem_name = type_name_in_arena(Some(*element_type), arena);
            if size.is_none() {
                format!("{elem_name}[]")
            } else {
                elem_name
            }
        }
        _ => String::new(),
    }
}
