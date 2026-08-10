// =========================================================================
// Ops — TypeOps trait + scalar/reference ops implementations + DynamicOpsRegistry.
// =========================================================================
//
// The TypeOps trait is kept separate from Ty: Ty is type identity (Copy data), while
// TypeOps covers runtime value operations. The TypeDescriptor struct has been removed;
// static DESC constants are no longer generated.

use super::Tag::*;
use crate::value::{Char, F128, F16, ValueArena, ValueHandle};

/// Type operations trait: describes the conversion semantics between a raw byte buffer
/// and a `ValueArena` handle for a given type. All methods are hot paths and
/// implementations should keep `#[inline]`.
///
/// Implementors require `Send + Sync + 'static` so that ops can be constructed
/// statically and shared across threads.
pub trait TypeOps: Send + Sync + 'static {
    /// Read a value from `ptr`, allocate it into `arena`, and return the handle.
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle;
    /// Write the value corresponding to handle `v` into `ptr`.
    fn write(&self, ptr: *mut u8, v: ValueHandle, arena: &ValueArena);
    /// Coerce an arbitrary handle `v` into a handle of this type.
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle;
    /// Compare two in-memory values for equality.
    fn equal(&self, a: *const u8, b: *const u8) -> bool;
    /// Format the value at `ptr` into `buf`, returning the written `&str` slice.
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str;
    /// Compute the hash of the value.
    fn hash_val(&self, ptr: *const u8) -> u64;
    /// Clone the value (scalars are value-semantic, so this is equivalent to `read`).
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle;
}

// =========================================================================
// Scalar ops macro generation (14 small-size scalars).
// =========================================================================
//
// Macro parameters: ops name, Rust native type, alloc method name, get method name,
// fmt expression, coerce kind.
//
// This macro generates a ZST ops struct and its `TypeOps` implementation. i128 / u128 /
// f128 / bool are manually implemented due to special semantics. TypeDescriptor static
// constants are no longer generated (TypeDescriptor has been removed).

macro_rules! impl_scalar_ops {
    // Internal arm: generates read/write/format/hash_val/clone_val (excludes equal).
    // The responsibility of these methods is to operate on typed memory behind raw
    // pointers; clippy's `not_unsafe_ptr_arg_deref` is a false positive here, so we
    // allow it uniformly.
    (@fns_core $ty:ty, $alloc:ident, $get:ident, [$v:ident => $fmt:expr]) => {
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
            // SAFETY: ptr points to valid $ty memory; read unaligned.
            let $v: $ty = unsafe { std::ptr::read_unaligned(ptr as *const $ty) };
            arena.$alloc($v)
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
            let $v: $ty = arena.$get(h);
            // SAFETY: ptr points to writable $ty memory; write unaligned.
            unsafe { std::ptr::write_unaligned(ptr as *mut $ty, $v) }
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
            // SAFETY: ptr points to a valid $ty value.
            let $v: $ty = unsafe { std::ptr::read_unaligned(ptr as *const $ty) };
            use std::io::Write;
            let mut cursor = std::io::Cursor::new(buf);
            let _ = write!(cursor, "{}", $fmt);
            let written = cursor.position() as usize;
            let buf_ref: &mut [u8] = cursor.into_inner();
            // SAFETY: Display output of native numeric/char types is ASCII or valid UTF-8.
            unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn hash_val(&self, ptr: *const u8) -> u64 {
            use std::hash::{Hash, Hasher};
            // SAFETY: ptr points to a valid $ty value; hash by its byte representation
            // (for types like f32/f64 that do not implement Hash, hash by bit pattern).
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<$ty>()) };
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
            // SAFETY: scalars are value-semantic; clone is equivalent to read.
            let $v: $ty = unsafe { std::ptr::read_unaligned(ptr as *const $ty) };
            arena.$alloc($v)
        }
    };

    // Full @fns = @fns_core + default equal (bit-pattern comparison).
    // The native == for f32/f64 already implements IEEE semantics (NaN != NaN, -0 == +0);
    // == for integers/bool/char is value equality. f16 is stored as u16; for IEEE
    // semantics, use the coerce=f16 branch.
    (@fns $ty:ty, $alloc:ident, $get:ident, [$v:ident => $fmt:expr]) => {
        impl_scalar_ops!(@fns_core $ty, $alloc, $get, [$v => $fmt]);
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn equal(&self, a: *const u8, b: *const u8) -> bool {
            // SAFETY: both pointers point to valid $ty values.
            unsafe {
                std::ptr::read_unaligned(a as *const $ty)
                    == std::ptr::read_unaligned(b as *const $ty)
            }
        }
    };

    // Integer target type
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=int
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                let val: $ty = match v.tag() {
                    ValueTag::Bool => arena.get_bool(v) as $ty,
                    ValueTag::Char => arena.get_char(v) as $ty,
                    ValueTag::I8 => arena.get_i8(v) as $ty,
                    ValueTag::I16 => arena.get_i16(v) as $ty,
                    ValueTag::I32 => arena.get_i32(v) as $ty,
                    ValueTag::I64 => arena.get_i64(v) as $ty,
                    ValueTag::I128 => arena.get_i128(v) as $ty,
                    ValueTag::U8 => arena.get_u8(v) as $ty,
                    ValueTag::U16 => arena.get_u16(v) as $ty,
                    ValueTag::U32 => arena.get_u32(v) as $ty,
                    ValueTag::U64 => arena.get_u64(v) as $ty,
                    ValueTag::U128 => arena.get_u128(v) as $ty,
                    ValueTag::Isize => arena.get_isize(v) as $ty,
                    ValueTag::Usize => arena.get_usize(v) as $ty,
                    ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as $ty,
                    ValueTag::F32 => arena.get_f32(v) as $ty,
                    ValueTag::F64 => arena.get_f64(v) as $ty,
                    ValueTag::F128 => arena.get_f128(v).to_f64() as $ty,
                    ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0 as $ty,
                };
                arena.$alloc(val)
            }
        }
    };

    // Float target type (f32 / f64)
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=float
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                let val: $ty = match v.tag() {
                    ValueTag::Bool => arena.get_bool(v) as u8 as $ty,
                    ValueTag::Char => arena.get_char(v) as u32 as $ty,
                    ValueTag::I8 => arena.get_i8(v) as $ty,
                    ValueTag::I16 => arena.get_i16(v) as $ty,
                    ValueTag::I32 => arena.get_i32(v) as $ty,
                    ValueTag::I64 => arena.get_i64(v) as $ty,
                    ValueTag::I128 => arena.get_i128(v) as $ty,
                    ValueTag::U8 => arena.get_u8(v) as $ty,
                    ValueTag::U16 => arena.get_u16(v) as $ty,
                    ValueTag::U32 => arena.get_u32(v) as $ty,
                    ValueTag::U64 => arena.get_u64(v) as $ty,
                    ValueTag::U128 => arena.get_u128(v) as $ty,
                    ValueTag::Isize => arena.get_isize(v) as $ty,
                    ValueTag::Usize => arena.get_usize(v) as $ty,
                    ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as $ty,
                    ValueTag::F32 => arena.get_f32(v) as $ty,
                    ValueTag::F64 => arena.get_f64(v) as $ty,
                    ValueTag::F128 => arena.get_f128(v).to_f64() as $ty,
                    ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0.0 as $ty,
                };
                arena.$alloc(val)
            }
        }
    };

    // f16 target type
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=f16
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns_core $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            fn equal(&self, a: *const u8, b: *const u8) -> bool {
                // IEEE 754 semantics: NaN != NaN, -0 == +0 (consistent with f32/f64/f128 equal).
                // F16 is stored as a u16 bit pattern; using u16 == u16 directly would give
                // NaN == NaN and -0 != +0.
                unsafe {
                    let x = std::ptr::read_unaligned(a as *const u16);
                    let y = std::ptr::read_unaligned(b as *const u16);
                    // NaN check: exponent all 1s (0x7C00) and mantissa nonzero (0x03FF).
                    let x_nan = (x & 0x7C00) == 0x7C00 && (x & 0x03FF) != 0;
                    let y_nan = (y & 0x7C00) == 0x7C00 && (y & 0x03FF) != 0;
                    if x_nan || y_nan {
                        return false;
                    }
                    // -0 == +0: when both exponent and mantissa are zero, treat as equal.
                    x == y || (x | y) & 0x7FFF == 0
                }
            }
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                let f: f32 = match v.tag() {
                    ValueTag::Bool => arena.get_bool(v) as u8 as f32,
                    ValueTag::Char => arena.get_char(v) as u32 as f32,
                    ValueTag::I8 => arena.get_i8(v) as f32,
                    ValueTag::I16 => arena.get_i16(v) as f32,
                    ValueTag::I32 => arena.get_i32(v) as f32,
                    ValueTag::I64 => arena.get_i64(v) as f32,
                    ValueTag::I128 => arena.get_i128(v) as f32,
                    ValueTag::U8 => arena.get_u8(v) as f32,
                    ValueTag::U16 => arena.get_u16(v) as f32,
                    ValueTag::U32 => arena.get_u32(v) as f32,
                    ValueTag::U64 => arena.get_u64(v) as f32,
                    ValueTag::U128 => arena.get_u128(v) as f32,
                    ValueTag::Isize => arena.get_isize(v) as f32,
                    ValueTag::Usize => arena.get_usize(v) as f32,
                    ValueTag::F16 => F16(arena.get_f16(v)).to_f32(),
                    ValueTag::F32 => arena.get_f32(v),
                    ValueTag::F64 => arena.get_f64(v) as f32,
                    ValueTag::F128 => arena.get_f128(v).to_f64() as f32,
                    ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0.0,
                };
                arena.$alloc(F16::from_f32(f).0)
            }
        }
    };

    // f128 target type
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=f128
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                // i128/u128 precision exceeds f64 (53 bits); use F128::from_i128/from_u128 for
                // exact construction. F128 sources are returned as-is (no loss); all others
                // go through f64 (values are within f64 precision, lossless).
                match v.tag() {
                    ValueTag::I128 => return arena.$alloc(F128::from_i128(arena.get_i128(v))),
                    ValueTag::U128 => return arena.$alloc(F128::from_u128(arena.get_u128(v))),
                    ValueTag::F128 => return v,
                    _ => {}
                }
                let f: f64 = match v.tag() {
                    ValueTag::Bool => arena.get_bool(v) as u8 as f64,
                    ValueTag::Char => arena.get_char(v) as u64 as f64,
                    ValueTag::I8 => arena.get_i8(v) as f64,
                    ValueTag::I16 => arena.get_i16(v) as f64,
                    ValueTag::I32 => arena.get_i32(v) as f64,
                    ValueTag::I64 => arena.get_i64(v) as f64,
                    ValueTag::U8 => arena.get_u8(v) as f64,
                    ValueTag::U16 => arena.get_u16(v) as f64,
                    ValueTag::U32 => arena.get_u32(v) as f64,
                    ValueTag::U64 => arena.get_u64(v) as f64,
                    ValueTag::Isize => arena.get_isize(v) as f64,
                    ValueTag::Usize => arena.get_usize(v) as f64,
                    ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as f64,
                    ValueTag::F32 => arena.get_f32(v) as f64,
                    ValueTag::F64 => arena.get_f64(v),
                    ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0.0,
                    // I128/U128/F128 were early-returned above and never reach here.
                    ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => unreachable!(),
                };
                arena.$alloc(F128::from_f64(f))
            }
        }
    };

    // bool target type
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=bool
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                let b: bool = match v.tag() {
                    ValueTag::Bool => arena.get_bool(v),
                    ValueTag::Char => arena.get_char(v) != 0,
                    ValueTag::I8 => arena.get_i8(v) != 0,
                    ValueTag::I16 => arena.get_i16(v) != 0,
                    ValueTag::I32 => arena.get_i32(v) != 0,
                    ValueTag::I64 => arena.get_i64(v) != 0,
                    ValueTag::I128 => arena.get_i128(v) != 0,
                    ValueTag::U8 => arena.get_u8(v) != 0,
                    ValueTag::U16 => arena.get_u16(v) != 0,
                    ValueTag::U32 => arena.get_u32(v) != 0,
                    ValueTag::U64 => arena.get_u64(v) != 0,
                    ValueTag::U128 => arena.get_u128(v) != 0,
                    ValueTag::Isize => arena.get_isize(v) != 0,
                    ValueTag::Usize => arena.get_usize(v) != 0,
                    ValueTag::F16 => F16(arena.get_f16(v)).to_f32() != 0.0,
                    ValueTag::F32 => arena.get_f32(v) != 0.0,
                    ValueTag::F64 => arena.get_f64(v) != 0.0,
                    ValueTag::F128 => arena.get_f128(v).to_f64() != 0.0,
                    ValueTag::Null | ValueTag::Void | ValueTag::Ref => false,
                };
                arena.$alloc(b)
            }
        }
    };

    // char target type
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=char
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                let c: u32 = match v.tag() {
                    ValueTag::Bool => arena.get_bool(v) as u32,
                    ValueTag::Char => arena.get_char(v),
                    ValueTag::I8 => arena.get_i8(v) as u32,
                    ValueTag::I16 => arena.get_i16(v) as u32,
                    ValueTag::I32 => arena.get_i32(v) as u32,
                    ValueTag::I64 => arena.get_i64(v) as u32,
                    ValueTag::I128 => arena.get_i128(v) as u32,
                    ValueTag::U8 => arena.get_u8(v) as u32,
                    ValueTag::U16 => arena.get_u16(v) as u32,
                    ValueTag::U32 => arena.get_u32(v),
                    ValueTag::U64 => arena.get_u64(v) as u32,
                    ValueTag::U128 => arena.get_u128(v) as u32,
                    ValueTag::Isize => arena.get_isize(v) as u32,
                    ValueTag::Usize => arena.get_usize(v) as u32,
                    ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as u32,
                    ValueTag::F32 => arena.get_f32(v) as u32,
                    ValueTag::F64 => arena.get_f64(v) as u32,
                    ValueTag::F128 => arena.get_f128(v).to_f64() as u32,
                    ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0,
                };
                arena.$alloc(c)
            }
        }
    };
}

// Macro instantiation for the 14 small-size scalars (generates ops structs and TypeOps impls).
impl_scalar_ops!(I8Ops, i8, alloc=alloc_i8, get=get_i8, fmt(v) => v, coerce=int);
impl_scalar_ops!(I16Ops, i16, alloc=alloc_i16, get=get_i16, fmt(v) => v, coerce=int);
impl_scalar_ops!(I32Ops, i32, alloc=alloc_i32, get=get_i32, fmt(v) => v, coerce=int);
impl_scalar_ops!(I64Ops, i64, alloc=alloc_i64, get=get_i64, fmt(v) => v, coerce=int);
impl_scalar_ops!(U8Ops, u8, alloc=alloc_u8, get=get_u8, fmt(v) => v, coerce=int);
impl_scalar_ops!(U16Ops, u16, alloc=alloc_u16, get=get_u16, fmt(v) => v, coerce=int);
impl_scalar_ops!(U32Ops, u32, alloc=alloc_u32, get=get_u32, fmt(v) => v, coerce=int);
impl_scalar_ops!(U64Ops, u64, alloc=alloc_u64, get=get_u64, fmt(v) => v, coerce=int);
impl_scalar_ops!(IsizeOps, isize, alloc=alloc_isize, get=get_isize, fmt(v) => v, coerce=int);
impl_scalar_ops!(UsizeOps, usize, alloc=alloc_usize, get=get_usize, fmt(v) => v, coerce=int);
impl_scalar_ops!(F16Ops, u16, alloc=alloc_f16, get=get_f16, fmt(v) => F16(v).to_f32(), coerce=f16);
impl_scalar_ops!(F32Ops, f32, alloc=alloc_f32, get=get_f32, fmt(v) => v, coerce=float);
impl_scalar_ops!(F64Ops, f64, alloc=alloc_f64, get=get_f64, fmt(v) => v, coerce=float);
impl_scalar_ops!(CharOps, u32, alloc=alloc_char, get=get_char, fmt(v) => Char::from_codepoint_unchecked(v), coerce=char);

// =========================================================================
// Manual implementations: i128 / u128 / f128 / bool.
// =========================================================================

/// `TypeOps` implementation for i128 (16 bytes; `ValueArena` returns i128).
pub struct I128Ops;
impl TypeOps for I128Ops {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr points to a valid 16-byte i128.
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        arena.alloc_i128(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: i128 = arena.get_i128(h);
        // SAFETY: ptr points to writable 16-byte i128 memory.
        unsafe { std::ptr::write_unaligned(ptr as *mut i128, v) }
    }
    #[inline]
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
        let val: i128 = match v.tag() {
            ValueTag::Bool => arena.get_bool(v) as i128,
            ValueTag::Char => arena.get_char(v) as i128,
            ValueTag::I8 => arena.get_i8(v) as i128,
            ValueTag::I16 => arena.get_i16(v) as i128,
            ValueTag::I32 => arena.get_i32(v) as i128,
            ValueTag::I64 => arena.get_i64(v) as i128,
            ValueTag::I128 => arena.get_i128(v),
            ValueTag::U8 => arena.get_u8(v) as i128,
            ValueTag::U16 => arena.get_u16(v) as i128,
            ValueTag::U32 => arena.get_u32(v) as i128,
            ValueTag::U64 => arena.get_u64(v) as i128,
            ValueTag::U128 => arena.get_u128(v) as i128,
            ValueTag::Isize => arena.get_isize(v) as i128,
            ValueTag::Usize => arena.get_usize(v) as i128,
            ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as i128,
            ValueTag::F32 => arena.get_f32(v) as i128,
            ValueTag::F64 => arena.get_f64(v) as i128,
            ValueTag::F128 => arena.get_f128(v).to_f64() as i128,
            ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0,
        };
        arena.alloc_i128(val)
    }
    #[inline]
    fn equal(&self, a: *const u8, b: *const u8) -> bool {
        // SAFETY: both pointers point to valid i128 values.
        unsafe {
            std::ptr::read_unaligned(a as *const i128)
                == std::ptr::read_unaligned(b as *const i128)
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr points to a valid i128.
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: i128 Display output is ASCII decimal.
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr points to a valid i128.
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: scalars are value-semantic; clone is equivalent to read.
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        arena.alloc_i128(v)
    }
}

/// `TypeOps` implementation for u128 (16 bytes; `ValueArena` returns u128).
pub struct U128Ops;
impl TypeOps for U128Ops {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr points to a valid 16-byte u128.
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        arena.alloc_u128(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: u128 = arena.get_u128(h);
        // SAFETY: ptr points to writable 16-byte u128 memory.
        unsafe { std::ptr::write_unaligned(ptr as *mut u128, v) }
    }
    #[inline]
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
        let val: u128 = match v.tag() {
            ValueTag::Bool => arena.get_bool(v) as u128,
            ValueTag::Char => arena.get_char(v) as u128,
            ValueTag::I8 => arena.get_i8(v) as u128,
            ValueTag::I16 => arena.get_i16(v) as u128,
            ValueTag::I32 => arena.get_i32(v) as u128,
            ValueTag::I64 => arena.get_i64(v) as u128,
            ValueTag::I128 => arena.get_i128(v) as u128,
            ValueTag::U8 => arena.get_u8(v) as u128,
            ValueTag::U16 => arena.get_u16(v) as u128,
            ValueTag::U32 => arena.get_u32(v) as u128,
            ValueTag::U64 => arena.get_u64(v) as u128,
            ValueTag::U128 => arena.get_u128(v),
            ValueTag::Isize => arena.get_isize(v) as u128,
            ValueTag::Usize => arena.get_usize(v) as u128,
            ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as u128,
            ValueTag::F32 => arena.get_f32(v) as u128,
            ValueTag::F64 => arena.get_f64(v) as u128,
            ValueTag::F128 => arena.get_f128(v).to_f64() as u128,
            ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0,
        };
        arena.alloc_u128(val)
    }
    #[inline]
    fn equal(&self, a: *const u8, b: *const u8) -> bool {
        // SAFETY: both pointers point to valid u128 values.
        unsafe {
            std::ptr::read_unaligned(a as *const u128)
                == std::ptr::read_unaligned(b as *const u128)
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr points to a valid u128.
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: u128 Display output is ASCII decimal.
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr points to a valid u128.
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: scalars are value-semantic; clone is equivalent to read.
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        arena.alloc_u128(v)
    }
}

/// `TypeOps` implementation for f128 (16 bytes; `ValueArena` returns `F128`).
pub struct F128Ops;
impl TypeOps for F128Ops {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr points to a valid 16-byte F128 (Copy).
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        arena.alloc_f128(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: F128 = arena.get_f128(h);
        // SAFETY: ptr points to writable 16-byte F128 memory.
        unsafe { std::ptr::write_unaligned(ptr as *mut F128, v) }
    }
    #[inline]
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
        // i128/u128 would lose precision via `as f64`; use from_i128/from_u128 for exact
        // construction. F128 sources are returned as-is.
        match v.tag() {
            ValueTag::I128 => return arena.alloc_f128(F128::from_i128(arena.get_i128(v))),
            ValueTag::U128 => return arena.alloc_f128(F128::from_u128(arena.get_u128(v))),
            ValueTag::F128 => return v,
            _ => {}
        }
        let f: f64 = match v.tag() {
            ValueTag::Bool => arena.get_bool(v) as u8 as f64,
            ValueTag::Char => arena.get_char(v) as u64 as f64,
            ValueTag::I8 => arena.get_i8(v) as f64,
            ValueTag::I16 => arena.get_i16(v) as f64,
            ValueTag::I32 => arena.get_i32(v) as f64,
            ValueTag::I64 => arena.get_i64(v) as f64,
            ValueTag::U8 => arena.get_u8(v) as f64,
            ValueTag::U16 => arena.get_u16(v) as f64,
            ValueTag::U32 => arena.get_u32(v) as f64,
            ValueTag::U64 => arena.get_u64(v) as f64,
            ValueTag::Isize => arena.get_isize(v) as f64,
            ValueTag::Usize => arena.get_usize(v) as f64,
            ValueTag::F16 => F16(arena.get_f16(v)).to_f32() as f64,
            ValueTag::F32 => arena.get_f32(v) as f64,
            ValueTag::F64 => arena.get_f64(v),
            ValueTag::Null | ValueTag::Void | ValueTag::Ref => 0.0,
            ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => unreachable!(),
        };
        arena.alloc_f128(F128::from_f64(f))
    }
    #[inline]
    fn equal(&self, a: *const u8, b: *const u8) -> bool {
        // IEEE 754 semantics: NaN != NaN, -0 == +0 (consistent with f32/f64 equal).
        unsafe {
            let x = std::ptr::read_unaligned(a as *const F128);
            let y = std::ptr::read_unaligned(b as *const F128);
            if x.is_nan() || y.is_nan() {
                return false;
            }
            // -0 == +0: treat as equal when bit patterns differ only in the sign bit.
            let xb = u128::from_le_bytes(x.0);
            let yb = u128::from_le_bytes(y.0);
            xb == yb || (xb | yb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr points to a valid F128.
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: F128 Display output is ASCII.
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr points to a valid F128.
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: scalars are value-semantic; clone is equivalent to read.
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        arena.alloc_f128(v)
    }
}

/// `TypeOps` implementation for bool (`ValueArena::bool` and `get_bool` both have
/// `&self` singleton semantics).
pub struct BoolOps;
impl TypeOps for BoolOps {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr points to a valid 1-byte bool.
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        arena.bool(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: bool = arena.get_bool(h);
        // SAFETY: ptr points to writable 1-byte bool memory.
        unsafe { std::ptr::write_unaligned(ptr as *mut bool, v) }
    }
    #[inline]
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
        let b: bool = match v.tag() {
            ValueTag::Bool => arena.get_bool(v),
            ValueTag::Char => arena.get_char(v) != 0,
            ValueTag::I8 => arena.get_i8(v) != 0,
            ValueTag::I16 => arena.get_i16(v) != 0,
            ValueTag::I32 => arena.get_i32(v) != 0,
            ValueTag::I64 => arena.get_i64(v) != 0,
            ValueTag::I128 => arena.get_i128(v) != 0,
            ValueTag::U8 => arena.get_u8(v) != 0,
            ValueTag::U16 => arena.get_u16(v) != 0,
            ValueTag::U32 => arena.get_u32(v) != 0,
            ValueTag::U64 => arena.get_u64(v) != 0,
            ValueTag::U128 => arena.get_u128(v) != 0,
            ValueTag::Isize => arena.get_isize(v) != 0,
            ValueTag::Usize => arena.get_usize(v) != 0,
            ValueTag::F16 => F16(arena.get_f16(v)).to_f32() != 0.0,
            ValueTag::F32 => arena.get_f32(v) != 0.0,
            ValueTag::F64 => arena.get_f64(v) != 0.0,
            ValueTag::F128 => arena.get_f128(v).to_f64() != 0.0,
            ValueTag::Null | ValueTag::Void | ValueTag::Ref => false,
        };
        arena.bool(b)
    }
    #[inline]
    fn equal(&self, a: *const u8, b: *const u8) -> bool {
        // SAFETY: both pointers point to valid bool values.
        unsafe {
            std::ptr::read_unaligned(a as *const bool)
                == std::ptr::read_unaligned(b as *const bool)
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr points to a valid bool.
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: bool Display output is ASCII ("true"/"false").
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr points to a valid bool.
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: scalars are value-semantic; clone is equivalent to read.
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        arena.bool(v)
    }
}

// =========================================================================
// Null / Void ops (simplified implementations).
// =========================================================================

/// `TypeOps` implementation for the null type (no data, singleton semantics).
pub struct NullOps;
impl TypeOps for NullOps {
    #[inline]
    fn read(&self, _ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        arena.null()
    }
    #[inline]
    fn write(&self, _ptr: *mut u8, _v: ValueHandle, _arena: &ValueArena) {}
    #[inline]
    fn coerce(&self, v: ValueHandle, _arena: &mut ValueArena) -> ValueHandle {
        v
    }
    #[inline]
    fn equal(&self, _a: *const u8, _b: *const u8) -> bool {
        // null is a singleton; all instances are equal (same type_id).
        true
    }
    #[inline]
    fn format<'a>(&self, _ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        if buf.len() < 4 {
            return "";
        }
        // SAFETY: "null" is valid ASCII.
        buf[..4].copy_from_slice(b"null");
        unsafe { std::str::from_utf8_unchecked(&buf[..4]) }
    }
    #[inline]
    fn hash_val(&self, _ptr: *const u8) -> u64 {
        0
    }
    #[inline]
    fn clone_val(&self, _ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        arena.null()
    }
}

/// `TypeOps` implementation for the void type (no data, singleton semantics).
pub struct VoidOps;
impl TypeOps for VoidOps {
    #[inline]
    fn read(&self, _ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        arena.void()
    }
    #[inline]
    fn write(&self, _ptr: *mut u8, _v: ValueHandle, _arena: &ValueArena) {}
    #[inline]
    fn coerce(&self, v: ValueHandle, _arena: &mut ValueArena) -> ValueHandle {
        v
    }
    #[inline]
    fn equal(&self, _a: *const u8, _b: *const u8) -> bool {
        // void is a singleton; all instances are equal (same type_id).
        true
    }
    #[inline]
    fn format<'a>(&self, _ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        if buf.len() < 4 {
            return "";
        }
        // SAFETY: "void" is valid ASCII.
        buf[..4].copy_from_slice(b"void");
        unsafe { std::str::from_utf8_unchecked(&buf[..4]) }
    }
    #[inline]
    fn hash_val(&self, _ptr: *const u8) -> u64 {
        0
    }
    #[inline]
    fn clone_val(&self, _ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        arena.void()
    }
}

// =========================================================================
// ref_ops / heap_ref_ops (reference type ops, simplified implementations).
// =========================================================================
//
// The full ref implementation (msync checks, ObjHeader validation, heap object
// materialization) is the responsibility of the engine layer. The shared layer only
// provides a read/write skeleton for the 8-byte pointer slot: 0 denotes null, and a
// nonzero address returns null on read (preserving null-paired semantics); write
// stores the handle index into the slot as a placeholder.

macro_rules! impl_ref_ops {
    ($ops:ident) => {
        pub struct $ops;
        impl TypeOps for $ops {
            #[inline]
            fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
                // SAFETY: ptr points to an 8-byte address slot (0 denotes null).
                let addr: usize = unsafe { std::ptr::read_unaligned(ptr as *const usize) };
                if addr == 0 {
                    arena.null()
                } else {
                    // Full ref materialization requires the engine layer (ObjHeader /
                    // msync validation); the shared layer preserves null-paired
                    // semantics and returns null for nonzero addresses.
                    arena.null()
                }
            }
            #[inline]
            fn write(&self, ptr: *mut u8, h: ValueHandle, _arena: &ValueArena) {
                // SAFETY: ptr points to a writable 8-byte address slot; store the handle
                // index as a placeholder.
                let raw: usize = h.index();
                unsafe { std::ptr::write_unaligned(ptr as *mut usize, raw) }
            }
            #[inline]
            fn coerce(&self, v: ValueHandle, _arena: &mut ValueArena) -> ValueHandle {
                v
            }
            #[inline]
            fn equal(&self, a: *const u8, b: *const u8) -> bool {
                // SAFETY: both pointers point to 8-byte address slots.
                unsafe {
                    std::ptr::read_unaligned(a as *const usize)
                        == std::ptr::read_unaligned(b as *const usize)
                }
            }
            #[inline]
            fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
                // SAFETY: ptr points to an 8-byte address.
                let addr: usize = unsafe { std::ptr::read_unaligned(ptr as *const usize) };
                use std::io::Write;
                let mut cursor = std::io::Cursor::new(buf);
                let _ = write!(cursor, "ref:0x{:x}", addr);
                let written = cursor.position() as usize;
                let buf_ref: &mut [u8] = cursor.into_inner();
                // SAFETY: the formatted text is ASCII.
                unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
            }
            #[inline]
            fn hash_val(&self, ptr: *const u8) -> u64 {
                // SAFETY: ptr points to an 8-byte address; reinterpret as u64.
                unsafe { std::ptr::read_unaligned(ptr as *const u64) }
            }
            #[inline]
            fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
                self.read(ptr, arena)
            }
        }
    };
}

impl_ref_ops!(RefOps);
impl_ref_ops!(HeapRefOps);

// =========================================================================
// type_id constants (migrated from TypeDesc.rs, authoritative source).
// =========================================================================

/// Builtin scalar type ID range: 1..=21 (21 types in total).
/// type_id 0 is reserved for "unknown"; 22+ are user/dynamic types (type_def_index offset).
pub const FIRST_DYNAMIC_TYPE_ID: u16 = 22;

/// type_id for str/null/void (consistent with BUILTIN_TABLE).
pub const STR_TYPE_ID: u16 = 19;
pub const NULL_TYPE_ID: u16 = 20;
pub const VOID_TYPE_ID: u16 = 21;

/// Integer type type_id range: 1..=12.
pub const FIRST_INT_TYPE_ID: u16 = 1;
pub const LAST_INT_TYPE_ID: u16 = 12;
/// Float type type_id range: 13..=16.
pub const FIRST_FLOAT_TYPE_ID: u16 = 13;
pub const LAST_FLOAT_TYPE_ID: u16 = 16;

/// Convert a `type_def_index` to a dynamic `type_id`.
#[inline]
pub const fn dynamic_type_id(type_def_index: u16) -> u16 {
    FIRST_DYNAMIC_TYPE_ID + type_def_index
}

/// Convert a dynamic `type_id` back to a `type_def_index`.
/// Only valid for type_id >= FIRST_DYNAMIC_TYPE_ID.
#[inline]
pub const fn type_def_index_of(type_id: u16) -> u16 {
    type_id - FIRST_DYNAMIC_TYPE_ID
}

// =========================================================================
// DynamicOpsRegistry (replaces TypeDescriptorPool).
// =========================================================================
//
// Manages ops for user-defined types (type_id starts at FIRST_DYNAMIC_TYPE_ID=22).
// The TypeDescriptor struct has been removed; user type ops are registered directly as
// `&'static dyn TypeOps`.

/// Dynamic type ops entry: stores the ops + size + name + type_id for a user-defined type.
///
/// Dynamic type ops registry: placeholder for user-defined type ops registration.
/// Currently no dynamic types are registered; methods will be added when user-defined
/// type registration is wired into sema.
pub struct DynamicOpsRegistry;

impl DynamicOpsRegistry {
    #[inline]
    pub fn new() -> Self {
        DynamicOpsRegistry
    }
}

impl Default for DynamicOpsRegistry {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
