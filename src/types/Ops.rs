// =========================================================================
// Ops — TypeOps trait + 标量/引用 ops 实现 + ops 查找表 + DynamicOpsRegistry
// =========================================================================
//
// TypeOps trait 与 Ty 分离：Ty 是类型身份（Copy 数据），TypeOps 是运行时值操作。
// ops 查找表（ops_of/ops_by_type_id）在下方提供。
// TypeDescriptor 结构体已删除，静态 DESC 常量不再生成。

use super::Tag::*;
use super::ty::*;
use crate::value::{Char, F128, F16, ValueArena, ValueHandle};
use rustc_hash::FxHashMap;

/// 类型操作 trait：描述某种类型在原始字节缓冲区与 `ValueArena` 句柄之间的
/// 转换语义。所有方法均为热路径，实现应保持 `#[inline]`。
///
/// 实现者要求 `Send + Sync + 'static`，以便 ops 可被静态构造并跨线程共享。
pub trait TypeOps: Send + Sync + 'static {
    /// 从 `ptr` 读取一个值，分配到 `arena` 并返回句柄。
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle;
    /// 将句柄 `v` 对应的值写入 `ptr`。
    fn write(&self, ptr: *mut u8, v: ValueHandle, arena: &ValueArena);
    /// 将任意句柄 `v` 强制转换为本类型句柄。
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle;
    /// 比较两块内存中的值是否相等。
    fn equal(&self, a: *const u8, b: *const u8) -> bool;
    /// 将 `ptr` 处的值格式化写入 `buf`，返回写入部分的 `&str` 切片。
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str;
    /// 计算值的哈希。
    fn hash_val(&self, ptr: *const u8) -> u64;
    /// 克隆值（标量为值语义，等价于 read）。
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle;
}

// =========================================================================
// 标量 ops 宏生成（14 种小尺寸标量）
// =========================================================================
//
// 宏参数：ops 名, Rust 原生类型, alloc 方法名, get 方法名, fmt 表达式, coerce 种类。
//
// 该宏生成 ZST ops 结构体与 `TypeOps` 实现。i128 / u128 / f128 / bool 因特殊语义
// 手动实现。TypeDescriptor 静态常量不再生成（TypeDescriptor 已删除）。

macro_rules! impl_scalar_ops {
    // 内部分支：生成 read/write/format/hash_val/clone_val 五个方法（不含 equal）。
    // 这些方法的职责就是操作裸指针指向的类型化内存，clippy 的
    // `not_unsafe_ptr_arg_deref` 在此为误报，统一 allow。
    (@fns_core $ty:ty, $alloc:ident, $get:ident, [$v:ident => $fmt:expr]) => {
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
            // SAFETY: ptr 指向合法的 $ty 值内存，按非对齐方式读取。
            let $v: $ty = unsafe { std::ptr::read_unaligned(ptr as *const $ty) };
            arena.$alloc($v)
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
            let $v: $ty = arena.$get(h);
            // SAFETY: ptr 指向可写的 $ty 内存，按非对齐方式写入。
            unsafe { std::ptr::write_unaligned(ptr as *mut $ty, $v) }
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
            // SAFETY: ptr 指向合法的 $ty 值。
            let $v: $ty = unsafe { std::ptr::read_unaligned(ptr as *const $ty) };
            use std::io::Write;
            let mut cursor = std::io::Cursor::new(buf);
            let _ = write!(cursor, "{}", $fmt);
            let written = cursor.position() as usize;
            let buf_ref: &mut [u8] = cursor.into_inner();
            // SAFETY: 原生数值/字符的 Display 输出为 ASCII 或合法 UTF-8。
            unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn hash_val(&self, ptr: *const u8) -> u64 {
            use std::hash::{Hash, Hasher};
            // SAFETY: ptr 指向合法的 $ty 值；按其字节表示哈希（对 f32/f64
            // 等未实现 Hash 的类型，统一按 bit pattern 哈希）。
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<$ty>()) };
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        }
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
            // SAFETY: 标量为值语义，clone 等价于 read。
            let $v: $ty = unsafe { std::ptr::read_unaligned(ptr as *const $ty) };
            arena.$alloc($v)
        }
    };

    // 完整 @fns = @fns_core + 默认 equal（bit-pattern 比较）。
    // f32/f64 的原生 == 已实现 IEEE 语义（NaN≠NaN，-0==+0）；
    // 整数/bool/char 的 == 即值相等。f16 存储为 u16，需 IEEE 语义时走 coerce=f16 分支。
    (@fns $ty:ty, $alloc:ident, $get:ident, [$v:ident => $fmt:expr]) => {
        impl_scalar_ops!(@fns_core $ty, $alloc, $get, [$v => $fmt]);
        #[inline]
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn equal(&self, a: *const u8, b: *const u8) -> bool {
            // SAFETY: 两个指针均指向合法的 $ty 值。
            unsafe {
                std::ptr::read_unaligned(a as *const $ty)
                    == std::ptr::read_unaligned(b as *const $ty)
            }
        }
    };

    // 整数目标类型
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

    // 浮点目标类型（f32 / f64）
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

    // f16 目标类型
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
                // IEEE 754 语义：NaN≠NaN，-0==+0（与 f32/f64/f128 的 equal 一致）。
                // F16 存储为 u16 bit pattern，不能用 u16 == u16（会得到 NaN==NaN、-0≠+0）。
                unsafe {
                    let x = std::ptr::read_unaligned(a as *const u16);
                    let y = std::ptr::read_unaligned(b as *const u16);
                    // NaN 判定：exponent 全 1 (0x7C00) 且 mantissa 非零 (0x03FF)
                    let x_nan = (x & 0x7C00) == 0x7C00 && (x & 0x03FF) != 0;
                    let y_nan = (y & 0x7C00) == 0x7C00 && (y & 0x03FF) != 0;
                    if x_nan || y_nan {
                        return false;
                    }
                    // -0 == +0：两者 exponent 与 mantissa 均为零时视为相等
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

    // f128 目标类型
    (
        $ops:ident, $ty:ty,
        alloc=$alloc:ident, get=$get:ident, fmt($v:ident) => $fmt:expr, coerce=f128
    ) => {
        pub struct $ops;
        impl TypeOps for $ops {
            impl_scalar_ops!(@fns $ty, $alloc, $get, [$v => $fmt]);
            #[inline]
            fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
                // i128/u128 精度超出 f64（53 位），用 F128::from_i128/from_u128 精确构造；
                // F128 源恒等返回，无精度损失；其余经 f64（值在 f64 精度内，无损）。
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
                    // I128/U128/F128 已在上方 early return，不会到达
                    ValueTag::I128 | ValueTag::U128 | ValueTag::F128 => unreachable!(),
                };
                arena.$alloc(F128::from_f64(f))
            }
        }
    };

    // bool 目标类型
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

    // char 目标类型
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

// 14 种小尺寸标量的宏实例化（生成 ops 结构体与 TypeOps 实现）。
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
// 手动实现：i128 / u128 / f128 / bool
// =========================================================================

/// `TypeOps` 实现：i128（16 字节，`ValueArena` 返回 i128）。
pub struct I128Ops;
impl TypeOps for I128Ops {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr 指向 16 字节合法 i128。
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        arena.alloc_i128(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: i128 = arena.get_i128(h);
        // SAFETY: ptr 指向可写 16 字节 i128。
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
        // SAFETY: 两指针均指向合法 i128。
        unsafe {
            std::ptr::read_unaligned(a as *const i128)
                == std::ptr::read_unaligned(b as *const i128)
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr 指向合法 i128。
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: i128 Display 输出为 ASCII 十进制。
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr 指向合法 i128。
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: 标量值语义，clone 等价于 read。
        let v: i128 = unsafe { std::ptr::read_unaligned(ptr as *const i128) };
        arena.alloc_i128(v)
    }
}

/// `TypeOps` 实现：u128（16 字节，`ValueArena` 返回 u128）。
pub struct U128Ops;
impl TypeOps for U128Ops {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr 指向 16 字节合法 u128。
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        arena.alloc_u128(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: u128 = arena.get_u128(h);
        // SAFETY: ptr 指向可写 16 字节 u128。
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
        // SAFETY: 两指针均指向合法 u128。
        unsafe {
            std::ptr::read_unaligned(a as *const u128)
                == std::ptr::read_unaligned(b as *const u128)
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr 指向合法 u128。
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: u128 Display 输出为 ASCII 十进制。
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr 指向合法 u128。
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: 标量值语义，clone 等价于 read。
        let v: u128 = unsafe { std::ptr::read_unaligned(ptr as *const u128) };
        arena.alloc_u128(v)
    }
}

/// `TypeOps` 实现：f128（16 字节，`ValueArena` 返回 `F128`）。
pub struct F128Ops;
impl TypeOps for F128Ops {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr 指向 16 字节合法 F128（Copy）。
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        arena.alloc_f128(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: F128 = arena.get_f128(h);
        // SAFETY: ptr 指向可写 16 字节 F128。
        unsafe { std::ptr::write_unaligned(ptr as *mut F128, v) }
    }
    #[inline]
    fn coerce(&self, v: ValueHandle, arena: &mut ValueArena) -> ValueHandle {
        // i128/u128 经 as f64 会丢精度，用 from_i128/from_u128 精确构造；F128 源恒等返回
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
        // IEEE 754 语义：NaN≠NaN，-0==+0（与 f32/f64 的 equal 一致）
        unsafe {
            let x = std::ptr::read_unaligned(a as *const F128);
            let y = std::ptr::read_unaligned(b as *const F128);
            if x.is_nan() || y.is_nan() {
                return false;
            }
            // -0 == +0：bit pattern 仅符号位不同时视为相等
            let xb = u128::from_le_bytes(x.0);
            let yb = u128::from_le_bytes(y.0);
            xb == yb || (xb | yb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr 指向合法 F128。
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: F128 Display 输出为 ASCII。
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr 指向合法 F128。
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: 标量值语义，clone 等价于 read。
        let v: F128 = unsafe { std::ptr::read_unaligned(ptr as *const F128) };
        arena.alloc_f128(v)
    }
}

/// `TypeOps` 实现：bool（`ValueArena::bool` 与 `get_bool` 均为 `&self` 单例语义）。
pub struct BoolOps;
impl TypeOps for BoolOps {
    #[inline]
    fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: ptr 指向 1 字节合法 bool。
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        arena.bool(v)
    }
    #[inline]
    fn write(&self, ptr: *mut u8, h: ValueHandle, arena: &ValueArena) {
        let v: bool = arena.get_bool(h);
        // SAFETY: ptr 指向可写 1 字节 bool。
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
        // SAFETY: 两指针均指向合法 bool。
        unsafe {
            std::ptr::read_unaligned(a as *const bool)
                == std::ptr::read_unaligned(b as *const bool)
        }
    }
    #[inline]
    fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        // SAFETY: ptr 指向合法 bool。
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        use std::io::Write;
        let mut cursor = std::io::Cursor::new(buf);
        let _ = write!(cursor, "{}", v);
        let written = cursor.position() as usize;
        let buf_ref: &mut [u8] = cursor.into_inner();
        // SAFETY: bool Display 输出为 ASCII（"true"/"false"）。
        unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
    }
    #[inline]
    fn hash_val(&self, ptr: *const u8) -> u64 {
        use std::hash::{Hash, Hasher};
        // SAFETY: ptr 指向合法 bool。
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
    #[inline]
    fn clone_val(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
        // SAFETY: 标量值语义，clone 等价于 read。
        let v: bool = unsafe { std::ptr::read_unaligned(ptr as *const bool) };
        arena.bool(v)
    }
}

// =========================================================================
// Null / Void ops（简化实现）
// =========================================================================

/// `TypeOps` 实现：null 类型（无数据，单例语义）。
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
        // null 为单例，所有实例相等（同 type_id）。
        true
    }
    #[inline]
    fn format<'a>(&self, _ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        if buf.len() < 4 {
            return "";
        }
        // SAFETY: "null" 为合法 ASCII。
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

/// `TypeOps` 实现：void 类型（无数据，单例语义）。
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
        // void 为单例，所有实例相等（同 type_id）。
        true
    }
    #[inline]
    fn format<'a>(&self, _ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
        if buf.len() < 4 {
            return "";
        }
        // SAFETY: "void" 为合法 ASCII。
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
// ref_ops / heap_ref_ops（引用类型 ops，简化实现）
// =========================================================================
//
// 完整的 ref 实现（msync 检查、ObjHeader 验证、堆对象物化）属于 engine 层职责。
// 共享层仅提供 8 字节指针槽的读写骨架：0 表示 null，非零地址在 read 时返回 null
// （保持无配语义），write 将句柄索引写入槽位作为占位。

macro_rules! impl_ref_ops {
    ($ops:ident) => {
        pub struct $ops;
        impl TypeOps for $ops {
            #[inline]
            fn read(&self, ptr: *const u8, arena: &mut ValueArena) -> ValueHandle {
                // SAFETY: ptr 指向 8 字节地址槽（0 表示 null）。
                let addr: usize = unsafe { std::ptr::read_unaligned(ptr as *const usize) };
                if addr == 0 {
                    arena.null()
                } else {
                    // 完整 ref 物化需要 engine 层（ObjHeader / msync 校验），
                    // 共享层保持无配语义，非零地址返回 null。
                    arena.null()
                }
            }
            #[inline]
            fn write(&self, ptr: *mut u8, h: ValueHandle, _arena: &ValueArena) {
                // SAFETY: ptr 指向可写 8 字节地址槽；存储句柄索引作为占位。
                let raw: usize = h.index();
                unsafe { std::ptr::write_unaligned(ptr as *mut usize, raw) }
            }
            #[inline]
            fn coerce(&self, v: ValueHandle, _arena: &mut ValueArena) -> ValueHandle {
                v
            }
            #[inline]
            fn equal(&self, a: *const u8, b: *const u8) -> bool {
                // SAFETY: 两指针均指向 8 字节地址槽。
                unsafe {
                    std::ptr::read_unaligned(a as *const usize)
                        == std::ptr::read_unaligned(b as *const usize)
                }
            }
            #[inline]
            fn format<'a>(&self, ptr: *const u8, buf: &'a mut [u8]) -> &'a str {
                // SAFETY: ptr 指向 8 字节地址。
                let addr: usize = unsafe { std::ptr::read_unaligned(ptr as *const usize) };
                use std::io::Write;
                let mut cursor = std::io::Cursor::new(buf);
                let _ = write!(cursor, "ref:0x{:x}", addr);
                let written = cursor.position() as usize;
                let buf_ref: &mut [u8] = cursor.into_inner();
                // SAFETY: 格式化文本为 ASCII。
                unsafe { std::str::from_utf8_unchecked(&buf_ref[..written]) }
            }
            #[inline]
            fn hash_val(&self, ptr: *const u8) -> u64 {
                // SAFETY: ptr 指向 8 字节地址，按 u64 重解释。
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
// type_id 常量（从 TypeDesc.rs 迁移，权威源）
// =========================================================================

/// 内置标量类型 ID 范围：1..=21（共 21 种）。
/// type_id 0 保留给 "unknown"；22+ 为用户/动态类型（type_def_index 偏移）。
pub const FIRST_DYNAMIC_TYPE_ID: u16 = 22;
pub const MAX_BUILTIN_TYPE_ID: u16 = 21;

/// str/null/void 的 type_id（与 BUILTIN_TABLE 一致）。
pub const STR_TYPE_ID: u16 = 19;
pub const NULL_TYPE_ID: u16 = 20;
pub const VOID_TYPE_ID: u16 = 21;

/// 整数类型 type_id 范围：1..=12。
pub const FIRST_INT_TYPE_ID: u16 = 1;
pub const LAST_INT_TYPE_ID: u16 = 12;
/// 浮点类型 type_id 范围：13..=16。
pub const FIRST_FLOAT_TYPE_ID: u16 = 13;
pub const LAST_FLOAT_TYPE_ID: u16 = 16;

/// 将 type_def_index 转为动态 type_id。
#[inline]
pub const fn dynamic_type_id(type_def_index: u16) -> u16 {
    FIRST_DYNAMIC_TYPE_ID + type_def_index
}

/// 将动态 type_id 还原为 type_def_index。
/// 仅对 type_id >= FIRST_DYNAMIC_TYPE_ID 有效。
#[inline]
pub const fn type_def_index_of(type_id: u16) -> u16 {
    type_id - FIRST_DYNAMIC_TYPE_ID
}

// =========================================================================
// ops 查找表（ops_of / ops_by_type_id）
// =========================================================================
//
// 从 Ty / type_id 派生 &'static dyn TypeOps，替代原 TypeDesc.rs 的
// scalar_tag_to_desc / lookup_by_type_id / lookup_by_int_kind / lookup_by_float_kind。
// TypeDescriptor 结构体已删除，不再有静态 DESC 常量中间层。

/// 按 Ty 查找内置类型的 ops。
///
/// 21 种内置类型（18 标量 + str/null/void）返回对应 ops；
/// 复合类型（Array/Ref/Fn/Nullable 等）和用户类型（Adt/Record 等）返回 None，
/// 需通过 DynamicOpsRegistry 查找。
#[inline]
pub fn ops_of(ty: &Ty) -> Option<&'static dyn TypeOps> {
    match ty {
        Ty::I8 => Some(&I8Ops),
        Ty::I16 => Some(&I16Ops),
        Ty::I32 => Some(&I32Ops),
        Ty::I64 => Some(&I64Ops),
        Ty::I128 => Some(&I128Ops),
        Ty::U8 => Some(&U8Ops),
        Ty::U16 => Some(&U16Ops),
        Ty::U32 => Some(&U32Ops),
        Ty::U64 => Some(&U64Ops),
        Ty::U128 => Some(&U128Ops),
        Ty::Isize => Some(&IsizeOps),
        Ty::Usize => Some(&UsizeOps),
        Ty::F16 => Some(&F16Ops),
        Ty::F32 => Some(&F32Ops),
        Ty::F64 => Some(&F64Ops),
        Ty::F128 => Some(&F128Ops),
        Ty::Bool => Some(&BoolOps),
        Ty::Char => Some(&CharOps),
        Ty::Str => Some(&HeapRefOps),
        Ty::Null => Some(&NullOps),
        Ty::Void => Some(&VoidOps),
        _ => None,
    }
}

/// 按 type_id 查找内置类型的 ops（1..=21）。
///
/// 派生自 BUILTIN_TABLE：type_id → ValueTag → Ty → ops_of。
/// 动态 type_id（>= 22）返回 None，需通过 DynamicOpsRegistry 查找。
#[inline]
pub fn ops_by_type_id(type_id: u16) -> Option<&'static dyn TypeOps> {
    let info = builtin_info_by_type_id(type_id)?;
    ops_of(&Ty::from(info.value_tag))
}

// =========================================================================
// DynamicOpsRegistry（替代 TypeDescriptorPool）
// =========================================================================
//
// 管理用户自定义类型的 ops（type_id 从 FIRST_DYNAMIC_TYPE_ID=22 开始）。
// TypeDescriptor 结构体已删除，用户类型 ops 直接注册为 &'static dyn TypeOps。

/// 动态类型 ops 条目：存储用户自定义类型的 ops + size + name + type_id。
///
/// 替代原 TypeDescriptor 的角色，但不聚合为结构体常量——
/// 仅在 DynamicOpsRegistry 中按 type_id 索引存储。
pub struct DynamicOpsEntry {
    pub ops: &'static dyn TypeOps,
    pub size: u8,
    pub type_id: u16,
    pub type_name: &'static str,
}

/// 动态类型 ops 注册表：管理用户自定义类型（type_id 从 FIRST_DYNAMIC_TYPE_ID 开始）。
///
/// 替代 TypeDescriptorPool。TypeDescriptor 结构体已删除，
/// 用户类型 ops 直接注册为 &'static dyn TypeOps。
///
/// `register` 会将类型名与条目泄漏为 &'static 以获得静态生命周期，
/// 适用于进程级类型注册（与 Sema/Ir 的类型表语义一致）。
pub struct DynamicOpsRegistry {
    entries: Vec<DynamicOpsEntry>,
    name_to_id: FxHashMap<String, u16>,
}

impl DynamicOpsRegistry {
    #[inline]
    pub fn new() -> Self {
        DynamicOpsRegistry {
            entries: Vec::new(),
            name_to_id: FxHashMap::default(),
        }
    }

    /// 注册一个用户类型，返回分配的 `type_id`（从 FIRST_DYNAMIC_TYPE_ID 开始递增）。
    pub fn register(&mut self, name: &str, size: u8, ops: &'static dyn TypeOps) -> u16 {
        let len = self.entries.len();
        assert!(
            FIRST_DYNAMIC_TYPE_ID as usize + len <= u16::MAX as usize,
            "type_id overflow: too many dynamic type descriptors"
        );
        let type_id = FIRST_DYNAMIC_TYPE_ID + len as u16;
        let name_static: &'static str = Box::leak(name.to_string().into_boxed_str());
        self.entries.push(DynamicOpsEntry {
            ops,
            size,
            type_id,
            type_name: name_static,
        });
        self.name_to_id.insert(name.to_string(), type_id);
        type_id
    }

    /// 按 `type_id` 查找 ops。
    ///
    /// type_id 1..=MAX_BUILTIN_TYPE_ID 委托给 ops_by_type_id（内置静态表）；
    /// type_id >= FIRST_DYNAMIC_TYPE_ID 查询动态池。
    #[inline]
    pub fn get_ops(&self, type_id: u16) -> Option<&'static dyn TypeOps> {
        if type_id <= MAX_BUILTIN_TYPE_ID {
            return ops_by_type_id(type_id);
        }
        let idx = type_def_index_of(type_id) as usize;
        self.entries.get(idx).map(|e| e.ops)
    }

    /// 按 `type_id` 查找完整条目（含 size / type_name）。
    ///
    /// 仅对动态 type_id（>= FIRST_DYNAMIC_TYPE_ID）有效；内置 type_id 返回 None
    /// （内置类型的 size/name 可从 Ty 派生：Ty::byte_width() / Ty::name()）。
    #[inline]
    pub fn get_entry(&self, type_id: u16) -> Option<&DynamicOpsEntry> {
        if type_id < FIRST_DYNAMIC_TYPE_ID {
            return None;
        }
        let idx = type_def_index_of(type_id) as usize;
        self.entries.get(idx)
    }

    /// 按类型名查找完整条目。
    #[inline]
    pub fn get_by_name(&self, name: &str) -> Option<&DynamicOpsEntry> {
        let id = *self.name_to_id.get(name)?;
        self.get_entry(id)
    }
}

impl Default for DynamicOpsRegistry {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
