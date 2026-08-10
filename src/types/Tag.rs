// =========================================================================
// 类型判别标签（ValueTag）+ 基础结构（TypeHandle / TypeFamily / FieldType / TraitMethodSig / EnvId）
// =========================================================================
//
// ValueTag 从 Value.rs 移入；TypeHandle 从 sema/Sema.rs 移入（放在 Type 模块以打破循环依赖）。

use super::ty::{builtin_info_by_name, builtin_info_by_tag};

// ---- ValueTag — 21 种类型标签（含 Null/Void/Ref，用于 ValueHandle 编码）----

/// 类型标签：涵盖标量、Null/Void/Ref 共 21 种。
/// `#[repr(u8)]` 保证 ABI 稳定（ValueHandle 高 8 位存储此 tag）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum ValueTag {
    Null = 0,
    Void = 1,
    Bool = 2,
    Char = 3,
    I8 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    U8 = 8,
    U16 = 9,
    U32 = 10,
    U64 = 11,
    Isize = 12,
    Usize = 13,
    I128 = 14,
    U128 = 15,
    F16 = 16,
    F32 = 17,
    F64 = 18,
    F128 = 19,
    Ref = 20,
}

impl ValueTag {
    pub fn is_scalar(self) -> bool {
        !matches!(self, ValueTag::Null | ValueTag::Void | ValueTag::Ref)
    }
}

impl ValueTag {
    /// 字节宽度（派生自 `BUILTIN_TABLE`，非标量返回 0）。
    #[inline]
    pub fn byte_width(self) -> usize {
        builtin_info_by_tag(self).map(|i| i.byte_width as usize).unwrap_or(0)
    }

    pub fn is_int(self) -> bool {
        matches!(
            self,
            ValueTag::I8 | ValueTag::I16 | ValueTag::I32 | ValueTag::I64 | ValueTag::I128
                | ValueTag::U8 | ValueTag::U16 | ValueTag::U32 | ValueTag::U64 | ValueTag::U128
                | ValueTag::Isize | ValueTag::Usize
        )
    }

    pub fn is_float(self) -> bool {
        matches!(self, ValueTag::F16 | ValueTag::F32 | ValueTag::F64 | ValueTag::F128)
    }

    pub fn is_signed(self) -> bool {
        matches!(
            self,
            ValueTag::I8 | ValueTag::I16 | ValueTag::I32 | ValueTag::I64 | ValueTag::I128 | ValueTag::Isize
        )
    }

    pub fn is_bool(self) -> bool {
        matches!(self, ValueTag::Bool)
    }

    pub fn is_char(self) -> bool {
        matches!(self, ValueTag::Char)
    }

    pub fn is_numeric(self) -> bool {
        self.is_int() || self.is_float()
    }

    /// 类型家族（派生自 ValueTag，供 IR/Sema 层统一分派）。
    ///
    /// 调用方用 `matches!` 合并有符号/无符号整数变体即可按位宽分派，
    /// 保持单一真相源（`TypeFamily`）。
    #[inline]
    pub const fn family(self) -> TypeFamily {
        match self {
            ValueTag::I8 | ValueTag::I16 | ValueTag::I32 => TypeFamily::SignedInt32,
            ValueTag::I64 | ValueTag::Isize => TypeFamily::SignedInt64,
            ValueTag::I128 => TypeFamily::SignedInt128,
            ValueTag::U8 | ValueTag::U16 | ValueTag::U32 => TypeFamily::UnsignedInt32,
            ValueTag::U64 | ValueTag::Usize => TypeFamily::UnsignedInt64,
            ValueTag::U128 => TypeFamily::UnsignedInt128,
            ValueTag::F16 | ValueTag::F32 | ValueTag::F64 | ValueTag::F128 => TypeFamily::Float,
            ValueTag::Bool => TypeFamily::Bool,
            ValueTag::Char => TypeFamily::Char,
            ValueTag::Ref => TypeFamily::Str, // str 的 ValueTag 是 Ref
            ValueTag::Null => TypeFamily::Null,
            ValueTag::Void => TypeFamily::Void,
        }
    }

    /// 类型名（派生自 `BUILTIN_TABLE`，非标量返回 "unknown"）。
    #[inline]
    pub fn name(self) -> &'static str {
        builtin_info_by_tag(self).map(|i| i.name).unwrap_or("unknown")
    }

    /// 所有 18 个标量 ValueTag（派生自 `BUILTIN_TABLE`，排除 Null/Void/Ref）。
    pub fn all() -> &'static [ValueTag] {
        const SCALAR_TAGS: &[ValueTag] = &[
            ValueTag::I8, ValueTag::I16, ValueTag::I32, ValueTag::I64, ValueTag::I128,
            ValueTag::U8, ValueTag::U16, ValueTag::U32, ValueTag::U64, ValueTag::U128,
            ValueTag::Isize, ValueTag::Usize,
            ValueTag::F16, ValueTag::F32, ValueTag::F64, ValueTag::F128,
            ValueTag::Bool, ValueTag::Char,
        ];
        SCALAR_TAGS
    }

    /// 按 name 查 ValueTag（派生自 `BUILTIN_TABLE`）。
    #[inline]
    pub fn from_name(name: &str) -> Option<ValueTag> {
        builtin_info_by_name(name).map(|i| i.value_tag)
    }

    /// 标量类型名（与 name() 相同，保留此方法名兼容旧调用方）。
    #[inline]
    pub fn type_name(self) -> &'static str {
        self.name()
    }
}

// =========================================================================
// TypeHandle — 类型 arena 句柄（从 sema/Sema.rs 移入）
// =========================================================================

/// 类型 arena 句柄（u32 索引到 TypeArena）。
/// 放在 Type 模块以打破 Type↔sema 循环依赖。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeHandle(pub u32);

// =========================================================================
// TypeFamily — 所有类型的家族分类（替代字符串 family）
// =========================================================================

/// 所有 Kuzo 类型的家族分类。
///
/// 替代现有碎片化判断：
/// - Ir.rs 的 family: &'static str（仅标量，"i32"/"i64"/"i128"/"float"/"bool"）
/// - Ir.rs 的 ty_name == "str"（名字特判）
/// - Ir.rs 的 starts_with("Channel")（前缀匹配）
/// - Inference.rs 的 name == "Throw"（名字特判）
///
/// 调用方通过 ty.family() 一次 match 完成所有分派。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeFamily {
    // ── 有符号整数（区分位宽，因 opcode 按位宽分派）──
    /// 8/16/32 位有符号整数（i8/i16/i32）
    SignedInt32,
    /// 64 位有符号整数（i64/isize）
    SignedInt64,
    /// 128 位有符号整数（i128）
    SignedInt128,

    // ── 无符号整数（除法/移位/比较需区分符号性）──
    /// 8/16/32 位无符号整数（u8/u16/u32）
    UnsignedInt32,
    /// 64 位无符号整数（u64/usize）
    UnsignedInt64,
    /// 128 位无符号整数（u128）
    UnsignedInt128,

    // ── 浮点（f16/f32/f64/f128 统一 f64 运算）──
    Float,

    // ── 非数值标量 ──
    Bool,
    Char,

    // ── 非标量内置 ──
    Str, Null, Void,

    // ── 内置泛型（替代 starts_with/名字特判）──
    /// Throw<V, E>（is_ok 分派）
    Throw,
    /// Channel<T>（send/recv/close 分派）
    Channel,
    /// Async<T>（await 分派）
    Async,
    /// Lazy<T>
    Lazy,
    /// Atomic<T>（swap/cas/load/store 分派）
    Atomic,
    /// Sender<T>
    Sender,
    /// Receiver<T>
    Receiver,
    /// Timer（事件源分派用，用户自定义类型但事件源语义内置）
    Timer,

    // ── 复合 ──
    Array, Ref, Fn, Nullable, Trait,

    // ── 用户类型 ──
    Adt, Record, TraitObject, ModuleRef, Generic,

    // ── 特殊 ──
    Never, TypeVar, Unknown,
}

// =========================================================================
// FieldType / TraitMethodSig / EnvId — 复合类型辅助结构
// =========================================================================

/// record 字段：`name == None` 表示位置字段。
///
/// 字段类型通过 `TypeHandle` 索引 arena，避免自引用。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldType {
    pub name: Option<Box<str>>,
    pub ty: TypeHandle,
}

/// Trait 方法签名（压平后的 sema TraitInfo 方法）。
///
/// `return_type` 为返回类型的 arena 句柄（原 `&'static TypeDescriptor` 改为
/// `TypeHandle`，避免 Type.rs 依赖 TypeDesc.rs 形成循环）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitMethodSig {
    pub name: Box<str>,
    pub param_count: u8,
    pub return_type: TypeHandle,
    pub is_async: bool,
    /// 是否有 default 实现体（IRBuilder 据此决定是否从 AST 取 body）
    pub has_body: bool,
}

/// 环境句柄：`EnvArena` 中的索引（从 sema/Sema.rs 移入）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvId(pub u32);
