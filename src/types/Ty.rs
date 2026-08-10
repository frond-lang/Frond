// =========================================================================
// Ty — 统一类型枚举（唯一类型来源，Copy，无外部依赖）
// =========================================================================

use super::Tag::*;
use std::fmt;

/// Kuzo 统一类型表示。
///
/// **唯一类型来源**：sema 和 IR 层都使用 `Ty`，不再有 `ConcreteType`。
///
/// **Copy 枚举**：所有载荷均为 `u32`（`TypeHandle` 或 `DetailId`），无堆分配。
/// 结构数据（params/fields/method_sigs/name 等）存于 `TypeArena` 附属表，
/// 通过 `DetailId` 索引。
///
/// **分层设计：**
/// - **Basic types**（24 内置 + 4 复合 + 7 泛型）：内置类型，变体本身可做家族判断
/// - **Other types**（6 用户类型）：用户自定义类型，携带 `DetailId` 索引结构数据
///
/// 通过 `is_basic()` / `is_other()` 区分两层，通过 `family()` 做家族分派。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty {
    // ── Basic: 18 个标量（无载荷）──
    Bool, Char,
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
    Isize, Usize,
    F16, F32, F64, F128,

    // ── Basic: 3 个非标量内置（无载荷）──
    Str,    // fat pointer
    Null,   // 无值
    Void,   // 无类型

    // ── Basic: 7 个内置泛型（DetailId 索引 arena 中的子类型结构）──
    /// Throw<V, E>（arena 存 { value: TypeHandle, error: TypeHandle }）
    Throw(DetailId),
    /// Channel<T>（arena 存 { elem: TypeHandle }）
    Channel(DetailId),
    /// Async<T>（arena 存 { value: TypeHandle }）
    Async(DetailId),
    /// Lazy<T>（arena 存 { value: TypeHandle }）
    Lazy(DetailId),
    /// Atomic<T>（arena 存 { elem: TypeHandle }）
    Atomic(DetailId),
    /// Sender<T>（arena 存 { elem: TypeHandle }）
    Sender(DetailId),
    /// Receiver<T>（arena 存 { elem: TypeHandle }）
    Receiver(DetailId),
    /// Timer（事件源分派用，用户自定义类型但事件源语义内置）
    Timer(DetailId),

    // ── Basic: 4 个复合类型（DetailId 索引 arena 中的结构详情）──
    /// 数组 [T; N]（arena 存 { elem: TypeHandle, size: Option<u64> }）
    Array(DetailId),
    /// 引用 &T / 裸指针 *T（arena 存 { inner: TypeHandle, is_raw: bool }）
    Ref(DetailId),
    /// 函数 (P1, P2) -> R（arena 存 { params: Box<[TypeHandle]>, return_type: TypeHandle }）
    Fn(DetailId),
    /// 可空 T?（arena 存 { inner: TypeHandle }）
    Nullable(DetailId),

    // ── Other: 用户自定义类型（DetailId 索引结构数据）──
    /// Adt（代数数据类型）（arena 存 { name: Box<str>, args: Box<[TypeHandle]> }）
    Adt(DetailId),
    /// 记录类型 { x: i32, y: i32 }（arena 存 { fields: Box<[FieldType]>, name: Option<Box<str>> }）
    Record(DetailId),
    /// trait 类型 Ord<T>（arena 存 { name: Box<str>, args: Box<[TypeHandle]> }）
    Trait(DetailId),
    /// trait 对象类型：inline_trait 值的存在类型
    /// （arena 存 { trait_name: Box<str>, method_sigs: Box<[TraitMethodSig]> }）
    TraitObject(DetailId),
    /// 模块引用类型（arena 存 { path: Box<str>, env: EnvId }）
    ModuleRef(DetailId),
    /// 用户泛型应用 List<i32>（arena 存 { name: Box<str>, args: Box<[TypeHandle]> }）
    Generic(DetailId),

    // ── 特殊 ──
    /// 发散类型（return/throw 早退路径，与任意类型统一为对方）
    Never,
    /// 类型变量（推断中，载荷为 TypeArena::type_vars 下标）
    TypeVar(u32),
    /// 未知类型
    Unknown,
}

/// 类型结构详情 ID（u32 索引到 TypeArena::details 表）。
///
/// 复合类型和用户类型的结构数据存于 TypeArena 附属表，通过此 ID 索引。
/// `Ty` 所有变体只携带 `TypeHandle`(u32) / `DetailId`(u32) / `u32`，因此 Copy。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetailId(pub u32);

impl Ty {
    /// 全类型家族分类（一次调用完成全部分派判断）。
    #[inline]
    pub fn family(&self) -> TypeFamily {
        match self {
            Ty::I8 | Ty::I16 | Ty::I32 => TypeFamily::SignedInt32,
            Ty::I64 | Ty::Isize => TypeFamily::SignedInt64,
            Ty::I128 => TypeFamily::SignedInt128,
            Ty::U8 | Ty::U16 | Ty::U32 => TypeFamily::UnsignedInt32,
            Ty::U64 | Ty::Usize => TypeFamily::UnsignedInt64,
            Ty::U128 => TypeFamily::UnsignedInt128,
            Ty::F16 | Ty::F32 | Ty::F64 | Ty::F128 => TypeFamily::Float,
            Ty::Bool => TypeFamily::Bool,
            Ty::Char => TypeFamily::Char,
            Ty::Str => TypeFamily::Str,
            Ty::Null => TypeFamily::Null,
            Ty::Void => TypeFamily::Void,
            Ty::Throw(_) => TypeFamily::Throw,
            Ty::Channel(_) => TypeFamily::Channel,
            Ty::Async(_) => TypeFamily::Async,
            Ty::Lazy(_) => TypeFamily::Lazy,
            Ty::Atomic(_) => TypeFamily::Atomic,
            Ty::Sender(_) => TypeFamily::Sender,
            Ty::Receiver(_) => TypeFamily::Receiver,
            Ty::Timer(_) => TypeFamily::Timer,
            Ty::Array(_) => TypeFamily::Array,
            Ty::Ref(_) => TypeFamily::Ref,
            Ty::Fn(_) => TypeFamily::Fn,
            Ty::Nullable(_) => TypeFamily::Nullable,
            Ty::Adt(_) => TypeFamily::Adt,
            Ty::Record(_) => TypeFamily::Record,
            Ty::Trait(_) => TypeFamily::Trait,
            Ty::TraitObject(_) => TypeFamily::TraitObject,
            Ty::ModuleRef(_) => TypeFamily::ModuleRef,
            Ty::Generic(_) => TypeFamily::Generic,
            Ty::Never => TypeFamily::Never,
            Ty::TypeVar(_) => TypeFamily::TypeVar,
            Ty::Unknown => TypeFamily::Unknown,
        }
    }

    // ── 谓词（全部从 family 派生）──

    #[inline]
    pub fn is_signed_int(&self) -> bool {
        matches!(self.family(),
            TypeFamily::SignedInt32 | TypeFamily::SignedInt64 | TypeFamily::SignedInt128)
    }
    #[inline]
    pub fn is_unsigned_int(&self) -> bool {
        matches!(self.family(),
            TypeFamily::UnsignedInt32 | TypeFamily::UnsignedInt64 | TypeFamily::UnsignedInt128)
    }
    #[inline]
    pub fn is_int(&self) -> bool { self.is_signed_int() || self.is_unsigned_int() }
    #[inline]
    pub fn is_float(&self) -> bool { matches!(self.family(), TypeFamily::Float) }
    #[inline]
    pub fn is_numeric(&self) -> bool { self.is_int() || self.is_float() }
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.is_numeric() || matches!(self.family(), TypeFamily::Bool | TypeFamily::Char)
    }
    #[inline]
    pub fn is_signed(&self) -> bool { self.is_signed_int() }
    #[inline]
    pub fn is_builtin_generic(&self) -> bool {
        matches!(self.family(),
            TypeFamily::Throw | TypeFamily::Channel | TypeFamily::Async
            | TypeFamily::Lazy | TypeFamily::Atomic | TypeFamily::Sender | TypeFamily::Receiver)
    }
    #[inline]
    pub fn is_builtin(&self) -> bool {
        self.is_scalar() || matches!(self.family(),
            TypeFamily::Str | TypeFamily::Null | TypeFamily::Void) || self.is_builtin_generic()
    }

    // ── 派生元信息（无 ScalarInfo 中间结构）──

    /// 位宽：标量返回 Some(bits)，Str/Null/Void/复合/特殊返回 None。
    #[inline]
    pub fn bit_width(&self) -> Option<u16> {
        match self {
            Ty::I8 | Ty::U8 => Some(8),
            Ty::I16 | Ty::U16 | Ty::F16 => Some(16),
            Ty::I32 | Ty::U32 | Ty::F32 | Ty::Char => Some(32),
            Ty::I64 | Ty::U64 | Ty::Isize | Ty::Usize | Ty::F64 => Some(64),
            Ty::I128 | Ty::U128 | Ty::F128 => Some(128),
            Ty::Bool => Some(1),
            _ => None,
        }
    }

    /// 整数位宽（仅整数有）。
    #[inline]
    pub fn int_bit_width(&self) -> Option<u16> {
        if self.is_int() { self.bit_width() } else { None }
    }

    /// 浮点位宽（仅浮点有）。
    #[inline]
    pub fn float_bit_width(&self) -> Option<u16> {
        if self.is_float() { self.bit_width() } else { None }
    }

    /// 整数宽化比较秩（同宽同符号共享秩）；非整数为 None。
    #[inline]
    pub fn int_rank(&self) -> Option<u8> {
        match self {
            Ty::I8 | Ty::U8 => Some(1),
            Ty::I16 | Ty::U16 => Some(2),
            Ty::I32 | Ty::U32 => Some(3),
            Ty::I64 | Ty::U64 | Ty::Isize | Ty::Usize => Some(4),
            Ty::I128 | Ty::U128 => Some(5),
            _ => None,
        }
    }

    /// 字节宽度（标量: 1/2/4/8/16；str: 8；null/void: 0；复合: None）。
    /// 派生自 BUILTIN_TABLE。
    #[inline]
    pub fn byte_width(&self) -> Option<u8> {
        builtin_info_by_tag(self.to_value_tag()).map(|i| i.byte_width)
    }

    /// 内置 type_id（1..=21），其他返回 None。派生自 BUILTIN_TABLE。
    #[inline]
    pub fn type_id(&self) -> Option<u16> {
        builtin_info_by_tag(self.to_value_tag()).map(|i| i.type_id)
    }

    /// 类型族名（用于诊断和格式化）。
    /// 标量返回 "i32" 等具体名；内置泛型返回 "Channel" 等族名；
    /// Adt/Trait/Generic 的具体名需通过 arena 查 TypeDetail。
    pub fn name(&self) -> &'static str {
        match self {
            Ty::I8 => "i8", Ty::I16 => "i16", Ty::I32 => "i32",
            Ty::I64 => "i64", Ty::I128 => "i128",
            Ty::U8 => "u8", Ty::U16 => "u16", Ty::U32 => "u32",
            Ty::U64 => "u64", Ty::U128 => "u128",
            Ty::Isize => "isize", Ty::Usize => "usize",
            Ty::F16 => "f16", Ty::F32 => "f32", Ty::F64 => "f64", Ty::F128 => "f128",
            Ty::Bool => "bool", Ty::Char => "char",
            Ty::Str => "str", Ty::Null => "null", Ty::Void => "void",
            Ty::Throw(_) => "Throw",
            Ty::Channel(_) => "Channel",
            Ty::Async(_) => "Async",
            Ty::Lazy(_) => "Lazy",
            Ty::Atomic(_) => "Atomic",
            Ty::Sender(_) => "Sender",
            Ty::Receiver(_) => "Receiver",
            Ty::Timer(_) => "Timer",
            Ty::Array(_) => "array",
            Ty::Ref(_) => "ref",
            Ty::Fn(_) => "fn",
            Ty::Nullable(_) => "nullable",
            Ty::Adt(_) => "adt",
            Ty::Record(_) => "record",
            Ty::Trait(_) => "trait",
            Ty::TraitObject(_) => "trait_object",
            Ty::ModuleRef(_) => "module_ref",
            Ty::Generic(_) => "generic",
            Ty::Never => "never",
            Ty::TypeVar(_) => "_",
            Ty::Unknown => "unknown",
        }
    }

    /// 运行时 ValueTag（用于 ValueHandle 编码）。
    #[inline]
    pub fn to_value_tag(&self) -> ValueTag {
        match self {
            Ty::Bool => ValueTag::Bool,
            Ty::Char => ValueTag::Char,
            Ty::I8 => ValueTag::I8, Ty::I16 => ValueTag::I16,
            Ty::I32 => ValueTag::I32, Ty::I64 => ValueTag::I64, Ty::I128 => ValueTag::I128,
            Ty::U8 => ValueTag::U8, Ty::U16 => ValueTag::U16,
            Ty::U32 => ValueTag::U32, Ty::U64 => ValueTag::U64, Ty::U128 => ValueTag::U128,
            Ty::Isize => ValueTag::Isize, Ty::Usize => ValueTag::Usize,
            Ty::F16 => ValueTag::F16, Ty::F32 => ValueTag::F32,
            Ty::F64 => ValueTag::F64, Ty::F128 => ValueTag::F128,
            Ty::Str => ValueTag::Ref,
            Ty::Null => ValueTag::Null,
            Ty::Void => ValueTag::Void,
            _ => ValueTag::Ref, // 复合类型运行时都是 Ref
        }
    }

    /// 是否携带 DetailId（即需查 arena 获取结构数据）。
    #[inline]
    pub fn has_detail(&self) -> bool {
        matches!(self,
            Ty::Throw(_) | Ty::Channel(_) | Ty::Async(_) | Ty::Lazy(_)
            | Ty::Atomic(_) | Ty::Sender(_) | Ty::Receiver(_) | Ty::Timer(_)
            | Ty::Array(_) | Ty::Ref(_) | Ty::Fn(_) | Ty::Nullable(_)
            | Ty::Adt(_) | Ty::Record(_) | Ty::Trait(_)
            | Ty::TraitObject(_) | Ty::ModuleRef(_) | Ty::Generic(_))
    }

    /// 从类型名构造无参内置类型（标量 + str/null/void + 裸内置泛型名）。
    /// 用户自定义类型由 sema type_binding_stack 解析，不在此函数职责内。
    pub fn from_type_name(name: &str) -> Option<Self> {
        // 内置标量 + str + null + void
        if let Some(info) = builtin_info_by_name(name) {
            return Some(info.value_tag.into());
        }
        // 内置泛型裸名（"Async" / "Async<i32>" 均识别为 Ty::Async）。
        // DetailId 用 DetailId(u32::MAX) 占位（family() 不读载荷，占位安全）。
        let base = name.split('<').next().unwrap_or(name);
        let placeholder = DetailId(u32::MAX);
        Some(match base {
            "Throw" => Ty::Throw(placeholder),
            "Channel" => Ty::Channel(placeholder),
            "Async" => Ty::Async(placeholder),
            "Lazy" => Ty::Lazy(placeholder),
            "Atomic" => Ty::Atomic(placeholder),
            "Sender" => Ty::Sender(placeholder),
            "Receiver" => Ty::Receiver(placeholder),
            "Timer" => Ty::Timer(placeholder),
            _ => return None,
        })
    }

    /// int→float 精确 widening 路径判定。
    /// 平台相关整数按 `isize::BITS` 归约到 i32/u32 或 i64/u64 后判定。
    pub fn int_to_float_widening(int_ty: &Ty, float_ty: &Ty) -> bool {
        let platform_bits = isize::BITS as u16;
        // 平台相关整数先归约到等价定长整数。
        let int_ty = match int_ty {
            Ty::Isize => {
                if platform_bits <= 32 {
                    return Self::int_to_float_widening(&Ty::I32, float_ty);
                } else {
                    return Self::int_to_float_widening(&Ty::I64, float_ty);
                }
            }
            Ty::Usize => {
                if platform_bits <= 32 {
                    return Self::int_to_float_widening(&Ty::U32, float_ty);
                } else {
                    return Self::int_to_float_widening(&Ty::U64, float_ty);
                }
            }
            other => *other,
        };
        match int_ty {
            Ty::I8 | Ty::U8 | Ty::I16 | Ty::U16 => {
                matches!(float_ty, Ty::F32 | Ty::F64 | Ty::F128)
            }
            Ty::I32 | Ty::U32 => {
                matches!(float_ty, Ty::F64 | Ty::F128)
            }
            Ty::I64 | Ty::U64 => matches!(float_ty, Ty::F128),
            Ty::I128 | Ty::U128 => false,
            _ => false,
        }
    }
}

impl From<ValueTag> for Ty {
    fn from(tag: ValueTag) -> Self {
        match tag {
            ValueTag::Bool => Ty::Bool,
            ValueTag::Char => Ty::Char,
            ValueTag::I8 => Ty::I8, ValueTag::I16 => Ty::I16,
            ValueTag::I32 => Ty::I32, ValueTag::I64 => Ty::I64, ValueTag::I128 => Ty::I128,
            ValueTag::U8 => Ty::U8, ValueTag::U16 => Ty::U16,
            ValueTag::U32 => Ty::U32, ValueTag::U64 => Ty::U64, ValueTag::U128 => Ty::U128,
            ValueTag::Isize => Ty::Isize, ValueTag::Usize => Ty::Usize,
            ValueTag::F16 => Ty::F16, ValueTag::F32 => Ty::F32,
            ValueTag::F64 => Ty::F64, ValueTag::F128 => Ty::F128,
            ValueTag::Ref => Ty::Str,
            ValueTag::Null => Ty::Null,
            ValueTag::Void => Ty::Void,
        }
    }
}

// =========================================================================
// BUILTIN_TABLE — 内置类型元信息单一真相源
// =========================================================================

/// 内置类型元信息（仅标量 + str/null/void）。
#[derive(Debug, Clone, Copy)]
pub struct BuiltinInfo {
    /// 类型名（如 "i32"），所有派生函数的唯一键
    pub name: &'static str,
    /// 对应的 ValueTag（运行时编码）
    pub value_tag: ValueTag,
    /// TypeDesc 层 type_id（1..=21 内置范围）
    pub type_id: u16,
    /// 字节大小（标量: 1/2/4/8/16；str: 8；null/void: 0）
    pub byte_width: u8,
}

/// 21 个内置类型的元信息表，按 type_id 升序排列。
///
/// **新增内置类型时，只需在此表追加一行**。全库派生设施自动同步：
/// - Ty::type_id() / Ty::byte_width() / Ty::to_value_tag()
/// - TypeDesc::lookup_by_type_id
/// - Reflect::__reflect_type_name / __reflect_layout_*
/// - Sema::int_kind_from_name / float_kind_from_name
pub const BUILTIN_TABLE: &[BuiltinInfo] = &[
    // ---- 整数（1..=12）----
    BuiltinInfo { name: "i8",    value_tag: ValueTag::I8,    type_id: 1,  byte_width: 1  },
    BuiltinInfo { name: "i16",   value_tag: ValueTag::I16,   type_id: 2,  byte_width: 2  },
    BuiltinInfo { name: "i32",   value_tag: ValueTag::I32,   type_id: 3,  byte_width: 4  },
    BuiltinInfo { name: "i64",   value_tag: ValueTag::I64,   type_id: 4,  byte_width: 8  },
    BuiltinInfo { name: "i128",  value_tag: ValueTag::I128,  type_id: 5,  byte_width: 16 },
    BuiltinInfo { name: "u8",    value_tag: ValueTag::U8,    type_id: 6,  byte_width: 1  },
    BuiltinInfo { name: "u16",   value_tag: ValueTag::U16,   type_id: 7,  byte_width: 2  },
    BuiltinInfo { name: "u32",   value_tag: ValueTag::U32,   type_id: 8,  byte_width: 4  },
    BuiltinInfo { name: "u64",   value_tag: ValueTag::U64,   type_id: 9,  byte_width: 8  },
    BuiltinInfo { name: "u128",  value_tag: ValueTag::U128,  type_id: 10, byte_width: 16 },
    BuiltinInfo { name: "isize", value_tag: ValueTag::Isize, type_id: 11, byte_width: 8  },
    BuiltinInfo { name: "usize", value_tag: ValueTag::Usize, type_id: 12, byte_width: 8  },
    // ---- 浮点（13..=16）----
    BuiltinInfo { name: "f16",   value_tag: ValueTag::F16,   type_id: 13, byte_width: 2  },
    BuiltinInfo { name: "f32",   value_tag: ValueTag::F32,   type_id: 14, byte_width: 4  },
    BuiltinInfo { name: "f64",   value_tag: ValueTag::F64,   type_id: 15, byte_width: 8  },
    BuiltinInfo { name: "f128",  value_tag: ValueTag::F128,  type_id: 16, byte_width: 16 },
    // ---- 非算术标量（17..=18）----
    BuiltinInfo { name: "bool",  value_tag: ValueTag::Bool,  type_id: 17, byte_width: 1  },
    BuiltinInfo { name: "char",  value_tag: ValueTag::Char,  type_id: 18, byte_width: 4  },
    // ---- 非标量内置（19..=21）----
    BuiltinInfo { name: "str",   value_tag: ValueTag::Ref,   type_id: 19, byte_width: 8  },
    BuiltinInfo { name: "null",  value_tag: ValueTag::Null,  type_id: 20, byte_width: 0  },
    BuiltinInfo { name: "void",  value_tag: ValueTag::Void,  type_id: 21, byte_width: 0  },
];

// =========================================================================
// 查找函数
// =========================================================================

/// 按 name 查 BuiltinInfo。
#[inline]
pub fn builtin_info_by_name(name: &str) -> Option<&'static BuiltinInfo> {
    BUILTIN_TABLE.iter().find(|s| s.name == name)
}

/// 按 ValueTag 查 BuiltinInfo。
#[inline]
pub fn builtin_info_by_tag(tag: ValueTag) -> Option<&'static BuiltinInfo> {
    BUILTIN_TABLE.iter().find(|s| s.value_tag == tag)
}

/// 按 type_id 查 BuiltinInfo。
#[inline]
pub fn builtin_info_by_type_id(type_id: u16) -> Option<&'static BuiltinInfo> {
    BUILTIN_TABLE.iter().find(|s| s.type_id == type_id)
}

// =========================================================================
// 编译期断言（保护表完整性）
// =========================================================================

const _: () = {
    assert!(BUILTIN_TABLE.len() == 21, "BUILTIN_TABLE must have 21 entries");
    // type_id 唯一性检查
    let mut seen = [false; 22];
    let mut i = 0;
    while i < BUILTIN_TABLE.len() {
        let id = BUILTIN_TABLE[i].type_id as usize;
        assert!(!seen[id], "duplicate type_id in BUILTIN_TABLE");
        seen[id] = true;
        i += 1;
    }
};

// =========================================================================
// TypeDetail — TypeArena::details 表的元素
// =========================================================================

/// TypeDetail — TypeArena::details 表的元素。
///
/// 每个变体对应一个携带 DetailId 的 Ty 变体，存储该类型的结构数据。
/// 所有变体均为 Box<[T]> / Box<str> 拥有所有权，arena 集中管理生命周期。
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDetail {
    /// Ty::Throw：Throw<V, E>
    Throw { value_type: TypeHandle, error_type: TypeHandle },
    /// Ty::Channel：Channel<T>
    Channel { elem: TypeHandle },
    /// Ty::Async：Async<T>
    Async { value: TypeHandle },
    /// Ty::Lazy：Lazy<T>
    Lazy { value: TypeHandle },
    /// Ty::Atomic：Atomic<T>
    Atomic { elem: TypeHandle },
    /// Ty::Sender：Sender<T>
    Sender { elem: TypeHandle },
    /// Ty::Receiver：Receiver<T>
    Receiver { elem: TypeHandle },
    /// Ty::Array：[T; N]，size == None 为切片
    Array { elem: TypeHandle, size: Option<u64> },
    /// Ty::Ref：&T / *T
    Ref { inner: TypeHandle, is_raw: bool },
    /// Ty::Fn：(P1, P2) -> R
    Fn { params: Box<[TypeHandle]>, return_type: TypeHandle },
    /// Ty::Nullable：T?
    Nullable { inner: TypeHandle },
    /// Ty::Adt：代数数据类型 Option<T> 等
    Adt { name: Box<str>, type_args: Box<[TypeHandle]> },
    /// Ty::Record：{ x: i32, y: i32 }
    Record { fields: Box<[FieldType]>, name: Option<Box<str>> },
    /// Ty::Trait：trait 类型 Ord<T>
    Trait { name: Box<str>, type_args: Box<[TypeHandle]> },
    /// Ty::TraitObject：inline_trait 值的存在类型
    TraitObject { trait_name: Box<str>, method_sigs: Box<[TraitMethodSig]> },
    /// Ty::ModuleRef：模块引用（path + env）
    ModuleRef { path: Box<str>, env: EnvId },
    /// Ty::Generic：用户泛型应用 List<i32>
    Generic { name: Box<str>, args: Box<[TypeHandle]> },
}

// =========================================================================
// SemKind / TypeVar / UnifyError（从 sema/Sema.rs 移入）
// =========================================================================

/// 语义层的 kind：描述类型的"类型"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemKind {
    /// 普通值类型
    Star,
    /// 类型构造器：`param -> result`
    Arrow { param: Box<SemKind>, result: Box<SemKind> },
    /// kind 变量：用于 kind 推断，载荷为 `TypeArena::kind_vars` 下标
    Var(u32),
}

impl SemKind {
    /// 默认 kind 为 Star
    #[inline]
    pub fn star() -> Self {
        SemKind::Star
    }

    /// 从 Ast::Kind 转换为 SemKind
    pub fn from_ast(kind: &crate::ast::Ast::Kind) -> Self {
        match kind {
            crate::ast::Ast::Kind::Star => SemKind::Star,
            crate::ast::Ast::Kind::Arrow { param, result } => SemKind::Arrow {
                param: Box::new(SemKind::from_ast(param)),
                result: Box::new(SemKind::from_ast(result)),
            },
        }
    }

    /// 计算 kind 的 arity（Star=0, * -> * =1, * -> * -> * =2）
    pub fn arity(&self) -> usize {
        match self {
            SemKind::Star => 0,
            SemKind::Arrow { result, .. } => 1 + result.arity(),
            SemKind::Var(_) => 0,
        }
    }

    /// 对 kind 进行应用：给定参数 kind 列表，返回结果 kind。
    pub fn apply(&self, args: &[SemKind]) -> Option<SemKind> {
        if args.is_empty() {
            return Some(self.clone());
        }
        match self {
            SemKind::Star => None,
            SemKind::Var(_) => None,
            SemKind::Arrow { param, result } => {
                if args.is_empty() {
                    return Some(self.clone());
                }
                if **param != args[0] {
                    return None;
                }
                result.apply(&args[1..])
            }
        }
    }

    /// 提取箭头 kind 的参数 kind 列表和结果 kind。
    pub fn decompose(&self) -> (Vec<SemKind>, &SemKind) {
        let mut params = Vec::new();
        let mut current = self;
        while let SemKind::Arrow { param, result } = current {
            params.push((**param).clone());
            current = result;
        }
        (params, current)
    }
}

/// 类型变量：用于局部推断（null 字面量、未标注 lambda 参数等）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeVar {
    pub bound: Option<TypeHandle>,
    pub is_rigid: bool,
    /// 类型变量的 kind：普通类型变量为 Star，泛型类型构造器参数为 Arrow。
    pub kind: SemKind,
}

impl TypeVar {
    #[inline]
    pub fn new(is_rigid: bool) -> Self {
        TypeVar {
            bound: None,
            is_rigid,
            kind: SemKind::Star,
        }
    }

    #[inline]
    pub fn new_with_kind(is_rigid: bool, kind: SemKind) -> Self {
        TypeVar {
            bound: None,
            is_rigid,
            kind,
        }
    }
}

/// 类型统一错误。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifyError {
    /// 两个类型结构不兼容
    TypeMismatch,
    /// occurs check 失败：类型变量出现在目标类型中（无限类型）
    OccursCheckFailed,
}

impl fmt::Display for UnifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch => write!(f, "type mismatch"),
            Self::OccursCheckFailed => write!(f, "occurs check failed (recursive type)"),
        }
    }
}

impl std::error::Error for UnifyError {}
