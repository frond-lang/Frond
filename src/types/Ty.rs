// =========================================================================
// Ty — the unified type enum (the single source of types, Copy, no external deps).
// =========================================================================

use super::Tag::*;
use std::fmt;

// ─── Builtin type name constants ───
pub const NAME_STR: &str = "str";
pub const NAME_THROW: &str = "Throw";
pub const NAME_CHANNEL: &str = "Channel";
pub const NAME_ASYNC: &str = "Async";
pub const NAME_LAZY: &str = "Lazy";
pub const NAME_ATOMIC: &str = "Atomic";
pub const NAME_SENDER: &str = "Sender";
pub const NAME_RECEIVER: &str = "Receiver";
pub const NAME_TIMER: &str = "Timer";

/// Mapping from builtin generic type name → Ty variant.
/// All string-based type resolution for generics should go through this table.
pub const BUILTIN_GENERIC_TABLE: &[(&str, TyKind)] = &[
    (NAME_THROW, TyKind::Throw),
    (NAME_CHANNEL, TyKind::Channel),
    (NAME_ASYNC, TyKind::Async),
    (NAME_LAZY, TyKind::Lazy),
    (NAME_ATOMIC, TyKind::Atomic),
    (NAME_SENDER, TyKind::Sender),
    (NAME_RECEIVER, TyKind::Receiver),
    (NAME_TIMER, TyKind::Timer),
];

/// The kind of a builtin generic type (used for name→variant mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TyKind {
    Throw, Channel, Async, Lazy, Atomic, Sender, Receiver, Timer,
}

impl TyKind {
    pub fn to_ty(self, detail: DetailId) -> Ty {
        match self {
            TyKind::Throw => Ty::Throw(detail),
            TyKind::Channel => Ty::Channel(detail),
            TyKind::Async => Ty::Async(detail),
            TyKind::Lazy => Ty::Lazy(detail),
            TyKind::Atomic => Ty::Atomic(detail),
            TyKind::Sender => Ty::Sender(detail),
            TyKind::Receiver => Ty::Receiver(detail),
            TyKind::Timer => Ty::Timer(detail),
        }
    }
}

/// The unified type representation for Kuzo.
///
/// **Single source of types**: both the sema and IR layers use `Ty`; there is no longer
/// a `ConcreteType`.
///
/// **Copy enum**: all payloads are `u32` (`TypeHandle` or `DetailId`), with no heap
/// allocation. Structural data (params/fields/method_sigs/name, etc.) lives in the
/// `TypeArena` side tables, indexed by `DetailId`.
///
/// **Layered design:**
/// - **Basic types** (24 builtin + 4 composite + 7 generic): builtin types whose
///   variant alone is sufficient for family classification.
/// - **Other types** (6 user types): user-defined types carrying a `DetailId` that
///   indexes their structural data.
///
/// The two layers are distinguished via `is_basic()` / `is_other()`, and family
/// dispatch is done via `family()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty {
    // -- Basic: 18 scalars (no payload) --
    Bool, Char,
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
    Isize, Usize,
    F16, F32, F64, F128,

    // -- Basic: 3 non-scalar builtins (no payload) --
    Str,    // fat pointer
    Null,   // no value
    Void,   // no type

    // -- Basic: 7 builtin generics (DetailId indexes subtype structure in the arena) --
    /// `Throw<V, E>` (arena stores `{ value: TypeHandle, error: TypeHandle }`).
    Throw(DetailId),
    /// `Channel<T>` (arena stores `{ elem: TypeHandle }`).
    Channel(DetailId),
    /// `Async<T>` (arena stores `{ value: TypeHandle }`).
    Async(DetailId),
    /// `Lazy<T>` (arena stores `{ value: TypeHandle }`).
    Lazy(DetailId),
    /// `Atomic<T>` (arena stores `{ elem: TypeHandle }`).
    Atomic(DetailId),
    /// `Sender<T>` (arena stores `{ elem: TypeHandle }`).
    Sender(DetailId),
    /// `Receiver<T>` (arena stores `{ elem: TypeHandle }`).
    Receiver(DetailId),
    /// Timer (used for event-source dispatch; a user-defined type with builtin event-source semantics).
    Timer(DetailId),

    // -- Basic: 4 composite types (DetailId indexes structural details in the arena) --
    /// Array `[T; N]` (arena stores `{ elem: TypeHandle, size: Option<u64> }`).
    Array(DetailId),
    /// Reference `&T` / raw pointer `*T` (arena stores `{ inner: TypeHandle, is_raw: bool }`).
    Ref(DetailId),
    /// Function `(P1, P2) -> R` (arena stores `{ params: Box<[TypeHandle]>, return_type: TypeHandle }`).
    Fn(DetailId),
    /// Nullable `T?` (arena stores `{ inner: TypeHandle }`).
    Nullable(DetailId),

    // -- Other: user-defined types (DetailId indexes structural data) --
    /// Adt (algebraic data type) (arena stores `{ name: Box<str>, args: Box<[TypeHandle]> }`).
    Adt(DetailId),
    /// Record type `{ x: i32, y: i32 }` (arena stores `{ fields: Box<[FieldType]>, name: Option<Box<str>> }`).
    Record(DetailId),
    /// Trait type `Ord<T>` (arena stores `{ name: Box<str>, args: Box<[TypeHandle]> }`).
    Trait(DetailId),
    /// Trait object type: the existential type of an `inline_trait` value
    /// (arena stores `{ trait_name: Box<str>, method_sigs: Box<[TraitMethodSig]> }`).
    TraitObject(DetailId),
    /// Module reference type (arena stores `{ path: Box<str>, env: EnvId }`).
    ModuleRef(DetailId),
    /// User generic application `List<i32>` (arena stores `{ name: Box<str>, args: Box<[TypeHandle]> }`).
    Generic(DetailId),

    // -- Special --
    /// Bottom type (return/throw early-exit path; unifies with any type as the other side).
    Never,
    /// Type variable (during inference; payload is an index into `TypeArena::type_vars`).
    TypeVar(u32),
    /// Unknown type.
    Unknown,
}

/// Structural detail ID (a `u32` index into the `TypeArena::details` table).
///
/// Structural data for composite and user types lives in the `TypeArena` side tables,
/// indexed by this ID. All `Ty` variants carry only `TypeHandle`(u32) / `DetailId`(u32) /
/// `u32`, so `Ty` is `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetailId(pub u32);

impl Ty {
    /// Full type-family classification (completes all dispatch checks in a single call).
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

    // -- Predicates (all derived from family) --

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

    // -- Derived metadata (no intermediate ScalarInfo struct) --

    /// Bit width: scalars return `Some(bits)`; Str/Null/Void/composite/special return `None`.
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

    /// Integer bit width (only available for integers).
    #[inline]
    pub fn int_bit_width(&self) -> Option<u16> {
        if self.is_int() { self.bit_width() } else { None }
    }

    /// Float bit width (only available for floats).
    #[inline]
    pub fn float_bit_width(&self) -> Option<u16> {
        if self.is_float() { self.bit_width() } else { None }
    }

    /// Integer widening-comparison rank (same width and signedness share a rank); `None` for non-integers.
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

    /// Byte width (scalars: 1/2/4/8/16; str: 8; null/void: 0; composite: `None`).
    /// Derived from `BUILTIN_TABLE`.
    #[inline]
    pub fn byte_width(&self) -> Option<u8> {
        builtin_info_by_tag(self.to_value_tag()).map(|i| i.byte_width)
    }

    /// Builtin `type_id` (1..=21); returns `None` for others. Derived from `BUILTIN_TABLE`.
    #[inline]
    pub fn type_id(&self) -> Option<u16> {
        builtin_info_by_tag(self.to_value_tag()).map(|i| i.type_id)
    }

    /// Type family name (used for diagnostics and formatting).
    /// Scalars return a concrete name like "i32"; builtin generics return a family name
    /// like "Channel"; the concrete names of Adt/Trait/Generic must be looked up in the
    /// arena via `TypeDetail`.
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

    /// Runtime `ValueTag` (used for `ValueHandle` encoding).
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
            _ => ValueTag::Ref, // composite types are all Ref at runtime
        }
    }

    /// Whether this variant carries a `DetailId` (i.e., requires an arena lookup for structural data).
    #[inline]
    pub fn has_detail(&self) -> bool {
        matches!(self,
            Ty::Throw(_) | Ty::Channel(_) | Ty::Async(_) | Ty::Lazy(_)
            | Ty::Atomic(_) | Ty::Sender(_) | Ty::Receiver(_) | Ty::Timer(_)
            | Ty::Array(_) | Ty::Ref(_) | Ty::Fn(_) | Ty::Nullable(_)
            | Ty::Adt(_) | Ty::Record(_) | Ty::Trait(_)
            | Ty::TraitObject(_) | Ty::ModuleRef(_) | Ty::Generic(_))
    }

    /// Construct a parameterless builtin type from a type name (scalars + str/null/void +
    /// bare builtin generic names).
    /// User-defined types are resolved by the sema `type_binding_stack` and are not the
    /// responsibility of this function.
    pub fn from_type_name(name: &str) -> Option<Self> {
        // Builtin scalars + str + null + void.
        if let Some(info) = builtin_info_by_name(name) {
            return Some(info.value_tag.into());
        }
        // Bare builtin generic names (both "Async" and "Async<i32>" are recognized as Ty::Async).
        // DetailId uses DetailId(u32::MAX) as a placeholder (family() does not read the payload,
        // so the placeholder is safe).
        let base = name.split('<').next().unwrap_or(name);
        let kind = BUILTIN_GENERIC_TABLE.iter().find(|(n, _)| *n == base)?.1;
        let placeholder = DetailId(u32::MAX);
        Some(kind.to_ty(placeholder))
    }

    /// Determine the exact int-to-float widening path.
    /// Platform-dependent integers are first reduced to i32/u32 or i64/u64 based on
    /// `isize::BITS` before the check.
    pub fn int_to_float_widening(int_ty: &Ty, float_ty: &Ty) -> bool {
        let platform_bits = isize::BITS as u16;
        // Reduce platform-dependent integers to their equivalent fixed-width integers first.
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
// BUILTIN_TABLE — single source of truth for builtin type metadata.
// =========================================================================

/// Builtin type metadata (scalars + str/null/void only).
#[derive(Debug, Clone, Copy)]
pub struct BuiltinInfo {
    /// Type name (e.g. "i32"); the unique key for all derived functions.
    pub name: &'static str,
    /// Corresponding `ValueTag` (runtime encoding).
    pub value_tag: ValueTag,
    /// TypeDesc-layer `type_id` (within the 1..=21 builtin range).
    pub type_id: u16,
    /// Byte size (scalars: 1/2/4/8/16; str: 8; null/void: 0).
    pub byte_width: u8,
}

/// Metadata table for the 21 builtin types, sorted by `type_id` ascending.
///
/// **To add a new builtin type, simply append a row to this table.** All derived
/// facilities across the codebase sync automatically:
/// - `Ty::type_id()` / `Ty::byte_width()` / `Ty::to_value_tag()`
/// - `TypeDesc::lookup_by_type_id`
/// - `Reflect::__reflect_type_name` / `__reflect_layout_*`
/// - `Sema::int_kind_from_name` / `float_kind_from_name`
pub const BUILTIN_TABLE: &[BuiltinInfo] = &[
    // ---- Integers (1..=12) ----
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
    // ---- Floats (13..=16) ----
    BuiltinInfo { name: "f16",   value_tag: ValueTag::F16,   type_id: 13, byte_width: 2  },
    BuiltinInfo { name: "f32",   value_tag: ValueTag::F32,   type_id: 14, byte_width: 4  },
    BuiltinInfo { name: "f64",   value_tag: ValueTag::F64,   type_id: 15, byte_width: 8  },
    BuiltinInfo { name: "f128",  value_tag: ValueTag::F128,  type_id: 16, byte_width: 16 },
    // ---- Non-arithmetic scalars (17..=18) ----
    BuiltinInfo { name: "bool",  value_tag: ValueTag::Bool,  type_id: 17, byte_width: 1  },
    BuiltinInfo { name: "char",  value_tag: ValueTag::Char,  type_id: 18, byte_width: 4  },
    // ---- Non-scalar builtins (19..=21) ----
    BuiltinInfo { name: "str",   value_tag: ValueTag::Ref,   type_id: 19, byte_width: 8  },
    BuiltinInfo { name: "null",  value_tag: ValueTag::Null,  type_id: 20, byte_width: 0  },
    BuiltinInfo { name: "void",  value_tag: ValueTag::Void,  type_id: 21, byte_width: 0  },
];

// =========================================================================
// Lookup functions.
// =========================================================================

/// Look up `BuiltinInfo` by name.
#[inline]
pub fn builtin_info_by_name(name: &str) -> Option<&'static BuiltinInfo> {
    BUILTIN_TABLE.iter().find(|s| s.name == name)
}

/// Look up `BuiltinInfo` by `ValueTag`.
#[inline]
pub fn builtin_info_by_tag(tag: ValueTag) -> Option<&'static BuiltinInfo> {
    BUILTIN_TABLE.iter().find(|s| s.value_tag == tag)
}

/// Look up `BuiltinInfo` by `type_id`.
#[inline]
pub fn builtin_info_by_type_id(type_id: u16) -> Option<&'static BuiltinInfo> {
    BUILTIN_TABLE.iter().find(|s| s.type_id == type_id)
}

// =========================================================================
// Compile-time assertions (guard table integrity).
// =========================================================================

const _: () = {
    assert!(BUILTIN_TABLE.len() == 21, "BUILTIN_TABLE must have 21 entries");
    // type_id uniqueness check
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
// TypeDetail — element type of the TypeArena::details table.
// =========================================================================

/// TypeDetail — element type of the `TypeArena::details` table.
///
/// Each variant corresponds to a `Ty` variant that carries a `DetailId`, storing that
/// type's structural data. All variants own their data via `Box<[T]>` / `Box<str>`;
/// the arena centrally manages lifetimes.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDetail {
    /// `Ty::Throw`: `Throw<V, E>`.
    Throw { value_type: TypeHandle, error_type: TypeHandle },
    /// `Ty::Channel`: `Channel<T>`.
    Channel { elem: TypeHandle },
    /// `Ty::Async`: `Async<T>`.
    Async { value: TypeHandle },
    /// `Ty::Lazy`: `Lazy<T>`.
    Lazy { value: TypeHandle },
    /// `Ty::Atomic`: `Atomic<T>`.
    Atomic { elem: TypeHandle },
    /// `Ty::Sender`: `Sender<T>`.
    Sender { elem: TypeHandle },
    /// `Ty::Receiver`: `Receiver<T>`.
    Receiver { elem: TypeHandle },
    /// `Ty::Array`: `[T; N]`; `size == None` denotes a slice.
    Array { elem: TypeHandle, size: Option<u64> },
    /// `Ty::Ref`: `&T` / `*T`.
    Ref { inner: TypeHandle, is_raw: bool },
    /// `Ty::Fn`: `(P1, P2) -> R`.
    Fn { params: Box<[TypeHandle]>, return_type: TypeHandle },
    /// `Ty::Nullable`: `T?`.
    Nullable { inner: TypeHandle },
    /// `Ty::Adt`: algebraic data type, e.g. `Option<T>`.
    Adt { name: Box<str>, type_args: Box<[TypeHandle]> },
    /// `Ty::Record`: `{ x: i32, y: i32 }`.
    Record { fields: Box<[FieldType]>, name: Option<Box<str>> },
    /// `Ty::Trait`: trait type `Ord<T>`.
    Trait { name: Box<str>, type_args: Box<[TypeHandle]> },
    /// `Ty::TraitObject`: existential type of an `inline_trait` value.
    TraitObject { trait_name: Box<str>, method_sigs: Box<[TraitMethodSig]> },
    /// `Ty::ModuleRef`: module reference (path + env).
    ModuleRef { path: Box<str>, env: EnvId },
    /// `Ty::Generic`: user generic application `List<i32>`.
    Generic { name: Box<str>, args: Box<[TypeHandle]> },
}

// =========================================================================
// SemKind / TypeVar / UnifyError (moved from sema/Sema.rs).
// =========================================================================

/// Semantic-layer kind: describes the "type of a type".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemKind {
    /// Ordinary value type.
    Star,
    /// Type constructor: `param -> result`.
    Arrow { param: Box<SemKind>, result: Box<SemKind> },
    /// Kind variable: used during kind inference; payload is an index into `TypeArena::kind_vars`.
    Var(u32),
}

impl SemKind {
    /// The default kind is `Star`.
    #[inline]
    pub fn star() -> Self {
        SemKind::Star
    }

    /// Convert from `Ast::Kind` to `SemKind`.
    pub fn from_ast(kind: &crate::ast::Ast::Kind) -> Self {
        match kind {
            crate::ast::Ast::Kind::Star => SemKind::Star,
            crate::ast::Ast::Kind::Arrow { param, result } => SemKind::Arrow {
                param: Box::new(SemKind::from_ast(param)),
                result: Box::new(SemKind::from_ast(result)),
            },
        }
    }

    /// Compute the arity of a kind (Star=0, * -> * =1, * -> * -> * =2).
    pub fn arity(&self) -> usize {
        match self {
            SemKind::Star => 0,
            SemKind::Arrow { result, .. } => 1 + result.arity(),
            SemKind::Var(_) => 0,
        }
    }

    /// Apply the kind to a list of argument kinds, returning the result kind.
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

    /// Decompose an arrow kind into its parameter kind list and result kind.
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

/// Type variable: used for local inference (null literals, unannotated lambda parameters, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeVar {
    pub bound: Option<TypeHandle>,
    pub is_rigid: bool,
    /// Kind of the type variable: `Star` for ordinary type variables, `Arrow` for
    /// generic type-constructor parameters.
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

/// Type unification error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifyError {
    /// The two type structures are incompatible.
    TypeMismatch,
    /// Occurs check failed: the type variable occurs within the target type (infinite type).
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
