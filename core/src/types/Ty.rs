// =========================================================================
// Type — the unified type enum (the single source of types, Copy, no external deps).
// =========================================================================

use super::Tag::*;
use crate::value::{builtin_info_by_name, builtin_info_by_tag, ValueTag, TypeFamily};
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
pub const NAME_LIB: &str = "Lib";
pub const NAME_FOREIGN_FN: &str = "ForeignFn";

/// Mapping from builtin generic type name → Type variant.
/// All string-based type resolution for generics should go through this table.
pub const BUILTIN_GENERIC_TABLE: &[(&str, TypeKind)] = &[
    (NAME_THROW, TypeKind::Throw),
    (NAME_CHANNEL, TypeKind::Channel),
    (NAME_ASYNC, TypeKind::Async),
    (NAME_LAZY, TypeKind::Lazy),
    (NAME_ATOMIC, TypeKind::Atomic),
    (NAME_SENDER, TypeKind::Sender),
    (NAME_RECEIVER, TypeKind::Receiver),
    (NAME_TIMER, TypeKind::Timer),
    (NAME_FOREIGN_FN, TypeKind::ForeignFn),
];

/// The kind of a builtin generic type (used for name→variant mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
    Throw, Channel, Async, Lazy, Atomic, Sender, Receiver, Timer, ForeignFn,
}

impl TypeKind {
    pub fn to_ty(self, detail: DetailId) -> Type {
        match self {
            TypeKind::Throw => Type::Throw(detail),
            TypeKind::Channel => Type::Channel(detail),
            TypeKind::Async => Type::Async(detail),
            TypeKind::Lazy => Type::Lazy(detail),
            TypeKind::Atomic => Type::Atomic(detail),
            TypeKind::Sender => Type::Sender(detail),
            TypeKind::Receiver => Type::Receiver(detail),
            TypeKind::Timer => Type::Timer(detail),
            TypeKind::ForeignFn => Type::ForeignFn(detail),
        }
    }
}

/// The unified type representation for Frond.
///
/// **Single source of types**: both the sema and IR layers use `Type`; there is no longer
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
pub enum Type {
    // -- Basic: 18 scalars (no payload) --
    Bool, Char,
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
    Isize, Usize,
    F16, F32, F64, F128,

    // -- Basic: 4 non-scalar builtins (no payload) --
    Str,    // fat pointer
    Null,   // no value
    Void,   // no type
    /// `Lib` — opaque builtin handle over a dynamically loaded native library.
    /// Unit variant: all identity lives in the runtime HeapObj; `==` suffices.
    Lib,

    // -- Basic: 8 builtin generics (DetailId indexes subtype structure in the arena) --
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
    /// `ForeignFn<R>` (arena stores `{ ret: TypeHandle }`) — resolved native symbol;
    /// `call` returns `R`. Constructor: `Lib.lookup` only.
    ForeignFn(DetailId),
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
/// indexed by this ID. All `Type` variants carry only `TypeHandle`(u32) / `DetailId`(u32) /
/// `u32`, so `Type` is `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetailId(pub u32);

impl Type {
    /// Full type-family classification (completes all dispatch checks in a single call).
    #[inline]
    pub fn family(&self) -> TypeFamily {
        match self {
            Type::I8 | Type::I16 | Type::I32 => TypeFamily::SignedInt32,
            Type::I64 | Type::Isize => TypeFamily::SignedInt64,
            Type::I128 => TypeFamily::SignedInt128,
            Type::U8 | Type::U16 | Type::U32 => TypeFamily::UnsignedInt32,
            Type::U64 | Type::Usize => TypeFamily::UnsignedInt64,
            Type::U128 => TypeFamily::UnsignedInt128,
            Type::F16 | Type::F32 | Type::F64 | Type::F128 => TypeFamily::Float,
            Type::Bool => TypeFamily::Bool,
            Type::Char => TypeFamily::Char,
            Type::Str => TypeFamily::Str,
            Type::Null => TypeFamily::Null,
            Type::Void => TypeFamily::Void,
            Type::Lib => TypeFamily::Lib,
            Type::Throw(_) => TypeFamily::Throw,
            Type::Channel(_) => TypeFamily::Channel,
            Type::Async(_) => TypeFamily::Async,
            Type::Lazy(_) => TypeFamily::Lazy,
            Type::Atomic(_) => TypeFamily::Atomic,
            Type::Sender(_) => TypeFamily::Sender,
            Type::Receiver(_) => TypeFamily::Receiver,
            Type::ForeignFn(_) => TypeFamily::ForeignFn,
            Type::Timer(_) => TypeFamily::Timer,
            Type::Array(_) => TypeFamily::Array,
            Type::Ref(_) => TypeFamily::Ref,
            Type::Fn(_) => TypeFamily::Fn,
            Type::Nullable(_) => TypeFamily::Nullable,
            Type::Adt(_) => TypeFamily::Adt,
            Type::Record(_) => TypeFamily::Record,
            Type::Trait(_) => TypeFamily::Trait,
            Type::TraitObject(_) => TypeFamily::TraitObject,
            Type::ModuleRef(_) => TypeFamily::ModuleRef,
            Type::Generic(_) => TypeFamily::Generic,
            Type::Never => TypeFamily::Never,
            Type::TypeVar(_) => TypeFamily::TypeVar,
            Type::Unknown => TypeFamily::Unknown,
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
            | TypeFamily::Lazy | TypeFamily::Atomic | TypeFamily::Sender | TypeFamily::Receiver
            | TypeFamily::ForeignFn)
    }
    #[inline]
    pub fn is_builtin(&self) -> bool {
        self.is_scalar() || matches!(self.family(),
            TypeFamily::Str | TypeFamily::Null | TypeFamily::Void | TypeFamily::Lib)
            || self.is_builtin_generic()
    }

    /// Whether this variant is a payload-less unit for which `==` equality is
    /// sufficient for unification (all scalars + Str + Null + Void + Never + Unknown).
    ///
    /// Variants carrying a `DetailId` are excluded: even when their family matches,
    /// the structural data behind the `DetailId` must be compared via the arena.
    #[inline]
    pub fn is_atomic_unit(&self) -> bool {
        matches!(self,
            Type::Bool | Type::Char
            | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128
            | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128
            | Type::Isize | Type::Usize
            | Type::F16 | Type::F32 | Type::F64 | Type::F128
            | Type::Str | Type::Null | Type::Void | Type::Lib
            | Type::Never | Type::Unknown
        )
    }

    // -- Derived metadata (no intermediate ScalarInfo struct) --

    /// Bit width: scalars return `Some(bits)`; Str/Null/Void/composite/special return `None`.
    #[inline]
    pub fn bit_width(&self) -> Option<u16> {
        match self {
            Type::I8 | Type::U8 => Some(8),
            Type::I16 | Type::U16 | Type::F16 => Some(16),
            Type::I32 | Type::U32 | Type::F32 | Type::Char => Some(32),
            Type::I64 | Type::U64 | Type::Isize | Type::Usize | Type::F64 => Some(64),
            Type::I128 | Type::U128 | Type::F128 => Some(128),
            Type::Bool => Some(1),
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
            Type::I8 | Type::U8 => Some(1),
            Type::I16 | Type::U16 => Some(2),
            Type::I32 | Type::U32 => Some(3),
            Type::I64 | Type::U64 | Type::Isize | Type::Usize => Some(4),
            Type::I128 | Type::U128 => Some(5),
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

    /// Constructor kind label: uniform lowercase snake_case ("i32", "channel",
    /// "trait_object", "type_var"). Never compare this against a user-written
    /// name and never display it as source spelling — use `source_name()` for
    /// both. Payload names (Adt/Trait/Generic) must be looked up in the arena.
    pub fn name(&self) -> &'static str {
        match self {
            Type::I8 => "i8", Type::I16 => "i16", Type::I32 => "i32",
            Type::I64 => "i64", Type::I128 => "i128",
            Type::U8 => "u8", Type::U16 => "u16", Type::U32 => "u32",
            Type::U64 => "u64", Type::U128 => "u128",
            Type::Isize => "isize", Type::Usize => "usize",
            Type::F16 => "f16", Type::F32 => "f32", Type::F64 => "f64", Type::F128 => "f128",
            Type::Bool => "bool", Type::Char => "char",
            Type::Str => "str", Type::Null => "null", Type::Void => "void",
            Type::Lib => "lib",
            Type::Throw(_) => "throw",
            Type::Channel(_) => "channel",
            Type::Async(_) => "async",
            Type::Lazy(_) => "lazy",
            Type::Atomic(_) => "atomic",
            Type::Sender(_) => "sender",
            Type::Receiver(_) => "receiver",
            Type::ForeignFn(_) => "foreign_fn",
            Type::Timer(_) => "timer",
            Type::Array(_) => "array",
            Type::Ref(_) => "ref",
            Type::Fn(_) => "fn",
            Type::Nullable(_) => "nullable",
            Type::Adt(_) => "adt",
            Type::Record(_) => "record",
            Type::Trait(_) => "trait",
            Type::TraitObject(_) => "trait_object",
            Type::ModuleRef(_) => "module_ref",
            Type::Generic(_) => "generic",
            Type::Never => "never",
            Type::TypeVar(_) => "type_var",
            Type::Unknown => "unknown",
        }
    }

    /// Source-level spelling: how the type is written in user code. Builtin
    /// generics and Lib keep their PascalCase names (`Channel`, `ForeignFn`,
    /// …); a TypeVar is the `_` placeholder; everything else delegates to
    /// `name()` (scalars already read like source). Used by display and by
    /// name matching against user-written identifiers.
    pub fn source_name(&self) -> &'static str {
        match self {
            Type::Lib => "Lib",
            Type::Throw(_) => "Throw",
            Type::Channel(_) => "Channel",
            Type::Async(_) => "Async",
            Type::Lazy(_) => "Lazy",
            Type::Atomic(_) => "Atomic",
            Type::Sender(_) => "Sender",
            Type::Receiver(_) => "Receiver",
            Type::ForeignFn(_) => "ForeignFn",
            Type::Timer(_) => "Timer",
            Type::TypeVar(_) => "_",
            t => t.name(),
        }
    }

    /// Runtime `ValueTag` (used for `ValueHandle` encoding).
    #[inline]
    pub fn to_value_tag(&self) -> ValueTag {
        match self {
            Type::Bool => ValueTag::Bool,
            Type::Char => ValueTag::Char,
            Type::I8 => ValueTag::I8, Type::I16 => ValueTag::I16,
            Type::I32 => ValueTag::I32, Type::I64 => ValueTag::I64, Type::I128 => ValueTag::I128,
            Type::U8 => ValueTag::U8, Type::U16 => ValueTag::U16,
            Type::U32 => ValueTag::U32, Type::U64 => ValueTag::U64, Type::U128 => ValueTag::U128,
            Type::Isize => ValueTag::Isize, Type::Usize => ValueTag::Usize,
            Type::F16 => ValueTag::F16, Type::F32 => ValueTag::F32,
            Type::F64 => ValueTag::F64, Type::F128 => ValueTag::F128,
            Type::Str => ValueTag::Ref,
            Type::Null => ValueTag::Null,
            Type::Void => ValueTag::Void,
            _ => ValueTag::Ref, // composite types are all Ref at runtime
        }
    }

    /// Whether this variant carries a `DetailId` (i.e., requires an arena lookup for structural data).
    #[inline]
    pub fn has_detail(&self) -> bool {
        matches!(self,
            Type::Throw(_) | Type::Channel(_) | Type::Async(_) | Type::Lazy(_)
            | Type::Atomic(_) | Type::Sender(_) | Type::Receiver(_) | Type::Timer(_)
            | Type::ForeignFn(_)
            | Type::Array(_) | Type::Ref(_) | Type::Fn(_) | Type::Nullable(_)
            | Type::Adt(_) | Type::Record(_) | Type::Trait(_)
            | Type::TraitObject(_) | Type::ModuleRef(_) | Type::Generic(_))
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
        // Opaque nongeneric builtins.
        if name == NAME_LIB {
            return Some(Type::Lib);
        }
        // Bare builtin generic names (both "Async" and "Async<i32>" are recognized as Type::Async).
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
    pub fn int_to_float_widening(int_ty: &Type, float_ty: &Type) -> bool {
        let platform_bits = isize::BITS as u16;
        // Reduce platform-dependent integers to their equivalent fixed-width integers first.
        let int_ty = match int_ty {
            Type::Isize => {
                if platform_bits <= 32 {
                    return Self::int_to_float_widening(&Type::I32, float_ty);
                } else {
                    return Self::int_to_float_widening(&Type::I64, float_ty);
                }
            }
            Type::Usize => {
                if platform_bits <= 32 {
                    return Self::int_to_float_widening(&Type::U32, float_ty);
                } else {
                    return Self::int_to_float_widening(&Type::U64, float_ty);
                }
            }
            other => *other,
        };
        match int_ty {
            Type::I8 | Type::U8 | Type::I16 | Type::U16 => {
                matches!(float_ty, Type::F32 | Type::F64 | Type::F128)
            }
            Type::I32 | Type::U32 => {
                matches!(float_ty, Type::F64 | Type::F128)
            }
            Type::I64 | Type::U64 => matches!(float_ty, Type::F128),
            Type::I128 | Type::U128 => false,
            _ => false,
        }
    }
}

impl From<ValueTag> for Type {
    fn from(tag: ValueTag) -> Self {
        match tag {
            ValueTag::Bool => Type::Bool,
            ValueTag::Char => Type::Char,
            ValueTag::I8 => Type::I8, ValueTag::I16 => Type::I16,
            ValueTag::I32 => Type::I32, ValueTag::I64 => Type::I64, ValueTag::I128 => Type::I128,
            ValueTag::U8 => Type::U8, ValueTag::U16 => Type::U16,
            ValueTag::U32 => Type::U32, ValueTag::U64 => Type::U64, ValueTag::U128 => Type::U128,
            ValueTag::Isize => Type::Isize, ValueTag::Usize => Type::Usize,
            ValueTag::F16 => Type::F16, ValueTag::F32 => Type::F32,
            ValueTag::F64 => Type::F64, ValueTag::F128 => Type::F128,
            ValueTag::Ref => Type::Str,
            ValueTag::Null => Type::Null,
            ValueTag::Void => Type::Void,
        }
    }
}

// =========================================================================
// TypeDetail — element type of the TypeArena::details table.
// =========================================================================

/// TypeDetail — element type of the `TypeArena::details` table.
///
/// Each variant corresponds to a `Type` variant that carries a `DetailId`, storing that
/// type's structural data. All variants own their data via `Box<[T]>` / `Box<str>`;
/// the arena centrally manages lifetimes.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDetail {
    /// `Type::Throw`: `Throw<V, E>`.
    Throw { value_type: TypeHandle, error_type: TypeHandle },
    /// `Type::Channel`: `Channel<T>`.
    Channel { elem: TypeHandle },
    /// `Type::Async`: `Async<T>`.
    Async { value: TypeHandle },
    /// `Type::Lazy`: `Lazy<T>`.
    Lazy { value: TypeHandle },
    /// `Type::Atomic`: `Atomic<T>`.
    Atomic { elem: TypeHandle },
    /// `Type::Sender`: `Sender<T>`.
    Sender { elem: TypeHandle },
    /// `Type::Receiver`: `Receiver<T>`.
    Receiver { elem: TypeHandle },
    /// `Type::ForeignFn`: `ForeignFn<R>` — the return type of `call`.
    ForeignFn { ret: TypeHandle },
    /// `Type::Array`: `[T; N]`; `size == None` denotes a slice.
    Array { elem: TypeHandle, size: Option<u64> },
    /// `Type::Ref`: `&T` / `*T`.
    Ref { inner: TypeHandle, is_raw: bool },
    /// `Type::Fn`: `(P1, P2) -> R`.
    Fn { params: Box<[TypeHandle]>, return_type: TypeHandle },
    /// `Type::Nullable`: `T?`.
    Nullable { inner: TypeHandle },
    /// `Type::Adt`: algebraic data type, e.g. `Option<T>`.
    Adt { name: Box<str>, type_args: Box<[TypeHandle]> },
    /// `Type::Record`: `{ x: i32, y: i32 }`.
    Record { fields: Box<[FieldType]>, name: Option<Box<str>> },
    /// `Type::Trait`: trait type `Ord<T>`.
    Trait { name: Box<str>, type_args: Box<[TypeHandle]> },
    /// `Type::TraitObject`: existential type of an `inline_trait` value.
    TraitObject { trait_name: Box<str>, method_sigs: Box<[TraitMethodSig]> },
    /// `Type::ModuleRef`: module reference (path + env).
    ModuleRef { path: Box<str>, env: EnvId },
    /// `Type::Generic`: user generic application `List<i32>`.
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
