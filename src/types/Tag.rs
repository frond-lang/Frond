// =========================================================================
// Type discriminator tags (ValueTag) + base structures
// (TypeHandle / TypeFamily / FieldType / TraitMethodSig / EnvId).
// =========================================================================
//
// ValueTag was moved from Value.rs; TypeHandle was moved from sema/Sema.rs (placed in
// the Type module to break a circular dependency).

use super::ty::{builtin_info_by_name, builtin_info_by_tag};

// ---- ValueTag — 21 type tags (including Null/Void/Ref, used for ValueHandle encoding) ----

/// Type tag: covers scalars and Null/Void/Ref, 21 in total.
/// `#[repr(u8)]` guarantees ABI stability (the high 8 bits of a ValueHandle store this tag).
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
    /// Byte width (derived from `BUILTIN_TABLE`; returns 0 for non-scalars).
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

    /// Type family (derived from `ValueTag`, used by the IR/Sema layers for unified dispatch).
    ///
    /// Callers can use `matches!` to merge signed/unsigned integer variants and dispatch
    /// by bit width, keeping a single source of truth (`TypeFamily`).
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
            ValueTag::Ref => TypeFamily::Str, // the ValueTag of str is Ref
            ValueTag::Null => TypeFamily::Null,
            ValueTag::Void => TypeFamily::Void,
        }
    }

    /// Type name (derived from `BUILTIN_TABLE`; returns "unknown" for non-scalars).
    #[inline]
    pub fn name(self) -> &'static str {
        builtin_info_by_tag(self).map(|i| i.name).unwrap_or("unknown")
    }

    /// All 18 scalar ValueTags (derived from `BUILTIN_TABLE`, excluding Null/Void/Ref).
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

    /// Look up a ValueTag by name (derived from `BUILTIN_TABLE`).
    #[inline]
    pub fn from_name(name: &str) -> Option<ValueTag> {
        builtin_info_by_name(name).map(|i| i.value_tag)
    }

    /// Scalar type name (identical to `name()`; kept for backward compatibility with old callers).
    #[inline]
    pub fn type_name(self) -> &'static str {
        self.name()
    }
}

// =========================================================================
// TypeHandle — type arena handle (moved from sema/Sema.rs).
// =========================================================================

/// Type arena handle (a `u32` index into `TypeArena`).
/// Placed in the Type module to break the Type ↔ sema circular dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeHandle(pub u32);

// =========================================================================
// TypeFamily — family classification for all types (replaces string-based family).
// =========================================================================

/// Family classification for all Kuzo types.
///
/// Replaces the previous fragmented checks:
/// - `Ir.rs`'s `family: &'static str` (scalars only: "i32"/"i64"/"i128"/"float"/"bool")
/// - `Ir.rs`'s `ty_name == "str"` (name-based special case)
/// - `Ir.rs`'s `starts_with("Channel")` (prefix match)
/// - `Inference.rs`'s `name == "Throw"` (name-based special case)
///
/// Callers perform all dispatch in a single `match` via `ty.family()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeFamily {
    // -- Signed integers (bit-width distinguished because opcodes dispatch by width) --
    /// 8/16/32-bit signed integers (i8/i16/i32).
    SignedInt32,
    /// 64-bit signed integers (i64/isize).
    SignedInt64,
    /// 128-bit signed integers (i128).
    SignedInt128,

    // -- Unsigned integers (division/shift/comparison need signedness distinction) --
    /// 8/16/32-bit unsigned integers (u8/u16/u32).
    UnsignedInt32,
    /// 64-bit unsigned integers (u64/usize).
    UnsignedInt64,
    /// 128-bit unsigned integers (u128).
    UnsignedInt128,

    // -- Floating point (f16/f32/f64/f128 unified under f64 operations) --
    Float,

    // -- Non-numeric scalars --
    Bool,
    Char,

    // -- Non-scalar builtins --
    Str, Null, Void,

    // -- Builtin generics (replace starts_with / name-based special cases) --
    /// `Throw<V, E>` (dispatched via is_ok).
    Throw,
    /// `Channel<T>` (dispatched via send/recv/close).
    Channel,
    /// `Async<T>` (dispatched via await).
    Async,
    /// `Lazy<T>`.
    Lazy,
    /// `Atomic<T>` (dispatched via swap/compare_exchange/load/store).
    Atomic,
    /// `Sender<T>`.
    Sender,
    /// `Receiver<T>`.
    Receiver,
    /// Timer (used for event-source dispatch; a user-defined type with builtin event-source semantics).
    Timer,

    // -- Composite --
    Array, Ref, Fn, Nullable, Trait,

    // -- User types --
    Adt, Record, TraitObject, ModuleRef, Generic,

    // -- Special --
    Never, TypeVar, Unknown,
}

// =========================================================================
// FieldType / TraitMethodSig / EnvId — helper structures for composite types.
// =========================================================================

/// Record field: `name == None` denotes a positional field.
///
/// The field type indexes the arena via `TypeHandle` to avoid self-reference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldType {
    pub name: Option<Box<str>>,
    pub ty: TypeHandle,
}

/// Trait method signature (flattened from sema `TraitInfo` methods).
///
/// `return_type` is an arena handle for the return type (the original
/// `&'static TypeDescriptor` was changed to `TypeHandle` to avoid a circular dependency
/// where Type.rs would depend on TypeDesc.rs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitMethodSig {
    pub name: Box<str>,
    pub param_count: u8,
    pub return_type: TypeHandle,
    pub is_async: bool,
    /// Whether a default implementation body exists (IRBuilder uses this to decide
    /// whether to take the body from the AST).
    pub has_body: bool,
}

/// Environment handle: an index into `EnvArena` (moved from sema/Sema.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvId(pub u32);
