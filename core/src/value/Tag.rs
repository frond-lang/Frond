// =========================================================================
// Tag — ValueTag + TypeFamily + BuiltinInfo: value system type metadata.
//
// Lives in the value module because ValueTag is fundamentally the runtime
// value discriminant (stored in ValueHandle's high 8 bits). The types module
// depends on value (unidirectional), accessing these via crate::value::*.
// =========================================================================

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
// TypeFamily — family classification for all types (replaces string-based family).
// =========================================================================

/// Family classification for all Frond types.
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
    /// `Lib` — opaque builtin handle over a dynamically loaded native library.
    Lib,

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
    /// `ForeignFn<R>` — resolved native symbol whose `call` returns `R`.
    ForeignFn,
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
/// - `Type::type_id()` / `Type::byte_width()` / `Type::to_value_tag()`
/// - `TypeDesc::lookup_by_type_id`
/// - `Compute::reflect_type_name` / `value::reflect_layout_*`
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
