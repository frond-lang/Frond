// =========================================================================
// Base structures: TypeHandle / FieldType / TraitMethodSig / EnvId.
//
// ValueTag and TypeFamily were extracted to crate::tag (tag.rs) to break the
// types ↔ value circular dependency. TypeHandle was moved from sema/Sema.rs.
// =========================================================================

// =========================================================================
// TypeHandle — type arena handle (moved from sema/Sema.rs).
// =========================================================================

/// Type arena handle (a `u32` index into `TypeArena`).
/// Placed in the Type module to break the Type ↔ sema circular dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeHandle(pub u32);

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
