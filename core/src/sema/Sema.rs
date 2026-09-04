//! Sema.rs — core data structures for semantic analysis.
//!
//! The single source of truth for the type system is `crate::types`
//! (`Type` / `TypeArena` / `TypeOps`). The legacy `ConcreteType` / `TypeDescriptor`
//! types are no longer used — a clean removal with no compatibility layer.
//!
//! Key design decisions:
//! - **Arena + index**: recursive subtypes are addressed via `TypeHandle(u32)` into
//!   `TypeArena`; `unify`/`occurs`/`resolve` are methods on `TypeArena`.
//! - **`TypeVar` identity equals its index in `type_vars`.**
//! - **`Box<str>` / `Box<[...]>`** hold the composite type's own data, keeping
//!   ownership clear.
//! - **`Result<(), UnifyError>`** replaces Zig's error union; `EnvArena` + `EnvId`
//!   provide index-based environment access.
//!
//! Dependencies: one-way dependency on `crate::types`
//! (`Type` / `TypeArena` / `TypeOps` / `DynamicOpsRegistry`) and `crate::Ast`
//! (`TypeRef`, referenced only by the GADT backtrack field of `CtorDefInfo`).

use crate::ast::Ast::{
    AstArena, Decl, TypeNode, TypeRef as AstTypeRef, Span,
};
use crate::types::{
    FIRST_DYNAMIC_TYPE_ID, type_def_index_of,
};
use rustc_hash::{FxHashMap, FxHashSet};

// Re-export all type-system symbols from the Type module (breaks the Type↔sema
// circular dependency). sema submodules (Inference.rs / Relations.rs /
// Monomorph.rs) obtain these symbols via `use crate::sema::Sema::*;` glob import.
pub use crate::types::{
    TypeHandle, Type, TypeFamily, DetailId, EnvId, FieldType, TraitMethodSig,
    SemKind, TypeVar, UnifyError,
    TypeArena, TypeDetail, TypeDisplay,
    TypeOps,
    DynamicOpsRegistry,
    RefOps, HeapRefOps,
    STR_TYPE_ID, NULL_TYPE_ID, VOID_TYPE_ID,
    FIRST_INT_TYPE_ID, LAST_INT_TYPE_ID,
    FIRST_FLOAT_TYPE_ID, LAST_FLOAT_TYPE_ID,
    dynamic_type_id,
};

// =========================================================================
// ConcreteEnv / EnvArena — type environments (replaces the old TypeEnv, no
// TypeScheme).
// =========================================================================

/// A type-environment node: own bindings plus an optional parent environment
/// (shared via index).
struct EnvNode {
    bindings: FxHashMap<String, TypeHandle>,
    parent: Option<EnvId>,
}

/// Type-environment arena: manages environment nodes by index, supports parent
/// sharing, and uses no `Rc`/`RefCell`.
///
/// Nodes are append-only and NEVER freed: EnvIds are baked into arena types
/// (ModuleRef carries its module's env), so removing nodes would invalidate
/// live types. A full re-sema builds a fresh arena (SemaResult::new).
pub struct EnvArena {
    envs: Vec<EnvNode>,
    root_id: Option<EnvId>,
}

impl Default for EnvArena {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvArena {
    pub fn new() -> Self {
        EnvArena { envs: Vec::new(), root_id: None }
    }

    /// The top-level environment (no parent) — get-or-create, so every
    /// caller across passes and incremental rechecks shares one root.
    pub fn root(&mut self) -> EnvId {
        if let Some(id) = self.root_id {
            return id;
        }
        let id = EnvId(self.envs.len() as u32);
        self.envs.push(EnvNode {
            bindings: FxHashMap::default(),
            parent: None,
        });
        self.root_id = Some(id);
        id
    }

    /// Create a child environment whose parent is `parent`.
    pub fn child(&mut self, parent: EnvId) -> EnvId {
        let id = EnvId(self.envs.len() as u32);
        self.envs.push(EnvNode {
            bindings: FxHashMap::default(),
            parent: Some(parent),
        });
        id
    }

    /// Names currently defined in `env` (local only, no parent traversal) —
    /// used for predeclare origin manifests.
    pub fn snapshot_names(&self, env: EnvId) -> std::collections::HashSet<String> {
        self.envs[env.0 as usize]
            .bindings
            .keys()
            .cloned()
            .collect()
    }

    /// Define a binding in `env`; returns `false` if a binding with the same name
    /// already exists.
    pub fn define(&mut self, env: EnvId, name: &str, ty: TypeHandle) -> bool {
        let node = &mut self.envs[env.0 as usize];
        if node.bindings.contains_key(name) {
            return false;
        }
        node.bindings.insert(name.to_string(), ty);
        true
    }

    /// Forcefully define a binding in `env` (overwrites any existing binding
    /// with the same name).
    ///
    /// Used during constructor registration: `register_module_aliases` first
    /// registers module path aliases (e.g. "DateTime" → ModuleRef), then
    /// `predeclare_declarations` overrides the alias when registering the
    /// constructor, so `DateTime(...)` resolves to the constructor rather than
    /// the ModuleRef.
    pub fn redefine(&mut self, env: EnvId, name: &str, ty: TypeHandle) {
        let node = &mut self.envs[env.0 as usize];
        node.bindings.insert(name.to_string(), ty);
    }

    /// Look up a name starting from `env` and walking up the parent chain;
    /// returns `None` if not found.
    pub fn lookup(&self, mut env: EnvId, name: &str) -> Option<TypeHandle> {
        loop {
            let node = &self.envs[env.0 as usize];
            if let Some(&ty) = node.bindings.get(name) {
                return Some(ty);
            }
            {
                let p = node.parent?;
                env = p
            }
        }
    }

    /// Look up a name in `env` itself only (does not walk the parent chain);
    /// returns `None` if not found.
    ///
    /// Used for module-qualified access (ModuleRef.field): only searches the
    /// module's own symbols without falling through to the parent env (prevents
    /// `std.io.File.println` from incorrectly resolving to the global `println`).
    pub fn lookup_local(&self, env: EnvId, name: &str) -> Option<TypeHandle> {
        self.envs[env.0 as usize].bindings.get(name).copied()
    }

    /// Look up `name` walking up from `env`, returning the type ONLY when the
    /// binding lives in the ROOT env (the std predeclare/global level). A hit
    /// in any deeper env (locals, same-module definitions, selective-import
    /// items) returns None — those are deliberate bindings that shadow the
    /// global level silently. Used by the bare-call multi-owner zero-silence
    /// check: only root-level resolutions are ambiguity candidates.
    pub fn lookup_at_root(&self, env: EnvId, name: &str) -> Option<TypeHandle> {
        let mut env = env;
        loop {
            let node = &self.envs[env.0 as usize];
            if let Some(&ty) = node.bindings.get(name) {
                return if node.parent.is_none() { Some(ty) } else { None };
            }
            env = node.parent?;
        }
    }

    /// Walk up from `env` looking for a binding named `name` that satisfies
    /// `pred` (skipping same-named bindings that do not satisfy the predicate).
    /// Used for method-call dispatch `recv.method(args)` → `method(recv, args)`,
    /// preventing a local variable from shadowing a same-named free function.
    pub fn lookup_with_pred(
        &self,
        mut env: EnvId,
        name: &str,
        pred: impl Fn(TypeHandle) -> bool,
    ) -> Option<TypeHandle> {
        loop {
            let node = &self.envs[env.0 as usize];
            if let Some(&ty) = node.bindings.get(name) {
                if pred(ty) {
                    return Some(ty);
                }
            }
            {
                let p = node.parent?;
                env = p
            }
        }
    }
}

// =========================================================================
// ConstVal — compile-time constant values.
// =========================================================================

/// A compile-time constant value (mirrors `ConstVal` in ir/meta.zig).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstVal {
    /// Integer literal.
    Int(i128),
    /// Bit pattern of a float literal (interpreted according to the target float type).
    Float(u128),
    /// Boolean literal.
    Bool(bool),
    /// Character literal (Unicode scalar value).
    Char(u32),
}

// =========================================================================
// SemaResult helper structures.
// =========================================================================

/// Records that a bare identifier or call resolved to an implicit `this` access.
/// IR consumes this to generate FieldAccess/MethodCall nodes with the this receiver.
#[derive(Debug, Clone, PartialEq)]
pub enum ImplicitThisAccess {
    /// Bare identifier resolved to instance field: `field` → `this.field`
    Field(Box<str>),
    /// Bare call resolved to instance method: `method(args)` → `this.method(args)`
    Method(Box<str>),
}

/// Semantic information for a single expression.
#[derive(Debug, Clone)]
pub struct ExprInfo {
    /// Type handle of the expression (determines channel width and read/write vtable).
    pub ty: TypeHandle,
    /// Compile-time constant value (when the expression is a constant).
    pub const_val: Option<ConstVal>,
    /// AST handle address of the expression (used as a key).
    pub expr_id: u64,
    /// Type name of the expression (for adt/generic scenarios, eliminating AST
    /// lookbacks on the IR side).
    pub type_name: Option<Box<str>>,
    /// Whether this is a trait object (`Type::TraitObject`): the IR layer uses this
    /// to drive vtable-based dynamic dispatch rather than matching the trait name
    /// by string. Applies to any trait (Iterator/Stream/Iterable, etc.).
    pub is_trait_object: bool,
    /// Whether this is a `&T` / `*T` reference type (preserves reference semantics
    /// at runtime without a deep copy).
    pub is_ref_type: bool,
    /// Distinguishes `&T` (false) from `*T` (true); meaningful only when
    /// `is_ref_type == true`.
    pub is_raw_ref: bool,
    /// When a bare identifier/call resolves to an implicit `this` field/method,
    /// records the target name. IR consumes this to emit FieldAccess/MethodCall.
    pub implicit_this: Option<ImplicitThisAccess>,
}

impl ExprInfo {
    /// Construct a minimal `ExprInfo` with the given `ty` (all other fields set
    /// to their defaults).
    pub fn new(ty: TypeHandle, expr_id: u64) -> Self {
        ExprInfo {
            ty,
            const_val: None,
            expr_id,
            type_name: None,
            is_trait_object: false,
            is_ref_type: false,
            is_raw_ref: false,
            implicit_this: None,
        }
    }
}

/// Kind of a type definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeDefKind {
    /// Algebraic data type.
    Adt,
    /// Record type.
    Record,
    /// Type alias.
    Alias,
    /// Newtype wrapper.
    Newtype,
}

/// Constructor definition info (the flattened constructor from sema AdtInfo).
#[derive(Debug, Clone)]
pub struct CtorDefInfo {
    pub name: Box<str>,
    pub type_name: Box<str>,
    pub field_names: Box<[Option<Box<str>>]>,
    pub field_types: Box<[TypeHandle]>,
    /// Per-field visibility (aligned with `field_names`): private (module-scoped)
    /// unless declared `pub`, even when the type itself is pub.
    pub field_is_pub: Box<[bool]>,
    pub is_newtype: bool,
    /// Return type name of a GADT constructor (only valid for GADTs).
    pub return_type_name: Option<Box<str>>,
    /// Return type `TypeNode` of a GADT constructor (eliminates AST fallback on
    /// the IR side).
    pub return_type_node: Option<AstTypeRef>,
    /// Self-contained representation of field types (does not depend on AST
    /// references), used to fully restore field types across modules (including
    /// composite types such as Array, Nullable, and Ref).
    /// Length matches `field_names`.
    pub field_type_reprs: Box<[TypeRepr]>,
    /// Source span of the type declaration that defines this constructor.
    pub def_span: Span,
    /// Module path of the type declaration that defines this constructor.
    pub def_module: Box<str>,
}

/// Signature info for a type's methods, indexed by `method_idx` (the position in
/// the `methods` array of the type block).
///
/// A self-contained type representation (independent of AST references), used to
/// carry method return-type information across modules. Converted from AST
/// `TypeNode` during `build_method_sig_info`, and restored to a `TypeHandle` in
/// `lookup_method_type` via `type_repr_to_handle`.
#[derive(Debug, Clone)]
pub enum TypeRepr {
    Named(Box<str>),
    ThisType,
    Generic(Box<str>, Box<[TypeRepr]>),
    Nullable(Box<TypeRepr>),
    Ref(Box<TypeRepr>),
    RawPtr(Box<TypeRepr>),
    Function(Box<[TypeRepr]>, Box<TypeRepr>),
    Array(Box<TypeRepr>, Option<u64>),
}

/// Lowering strategy for built-in intrinsic methods.
///
/// Stored in `MethodSigInfo.intrinsic`; the IR layer looks up the method
/// signature via (type_id, method_idx) and then uses this field to select the
/// node kind and `compute_fn`, eliminating special-case name-based lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicKind {
    /// Single-node unary operation (no arguments): len/close/bytes/cancel →
    /// `compute_fn(idx)`.
    UnOp(u32),
    /// Suspend waiting on an event source (no arguments): await → Await node
    /// (unconditional lowering).
    Await,
    /// Channel receive (no arguments): recv → Await node (Channel/Receiver types
    /// only).
    ChannelAwait,
    /// Binary operation (recv + 1 argument): send(value) → `compute_fn(idx)`.
    BinOp(u32),
    /// Ternary operation (recv + 2 arguments): atomic.compare_exchange(expected, new) →
    /// `compute_fn(idx)`.
    TriOp(u32),
}

/// Replaces the legacy `func_sigs` mangled-name ("TypeName.method") registration,
/// driving method dispatch through the structured (type_id, method_idx) key.
#[derive(Debug, Clone)]
pub struct MethodSigInfo {
    pub name: Box<str>,
    pub param_is_ref: Box<[bool]>,
    pub return_is_ref: bool,
    pub is_async: bool,
    pub is_throwing: bool,
    /// Self-contained representation of parameter types (independent of AST
    /// references), used to fully restore parameter types across modules
    /// (including composite types such as Array, Nullable, and Ref).
    pub param_type_reprs: Box<[TypeRepr]>,
    /// Self-contained representation of the return type (independent of AST
    /// references), used to fully resolve nested generic types across modules
    /// (e.g. `Async<Throw<T, E>>`).
    pub return_type_repr: Option<TypeRepr>,
    /// Intrinsic lowering strategy: `None` for ordinary methods (with a body or
    /// trait methods); `Some` for built-in intrinsic methods (no body, lowered
    /// directly to a `compute_fn` node by the IR layer).
    pub intrinsic: Option<IntrinsicKind>,
    /// Whether the method declares its own body. Pure delegations
    /// (`fun m(): R = A.m`) have `has_body == false` — their method slot is an
    /// alias for the bound trait default (dispatch falls through to Path 3).
    pub has_body: bool,
    /// Explicit trait binding from a delegate annotation (`= A.m`): the trait
    /// whose default implementation this method is bound to. Also present when
    /// the annotation coexists with a body (`override fun m(): R = A.m { ... }`),
    /// where it fixes the target of `super.m()` under multi-trait conflicts.
    pub delegate_trait: Option<Box<str>>,
    /// Method visibility: private (module-scoped) unless declared `pub`,
    /// even when the type is pub.
    pub is_pub: bool,
}

/// Type definition info (replaces IRBuilder's type_table + ctor_table).
#[derive(Debug, Clone)]
pub struct TypeDefInfo {
    pub name: Box<str>,
    pub kind: TypeDefKind,
    /// adt/newtype/error_newtype: list of constructors.
    /// record: `constructors[0]` holds the fields (name == type_name).
    /// alias: empty slice.
    pub constructors: Box<[CtorDefInfo]>,
    pub type_params: Box<[Box<str>]>,
    /// alias/newtype only: target type name.
    pub target_type_name: Option<Box<str>>,
    /// alias/newtype only: target type descriptor.
    pub target_type: Option<TypeHandle>,
    /// Method signature table for the type block, indexed by `method_idx` (AST
    /// declaration order). An empty slice means the type has no methods (alias,
    /// or a record/adt without methods).
    pub methods: Box<[MethodSigInfo]>,
    /// Concrete base type names, declaration order (inheritance). Field layout
    /// = each base's fields (in order, generic args applied) then own fields;
    /// method lookup = own → bases in order → trait defaults. Empty for
    /// non-child types.
    pub bases: Box<[Box<str>]>,
}

/// Trait definition info (replaces the signature portion of IRBuilder's
/// trait_table).
#[derive(Debug, Clone)]
pub struct TraitDefInfo {
    pub name: Box<str>,
    /// Parent trait names (declaration order). `trait Pet(Animal)`.
    pub parents: Box<[Box<str>]>,
    pub methods: Box<[TraitMethodSig]>,
}

/// Function signature reference (embedded in `ExprInfo`, meaningful only for
/// callee expressions).
#[derive(Debug, Clone)]
pub struct FnSigRef {
    pub param_types: Box<[TypeHandle]>,
    pub return_type: TypeHandle,
    pub is_async: bool,
    pub is_throwing: bool,
}

/// Function signature info (replaces IRBuilder's func_generic_info).
#[derive(Debug, Clone)]
pub struct FuncSigInfo {
    /// Function name or mangled name (TypeName.method).
    pub name: Box<str>,
    /// Owning module (registration-qualified; top-level funs only).
    pub module_name: Box<str>,
    pub type_params: Box<[Box<str>]>,
    pub return_type: TypeHandle,
    /// Whether each parameter has `&T` reference semantics.
    pub param_is_ref: Box<[bool]>,
    pub return_is_ref: bool,
    pub is_async: bool,
    pub is_throwing: bool,
}

/// Import alias target (distinguishes module references from symbol references).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AliasTarget {
    /// Module short name → full module path.
    /// `import std.time.Calendar` → "Calendar" → `AliasTarget::Module("std.time.Calendar")`
    Module(Box<str>),
    /// Function/constant short name → mangled name.
    /// `import std.time.Calendar { is_leap_year }` → "is_leap_year" → `AliasTarget::Symbol("std.time.Calendar.is_leap_year")`
    Symbol(Box<str>),
}

/// Channel layout (the channel allocation scheme for a monomorphized instance).
#[derive(Debug, Clone)]
pub struct ChanLayout {
    pub local_chan_count: u16,
    pub return_channel: u16,
    pub local_offsets: Box<[u32]>,
    pub chan_types: Box<[TypeHandle]>,
    pub chan_total_bytes: u32,
}

impl ChanLayout {
    /// Empty layout (placeholder when channel allocation has not been computed).
    pub fn empty() -> Self {
        ChanLayout {
            local_chan_count: 0,
            return_channel: 0,
            local_offsets: Box::new([]),
            chan_types: Box::new([]),
            chan_total_bytes: 0,
        }
    }
}

/// Field-access metadata (for runtime `field_id` lookup).
#[derive(Debug, Clone)]
pub struct FieldAccessInfo {
    pub obj_type: TypeHandle,
    pub field_idx: u16,
    pub field_type: TypeHandle,
}

/// Method-dispatch metadata (trait method → concrete impl function).
#[derive(Debug, Clone, Copy)]
pub struct DispatchInfo {
    pub trait_id: u16,
    pub method_idx: u16,
    pub impl_fn_idx: u16,
    /// Monomorphization instance ID for generic method calls (0 for non-generic).
    pub instance_id: u32,
    /// Language-level intrinsic marker (await/recv etc. are recognized uniformly
    /// by sema, independent of type registration). User-defined types (Timer,
    /// etc.) do not register `MethodSigInfo.intrinsic`, but await/recv are
    /// universal semantics tagged here by sema during `MethodCall` inference.
    pub intrinsic: Option<IntrinsicKind>,
}

/// The trait-default binding of a method on an implementing type — the single
/// source of truth shared by override/delegate validation, `super` resolution,
/// and trait-default instance collection.
#[derive(Debug, Clone)]
pub enum MethodBinding {
    /// The method's default layer is bound to `trait_name` (explicitly via a
    /// delegate `= A.m`, or implicitly because it is the unique provider).
    /// `overridden`: the type declares its own method (with a body) on top of
    /// the default.
    Bound { trait_name: Box<str>, overridden: bool },
    /// No implemented trait provides a default for this method.
    Unbound,
    /// Multiple traits provide a default and no delegate disambiguates.
    Ambiguous(Vec<Box<str>>),
}

/// Resolved reflect metadata.
#[derive(Debug, Clone, Copy)]
pub struct ReflectMeta {
    pub ty: TypeHandle,
}

/// A monomorphization instance (one generic function + one set of type_args →
/// one instance).
#[derive(Debug, Clone)]
pub struct MonomorphInstance {
    pub instance_id: u32,
    pub func_name: Box<str>,
    /// Name of the module containing the function (used in the composite
    /// `expr_types` key to keep keys consistent across cross-module
    /// monomorphization).
    pub module_name: Box<str>,
    pub type_args: Box<[TypeHandle]>,
    pub chan_layout: ChanLayout,
    pub return_type: TypeHandle,
    pub is_async: bool,
    /// Instance-local expression type table (key =
    /// `module_expr_key(module_name, expr_id)`).
    pub expr_types: FxHashMap<u64, ExprInfo>,
    /// Field-access metadata (key = AST `field_access` Expr handle address).
    pub field_accesses: FxHashMap<u64, FieldAccessInfo>,
}

/// Monomorphization instance for a trait default method.
///
/// Each type that implements a trait but does not explicitly override the method
/// corresponds to a specialized instance. Collected during the later Sema phase
/// by `Monomorph::collect_trait_default_instances`, and pre-registered and
/// compiled into specialized subgraphs by the IR layer (IrBuilder).
///
/// Key semantics: `(type_id, trait_idx, method_idx)` matches the key used in
/// `Ir.trait_default_subgraphs`.
#[derive(Debug, Clone)]
pub struct TraitDefaultInstance {
    /// `type_id` of the implementing type (matches the `type_id` on `Type`).
    pub type_id: u16,
    /// Name of the implementing type (e.g. "Lt", "Ordering").
    pub type_name: Box<str>,
    /// Index of the trait in `trait_defs`.
    pub trait_idx: u16,
    /// Trait name (e.g. "Greet", "Show").
    pub trait_name: Box<str>,
    /// Index of the method within the trait's methods (a default method with a
    /// body).
    pub method_idx: u16,
}

/// An inherited-method expansion instance (inheritance). The child's method
/// table gains an entry at `method_idx` whose BODY is the base type's method
/// (located in `base_module`'s AST by `base_method_idx`); the IR stage
/// compiles it with the CHILD as receiver type — per-child compilation is
/// what makes `this`-dispatch late-bound (same mechanism as
/// `TraitDefaultInstance`).
#[derive(Debug, Clone)]
pub struct InheritedMethodInstance {
    /// `type_id` of the child type.
    pub type_id: u16,
    /// Name of the child type.
    pub type_name: Box<str>,
    /// The method's index within the child's merged method table
    /// (own methods first in AST order, then inherited ones in base order).
    pub method_idx: u16,
    /// Module path of the base's declaring module (AST lookup hint).
    pub base_module: Box<str>,
    /// Name of the base type owning the method body.
    pub base_type_name: Box<str>,
    /// Index of the method within the base type's method table.
    pub base_method_idx: u16,
}

/// Coroutine metadata (the product of the async-function state-machine
/// transformation).
///
/// Sema emits minimal metadata: `func_idx` locates the function and
/// `segment_count` describes the number of state segments. The full
/// state-machine transformation (segment/frame/defer/catch/loop tables) is built
/// by the IR layer based on this metadata, not maintained in the Sema layer,
/// preserving the separation of concerns between Sema and IR.
#[derive(Debug, Clone)]
pub struct CoroutineMeta {
    /// Index of the async function (index into the `functions` table).
    pub func_idx: u16,
    /// Number of state segments.
    pub segment_count: u16,
}

/// Capture mode for a variable referenced from a nested scope (lambda / defer /
/// nested function).
///
/// This is the single unified descriptor for "how does a nested scope access an
/// outer local variable", replacing the three parallel mechanisms (frame-chain
/// for defer/non-escaping closures, Cell-upvalues for escaping closures, and the
/// Bug #49 WriteBack-to-original hack) with one explicit per-capture decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    /// by-val snapshot: the nested scope captures the value at the declaration
    /// site (the `val` copy-node provides this naturally). Reads return the
    /// declaration-time value and never observe later mutations.
    Snapshot,
    /// by-ref reference: the nested scope captures a reference to the mutable
    /// slot. Reads reflect the latest value and writes propagate back. Used for
    /// `var` bindings and always forced for `defer` bodies (defer semantics
    /// require observing the value at function/block exit, not at defer-site).
    Reference,
}

/// Description of a single captured variable for a nested scope.
#[derive(Debug, Clone)]
pub struct CaptureInfo {
    /// Variable name (e.g. `x`, `this`).
    pub name: Box<str>,
    /// Module-unique key of the *declaration site* (ValDecl/VarDecl/Param expr
    /// or stmt handle, via `module_expr_key`). The IR builder uses this to
    /// locate the original NodeId so WriteBacks target the root-frame slot.
    pub decl_key: u64,
    /// by-val or by-ref.
    pub mode: CaptureMode,
}

// =========================================================================
// SemaError — semantic errors.
// =========================================================================

/// A semantic error.
#[derive(Debug, Clone)]
pub struct SemaError {
    pub message: Box<str>,
    pub line: u32,
    pub column: u32,
    /// Optional file path override: when set, diagnostics print this path instead
    /// of the module-check loop's path. Used by cross-module checks (e.g.
    /// `check_duplicate_constructors`) where the warning originates from a
    /// different module than the one currently being checked.
    pub file_path: Option<Box<str>>,
}

impl SemaError {
    pub fn new(message: &str, line: u32, column: u32) -> Self {
        SemaError {
            message: message.into(),
            line,
            column,
            file_path: None,
        }
    }

    /// Create a `SemaError` with an explicit file path override.
    pub fn new_with_path(message: &str, file_path: &str, line: u32, column: u32) -> Self {
        SemaError {
            message: message.into(),
            line,
            column,
            file_path: Some(file_path.into()),
        }
    }
}

// =========================================================================
// ModuleOwnership — module → global table indices owned by that module.
// =========================================================================

/// Module → global table indices owned by that module.
/// Used to drain old entries during incremental recheck.
#[derive(Default, Clone)]
pub struct ModuleOwnership {
    /// type_defs u16 indices owned by each module
    pub type_def_indices: FxHashMap<String, FxHashSet<u16>>,
    /// func_sigs u16 indices owned by each module
    pub func_sig_indices: FxHashMap<String, FxHashSet<u16>>,
    /// trait_defs u16 indices owned by each module
    pub trait_def_indices: FxHashMap<String, FxHashSet<u16>>,
    /// witness_table (trait_name, type_id) pairs owned by each module
    pub witness_keys: FxHashMap<String, FxHashSet<(Box<str>, u16)>>,
    /// import_aliases keys owned by each module
    pub alias_keys: FxHashMap<String, FxHashSet<String>>,
    /// monomorph_instances u32 indices owned by each module
    pub monomorph_indices: FxHashMap<String, FxHashSet<u32>>,
    /// field_id_map keys (type_name\x00field_name) owned by each module
    pub field_id_keys: FxHashMap<String, FxHashSet<String>>,
    /// expr_types u64 keys owned by each module (reverse map for purge)
    pub expr_type_keys: FxHashMap<String, FxHashSet<u64>>,
    /// captures u64 keys (nested-scope entry expressions) owned by each module.
    pub capture_keys: FxHashMap<String, FxHashSet<u64>>,
}

// =========================================================================
// SemaResult — graph-construction metadata produced by sema.
// =========================================================================

/// Graph-construction metadata produced by sema.
///
/// Upgraded from a "checker" to a "graph-construction driver", emitting all the
/// metadata needed to build the graph. All fields own their data (`Box<str>` /
/// `Vec` / `FxHashMap`), requiring no additional arena ownership.
pub struct SemaResult {
    /// Bug #103 layering manifest: std-layer binding name → origin module
    /// logical paths (multi-valued: `parse` lives in std.core.types.I8 AND
    /// std.json.Parse — a single-valued map silently dropped the earlier
    /// origin). The import re-export filters on this.
    pub std_binding_origins: rustc_hash::FxHashMap<String, Vec<String>>,
    /// Expression → type info (determines channel width); key = AST expression
    /// handle address.
    pub expr_types: FxHashMap<u64, ExprInfo>,
    /// Compile-time errors.
    pub errors: Vec<SemaError>,
    /// Compile-time warnings (do not stop compilation).
    pub warnings: Vec<SemaError>,
    /// Whether any error occurred.
    pub has_error: bool,
    /// Type definition table (replaces IRBuilder's type_table + ctor_table).
    /// Keyed by a u16 index that never recycles; entries are truly removed on
    /// `purge_module` (no stale holes).
    pub type_defs: FxHashMap<u16, TypeDefInfo>,
    /// u16 index allocator for `type_defs` (never recycles).
    pub next_type_def_id: u16,
    /// Type name → index into `type_defs`. Keys are CANONICAL type names:
    /// user-module types are module-qualified (`src.Main.List`), std/builtin
    /// types keep their bare name (one std tree, unique by construction).
    pub type_def_index: FxHashMap<crate::sema::Symbols::Sym, u16>,
    /// Dynamic type_id → type name (reverse index for O(1) lookup).
    /// Updated in tandem with `type_defs` (put_type_def / purge_module).
    pub type_id_to_name: FxHashMap<u16, Box<str>>,
    /// The module currently being populated/checked — the module context
    /// `resolve_type_key` resolves bare names against. Maintained by
    /// populate_module and check_module_with_env.
    /// 可见性收紧到 sema 子树(NAME_RESOLUTION_PLAN S3 收尾):IR/CLI
    /// 曾经直读此字段 = 陈旧上下文 bug 族;S3 显式 `*_in` 变体落地后读者
    /// 清零,`pub(in crate::sema)` 让子树外的再访问直接编译报错。
    pub(in crate::sema) current_module_name: String,
    /// Import bookkeeping for module-scoped type resolution: module name →
    /// logical paths of the user modules it imports (std imports are
    /// excluded — the bare-key fallback covers them). Filled by
    /// process_import_decls.
    pub module_imports: FxHashMap<String, Vec<String>>,
    /// Logical paths of all user (non-std/builtin) modules in the compile,
    /// used to map import spellings to module paths.
    pub user_module_paths: Vec<String>,
    /// Bare type names of the module currently being POPULATED (declared but
    /// possibly not yet registered — forward references). Lets
    /// `resolve_type_key_in` resolve a field/alias/base referencing a type
    /// declared later in the same module to its canonical key before the
    /// registration lands.
    pub(in crate::sema) pending_own_types: std::collections::HashSet<String>,
    /// Bare trait names of the module currently being populated — the trait
    /// twin of `pending_own_types` (forward-referenced parent traits, impl
    /// bounds ahead of the trait's declaration).
    pub(in crate::sema) pending_own_traits: std::collections::HashSet<String>,
    /// Trait definition table.
    pub trait_defs: FxHashMap<u16, TraitDefInfo>,
    /// u16 index allocator for `trait_defs` (never recycles).
    pub next_trait_def_id: u16,
    /// Trait name → index into `trait_defs`.
    pub trait_def_index: FxHashMap<crate::sema::Symbols::Sym, u16>,
    /// Function signature table.
    pub func_sigs: FxHashMap<u16, FuncSigInfo>,
    /// u16 index allocator for `func_sigs` (never recycles).
    pub next_func_sig_id: u16,
    /// Function name → index into `func_sigs` (module-qualified keys,
    /// "module\x00name").
    pub func_sig_index: FxHashMap<crate::sema::Symbols::Sym, u16>,
    /// Bare function name → owning modules in registration order (bare-name
    /// resolution for `get_func_sig`: unique owner wins, contested → first).
    pub func_sig_owners: FxHashMap<String, Vec<String>>,
    /// Shared variable/symbol environment arena (moved from InferContext).
    ///
    /// One arena for the whole compile: EnvIds baked into arena types
    /// (ModuleRef carries its module's env) stay valid in EVERY context —
    /// the temporary InferContext that monomorphization builds to re-infer
    /// generic bodies used to have a private arena, so replayed HM-pass
    /// types (instantiation mode resolves Idents through recorded
    /// expr_types) indexed a 2-node arena with env ids from the original
    /// pass and panicked (out of bounds) on any module-qualified call
    /// inside a generic function body.
    pub env: EnvArena,
    /// Module path (dotted) → module-specific EnvId (shared with the env
    /// arena). Kept across incremental rechecks so a re-checked module
    /// REUSES its env and redefines bindings in place — EnvIds baked into
    /// recorded types (ModuleRef) then resolve to the EDITED symbols, not
    /// stale pre-edit bindings.
    pub module_envs: FxHashMap<String, EnvId>,
    /// Coroutine metadata table.
    pub coroutine_metas: Vec<CoroutineMeta>,
    /// Constructor name → list of (type_def_index << 16 | ctor_index).
    /// Supports multiple types having same-named constructors (e.g. `FileKind.File`
    /// and `type File`); disambiguation is done by type context or qualified names.
    /// 名称驻留表(S1):Sym ↔ 字符串唯一身份,五张 Sym 键表的登记/读取通道。
    pub symbols: crate::sema::Symbols::Symbols,

    pub ctor_def_index: FxHashMap<crate::sema::Symbols::Sym, Vec<u32>>,
    /// Import alias table: short name → alias target.
    pub import_aliases: FxHashMap<String, AliasTarget>,
    /// Monomorphization instance table.
    pub monomorph_instances: Vec<MonomorphInstance>,
    /// Monomorphization instance name → index into `monomorph_instances`.
    pub monomorph_index: FxHashMap<u64, u32>,
    /// Trait-default-method monomorphization instance table (collected during
    /// the later Sema phase by the Monomorph module).
    pub trait_default_instances: Vec<TraitDefaultInstance>,
    /// Inherited-method expansion instances (inheritance), consumed by the IR
    /// stage to compile base method bodies with the child as receiver.
    pub inherited_method_instances: Vec<InheritedMethodInstance>,
    /// Dynamic ops registry (ops for user types, replaces TypeDescriptorPool).
    pub dynamic_ops: DynamicOpsRegistry,
    /// Call-site → instance mapping.
    pub call_instantiations: FxHashMap<u64, u32>,
    /// Resolved constructor calls (NAME_RESOLUTION_PLAN S2): call-expr key →
    /// Sym of the CONSTRUCTED TYPE's canonical name. Sema's single resolution
    /// decision; the IR construct path consumes it FIRST — the string-keyed
    /// scope lookups (and their registration-order / same-named-variant
    /// hazards) become a measured fallback.
    pub ctor_resolutions: FxHashMap<u64, crate::sema::Symbols::Sym>,
    /// Resolved method-dispatch targets (S4): call-expr key →
    /// (type_def_idx, method_idx)。Sema's single dispatch resolution for
    /// `recv.method(...)`; IR path-2 consumes it FIRST — the name-based
    /// method_idx lookup becomes a measured fallback.
    pub dispatch_targets: FxHashMap<u64, (u16, u16)>,
    /// Field-access metadata (global; key = AST `field_access` Expr handle
    /// address).
    pub field_accesses: FxHashMap<u64, FieldAccessInfo>,
    /// Method-dispatch metadata (key = AST `call` Expr handle address).
    pub method_dispatches: FxHashMap<u64, DispatchInfo>,
    /// Resolved reflect metadata.
    pub reflect_metas: FxHashMap<u64, ReflectMeta>,
    /// Resolved type handles (key = AST Expr handle address).
    pub resolved_types: FxHashMap<u64, TypeHandle>,
    /// Field ID map (key = "type_name\x00field_name" → field_id).
    /// ADT/newtype/error_newtype: `__tag=0`, fields start at 1.
    /// Record: fields in declaration order, 0..N-1.
    pub field_id_map: FxHashMap<crate::sema::Symbols::Sym, u16>,
    /// Witness table (static dispatch table for trait implementations).
    ///
    /// Maintained by `InferContext` during sema checking and accumulated across
    /// modules; mirrored into this field after `check` completes so the IR layer
    /// (IrBuilder) can access trait method dispatch info.
    pub witness_table: WitnessTable,
    /// Set of recv ExprId keys for module-function calls (Zig `@This` semantics).
    ///
    /// When `import std.time.Duration` and the module defines `pub type Duration`,
    /// predeclare uses `redefine` to overwrite the ModuleRef with the constructor
    /// Fn. Sema's `MethodCall` path 0b detects this and records the recv's expr
    /// key in this set, so the IR compiler omits the recv (`Duration.from_millis(100)`
    /// → `from_millis(100)` rather than `from_millis(Duration, 100)`).
    pub module_func_recv_exprs: FxHashSet<u64>,
    /// Module-qualified call targets (root fix for module short-name
    /// collisions): recv expr key → the import-resolved module LOGICAL path
    /// ("sub.Parse"). Sema's Path 0a resolves the qualifier through the
    /// import binding (env-aware, unambiguous); IR consumes this to bind
    /// `Parse.parse(...)` by the full mangled key instead of the contested
    /// short key, so a user module named like a std module no longer errors
    /// or misbinds (BOOTSTRAP 1C).
    pub module_func_call_targets: FxHashMap<u64, Box<str>>,
    /// S2c: bare free-function calls adjudicated by a selective-import item
    /// (explicit binding beats the std predeclare level). Key =
    /// module_expr_key(owner module, call expr) — same family as
    /// call_instantiations; value = target module logical path. Consumed by
    /// the IR bare-call branch to bind by the full mangled key, bypassing
    /// the std-contested bare-key tripwire.
    pub bare_call_targets: FxHashMap<u64, Box<str>>,
    /// Module-constant-access recv ExprId key → mangled name
    /// (module_path.field).
    ///
    /// When a `FieldAccess` like `Math.PI` has a ModuleRef receiver and the field
    /// is a module-level constant, sema records the recv's expr key → the global
    /// variable mangled name (e.g. "std.math.Math.PI") in this map. The IR
    /// compiler uses this to skip recv compilation and emit `compile_global_load`
    /// to read the global variable slot directly, taking the same path as local
    /// global-variable access and preventing the module name from being compiled
    /// into a zombie Const node.
    pub module_const_recv_exprs: FxHashMap<u64, String>,
    /// Pattern constructor disambiguation results: (module_name, pattern_id) → type_name.
    /// Stored by sema when multiple types share the same constructor name; the IR
    /// builder queries this to set `pattern_type_names` for runtime disambiguation.
    pub pattern_ctor_types: FxHashMap<(String, u32), Box<str>>,
    /// Unified capture table: for each nested scope (lambda / defer / nested
    /// function), the list of outer variables it captures and the capture mode.
    /// Key = module-unique key of the nested-scope entry expression
    /// (`module_expr_key`). Produced by Sema as the single source of truth;
    /// consumed by the IR builder (replaces its `collect_free_idents_expr` scan
    /// and the 7-branch assignment decision tree).
    pub captures: FxHashMap<u64, Vec<CaptureInfo>>,
    /// `super.method(...)` static-dispatch results: call-expr key
    /// (`module_expr_key`) → (trait_def_idx, method_idx within the trait).
    /// Produced by sema when inferring a method call whose receiver is `super`;
    /// consumed by the IR builder to bypass the type's own override (Path 2)
    /// and the vtable (Path 1), targeting the trait-default subgraph directly.
    pub super_dispatches: FxHashMap<u64, (u16, u16)>,
    /// `super.m()` resolved to a CONCRETE BASE type's method (inheritance):
    /// expr_key → (base type name, method name). Consumed by the IR builder
    /// ahead of the trait-default path — target is the base's own method
    /// subgraph (one level up), receiver is the enclosing `this`.
    pub super_base_dispatches: FxHashMap<u64, (Box<str>, Box<str>)>,
    /// `(type_name, trait_name, method_name)` triples for which a `super` call
    /// was resolved. Drives trait-default instance generation: an overriding
    /// type normally gets no specialized default subgraph, but one must exist
    /// when its override calls `super` into the default layer.
    pub super_targets: FxHashSet<(Box<str>, Box<str>, Box<str>)>,
    /// Phase 2: module-origin index for incremental purge
    pub module_ownership: ModuleOwnership,
}

impl Default for SemaResult {
    fn default() -> Self {
        Self::new()
    }
}


/// Generates the standard "table + index + put/get" trio of registration
/// functions. `$put`/`$get` are the method names, `$field` is the table field
/// name, `$index` is the index field name, `$ty` is the element type, and
/// `$next_id` is the u16 allocator field (never recycles).
macro_rules! define_table_registry {
    ($put:ident, $get:ident, $field:ident, $index:ident, $ty:ty, $next_id:ident) => {
        /// Insert an element and register its index; returns `false` on a duplicate
        /// name. The u16 index is allocated from `$next_id` and never recycles.
        pub fn $put(&mut self, def: $ty) -> bool {
            if self.$index.contains_key(def.name.as_ref()) {
                return false;
            }
            assert!(
                self.$next_id < u16::MAX,
                concat!(stringify!($field), " index overflow: too many entries"),
            );
            let idx = self.$next_id;
            self.$next_id += 1;
            self.$index.insert(def.name.to_string(), idx);
            self.$field.insert(idx, def);
            true
        }
        /// Look up an element by name.
        pub fn $get(&self, name: &str) -> Option<&$ty> {
            let idx = *self.$index.get(name)?;
            self.$field.get(&idx)
        }
    };
}

impl SemaResult {
    pub fn new() -> Self {
        SemaResult {
            std_binding_origins: FxHashMap::default(),
            expr_types: FxHashMap::default(),
            errors: Vec::new(),
            warnings: Vec::new(),
            has_error: false,
            type_defs: FxHashMap::default(),
            next_type_def_id: 0,
            type_def_index: FxHashMap::default(),
            symbols: crate::sema::Symbols::Symbols::new(),
            type_id_to_name: FxHashMap::default(),
            current_module_name: String::new(),
            module_imports: FxHashMap::default(),
            user_module_paths: Vec::new(),
            pending_own_types: std::collections::HashSet::new(),
            pending_own_traits: std::collections::HashSet::new(),
            trait_defs: FxHashMap::default(),
            next_trait_def_id: 0,
            trait_def_index: FxHashMap::default(),
            func_sigs: FxHashMap::default(),
            next_func_sig_id: 0,
            func_sig_index: FxHashMap::default(),
            func_sig_owners: FxHashMap::default(),
            env: EnvArena::new(),
            module_envs: FxHashMap::default(),
            coroutine_metas: Vec::new(),
            ctor_def_index: FxHashMap::default(),
            import_aliases: FxHashMap::default(),
            monomorph_instances: Vec::new(),
            monomorph_index: FxHashMap::default(),
            trait_default_instances: Vec::new(),
            inherited_method_instances: Vec::new(),
            dynamic_ops: DynamicOpsRegistry::new(),
            call_instantiations: FxHashMap::default(),
            ctor_resolutions: FxHashMap::default(),
            dispatch_targets: FxHashMap::default(),
            field_accesses: FxHashMap::default(),
            method_dispatches: FxHashMap::default(),
            reflect_metas: FxHashMap::default(),
            resolved_types: FxHashMap::default(),
            field_id_map: FxHashMap::default(),
            witness_table: WitnessTable::new(),
            module_func_recv_exprs: FxHashSet::default(),
            module_func_call_targets: FxHashMap::default(),
            bare_call_targets: FxHashMap::default(),
            module_const_recv_exprs: FxHashMap::default(),
            pattern_ctor_types: FxHashMap::default(),
            captures: FxHashMap::default(),
            super_dispatches: FxHashMap::default(),
            super_base_dispatches: FxHashMap::default(),
            super_targets: FxHashSet::default(),
            module_ownership: ModuleOwnership::default(),
        }
    }

    // ── Expressions ──

    /// Record the type of an expression.
    pub fn put_expr(&mut self, expr_id: u64, info: ExprInfo) {
        self.expr_types.insert(expr_id, info);
    }

    /// Record the capture list for a nested scope (lambda / defer / nested
    /// function). `scope_key` is the module-unique key of the nested-scope entry
    /// expression; `module_name` is the owning module (for incremental purge).
    pub fn put_captures(&mut self, scope_key: u64, module_name: &str, captures: Vec<CaptureInfo>) {
        self.captures.insert(scope_key, captures);
        self.module_ownership.capture_keys
            .entry(module_name.to_string())
            .or_default()
            .insert(scope_key);
    }

    /// Look up the capture list for a nested scope. Returns an empty slice if
    /// none recorded (the scope captures nothing).
    pub fn get_captures(&self, scope_key: u64) -> &[CaptureInfo] {
        self.captures.get(&scope_key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Look up the type of an expression.
    pub fn get_expr(&self, expr_id: u64) -> Option<&ExprInfo> {
        self.expr_types.get(&expr_id)
    }

    // ── Import aliases ──

    /// Register an import alias; returns `false` on a duplicate short name.
    pub fn put_import_alias(&mut self, short_name: &str, target: AliasTarget, module_name: &str) -> bool {
        if self.import_aliases.contains_key(short_name) {
            return false;
        }
        self.import_aliases.insert(short_name.to_string(), target);
        // Record module ownership for incremental purge (import_aliases key).
        self.module_ownership.alias_keys
            .entry(module_name.to_string())
            .or_default()
            .insert(short_name.to_string());
        true
    }

    /// Look up an import alias.
    pub fn get_import_alias(&self, short_name: &str) -> Option<&AliasTarget> {
        self.import_aliases.get(short_name)
    }

    // ── Errors ──

    /// Record an error.
    pub fn add_error(&mut self, err: SemaError) {
        self.has_error = true;
        self.errors.push(err);
    }

    /// Record a warning (does not set has_error).
    pub fn add_warning(&mut self, err: SemaError) {
        self.warnings.push(err);
    }

    // ── Type definitions ──

    /// Add a type definition and register `type_def_index` / `ctor_def_index`,
    /// automatically populating `field_id_map` as well.
    ///
    /// Returns `false` on a type-name conflict (same-named types cannot be
    /// redefined). Constructor names are allowed to conflict across different
    /// types: all matching constructors are registered in `ctor_def_index`
    /// (multi-map), and disambiguation is deferred to type-context resolution
    /// or qualified-name syntax (`Type.Ctor`).
    pub fn put_type_def(&mut self, def: TypeDefInfo, module_name: &str) -> bool {
        // The type namespace is module-scoped: `def.name` arrives CANONICAL
        // (user types module-qualified `src.Main.List` by
        // `ast_type_decl_to_type_def`; std/builtin/synthetic keep the bare
        // name), so a local `type List` alongside std.collections.List no
        // longer conflicts — the two coexist and every downstream consumer
        // (runtime equality, match, dispatch, reflect) compares the
        // qualified names. Remaining conflicts: a canonical key held by
        // ANOTHER module (possible only for bare std keys — two std modules
        // declaring the same bare type) or a same-module re-registration
        // (the populate pipeline runs twice per module — idempotent no-op).
        if let Some(existing_idx) = self.type_def_idx(def.name.as_ref()) {
            let existing_module = self
                .module_ownership
                .type_def_indices
                .iter()
                .find_map(|(m, ids)| {
                    if ids.contains(&existing_idx) {
                        Some(m.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("");
            // Same-module re-registration: silent no-op.
            if existing_module == module_name {
                return false;
            }
            self.add_error(SemaError::new_with_path(
                &format!("duplicate type definition: '{}'", def.name),
                module_name,
                1,
                1,
            ));
            return false;
        }
        // u16 index overflow check (aligned with register in TypeDesc.rs).
        assert!(
            self.next_type_def_id < u16::MAX,
            "type_def index overflow: too many type definitions"
        );
        let idx = self.next_type_def_id;
        self.next_type_def_id += 1;
        self.populate_field_ids(&def, module_name);
        // Register all constructors (same-named constructors across different
        // types are appended to the multi-map entry).
        for (ci, ctor) in def.constructors.iter().enumerate() {
            let packed_idx: u32 = ((idx as u32) << 16) | (ci as u32);
            self.ctor_def_push(ctor.name.as_ref(), packed_idx);
        }
        self.type_def_set(def.name.as_ref(), idx);
        // Record module ownership for incremental purge (type_def index).
        self.module_ownership.type_def_indices
            .entry(module_name.to_string())
            .or_default()
            .insert(idx);
        // Populate the type_id → name reverse index (O(1) lookup replacing
        // the former O(n) linear scan in collect_trait_default_instances).
        let type_id = dynamic_type_id(idx);
        self.type_id_to_name.insert(type_id, def.name.clone());
        self.type_defs.insert(idx, def);
        true
    }

    /// Populate `field_id_map` according to the type_def's kind rules.
    /// - adt/newtype/error_newtype: `__tag=0`, fields start at 1.
    /// - record: fields in declaration order, 0..N-1.
    /// - alias: no fields.
    fn populate_field_ids(&mut self, def: &TypeDefInfo, module_name: &str) {
        match def.kind {
            TypeDefKind::Adt => {
                for ctor in def.constructors.iter() {
                    for (fi, fname) in ctor.field_names.iter().enumerate() {
                        if let Some(name) = fname {
                            let field_id = (fi + 1) as u16;
                            self.put_field_id(&def.name, name, field_id, module_name);
                        }
                    }
                }
                self.put_field_id(&def.name, "__tag", 0, module_name);
            }
            TypeDefKind::Newtype => {
                for (fi, fname) in def.constructors.iter().flat_map(|c| c.field_names.iter()).enumerate() {
                    let field_id = (fi + 1) as u16;
                    match fname {
                        Some(name) => self.put_field_id(&def.name, name, field_id, module_name),
                        None => {
                            let positional = format!("_{}", fi);
                            self.put_field_id(&def.name, &positional, field_id, module_name);
                        }
                    }
                }
                self.put_field_id(&def.name, "__tag", 0, module_name);
            }
            TypeDefKind::Record => {
                if let Some(ctor) = def.constructors.first() {
                    for (fi, fname) in ctor.field_names.iter().enumerate() {
                        if let Some(name) = fname {
                            let field_id = fi as u16;
                            self.put_field_id(&def.name, name, field_id, module_name);
                        }
                    }
                }
            }
            TypeDefKind::Alias => {}
        }
    }

    /// Build the `field_id_map` key: `"type_name\x00field_name"`.
    fn make_field_key(type_name: &str, field_name: &str) -> String {
        format!("{}\0{}", type_name, field_name)
    }

    /// Build the `field_id_map` key and insert it (overwrites if already present).
    fn put_field_id(&mut self, type_name: &str, field_name: &str, field_id: u16, module_name: &str) {
        let key = Self::make_field_key(type_name, field_name);
        let ks = self.symbols.intern(&key);
        self.field_id_map.insert(ks, field_id);
        // Record module ownership for incremental purge (field_id_map key).
        self.module_ownership.field_id_keys
            .entry(module_name.to_string())
            .or_default()
            .insert(key);
    }

    /// Look up a field_id (returns `None` if not found).
    /// Key = "type_name\x00field_name".
    pub fn lookup_field_id(&self, type_name: &str, field_name: &str) -> Option<u16> {
        let key = Self::make_field_key(type_name, field_name);
        self.symbols.find(&key).and_then(|s| self.field_id_map.get(&s).copied())
    }

    /// Look up a type definition by name (concrete array names map to the
    /// synthetic builtin "array" def — see `canonical_type_name`).
    ///
    /// Module-scoped resolution for every caller: a CANONICAL (dotted) name
    /// passes through exactly; a BARE name resolves own-module-first (a
    /// local `type List` shadows the std type), then imports, then the
    /// std/builtin bare key, then a unique global user type. The exact-hit
    /// fast path is deliberately NOT taken for bare names — std occupying
    /// the bare key must not shadow the current module's own type.
    pub fn get_type_def(&self, name: &str) -> Option<&TypeDefInfo> {
        let key = Self::canonical_type_name(name);
        let resolved = self.resolve_type_key(key);
        let idx = self.type_def_idx(&resolved)?;
        self.type_defs.get(&idx)
    }

    /// Module-scoped bare-name → canonical type key resolution.
    ///
    /// Resolution order (the module-scoped type system's single choke point):
    /// a dotted input is treated as already canonical (exact pass-through);
    /// a bare name resolves as
    ///   1. the CURRENT module's own types (`src.Main.List` — local types
    ///      shadow std names);
    ///   2. user modules the current module imports;
    ///   3. the std/builtin bare key (one std tree occupies bare names);
    ///   4. a globally unique user type with this bare name (dep-module
    ///      references written bare, the legacy flat-table visibility);
    ///   5. bare fallback (unknown/builtin scalars — error recovery paths).
    // ── Sym 键表访问(NAME_RESOLUTION_PLAN S1)──
    // 写入路径驻留(intern),读取路径只查(find)——语义与原 String 键
    // 完全一致:键没登记过就是 miss。

    /// Record sema's constructor-call resolution (S2). Key mirrors
    /// expr_types/method_dispatches: instantiation-mode replays key by the
    /// instance's declaring module.
    // ── S3 显式上下文变体:IR 期调用 ──
    // IR 读取 sema 的 ambient current_module_name 是陈旧值(最后检查的模块)。
    // IR 侧一律传自己的 current_module().name。

    /// get_type_def with an explicit module context (IR-time safe).
    pub fn get_type_def_in(&self, module: &str, name: &str) -> Option<&TypeDefInfo> {
        let key = Self::canonical_type_name(name);
        let resolved = self.resolve_type_key_in(module, key);
        let idx = self.type_def_idx(&resolved)?;
        self.type_defs.get(&idx)
    }

    /// lookup_method_idx with an explicit module context (IR-time safe).
    pub fn lookup_method_idx_in(&self, module: &str, type_name: &str, method_name: &str) -> Option<u16> {
        let key = Self::canonical_type_name(type_name);
        let idx = match self
            .type_def_idx(&self.resolve_type_key_in(module, key))
            .or_else(|| self.type_def_idx(key))
        {
            Some(i) => i,
            None => return None,
        };
        self.type_defs
            .get(&idx)?
            .methods
            .iter()
            .position(|m| m.name.as_ref() == method_name)
            .map(|p| p as u16)
    }

    /// get_trait_def with an explicit module context (IR-time safe).
    pub fn get_trait_def_in(&self, module: &str, name: &str) -> Option<&TraitDefInfo> {
        let resolved = self.resolve_trait_key_in(module, name);
        let idx = self.trait_def_idx(&resolved)?;
        self.trait_defs.get(&idx)
    }

    /// Record sema's method-dispatch resolution (S4). Key mirrors
    /// expr_types/ctor_resolutions (instantiation replays key by the
    /// instance's declaring module).
    pub fn record_dispatch_target(&mut self, module_name: &str, expr_id: u64, type_idx: u16, method_idx: u16) {
        self.dispatch_targets
            .insert(crate::sema::Sema::module_expr_key(module_name, expr_id), (type_idx, method_idx));
    }

    pub fn record_ctor_resolution(&mut self, module_name: &str, expr_id: u64, canonical_type: &str) {
        let sym = self.symbols.intern(canonical_type);
        self.ctor_resolutions
            .insert(crate::sema::Sema::module_expr_key(module_name, expr_id), sym);
    }

    pub fn type_def_idx(&self, name: &str) -> Option<u16> {
        self.symbols.find(name).and_then(|s| self.type_def_index.get(&s).copied())
    }
    pub fn type_def_has(&self, name: &str) -> bool {
        self.type_def_idx(name).is_some()
    }
    pub fn type_def_set(&mut self, name: &str, idx: u16) {
        let s = self.symbols.intern(name);
        self.type_def_index.insert(s, idx);
    }
    pub fn type_def_remove(&mut self, name: &str) {
        if let Some(s) = self.symbols.find(name) {
            self.type_def_index.remove(&s);
        }
    }
    pub fn trait_def_has(&self, name: &str) -> bool {
        self.symbols.find(name).map(|s| self.trait_def_index.contains_key(&s)).unwrap_or(false)
    }
    pub fn trait_def_idx(&self, name: &str) -> Option<u16> {
        self.symbols.find(name).and_then(|s| self.trait_def_index.get(&s).copied())
    }
    pub fn trait_def_set(&mut self, name: &str, idx: u16) {
        let s = self.symbols.intern(name);
        self.trait_def_index.insert(s, idx);
    }
    pub fn func_sig_has(&self, key: &str) -> bool {
        self.symbols.find(key).map(|s| self.func_sig_index.contains_key(&s)).unwrap_or(false)
    }
    pub fn func_sig_idx(&self, key: &str) -> Option<u16> {
        self.symbols.find(key).and_then(|s| self.func_sig_index.get(&s).copied())
    }
    pub fn func_sig_set(&mut self, key: &str, idx: u16) {
        let s = self.symbols.intern(key);
        self.func_sig_index.insert(s, idx);
    }
    pub fn ctor_def_has(&self, name: &str) -> bool {
        self.symbols.find(name).map(|s| self.ctor_def_index.contains_key(&s)).unwrap_or(false)
    }
    pub fn ctor_def_list(&self, name: &str) -> Option<&Vec<u32>> {
        self.symbols.find(name).and_then(|s| self.ctor_def_index.get(&s))
    }
    pub fn ctor_def_push(&mut self, name: &str, packed: u32) {
        let s = self.symbols.intern(name);
        self.ctor_def_index.entry(s).or_default().push(packed);
    }
    /// field_id_map 键是 "type<NUL>field" 复合串;构造与查询统一走此口。
    pub fn field_id_of(&self, type_name: &str, field: &str) -> Option<u16> {
        let key = format!("{}\u{0}{}", type_name, field);
        self.symbols.find(&key).and_then(|s| self.field_id_map.get(&s).copied())
    }
    pub fn field_id_put(&mut self, type_name: &str, field: &str, id: u16) {
        let key = format!("{}\u{0}{}", type_name, field);
        let s = self.symbols.intern(&key);
        self.field_id_map.insert(s, id);
    }

    pub fn resolve_type_key_in(&self, module_name: &str, bare: &str) -> String {
        // A dotted input is either an internal canonical (module-qualified)
        // name or a source-qualified path (`A.Point`): exact key first, then
        // the source spelling mapped through its module qualifier (imports →
        // std/builtin prefix → globally unique user module). Unmapped
        // spellings pass through unchanged (unknown/forward reference —
        // bare-key behavior).
        if bare.contains('.') {
            if self.type_def_has(bare) {
                return bare.to_string();
            }
            if let Some(key) = self.map_qualified_key_in(module_name, bare) {
                return key;
            }
            return bare.to_string();
        }
        // 1. Own module — local `type List` shadows the std type. The
        //    pending set covers forward references during populate (declared
        //    but not yet registered).
        let own = canonical_type_key(module_name, bare);
        if self.type_def_has(&own)
            || (own != bare && self.pending_own_types.contains(bare))
        {
            return own;
        }
        // 2. Imported user modules (declaration order).
        if let Some(imports) = self.module_imports.get(module_name) {
            for path in imports {
                let key = format!("{}.{}", path, bare);
                if self.type_def_has(&key) {
                    return key;
                }
            }
        }
        // 3. std/builtin bare key — one std tree occupies bare names.
        if self.type_def_has(bare) {
            return bare.to_string();
        }
        // 4. A globally unique user type with this bare name (dep-module
        //    references written bare — the legacy flat-table visibility).
        //    Contested names fall through to the bare fallback.
        let mut unique: Option<String> = None;
        for (&ksym, _) in self.type_def_index.iter() {
            let key = self.symbols.resolve(ksym);
            if key.contains('.') && key.rsplit('.').next() == Some(bare) {
                if unique.is_some() {
                    unique = None;
                    break;
                }
                unique = Some(key.to_string());
            }
        }
        if let Some(key) = unique {
            return key.clone();
        }
        // 5. Unknown/builtin-scalar bare fallback (error recovery paths).
        bare.to_string()
    }

    /// `resolve_type_key_in` against the module currently being
    /// populated/checked (`current_module_name`).
    pub fn resolve_type_key(&self, bare: &str) -> String {
        self.resolve_type_key_in(&self.current_module_name, bare)
    }

    /// Resolve a source module qualifier (`A`, `std.collections`) to its
    /// canonical anchor: imported user modules first (declaration order,
    /// mirroring the bare chain's import step), then std/builtin prefixes
    /// (whose types register under bare keys — `is_std = true`), then a
    /// globally unique user module tail match. `None` when the qualifier
    /// names no module (contested tail matches stay unresolved too).
    pub(in crate::sema) fn resolve_module_qualifier(&self, module_name: &str, qual: &str) -> Option<(String, bool)> {
        if let Some(imports) = self.module_imports.get(module_name) {
            for path in imports {
                if path == qual || path.ends_with(&format!(".{}", qual)) {
                    return Some((anchored_logical_path(path).to_string(), false));
                }
            }
        }
        if qual == "std" || qual.starts_with("std.")
            || qual == "builtin" || qual.starts_with("builtin.")
        {
            return Some((String::new(), true));
        }
        let mut unique: Option<&String> = None;
        for lp in &self.user_module_paths {
            if lp == qual || lp.ends_with(&format!(".{}", qual)) {
                if unique.is_some() {
                    return None;
                }
                unique = Some(lp);
            }
        }
        unique.map(|lp| (anchored_logical_path(lp).to_string(), false))
    }

    /// Map a source-qualified key (`A.Point`, `std.collections.List`) to its
    /// canonical registration key: last segment = bare type/trait name,
    /// prefix = module qualifier (`resolve_module_qualifier`).
    fn map_qualified_key_in(&self, module_name: &str, dotted: &str) -> Option<String> {
        let pos = dotted.rfind('.')?;
        let (qual, tail) = (&dotted[..pos], &dotted[pos + 1..]);
        match self.resolve_module_qualifier(module_name, qual)? {
            (_, true) => Some(tail.to_string()),
            (anchor, false) => Some(format!("{}.{}", anchor, tail)),
        }
    }

    /// Module-scoped bare-name → canonical TRAIT key resolution (the trait
    /// twin of `resolve_type_key_in`; own-pending covers forward-referenced
    /// parents and impl bounds).
    pub fn resolve_trait_key_in(&self, module_name: &str, bare: &str) -> String {
        // Dotted: canonical exact hit, else source-qualified spelling mapped
        // through its module qualifier (the `resolve_type_key_in` twin).
        if bare.contains('.') {
            if self.trait_def_has(bare) {
                return bare.to_string();
            }
            if let Some(key) = self.map_qualified_key_in(module_name, bare) {
                return key;
            }
            return bare.to_string();
        }
        let own = canonical_type_key(module_name, bare);
        if self.trait_def_has(&own)
            || (own != bare && self.pending_own_traits.contains(bare))
        {
            return own;
        }
        if let Some(imports) = self.module_imports.get(module_name) {
            for path in imports {
                let key = format!("{}.{}", path, bare);
                if self.trait_def_has(&key) {
                    return key;
                }
            }
        }
        if self.trait_def_has(bare) {
            return bare.to_string();
        }
        let mut unique: Option<String> = None;
        for (&ksym, _) in self.trait_def_index.iter() {
            let key = self.symbols.resolve(ksym);
            if key.contains('.') && key.rsplit('.').next() == Some(bare) {
                if unique.is_some() {
                    unique = None;
                    break;
                }
                unique = Some(key.to_string());
            }
        }
        if let Some(key) = unique {
            return key;
        }
        bare.to_string()
    }

    pub fn resolve_trait_key(&self, bare: &str) -> String {
        self.resolve_trait_key_in(&self.current_module_name, bare)
    }

    /// Transitive parent-trait closure of `trait_name` (excluding itself),
    /// declaration-order BFS with dedup; cycle- and missing-safe.
    pub fn trait_parent_closure(&self, trait_name: &str) -> Vec<Box<str>> {
        let mut out: Vec<Box<str>> = Vec::new();
        let mut queue: std::collections::VecDeque<Box<str>> = std::collections::VecDeque::new();
        if let Some(td) = self.get_trait_def(trait_name) {
            for p in td.parents.iter() {
                queue.push_back(p.clone());
            }
        }
        let mut hops = 0;
        while let Some(t) = queue.pop_front() {
            hops += 1;
            if hops > 128 {
                break;
            }
            if out.iter().any(|x| x.as_ref() == t.as_ref()) {
                continue;
            }
            if let Some(td) = self.get_trait_def(t.as_ref()) {
                out.push(t.clone());
                for p in td.parents.iter() {
                    queue.push_back(p.clone());
                }
            }
        }
        out
    }

    /// The effective method surface of a trait: own methods plus every
    /// transitive parent's (child's own entries shadow same-named parents').
    pub fn trait_effective_methods(&self, trait_name: &str) -> Vec<(Box<str>, &TraitMethodSig)> {
        // (origin trait, sig) pairs; own first so shadowing picks the child's.
        let mut out: Vec<(Box<str>, &TraitMethodSig)> = Vec::new();
        if let Some(td) = self.get_trait_def(trait_name) {
            for m in td.methods.iter() {
                out.push((trait_name.into(), m));
            }
        }
        for p in self.trait_parent_closure(trait_name) {
            if let Some(td) = self.get_trait_def(p.as_ref()) {
                for m in td.methods.iter() {
                    if !out.iter().any(|(_, om)| om.name == m.name) {
                        out.push((p.clone(), m));
                    }
                }
            }
        }
        out
    }

    /// Look up a constructor definition by constructor name.    /// Look up a constructor definition by constructor name.
    /// Returns the first match when multiple types share the same constructor
    /// name; use `get_ctor_defs` for disambiguation.
    pub fn get_ctor_def(&self, name: &str) -> Option<&CtorDefInfo> {
        // Canonical (dotted) type names arrive here for record types whose
        // constructor shares the type's name — the ctor map is keyed by the
        // bare ctor name, so retry with the stripped tail and pick the entry
        // whose OWNING type matches the canonical name.
        if let Some(list) = self.ctor_def_list(name) {
            if let Some(&packed_idx) = list.first() {
                let type_idx = (packed_idx >> 16) as u16;
                let ctor_idx = (packed_idx & 0xFFFF) as u16;
                return self
                    .type_defs
                    .get(&type_idx)
                    .and_then(|def| def.constructors.get(ctor_idx as usize));
            }
        }
        if !name.contains('.') {
            return None;
        }
        let bare = name.rsplit('.').next()?;
        // Source-qualified spelling (`A.TEf`): candidates whose OWNING type
        // lives in the qualifier's module. Canonical exact match (internal
        // dotted names) is kept first.
        let qual = &name[..name.len() - bare.len() - 1];
        let mod_anchor = self.resolve_module_qualifier(&self.current_module_name, qual);
        for &packed_idx in self.ctor_def_list(bare)? {
            let type_idx = (packed_idx >> 16) as u16;
            let ctor_idx = (packed_idx & 0xFFFF) as u16;
            if let Some(def) = self.type_defs.get(&type_idx) {
                let dn = def.name.as_ref();
                let in_module = match &mod_anchor {
                    // std/builtin: owning type names are bare.
                    Some((_, true)) => !dn.contains('.'),
                    Some((anchor, false)) => dn
                        .strip_prefix(anchor.as_str())
                        .is_some_and(|r| r.starts_with('.')),
                    None => false,
                };
                if dn == name || in_module {
                    return def.constructors.get(ctor_idx as usize);
                }
            }
        }
        None
    }

    /// Look up all constructor definitions matching a constructor name.
    /// Returns an empty slice when no match is found; returns multiple entries
    /// when different types share the same constructor name (e.g. `FileKind.File`
    /// and `type File`).
    pub fn get_ctor_defs(&self, name: &str) -> Vec<&CtorDefInfo> {
        match self.ctor_def_list(name) {
            Some(indices) => indices
                .iter()
                .filter_map(|&packed_idx| {
                    let type_idx = (packed_idx >> 16) as u16;
                    let ctor_idx = (packed_idx & 0xFFFF) as u16;
                    self.type_defs
                        .get(&type_idx)
                        .and_then(|def| def.constructors.get(ctor_idx as usize))
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Resolve a record/ADT field type descriptor.
    /// Distinguishes Record (field_id starts at 0) from ADT (field_id starts at
    /// 1) based on `TypeDefKind`.
    /// Returns `(field_id, field_type_desc)`, or `None` if not found.
    pub fn resolve_field_td(
        &self,
        type_name: &str,
        field: &str,
    ) -> Option<(u16, TypeHandle)> {
        let field_id = self.lookup_field_id(type_name, field)?;
        let ctor = self.get_ctor_def(type_name)?;
        let idx = match self.get_type_def(type_name) {
            Some(def) if def.kind == TypeDefKind::Record => field_id as usize,
            _ => (field_id as usize).saturating_sub(1),
        };
        let &field_td = ctor.field_types.get(idx)?;
        Some((field_id, field_td))
    }

    // ── Trait definitions ──
    // Trait registry: hand-written (not the shared macro) — the GET is
    // module-scoped, mirroring the type namespace (user-module traits are
    // canonical `src.Main.Eq`; std/builtin keep bare names).
    /// Insert a trait def; returns `false` on a duplicate canonical name
    /// (same-module re-population lands here and is a silent no-op, exactly
    /// like the macro version).
    pub fn put_trait_def(&mut self, def: TraitDefInfo) -> bool {
        if self.trait_def_has(def.name.as_ref()) {
            return false;
        }
        assert!(
            self.next_trait_def_id < u16::MAX,
            "trait_defs index overflow: too many entries",
        );
        let idx = self.next_trait_def_id;
        self.next_trait_def_id += 1;
        self.trait_def_set(def.name.as_ref(), idx);
        self.trait_defs.insert(idx, def);
        true
    }

    /// Module-scoped trait lookup: a CANONICAL (dotted) name passes through;
    /// a bare name resolves own-module-first, then imports, then the
    /// std/builtin bare key, then a unique global user trait.
    pub fn get_trait_def(&self, name: &str) -> Option<&TraitDefInfo> {
        let resolved = self.resolve_trait_key(name);
        let idx = self.trait_def_idx(&resolved)?;
        self.trait_defs.get(&idx)
    }

    // ── Function signatures ──
    // Module-qualified registry ("module\x00name") — hand-written, not the
    // shared define_table_registry! macro. The legacy bare-name keying made
    // same-named top-level functions in different modules collide:
    // put_func_sig silently dropped every sig after the first (first-wins),
    // and bare lookups then answered with the wrong module's flags/arity.
    // Registration keys by (module, name); the bare-name `get_func_sig`
    // resolves through the owner table (unique owner wins; contested names
    // answer with the first-registered module — callers with module context
    // use `get_func_sig_in`).

    fn func_sig_qualified_key(module: &str, name: &str) -> String {
        let mut k = String::with_capacity(module.len() + name.len() + 1);
        k.push_str(module);
        k.push('\x00');
        k.push_str(name);
        k
    }

    /// Insert a signature under its (module, name) key; returns `false` on a
    /// same-module redefinition. The u16 index is allocated from
    /// `next_func_sig_id` and never recycles.
    pub fn put_func_sig(&mut self, sig: FuncSigInfo) -> bool {
        let qualified = Self::func_sig_qualified_key(&sig.module_name, &sig.name);
        if self.func_sig_has(&qualified) {
            return false;
        }
        assert!(
            self.next_func_sig_id < u16::MAX,
            "func_sigs index overflow: too many entries"
        );
        let idx = self.next_func_sig_id;
        self.next_func_sig_id += 1;
        self.func_sig_owners
            .entry(sig.name.to_string())
            .or_default()
            .push(sig.module_name.to_string());
        self.func_sig_set(&qualified, idx);
        self.func_sigs.insert(idx, sig);
        true
    }

    /// Bare-name lookup, STRICT: `Some` only when exactly one module defines
    /// `name` — a contested name has no single correct answer, so it resolves
    /// to `None` (no legacy first-registered fallback). Callers with module
    /// context use `get_func_sig_in`; callers checking a property
    /// conservatively use `func_sigs_named`.
    pub fn get_func_sig(&self, name: &str) -> Option<&FuncSigInfo> {
        let owners = self.func_sig_owners.get(name)?;
        if owners.len() != 1 {
            return None;
        }
        let module = owners.first()?;
        let idx = self.func_sig_idx(&Self::func_sig_qualified_key(module, name))?;
        self.func_sigs.get(&idx)
    }

    /// Every signature registered under a bare name (cross-module same-name
    /// aware) — for conservative checks ("is ANY owner async/throwing") where
    /// picking a single owner would be unsound.
    pub fn func_sigs_named(&self, name: &str) -> Vec<&FuncSigInfo> {
        match self.func_sig_owners.get(name) {
            None => Vec::new(),
            Some(owners) => owners
                .iter()
                .filter_map(|m| {
                    let idx = self.func_sig_idx(&Self::func_sig_qualified_key(m, name))?;
                    self.func_sigs.get(&idx)
                })
                .collect(),
        }
    }

    /// Module-qualified lookup — the exact signature of `module`'s `name`.
    pub fn get_func_sig_in(&self, module: &str, name: &str) -> Option<&FuncSigInfo> {
        let idx = self.func_sig_idx(&Self::func_sig_qualified_key(module, name))?;
        self.func_sigs.get(&idx)
    }

    /// Record that a func_sig belongs to a module (for incremental purge).
    /// Looks up the current index by (module, name); call after a successful `put_func_sig`.
    pub fn record_func_sig_owner(&mut self, name: &str, module_name: &str) {
        if let Some(idx) = self
            .func_sig_idx(&Self::func_sig_qualified_key(module_name, name))
        {
            self.module_ownership.func_sig_indices
                .entry(module_name.to_string())
                .or_default()
                .insert(idx);
        }
    }

    /// Record that a trait_def belongs to a module (for incremental purge).
    /// Looks up the current index by name; call after a successful `put_trait_def`.
    pub fn record_trait_def_owner(&mut self, name: &str, module_name: &str) {
        if let Some(idx) = self.trait_def_idx(name) {
            self.module_ownership.trait_def_indices
                .entry(module_name.to_string())
                .or_default()
                .insert(idx);
        }
    }

    // ── Method signatures (Type-driven) ──

    /// Look up `method_idx` (the position in `TypeDefInfo.methods`) by type name
    /// and method name.
    ///
    /// The IR layer uses (type_id, method_idx) to look up the subgraph in
    /// `method_subgraphs`. Returning `None` means the type has no such method
    /// (it may be a trait default method; consult the witness_table).
    /// Method-index lookup by dynamic type_id — unambiguous (no name
    /// resolution, no module context). The fallback for name-based lookups
    /// when the global `current_module_name` is stale (IR-time dispatch for
    /// a module other than the last-checked one).
    pub fn lookup_method_idx_by_type_id(&self, type_id: u16, method_name: &str) -> Option<u16> {
        if type_id < FIRST_DYNAMIC_TYPE_ID {
            return None;
        }
        let idx = type_def_index_of(type_id);
        self.type_defs
            .get(&idx)?
            .methods
            .iter()
            .position(|m| m.name.as_ref() == method_name)
            .map(|p| p as u16)
    }

    pub fn lookup_method_idx(&self, type_name: &str, method_name: &str) -> Option<u16> {
        // Module-scoped: bare AST names resolve to their canonical key.
        // Resolve FIRST (own module → imports → std bare → unique user): a
        // bare key held by a STD type (e.g. std.json's `Parser`) used to win
        // the old exact-hit fast path and hijack every same-named USER type
        // — method dispatch then read the std twin's (empty) method table
        // and every implicit-this sibling call failed to compile.
        let key = Self::canonical_type_name(type_name);
        let key = self
            .type_def_idx(&self.resolve_type_key(key))
            .or_else(|| self.type_def_idx(key))?;

        let type_def = &self.type_defs[&key];
        type_def
            .methods
            .iter()
            .position(|m| m.name.as_ref() == method_name)
            .map(|i| i as u16)
    }

    /// Canonical registry name for type-name-keyed lookups: concrete array
    /// names ("u8[]", nested "u8[][]" — from ExprInfo.type_name /
    /// expr_type_name) address the synthetic builtin "array" TypeDefInfo.
    pub fn canonical_type_name(name: &str) -> &str {
        if name.ends_with("[]") {
            "array"
        } else {
            name
        }
    }

    /// Get the method signature by `type_id` and `method_idx`.
    pub fn get_method_sig(&self, type_id: u16, method_idx: u16) -> Option<&MethodSigInfo> {
        if type_id < FIRST_DYNAMIC_TYPE_ID {
            return None;
        }
        let type_idx = type_def_index_of(type_id);
        let type_def = self.type_defs.get(&type_idx)?;
        type_def.methods.get(method_idx as usize)
    }

    /// Resolve the trait-default binding of `method_name` on an implementing
    /// type (see `MethodBinding`).
    ///
    /// `implemented_traits` are the trait names bound by the type declaration
    /// (`type T: (A, B)`); pass the witness-table entries' trait names when the
    /// declaration AST is not at hand.
    ///
    /// Resolution order:
    /// 1. An explicit delegate annotation on the declared method
    ///    (`fun m(): R = A.m`, with or without a body) wins outright.
    /// 2. Otherwise, among implemented traits that provide a default (a method
    ///    with a body) named `method_name`: a unique provider binds implicitly;
    ///    multiple providers make the binding ambiguous.
    pub fn resolve_method_binding(
        &self,
        implemented_traits: &[Box<str>],
        type_name: &str,
        method_name: &str,
    ) -> MethodBinding {
        let declared = self
            .type_def_idx(type_name)
            .and_then(|idx| self.type_defs.get(&idx))
            .and_then(|td| td.methods.iter().find(|m| m.name.as_ref() == method_name));
        let mut providers: Vec<Box<str>> = Vec::new();
        for t in implemented_traits {
            let has_default = self
                .get_trait_def(t)
                .map(|td| {
                    td.methods
                        .iter()
                        .any(|m| m.name.as_ref() == method_name && m.has_body)
                })
                .unwrap_or(false);
            if has_default {
                providers.push(t.clone());
            }
        }
        if let Some(m) = declared {
            if let Some(ref t) = m.delegate_trait {
                return MethodBinding::Bound { trait_name: t.clone(), overridden: m.has_body };
            }
        }
        let overridden = declared.map(|m| m.has_body).unwrap_or(false);
        // Trait-inheritance shadowing: when one provider is a descendant of
        // every other (its parent closure contains them all), the child
        // trait's declaration shadows the parents' — `trait C(A, B)` can
        // resolve A/B's conflict by declaring its own default `m`.
        if providers.len() > 1 {
            let mut winner: Option<usize> = None;
            for (i, c) in providers.iter().enumerate() {
                if providers.iter().all(|o| {
                    o.as_ref() == c.as_ref()
                        || self.trait_parent_closure(c.as_ref())
                            .iter()
                            .any(|a| a.as_ref() == o.as_ref())
                }) {
                    winner = Some(i);
                    break;
                }
            }
            if let Some(i) = winner {
                let t = providers.remove(i);
                return MethodBinding::Bound { trait_name: t, overridden };
            }
        }
        match providers.len() {
            0 => MethodBinding::Unbound,
            1 => MethodBinding::Bound { trait_name: providers.remove(0), overridden },
            _ => MethodBinding::Ambiguous(providers),
        }
    }

    // ── Coroutine metadata ──

    /// Add coroutine metadata.
    pub fn put_coroutine_meta(&mut self, meta: CoroutineMeta) {
        self.coroutine_metas.push(meta);
    }

    /// Look up coroutine metadata by `func_idx`.
    pub fn get_coroutine_meta_by_func_idx(&self, func_idx: u16) -> Option<&CoroutineMeta> {
        self.coroutine_metas.iter().find(|m| m.func_idx == func_idx)
    }

    // ── Incremental purge ──

    /// Purge all sema products for a module (prepare for incremental recheck).
    /// Removes expr_types, resolved_types, type_defs entries, func_sigs entries,
    /// trait_defs entries, witness_table entries, import_aliases, field_id_map entries.
    /// type_defs/func_sigs/trait_defs are truly removed from their HashMaps (the
    /// u16 index allocator never recycles, so freed indices are simply never reused).
    pub fn purge_module(&mut self, module_name: &str) {
        // === Category A: u64-key tables ===
        // Use expr_type_keys reverse map to find which keys to remove
        if let Some(keys) = self.module_ownership.expr_type_keys.remove(module_name) {
            for k in &keys {
                self.expr_types.remove(k);
                self.resolved_types.remove(k);
                self.field_accesses.remove(k);
                self.method_dispatches.remove(k);
                self.reflect_metas.remove(k);
                self.call_instantiations.remove(k);
                self.module_func_recv_exprs.remove(k);
                self.module_func_call_targets.remove(k);
                self.bare_call_targets.remove(k);
                self.module_const_recv_exprs.remove(k);
                self.super_dispatches.remove(k);
            }
        }
        // pattern_ctor_types: key is (String, u32), filter by module name
        self.pattern_ctor_types.retain(|(m, _), _| m.as_str() != module_name);

        // captures: u64-key table (nested-scope entry expr keys), tracked via
        // the dedicated capture_keys reverse map.
        if let Some(keys) = self.module_ownership.capture_keys.remove(module_name) {
            for k in keys {
                self.captures.remove(&k);
            }
        }

        // === Category B: global definition tables ===
        // Truly remove entries from both the index HashMap and the value HashMap.
        // Freed u16 indices are never reused (the allocator is monotonic).

        // type_defs
        if let Some(indices) = self.module_ownership.type_def_indices.remove(module_name) {
            for idx in indices {
                if let Some(def) = self.type_defs.remove(&idx) {
                    self.type_def_remove(def.name.as_ref());
                    // Remove the type_id → name reverse-index entry.
                    let type_id = dynamic_type_id(idx);
                    self.type_id_to_name.remove(&type_id);
                    // Remove constructor entries from ctor_def_index
                    for ctor in &def.constructors {
                        if let Some(sym) = self.symbols.find(ctor.name.as_ref()) {
                            if let Some(vec) = self.ctor_def_index.get_mut(&sym) {
                                vec.retain(|&packed| (packed >> 16) as u16 != idx);
                                if vec.is_empty() {
                                    self.ctor_def_index.remove(&sym);
                                }
                            }
                        }
                    }
                }
            }
        }

        // func_sigs
        if let Some(indices) = self.module_ownership.func_sig_indices.remove(module_name) {
            for idx in indices {
                if let Some(sig) = self.func_sigs.remove(&idx) {
                    // Qualified key + owner-list cleanup (see put_func_sig).
                    if let Some(sym) = self
                        .symbols
                        .find(&Self::func_sig_qualified_key(&sig.module_name, &sig.name))
                    {
                        self.func_sig_index.remove(&sym);
                    }
                    if let Some(owners) = self.func_sig_owners.get_mut(sig.name.as_ref()) {
                        owners.retain(|m| m != sig.module_name.as_ref());
                        if owners.is_empty() {
                            self.func_sig_owners.remove(sig.name.as_ref());
                        }
                    }
                }
            }
        }

        // trait_defs
        if let Some(indices) = self.module_ownership.trait_def_indices.remove(module_name) {
            for idx in indices {
                if let Some(def) = self.trait_defs.remove(&idx) {
                    if let Some(sym) = self.symbols.find(def.name.as_ref()) {
                        self.trait_def_index.remove(&sym);
                    }
                }
            }
        }

        // field_id_map
        if let Some(keys) = self.module_ownership.field_id_keys.remove(module_name) {
            for k in keys {
                if let Some(sym) = self.symbols.find(&k) {
                    self.field_id_map.remove(&sym);
                }
            }
        }

        // === Category C: cross-module accumulated ===
        // witness_table
        if let Some(keys) = self.module_ownership.witness_keys.remove(module_name) {
            for (trait_name, type_id) in keys {
                self.witness_table.remove(&trait_name, type_id);
            }
        }
        // import_aliases
        if let Some(keys) = self.module_ownership.alias_keys.remove(module_name) {
            for k in keys {
                self.import_aliases.remove(&k);
            }
        }
        // monomorph_instances: leave stale entries (Vec, indexed by hash, won't be looked up)
        // monomorph_index: also leave (stale entries harmless)
    }
}

// =========================================================================
// builtin_types — built-in type registry.
//
// A Rust port of `src/sema/builtin_types.zig`. Unifies the scalar name → Type
// mapping and the arity table for built-in generic types.
// Data source: BUILTIN_TABLE in Type.rs (type_id 1..=21), the single source of
// truth.
// =========================================================================

/// Entry for a built-in generic type (a higher-kinded type with a fixed arity).
#[derive(Debug, Clone, Copy)]
pub struct BuiltinGenericEntry {
    pub name: &'static str,
    pub arity: u8,
}

/// Built-in type declaration macro: a single declaration produces two outputs.
///
/// - **Output 1**: `BUILTIN_GENERIC_TYPES` static arity table (`generic` group
///   only). kind_check (step 7) runs before `register_builtin_method_sigs`
///   (step 8), so the arity table must be a static constant.
/// - **Output 2**: the `register_builtin_method_sigs` function body (`generic`
///   + `nongeneric` groups). At runtime it registers synthetic `TypeDefInfo`
///   (including the method signature table), so built-in type method lookup goes
///   through the same `(type_id, method_idx)` path used for user-defined types.
///
/// Types in the `generic` group use the `TypeNode::Generic { name, .. }` AST
/// node and must have an entry in `BUILTIN_GENERIC_TYPES` so kind_check can look
/// up their arity. The `nongeneric` group has dedicated `Type`/`TypeNode` variants
/// (e.g. Array/Nullable/Str) and does not need an arity-table entry.
///
/// Declaration syntax:
/// ```ignore
/// define_builtin_types! {
///     generic {
///         "TypeName" : ["T", "E"] = [ sig(...), sig(...), ... ],
///         ...
///     }
///     nongeneric {
///         "TypeName" : ["T"] = [ sig(...), ... ],
///         ...
///     }
/// }
/// ```
macro_rules! define_builtin_types {
    (
        generic { $($gname:literal : [$($gp:literal),*] = [$($gmethod:expr),* $(,)?]),* $(,)? }
        nongeneric { $($nname:literal : [$($np:literal),*] = [$($nmethod:expr),* $(,)?]),* $(,)? }
    ) => {
        /// Built-in generic type constructor table (derived from the `generic`
        /// group by the `define_builtin_types!` macro).
        pub const BUILTIN_GENERIC_TYPES: &[BuiltinGenericEntry] = &[
            $( BuiltinGenericEntry {
                name: $gname,
                arity: <[&'static str]>::len(&[$($gp),*]) as u8,
            } ),*
        ];

        /// Register synthetic `TypeDefInfo` (including the method signature table)
        /// for built-in types, so built-in type method lookup goes through the
        /// same (type_id, method_idx) path used for user-defined types,
        /// eliminating the match-branch special casing in `lookup_builtin_method`.
        ///
        /// Within method signatures:
        /// - param_type_reprs[0] = ThisType (the `self` parameter, matching user
        ///   type blocks).
        /// - Generic parameters use Named("T")/Named("E"), resolved via
        ///   `type_binding_stack`.
        /// - Scalar return types use Named("usize")/Named("bool")/Named("void")/Named("str").
        /// - `build_fn_type_from_sig` reconstructs the full `Type::Fn` via
        ///   `type_repr_to_handle`.
        pub fn register_builtin_method_sigs(sema_result: &mut SemaResult) {
            /// Build a single built-in method signature. The `type` field is a
            /// `Type::Void` placeholder (does not affect type checking;
            /// `build_fn_type_from_sig` only reads `param_type_reprs` /
            /// `return_type_repr`).
            /// The `intrinsic` parameter tags the lowering strategy; `None` means
            /// an ordinary method (with a body or a trait method).
            fn sig(
                name: &str,
                param_reprs: Vec<TypeRepr>,
                return_repr: Option<TypeRepr>,
                intrinsic: Option<IntrinsicKind>,
            ) -> MethodSigInfo {
                let n = param_reprs.len();
                MethodSigInfo {
                    name: name.into(),
                    param_is_ref: vec![false; n].into_boxed_slice(),
                    return_is_ref: false,
                    is_async: false,
                    is_throwing: false,
                    param_type_reprs: param_reprs.into_boxed_slice(),
                    return_type_repr: return_repr,
                    intrinsic,
                    has_body: true,
                    delegate_trait: None,
                    is_pub: true,
                }
            }

            /// Register a synthetic `TypeDefInfo` for a single built-in type.
            fn register(
                sema_result: &mut SemaResult,
                type_name: &str,
                type_params: &[&str],
                methods: Vec<MethodSigInfo>,
            ) {
                if sema_result.type_def_has(type_name) {
                    return; // Already registered (e.g. user stdlib declared a same-named type block).
                }
                let def = TypeDefInfo {
                    name: type_name.into(),
                    kind: TypeDefKind::Alias,
                    constructors: Box::new([]),
                    type_params: type_params.iter().map(|t| (*t).into()).collect(),
                    target_type_name: None,
                    target_type: None,
                    methods: methods.into_boxed_slice(),
                    bases: Box::new([]),
                };
                sema_result.put_type_def(def, "");
            }

            // ── generic group: enters BUILTIN_GENERIC_TYPES + method registration ──
            $(
                register(sema_result, $gname, &[$($gp),*], vec![$($gmethod),*]);
            )*
            // ── nongeneric group: method registration only (has dedicated Type variant) ──
            $(
                register(sema_result, $nname, &[$($np),*], vec![$($nmethod),*]);
            )*
        }
    };
}

define_builtin_types! {
    generic {
        "Throw" : ["T", "E"] = [
            sig("is_ok", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), None),
        ],
        "Channel" : ["T"] = [
            sig("send", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(282))),
            sig("recv", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::ChannelAwait)),
            sig("close", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(283))),
        ],
        "Atomic" : ["T"] = [
            sig("swap", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::BinOp(314))),
            sig("compare_exchange", vec![TypeRepr::ThisType, TypeRepr::Named("T".into()), TypeRepr::Named("T".into())], Some(TypeRepr::Named("bool".into())), Some(IntrinsicKind::TriOp(315))),
            sig("load", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::UnOp(312))),
            sig("store", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(313))),
        ],
        "Async" : ["T"] = [
            sig("status", vec![TypeRepr::ThisType], Some(TypeRepr::Named("str".into())), None),
            sig("await", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::Await)),
            sig("cancel", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(42))),
        ],
        "Sender" : ["T"] = [
            sig("send", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(282))),
            sig("close", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(283))),
        ],
        "Receiver" : ["T"] = [
            sig("recv", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::ChannelAwait)),
            sig("close", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(283))),
        ],
        "Lazy" : ["T"] = [],
        // Lib/ForeignFn methods dispatch structurally (name-based, like reflect):
        // `InferContext::lib_method_return_type` + `Builder::lib_method_intrinsic`.
        "ForeignFn" : ["R"] = [],
    }
    nongeneric {
        "Lib" : [] = [],
        "array" : ["T"] = [
            sig("len", vec![TypeRepr::ThisType], Some(TypeRepr::Named("usize".into())), Some(IntrinsicKind::UnOp(35))),
            sig("is_empty", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), Some(IntrinsicKind::UnOp(340))),
        ],
        "str" : [] = [
            sig("len", vec![TypeRepr::ThisType], Some(TypeRepr::Named("usize".into())), Some(IntrinsicKind::UnOp(35))),
            sig("is_empty", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), Some(IntrinsicKind::UnOp(340))),
            sig("bytes", vec![TypeRepr::ThisType], Some(TypeRepr::Array(Box::new(TypeRepr::Named("u8".into())), None)), Some(IntrinsicKind::UnOp(285))),
        ],
        "nullable" : ["T"] = [
            sig("is_null", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), Some(IntrinsicKind::UnOp(34))),
        ],
    }
}

/// Built-in generic type name → arity (returns `None` if unmatched).
pub fn generic_type_arity(name: &str) -> Option<u8> {
    BUILTIN_GENERIC_TYPES
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.arity)
}

/// Whether `name` is a built-in generic type constructor.
#[inline]
pub fn is_builtin_generic_type(name: &str) -> bool {
    generic_type_arity(name).is_some()
}

// =========================================================================
// type_resolver — type resolver.
//
// Responsibility: resolve an AST type node + a type_args binding context into a
// `TypeHandle`.
// All functions are free functions (not methods), because resolution requires
// multiple inputs (`&AstArena` + `&mut TypeArena` + `&mut SemaResult`) with no
// single `self` holding state.
// =========================================================================

/// Extract the type name from a `TypeNode` (used for type inference of variable
/// bindings).
///
/// Recurses into the inner type for `&T` / `*T`, returns the base name for
/// generics, and returns the name itself for named types.
/// Returns `None` for other type nodes.
pub fn type_name_from_node<'a>(
    type_ref: Option<AstTypeRef>,
    ast: &AstArena<'a>,
) -> Option<&'a str> {
    let type_ref = type_ref?;
    let tn = &ast.ty(type_ref).node;
    let effective = match tn {
        TypeNode::RefType { inner } | TypeNode::RawPtr { inner } => &ast.ty(*inner).node,
        _ => tn,
    };
    match effective {
        TypeNode::Named { name } => Some(name),
        TypeNode::Generic { name, .. } => Some(name),
        _ => None,
    }
}

/// Check if a TypeHandle corresponds to a type with the given name.
fn type_handle_name_matches(arena: &TypeArena, h: TypeHandle, name: &str) -> bool {
    match arena.get(h) {
        Type::Adt(_) => arena.adt_parts(h).0 == name,
        Type::Generic(_) => arena.generic_parts(h).0 == name,
        Type::Trait(_) => arena.trait_parts(h).0 == name,
        // Other types (including built-in generics Throw/Channel/Async/Lazy/Atomic/
        // Sender/Receiver/Timer and scalars/str/void) uniformly go through
        // `ty.source_name()`: the name being matched is a user-written identifier,
        // so the source spelling is the single source of truth.
        ty => ty.source_name() == name,
    }
}

/// Maximum recursion depth for type resolution: prevents stack overflow from
/// extremely deep alias/newtype chains.
/// `visiting.len()` is the current recursion depth; recursion stops when the
/// limit is reached.
const MAX_TYPE_RECURSION_DEPTH: usize = 256;

/// Resolve a type by name (resolved version, with alias/newtype chain expansion).
///
/// Priority: type_args binding → built-in scalar/str/void → type_defs
/// alias/newtype recursion → user-defined type (`arena.make_adt`).
/// Extracted as a free function so that `resolve_type_node_resolved` can recurse
/// on aliases without constructing a temporary `TypeNode`.
///
/// `visiting` is used for cyclic-alias detection: if `name` is already in the
/// set, a cyclic alias chain has been encountered, and `arena.make_adt(name)` is
/// returned instead of recursing further (preventing infinite recursion / stack
/// overflow).
fn resolve_named_type_resolved(
    arena: &mut TypeArena,
    name: &str,
    type_args: &[TypeHandle],
    sema_result: &mut SemaResult,
    visiting: &mut FxHashSet<String>,
) -> TypeHandle {
    // 1. Prefer type_args bindings (generic type parameters — bare names).
    for &ta in type_args {
        if type_handle_name_matches(arena, ta, name) {
            return ta;
        }
    }
    // 2. Built-in scalar/str/null/void (bare names).
    if let Some(ty) = Type::from_type_name(name) {
        return arena.make(ty);
    }
    // Module-scoped canonicalization: a bare AST name resolves against the
    // module being populated/checked (own → imports → std → unique user);
    // everything downstream (type_defs lookup, Adt identity, alias-chain
    // traversal) uses the canonical key.
    let canonical = sema_result.resolve_type_key(name);
    // Cyclic-alias detection: `name` already in `visiting` means a cycle;
    // stop recursing.
    if visiting.contains(canonical.as_str()) {
        return arena.make_adt(canonical.into(), Box::new([]));
    }
    // Recursion-depth limit: `visiting.len()` is the current depth; stop
    // recursing past the limit to prevent stack overflow.
    if visiting.len() >= MAX_TYPE_RECURSION_DEPTH {
        return arena.make_adt(canonical.into(), Box::new([]));
    }
    visiting.insert(canonical.clone());
    // 3. Consult type_defs to resolve the alias/newtype chain.
    //    Extract the needed info (owned) to release the immutable borrow,
    //    allowing subsequent `&mut` calls.
    let (target_ty, target_name): (Option<TypeHandle>, Option<String>) =
        match sema_result.get_type_def(&canonical) {
            Some(td) => (
                td.target_type,
                td.target_type_name.as_deref().map(|n| sema_result.resolve_type_key(n)),
            ),
            None => (None, None),
        };
    if let Some(inner_ty) = target_ty {
        // alias/newtype has a target TypeHandle: return it directly.
        visiting.remove(canonical.as_str());
        return inner_ty;
    }
    if let Some(ttn) = target_name {
        // target_type_name is known: recursively resolve to the final concrete
        // type.
        let result = resolve_named_type_resolved(arena, &ttn, type_args, sema_result, visiting);
        visiting.remove(canonical.as_str());
        return result;
    }
    // 4. Other user-defined types → create a named Adt.
    visiting.remove(canonical.as_str());
    arena.make_adt(canonical.into(), Box::new([]))
}

/// Resolve a `TypeNode` into a `TypeHandle` (resolved version, with alias/newtype
/// chain expansion).
///
/// Differs from `concretize_type`: the Named branch consults
/// `sema_result.type_defs` and, if the type is an alias/newtype with a known
/// `target_type`, recursively resolves to the concrete scalar type.
/// Used when the alias chain must be transparently traversed to obtain the final
/// scalar channel type (e.g. scalar monomorphization of `field_value`).
pub fn resolve_type_node_resolved<'a>(
    arena: &mut TypeArena,
    type_ref: Option<AstTypeRef>,
    type_args: &[TypeHandle],
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> Option<TypeHandle> {
    let type_ref = type_ref?;
    let tn = &ast.ty(type_ref).node;
    let mut visiting: FxHashSet<String> = FxHashSet::default();
    Some(match tn {
        TypeNode::Named { name } => resolve_named_type_resolved(arena, name, type_args, sema_result, &mut visiting),
        TypeNode::Generic { name, args } => {
            // Lazy<T>: recursively resolve the inner type.
            if Type::from_type_name(name).is_some_and(|t| t.family() == TypeFamily::Lazy)
                && !args.is_empty() {
                if let Some(inner_ty) =
                    resolve_type_node_resolved(arena, Some(args[0]), type_args, ast, sema_result)
                {
                    return Some(inner_ty);
                }
            }
            arena.make_generic((*name).into(), Box::new([]))
        }
        TypeNode::Nullable { inner } => {
            return resolve_type_node_resolved(arena, Some(*inner), type_args, ast, sema_result);
        }
        TypeNode::RefType { inner } => {
            let inner_name = type_name_from_node(Some(*inner), ast).unwrap_or("ref");
            let canonical = sema_result.resolve_type_key(inner_name);
            arena.make_adt(canonical.into(), Box::new([]))
        }
        TypeNode::RawPtr { inner } => {
            let inner_name = type_name_from_node(Some(*inner), ast).unwrap_or("ptr");
            let canonical = sema_result.resolve_type_key(inner_name);
            arena.make_adt(canonical.into(), Box::new([]))
        }
        TypeNode::Record { .. } => arena.make_record(Vec::<FieldType>::new().into_boxed_slice(), None),
        TypeNode::Function { .. } => {
            let ret = arena.make(Type::Unknown);
            arena.make_fn(Vec::<TypeHandle>::new().into_boxed_slice(), ret)
        }
        TypeNode::Array { .. } => arena.make_adt("array".into(), Box::new([])),
        TypeNode::ThisType => {
            for &ta in type_args {
                if type_handle_name_matches(arena, ta, "This") {
                    return Some(ta);
                }
            }
            arena.make_adt("This".into(), Box::new([]))
        }
        TypeNode::KindAnnotated { inner, .. } => {
            return resolve_type_node_resolved(arena, Some(*inner), type_args, ast, sema_result);
        }
    })
}

// =========================================================================
// inference — core type inference.
//
// A Rust port of `src/sema/inference.zig`.
// Responsibility: generic argument inference, self-parameter binding, literal
// promotion, and GADT inference.
//
// Deliberate differences from the Zig original (intentional improvements,
// confirmed by the user):
// - **self parameter requires scope binding**: top-level extension functions
//   can no longer use `self: TypeName`. `self` is only allowed inside type/trait
//   blocks and cannot have a type annotation. The Zig original's 3-tier
//   fallback (scope→annotation→fresh var) is simplified to scope→error.
// - **Literal promotion**: Zig semantics are preserved — literals are promoted
//   to the variable's type when combined with a variable.
// - **Generic deferred solving**: Zig semantics are preserved — unsolved
//   TypeVars are left for later unification.
//
// Binding-stack architecture:
// - TypeBindingStack: generic parameter name → TypeHandle (rigid var).
// - ThisBindingStack: Self → TypeHandle (scope type).
// The two stacks push/pop in lockstep: entering an `impl Type<T>` block pushes T
// onto TypeBindingStack and Type<T> onto ThisBindingStack; both are popped on
// exit.
// =========================================================================

/// A type-binding stack frame: generic parameter name → TypeHandle (usually a
/// rigid TypeVar).
#[derive(Debug, Default)]
pub struct TypeBindingFrame {
    bindings: FxHashMap<Box<str>, TypeHandle>,
}

impl TypeBindingFrame {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: &str, ty: TypeHandle) {
        self.bindings.insert(name.into(), ty);
    }

    pub fn get(&self, name: &str) -> Option<TypeHandle> {
        self.bindings.get(name).copied()
    }
}

/// Type-binding stack: manages type-parameter bindings during generic
/// instantiation.
///
/// Push a frame when entering `impl Type<T>` or `fn method<U>`; pop on exit.
/// `lookup` searches from the top of the stack down, so inner bindings take
/// precedence (shadowing semantics).
///
/// Note: this stack holds `TypeHandle`s (Type indices); type resolution goes
/// through `InferContext::lookup_type_binding`, without a separate trait
/// abstraction to avoid type confusion.
#[derive(Debug, Default)]
pub struct TypeBindingStack {
    frames: Vec<TypeBindingFrame>,
}

impl TypeBindingStack {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an empty frame; add bindings later via `insert`.
    pub fn push(&mut self) {
        self.frames.push(TypeBindingFrame::new());
    }

    /// Push a pre-constructed frame.
    pub fn push_frame(&mut self, frame: TypeBindingFrame) {
        self.frames.push(frame);
    }

    /// Pop the top frame.
    pub fn pop(&mut self) -> Option<TypeBindingFrame> {
        self.frames.pop()
    }

    /// Current stack depth.
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Look up a type-parameter binding from the top down (inner bindings first).
    pub fn lookup(&self, name: &str) -> Option<TypeHandle> {
        for frame in self.frames.iter().rev() {
            if let Some(ty) = frame.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Add a binding to the top frame (only when the stack is non-empty).
    pub fn insert_top(&mut self, name: &str, ty: TypeHandle) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name, ty);
        }
    }
}

/// Self-binding stack: manages the `Self` type binding for type/trait blocks.
///
/// Push the `TypeHandle` of `T` when entering a `type T { ... }` block;
/// push a `fresh_type_var` when entering `trait Foo<T> { default methods }`;
/// pop on exit. `lookup` returns the top of the stack (inner bindings first).
#[derive(Debug, Default)]
pub struct ThisBindingStack {
    stack: Vec<TypeHandle>,
}

impl ThisBindingStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, self_ty: TypeHandle) {
        self.stack.push(self_ty);
    }

    pub fn pop(&mut self) -> Option<TypeHandle> {
        self.stack.pop()
    }

    pub fn current(&self) -> Option<TypeHandle> {
        self.stack.last().copied()
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// =========================================================================
// populate — populate SemaResult definition tables from the AST.
//
// A Rust port of `src/sema/populate.zig`.
// Responsibility: walk module declarations and dispatch to the corresponding
// conversion functions, filling type_defs/func_sigs/trait_defs.
//
// Differences from the Zig original:
// - The arena_alloc parameter is dropped: Rust uses Box<[T]> / Vec<T> for owned
//   data.
// - anytype parameters → concrete types (via Decl variant destructuring).
// - *const TypeNode → TypeRef + &AstArena dereference.
// - `orelse ... catch unreachable` → `unwrap_or_else`.
// - No `_force_analysis` (Rust has no lazy analysis).
//
// Dependencies: one-way dependency on crate::Ast (Module/Decl/TypeNode, etc.) +
// the existing SemaResult put methods.
// =========================================================================

use crate::ast::Ast::{
    ConstructorDef, RecordFieldType, TypeDef as AstTypeDef,
};

/// populate main entry: walk a module's declarations and fill SemaResult's
/// definition tables.
///
/// Iterates `module.declarations`, dispatching by declaration kind:
/// - `Decl::FunDecl` → `ast_fun_decl_to_func_sig`
/// - `Decl::TypeDecl` → `ast_type_decl_to_type_def`
/// - `Decl::TraitDecl` → `ast_trait_decl_to_trait_def`
/// - Others (ImportDecl/PackDecl/ExprDecl) → skipped
///
/// Returns `false` if a duplicate-definition error occurs (when a put method
/// returns `false`).
pub fn populate_sema_result_from_ast<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    decl: &'a crate::ast::Ast::Spanned<Decl<'a>>,
    ast: &AstArena<'a>,
    module_name: &str,
) -> bool {
    match &decl.node {
        Decl::FunDecl { name, type_params, params, return_type, is_async, .. } => {
            if ast_fun_decl_to_func_sig(arena, sema_result, name, type_params, params, *return_type, *is_async, ast, module_name) {
                sema_result.record_func_sig_owner(name, module_name);
                true
            } else {
                false
            }
        }
        Decl::TypeDecl { name, type_params, base_types, def, methods, .. } => {
            ast_type_decl_to_type_def(arena, sema_result, name, type_params, base_types, def, ast, decl.span, module_name);
            // Register methods inside the type block into
            // TypeDefInfo.methods (indexed by method_idx).
            for method in methods.iter() {
                ast_method_to_func_sig(arena, sema_result, name, method, ast);
            }
            true
        }
        Decl::TraitDecl { name, parents, methods, .. } => {
            if ast_trait_decl_to_trait_def(arena, sema_result, name, parents, methods, ast, module_name) {
                sema_result.record_trait_def_owner(
                    &sema_result.resolve_trait_key_in(module_name, name),
                    module_name,
                );
                true
            } else {
                false
            }
        }
        _ => true, // ImportDecl/PackDecl/ExprDecl skipped
    }
}

/// Walk all declarations of a module, populating SemaResult in bulk.
///
/// Convenience wrapper: calls `populate_sema_result_from_ast` for each
/// declaration in `module.declarations`. Returns `false` if any declaration
/// fails to populate (returns `false`).
pub fn populate_module<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    module: &'a crate::ast::Ast::Module<'a>,
    all_modules: &[&'a crate::ast::Ast::Module<'a>],
) -> bool {
    // Module context for name canonicalization during populate (ctor fields,
    // alias targets, bases) — forward references within the module resolve
    // through the pending own-type set. NESTED types (declared inside
    // function bodies) are included: they register during check, and their
    // canonical keys must be predictable from the start.
    sema_result.current_module_name = module.name.to_string();
    sema_result.pending_own_types = collect_module_type_names(module);
    sema_result.pending_own_traits = collect_module_trait_names(module);
    // Import bookkeeping EARLY: populate-time canonicalization (bases, alias
    // targets, ctor fields) resolves against the module's imports, which the
    // check-time process_import_decls pass would otherwise record too late.
    for decl in &module.declarations {
        if let crate::ast::Ast::Decl::ImportDecl { module_path, .. } = &decl.node {
            if !module_path.is_empty() {
                record_module_import(sema_result, module.name, &module_path.join("."));
            }
        }
    }
    let mut ok = true;
    // Same-module duplicate type declarations are a hard error. put_type_def
    // treats same-module re-registration as an idempotent no-op (the populate
    // pipeline runs twice per module), so genuine duplicates must be caught
    // here, at the declaration level.
    let mut declared_traits: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut declared_types: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for decl in &module.declarations {
        if let crate::ast::Ast::Decl::TraitDecl { name, .. } = &decl.node {
            if !declared_traits.insert(name) {
                sema_result.add_error(SemaError::new_with_path(
                    &format!(
                        "duplicate trait definition: '{}'",
                        canonical_type_key(module.name, name)
                    ),
                    module.name,
                    decl.span.line,
                    decl.span.column,
                ));
                ok = false;
                continue; // skip only the duplicate declaration
            }
        }
        if let crate::ast::Ast::Decl::TypeDecl { name, .. } = &decl.node {
            if !declared_types.insert(name) {
                sema_result.add_error(SemaError::new_with_path(
                    &format!(
                        "duplicate type definition: '{}'",
                        canonical_type_key(module.name, name)
                    ),
                    module.name,
                    decl.span.line,
                    decl.span.column,
                ));
                ok = false;
                continue;
            }
        }
        if !populate_sema_result_from_ast(arena, sema_result, decl, &module.arena, module.name) {
            ok = false;
        }
    }
    expand_inherited_methods(module, all_modules, arena, sema_result);
    ok
}

/// Every trait name declared anywhere in the module — the trait twin of
/// `collect_module_type_names` (feeds `pending_own_traits`).
pub fn collect_module_trait_names(
    module: &crate::ast::Ast::Module<'_>,
) -> std::collections::HashSet<String> {
    struct TraitDeclNames<'m> {
        names: std::collections::HashSet<String>,
        arena: &'m crate::ast::Ast::AstArena<'m>,
    }
    impl<'a, 'm> crate::ast::Ast::AstVisitor<'a> for TraitDeclNames<'m> {
        fn visit_decl(&mut self, decl: &'a crate::ast::Ast::Spanned<crate::ast::Ast::Decl<'a>>) {
            if let crate::ast::Ast::Decl::TraitDecl { name, .. } = &decl.node {
                self.names.insert(name.to_string());
            }
        }
        fn visit_stmt(&mut self, stmt: crate::ast::Ast::StmtId) {
            if let crate::ast::Ast::Stmt::LocalDecl { decl } = &self.arena.stmt(stmt).node {
                if let crate::ast::Ast::Decl::TraitDecl { name, .. } = decl.as_ref() {
                    self.names.insert(name.to_string());
                }
            }
        }
    }
    let mut v = TraitDeclNames {
        names: std::collections::HashSet::new(),
        arena: &module.arena,
    };
    crate::ast::Ast::walk_module(&mut v, &module.arena, module);
    v.names
}

/// Record one import declaration into `module_imports`: the import spelling
/// is mapped to the logical paths of the user modules it names (exact or
/// tail-segment match — `import Counter` → `src.Counter`). std/builtin
/// imports are skipped (std types live on the bare-key fallback). Called from
/// BOTH populate_module (early — bases/alias targets resolve against imports
/// during populate, before check-time processing) and process_import_decls
/// (idempotent).
pub fn record_module_import(
    sema_result: &mut SemaResult,
    module_name: &str,
    full_path: &str,
) {
    if full_path.starts_with("std.") || full_path.starts_with("builtin.") {
        return;
    }
    let matches: Vec<String> = sema_result
        .user_module_paths
        .iter()
        .filter(|lp| lp.as_str() == full_path || lp.ends_with(&format!(".{}", full_path)))
        .cloned()
        .collect();
    let entry = sema_result
        .module_imports
        .entry(module_name.to_string())
        .or_default();
    for m in matches {
        if !entry.contains(&m) {
            entry.push(m);
        }
    }
}

/// Every type name declared anywhere in the module — top-level `TypeDecl`s
/// plus nested ones inside function/method bodies (`Stmt::LocalDecl`).
/// Feeds `pending_own_types` so module-scoped canonical resolution can
/// resolve forward references to not-yet-registered same-module types
/// (e.g. `type L4 = L4(v: L3)` ahead of L3's declaration).
pub fn collect_module_type_names(
    module: &crate::ast::Ast::Module<'_>,
) -> std::collections::HashSet<String> {
    struct TypeDeclNames<'m> {
        names: std::collections::HashSet<String>,
        arena: &'m crate::ast::Ast::AstArena<'m>,
    }
    impl<'a, 'm> crate::ast::Ast::AstVisitor<'a> for TypeDeclNames<'m> {
        fn visit_decl(&mut self, decl: &'a crate::ast::Ast::Spanned<crate::ast::Ast::Decl<'a>>) {
            if let crate::ast::Ast::Decl::TypeDecl { name, .. } = &decl.node {
                self.names.insert(name.to_string());
            }
        }
        // The shared walker's LocalDecl arm does not invoke visit_decl for
        // nested declarations (it only recurses into method bodies), so
        // nested TypeDecls are intercepted here, at the statement level.
        fn visit_stmt(&mut self, stmt: crate::ast::Ast::StmtId) {
            if let crate::ast::Ast::Stmt::LocalDecl { decl } = &self.arena.stmt(stmt).node {
                if let crate::ast::Ast::Decl::TypeDecl { name, .. } = decl.as_ref() {
                    self.names.insert(name.to_string());
                }
            }
        }
    }
    let mut v = TypeDeclNames {
        names: std::collections::HashSet::new(),
        arena: &module.arena,
    };
    crate::ast::Ast::walk_module(&mut v, &module.arena, module);
    v.names
}

/// Inheritance method expansion: append the bases' methods to each child's
/// method table AFTER the child's own methods (own methods shadow/override by
/// name), recording an `InheritedMethodInstance` per appended entry for the
/// IR stage. v1: base TypeDecls are located in the SAME module as the child
/// (stdlib cross-module bases land with the Map-family reorganization, which
/// will widen this lookup by `base_module`).
/// Base-bound substitution pairs at the REPR level: base type-param names →
/// the bound argument's TypeRepr (read in the child module's arena).
fn inheritance_bound_repr_pairs<'a>(
    b: &crate::ast::Ast::TraitBound<'a>,
    base_params: &[Box<str>],
    child_arena: &AstArena<'a>,
) -> Vec<(Box<str>, TypeRepr)> {
    b.type_args
        .iter()
        .enumerate()
        .filter_map(|(i, &arg)| {
            let tp = base_params.get(i)?;
            let repr = type_node_to_repr(&child_arena.ty(arg).node, child_arena);
            Some((tp.clone(), repr))
        })
        .collect()
}

/// TypeRepr-tree substitution by name (the sig-level twin of
/// `substitute_named_adts_free`).
fn substitute_type_repr(ty: TypeRepr, pairs: &[(Box<str>, TypeRepr)]) -> TypeRepr {
    if pairs.is_empty() {
        return ty;
    }
    match ty {
        TypeRepr::Named(n) => {
            if let Some((_, r)) = pairs.iter().find(|(pn, _)| *pn == n) {
                r.clone()
            } else {
                TypeRepr::Named(n)
            }
        }
        TypeRepr::ThisType => TypeRepr::ThisType,
        TypeRepr::Generic(g, args) => TypeRepr::Generic(
            g,
            args.iter().map(|a| substitute_type_repr(a.clone(), pairs)).collect(),
        ),
        TypeRepr::Nullable(i) => TypeRepr::Nullable(Box::new(substitute_type_repr(*i, pairs))),
        TypeRepr::Ref(i) => TypeRepr::Ref(Box::new(substitute_type_repr(*i, pairs))),
        TypeRepr::RawPtr(i) => TypeRepr::RawPtr(Box::new(substitute_type_repr(*i, pairs))),
        TypeRepr::Function(ps, r) => TypeRepr::Function(
            ps.iter().map(|a| substitute_type_repr(a.clone(), pairs)).collect(),
            Box::new(substitute_type_repr(*r, pairs)),
        ),
        TypeRepr::Array(e, sz) => TypeRepr::Array(Box::new(substitute_type_repr(*e, pairs)), sz),
    }
}

fn expand_inherited_methods<'a>(
    module: &'a crate::ast::Ast::Module<'a>,
    all_modules: &[&'a crate::ast::Ast::Module<'a>],
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
) {
    for decl in &module.declarations {
        let crate::ast::Ast::Decl::TypeDecl { name, base_types, .. } = &decl.node
        else {
            continue;
        };
        if base_types.is_empty() {
            continue;
        }
        // Idempotence: populate runs twice per module (pipeline + the
        // check-phase re-populate); a second expansion would append the
        // inherited methods AGAIN — duplicated method-table entries and
        // InheritedMethodInstances compile the body twice into overlapping
        // subgraph ranges (observed: while-loops in inherited methods hang).
        if sema_result
            .inherited_method_instances
            .iter()
            .any(|i| i.type_name.as_ref() == sema_result.resolve_type_key(name))
        {
            continue;
        }
        let canonical_name: Box<str> = sema_result.resolve_type_key(name).into();
        let Some(child_idx) = sema_result.type_def_idx(canonical_name.as_ref()) else {
            continue;
        };
        let child_type_id = crate::types::dynamic_type_id(child_idx);
        // Name → source already available on the child. `None` = the child's
        // OWN method (shadowing = the override, always wins silently);
        // `Some(base)` = inherited from that base — a SECOND base offering the
        // same name is an ambiguity error (must override to disambiguate).
        let own_method_count: usize = module
            .declarations
            .iter()
            .find_map(|d| match &d.node {
                crate::ast::Ast::Decl::TypeDecl { name: n, methods, .. } if *n == *name => {
                    Some(methods.iter().filter(|m| m.body.is_some()).count())
                }
                _ => None,
            })
            .unwrap_or(0);
        let mut present: Vec<(Box<str>, Option<Box<str>>)> = sema_result
            .get_type_def(canonical_name.as_ref())
            .map(|d| {
                d.methods
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        (
                            m.name.clone(),
                            if i < own_method_count { None } else { Some(Box::from("__pending__")) },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        for b in base_types {
            // Locate the base's method AST: same module first, then any
            // module (cross-module bases, e.g. std collections children).
            let base_lookup = |mods: &[&'a crate::ast::Ast::Module<'a>]| -> Option<(&'a [crate::ast::Ast::MethodDecl<'a>], &'a crate::ast::Ast::Module<'a>)> {
                mods.iter().find_map(|m| {
                    m.declarations.iter().find_map(|d| match &d.node {
                        crate::ast::Ast::Decl::TypeDecl { name: bn, methods, .. }
                            if *bn == b.trait_name =>
                        {
                            Some((methods.as_slice(), *m))
                        }
                        _ => None,
                    })
                })
            };
            let Some((base_methods, base_mod)) = base_lookup(&[module])
                .or_else(|| base_lookup(all_modules))
            else {
                continue;
            };
            let base_ast_method_count = base_methods.len();
            let bname: Box<str> = sema_result.resolve_type_key(b.trait_name).into();
            // Iterate the base's METHOD TABLE (own AST methods first — table
            // index == AST index for the own part — then its inherited
            // entries), so chains (base-of-base methods) come along.
            let base_table_len = sema_result
                .get_type_def(bname.as_ref())
                .map(|d| d.methods.len())
                .unwrap_or(0);
            for table_idx in 0..base_table_len {
                let m_name: Box<str> = sema_result
                    .get_type_def(bname.as_ref())
                    .and_then(|d| d.methods.get(table_idx).map(|m| m.name.clone()))
                    .unwrap_or_default();
                if m_name.is_empty() {
                    continue;
                }
                // Body source: own AST method (table_idx < AST count and it
                // has a body), or the ORIGINAL declaring type via the base's
                // own instance record (inheritance chains).
                let mut src: Option<(Box<str>, Box<str>, u16)> = None;
                if table_idx < base_ast_method_count {
                    if base_methods[table_idx].body.is_some() {
                        src = Some((base_mod.name.into(), bname.clone(), table_idx as u16));
                    }
                } else if let Some(inst) = sema_result
                    .inherited_method_instances
                    .iter()
                    .find(|i| i.type_name.as_ref() == bname.as_ref() && i.method_idx as usize == table_idx)
                {
                    src = Some((inst.base_module.clone(), inst.base_type_name.clone(), inst.base_method_idx));
                }
                let Some((src_module, src_type, src_idx)) = src else {
                    continue; // abstract/body-less or unresolvable — nothing to inherit
                };
                if let Some((_, psrc)) = present.iter().find(|(n, _)| n.as_ref() == m_name.as_ref()) {
                    if let Some(prev) = psrc {
                        if prev.as_ref() != "__pending__" && prev.as_ref() != bname.as_ref() {
                            sema_result.add_error(SemaError::new_with_path(
                                &format!(
                                    "ambiguous inherited method '{}': offered by bases '{}' and '{}' — override it in '{}' to disambiguate",
                                    m_name, prev, bname, name
                                ), module.name, decl.span.line, decl.span.column));
                        }
                    }
                    continue; // own override (silent) / same base (dedup) / reported ambiguity
                }
                present.push((m_name.clone(), Some(bname.clone())));
                // Locate the declaring type's AST method for the sig + the
                // body (the ORIGINAL type, which may be the base's base in
                // another module). `src_type` is a canonical key — match the
                // AST declaration by bare name (tail segment).
                let find_decl = |mods: &[&'a crate::ast::Ast::Module<'a>]| -> Option<(&'a crate::ast::Ast::MethodDecl<'a>, &'a crate::ast::Ast::Module<'a>)> {
                    mods.iter().find_map(|m| {
                        m.declarations.iter().find_map(|d| match &d.node {
                            crate::ast::Ast::Decl::TypeDecl { name: bn, methods, .. }
                                if src_type.as_ref().rsplit('.').next() == Some(bn.as_ref()) =>
                            {
                                methods.get(src_idx as usize).map(|bm| (bm, *m))
                            }
                            _ => None,
                        })
                    })
                };
                let Some((bm, decl_mod)) = find_decl(&[module])
                    .or_else(|| find_decl(all_modules))
                else {
                    continue;
                };
                // Append the MethodSigInfo (built from the declaring type's
                // AST, under the child's name — param/return placeholders ride
                // by name like the base's own registration).
                ast_method_to_func_sig(arena, sema_result, name, bm, &decl_mod.arena);
                let method_idx = sema_result
                    .get_type_def(*name)
                    .map(|d| d.methods.len() as u16 - 1)
                    .unwrap_or(0);
                // Pinned-arg substitution on the appended sig's REPRs: a
                // child bound like IntMap<V>(Map<i64, V>) must see the base's
                // K-param references as i64 (the child has no K of its own).
                {
                    let base_sig_params: Vec<Box<str>> = sema_result
                        .get_type_def(bname.as_ref())
                        .map(|d| d.type_params.to_vec())
                        .unwrap_or_default();
                    let pairs = inheritance_bound_repr_pairs(b, &base_sig_params, &module.arena);
                    if !pairs.is_empty() {
                        if let Some(child_idx2) = sema_result.type_def_idx(canonical_name.as_ref()) {
                            if let Some(d) = sema_result.type_defs.get_mut(&child_idx2) {
                                if let Some(sig) = d.methods.last_mut() {
                                    let pt: Vec<TypeRepr> = sig
                                        .param_type_reprs
                                        .to_vec()
                                        .into_iter()
                                        .map(|t| substitute_type_repr(t, &pairs))
                                        .collect();
                                    sig.param_type_reprs = pt.into_boxed_slice();
                                    sig.return_type_repr =
                                        sig.return_type_repr.take().map(|t| substitute_type_repr(t, &pairs));
                                }
                            }
                        }
                    }
                }
                sema_result.inherited_method_instances.push(InheritedMethodInstance {
                    type_id: child_type_id,
                    type_name: canonical_name.clone().into(),
                    method_idx,
                    base_module: decl_mod.name.into(),
                    base_type_name: src_type,
                    base_method_idx: src_idx,
                });
            }
        }
    }
}

// ── Private conversion functions ──

/// Convert a module file path to a logical module path.
///
/// `std/io/Path.frond` → `std.io.Path`
/// `stdlib/std/io/Path.frond` → `std.io.Path` (strips the stdlib/ prefix)
/// `builtin/error/Err.frond` → `builtin.error.Err`
/// Returns `None` if there is no `.frond` suffix or the path is empty.
pub fn module_logical_path(name: &str) -> Option<String> {
    let path = name.strip_suffix(".frond")?;
    // Strip the stdlib/ prefix if present.
    let path = path.strip_prefix("stdlib/").unwrap_or(path);
    if path.is_empty() {
        return None;
    }
    Some(path.replace('/', "."))
}

/// Anchor a module logical path at its last `src` segment
/// (`F:/…/pkg/src.Main` → `src.Main`); already-anchored paths pass through.
fn anchored_logical_path(lp: &str) -> &str {
    match lp.rfind(".src.") {
        Some(pos) => &lp[pos + 1..],
        None => lp,
    }
}

/// Canonical registration key for a type declared in `module_name`.
///
/// Module-scoped type identity: user-module types are registered under a
/// module-qualified name (`src.Main.List` — logical path + bare name), so
/// same-named types in different modules coexist in `type_def_index` and,
/// downstream, in the runtime `type_name` strings that equality, match,
/// dispatch and reflect all compare. std/builtin modules (and synthetic
/// registrations with an empty module) keep the bare name — one std tree,
/// unique by construction, and every existing runtime string, test and
/// .fndo artifact for std types stays unchanged.
pub fn canonical_type_key(module_name: &str, bare: &str) -> String {
    let is_std = module_name.is_empty()
        || module_name.starts_with("std/")
        || module_name.starts_with("builtin/");
    if is_std {
        return bare.to_string();
    }
    match module_logical_path(module_name) {
        Some(mp) => format!("{}.{}", anchored_logical_path(&mp), bare),
        None => bare.to_string(),
    }
}

/// Strip the module prefix from a canonical type name for DISPLAY purposes
/// (`src.Main.Pair<src.Main.Point, i32>` → `Pair<Point, i32>`). Names without
/// a module-qualified head (std/builtin bare names, builtin composites like
/// `u8[]`) pass through unchanged. Generic arguments are stripped
/// recursively; the head prefix is the last '.' outside any `<...>` group.
pub fn display_type_name(name: &str) -> String {
    // Split into head (before the first '<') and the arg list.
    let (head, args) = match name.find('<') {
        Some(open) => (&name[..open], &name[open..]),
        None => (name, ""),
    };
    let bare_head = match head.rfind('.') {
        Some(p) => &head[p + 1..],
        None => head,
    };
    if args.is_empty() {
        return bare_head.to_string();
    }
    let mut out = String::with_capacity(name.len());
    out.push_str(bare_head);
    strip_args_display(args, &mut out);
    out
}

/// `args` starts at the '<' of an argument list; appends the display-stripped
/// arguments (recursively) to `out`, preserving the `<a, b>` shape.
fn strip_args_display(args: &str, out: &mut String) {
    let inner = &args[1..args.len().saturating_sub(1)];
    out.push('<');
    let mut depth: usize = 0;
    let mut seg_start = 0;
    let mut first = true;
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                if !first {
                    out.push_str(", ");
                }
                out.push_str(&display_type_name(inner[seg_start..i].trim()));
                first = false;
                seg_start = i + 1;
            }
            _ => {}
        }
    }
    if !first {
        out.push_str(", ");
    }
    out.push_str(&display_type_name(inner[seg_start..].trim()));
    out.push('>');
}

/// Compute a module-aware expression key: combines a hash of the module name with
/// the ExprId.
///
/// ExprIds are module-specific (each module's AST arena numbers independently),
/// so using an ExprId directly as a global key would cause cross-module
/// collisions. This function combines the module name with the ExprId into a
/// globally unique u64 key.
pub fn module_expr_key(module_name: &str, expr_id: u64) -> u64 {
    use rustc_hash::FxHasher;
    use std::hash::Hasher;
    let mut hasher = FxHasher::default();
    hasher.write(module_name.as_bytes());
    hasher.write_u64(expr_id);
    hasher.finish()
}

/// fun_decl → FuncSigInfo, registered into `sema_result.func_sigs`.
///
/// Top-level functions are registered under their module-qualified key
/// (module, bare name) — see `put_func_sig`.
fn ast_fun_decl_to_func_sig<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    params: &[crate::ast::Ast::Param<'a>],
    return_type: Option<AstTypeRef>,
    is_async: bool,
    ast: &AstArena<'a>,
    module_name: &str,
) -> bool {
    let name: Box<str> = name.into();
    ast_fun_decl_to_func_sig_inner(arena, sema_result, name, type_params, params, return_type, is_async, ast, module_name)
}

/// Construct a `MethodSigInfo` from an AST `MethodDecl` (not registered into
/// `func_sigs`).
///
/// Reuses `resolve_param_type` / `concretize_type` for type resolution, producing
/// the method signature indexed by `method_idx`, stored into
/// `TypeDefInfo.methods`.
fn build_method_sig_info<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    method: &crate::ast::Ast::MethodDecl<'a>,
    ast: &AstArena<'a>,
) -> MethodSigInfo {
    let mut param_is_ref: Vec<bool> = Vec::with_capacity(method.params.len());
    let mut param_type_reprs: Vec<TypeRepr> = Vec::with_capacity(method.params.len());

    for param in &method.params {
        let (_, is_ref, _, repr) = resolve_param_type(arena, param, ast, sema_result);
        param_is_ref.push(is_ref);
        param_type_reprs.push(repr);
    }

    let (_, return_type_repr, is_throwing) = match method.return_type {
        Some(rt) => {
            // The self-contained representation (TypeRepr) of the return type is
            // constructed directly from the AST via `type_node_to_repr`; no need
            // to resolve it into a TypeHandle here (the old `concretize_type`
            // call's result was discarded, had no side effects, and has been
            // removed).
            let repr = type_node_to_repr(&ast.ty(rt).node, ast);
            ((), Some(repr), is_throw_type(&ast.ty(rt).node))
        }
        None => ((), None, false),
    };

    let return_is_ref = match method.return_type {
        Some(rt) => matches!(ast.ty(rt).node, TypeNode::RefType { .. }),
        None => false,
    };

    MethodSigInfo {
        name: method.name.into(),
        param_is_ref: param_is_ref.into_boxed_slice(),
        return_is_ref,
        is_async: method.is_async,
        is_throwing,
        param_type_reprs: param_type_reprs.into_boxed_slice(),
        return_type_repr,
        intrinsic: None,
        has_body: method.body.is_some(),
        // Canonical trait key (module-scoped) — consumers index the trait
        // tables and trait-default subgraphs by the registered name.
        delegate_trait: method
            .delegate
            .as_ref()
            .map(|d| sema_result.resolve_trait_key(d.trait_name).into_boxed_str()),
        // Trait implementations travel with the trait: override and delegate
        // methods are public even without an explicit `pub`, mirroring the
        // "trait methods are public with the trait" rule.
        is_pub: matches!(method.visibility, crate::ast::Ast::Visibility::Public)
            || method.is_override
            || method.delegate.is_some(),
    }
}

/// Public entry point for local type method registration (called from Inference check_decl).
pub fn ast_method_to_func_sig_pub<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    type_name: &str,
    method: &crate::ast::Ast::MethodDecl<'a>,
    ast: &AstArena<'a>,
) -> bool {
    ast_method_to_func_sig(arena, sema_result, type_name, method, ast)
}

/// type-block method → MethodSigInfo, stored into TypeDefInfo.methods (indexed
/// by method_idx).
///
/// `method_idx` is the method's position in the type block's `methods` array
/// (AST declaration order). The IR stage looks up the subgraph in
/// `method_subgraphs` via (type_id, method_idx).
fn ast_method_to_func_sig<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    type_name: &str,
    method: &crate::ast::Ast::MethodDecl<'a>,
    ast: &AstArena<'a>,
) -> bool {
    let sig = build_method_sig_info(arena, sema_result, method, ast);
    // Module-scoped: the AST type name resolves to its canonical key.
    let canonical = sema_result.resolve_type_key(type_name);
    if let Some(type_idx) = sema_result.type_def_idx(canonical.as_str()) {
        if let Some(type_def) = sema_result.type_defs.get_mut(&type_idx) {
            // Idempotent append (S3 / R4): populate runs 2-3x per module and
            // this used to append every time — the method table carried
            // duplicates (dump showed ArrayIter with three `next` entries)
            // and positions drifted off the AST method_idx alignment.
            // Method names are unique within a type: a name match means the
            // registration already landed.
            if type_def.methods.iter().any(|m| m.name == sig.name) {
                return true;
            }
            let mut methods_vec: Vec<MethodSigInfo> = type_def.methods.to_vec();
            methods_vec.push(sig);
            type_def.methods = methods_vec.into_boxed_slice();
        }
        true
    } else {
        false
    }
}

fn ast_fun_decl_to_func_sig_inner<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: Box<str>,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    params: &[crate::ast::Ast::Param<'a>],
    return_type: Option<AstTypeRef>,
    is_async: bool,
    ast: &AstArena<'a>,
    module_name: &str,
) -> bool {

    // type_params: take each TypeParam's name.
    let type_params: Box<[Box<str>]> = type_params.iter().map(|tp| tp.name.into()).collect();

    // param_is_ref: resolve whether each parameter is a reference type.
    let mut param_is_ref: Vec<bool> = Vec::with_capacity(params.len());

    for param in params {
        let (_, is_ref, _, _) = resolve_param_type(arena, param, ast, sema_result);
        param_is_ref.push(is_ref);
    }

    // return_type + is_throwing
    let (return_ty, is_throwing) = match return_type {
        Some(rt) => {
            let ty = concretize_type(arena, rt, &[], ast, sema_result);
            (ty, is_throw_type(&ast.ty(rt).node))
        }
        None => (arena.make(Type::Void), false),
    };

    // return_is_ref: true when the return type is a RefType.
    let return_is_ref = match return_type {
        Some(rt) => matches!(ast.ty(rt).node, TypeNode::RefType { .. }),
        None => false,
    };

    let sig = FuncSigInfo {
        name,
        module_name: module_name.into(),
        type_params,
        return_type: return_ty,
        param_is_ref: param_is_ref.into_boxed_slice(),
        return_is_ref,
        is_async,
        is_throwing,
    };

    sema_result.put_func_sig(sig)
}

/// trait_decl → TraitDefInfo, registered into `sema_result.trait_defs`.
pub(crate) fn ast_trait_decl_to_trait_def<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    parents: &[crate::ast::Ast::TraitBound<'a>],
    methods: &[crate::ast::Ast::MethodDecl<'a>],
    ast: &AstArena<'a>,
    def_module: &str,
) -> bool {
    // Canonical registration name (module-qualified for user modules).
    let name: Box<str> = canonical_type_key(def_module, name).into();
    // Parent validation: must be known traits; a parent naming THIS trait
    // (directly or via a cycle of already-registered links) is rejected.
    let mut parent_names: Vec<Box<str>> = Vec::new();
    for p in parents {
        let pn: Box<str> = sema_result.resolve_trait_key_in(def_module, p.trait_name).into();
        if pn.as_ref() == name.as_ref() {
            sema_result.add_error(SemaError::new(
                &format!("trait '{name}' cannot inherit from itself"),
                0, 1,
            ));
            continue;
        }
        if parent_names.iter().any(|x| x.as_ref() == pn.as_ref()) {
            continue; // duplicate parent — dedup silently
        }
        if sema_result.get_trait_def(pn.as_ref()).is_none() {
            sema_result.add_error(SemaError::new(
                &format!("trait '{name}' parent '{pn}' is not a known trait (parents must be declared before the child)"),
                0, 1,
            ));
            continue;
        }
        if sema_result.trait_parent_closure(pn.as_ref()).iter().any(|a| a.as_ref() == name.as_ref()) {
            sema_result.add_error(SemaError::new(
                &format!("cyclic trait inheritance: '{pn}' already inherits from '{name}'"),
                0, 1,
            ));
            continue;
        }
        parent_names.push(pn);
    }

    let methods: Vec<TraitMethodSig> = methods
        .iter()
        .map(|m| {
            let return_type = match m.return_type {
                Some(rt) => concretize_type(arena, rt, &[], ast, sema_result),
                None => arena.make(Type::Void),
            };
            TraitMethodSig {
                name: m.name.into(),
                param_count: m.params.len() as u8,
                return_type,
                is_async: m.is_async,
                has_body: m.body.is_some(),
            }
        })
        .collect();

    let trait_def = TraitDefInfo {
        name,
        parents: parent_names.into_boxed_slice(),
        methods: methods.into_boxed_slice(),
    };

    sema_result.put_trait_def(trait_def)
}

/// type_decl → TypeDefInfo, dispatched across the 5 def variants, registered
/// into `sema_result.type_defs`.
pub(crate) fn ast_type_decl_to_type_def<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    base_types: &[crate::ast::Ast::TraitBound<'a>],
    def: &AstTypeDef<'a>,
    ast: &AstArena<'a>,
    def_span: Span,
    def_module: &str,
) -> bool {
    // Canonical registration name: user-module types are module-qualified
    // (`src.Main.List`) so same-named types coexist across modules; std and
    // builtin types keep the bare name (see `canonical_type_key`).
    let name: Box<str> = canonical_type_key(def_module, name).into();
    let type_params: Box<[Box<str>]> = type_params.iter().map(|tp| tp.name.into()).collect();

    // ── Inheritance: validate bases + merge their fields ahead of the own
    // fields in the Record branch. A base must be a record or single-ctor
    // ADT (the `type X = X(f: T, ...)` shape the stdlib containers use —
    // those register as Adt kind) declared earlier (std loads before user
    // modules; same-module bases must be declared before the child). Diamond
    // (a shared transitive ancestor) and duplicate field names across
    // bases+own are hard errors — field access is by name and has no
    // qualification syntax. Idempotence: a type already registered (this
    // decl re-processed by a later pass) skips the merge entirely.
    let mut base_names: Vec<Box<str>> = Vec::new();
    // (field name, substituted type, is_pub, repr) collected from bases.
    let mut base_fields: Vec<(Box<str>, TypeHandle, bool, TypeRepr)> = Vec::new();
    // Full substituted ctor sets per base (multi-ctor ADT inheritance).
    let mut base_ctor_sets: Vec<Vec<CtorDefInfo>> = Vec::new();
    let already_registered = sema_result.type_def_has(name.as_ref());
    for b in base_types.iter().take(if already_registered { 0 } else { usize::MAX }) {
        // Base names resolve module-scoped (own module → std → unique user),
        // so an inherited base with a std-colliding name still binds to the
        // local type when the child declares it locally.
        let bname: Box<str> = sema_result.resolve_type_key_in(def_module, b.trait_name).into();
        // Snapshot the base's data (owned) so validation and substitution
        // below can freely re-borrow sema_result.
        let base_snapshot = sema_result.get_type_def(bname.as_ref()).map(|d| {
            (
                d.kind,
                d.constructors.len(),
                d.type_params.to_vec(),
                d.bases.to_vec(),
                d.constructors.to_vec(),
            )
        });
        let (base_kind, base_ctor_count, base_params, _base_own_bases, base_all_ctors) = match base_snapshot {
            Some(x) => x,
            None => {
                sema_result.add_error(SemaError::new_with_path(
                    &format!("inheritance base '{bname}' is not a known type (bases must be declared before the child)"), def_module, def_span.line, def_span.column));
                continue;
            }
        };
        let base_is_admissible =
            matches!(base_kind, TypeDefKind::Record)
                || matches!(base_kind, TypeDefKind::Adt);
        if !base_is_admissible {
            sema_result.add_error(SemaError::new_with_path(
                &format!("inheritance base '{bname}' must be a record or ADT (newtype/alias bases are rejected)"), def_module, def_span.line, def_span.column));
            continue;
        }
        // Diamond check: the new base's ancestor chain must not contain an
        // already-accepted base, and vice versa.
        let mut chain: Vec<String> = Vec::new();
        collect_ancestor_names(sema_result, bname.as_ref(), &mut chain);
        let mut earlier_chain: Vec<String> = Vec::new();
        for other in &base_names {
            collect_ancestor_names(sema_result, other.as_ref(), &mut earlier_chain);
        }
        for other in &base_names {
            if chain.iter().any(|a| a == other.as_ref()) {
                sema_result.add_error(SemaError::new_with_path(
                    &format!("diamond inheritance forbidden: base '{other}' is a transitive ancestor of base '{bname}'"), def_module, def_span.line, def_span.column));
            }
        }
        if earlier_chain.iter().any(|a| a == bname.as_ref()) {
            sema_result.add_error(SemaError::new_with_path(
                &format!("diamond inheritance forbidden: base '{bname}' is a transitive ancestor of an earlier base"), def_module, def_span.line, def_span.column));
        }
        base_names.push(bname.clone());
        // Substitution: base's declared type params → the bound's arg handles
        // (name-keyed placeholder Adts, the alias convention).
        let arg_handles: Vec<TypeHandle> = b
            .type_args
            .iter()
            .map(|&arg| concretize_type(arena, arg, &[], ast, sema_result))
            .collect();
        let mut pairs: Vec<(Box<str>, TypeHandle)> = Vec::new();
        let mut repr_pairs: Vec<(Box<str>, TypeRepr)> = Vec::new();
        for (i, tp) in base_params.iter().enumerate() {
            if let Some(&h) = arg_handles.get(i) {
                pairs.push((tp.clone(), h));
                if let Some(&arg_node) = b.type_args.get(i) {
                    repr_pairs.push((
                        tp.clone(),
                        type_node_to_repr(&ast.ty(arg_node).node, ast),
                    ));
                }
            }
        }
        // Substitute the base's FULL ctor set (multi-ctor ADT bases use it
        // for ctor-set inheritance; single-ctor bases only feed base_fields).
        let mut sub_ctors: Vec<CtorDefInfo> = Vec::with_capacity(base_all_ctors.len());
        for bc in &base_all_ctors {
            let mut c = bc.clone();
            c.type_name = name.as_ref().into();
            let ft: Vec<TypeHandle> = c
                .field_types
                .to_vec()
                .into_iter()
                .map(|t| substitute_named_adts_free(arena, t, &pairs))
                .collect();
            c.field_types = ft.into_boxed_slice();
            let fr: Vec<TypeRepr> = c
                .field_type_reprs
                .to_vec()
                .into_iter()
                .map(|r| substitute_type_repr(r, &repr_pairs))
                .collect();
            c.field_type_reprs = fr.into_boxed_slice();
            sub_ctors.push(c);
        }
        base_ctor_sets.push(sub_ctors);
        let base_ctor = &base_all_ctors[0];
        for i in 0..base_ctor.field_names.len() {
            let fname = base_ctor.field_names[i].clone().unwrap_or_else(|| "_".into());
            base_fields.push((
                fname,
                substitute_named_adts_free(arena, base_ctor.field_types[i], &pairs),
                base_ctor.field_is_pub[i],
                substitute_type_repr(base_ctor.field_type_reprs[i].clone(), &repr_pairs),
            ));
        }
    }

    let (kind, constructors, target_type_name, target_type) = match def {
        AstTypeDef::Adt { constructors: ctor_defs } => {
            if !base_names.is_empty() && ctor_defs.len() > 1 {
                sema_result.add_error(SemaError::new_with_path(
                    "inheritance on a multi-constructor ADT is not supported (ADT children may not add constructors)", def_module, def_span.line, def_span.column));
            }
            let mut ctors: Vec<CtorDefInfo> = ctor_defs
                .iter()
                .map(|c| constructor_def_to_ctor_info(arena, c, name.as_ref(), ast, sema_result, def_span, def_module))
                .collect();
            let multi_ctor_base = base_ctor_sets.iter().any(|cs| cs.len() > 1);
            if multi_ctor_base {
                // ADT child of a multi-ctor base: inherit the ctor set
                // VERBATIM (children may not add constructors — an open sum
                // would break match exhaustiveness). Requires the zero-field
                // child form `= X()`.
                if ctors.len() != 1 || !ctors[0].field_names.is_empty() || ctor_defs[0].fields.len() != 0 {
                    sema_result.add_error(SemaError::new_with_path(
                        "a multi-constructor ADT base requires the zero-field child form '= Child() { ... }' (no own fields, no added constructors)", def_module, def_span.line, def_span.column));
                }
                ctors = base_ctor_sets.concat();
            } else if ctors.len() == 1 && !base_fields.is_empty() {
                // Single-ctor ADT children (= X(...) / = X()) merge base fields
                // exactly like records — this is the stdlib container shape.
                merge_base_fields_into_ctor(&mut ctors[0], base_fields, sema_result, def_span, def_module);
            }
            (TypeDefKind::Adt, ctors, None, None)
        }
        AstTypeDef::Record { fields } => {
            let mut ctor = record_fields_to_ctor_info(arena, fields, name.as_ref(), ast, sema_result, def_span, def_module);
            if !base_fields.is_empty() {
                merge_base_fields_into_ctor(&mut ctor, base_fields, sema_result, def_span, def_module);
            }
            (TypeDefKind::Record, vec![ctor], None, None)
        }
        AstTypeDef::Alias { target } => {
            // Generic aliases: bind each type param to a name-keyed
            // placeholder Adt so the stored target keeps them recoverable —
            // use sites substitute real arguments by name
            // (`substitute_named_adts`). Non-generic aliases pass no
            // placeholders and are unaffected.
            let placeholder_args: Vec<TypeHandle> = type_params
                .iter()
                .map(|tp| arena.make_adt(tp.clone(), Box::new([])))
                .collect();
            let target_ty = concretize_type(arena, *target, &placeholder_args, ast, sema_result);
            let target_name = type_name_from_node(Some(*target), ast)
                .map(|n| sema_result.resolve_type_key_in(def_module, n));
            (
                TypeDefKind::Alias,
                Vec::new(),
                target_name.map(|n| n.into()),
                Some(target_ty),
            )
        }
        AstTypeDef::Newtype { name: nt_name, inner } => {
            let target_ty = concretize_type(arena, *inner, &[], ast, sema_result);
            let target_name = type_name_from_node(Some(*inner), ast)
                .map(|n| sema_result.resolve_type_key_in(def_module, n));
            let target_repr = type_node_to_repr(&ast.ty(*inner).node, ast);
            let ctor = CtorDefInfo {
                name: (*nt_name).into(),
                type_name: name.clone(),
                field_names: Box::new([Some("_0".into())]),
                field_types: Box::new([target_ty]),
                // Newtype 的单字段即值本身:构造即解构,newtype 保持可构造。
                field_is_pub: Box::new([true]),
                is_newtype: true,
                return_type_name: None,
                return_type_node: None,
                field_type_reprs: Box::new([target_repr]),
                def_span,
                def_module: def_module.into(),
            };
            (
                TypeDefKind::Newtype,
                vec![ctor],
                target_name.map(|n| n.into()),
                Some(target_ty),
            )
        }
    };

    let type_def = TypeDefInfo {
        name,
        kind,
        constructors: constructors.into_boxed_slice(),
        type_params,
        target_type_name,
        target_type,
        methods: Box::new([]),
        bases: base_names.into_boxed_slice(),
    };

    sema_result.put_type_def(type_def, def_module)
}

// ── Helper functions ──

/// Prepend the (already substituted) base fields ahead of the child ctor's
/// own fields, rejecting duplicate names across bases+own.
fn merge_base_fields_into_ctor(
    ctor: &mut CtorDefInfo,
    base_fields: Vec<(Box<str>, TypeHandle, bool, TypeRepr)>,
    sema_result: &mut SemaResult,
    def_span: Span,
    def_module: &str,
) {
    let mut seen: std::collections::HashSet<Box<str>> = std::collections::HashSet::new();
    for (fname, _, _, _) in &base_fields {
        if !seen.insert(fname.clone()) {
            sema_result.add_error(SemaError::new_with_path(
                &format!("inherited field name collision: '{fname}' appears in more than one base"),
                def_module,
                def_span.line,
                def_span.column,
            ));
        }
    }
    for n in ctor.field_names.iter().flatten() {
        if !seen.insert(n.clone()) {
            sema_result.add_error(SemaError::new_with_path(
                &format!("field '{n}' collides with an inherited base field"),
                def_module,
                def_span.line,
                def_span.column,
            ));
        }
    }
    let mut names: Vec<Option<Box<str>>> = Vec::with_capacity(base_fields.len() + ctor.field_names.len());
    let mut types: Vec<TypeHandle> = Vec::with_capacity(base_fields.len() + ctor.field_types.len());
    let mut pubs: Vec<bool> = Vec::with_capacity(base_fields.len() + ctor.field_is_pub.len());
    let mut reprs: Vec<TypeRepr> = Vec::with_capacity(base_fields.len() + ctor.field_type_reprs.len());
    for (fname, fty, fpub, frepr) in base_fields {
        // Keep ONE copy per name (collisions were already reported above;
        // duplicating them here would trip the ctor duplicate-field check).
        if names.iter().flatten().any(|n| n.as_ref() == fname.as_ref()) {
            continue;
        }
        names.push(Some(fname));
        types.push(fty);
        pubs.push(fpub);
        reprs.push(frepr);
    }
    names.extend(ctor.field_names.iter().cloned());
    types.extend(ctor.field_types.iter().copied());
    pubs.extend(ctor.field_is_pub.iter().copied());
    reprs.extend(ctor.field_type_reprs.iter().cloned());
    ctor.field_names = names.into_boxed_slice();
    ctor.field_types = types.into_boxed_slice();
    ctor.field_is_pub = pubs.into_boxed_slice();
    ctor.field_type_reprs = reprs.into_boxed_slice();
}

/// Recursively collect the transitive base-type names of `type_name`
/// (direct + indirect; the type itself excluded). Cycle-safe via the
/// seen-dedup on `out`.
fn collect_ancestor_names(sema_result: &SemaResult, type_name: &str, out: &mut Vec<String>) {
    let Some(def) = sema_result.get_type_def(type_name) else { return };
    for b in def.bases.iter() {
        if out.iter().any(|x| x == b.as_ref()) {
            continue;
        }
        out.push(b.as_ref().to_string());
        collect_ancestor_names(sema_result, b.as_ref(), out);
    }
}

/// Name-keyed structural substitution over a TypeHandle — registration-phase
/// twin of `InferContext::substitute_named_adts` (arm-for-arm), keyed on
/// placeholder `Adt(name)` nodes so a base's generic fields unfold against
/// the inheritance bound's arguments (`type IntMap(Map<i64, V>)`).
pub(crate) fn substitute_named_adts_free(
    arena: &mut TypeArena,
    ty: TypeHandle,
    pairs: &[(Box<str>, TypeHandle)],
) -> TypeHandle {
    if pairs.is_empty() {
        return ty;
    }
    substitute_named_adts_free_inner(arena, ty, pairs)
}

fn substitute_named_adts_free_inner(
    arena: &mut TypeArena,
    ty: TypeHandle,
    pairs: &[(Box<str>, TypeHandle)],
) -> TypeHandle {
    use crate::types::Type;
    let resolved = arena.resolve(ty);
    match arena.get(resolved) {
        Type::Adt(_) => {
            let (name, type_args) = arena.adt_parts(resolved);
            if let Some((_, h)) = pairs.iter().find(|(n, _)| n.as_ref() == name) {
                return *h;
            }
            let name: Box<str> = name.into();
            let type_args: Vec<TypeHandle> = type_args.to_vec();
            let new_args: Vec<TypeHandle> = type_args
                .iter()
                .map(|&a| substitute_named_adts_free_inner(arena, a, pairs))
                .collect();
            arena.make_adt(name, new_args.into_boxed_slice())
        }
        Type::Fn(_) => {
            let (params, return_type) = arena.fn_parts(resolved);
            let params: Vec<TypeHandle> = params.to_vec();
            let new_params: Vec<TypeHandle> = params
                .iter()
                .map(|&p| substitute_named_adts_free_inner(arena, p, pairs))
                .collect();
            let new_ret = substitute_named_adts_free_inner(arena, return_type, pairs);
            arena.make_fn(new_params.into_boxed_slice(), new_ret)
        }
        Type::Record(_) => {
            let fields = arena.record_fields(resolved).to_vec();
            let name = arena.record_name(resolved).map(|s| s.into());
            let new_fields: Vec<crate::types::FieldType> = fields
                .iter()
                .map(|f| crate::types::FieldType {
                    name: f.name.clone(),
                    ty: substitute_named_adts_free_inner(arena, f.ty, pairs),
                })
                .collect();
            arena.make_record(new_fields.into_boxed_slice(), name)
        }
        Type::Nullable(_) => {
            let inner = arena.nullable_inner(resolved);
            let new_inner = substitute_named_adts_free_inner(arena, inner, pairs);
            arena.make_nullable(new_inner)
        }
        Type::Generic(_) => {
            let (name, args) = arena.generic_parts(resolved);
            let name: Box<str> = name.into();
            let args: Vec<TypeHandle> = args.to_vec();
            let new_args: Vec<TypeHandle> = args
                .iter()
                .map(|&a| substitute_named_adts_free_inner(arena, a, pairs))
                .collect();
            arena.make_generic(name, new_args.into_boxed_slice())
        }
        Type::Array(_) => {
            let (element_type, size) = arena.array_parts(resolved);
            let new_elem = substitute_named_adts_free_inner(arena, element_type, pairs);
            arena.make_array(new_elem, size)
        }
        Type::Throw(_) => {
            let (value_type, error_type) = arena.throw_parts(resolved);
            let new_v = substitute_named_adts_free_inner(arena, value_type, pairs);
            let new_e = substitute_named_adts_free_inner(arena, error_type, pairs);
            arena.make_throw(new_v, new_e)
        }
        Type::Trait(_) => {
            let (name, type_args) = arena.trait_parts(resolved);
            let name: Box<str> = name.into();
            let type_args: Vec<TypeHandle> = type_args.to_vec();
            let new_args: Vec<TypeHandle> = type_args
                .iter()
                .map(|&a| substitute_named_adts_free_inner(arena, a, pairs))
                .collect();
            arena.make_trait(name, new_args.into_boxed_slice())
        }
        Type::Ref(_) => {
            let (inner, is_raw) = arena.ref_parts(resolved);
            let new_inner = substitute_named_adts_free_inner(arena, inner, pairs);
            arena.make_ref(new_inner, is_raw)
        }
        Type::Channel(_) => {
            let elem = arena.channel_elem(resolved);
            let new_elem = substitute_named_adts_free_inner(arena, elem, pairs);
            arena.make_channel(new_elem)
        }
        Type::Async(_) => {
            let value = arena.async_value(resolved);
            let new_value = substitute_named_adts_free_inner(arena, value, pairs);
            arena.make_async(new_value)
        }
        Type::Lazy(_) => {
            let value = arena.lazy_value(resolved);
            let new_value = substitute_named_adts_free_inner(arena, value, pairs);
            arena.make_lazy(new_value)
        }
        Type::Atomic(_) => {
            let elem = arena.atomic_elem(resolved);
            let new_elem = substitute_named_adts_free_inner(arena, elem, pairs);
            arena.make_atomic(new_elem)
        }
        Type::Sender(_) => {
            let elem = arena.sender_elem(resolved);
            let new_elem = substitute_named_adts_free_inner(arena, elem, pairs);
            arena.make_sender(new_elem)
        }
        Type::Receiver(_) => {
            let elem = arena.receiver_elem(resolved);
            let new_elem = substitute_named_adts_free_inner(arena, elem, pairs);
            arena.make_receiver(new_elem)
        }
        Type::ForeignFn(_) => {
            let ret = arena.foreign_fn_ret(resolved);
            let new_ret = substitute_named_adts_free_inner(arena, ret, pairs);
            arena.make_foreign_fn(new_ret)
        }
        // Scalars, Never, Unknown, Void, Null, TraitObject, ModuleRef have no
        // sub-nodes → as-is.
        _ => resolved,
    }
}


/// Resolve a parameter type: returns (TypeHandle, is_ref, type_name, type_repr).
fn resolve_param_type<'a>(
    arena: &mut TypeArena,
    param: &crate::ast::Ast::Param<'a>,
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> (TypeHandle, bool, Option<Box<str>>, TypeRepr) {
    match param.type_annotation {
        Some(tr) => {
            let node = &ast.ty(tr).node;
            let is_ref = matches!(node, TypeNode::RefType { .. });
            let ty = concretize_type(arena, tr, &[], ast, sema_result);
            let name = type_name_from_node(Some(tr), ast).map(|n| n.into());
            let repr = type_node_to_repr(node, ast);
            (ty, is_ref, name, repr)
        }
        None => (
            arena.make_adt("param".into(), Box::new([])),
            false,
            None,
            TypeRepr::Named("unknown".into()),
        ),
    }
}

/// Single type concretization entry point (registration phase): resolves an AST
/// `TypeNode` into a `TypeHandle`.
///
/// Unifies type resolution in the registration phase: structure-preserving
/// (`make_ref`/`make_array`/`make_nullable`/`make_fn`/`make_record`) + Named
/// alias/newtype chain expansion (with cycle detection and a depth limit).
///
/// Differs from `resolve_type_node_resolved`: the latter is specialized for
/// scalar channel-width computation and projects Ref/Array down to Adt(name);
/// this function preserves structure and is the general concretization entry
/// point. The inference phase (which needs the `type_binding_stack`/
/// `this_binding_stack` context) uses `InferContext::type_from_ast_with_params`;
/// since that context cannot be provided by this free function, they are separate
/// phase entry points.
pub(crate) fn concretize_type<'a>(
    arena: &mut TypeArena,
    type_ref: AstTypeRef,
    type_args: &[TypeHandle],
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> TypeHandle {
    let tn = &ast.ty(type_ref).node;
    match tn {
        TypeNode::Named { name } => {
            // Delegate to resolve_named_type_resolved: type_args binding →
            // built-in scalar → alias/newtype chain expansion (visiting cycle
            // detection + MAX_TYPE_RECURSION_DEPTH depth limit) → user-defined
            // Adt. Fixes the precision gap in the old Named branch, which only
            // used Type::from_type_name/make_adt.
            let mut visiting = FxHashSet::default();
            resolve_named_type_resolved(arena, name, type_args, sema_result, &mut visiting)
        }
        TypeNode::Generic { name, args } => {
            // Builtin generics must carry their REAL type arguments. The old
            // code called `Type::from_type_name(name)`, which returns a
            // placeholder (`DetailId(u32::MAX)`) that DISCARDS the args — so
            // every signature declaring `Async<X>` / `Throw<X, E>` /
            // `Channel<T>`… was registered with an opaque placeholder type.
            // Downstream consumers of `func_sigs` then silently
            // mis-handled them (the await value-propagation bug family:
            // #97's tail-`?` Ok re-wrap sees Async<placeholder> instead of
            // Async<Throw<…>> and skips the re-wrap; the raw payload leaks
            // out and `match f().await() { Ok/Err }` hits the fallback
            // panic). Construction mirrors InferContext::make_builtin_generic.
            let resolved: Vec<TypeHandle> = args.iter()
                .map(|&a| concretize_type(arena, a, type_args, ast, sema_result))
                .collect();
            match (*name, resolved.as_slice()) {
                ("Throw", [v, e]) => arena.make_throw(*v, *e),
                ("Channel", [t]) => arena.make_channel(*t),
                ("Async", [t]) => arena.make_async(*t),
                ("Lazy", [t]) => arena.make_lazy(*t),
                ("Atomic", [t]) => arena.make_atomic(*t),
                ("Sender", [t]) => arena.make_sender(*t),
                ("Receiver", [t]) => arena.make_receiver(*t),
                ("ForeignFn", [t]) => arena.make_foreign_fn(*t),
                _ => {
                    if let Some(ty) = Type::from_type_name(name) {
                        // Bare builtin generic name written without args.
                        arena.make(ty)
                    } else {
                        // Source-qualified spellings canonicalize so the
                        // signature identity matches the registered key; bare
                        // names keep their raw spelling (status quo).
                        let key: Box<str> = if name.contains('.') {
                            sema_result.resolve_type_key(name).into()
                        } else {
                            (*name).into()
                        };
                        arena.make_generic(key, Box::new([]))
                    }
                }
            }
        }
        TypeNode::Nullable { inner } => {
            let inner = concretize_type(arena, *inner, type_args, ast, sema_result);
            arena.make_nullable(inner)
        }
        TypeNode::RefType { inner } => {
            let inner = concretize_type(arena, *inner, type_args, ast, sema_result);
            arena.make_ref(inner, false)
        }
        TypeNode::RawPtr { inner } => {
            let inner = concretize_type(arena, *inner, type_args, ast, sema_result);
            arena.make_ref(inner, true)
        }
        TypeNode::Record { .. } => arena.make_record(Vec::<FieldType>::new().into_boxed_slice(), None),
        TypeNode::Function { params, return_type } => {
            // Recursively concretize the parameter and return types so a function
            // type used as an alias target (e.g. `type IntToInt = (i32) -> i32`)
            // preserves its arity and signature. Previously this discarded both,
            // producing a 0-arity `Fn` returning Unknown, which caused callers to
            // skip argument inference and left argument expressions without ExprInfo
            // (surfacing as "missing ExprInfo" ICEs in the IR builder).
            let param_tys: Vec<TypeHandle> = params
                .iter()
                .map(|&p| concretize_type(arena, p, type_args, ast, sema_result))
                .collect();
            let ret_ty = concretize_type(arena, *return_type, type_args, ast, sema_result);
            arena.make_fn(param_tys.into_boxed_slice(), ret_ty)
        }
        TypeNode::Array { element_type, size } => {
            let elem = concretize_type(arena, *element_type, type_args, ast, sema_result);
            arena.make_array(elem, *size)
        }
        TypeNode::ThisType => {
            // `This` is stored as an `Adt` with the detail name "This", so the
            // match must be detail-aware (same as `resolve_type_node_resolved`);
            // a bare `name()` compare would never match ("adt" != "This").
            for &ta in type_args {
                if type_handle_name_matches(arena, ta, "This") {
                    return ta;
                }
            }
            arena.make_adt("This".into(), Box::new([]))
        }
        TypeNode::KindAnnotated { inner, .. } => {
            concretize_type(arena, *inner, type_args, ast, sema_result)
        }
    }
}

/// Whether a `TypeNode` is a `Throw<T, E>` type.
///
/// `Throw` is represented in `TypeNode` as `Generic { name: "Throw", args: [V, E] }`.
fn is_throw_type(tn: &TypeNode) -> bool {
    matches!(tn, TypeNode::Generic { name, .. }
        if Type::from_type_name(name).is_some_and(|t| t.family() == TypeFamily::Throw))
}

/// Recursively convert an AST `TypeNode` into a self-contained `TypeRepr`
/// (independent of `AstArena` references).
/// Used during the sema phase to serialize method return-type information for
/// later cross-module `lookup_method_type` use.
fn type_node_to_repr<'a>(tn: &TypeNode<'a>, ast: &AstArena<'a>) -> TypeRepr {
    match tn {
        TypeNode::Named { name } => TypeRepr::Named((*name).into()),
        TypeNode::ThisType => TypeRepr::ThisType,
        TypeNode::Generic { name, args } => {
            let repr_args: Vec<TypeRepr> = args
                .iter()
                .map(|&a| type_node_to_repr(&ast.ty(a).node, ast))
                .collect();
            TypeRepr::Generic((*name).into(), repr_args.into_boxed_slice())
        }
        TypeNode::Nullable { inner } => {
            TypeRepr::Nullable(Box::new(type_node_to_repr(&ast.ty(*inner).node, ast)))
        }
        TypeNode::RefType { inner } => {
            TypeRepr::Ref(Box::new(type_node_to_repr(&ast.ty(*inner).node, ast)))
        }
        TypeNode::RawPtr { inner } => {
            TypeRepr::RawPtr(Box::new(type_node_to_repr(&ast.ty(*inner).node, ast)))
        }
        TypeNode::Function {
            params,
            return_type,
        } => {
            let p: Vec<TypeRepr> = params
                .iter()
                .map(|&a| type_node_to_repr(&ast.ty(a).node, ast))
                .collect();
            let r = type_node_to_repr(&ast.ty(*return_type).node, ast);
            TypeRepr::Function(p.into_boxed_slice(), Box::new(r))
        }
        TypeNode::Record { .. } => TypeRepr::Named("record".into()),
        TypeNode::Array {
            element_type,
            size,
        } => TypeRepr::Array(
            Box::new(type_node_to_repr(&ast.ty(*element_type).node, ast)),
            *size,
        ),
        TypeNode::KindAnnotated { inner, .. } => {
            type_node_to_repr(&ast.ty(*inner).node, ast)
        }
    }
}

/// Convert a `ConstructorDef` into a `CtorDefInfo`.
fn constructor_def_to_ctor_info<'a>(
    arena: &mut TypeArena,
    c: &ConstructorDef<'a>,
    type_name: &str,
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
    def_span: Span,
    def_module: &str,
) -> CtorDefInfo {
    let mut field_names: Vec<Option<Box<str>>> = Vec::with_capacity(c.fields.len());
    let mut field_types: Vec<TypeHandle> = Vec::with_capacity(c.fields.len());
    let mut field_type_reprs: Vec<TypeRepr> = Vec::with_capacity(c.fields.len());

    let mut field_is_pub: Vec<bool> = Vec::with_capacity(c.fields.len());
    for f in &c.fields {
        field_names.push(f.name.map(|n| n.into()));
        let ty = concretize_type(arena, f.ty, &[], ast, sema_result);
        field_types.push(ty);
        field_type_reprs.push(type_node_to_repr(&ast.ty(f.ty).node, ast));
        field_is_pub.push(f.is_pub);
    }

    CtorDefInfo {
        name: c.name.into(),
        type_name: type_name.into(),
        field_names: field_names.into_boxed_slice(),
        field_types: field_types.into_boxed_slice(),
        field_is_pub: field_is_pub.into_boxed_slice(),
        is_newtype: false,
        return_type_name: None,
        return_type_node: c.return_type,
        field_type_reprs: field_type_reprs.into_boxed_slice(),
        def_span,
        def_module: def_module.into(),
    }
}

/// Convert a list of `RecordFieldType` into a single-constructor `CtorDefInfo`
/// (for record types).
fn record_fields_to_ctor_info<'a>(
    arena: &mut TypeArena,
    fields: &[RecordFieldType<'a>],
    type_name: &str,
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
    def_span: Span,
    def_module: &str,
) -> CtorDefInfo {
    let mut field_names: Vec<Option<Box<str>>> = Vec::with_capacity(fields.len());
    let mut field_types: Vec<TypeHandle> = Vec::with_capacity(fields.len());
    let mut field_type_reprs: Vec<TypeRepr> = Vec::with_capacity(fields.len());

    let mut field_is_pub: Vec<bool> = Vec::with_capacity(fields.len());
    for f in fields {
        field_names.push(Some(f.name.into()));
        let ty = concretize_type(arena, f.ty, &[], ast, sema_result);
        field_types.push(ty);
        field_type_reprs.push(type_node_to_repr(&ast.ty(f.ty).node, ast));
        field_is_pub.push(f.is_pub);
    }

    CtorDefInfo {
        name: type_name.into(), // record constructor name == type name
        type_name: type_name.into(),
        field_names: field_names.into_boxed_slice(),
        field_types: field_types.into_boxed_slice(),
        field_is_pub: field_is_pub.into_boxed_slice(),
        is_newtype: false,
        return_type_name: None,
        return_type_node: None,
        field_type_reprs: field_type_reprs.into_boxed_slice(),
        def_span,
        def_module: def_module.into(),
    }
}

// =========================================================================
// sema v2: Witness Table — static dispatch table for trait implementations.
//
// Design rationale (original, not a copy of Swift/Haskell):
// - Trait implementations are materialized at compile time into a WitnessEntry
//   (a function-pointer table).
// - Dispatch is indexed by the type_id on `Type`, in O(1).
// - Replaces the current mangled-name ("TypeName.method") lookup.
// - Naturally fits Frond's type_id / reflection mechanism.
//
// Data structures:
// - WitnessEntry { trait_name, type_id, method_slots }
// - WitnessTable uses FxHashMap<u32, WitnessEntry> (never-recycling u32 allocator)
//   + FxHashMap<(trait_name, type_id), u32> for indexing. Hard delete on purge
//   (no stale holes), consistent with type_defs/func_sigs/trait_defs.
//
// Dispatch flow:
// 1. Infer the receiver type → resolve → obtain type_id (scalars have one
//    directly; ADTs consult type_def).
// 2. Build key = (trait_name, type_id).
// 3. Look up the witness table → obtain method_slots.
// 4. method_slots[method_name] → method slot index.
// 5. The slot index points to a MonomorphInstance (the compiled method body).
// =========================================================================

/// A witness-table entry: the implementation of a trait on a type.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Trait name (e.g. "Show", "Eq", "Error").
    pub trait_name: Box<str>,
    /// `type_id` of the implementing type (matches the `type_id` on `Type`).
    pub type_id: u16,
    /// Method slots: method_name → method_idx (position in
    /// `TypeDefInfo.methods`).
    pub method_slots: FxHashMap<Box<str>, u16>,
    /// Name of the implementing type (used in error messages).
    pub type_name: Box<str>,
}

/// Witness table: an index over all trait implementations.
///
/// Indexed by (trait_name, type_id) to reach a `WitnessEntry`, then by
/// `method_name` to reach a method slot.
///
/// Uses `FxHashMap<u32, WitnessEntry>` with a never-recycling u32 allocator
/// (consistent with `type_defs`/`func_sigs`/`trait_defs`). `remove` performs a
/// true hard delete — no stale holes left in the table.
#[derive(Default, Clone)]
pub struct WitnessTable {
    /// Entry storage: entry_id → WitnessEntry. Freed ids are never reused.
    entries: FxHashMap<u32, WitnessEntry>,
    /// Monotonic entry-id allocator (never recycles, so stale indices are never
    /// accidentally reused after a purge).
    next_entry_id: u32,
    /// Index: (trait_name, type_id) → entry_id into `entries`.
    index: FxHashMap<(Box<str>, u16), u32>,
}

impl WitnessTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trait implementation.
    ///
    /// If (trait_name, type_id) already exists, the old implementation is
    /// overwritten (redefinition is allowed) at the same entry_id — no new id is
    /// allocated.
    pub fn register(
        &mut self,
        trait_name: &str,
        type_id: u16,
        type_name: &str,
        method_slots: FxHashMap<Box<str>, u16>,
    ) {
        let key = (trait_name.into(), type_id);
        if let Some(&id) = self.index.get(&key) {
            // Overwrite the existing implementation at the same entry_id.
            self.entries.insert(id, WitnessEntry {
                trait_name: trait_name.into(),
                type_id,
                method_slots,
                type_name: type_name.into(),
            });
        } else {
            let id = self.next_entry_id;
            self.next_entry_id += 1;
            self.entries.insert(id, WitnessEntry {
                trait_name: trait_name.into(),
                type_id,
                method_slots,
                type_name: type_name.into(),
            });
            self.index.insert(key, id);
        }
    }

    /// Whether a type implements a given trait.
    #[inline]
    pub fn implements(&self, trait_name: &str, type_id: u16) -> bool {
        self.index.contains_key(&(trait_name.into(), type_id))
    }

    /// Look up the `method_idx` for a method of a trait implementation.
    ///
    /// Returns the method's position index in `TypeDefInfo.methods`.
    /// The IR layer uses (type_id, method_idx) to look up the subgraph in
    /// `method_subgraphs`.
    pub fn resolve_method(
        &self,
        trait_name: &str,
        type_id: u16,
        method_name: &str,
    ) -> Option<u16> {
        let key = (trait_name.into(), type_id);
        let &id = self.index.get(&key)?;
        let entry = self.entries.get(&id)?;
        entry.method_slots.get(method_name).copied()
    }

    /// Get all method names of a trait implementation.
    pub fn trait_methods(&self, trait_name: &str, type_id: u16) -> Vec<&str> {
        let key = (trait_name.into(), type_id);
        match self.index.get(&key) {
            Some(&id) => self.entries.get(&id).map(|e| {
                e.method_slots
                    .keys()
                    .map(|k| k.as_ref())
                    .collect()
            }).unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Get all entries (for reflection / diagnostics).
    ///
    /// Returns an iterator over `&WitnessEntry`. Callers should iterate directly
    /// (no `.iter()` needed).
    #[inline]
    pub fn entries(&self) -> std::collections::hash_map::Values<'_, u32, WitnessEntry> {
        self.entries.values()
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a witness entry by (trait_name, type_id).
    /// Used during incremental sema purge.
    ///
    /// Performs a true hard delete: removes the key from `index` and the entry
    /// from `entries`. The freed entry_id is never reused (the allocator is
    /// monotonic), so no stale references can accidentally resolve to a different
    /// trait implementation after a purge.
    pub fn remove(&mut self, trait_name: &str, type_id: u16) {
        let key: (Box<str>, u16) = (trait_name.into(), type_id);
        if let Some(id) = self.index.remove(&key) {
            self.entries.remove(&id);
        }
    }
}
