//! Sema.rs — core data structures for semantic analysis.
//!
//! The single source of truth for the type system is `crate::types`
//! (`Ty` / `TypeArena` / `TypeOps`). The legacy `ConcreteType` / `TypeDescriptor`
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
//! (`Ty` / `TypeArena` / `TypeOps` / `DynamicOpsRegistry`) and `crate::Ast`
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
    TypeHandle, Ty, TypeFamily, DetailId, EnvId, FieldType, TraitMethodSig,
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
pub struct EnvArena {
    envs: Vec<EnvNode>,
}

impl Default for EnvArena {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvArena {
    pub fn new() -> Self {
        EnvArena { envs: Vec::new() }
    }

    /// Create the top-level environment (no parent).
    pub fn root(&mut self) -> EnvId {
        let id = EnvId(self.envs.len() as u32);
        self.envs.push(EnvNode {
            bindings: FxHashMap::default(),
            parent: None,
        });
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
    /// Whether this is a trait object (`Ty::TraitObject`): the IR layer uses this
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
}

/// Trait definition info (replaces the signature portion of IRBuilder's
/// trait_table).
#[derive(Debug, Clone)]
pub struct TraitDefInfo {
    pub name: Box<str>,
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
    /// `type_id` of the implementing type (matches the `type_id` on `Ty`).
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
    /// Type name → index into `type_defs`.
    pub type_def_index: FxHashMap<String, u16>,
    /// Dynamic type_id → type name (reverse index for O(1) lookup).
    /// Updated in tandem with `type_defs` (put_type_def / purge_module).
    pub type_id_to_name: FxHashMap<u16, Box<str>>,
    /// Trait definition table.
    pub trait_defs: FxHashMap<u16, TraitDefInfo>,
    /// u16 index allocator for `trait_defs` (never recycles).
    pub next_trait_def_id: u16,
    /// Trait name → index into `trait_defs`.
    pub trait_def_index: FxHashMap<String, u16>,
    /// Function signature table.
    pub func_sigs: FxHashMap<u16, FuncSigInfo>,
    /// u16 index allocator for `func_sigs` (never recycles).
    pub next_func_sig_id: u16,
    /// Function name → index into `func_sigs`.
    pub func_sig_index: FxHashMap<String, u16>,
    /// Coroutine metadata table.
    pub coroutine_metas: Vec<CoroutineMeta>,
    /// Constructor name → list of (type_def_index << 16 | ctor_index).
    /// Supports multiple types having same-named constructors (e.g. `FileKind.File`
    /// and `type File`); disambiguation is done by type context or qualified names.
    pub ctor_def_index: FxHashMap<String, Vec<u32>>,
    /// Import alias table: short name → alias target.
    pub import_aliases: FxHashMap<String, AliasTarget>,
    /// Monomorphization instance table.
    pub monomorph_instances: Vec<MonomorphInstance>,
    /// Monomorphization instance name → index into `monomorph_instances`.
    pub monomorph_index: FxHashMap<u64, u32>,
    /// Trait-default-method monomorphization instance table (collected during
    /// the later Sema phase by the Monomorph module).
    pub trait_default_instances: Vec<TraitDefaultInstance>,
    /// Dynamic ops registry (ops for user types, replaces TypeDescriptorPool).
    pub dynamic_ops: DynamicOpsRegistry,
    /// Call-site → instance mapping.
    pub call_instantiations: FxHashMap<u64, u32>,
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
    pub field_id_map: FxHashMap<String, u16>,
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
            expr_types: FxHashMap::default(),
            errors: Vec::new(),
            warnings: Vec::new(),
            has_error: false,
            type_defs: FxHashMap::default(),
            next_type_def_id: 0,
            type_def_index: FxHashMap::default(),
            type_id_to_name: FxHashMap::default(),
            trait_defs: FxHashMap::default(),
            next_trait_def_id: 0,
            trait_def_index: FxHashMap::default(),
            func_sigs: FxHashMap::default(),
            next_func_sig_id: 0,
            func_sig_index: FxHashMap::default(),
            coroutine_metas: Vec::new(),
            ctor_def_index: FxHashMap::default(),
            import_aliases: FxHashMap::default(),
            monomorph_instances: Vec::new(),
            monomorph_index: FxHashMap::default(),
            trait_default_instances: Vec::new(),
            dynamic_ops: DynamicOpsRegistry::new(),
            call_instantiations: FxHashMap::default(),
            field_accesses: FxHashMap::default(),
            method_dispatches: FxHashMap::default(),
            reflect_metas: FxHashMap::default(),
            resolved_types: FxHashMap::default(),
            field_id_map: FxHashMap::default(),
            witness_table: WitnessTable::new(),
            module_func_recv_exprs: FxHashSet::default(),
            module_const_recv_exprs: FxHashMap::default(),
            pattern_ctor_types: FxHashMap::default(),
            module_ownership: ModuleOwnership::default(),
        }
    }

    // ── Expressions ──

    /// Record the type of an expression.
    pub fn put_expr(&mut self, expr_id: u64, info: ExprInfo) {
        self.expr_types.insert(expr_id, info);
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
        // Type-name conflict: reject (same-named types cannot be redefined).
        if self.type_def_index.contains_key(def.name.as_ref()) {
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
            self.ctor_def_index
                .entry(ctor.name.to_string())
                .or_default()
                .push(packed_idx);
        }
        self.type_def_index.insert(def.name.to_string(), idx);
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
        self.field_id_map.insert(key.clone(), field_id);
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
        self.field_id_map.get(&key).copied()
    }

    /// Look up a type definition by name.
    pub fn get_type_def(&self, name: &str) -> Option<&TypeDefInfo> {
        let idx = *self.type_def_index.get(name)?;
        self.type_defs.get(&idx)
    }

    /// Look up a constructor definition by constructor name.
    /// Returns the first match when multiple types share the same constructor
    /// name; use `get_ctor_defs` for disambiguation.
    pub fn get_ctor_def(&self, name: &str) -> Option<&CtorDefInfo> {
        let packed_idx = self.ctor_def_index.get(name)?.first()?;
        let type_idx = (*packed_idx >> 16) as u16;
        let ctor_idx = (*packed_idx & 0xFFFF) as u16;
        let def = self.type_defs.get(&type_idx)?;
        def.constructors.get(ctor_idx as usize)
    }

    /// Look up all constructor definitions matching a constructor name.
    /// Returns an empty slice when no match is found; returns multiple entries
    /// when different types share the same constructor name (e.g. `FileKind.File`
    /// and `type File`).
    pub fn get_ctor_defs(&self, name: &str) -> Vec<&CtorDefInfo> {
        match self.ctor_def_index.get(name) {
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
    define_table_registry!(put_trait_def, get_trait_def, trait_defs, trait_def_index, TraitDefInfo, next_trait_def_id);

    // ── Function signatures ──
    define_table_registry!(put_func_sig, get_func_sig, func_sigs, func_sig_index, FuncSigInfo, next_func_sig_id);

    /// Record that a func_sig belongs to a module (for incremental purge).
    /// Looks up the current index by name; call after a successful `put_func_sig`.
    pub fn record_func_sig_owner(&mut self, name: &str, module_name: &str) {
        if let Some(&idx) = self.func_sig_index.get(name) {
            self.module_ownership.func_sig_indices
                .entry(module_name.to_string())
                .or_default()
                .insert(idx);
        }
    }

    /// Record that a trait_def belongs to a module (for incremental purge).
    /// Looks up the current index by name; call after a successful `put_trait_def`.
    pub fn record_trait_def_owner(&mut self, name: &str, module_name: &str) {
        if let Some(&idx) = self.trait_def_index.get(name) {
            self.module_ownership.trait_def_indices
                .entry(module_name.to_string())
                .or_default()
                .insert(idx);
        }
    }

    // ── Method signatures (Ty-driven) ──

    /// Look up `method_idx` (the position in `TypeDefInfo.methods`) by type name
    /// and method name.
    ///
    /// The IR layer uses (type_id, method_idx) to look up the subgraph in
    /// `method_subgraphs`. Returning `None` means the type has no such method
    /// (it may be a trait default method; consult the witness_table).
    pub fn lookup_method_idx(&self, type_name: &str, method_name: &str) -> Option<u16> {
        let &type_idx = self.type_def_index.get(type_name)?;
        let type_def = &self.type_defs[&type_idx];
        type_def
            .methods
            .iter()
            .position(|m| m.name.as_ref() == method_name)
            .map(|i| i as u16)
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
                self.module_const_recv_exprs.remove(k);
            }
        }
        // pattern_ctor_types: key is (String, u32), filter by module name
        self.pattern_ctor_types.retain(|(m, _), _| m.as_str() != module_name);

        // === Category B: global definition tables ===
        // Truly remove entries from both the index HashMap and the value HashMap.
        // Freed u16 indices are never reused (the allocator is monotonic).

        // type_defs
        if let Some(indices) = self.module_ownership.type_def_indices.remove(module_name) {
            for idx in indices {
                if let Some(def) = self.type_defs.remove(&idx) {
                    self.type_def_index.remove(def.name.as_ref());
                    // Remove the type_id → name reverse-index entry.
                    let type_id = dynamic_type_id(idx);
                    self.type_id_to_name.remove(&type_id);
                    // Remove constructor entries from ctor_def_index
                    for ctor in &def.constructors {
                        if let Some(vec) = self.ctor_def_index.get_mut(ctor.name.as_ref()) {
                            vec.retain(|&packed| (packed >> 16) as u16 != idx);
                            if vec.is_empty() {
                                self.ctor_def_index.remove(ctor.name.as_ref());
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
                    self.func_sig_index.remove(sig.name.as_ref());
                }
            }
        }

        // trait_defs
        if let Some(indices) = self.module_ownership.trait_def_indices.remove(module_name) {
            for idx in indices {
                if let Some(def) = self.trait_defs.remove(&idx) {
                    self.trait_def_index.remove(def.name.as_ref());
                }
            }
        }

        // field_id_map
        if let Some(keys) = self.module_ownership.field_id_keys.remove(module_name) {
            for k in keys {
                self.field_id_map.remove(&k);
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
// A Rust port of `src/sema/builtin_types.zig`. Unifies the scalar name → Ty
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
/// up their arity. The `nongeneric` group has dedicated `Ty`/`TypeNode` variants
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
        /// - `build_fn_type_from_sig` reconstructs the full `Ty::Fn` via
        ///   `type_repr_to_handle`.
        pub fn register_builtin_method_sigs(sema_result: &mut SemaResult) {
            /// Build a single built-in method signature. The `type` field is a
            /// `Ty::Void` placeholder (does not affect type checking;
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
                }
            }

            /// Register a synthetic `TypeDefInfo` for a single built-in type.
            fn register(
                sema_result: &mut SemaResult,
                type_name: &str,
                type_params: &[&str],
                methods: Vec<MethodSigInfo>,
            ) {
                if sema_result.type_def_index.contains_key(type_name) {
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
                };
                sema_result.put_type_def(def, "");
            }

            // ── generic group: enters BUILTIN_GENERIC_TYPES + method registration ──
            $(
                register(sema_result, $gname, &[$($gp),*], vec![$($gmethod),*]);
            )*
            // ── nongeneric group: method registration only (has dedicated Ty variant) ──
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
            sig("send", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(284))),
            sig("recv", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::ChannelAwait)),
            sig("close", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(285))),
        ],
        "Atomic" : ["T"] = [
            sig("swap", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::BinOp(317))),
            sig("compare_exchange", vec![TypeRepr::ThisType, TypeRepr::Named("T".into()), TypeRepr::Named("T".into())], Some(TypeRepr::Named("bool".into())), Some(IntrinsicKind::TriOp(318))),
            sig("load", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::UnOp(315))),
            sig("store", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(316))),
        ],
        "Async" : ["T"] = [
            sig("status", vec![TypeRepr::ThisType], Some(TypeRepr::Named("str".into())), None),
            sig("await", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::Await)),
            sig("cancel", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(42))),
        ],
        "Sender" : ["T"] = [
            sig("send", vec![TypeRepr::ThisType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(284))),
            sig("close", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(285))),
        ],
        "Receiver" : ["T"] = [
            sig("recv", vec![TypeRepr::ThisType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::ChannelAwait)),
            sig("close", vec![TypeRepr::ThisType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(285))),
        ],
        "Lazy" : ["T"] = [],
    }
    nongeneric {
        "array" : ["T"] = [
            sig("len", vec![TypeRepr::ThisType], Some(TypeRepr::Named("usize".into())), Some(IntrinsicKind::UnOp(35))),
            sig("is_empty", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), None),
        ],
        "str" : [] = [
            sig("len", vec![TypeRepr::ThisType], Some(TypeRepr::Named("usize".into())), Some(IntrinsicKind::UnOp(35))),
            sig("is_empty", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), None),
            sig("bytes", vec![TypeRepr::ThisType], Some(TypeRepr::Array(Box::new(TypeRepr::Named("u8".into())), None)), Some(IntrinsicKind::UnOp(287))),
        ],
        "nullable" : ["T"] = [
            sig("is_null", vec![TypeRepr::ThisType], Some(TypeRepr::Named("bool".into())), None),
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
        Ty::Adt(_) => arena.adt_parts(h).0 == name,
        Ty::Generic(_) => arena.generic_parts(h).0 == name,
        Ty::Trait(_) => arena.trait_parts(h).0 == name,
        // Other types (including built-in generics Throw/Channel/Async/Lazy/Atomic/
        // Sender/Receiver/Timer and scalars/str/void) uniformly go through
        // `ty.name()`, the single source of truth.
        ty => ty.name() == name,
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
    // 1. Prefer type_args bindings (generic type parameters).
    for &ta in type_args {
        if type_handle_name_matches(arena, ta, name) {
            return ta;
        }
    }
    // 2. Built-in scalar/str/null/void.
    if let Some(ty) = Ty::from_type_name(name) {
        return arena.make(ty);
    }
    // Cyclic-alias detection: `name` already in `visiting` means a cycle;
    // stop recursing.
    if visiting.contains(name) {
        return arena.make_adt(name.into(), Box::new([]));
    }
    // Recursion-depth limit: `visiting.len()` is the current depth; stop
    // recursing past the limit to prevent stack overflow.
    if visiting.len() >= MAX_TYPE_RECURSION_DEPTH {
        return arena.make_adt(name.into(), Box::new([]));
    }
    visiting.insert(name.to_string());
    // 3. Consult type_defs to resolve the alias/newtype chain.
    //    Extract the needed info (owned) to release the immutable borrow,
    //    allowing subsequent `&mut` calls.
    let (target_ty, target_name): (Option<TypeHandle>, Option<String>) =
        match sema_result.get_type_def(name) {
            Some(td) => (
                td.target_type,
                td.target_type_name.as_deref().map(String::from),
            ),
            None => (None, None),
        };
    if let Some(inner_ty) = target_ty {
        // alias/newtype has a target TypeHandle: return it directly.
        visiting.remove(name);
        return inner_ty;
    }
    if let Some(ttn) = target_name {
        // target_type_name is known: recursively resolve to the final concrete
        // type.
        let result = resolve_named_type_resolved(arena, &ttn, type_args, sema_result, visiting);
        visiting.remove(name);
        return result;
    }
    // 4. Other user-defined types → create a named Adt.
    visiting.remove(name);
    arena.make_adt(name.into(), Box::new([]))
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
            if Ty::from_type_name(name).is_some_and(|t| t.family() == TypeFamily::Lazy)
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
            arena.make_adt(inner_name.into(), Box::new([]))
        }
        TypeNode::RawPtr { inner } => {
            let inner_name = type_name_from_node(Some(*inner), ast).unwrap_or("ptr");
            arena.make_adt(inner_name.into(), Box::new([]))
        }
        TypeNode::Record { .. } => arena.make_record(Vec::<FieldType>::new().into_boxed_slice(), None),
        TypeNode::Function { .. } => {
            let ret = arena.make(Ty::Unknown);
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
/// Note: this stack holds `TypeHandle`s (Ty indices); type resolution goes
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
            if ast_fun_decl_to_func_sig(arena, sema_result, name, type_params, params, *return_type, *is_async, ast) {
                sema_result.record_func_sig_owner(name, module_name);
                true
            } else {
                false
            }
        }
        Decl::TypeDecl { name, type_params, def, methods, .. } => {
            ast_type_decl_to_type_def(arena, sema_result, name, type_params, def, ast, decl.span, module_name);
            // Register methods inside the type block into
            // TypeDefInfo.methods (indexed by method_idx).
            for method in methods.iter() {
                ast_method_to_func_sig(arena, sema_result, name, method, ast);
            }
            true
        }
        Decl::TraitDecl { name, methods, .. } => {
            if ast_trait_decl_to_trait_def(arena, sema_result, name, methods, ast) {
                sema_result.record_trait_def_owner(name, module_name);
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
) -> bool {
    let mut ok = true;
    for decl in &module.declarations {
        if !populate_sema_result_from_ast(arena, sema_result, decl, &module.arena, module.name) {
            ok = false;
        }
    }
    ok
}

// ── Private conversion functions ──

/// Convert a module file path to a logical module path.
///
/// `std/io/Path.kz` → `std.io.Path`
/// `stdlib/std/io/Path.kz` → `std.io.Path` (strips the stdlib/ prefix)
/// `builtin/error/Err.kz` → `builtin.error.Err`
/// Returns `None` if there is no `.kz` suffix or the path is empty.
pub fn module_logical_path(name: &str) -> Option<String> {
    let path = name.strip_suffix(".kz")?;
    // Strip the stdlib/ prefix if present.
    let path = path.strip_prefix("stdlib/").unwrap_or(path);
    if path.is_empty() {
        return None;
    }
    Some(path.replace('/', "."))
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
/// Top-level functions are registered under their bare name. Methods inside a
/// type block are registered with a mangled name `TypeName.method` via
/// `ast_method_to_func_sig`.
fn ast_fun_decl_to_func_sig<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    params: &[crate::ast::Ast::Param<'a>],
    return_type: Option<AstTypeRef>,
    is_async: bool,
    ast: &AstArena<'a>,
) -> bool {
    let name: Box<str> = name.into();
    ast_fun_decl_to_func_sig_inner(arena, sema_result, name, type_params, params, return_type, is_async, ast)
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
    }
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
    if let Some(&type_idx) = sema_result.type_def_index.get(type_name) {
        if let Some(type_def) = sema_result.type_defs.get_mut(&type_idx) {
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
        None => (arena.make(Ty::Void), false),
    };

    // return_is_ref: true when the return type is a RefType.
    let return_is_ref = match return_type {
        Some(rt) => matches!(ast.ty(rt).node, TypeNode::RefType { .. }),
        None => false,
    };

    let sig = FuncSigInfo {
        name,
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
    methods: &[crate::ast::Ast::MethodDecl<'a>],
    ast: &AstArena<'a>,
) -> bool {
    let name: Box<str> = name.into();

    let methods: Vec<TraitMethodSig> = methods
        .iter()
        .map(|m| {
            let return_type = match m.return_type {
                Some(rt) => concretize_type(arena, rt, &[], ast, sema_result),
                None => arena.make(Ty::Void),
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
    def: &AstTypeDef<'a>,
    ast: &AstArena<'a>,
    def_span: Span,
    def_module: &str,
) -> bool {
    let name: Box<str> = name.into();
    let type_params: Box<[Box<str>]> = type_params.iter().map(|tp| tp.name.into()).collect();

    let (kind, constructors, target_type_name, target_type) = match def {
        AstTypeDef::Adt { constructors: ctor_defs } => {
            let ctors: Vec<CtorDefInfo> = ctor_defs
                .iter()
                .map(|c| constructor_def_to_ctor_info(arena, c, name.as_ref(), ast, sema_result, def_span, def_module))
                .collect();
            (TypeDefKind::Adt, ctors, None, None)
        }
        AstTypeDef::Record { fields } => {
            let ctor = record_fields_to_ctor_info(arena, fields, name.as_ref(), ast, sema_result, def_span, def_module);
            (TypeDefKind::Record, vec![ctor], None, None)
        }
        AstTypeDef::Alias { target } => {
            let target_ty = concretize_type(arena, *target, &[], ast, sema_result);
            let target_name = type_name_from_node(Some(*target), ast);
            (
                TypeDefKind::Alias,
                Vec::new(),
                target_name.map(|n| n.into()),
                Some(target_ty),
            )
        }
        AstTypeDef::Newtype { name: nt_name, inner } => {
            let target_ty = concretize_type(arena, *inner, &[], ast, sema_result);
            let target_name = type_name_from_node(Some(*inner), ast);
            let target_repr = type_node_to_repr(&ast.ty(*inner).node, ast);
            let ctor = CtorDefInfo {
                name: (*nt_name).into(),
                type_name: name.clone(),
                field_names: Box::new([Some("_0".into())]),
                field_types: Box::new([target_ty]),
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
    };

    sema_result.put_type_def(type_def, def_module)
}

// ── Helper functions ──

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
            // used Ty::from_type_name/make_adt.
            let mut visiting = FxHashSet::default();
            resolve_named_type_resolved(arena, name, type_args, sema_result, &mut visiting)
        }
        TypeNode::Generic { name, .. } => {
            if let Some(ty) = Ty::from_type_name(name) {
                arena.make(ty)
            } else {
                arena.make_generic((*name).into(), Box::new([]))
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
        TypeNode::Function { .. } => {
            let ret = arena.make(Ty::Unknown);
            arena.make_fn(Vec::<TypeHandle>::new().into_boxed_slice(), ret)
        }
        TypeNode::Array { element_type, size } => {
            let elem = concretize_type(arena, *element_type, type_args, ast, sema_result);
            arena.make_array(elem, *size)
        }
        TypeNode::ThisType => {
            for &ta in type_args {
                if arena.get(ta).name() == "This" {
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
        if Ty::from_type_name(name).is_some_and(|t| t.family() == TypeFamily::Throw))
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

    for f in &c.fields {
        field_names.push(f.name.map(|n| n.into()));
        let ty = concretize_type(arena, f.ty, &[], ast, sema_result);
        field_types.push(ty);
        field_type_reprs.push(type_node_to_repr(&ast.ty(f.ty).node, ast));
    }

    CtorDefInfo {
        name: c.name.into(),
        type_name: type_name.into(),
        field_names: field_names.into_boxed_slice(),
        field_types: field_types.into_boxed_slice(),
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

    for f in fields {
        field_names.push(Some(f.name.into()));
        let ty = concretize_type(arena, f.ty, &[], ast, sema_result);
        field_types.push(ty);
        field_type_reprs.push(type_node_to_repr(&ast.ty(f.ty).node, ast));
    }

    CtorDefInfo {
        name: type_name.into(), // record constructor name == type name
        type_name: type_name.into(),
        field_names: field_names.into_boxed_slice(),
        field_types: field_types.into_boxed_slice(),
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
// - Dispatch is indexed by the type_id on `Ty`, in O(1).
// - Replaces the current mangled-name ("TypeName.method") lookup.
// - Naturally fits Kuzo's type_id / reflection mechanism.
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
    /// `type_id` of the implementing type (matches the `type_id` on `Ty`).
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
