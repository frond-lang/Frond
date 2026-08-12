//! Inference.rs — type inference algorithm layer
//!
//! Split out from Sema.rs. Depends on crate::Sema (type system foundations) + crate::Relations (type relation checks).
//! Responsibilities: type inference, constraint solving, flow-sensitive narrowing.
//! Monomorphization instance collection has moved to crate::Monomorph (unified entry point for the post-sema stage).

use crate::sema::Sema::*;
use crate::sema::Relations::*;
use crate::ast::Ast::{
    AstArena, BinaryOp, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module,
    Pattern, PatternId, PatternLiteral, PatternRef, Stmt, StmtId,
    TypeNode, TypeRef as AstTypeRef,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Generates the match arm for numeric literal inference (shared structure for IntLit/FloatLit).
/// `$suffix_fn` is the suffix→TypeHandle method, `$predicate` is the expected-type predicate, and `$fallback` is the default Type variant.
macro_rules! numeric_lit {
    ($self:expr, $suffix:expr, $expected:expr, $suffix_fn:ident, $predicate:ident, $fallback:ident) => {{
        if let Some(suf) = $suffix {
            if let Some(ty) = $self.$suffix_fn(suf) {
                return ty;
            }
        }
        if let Some(exp) = $expected {
            let resolved = $self.arena.resolve(exp);
            if $self.arena.get(resolved).$predicate() {
                return exp;
            }
        }
        $self.make_builtin(Type::$fallback)
    }};
}

/// Range-check an integer literal's raw text against the target scalar type's range.
/// Returns `Some(error message)` when out of range or unparseable; `None` when in range.
/// Mirrors `ir::Builder::check_int_range` so sema and IR report consistently (Bug #72: stage consistency).
fn check_int_literal_range(raw: &str, tag: crate::types::ValueTag) -> Option<String> {
    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = cleaned
        .strip_prefix("0x").map(|s| (s, 16u32))
        .or_else(|| cleaned.strip_prefix("0o").map(|s| (s, 8)))
        .or_else(|| cleaned.strip_prefix("0b").map(|s| (s, 2)))
        .unwrap_or((cleaned.as_str(), 10));
    // i128/u128 literals cannot overflow their own parse; only syntax errors are possible.
    match tag {
        crate::types::ValueTag::I128 => {
            return match i128::from_str_radix(digits, radix) {
                Ok(_) => None,
                Err(_) => Some(format!("invalid integer literal '{}'", raw)),
            };
        }
        crate::types::ValueTag::U128 => {
            return match u128::from_str_radix(digits, radix) {
                Ok(_) => None,
                Err(_) => Some(format!("invalid integer literal '{}'", raw)),
            };
        }
        _ => {}
    }
    let (min, max, name): (i128, i128, &str) = match tag {
        crate::types::ValueTag::I8    => (i8::MIN    as i128, i8::MAX    as i128, "i8"),
        crate::types::ValueTag::I16   => (i16::MIN   as i128, i16::MAX   as i128, "i16"),
        crate::types::ValueTag::I32   => (i32::MIN   as i128, i32::MAX   as i128, "i32"),
        crate::types::ValueTag::I64   => (i64::MIN   as i128, i64::MAX   as i128, "i64"),
        crate::types::ValueTag::U8    => (0,                   u8::MAX    as i128, "u8"),
        crate::types::ValueTag::U16   => (0,                   u16::MAX   as i128, "u16"),
        crate::types::ValueTag::U32   => (0,                   u32::MAX   as i128, "u32"),
        crate::types::ValueTag::U64   => (0,                   u64::MAX   as i128, "u64"),
        crate::types::ValueTag::Isize => (isize::MIN as i128, isize::MAX as i128, "isize"),
        crate::types::ValueTag::Usize => (0,                   usize::MAX as i128, "usize"),
        _ => return None,
    };
    match i128::from_str_radix(digits, radix) {
        Ok(v) if v < min || v > max => Some(format!(
            "integer literal '{}' is out of range for {} (valid range: {}..={})",
            raw, name, min, max
        )),
        Ok(_) => None,
        Err(_) => Some(format!("invalid integer literal '{}'", raw)),
    }
}

/// Inference context: encapsulates all state needed for type inference.
///
/// Lifetime: a single TypeArena is shared across the whole module's sema stage; InferContext holds a `&mut` reference to it.
/// Instantiation-mode context: used when resolving types in a monomorphized function body.
///
/// Design: two-phase writes to avoid aliasing conflicts.
/// - Runtime type results are staged in local_expr_types.
/// - After the run completes, take_local_expr_types() transfers them into MonomorphInstance.expr_types.
///
/// Does not hold func_decls (lifetime-bearing references); monomorphization triggers in the Call branch are orchestrated externally.
pub struct InstantiationCtx {
    /// Function name of the current instance (used to short-circuit generic recursion: func_name == cur_func_name).
    pub func_name: Box<str>,
    /// type_args of the current instance (same length as type_params, positionally aligned).
    pub type_args: Box<[TypeHandle]>,
    /// Type-parameter name → type_args index (fast lookup).
    pub type_param_map: FxHashMap<String, u16>,
    /// Module name of the callee (for cross-module cases the expr_types key must match what the IR Builder looks up).
    pub module_name: String,
    /// Staged expression type table (key = module_expr_key(module_name, expr_id)).
    pub local_expr_types: FxHashMap<u64, ExprInfo>,
    /// Staged field_accesses metadata (key = module_expr_key).
    pub local_field_accesses: FxHashMap<u64, FieldAccessInfo>,
    /// Cycle detection: cache_key currently being instantiated → instance_id (supports forward references).
    pub in_progress: FxHashMap<u64, u32>,
}

/// type_binding_stack and this_binding_stack push/pop as impl/trait/fn blocks are entered and exited.
/// env is the local variable environment (EnvArena); expected_return drives reverse inference of the return type.
pub struct InferContext<'a> {
    pub arena: &'a mut TypeArena,
    pub sema_result: &'a mut SemaResult,
    pub type_binding_stack: TypeBindingStack,
    pub this_binding_stack: ThisBindingStack,
    /// Pending implicit-this access marker, set by Ident/Call fallback when a bare
    /// identifier/call resolves to an instance field/method. Consumed by infer_expr
    /// after store_expr_info to update the staged ExprInfo.
    pub pending_implicit_this: Option<(ExprId, crate::sema::Sema::ImplicitThisAccess)>,
    pub env: EnvArena,
    /// Expected return type of the current function (used for reverse inference of throw expressions, etc.).
    pub expected_return: Option<TypeHandle>,
    /// sema v2: constraint solver (lazy solving + snapshot/rollback).
    pub solver: ConstraintSolver,
    /// sema v2: flow-sensitive narrowing context (path-sensitive type refinement).
    pub flow_ctx: FlowContext,
    /// sema v2: witness table (static dispatch table for trait implementations).
    pub witness_table: WitnessTable,
    /// Module path → module-specific EnvId mapping.
    ///
    /// Each module (including path prefixes) creates its own env at registration time (parent points to root_env or the parent path's env).
    /// The module's functions/types are registered in this env. ModuleRef lookups search by bare name directly in the corresponding env, with no mangled name required.
    ///
    /// Hierarchical example:
    ///   "std"            → env_std (parent=root_env), binds "io"→ModuleRef("std.io", env_std_io)
    ///   "std.io"         → env_std_io (parent=env_std), binds "File"→ModuleRef("std.io.File", env_std_io_file)
    ///   "std.io.File"    → env_std_io_file (parent=env_std_io), binds "open"→Fn(...)
    ///
    /// This makes lookups for `std.io.File.open(...)` fully structured through the env chain:
    ///   std → env_std.lookup("io") → ModuleRef("std.io", env_std_io)
    ///       → env_std_io.lookup("File") → ModuleRef("std.io.File", env_std_io_file)
    ///       → Call: env_std_io_file.lookup("open") → Fn(...)
    pub module_envs: FxHashMap<String, EnvId>,
    /// Logical path of the module currently being checked (e.g. "Math.Geometry"), used to register mangled names.
    /// Set at the start of check_module_with_env for use by methods like infer_stmt that do not take a module parameter.
    pub current_module_logical_path: Option<String>,
    /// Module-specific EnvId of the module currently being checked.
    /// Looked up from module_envs at the start of check_module_with_env; used to register symbols during predeclare_declarations.
    pub current_module_env: Option<EnvId>,
    /// Filename of the module currently being checked (e.g. "Math/Geometry.kz"), used as part of the expr_types composite key.
    /// Prevents ExprIds from different modules from colliding in the global expr_types.
    pub current_module_name: String,
    /// Diagnostic trace table: records (TypeHandle, Span) for each expression's inference result, used to trace unresolved TypeVars back to their source locations.
    /// Only populated when KUZO_SEMA_TRACE is enabled, to avoid memory overhead during normal compilation.
    pub type_trace: Vec<(TypeHandle, crate::ast::Ast::Span)>,
    /// Constructor short name → module EnvId where the type is defined (Zig-style @This semantics).
    ///
    /// When `import std.time.Duration` and the module defines `pub type Duration`,
    /// predefine uses redefine to overwrite the ModuleRef with the constructor Fn. This map retains
    /// "type name → source module env", allowing the MethodCall path 0b to fall back to free functions in the module
    /// (when the type name equals the filename, the type is treated as a module namespace).
    pub ctor_module_envs: FxHashMap<String, EnvId>,
    /// Instantiation-mode context: None = HM mode, Some = instantiation mode.
    /// Set to Some when resolving types in a monomorphized function body; None during HM type checking.
    pub instantiation_ctx: Option<InstantiationCtx>,
    /// Tracks local binding mutability per environment scope: (env_id, name) → is_mutable.
    /// Used to detect val→var / var→val mutability-changing shadowing (Bug #76).
    pub local_mutability: FxHashMap<(u32, String), bool>,
    /// Name of the trait whose default methods are currently being inferred (None outside trait blocks).
    /// Used by lookup_method_type to resolve bare method calls (implicit this) inside trait default
    /// methods, where current_this_type() is a rigid TypeVar that has no method table of its own.
    pub current_trait_name: Option<Box<str>>,
}

/// Checks whether a type references any unresolved TypeVar (in unresolved_set).
/// Used during diagnostics to trace unresolved TypeVars back to their expression locations.
fn type_contains_any_unresolved(
    ty: TypeHandle,
    arena: &TypeArena,
    unresolved_set: &FxHashSet<u32>,
) -> bool {
    let resolved = arena.resolve(ty);
    match arena.get(resolved) {
        Type::TypeVar(idx) => unresolved_set.contains(&idx),
        Type::Fn(_) => {
            let (params, return_type) = arena.fn_parts(resolved);
            params.iter().any(|&p| type_contains_any_unresolved(p, arena, unresolved_set))
                || type_contains_any_unresolved(return_type, arena, unresolved_set)
        }
        Type::Record(_) => arena.record_fields(resolved)
            .iter()
            .any(|f| type_contains_any_unresolved(f.ty, arena, unresolved_set)),
        Type::Adt(_) => {
            let (_, type_args) = arena.adt_parts(resolved);
            type_args
                .iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        Type::Nullable(_) => {
            let inner = arena.nullable_inner(resolved);
            type_contains_any_unresolved(inner, arena, unresolved_set)
        }
        Type::Ref(_) => {
            let (inner, _) = arena.ref_parts(resolved);
            type_contains_any_unresolved(inner, arena, unresolved_set)
        }
        Type::Generic(_) => {
            let (_, args) = arena.generic_parts(resolved);
            args.iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        Type::Array(_) => {
            let (element_type, _) = arena.array_parts(resolved);
            type_contains_any_unresolved(element_type, arena, unresolved_set)
        }
        Type::Throw(_) => {
            let (value_type, error_type) = arena.throw_parts(resolved);
            type_contains_any_unresolved(value_type, arena, unresolved_set)
                || type_contains_any_unresolved(error_type, arena, unresolved_set)
        }
        Type::Trait(_) => {
            let (_, type_args) = arena.trait_parts(resolved);
            type_args
                .iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        _ => false,
    }
}

// =========================================================================
// Usefulness algorithm (Maranget) — pattern matrix exhaustiveness checking
// =========================================================================

/// Constructor identifier used by the usefulness algorithm.
#[derive(Clone, PartialEq)]
enum PatCtor {
    Adt(Box<str>),
    Bool(bool),
    Int(Box<str>),
    Float(Box<str>),
    Char(u32),
    Str(Box<str>),
    Null,
}

/// Normalized pattern (arena-independent) for the usefulness algorithm.
/// Or-patterns are expanded into multiple alternatives during normalization.
#[derive(Clone)]
enum NormPat {
    Wild,
    Ctor(PatCtor, Vec<NormPat>),
}

/// Unwrap an inline `Pattern::Guard` to retrieve the inner pattern.
fn unwrap_guard_pat(ast: &AstArena<'_>, pat: PatternRef) -> PatternRef {
    match &ast.pattern(pat).node {
        Pattern::Guard { pattern, .. } => *pattern,
        _ => pat,
    }
}

/// Normalize an AST pattern into one or more `NormPat` alternatives.
/// Or-patterns expand to multiple alternatives; sub-pattern or-patterns produce
/// the cartesian product of alternatives.
fn normalize_pattern(ast: &AstArena<'_>, pat: PatternRef) -> Vec<NormPat> {
    match &ast.pattern(pat).node {
        Pattern::Wildcard => vec![NormPat::Wild],
        Pattern::Variable { name } => {
            if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                vec![NormPat::Ctor(PatCtor::Adt(name.to_string().into_boxed_str()), Vec::new())]
            } else {
                vec![NormPat::Wild]
            }
        }
        Pattern::Constructor { name, patterns } => {
            let ctor = PatCtor::Adt(name.to_string().into_boxed_str());
            // Cartesian product of sub-pattern alternatives.
            let mut alternatives: Vec<Vec<NormPat>> = vec![Vec::new()];
            for &sub_pat in patterns.iter() {
                let sub_alts = normalize_pattern(ast, sub_pat);
                let mut next = Vec::new();
                for existing in &alternatives {
                    for sub_alt in &sub_alts {
                        let mut combined = existing.clone();
                        combined.push(sub_alt.clone());
                        next.push(combined);
                    }
                }
                alternatives = next;
            }
            alternatives.into_iter()
                .map(|subs| NormPat::Ctor(ctor.clone(), subs))
                .collect()
        }
        Pattern::Literal(lit) => {
            let c = match lit {
                PatternLiteral::Bool(b) => PatCtor::Bool(*b),
                PatternLiteral::Int(s) => PatCtor::Int(s.to_string().into_boxed_str()),
                PatternLiteral::Float(s) => PatCtor::Float(s.to_string().into_boxed_str()),
                PatternLiteral::Char(c) => PatCtor::Char(*c),
                PatternLiteral::String(s) => PatCtor::Str(s.to_string().into_boxed_str()),
                PatternLiteral::Null => PatCtor::Null,
            };
            vec![NormPat::Ctor(c, Vec::new())]
        }
        // Record patterns always match the single record constructor; treat as catch-all
        // (field sub-pattern exhaustiveness is not tracked — conservative but safe).
        Pattern::Record { .. } => vec![NormPat::Wild],
        Pattern::OrPattern { left, right } => {
            let mut result = normalize_pattern(ast, *left);
            result.extend(normalize_pattern(ast, *right));
            result
        }
        Pattern::Guard { pattern, .. } => normalize_pattern(ast, *pattern),
    }
}

/// Specialize a pattern matrix with respect to a constructor `target` of arity `arity`.
///
/// For each row: wildcard → expand to `arity` wildcards + remaining columns;
/// matching constructor → extract subpatterns (padded/truncated to `arity`) + remaining;
/// non-matching constructor → drop the row.
fn specialize_matrix(
    matrix: &[Vec<NormPat>],
    col_types: &[TypeHandle],
    target: &PatCtor,
    arity: usize,
    field_types: &[TypeHandle],
) -> (Vec<Vec<NormPat>>, Vec<TypeHandle>) {
    let mut new_matrix = Vec::new();
    for row in matrix {
        if row.is_empty() {
            continue;
        }
        match &row[0] {
            NormPat::Wild => {
                let mut new_row = vec![NormPat::Wild; arity];
                new_row.extend(row[1..].iter().cloned());
                new_matrix.push(new_row);
            }
            NormPat::Ctor(c, sub) if c == target => {
                let mut new_row = sub.clone();
                if new_row.len() > arity {
                    new_row.truncate(arity);
                }
                while new_row.len() < arity {
                    new_row.push(NormPat::Wild);
                }
                new_row.extend(row[1..].iter().cloned());
                new_matrix.push(new_row);
            }
            _ => {} // different constructor: drop row
        }
    }
    let mut new_col_types = field_types.to_vec();
    new_col_types.extend(col_types[1..].iter().copied());
    (new_matrix, new_col_types)
}

/// Default matrix: keep only wildcard rows, dropping the first column.
fn default_matrix(
    matrix: &[Vec<NormPat>],
    col_types: &[TypeHandle],
) -> (Vec<Vec<NormPat>>, Vec<TypeHandle>) {
    let mut new_matrix = Vec::new();
    for row in matrix {
        if row.is_empty() {
            new_matrix.push(Vec::new());
        } else if matches!(row[0], NormPat::Wild) {
            new_matrix.push(row[1..].to_vec());
        }
    }
    (new_matrix, col_types[1..].to_vec())
}

impl<'a> InferContext<'a> {
    pub fn new(arena: &'a mut TypeArena, sema_result: &'a mut SemaResult) -> Self {
        InferContext {
            arena,
            sema_result,
            type_binding_stack: TypeBindingStack::new(),
            this_binding_stack: ThisBindingStack::new(),
            pending_implicit_this: None,
            env: EnvArena::new(),
            expected_return: None,
            solver: ConstraintSolver::new(),
            flow_ctx: FlowContext::new(),
            witness_table: WitnessTable::new(),
            module_envs: FxHashMap::default(),
            current_module_logical_path: None,
            current_module_env: None,
            current_module_name: String::new(),
            type_trace: Vec::new(),
            ctor_module_envs: FxHashMap::default(),
            instantiation_ctx: None,
            local_mutability: FxHashMap::default(),
            current_trait_name: None,
        }
    }

    /// Construct from existing TypeArena and SemaResult (for incremental recheck).
    /// Preserves global state (witness_table) from previous run.
    /// Unlike `new()`, this clones the witness_table from sema_result to preserve
    /// clean modules' witness entries.
    pub fn from_existing(
        arena: &'a mut TypeArena,
        sema_result: &'a mut SemaResult,
    ) -> Self {
        // Clone witness_table before moving sema_result into the struct to avoid
        // a simultaneous mutable + immutable borrow conflict.
        let witness_table = sema_result.witness_table.clone();
        InferContext {
            arena,
            sema_result,
            type_binding_stack: TypeBindingStack::new(),
            this_binding_stack: ThisBindingStack::new(),
            pending_implicit_this: None,
            env: EnvArena::new(),
            expected_return: None,
            solver: ConstraintSolver::new(),
            flow_ctx: FlowContext::new(),
            witness_table,
            module_envs: FxHashMap::default(),
            current_module_logical_path: None,
            current_module_env: None,
            current_module_name: String::new(),
            type_trace: Vec::new(),
            ctor_module_envs: FxHashMap::default(),
            instantiation_ctx: None,
            local_mutability: FxHashMap::default(),
            current_trait_name: None,
        }
    }

    // ── Type binding stack operations ──

    /// Enters a generic scope: allocates a rigid var for each type parameter and pushes it onto the stack.
    /// Parameters without a declared kind default to Star; declared kinds are used for HKT checks.
    pub fn push_type_bindings(&mut self, type_params: &[(&str, Option<SemKind>)]) {
        self.type_binding_stack.push();
        for &(name, ref kind_opt) in type_params {
            let var = match kind_opt {
                Some(kind) => self.arena.fresh_rigid_var_with_kind(kind.clone()),
                None => self.arena.fresh_rigid_var(),
            };
            self.type_binding_stack.insert_top(name, var);
        }
    }

    /// Leaves a generic scope: pops the top stack frame.
    pub fn pop_type_bindings(&mut self) {
        self.type_binding_stack.pop();
    }

    /// Looks up a type-parameter binding.
    pub fn lookup_type_binding(&self, name: &str) -> Option<TypeHandle> {
        self.type_binding_stack.lookup(name)
    }

    // ── Instantiation mode (monomorphized function-body type resolution) ──

    /// Enters instantiation mode: replaces type-parameter bindings with concrete type_args.
    ///
    /// push_type_bindings (rigid var) should already have been called; this method replaces the rigid vars in the top stack frame
    /// with concrete TypeHandles from type_args (insert_top internally uses HashMap::insert to overwrite the same-name key).
    pub fn enter_instantiation_mode(
        &mut self,
        func_name: Box<str>,
        type_args: Box<[TypeHandle]>,
        type_param_names: &[&str],
        module_name: String,
        in_progress: FxHashMap<u64, u32>,
    ) {
        // Replace the rigid vars at the top of type_binding_stack with concrete type_args.
        for (i, &name) in type_param_names.iter().enumerate() {
            if i < type_args.len() {
                self.type_binding_stack.insert_top(name, type_args[i]);
            }
        }

        // Build type_param_map.
        let mut type_param_map: FxHashMap<String, u16> = FxHashMap::default();
        for (i, &name) in type_param_names.iter().enumerate() {
            type_param_map.insert(name.to_string(), i as u16);
        }

        self.instantiation_ctx = Some(InstantiationCtx {
            func_name,
            type_args,
            type_param_map,
            module_name,
            local_expr_types: FxHashMap::default(),
            local_field_accesses: FxHashMap::default(),
            in_progress,
        });
    }

    /// Leaves instantiation mode: takes out the staged local_expr_types and local_field_accesses.
    ///
    /// The caller is responsible for transferring the returned values into MonomorphInstance.
    pub fn leave_instantiation_mode(
        &mut self,
    ) -> Option<(
        FxHashMap<u64, ExprInfo>,
        FxHashMap<u64, FieldAccessInfo>,
        FxHashMap<u64, u32>,
    )> {
        self.instantiation_ctx.take().map(|ctx| {
            (
                ctx.local_expr_types,
                ctx.local_field_accesses,
                ctx.in_progress,
            )
        })
    }

    // ── Self binding stack operations ──

    /// Enters a type block: binds Self to a concrete type.
    /// `self_ty` should be in `Adt { name, type_args }` form, where type_args reference vars in the TypeBindingStack.
    pub fn push_this_type(&mut self, self_ty: TypeHandle) {
        self.this_binding_stack.push(self_ty);
    }

    /// Enters a trait default method: binds Self to a fresh_rigid_var (to be solved by unification at impl time).
    /// Using a rigid var marks Self as a template parameter and is automatically excluded from diagnostics (only unbound non-rigid TypeVars are reported as errors).
    pub fn push_this_type_var(&mut self) -> TypeHandle {
        let var = self.arena.fresh_rigid_var();
        self.this_binding_stack.push(var);
        var
    }

    /// Leaves a type/trait block: pops the Self binding.
    pub fn pop_this_type(&mut self) {
        self.this_binding_stack.pop();
    }

    /// Current Self type (top of stack).
    pub fn current_this_type(&self) -> Option<TypeHandle> {
        self.this_binding_stack.current()
    }

    // ── Error recording ──

    pub fn add_error(&mut self, message: &str) {
        // line=0/column=0 means no location info (the sema inference stage has not yet associated an AST location).
        self.sema_result.add_error(SemaError::new(message, 0, 0));
    }

    /// Adds an error with location info (for call sites that have AST span context).
    pub fn add_error_at(&mut self, message: &str, line: u32, column: u32) {
        self.sema_result.add_error(SemaError::new(message, line, column));
    }

    /// Adds a warning with location info (does not set has_error; does not stop compilation).
    pub fn add_warning_at(&mut self, message: &str, line: u32, column: u32) {
        self.sema_result.add_warning(SemaError::new(message, line, column));
    }

    // ── self parameter resolution (phase3b) ──

    /// Determines whether a parameter's type_annotation is ThisType (or RefType<ThisType>).
    ///
    /// The parser auto-fills a ThisType annotation for `self`/`&self` parameters of methods inside type/trait blocks;
    /// Sema uses this type node to detect a self parameter, rather than relying on the parameter name.
    fn is_this_param(&self, type_annotation: Option<AstTypeRef>, ast: &AstArena<'_>) -> bool {
        match type_annotation {
            Some(ta) => match &ast.ty(ta).node {
                crate::ast::Ast::TypeNode::ThisType => true,
                crate::ast::Ast::TypeNode::RefType { inner } => {
                    matches!(ast.ty(*inner).node, crate::ast::Ast::TypeNode::ThisType)
                }
                _ => false,
            },
            None => false,
        }
    }

    /// Infers the type of a self parameter.
    ///
    /// **Semantic rules (Rust port, intentionally refined)**:
    /// - `self` may only appear in methods inside type/trait blocks (ThisBindingStack must be non-empty).
    /// - `self` parameters disallow type annotations (the parser auto-fills ThisType or RefType<ThisType>).
    /// - A top-level fun with a self parameter → error.
    /// - A self parameter with an explicit `: Type` annotation → error.
    ///
    /// **Return value**:
    /// - `self` (no annotation, inside a type block) → the scope type.
    /// - `&self` (no annotation, inside a type block) → `Ref<scope type>`.
    /// - Illegal usage → reports an error and returns a fresh_type_var (error recovery).
    pub fn infer_this_param(
        &mut self,
        type_annotation: Option<AstTypeRef>,
        ast: &AstArena<'_>,
    ) -> TypeHandle {
        let self_ty = match self.current_this_type() {
            Some(ty) => ty,
            None => {
                // Get the span from type_annotation if present, otherwise no location info.
                let (line, column) = type_annotation
                    .map(|ta| {
                        let s = ast.ty(ta).span;
                        (s.line, s.column)
                    })
                    .unwrap_or((0, 0));
                self.add_error_at(
                    "this parameter requires enclosing type or trait block",
                    line,
                    column,
                );
                return self.arena.fresh_type_var();
            }
        };

        // Check the type annotation: self parameters disallow explicit annotations.
        // The parser auto-fills ThisType for `self` (no `:`) and RefType<ThisType> for `&self`.
        // A user-written `self: Foo` goes through parse_param's `:` branch, where type_annotation is the user's type.
        match type_annotation {
            None => {
                // No annotation (theoretically unreachable; the parser always fills one for self).
                self_ty
            }
            Some(ta) => {
                let tn = &ast.ty(ta).node;
                let span = ast.ty(ta).span;
                match tn {
                    // `self` (parser auto-fills ThisType) → return the scope type.
                    TypeNode::ThisType => self_ty,
                    // `&self` (parser auto-fills RefType<ThisType>) → return Ref<scope type>.
                    TypeNode::RefType { inner } => {
                        if matches!(ast.ty(*inner).node, TypeNode::ThisType) {

                            self.arena.make_ref(self_ty, false)
                        } else {
                            // `&self: &Foo` with an explicit reference annotation written by the user → error.
                            self.add_error_at(
                                "this parameter does not allow explicit type annotation",
                                span.line,
                                span.column,
                            );
                            self.arena.fresh_type_var()
                        }
                    }
                    // `self: Foo` with an explicit annotation written by the user → error.
                    _ => {
                        self.add_error_at(
                            "this parameter does not allow explicit type annotation",
                            span.line,
                            span.column,
                        );
                        self.arena.fresh_type_var()
                    }
                }
            }
        }
    }

    /// Recursively collects every TypeVar idx in a type, inserting it into subst (with a placeholder value TypeHandle(0); only the key matters).
    fn collect_type_vars(&self, ty: TypeHandle, subst: &mut FxHashMap<u32, TypeHandle>) {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::TypeVar(idx) => {
                subst.entry(idx).or_insert(TypeHandle(0));
            }
            // All composite types (incl. Channel/Async/Lazy/Atomic/Sender/Receiver) delegate
            // their child traversal to `for_each_child`, the single source of truth.
            _ => self
                .arena
                .for_each_child(resolved, |c| self.collect_type_vars(c, subst)),
        }
    }

    /// Instantiates a function type: creates fresh non-rigid copies of every unbound TypeVar in the signature.
    ///
    /// Polymorphic built-in functions (e.g. Ok/i8 registered as generics with rigid vars) must be instantiated on every call,
    /// otherwise type constraints from different calls would clash (after the first call permanently binds a var, subsequent calls cannot unify).
    /// Non-polymorphic functions (signatures with no TypeVar) are returned as-is.
    fn instantiate_fn_type(&mut self, fn_ty: TypeHandle) -> TypeHandle {
    let resolved = self.arena.resolve(fn_ty);
    // Collect all unbound TypeVar idxs in the function signature (collect_type_vars follows resolve;
    // already-bound TypeVars are not collected).
    let mut subst: FxHashMap<u32, TypeHandle> = FxHashMap::default();
    if !matches!(self.arena.get(resolved), Type::Fn(_)) {
        return resolved;
    }
    {
        let (params, return_type) = self.arena.fn_parts(resolved);
        for &p in params.iter() {
            self.collect_type_vars(p, &mut subst);
        }
        self.collect_type_vars(return_type, &mut subst);
    }
    if subst.is_empty() {
        return resolved;
    }
    // Create a fresh non-rigid var for each idx (the collect_type_vars borrow has been released, so a mutable borrow is safe).
    let indices: Vec<u32> = subst.keys().copied().collect();
    for idx in indices {
        let fresh = self.arena.fresh_type_var();
        subst.insert(idx, fresh);
    }
    self.substitute_type(resolved, &subst)
}

    /// Type substitution: replaces the specified TypeVars (by idx) in a type with the bindings from the substitution table.
    ///
    /// Recursively traverses compound types, replacing matching TypeVars. Used to replace a formal parameter's rigid var
    /// with a fresh non-rigid var at the call site, so it can be bound by unification.
    fn substitute_type(&mut self, ty: TypeHandle, subst: &FxHashMap<u32, TypeHandle>) -> TypeHandle {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::TypeVar(idx) => {
                // Hit in the substitution table → return the substituted type; otherwise leave as-is.
                subst.get(&idx).copied().unwrap_or(resolved)
            }
            Type::Fn(_) => {
                let (params, return_type) = self.arena.fn_parts(resolved);
                let params: Vec<TypeHandle> = params.to_vec();
                let new_params: Vec<TypeHandle> = params
                    .iter()
                    .map(|&p| self.substitute_type(p, subst))
                    .collect();
                let new_ret = self.substitute_type(return_type, subst);
                self.arena.make_fn(new_params.into_boxed_slice(), new_ret)
            }
            Type::Record(_) => {
                let fields = self.arena.record_fields(resolved).to_vec();
                let name = self.arena.record_name(resolved).map(|s| s.into());
                let new_fields: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: f.name.clone(),
                        ty: self.substitute_type(f.ty, subst),
                    })
                    .collect();
                self.arena.make_record(new_fields.into_boxed_slice(), name)
            }
            Type::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                let name: Box<str> = name.into();
                let type_args: Vec<TypeHandle> = type_args.to_vec();
                let new_args: Vec<TypeHandle> = type_args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_adt(name, new_args.into_boxed_slice())
            }
            Type::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                let new_inner = self.substitute_type(inner, subst);
                self.arena.make_nullable(new_inner)
            }
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                let name: Box<str> = name.into();
                let args: Vec<TypeHandle> = args.to_vec();
                let new_args: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_generic(name, new_args.into_boxed_slice())
            }
            Type::Array(_) => {
                let (element_type, size) = self.arena.array_parts(resolved);
                let new_elem = self.substitute_type(element_type, subst);
                self.arena.make_array(new_elem, size)
            }
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                let new_v = self.substitute_type(value_type, subst);
                let new_e = self.substitute_type(error_type, subst);
                self.arena.make_throw(new_v, new_e)
            }
            Type::Trait(_) => {
                let (name, type_args) = self.arena.trait_parts(resolved);
                let name: Box<str> = name.into();
                let type_args: Vec<TypeHandle> = type_args.to_vec();
                let new_args: Vec<TypeHandle> = type_args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_trait(name, new_args.into_boxed_slice())
            }
            Type::TraitObject(_) => {
                let (trait_name, method_sigs) = self.arena.trait_object_parts(resolved);
                self.arena.make_trait_object(trait_name.into(), method_sigs.to_vec().into_boxed_slice())
            }
            Type::Ref(_) => {
                let (inner, is_raw) = self.arena.ref_parts(resolved);
                let new_inner = self.substitute_type(inner, subst);
                self.arena.make_ref(new_inner, is_raw)
            }
            // Single-element builtin generics — kept in lockstep with `for_each_child` so that
            // TypeVars nested inside them are substituted (otherwise instantiate_fn_type would
            // collect a var via collect_type_vars but fail to replace it here).
            Type::Channel(_) => {
                let elem = self.arena.channel_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_channel(new_elem)
            }
            Type::Async(_) => {
                let value = self.arena.async_value(resolved);
                let new_value = self.substitute_type(value, subst);
                self.arena.make_async(new_value)
            }
            Type::Lazy(_) => {
                let value = self.arena.lazy_value(resolved);
                let new_value = self.substitute_type(value, subst);
                self.arena.make_lazy(new_value)
            }
            Type::Atomic(_) => {
                let elem = self.arena.atomic_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_atomic(new_elem)
            }
            Type::Sender(_) => {
                let elem = self.arena.sender_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_sender(new_elem)
            }
            Type::Receiver(_) => {
                let elem = self.arena.receiver_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_receiver(new_elem)
            }
            // Scalars, Never, Unknown, Void, Null, TraitObject, ModuleRef, Timer have no sub-nodes → return as-is.
            _ => resolved,
        }
    }

    // ── Literal promotion ──
    // v2 convergence: literal_promotion has been replaced by peer_type_binary,
    // literal promotion rules are inlined into peer_type_binary, eliminating the dual-track scheme.

    // ── GADT inference (phase3e) ──

    /// Performs GADT type refinement for a constructor pattern.
    ///
    /// **Semantics** (ported from `src/sema/gadt_check.zig` refineConstructorPattern):
    /// 1. Look up the constructor definition (CtorDefInfo) from sema_result.
    /// 2. Unify the constructor's return type with expected_ty to refine type variables.
    /// 3. Recursively infer sub-patterns against the constructor's field types.
    ///
    /// **Return value**: `true` means this function handled it (the constructor is registered);
    /// `false` means the constructor is unregistered and falls back to regular pattern inference.
    ///
    /// **Throw error branch**: when expected_ty is a Throw type and the constructor is an error_newtype
    /// ADT constructor, the `is_throw_error_branch` flag is true, and the constructor's return type and
    /// sub-patterns are unified and bound to error_type. This flag flows through both the return-type
    /// resolution and sub-pattern binding steps, with no separate early-exit branch; it shares the same
    /// control flow as the regular GADT path.
    pub fn refine_constructor_pattern(
        &mut self,
        ctor_name: &str,
        sub_patterns: &[PatternRef],
        expected_ty: TypeHandle,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> bool {
        // Use field_type_reprs (self-contained TypeRepr) instead of field_type_nodes (AST references),
        // to avoid AST arena mismatch when used cross-module, which would make TypeRef indices point at the wrong type nodes.
        // return_type_node still uses AstTypeRef (GADT cases are rare and usually intra-module).
        type CtorInfoSnapshot = (Box<str>, bool, Option<AstTypeRef>, Box<[TypeRepr]>);
        let resolved_expected = self.arena.resolve(expected_ty);

        // Clone the constructor info first, to avoid the &CtorDefInfo borrow blocking later &mut self calls.
        let ctor_info: Option<CtorInfoSnapshot> =
            self.find_ctor_def(ctor_name, expected_ty).map(|c| {
                (
                    c.type_name.clone(),
                    c.is_newtype,
                    c.return_type_node,
                    c.field_type_reprs.clone(),
                )
            });

        // Throw<T, E> builtin type variant matching:
        // Throw is a builtin sum type; its variants Ok(T) / Error(E) are not registered as CtorDefInfo.
        // - Ok (or any non-error constructor) → value variant → sub-patterns bind to value_type.
        // - Error / Err (by name) or any registered error ADT constructor → error variant → sub-patterns bind to error_type.
        //   (When the constructor name collides with the Throw error-variant name "Error"/"Err",
        //    regardless of what error_type is, the pattern matches the Throw error variant
        //    and sub-patterns bind to error_type rather than the constructor's field types.)
        if let Type::Throw(_) = self.arena.get(resolved_expected) {
            let (value_type, error_type) = self.arena.throw_parts(resolved_expected);
            let is_error_variant = ctor_name == crate::ir::Compute::CTOR_ERR
                || ctor_name == crate::ir::Compute::CTOR_ERR_ALT
                || ctor_info.is_some();
            let branch_ty = if is_error_variant { error_type } else { value_type };
            for &sub_pat in sub_patterns.iter() {
                self.infer_pattern(sub_pat, ast, branch_ty, env);
            }
            return true;
        }

        let (type_name, is_newtype, return_type_node, field_type_reprs) = match ctor_info {
            Some(info) => info,
            None => return false,
        };
        let _ = is_newtype;

        // Resolve the constructor's return type (GADT → return_type_node; regular ADT → the Adt for type_name).
        // Goes through InferContext's full type resolution (type_from_ast), unifying handling of all TypeNode variants,
        // to eliminate type loss caused by the simplified resolve_type_node_to_handle falling back to fresh_type_var for complex types.
        let ctor_return_ty = if let Some(rtn) = return_type_node {
            self.type_from_ast(rtn, ast)
        } else {
            self.arena.make_adt(type_name, Box::new([]))
        };

        // Unify the constructor's return type with the expected type to perform GADT type refinement.
        // On failure, register a constraint for the fixpoint iteration to retry.
        let ctor_compatible = self.arena.unify(ctor_return_ty, expected_ty).is_ok();
        if !ctor_compatible {
            self.unify_or_constrain(ctor_return_ty, expected_ty);
        }

        // Recursively infer sub-patterns against the constructor's field types and bind variables.
        // When the constructor's return type is incompatible with the expected type (e.g. an Error ADT used to unwrap a Throw's error_type),
        // sub-patterns bind to expected_ty rather than the constructor's field types, ensuring pattern variables get the correct runtime type.
        for (i, &sub_pat) in sub_patterns.iter().enumerate() {
            let sub_ty = if !ctor_compatible {
                expected_ty
            } else if i < field_type_reprs.len() {
                self.type_repr_to_handle(&field_type_reprs[i])
            } else {
                self.arena.fresh_type_var()
            };
            self.infer_pattern(sub_pat, ast, sub_ty, env);
        }

        true
    }

    /// Looks up a constructor definition by name from sema_result.
    /// When multiple types define the same constructor name, uses `expected_ty`
    /// to disambiguate (type-oriented pattern constructor resolution).
    fn find_ctor_def(&self, ctor_name: &str, expected_ty: TypeHandle) -> Option<&CtorDefInfo> {
        let candidates = self.sema_result.get_ctor_defs(ctor_name);
        if candidates.len() <= 1 {
            return candidates.into_iter().next();
        }
        // Type-oriented disambiguation: select by the Adt type_name of expected_ty
        let exp_resolved = self.arena.resolve(expected_ty);
        if let Type::Adt(_) = self.arena.get(exp_resolved) {
            let (exp_type_name, _) = self.arena.adt_parts(exp_resolved);
            let matches: Vec<_> = candidates.iter()
                .filter(|c| c.type_name.as_ref() == exp_type_name)
                .collect();
            if matches.len() == 1 {
                return Some(matches[0]);
            }
        }
        // Fall back to the first candidate (preserves backward compatibility)
        candidates.into_iter().next()
    }

    // ── Usefulness algorithm (Maranget) for match exhaustiveness checking ──

    /// Convert a `TypeRepr` to a `TypeHandle`, substituting type parameters with
    /// the actual type arguments from the scrutinee type (for generic ADTs).
    fn ctor_field_type(
        &mut self,
        repr: &TypeRepr,
        params: &[Box<str>],
        args: &[TypeHandle],
    ) -> TypeHandle {
        if let TypeRepr::Named(name) = repr {
            if let Some(idx) = params.iter().position(|p| p.as_ref() == name.as_ref()) {
                if idx < args.len() {
                    return args[idx];
                }
            }
        }
        self.type_repr_to_handle(repr)
    }

    /// Get the arity and field types of a constructor, given the column type
    /// (used to disambiguate same-named constructors across types and to substitute
    /// generic type parameters).
    fn ctor_arity_and_fields(
        &mut self,
        col_type: TypeHandle,
        ctor: &PatCtor,
    ) -> (usize, Vec<TypeHandle>) {
        // Throw<T, E> builtin: Ok has arity 1 (field = value_type),
        // Error (or any registered error-variant constructor) has arity 1
        // (field = error_type). These constructors are NOT registered as
        // CtorDefInfo (see refine_constructor_pattern), so they need explicit
        // handling to avoid arity-0 fallback that loses sub-pattern information.
        if let PatCtor::Adt(name) = ctor {
            let resolved = self.arena.resolve(col_type);
            if let Type::Throw(_) = self.arena.get(resolved) {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                let field_ty = if name.as_ref() == "Ok" { value_type } else { error_type };
                return (1, vec![field_ty]);
            }
        }

        // Phase 1: collect data via immutable borrows (all cloned to release borrows).
        let collected: Option<(Box<[TypeRepr]>, Box<[Box<str>]>, Vec<TypeHandle>)> = match ctor {
            PatCtor::Adt(name) => {
                let resolved = self.arena.resolve(col_type);
                match self.arena.get(resolved) {
                    Type::Adt(_) => {
                        let (type_name, type_args) = self.arena.adt_parts(resolved);
                        self.sema_result.get_type_def(type_name).and_then(|td| {
                            td.constructors.iter()
                                .find(|c| c.name.as_ref() == name.as_ref())
                                .map(|c| {
                                    (c.field_type_reprs.clone(), td.type_params.clone(), type_args.to_vec())
                                })
                        })
                    }
                    _ => self.sema_result.get_ctor_def(name).map(|c| {
                        (c.field_type_reprs.clone(), Box::new([]) as Box<[Box<str>]>, Vec::new())
                    }),
                }
            }
            _ => None,
        };

        // Phase 2: convert field type reprs to handles (mutable borrow).
        match collected {
            Some((reprs, params, args)) => {
                let field_types: Vec<TypeHandle> = reprs.iter()
                    .map(|r| self.ctor_field_type(r, &params, &args))
                    .collect();
                (field_types.len(), field_types)
            }
            None => (0, Vec::new()),
        }
    }

    /// Return all constructors of a type if it has a finite (complete) signature.
    /// Returns `None` for types with infinite value spaces (int, float, char, str,
    /// nullable, etc.).
    fn type_all_ctors(&self, col_type: TypeHandle) -> Option<Vec<PatCtor>> {
        let resolved = self.arena.resolve(col_type);
        match self.arena.get(resolved) {
            Type::Adt(_) => {
                let (type_name, _) = self.arena.adt_parts(resolved);
                let type_def = self.sema_result.get_type_def(type_name)?;
                if type_def.constructors.is_empty() {
                    return None;
                }
                Some(type_def.constructors.iter()
                    .map(|c| PatCtor::Adt(c.name.clone()))
                    .collect())
            }
            Type::Bool => Some(vec![PatCtor::Bool(true), PatCtor::Bool(false)]),
            _ => None,
        }
    }

    /// Core usefulness check: is `query` useful w.r.t. `matrix`?
    /// Implements Maranget's algorithm U(M, q).
    fn is_useful(
        &mut self,
        matrix: &[Vec<NormPat>],
        col_types: &[TypeHandle],
        query: &[NormPat],
        depth: usize,
    ) -> bool {
        // Base case: no columns → useful iff the matrix has no rows.
        if query.is_empty() {
            return matrix.is_empty();
        }
        if col_types.is_empty() {
            return false; // defensive: no type info
        }
        // Safety valve: prevent exponential blowup on pathological nesting.
        if depth > 48 {
            return false;
        }

        let col_type = col_types[0];
        match &query[0] {
            NormPat::Wild => {
                // Collect distinct constructors appearing in the first column.
                let mut seen: Vec<PatCtor> = Vec::new();
                for row in matrix {
                    if !row.is_empty() {
                        if let NormPat::Ctor(c, _) = &row[0] {
                            if !seen.contains(c) {
                                seen.push(c.clone());
                            }
                        }
                    }
                }

                if seen.is_empty() {
                    // No constructors in column → check default matrix.
                    let (dm, dt) = default_matrix(matrix, col_types);
                    return self.is_useful(&dm, &dt, &query[1..], depth + 1);
                }

                // Try each seen constructor: if any makes the query useful, done.
                for ctor in &seen {
                    let (arity, field_types) = self.ctor_arity_and_fields(col_type, ctor);
                    let (sm, st) = specialize_matrix(matrix, col_types, ctor, arity, &field_types);
                    let mut sq = vec![NormPat::Wild; arity];
                    sq.extend(query[1..].iter().cloned());
                    if self.is_useful(&sm, &st, &sq, depth + 1) {
                        return true;
                    }
                }

                // All seen constructors failed; check if the signature is complete.
                let is_complete = self.type_all_ctors(col_type)
                    .map(|all| all.iter().all(|c| seen.contains(c)))
                    .unwrap_or(false);
                if is_complete {
                    return false;
                }
                // Incomplete signature → wildcard is useful via the default matrix.
                let (dm, dt) = default_matrix(matrix, col_types);
                self.is_useful(&dm, &dt, &query[1..], depth + 1)
            }
            NormPat::Ctor(target, sub) => {
                let (arity, field_types) = self.ctor_arity_and_fields(col_type, target);
                let (sm, st) = specialize_matrix(matrix, col_types, target, arity, &field_types);
                let mut sq = sub.clone();
                if sq.len() > arity {
                    sq.truncate(arity);
                }
                while sq.len() < arity {
                    sq.push(NormPat::Wild);
                }
                sq.extend(query[1..].iter().cloned());
                self.is_useful(&sm, &st, &sq, depth + 1)
            }
        }
    }

    /// Generate a human-readable witness for a non-exhaustive match (what's missing).
    fn witness(
        &mut self,
        matrix: &[Vec<NormPat>],
        col_types: &[TypeHandle],
        depth: usize,
    ) -> String {
        if col_types.is_empty() || depth > 8 {
            return String::new();
        }
        let col_type = col_types[0];
        let resolved = self.arena.resolve(col_type);
        let ty = self.arena.get(resolved);

        let seen: Vec<PatCtor> = matrix.iter()
            .filter_map(|r| {
                if !r.is_empty() {
                    if let NormPat::Ctor(c, _) = &r[0] { Some(c.clone()) } else { None }
                } else { None }
            })
            .collect();
        let has_wild = matrix.iter().any(|r| !r.is_empty() && matches!(r[0], NormPat::Wild));

        match ty {
            Type::Adt(_) => {
                let (type_name, _) = self.arena.adt_parts(resolved);
                // Collect constructor names (cloned to release the borrow).
                let ctor_names: Vec<Box<str>> = self.sema_result.get_type_def(type_name)
                    .map(|td| td.constructors.iter().map(|c| c.name.clone()).collect())
                    .unwrap_or_default();

                // Check for missing top-level constructors.
                let missing: Vec<&str> = ctor_names.iter()
                    .map(|n| n.as_ref())
                    .filter(|n| !seen.iter().any(|c| matches!(c, PatCtor::Adt(a) if a.as_ref() == *n)))
                    .collect();
                if !missing.is_empty() {
                    return format!(": missing {}", missing.join(", "));
                }

                // All top-level constructors present but nested issue; recurse.
                for name in &ctor_names {
                    let pc = PatCtor::Adt(name.clone());
                    if seen.contains(&pc) {
                        let (arity, ft) = self.ctor_arity_and_fields(col_type, &pc);
                        let (sm, st) = specialize_matrix(matrix, col_types, &pc, arity, &ft);
                        let q = vec![NormPat::Wild; arity];
                        if self.is_useful(&sm, &st, &q, 0) {
                            return format!(": `{}` not exhaustive{}", name,
                                self.witness(&sm, &st, depth + 1));
                        }
                    }
                }
                String::new()
            }
            Type::Bool => {
                if !seen.iter().any(|c| matches!(c, PatCtor::Bool(true))) {
                    ": missing `true`".to_string()
                } else if !seen.iter().any(|c| matches!(c, PatCtor::Bool(false))) {
                    ": missing `false`".to_string()
                } else {
                    String::new()
                }
            }
            _ => {
                if !has_wild {
                    ": missing catch-all `_`".to_string()
                } else {
                    String::new()
                }
            }
        }
    }

    /// Check match exhaustiveness using the usefulness algorithm (Maranget).
    ///
    /// Reports a non-exhaustive error if any value of the scrutinee type is not
    /// covered by the (unguarded) match arms. Also warns on unreachable arms.
    fn check_match_exhaustive(
        &mut self,
        ast: &AstArena<'_>,
        scrutinee: crate::ast::Ast::ExprId,
        scrutinee_ty: TypeHandle,
        arms: &[crate::ast::Ast::MatchArm],
    ) {
        let col_types = vec![scrutinee_ty];

        // An arm is "guarded" if it has an arm-level guard OR an inline Pattern::Guard.
        // Guarded arms do not guarantee coverage (the guard may fail) and do not make
        // subsequent arms unreachable.
        let is_arm_guarded = |arm: &crate::ast::Ast::MatchArm| -> bool {
            arm.guard.is_some() || matches!(ast.pattern(arm.pattern).node, Pattern::Guard { .. })
        };

        // ── Exhaustiveness check ──
        // Build matrix from unguarded arms only (guarded arms don't guarantee coverage).
        let exhaust_matrix: Vec<Vec<NormPat>> = arms.iter()
            .filter(|arm| !is_arm_guarded(arm))
            .flat_map(|arm| {
                let pat = unwrap_guard_pat(ast, arm.pattern);
                normalize_pattern(ast, pat).into_iter().map(|p| vec![p])
            })
            .collect();

        // Only report non-exhaustive for types with finite constructor sets (ADT, Bool).
        // For infinite types (int, str, char, ...), preserve existing lenient behavior.
        let resolved = self.arena.resolve(scrutinee_ty);
        let is_finite = matches!(self.arena.get(resolved), Type::Adt(_) | Type::Bool);

        if is_finite && self.is_useful(&exhaust_matrix, &col_types, &[NormPat::Wild], 0) {
            let span = ast.expr(scrutinee).span;
            let witness = self.witness(&exhaust_matrix, &col_types, 0);
            self.add_error_at(
                &format!("non-exhaustive match{}", witness),
                span.line,
                span.column,
            );
        }

        // ── Unreachable arm detection ──
        // Arm i is unreachable iff its pattern is not useful given the matrix of
        // previous *unguarded* arms (guarded arms don't block subsequent arms).
        let mut prev_matrix: Vec<Vec<NormPat>> = Vec::new();
        for arm in arms.iter() {
            let pat = unwrap_guard_pat(ast, arm.pattern);
            let alternatives = normalize_pattern(ast, pat);

            let any_useful = alternatives.iter()
                .any(|alt| self.is_useful(&prev_matrix, &col_types, &[alt.clone()], 0));

            if !any_useful && !alternatives.is_empty() {
                let span = ast.pattern(arm.pattern).span;
                self.add_warning_at("unreachable match arm", span.line, span.column);
            }

            // Add this arm's patterns to the previous matrix (only if unguarded).
            if !is_arm_guarded(arm) {
                for alt in alternatives {
                    prev_matrix.push(vec![alt]);
                }
            }
        }
    }

    /// Check if `Type.Ctor` is a qualified constructor access.
    /// Returns `Some((type_name, field_type_reprs))` when `type_name` is a
    /// registered type and `ctor_name` is one of its constructors.
    fn check_qualified_ctor(
        &self,
        type_name: &str,
        ctor_name: &str,
    ) -> Option<(Box<str>, Box<[TypeRepr]>)> {
        let &type_idx = self.sema_result.type_def_index.get(type_name)?;
        let type_def = &self.sema_result.type_defs[&type_idx];
        let ctor = type_def
            .constructors
            .iter()
            .find(|c| c.name.as_ref() == ctor_name)?;
        Some((ctor.type_name.clone(), ctor.field_type_reprs.clone()))
    }
}


// =========================================================================
// phase5: InferContext extensions — type resolution, freshening, structural equality, throw checks
//
// Adds InferContext methods ported from `src/sema/type_check.zig` and `throw_check.zig`.
// =========================================================================

/// Registry of builtin cast functions: (function name, whether it is the try variant).
/// Adding a cast variant only requires appending a row; no new name-specific branch is needed.
const CAST_BUILTINS: &[(&str, bool)] = &[
    ("__cast_to", false),
    ("__cast_try_to", true),
];

impl<'a> InferContext<'a> {
    // ── Type resolution (typeFromAst) ──

    /// Resolves an AST TypeNode to a TypeHandle (convenience overload, with no type-parameter map).
    pub fn type_from_ast(&mut self, type_ref: AstTypeRef, ast: &AstArena<'_>) -> TypeHandle {
        let empty = FxHashMap::default();
        self.type_from_ast_with_params(type_ref, ast, &empty)
    }

    /// Resolves a name to a TypeHandle (alias unfolding + cycle detection).
    ///
    /// This is the core of Named type resolution: type_param_map → type_binding → builtin scalars →
    /// trait → recursive Alias unfolding in type_defs → user-defined Adt.
    /// `visiting` is used for alias cycle detection (A→B→A); on a cycle it returns Adt(name) to terminate.
    fn resolve_name_to_type(
        &mut self,
        name: &str,
        type_param_map: &FxHashMap<String, TypeHandle>,
        visiting: &mut FxHashSet<String>,
    ) -> TypeHandle {
        // 1. Type-parameter map.
        if let Some(ty) = type_param_map.get(name) {
            return *ty;
        }
        // 2. Type binding stack (generic scope).
        if let Some(ty) = self.lookup_type_binding(name) {
            return ty;
        }
        // 3. Builtin scalars + str/null/void: derived from BUILTIN_TABLE.
        if let Some(ct) = name_to_concrete(name) {
            return self.arena.make(ct);
        }
        // 4. trait definition → Trait type.
        if self.sema_result.get_trait_def(name).is_some() {
            return self.arena.make_trait(name.into(), Box::new([]));
        }
        // Alias cycle detection.
        if visiting.contains(name) {
            return self.arena.make_adt(name.into(), Box::new([]));
        }
        visiting.insert(name.to_string());
        // 5. Alias unfolding: type Name = T → resolve T.
        // Prefer the already-resolved target_type (TypeHandle), which covers non-named targets like functions/Records/Arrays;
        // fall back to target_type_name (a named target, e.g. type A = B).
        let (alias_target_ty, alias_target_name): (Option<TypeHandle>, Option<String>) = self
            .sema_result
            .get_type_def(name)
            .filter(|td| td.kind == TypeDefKind::Alias)
            .map(|td| (td.target_type, td.target_type_name.as_deref().map(String::from)))
            .unwrap_or((None, None));
        if let Some(inner_ty) = alias_target_ty {
            visiting.remove(name);
            return inner_ty;
        }
        if let Some(target_name) = alias_target_name {
            let result = self.resolve_name_to_type(&target_name, type_param_map, visiting);
            visiting.remove(name);
            return result;
        }
        visiting.remove(name);
        // 6. User-defined type → Adt.
        self.arena.make_adt(name.into(), Box::new([]))
    }

    /// Resolves an AST TypeNode to a TypeHandle (full version, with a type-parameter map).
    ///
    /// Handles all TypeNode variants: Named, ThisType, Generic, Nullable, RefType, RawPtr,
    /// Function, Record, Array, KindAnnotated. Builtin scalars go through from_scalar_name;
    /// the generic Throw is special-cased into a Throw type; other builtin generics become Generic;
    /// user-defined ADTs become Adt; traits become Trait.
    pub fn type_from_ast_with_params(
        &mut self,
        type_ref: AstTypeRef,
        ast: &AstArena<'_>,
        type_param_map: &FxHashMap<String, TypeHandle>,
    ) -> TypeHandle {
        let tn = &ast.ty(type_ref).node;
        match tn {
            TypeNode::Named { name } => {
                // Delegate to resolve_name_to_type: builtin scalar → trait → alias unfolding → Adt.
                let mut visiting = FxHashSet::default();
                self.resolve_name_to_type(name, type_param_map, &mut visiting)
            }
            TypeNode::ThisType => match self.current_this_type() {
                Some(ty) => ty,
                None => {
                    let span = ast.ty(type_ref).span;
                    self.add_error_at("This type can only be used within type or trait methods", span.line, span.column);
                    self.arena.make(Type::Void)
                }
            },
            TypeNode::Generic { name, args } => {
                // Recursively resolve type arguments.
                let new_args: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.type_from_ast_with_params(a, ast, type_param_map))
                    .collect();
                let args_box: Box<[TypeHandle]> = new_args.into_boxed_slice();

                // Higher-kinded type (HKT) in the type-parameter map: F<T> where F is a type parameter.
                if let Some(&param_handle) = type_param_map.get(*name) {
                    // Kind check: verify that F's kind matches the number and kinds of the arguments.
                    let constructor_kind = self.arena.kind_of(param_handle);
                    // If constructor_kind is not Star (i.e. F is a type constructor),
                    // or args is non-empty (i.e. an F<T> application), perform the kind check.
                    if !matches!(constructor_kind, SemKind::Star) || !args_box.is_empty() {
                        let arg_kinds: Vec<SemKind> = args_box
                            .iter()
                            .map(|&a| self.arena.kind_of(a))
                            .collect();
                        if let Err(kind_err) = self.arena.check_kind_application(&constructor_kind, &arg_kinds) {
                            // Error recovery: record the error but keep constructing the type.
                            let span = ast.ty(type_ref).span;
                            self.add_error_at(&kind_err, span.line, span.column);
                        }
                    }
                    return self.arena.make_generic((*name).into(), args_box);
                }
                // Builtin generic types (Throw/Atomic/Async/Channel, etc.) construct dedicated Type variants
                // and do not go through the Type::Generic path — this avoids later matching builtin generics by string name.
                if is_builtin_generic_type(name) {
                    return self.make_builtin_generic((*name).into(), args_box);
                }
                // trait definition → Trait type.
                if self.sema_result.get_trait_def(name).is_some() {
                    return self.arena.make_trait((*name).into(), args_box);
                }
                // User-defined generic ADT.
                let has_type_params = self
                    .sema_result
                    .get_type_def(name)
                    .map(|d| !d.type_params.is_empty())
                    .unwrap_or(false);
                if has_type_params {
                    return self.arena.make_adt((*name).into(), args_box);
                }
                // Fallback: construct a Generic (may be undefined or a forward reference; reported on later use).
                self.arena.make_generic((*name).into(), args_box)
            }
            TypeNode::Nullable { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_nullable(inner_ty)
            }
            TypeNode::RefType { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_ref(inner_ty, false)
            }
            TypeNode::RawPtr { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_ref(inner_ty, true)
            }
            TypeNode::Function { params, return_type } => {
                let new_params: Vec<TypeHandle> = params
                    .iter()
                    .map(|&p| self.type_from_ast_with_params(p, ast, type_param_map))
                    .collect();
                let new_ret = self.type_from_ast_with_params(*return_type, ast, type_param_map);
                self.arena.make_fn(new_params.into_boxed_slice(), new_ret)
            }
            TypeNode::Record { fields } => {
                if fields.is_empty() {
                    return self.arena.make(Type::Void);
                }
                let new_fields: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: Some(f.name.into()),
                        ty: self.type_from_ast_with_params(f.ty, ast, type_param_map),
                    })
                    .collect();
                self.arena.make_record(new_fields.into_boxed_slice(), None)
            }
            TypeNode::Array { element_type, size } => {
                let elem_ty = self.type_from_ast_with_params(*element_type, ast, type_param_map);
                self.arena.make_array(elem_ty, *size)
            }
            TypeNode::KindAnnotated { inner, .. } => {
                self.type_from_ast_with_params(*inner, ast, type_param_map)
            }
        }
    }

    // ── freshen_type / apply_type_subst ──

    /// Freshens a type: replaces unbound TypeVars in the type with new TypeVars.
    /// Used when looking up a generic function type from the environment, to keep each call independent (replaces the old HM instantiate).
    pub fn freshen_type(&mut self, ty: TypeHandle) -> TypeHandle {
        // 1. Collect all unbound TypeVar idxs.
        let mut free_vars: Vec<u32> = Vec::new();
        self.collect_free_vars(ty, &mut free_vars);
        if free_vars.is_empty() {
            return ty;
        }
        // 2. Allocate a fresh var for each free var and build the substitution table.
        let mut subst: FxHashMap<u32, TypeHandle> = FxHashMap::default();
        for idx in free_vars.iter() {
            let fresh = self.arena.fresh_type_var();
            subst.insert(*idx, fresh);
        }
        // 3. Apply the substitution.
        self.apply_type_subst(ty, &subst)
    }

    /// Recursively collects the idxs of unbound TypeVars in a type (deduplicated).
    ///
    /// Note: Fn types do not collect their internal TypeVars. A function type is a "type scheme";
    /// instantiation of its free variables is handled uniformly by `instantiate_fn_type` at the call site.
    /// If freshen_type also instantiated Fn-internal variables, it would duplicate the work of instantiate_fn_type,
    /// leaving the first set of fresh copies orphaned (referenced by no unify) and ultimately reported as unresolved TypeVars.
    fn collect_free_vars(&self, ty: TypeHandle, free_vars: &mut Vec<u32>) {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::TypeVar(idx) => {
                // A rigid var represents a generic parameter declaration (e.g. T in type ArrayIter<T>);
                // it is fixed within the current scope and must not be freshened/instantiated.
                // Collect only unbound non-rigid TypeVars (local inference variables).
                if !self.arena.type_var(idx).is_rigid && !free_vars.contains(&idx) {
                    free_vars.push(idx);
                }
            }
            // Skip Fn types: instantiation is handled by instantiate_fn_type at the call site.
            Type::Fn(_) => {}
            // All other composite types delegate child traversal to `for_each_child`.
            _ => self
                .arena
                .for_each_child(resolved, |c| self.collect_free_vars(c, free_vars)),
        }
    }

    /// Substitutes TypeVars in a type using the substitution table (by idx). Side-effect free; returns a new type.
    /// Delegates to the existing `substitute_type` implementation.
    pub fn apply_type_subst(
        &mut self,
        ty: TypeHandle,
        subst: &FxHashMap<u32, TypeHandle>,
    ) -> TypeHandle {
        self.substitute_type(ty, subst)
    }

    // ── types_structurally_equal ──

    /// Side-effect-free structural type equality check: does not modify any TypeVar and triggers no unify side effects.
    /// Used when matching trait method signatures to compare parameter and return types.
    ///
    /// Delegates to the free function `types_equal` to avoid maintaining two copies of the structural-equality logic.
    pub fn types_structurally_equal(&self, a: TypeHandle, b: TypeHandle) -> bool {
        types_equal(self.arena, a, b)
    }

    // ── throw_check methods ──

    /// Unifies a function's declared return type with the type inferred from its body.
    /// Applies special relaxation for nullable/throw return types: when the body returns void (early exit / throw),
    /// it is not treated as a mismatch; otherwise it tries a widening unification and falls back to a strict unify on failure.
    pub fn unify_return_type(
        &mut self,
        declared: TypeHandle,
        inferred: TypeHandle,
    ) -> Result<(), UnifyError> {
        let r_declared = self.arena.resolve(declared);
        let r_inferred = self.arena.resolve(inferred);

        let declared_ty = self.arena.get(r_declared);
        let inferred_ty = self.arena.get(r_inferred);

        // async function: the declared return type should be Async<X> and the body infers Async<Y>;
        // recursively unify the inner types X and Y.
        if let (Type::Async(_), Type::Async(_)) = (declared_ty, inferred_ty) {
            let da = self.arena.async_value(r_declared);
            let ia = self.arena.async_value(r_inferred);
            return self.unify_return_type(da, ia);
        }
        // async function body directly returns the inner value (not Async-wrapped):
        // declared Async<X>, body infers Y → recursively unify X with Y.
        if let Type::Async(_) = declared_ty {
            let da = self.arena.async_value(r_declared);
            return self.unify_return_type(da, r_inferred);
        }

        match declared_ty {
            Type::Nullable(_) => match inferred_ty {
                Type::Nullable(_) => self.arena.unify(declared, inferred),
                Type::Void => Ok(()), // The body produced no value; compatible with nullable.
                _ => {
                    let inner_ty = self.arena.nullable_inner(r_declared);
                    match self.try_widen_unify(inner_ty, r_inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(inner_ty, r_inferred),
                    }
                }
            },
            Type::Throw(_) => match inferred_ty {
                Type::Throw(_) => {
                    match self.try_widen_unify(declared, inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(declared, inferred),
                    }
                }
                Type::Void => Ok(()), // The body produced no value; compatible with throw.
                _ => {
                    let (vt, _) = self.arena.throw_parts(r_declared);
                    match self.try_widen_unify(vt, r_inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(vt, r_inferred),
                    }
                }
            },
            _ => {
                match self.try_widen_unify(r_declared, r_inferred) {
                    Ok(_) => Ok(()),
                    Err(_) => self.arena.unify(declared, inferred),
                }
            }
        }
    }

    /// Immediately unifies two types; on failure, registers an Equality constraint for the fixpoint iteration to retry.
    ///
    /// Replaces the `let _ = self.arena.unify(t1, t2)` pattern:
    /// - unify succeeds → bind immediately (preserves inference-order advantages).
    /// - unify fails → register an Equality constraint with the solver, retried by the fixpoint iteration
    ///   (other constraints may bind the relevant TypeVars first, making a later unify succeed).
    #[inline]
    pub fn unify_or_constrain(&mut self, t1: TypeHandle, t2: TypeHandle) {
        // Instantiation mode: skip HM constraint solving (types were already checked in the sema HM stage).
        if self.instantiation_ctx.is_some() {
            return;
        }
        // Lazy<T> subsumption: Lazy<T> auto-unwraps to T when the context expects T.
        // This makes `lazy(1i32) + 3i32` type-check (Lazy<i32> unwraps to i32).
        {
            let InferContext { arena, .. } = self;
            let ra = arena.resolve(t1);
            let rb = arena.resolve(t2);
            let a_is_lazy = matches!(arena.get(ra), Type::Lazy(_));
            let b_is_lazy = matches!(arena.get(rb), Type::Lazy(_));
            if a_is_lazy && !b_is_lazy {
                let inner = arena.lazy_value(ra);
                self.unify_or_constrain(inner, t2);
                return;
            }
            if b_is_lazy && !a_is_lazy {
                let inner = arena.lazy_value(rb);
                self.unify_or_constrain(t1, inner);
                return;
            }
        }
        // Record candidate before unify so finalize_solution can detect ambiguity when
        // the same TypeVar is required to bind to multiple distinct concrete types
        // (Bug #83: `pair(1i32, 2i64)` silently bound T to i32). Without this, only
        // failed unifies recorded candidates, so a TypeVar bound by the first
        // successful unify + a conflicting failed unify would show a single candidate.
        let InferContext { arena, solver, .. } = self;
        solver.record_candidate(arena, t1, t2);
        if arena.unify(t1, t2).is_err() {
            solver.add_equality(t1, t2);
        }
    }

    /// Attempts a widening unification of two types and returns the unified type.
    /// First tries a strict unify; on failure, if both are numeric, picks one per the widening rules;
    /// otherwise handles structural compatibility for nullable/throw vs. ordinary types, void, etc.
    pub fn try_widen_unify(
        &mut self,
        t1: TypeHandle,
        t2: TypeHandle,
    ) -> Result<TypeHandle, UnifyError> {
        let r1 = self.arena.resolve(t1);
        let r2 = self.arena.resolve(t2);

        // never unifies with any type as the other type.
        if matches!(self.arena.get(r1), Type::Never) {
            return Ok(r2);
        }
        if matches!(self.arena.get(r2), Type::Never) {
            return Ok(r1);
        }

        // First try a strict unify.
        if self.arena.unify(r1, r2).is_ok() { return Ok(r1) }

        let c1 = self.arena.get(r1);
        let c2 = self.arena.get(r2);

        // Async unfolding: Async<X> with Y (non-Async) → recursively unify X with Y.
        // Scenario: inside an async function body, Ok(void) returns Throw<void, '_E>,
        // expected is Async<Throw<void, IOError>>; the Async layer must be unfolded to solve '_E.
        if let Type::Async(_) = c1 {
            let inner = self.arena.async_value(r1);
            return self.try_widen_unify(inner, r2);
        }
        if let Type::Async(_) = c2 {
            let inner = self.arena.async_value(r2);
            return self.try_widen_unify(r1, inner);
        }

        // Bug #60 (Plan A fully strict): numeric widening is removed — different numeric types
        // are no longer implicitly promoted; an explicit cast is required. Strict unify has already
        // been attempted above and failed; numeric type pairs fall through to the match's _ branch and return TypeMismatch.

        match (c1, c2) {
            (Type::Nullable(_), _) => match c2 {
                Type::Nullable(_) => {
                    let i1 = self.arena.resolve(self.arena.nullable_inner(r1));
                    let i2 = self.arena.resolve(self.arena.nullable_inner(r2));
                    match self.arena.unify(i1, i2) {
                        Ok(_) => Ok(r1),
                        Err(_) => Err(UnifyError::TypeMismatch),
                    }
                }
                Type::Void => Ok(r1), // void can be treated as the "empty" value of a nullable.
                _ => {
                    // nullable<T> is compatible with T.
                    let inner1_ty = self.arena.nullable_inner(r1);
                    match self.arena.unify(inner1_ty, r2) {
                        Ok(_) => Ok(r1),
                        Err(_) => Err(UnifyError::TypeMismatch),
                    }
                }
            },
            (Type::Throw(_), _) => match c2 {
                Type::Throw(_) => {
                    let (vt1, et1) = self.arena.throw_parts(r1);
                    let (vt2, et2) = self.arena.throw_parts(r2);
                    let v1 = self.arena.resolve(vt1);
                    let v2 = self.arena.resolve(vt2);
                    let e1 = self.arena.resolve(et1);
                    let e2 = self.arena.resolve(et2);
                    self.arena.unify(e1, e2)?;
                    match self.arena.unify(v1, v2) {
                        Ok(_) => Ok(r1),
                        Err(_) => {
                            // Recursively call try_widen_unify to handle structural compatibility for Nullable/Throw etc.,
                            // but no longer performs numeric widening (Plan A fully strict).
                            match self.try_widen_unify(v1, v2) {
                                Ok(_) => Ok(r1),
                                Err(_) => Err(UnifyError::TypeMismatch),
                            }
                        }
                    }
                }
                Type::Void => Ok(r1), // void can be treated as throw's "no value taken".
                _ => {
                    // Throw<T, E> is compatible with T (value dimension only).
                    let (vt1_ty, _) = self.arena.throw_parts(r1);
                    match self.arena.unify(vt1_ty, r2) {
                        Ok(_) => Ok(r1),
                        Err(_) => Err(UnifyError::TypeMismatch),
                    }
                }
            },
            (Type::Void, _) => match c2 {
                Type::Nullable(_) | Type::Throw(_) => Ok(r2),
                _ => Err(UnifyError::TypeMismatch),
            },
            (_, Type::Nullable(_)) => {
                // T is compatible with nullable<T>; unify as nullable.
                let inner2_ty = self.arena.nullable_inner(r2);
                self.arena.unify(r1, inner2_ty)?;
                Ok(r2)
            }
            (_, Type::Throw(_)) => {
                // T is compatible with Throw<T, E>; unify as throw.
                let (vt2_ty, _) = self.arena.throw_parts(r2);
                self.arena.unify(r1, vt2_ty)?;
                Ok(r2)
            }
            _ => Err(UnifyError::TypeMismatch),
        }
    }

    /// Checks the legality of the propagation operator `?` on an expression and returns the unfolded type.
    ///
    /// `expected_return` is the enclosing function's return type (possibly `Async<Throw<V, E>>` or `Throw<V, E>`),
    /// used to unify error_type so that throw propagation types correctly.
    ///
    /// - nullable: unfolds to the inner type.
    /// - throw: unfolds to the value type and unifies error_type with the enclosing function's error_type.
    /// - TypeVar: deferred to the solver; returns a fresh_type_var to avoid cascading false positives.
    /// - other types: reports an error and returns the original type.
    pub fn check_propagate(
        &mut self,
        resolved_inner: TypeHandle,
        inner_ty: TypeHandle,
        expected_return: Option<TypeHandle>,
        line: u32,
        column: u32,
    ) -> TypeHandle {
        let ct = self.arena.get(resolved_inner);
        match ct {
            Type::Nullable(_) => self.arena.nullable_inner(resolved_inner),
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved_inner);
                // Unify error_type with the enclosing function's error_type (when the outer function is a throwing one).
                // Kuzo allows `?` in non-throwing functions (panics/exits on failure); in that case error_type is not propagated.
                if let Some(er) = expected_return {
                    let er_resolved = self.arena.resolve(er);
                    let er_ty = self.arena.get(er_resolved);
                    // async function: expected_return may be Async<Throw<V', E'>>.
                    let outer_throw_handle = if let Type::Async(_) = er_ty {
                        Some(self.arena.resolve(self.arena.async_value(er_resolved)))
                    } else {
                        None
                    };
                    let outer_resolved = outer_throw_handle.unwrap_or(er_resolved);
                    if let Type::Throw(_) = self.arena.get(outer_resolved) {
                        let (_, outer_err) = self.arena.throw_parts(outer_resolved);
                        self.unify_or_constrain(error_type, outer_err);
                    }
                    // Non-Throw outer (e.g. void) or TypeVar: silently skip, no error.
                }
                value_type
            }
            Type::TypeVar(_) => {
                // Operand type not yet determined; defer to the solver for later judgment.
                // Return a fresh_type_var to avoid cascading false positives in downstream lookups.
                self.arena.fresh_type_var()
            }
            _ => {
                self.add_error_at(
                    "propagation operator '?' cannot be used on a non-nullable, non-throw expression",
                    line,
                    column,
                );
                inner_ty
            }
        }
    }

    /// Checks the expression type of a throw statement.
    /// Kuzo has no try-catch; throw is a general-purpose raising mechanism that accepts any ADT/Record/Throw/TypeVar.
    pub fn check_throw_stmt(&mut self, thrown_ty: TypeHandle, _line: u32, _column: u32) {
        let resolved = self.arena.resolve(thrown_ty);
        let ct = self.arena.get(resolved);
        match ct {
            Type::TypeVar(_) => return,   // Deferred to the unification stage.
            Type::Throw(_) => return, // throw Error("...") returns Throw; legal.
            Type::Adt(_) | Type::Generic(_) => return, // Error type (ordinary ADT).
            _ => return, // Permissive: throw is a general-purpose mechanism.
        }
    }

    // ── infer_expr / infer_stmt / infer_pattern placeholders (implemented below) ──

    /// Returns the TypeHandle for a builtin scalar type (helper).
    fn make_builtin(&mut self, ty: Type) -> TypeHandle {
        self.arena.make(ty)
    }

    /// Constructs the dedicated Type variant for a builtin generic type (Throw/Channel/Async/Lazy/Atomic/Sender/Receiver).
    /// Falls back to Type::Generic on arity mismatch (fault-tolerant; sema already constrains builtin generic arity).
    fn make_builtin_generic(&mut self, name: Box<str>, args: Box<[TypeHandle]>) -> TypeHandle {
        match name.as_ref() {
            "Throw" if args.len() == 2 => self.arena.make_throw(args[0], args[1]),
            "Channel" if args.len() == 1 => self.arena.make_channel(args[0]),
            "Async" if args.len() == 1 => self.arena.make_async(args[0]),
            "Lazy" if args.len() == 1 => self.arena.make_lazy(args[0]),
            "Atomic" if args.len() == 1 => self.arena.make_atomic(args[0]),
            "Sender" if args.len() == 1 => self.arena.make_sender(args[0]),
            "Receiver" if args.len() == 1 => self.arena.make_receiver(args[0]),
            _ => self.arena.make_generic(name, args),
        }
    }

    /// Determines whether an expression is a literal (used by peer_type_binary callers).
    fn expr_is_literal(ast: &AstArena<'_>, expr: ExprId) -> bool {
        matches!(
            ast.expr(expr).node,
            Expr::IntLit { .. }
                | Expr::FloatLit { .. }
                | Expr::BoolLit(_)
                | Expr::CharLit(_)
                | Expr::StrLit(_)
                | Expr::NullLit
                | Expr::VoidLit
        )
    }

    /// Returns true if the expression has an explicitly declared numeric type
    /// that cannot be silently promoted. This includes:
    /// - Suffixed numeric literals (e.g. `1i32`, `2.0f64`)
    /// - Identifier references (variables with declared types)
    /// Computed expressions (binary ops, calls, etc.) are NOT "explicitly typed"
    /// because their type may result from bare-literal promotion internally.
    fn expr_is_explicitly_typed_numeric(ast: &AstArena<'_>, expr: ExprId) -> bool {
        match ast.expr(expr).node {
            Expr::IntLit { suffix: Some(_), .. }
            | Expr::FloatLit { suffix: Some(_), .. } => true,
            Expr::Ident(_) => true,
            _ => false,
        }
    }

    /// Check numeric binary operation type compatibility (Bug #73, #74).
    ///
    /// Rules (consistent with user preference: Rust-style strict typing,
    /// bare literals promotable, explicitly typed operands require cast):
    /// 1. Types already equal → OK
    /// 2. Cross-category (int vs float): always error (no implicit int↔float conversion)
    /// 3. Same category different bit width: error only if both sides are
    ///    explicitly typed (suffixed literal or variable identifier)
    fn check_numeric_binop_compat(
        &mut self,
        ast: &AstArena<'_>,
        lhs: ExprId,
        rhs: ExprId,
        left_ty: TypeHandle,
        right_ty: TypeHandle,
        span: crate::ast::Ast::Span,
    ) {
        // If types are already equal, no issue.
        if types_equal(self.arena, left_ty, right_ty) {
            return;
        }

        let lc = self.arena.get(left_ty);
        let rc = self.arena.get(right_ty);

        let left_str = format!("{}", self.arena.display(left_ty));
        let right_str = format!("{}", self.arena.display(right_ty));

        // Cross-category (int vs float): always error — no implicit int↔float conversion.
        if (lc.is_int() && rc.is_float()) || (lc.is_float() && rc.is_int()) {
            self.add_error_at(
                &format!(
                    "type mismatch: cannot operate on '{}' and '{}' without explicit cast (int/float category mismatch)",
                    left_str, right_str
                ),
                span.line,
                span.column,
            );
            return;
        }

        // Same category, different bit widths: error only if both sides are
        // explicitly typed (suffixed literal or variable identifier).
        // Computed expressions (e.g. `1.0 / 0.0`) may derive their type from
        // bare-literal promotion, so they are not treated as "explicitly typed".
        let left_explicit = Self::expr_is_explicitly_typed_numeric(ast, lhs);
        let right_explicit = Self::expr_is_explicitly_typed_numeric(ast, rhs);
        if left_explicit && right_explicit {
            self.add_error_at(
                &format!(
                    "type mismatch: cannot operate on '{}' and '{}' without explicit cast (different bit widths)",
                    left_str, right_str
                ),
                span.line,
                span.column,
            );
        }
    }

    /// Dereferences a ref/nullable type, returning the inner type; for non-ref/nullable types returns the original type.
    /// SafeAccess `?.` on a Nullable needs to unwrap the inner type to look up fields, matching how method calls unwrap Nullable.
    fn unwrap_ref(&self, ty: TypeHandle) -> TypeHandle {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::Ref(_) => self.arena.ref_parts(resolved).0,
            Type::Nullable(_) => self.arena.nullable_inner(resolved),
            _ => resolved,
        }
    }

    /// Structurally extracts the element type from an iterator type.
    /// Covers all standard iterator shapes:
    /// - Array<T> → T (arrays are not iterators, but the element type is extracted for constraints).
    /// - ArrayIter<T> / Iter<T> / RangeIterator → T.
    /// - Iterator of Map<K,V> → Entry<K,V>.
    /// - Str → char.
    /// - Throw<T,E> → T (destructured directly so that iterating over a Throw yields value-typed elements).
    /// Returns None on failure (the caller falls back to fresh_type_var + a constraint).
    fn extract_iterator_element(&mut self, h: TypeHandle) -> Option<TypeHandle> {
        let ty = self.arena.get(h);
        match ty {
            Type::Array(_) => Some(self.arena.array_parts(h).0),
            Type::Str => Some(self.make_builtin(Type::Char)),
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(h);
                match name {
                    // Standard iterators: ArrayIter<T>, Iter<T>, RangeIterator (no args; element is i64).
                    "ArrayIter" | "Iter" if args.len() == 1 => Some(args[0]),
                    "RangeIterator" => Some(self.make_builtin(Type::I64)),
                    // Map iterators return Entry<K,V>.
                    "MapIter" | "MapKeys" | "MapValues" if args.len() == 1 => Some(args[0]),
                    "Map" if args.len() == 2 => {
                        let entry_ty = self.arena.make_generic(
                            "Entry".into(),
                            args.to_vec().into_boxed_slice(),
                        );
                        Some(entry_ty)
                    }
                    _ => None,
                }
            }
            Type::Throw(_) => Some(self.arena.throw_parts(h).0),
            _ => None,
        }
    }
}

// =========================================================================
// phase5: InferContext extensions — expression/statement/pattern inference + module check entry
//
// Ported from `src/sema/type_check.zig`: inferExpr / inferStmt / inferPattern /
// registerBuiltins / checkModuleWithName.
// =========================================================================

impl<'a> InferContext<'a> {
    /// Stores the inferred type as ExprInfo into sema_result.expr_types.
    fn store_expr_info(&mut self, expr: ExprId, ty: TypeHandle) {
        let resolved = self.arena.resolve(ty);
        let ct = self.arena.get(resolved);
        let type_name: Option<String> = self.arena.type_name(resolved).map(|s| s.to_string());
        let is_ref = matches!(ct, Type::Ref(_));
        let is_raw_ref = matches!(ct, Type::Ref(_)) && self.arena.ref_parts(resolved).1;

        let is_trait_object = matches!(ct, Type::TraitObject(_));

        let key = if let Some(ref ictx) = self.instantiation_ctx {
            // Instantiation mode: compute the key with the instance's module name; write to the instance-local staging table + global resolved_types.
            module_expr_key(&ictx.module_name, expr.0 as u64)
        } else {
            // HM mode: compute the key with the current module name.
            module_expr_key(&self.current_module_name, expr.0 as u64)
        };

        // In instantiation mode, the original HM pass may have set `implicit_this`
        // on the ExprInfo (marking bare identifiers/calls that resolve to implicit
        // `this` field/method access). The instantiation pass re-infers types with
        // concrete type_args but does NOT set up `this_binding_stack`, so
        // `pending_implicit_this` is never set. Preserve the original marker by
        // copying it from the pre-existing ExprInfo.
        let implicit_this = if self.instantiation_ctx.is_some() {
            self.sema_result
                .expr_types
                .get(&key)
                .and_then(|info| info.implicit_this.clone())
        } else {
            None
        };

        let info = ExprInfo {
            ty: resolved,
            const_val: None,
            expr_id: expr.0 as u64,
            type_name: type_name.map(|s| s.into_boxed_str()),
            is_trait_object,
            is_ref_type: is_ref,
            is_raw_ref,
            implicit_this,
        };

        if let Some(ref mut ictx) = self.instantiation_ctx {
            ictx.local_expr_types.insert(key, info);
            self.sema_result.resolved_types.insert(key, resolved);
        } else {
            self.sema_result.put_expr(key, info);
            // Record module ownership for incremental purge (expr_types key).
            let mod_name = self.current_module_name.clone();
            self.sema_result.module_ownership.expr_type_keys
                .entry(mod_name)
                .or_default()
                .insert(key);
        }
    }

    // ── Unified capture analysis ──
    //
    // Computes the capture list (outer variables referenced by a nested scope:
    // lambda / defer / nested function) with per-capture mutability. This is the
    // single source of truth consumed by the IR builder (replacing its own
    // `collect_free_idents_expr` re-scan) and by the assignment-decision-tree
    // simplification.
    //
    // Capture mode rules:
    //   - `var` binding            → Reference (reads/writes reflect latest value)
    //   - `val` binding            → Snapshot  (captures declaration-time value)
    //   - parameter                → Snapshot  (params are immutable; no `var param`)
    //   - `this`                   → Reference (receiver is shared; mutations via it)
    //   - defer bodies             → ALL captures forced to Reference (defer semantics
    //                                require reading the value at exit, not at defer-site;
    //                                this directly resolves the Bug #49 tension where a
    //                                `val` snapshot and defer-latest conflicted on the
    //                                same node).
    //
    // The walk is scope-aware: bindings introduced *inside* the nested scope
    // (block ValDecl/VarDecl, for-loop binders, nested lambda params, pattern
    // binders) are excluded — only names resolved from the enclosing scope count
    // as captures. This correctly handles nested lambdas/defers whose own
    // captures are computed independently.

    /// Collect free identifier names referenced by a nested-scope body, excluding
    /// names bound *inside* the body. `bound` is seeded with the nested scope's
    /// own parameter names (plus `self`-reference for named functions); the walk
    /// adds inner bindings as it descends.
    fn collect_free_idents_scoped(
        &self,
        ast: &AstArena<'_>,
        expr: ExprId,
        bound: &mut rustc_hash::FxHashSet<String>,
        out: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &ast.expr(expr).node;
        match node {
            Ast::Expr::Ident(name) => {
                if !bound.contains(*name) {
                    out.insert(name.to_string());
                }
            }
            Ast::Expr::Assign { target, value } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Expr::Binary { lhs, rhs, .. } => {
                self.collect_free_idents_scoped(ast, *lhs, bound, out);
                self.collect_free_idents_scoped(ast, *rhs, bound, out);
            }
            Ast::Expr::CompoundAssign { target, value, .. } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Expr::Unary { operand, .. } => {
                self.collect_free_idents_scoped(ast, *operand, bound, out);
            }
            Ast::Expr::Call { callee, args, .. } => {
                self.collect_free_idents_scoped(ast, *callee, bound, out);
                for &a in args {
                    self.collect_free_idents_scoped(ast, a, bound, out);
                }
            }
            Ast::Expr::MethodCall { recv, args, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                for &a in args {
                    self.collect_free_idents_scoped(ast, a, bound, out);
                }
            }
            Ast::Expr::FieldAccess { recv, .. }
            | Ast::Expr::Index { recv, .. }
            | Ast::Expr::RefOf(recv)
            | Ast::Expr::Deref(recv)
            | Ast::Expr::NonNullAssert(recv)
            | Ast::Expr::Propagate(recv) => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
            }
            Ast::Expr::SafeAccess { recv, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
            }
            Ast::Expr::SafeMethodCall { recv, args, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                for &a in args {
                    self.collect_free_idents_scoped(ast, a, bound, out);
                }
            }
            Ast::Expr::Slice { recv, start, end, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                self.collect_free_idents_scoped(ast, *start, bound, out);
                self.collect_free_idents_scoped(ast, *end, bound, out);
            }
            Ast::Expr::Elvis { lhs, rhs } => {
                self.collect_free_idents_scoped(ast, *lhs, bound, out);
                self.collect_free_idents_scoped(ast, *rhs, bound, out);
            }
            Ast::Expr::If { cond, then_branch, else_branch } => {
                self.collect_free_idents_scoped(ast, *cond, bound, out);
                self.collect_free_idents_scoped(ast, *then_branch, bound, out);
                if let Some(eb) = else_branch {
                    self.collect_free_idents_scoped(ast, *eb, bound, out);
                }
            }
            Ast::Expr::Block { stmts, trailing } => {
                // Block introduces a new lexical scope: snapshot `bound`, recurse
                // (inner ValDecl/VarDecl names are added to `bound` by the stmt
                // walker), then restore.
                let snapshot = bound.clone();
                for &s in stmts {
                    self.collect_free_idents_stmt_scoped(ast, s, bound, out);
                }
                if let Some(t) = trailing {
                    self.collect_free_idents_scoped(ast, *t, bound, out);
                }
                *bound = snapshot;
            }
            Ast::Expr::Match { scrutinee, arms } => {
                self.collect_free_idents_scoped(ast, *scrutinee, bound, out);
                for arm in arms {
                    let snapshot = bound.clone();
                    self.collect_pattern_binders(ast, arm.pattern, bound);
                    if let Some(g) = arm.guard {
                        self.collect_free_idents_scoped(ast, g, bound, out);
                    }
                    self.collect_free_idents_scoped(ast, arm.body, bound, out);
                    *bound = snapshot;
                }
            }
            Ast::Expr::Lambda { params, body, .. } => {
                // Nested lambda: its params belong to it, not to the outer scope.
                let snapshot = bound.clone();
                for p in params {
                    bound.insert(p.name.to_string());
                }
                match body {
                    Ast::LambdaBody::Block(b) => self.collect_free_idents_scoped(ast, *b, bound, out),
                    Ast::LambdaBody::Expression(e) => self.collect_free_idents_scoped(ast, *e, bound, out),
                }
                *bound = snapshot;
            }
            Ast::Expr::ArrayLit { elements, fill } => {
                for &e in elements {
                    self.collect_free_idents_scoped(ast, e, bound, out);
                }
                if let Some((v, c)) = fill {
                    self.collect_free_idents_scoped(ast, *v, bound, out);
                    self.collect_free_idents_scoped(ast, *c, bound, out);
                }
            }
            Ast::Expr::RecordLit(fields) => {
                for f in fields {
                    self.collect_free_idents_scoped(ast, f.value, bound, out);
                }
            }
            Ast::Expr::RecordExtend { base, updates } => {
                self.collect_free_idents_scoped(ast, *base, bound, out);
                for f in updates {
                    self.collect_free_idents_scoped(ast, f.value, bound, out);
                }
            }
            Ast::Expr::StrInterp(parts) => {
                for part in parts {
                    if let Ast::InterpolationPart::Expression(e) = part {
                        self.collect_free_idents_scoped(ast, *e, bound, out);
                    }
                }
            }
            Ast::Expr::Select(arms) => {
                for arm in arms {
                    match arm {
                        Ast::SelectArm::Receive { channel_expr, binding, body } => {
                            self.collect_free_idents_scoped(ast, *channel_expr, bound, out);
                            // The `binding` name is local to the arm body.
                            let snapshot = bound.clone();
                            if let Some(b) = binding {
                                bound.insert(b.to_string());
                            }
                            self.collect_free_idents_scoped(ast, *body, bound, out);
                            *bound = snapshot;
                        }
                        Ast::SelectArm::Timeout { duration, body } => {
                            self.collect_free_idents_scoped(ast, *duration, bound, out);
                            self.collect_free_idents_scoped(ast, *body, bound, out);
                        }
                    }
                }
            }
            Ast::Expr::Atomic(inner) | Ast::Expr::Lazy(inner) => {
                self.collect_free_idents_scoped(ast, *inner, bound, out);
            }
            Ast::Expr::InlineTrait(methods) => {
                for m in methods {
                    if let Some(body_expr) = m.body {
                        self.collect_free_idents_scoped(ast, body_expr, bound, out);
                    }
                }
            }
            // Literals and other leaf variants: no sub-expressions.
            _ => {}
        }
    }

    /// Statement-level scoped free-identifier collection (companion to
    /// `collect_free_idents_scoped`).
    fn collect_free_idents_stmt_scoped(
        &self,
        ast: &AstArena<'_>,
        stmt: StmtId,
        bound: &mut rustc_hash::FxHashSet<String>,
        out: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &ast.stmt(stmt).node;
        match node {
            Ast::Stmt::ValDecl { name, value, .. }
            | Ast::Stmt::VarDecl { name, value, .. } => {
                self.collect_free_idents_scoped(ast, *value, bound, out);
                bound.insert(name.to_string());
            }
            Ast::Stmt::Expression { expr } => {
                self.collect_free_idents_scoped(ast, *expr, bound, out);
            }
            Ast::Stmt::Assignment { target, value } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Stmt::FieldAssignment { object, value, .. } => {
                self.collect_free_idents_scoped(ast, *object, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Stmt::CompoundAssignment { target, value, .. } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Stmt::Return { value } => {
                if let Some(v) = value {
                    self.collect_free_idents_scoped(ast, *v, bound, out);
                }
            }
            Ast::Stmt::Throw { expr } => {
                self.collect_free_idents_scoped(ast, *expr, bound, out);
            }
            Ast::Stmt::For { name, iterable, body } => {
                self.collect_free_idents_scoped(ast, *iterable, bound, out);
                let snapshot = bound.clone();
                bound.insert(name.to_string());
                self.collect_free_idents_scoped(ast, *body, bound, out);
                *bound = snapshot;
            }
            Ast::Stmt::While { condition, body } => {
                self.collect_free_idents_scoped(ast, *condition, bound, out);
                self.collect_free_idents_scoped(ast, *body, bound, out);
            }
            Ast::Stmt::Loop { body } => {
                self.collect_free_idents_scoped(ast, *body, bound, out);
            }
            Ast::Stmt::Defer { expr } => {
                self.collect_free_idents_scoped(ast, *expr, bound, out);
            }
            Ast::Stmt::Break | Ast::Stmt::Continue => {}
            Ast::Stmt::LocalDecl { decl } => match decl.as_ref() {
                Ast::Decl::FunDecl { name, params, body, .. } => {
                    // A nested function declaration binds its own name in the
                    // current scope. Its params belong to it. References inside
                    // its body that resolve to the *current* scope are captures
                    // of the current scope — record them, treating the nested
                    // function's name + params as bound.
                    bound.insert(name.to_string());
                    let snapshot = bound.clone();
                    for p in params {
                        bound.insert(p.name.to_string());
                    }
                    self.collect_free_idents_scoped(ast, *body, bound, out);
                    *bound = snapshot;
                }
                _ => {}
            },
        }
    }

    /// Add pattern-bound variable names to `bound`.
    fn collect_pattern_binders(
        &self,
        ast: &AstArena<'_>,
        pat: PatternId,
        bound: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &ast.pattern(pat).node;
        match node {
            Ast::Pattern::Variable { name } => {
                bound.insert(name.to_string());
            }
            Ast::Pattern::Wildcard | Ast::Pattern::Literal(_) => {}
            Ast::Pattern::Constructor { patterns, .. } => {
                for &p in patterns {
                    self.collect_pattern_binders(ast, p, bound);
                }
            }
            Ast::Pattern::Record { fields } => {
                for f in fields {
                    self.collect_pattern_binders(ast, f.pattern, bound);
                }
            }
            Ast::Pattern::OrPattern { left, right } => {
                self.collect_pattern_binders(ast, *left, bound);
                self.collect_pattern_binders(ast, *right, bound);
            }
            Ast::Pattern::Guard { pattern, .. } => {
                self.collect_pattern_binders(ast, *pattern, bound);
            }
        }
    }

    /// Compute the capture list for a nested scope.
    ///
    /// `body_expr` is the scope's body expression; `param_names` are the scope's
    /// own parameter names (excluded from captures); `force_reference` overrides
    /// all captures to `Reference` mode (used for defer).
    fn compute_captures(
        &self,
        ast: &AstArena<'_>,
        body_expr: ExprId,
        param_names: &[&str],
        force_reference: bool,
    ) -> Vec<crate::sema::Sema::CaptureInfo> {
        use crate::sema::Sema::{CaptureInfo, CaptureMode};

        let mut bound: rustc_hash::FxHashSet<String> = param_names.iter().map(|s| s.to_string()).collect();
        let mut free: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
        self.collect_free_idents_scoped(ast, body_expr, &mut bound, &mut free);

        // Deterministic order for stable output.
        let mut names: Vec<String> = free.into_iter().collect();
        names.sort();

        names
            .into_iter()
            .map(|name| {
                let mode = if force_reference {
                    CaptureMode::Reference
                } else {
                    self.capture_mode_for(&name)
                };
                CaptureInfo {
                    name: name.into_boxed_str(),
                    decl_key: 0, // resolved by name on the IR side (lookup_var)
                    mode,
                }
            })
            .collect()
    }

    /// Decide the capture mode for a referenced variable based on its declared
    /// mutability. `this` is always `Reference` (shared receiver). `var`
    /// bindings are `Reference`; `val`/params/unresolved default to `Snapshot`.
    fn capture_mode_for(&self, name: &str) -> crate::sema::Sema::CaptureMode {
        use crate::sema::Sema::CaptureMode;
        if name == "this" {
            return CaptureMode::Reference;
        }
        // Search local_mutability for any entry with this name (the env_id
        // component is not known here; a `var` binding will have an entry with
        // `is_mutable == true` somewhere in the table).
        let is_var = self
            .local_mutability
            .iter()
            .any(|((_, n), mutable)| n.as_str() == name && *mutable);
        if is_var {
            CaptureMode::Reference
        } else {
            CaptureMode::Snapshot
        }
    }

    // ── infer_expr ──

    /// Infers the type of an expression. This is the core entry point of type checking and recursively handles all expression variants.
    /// After inference, stores the ExprInfo into sema_result.
    pub fn infer_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        let ty = self.infer_expr_inner(expr, ast, env, expected);
        self.store_expr_info(expr, ty);
        // Flush pending implicit-this marker into the staged ExprInfo.
        if let Some((eid, access)) = self.pending_implicit_this.take() {
            let key = if let Some(ref ictx) = self.instantiation_ctx {
                module_expr_key(&ictx.module_name, eid.0 as u64)
            } else {
                module_expr_key(&self.current_module_name, eid.0 as u64)
            };
            let info = if let Some(ref mut ictx) = self.instantiation_ctx {
                ictx.local_expr_types.get_mut(&key)
            } else {
                self.sema_result.expr_types.get_mut(&key)
            };
            if let Some(info) = info {
                info.implicit_this = Some(access);
            }
        }
        // Diagnostic trace: only record (TypeHandle, Span) when KUZO_SEMA_TRACE is enabled.
        if std::env::var("KUZO_SEMA_TRACE").is_ok() {
            let span = ast.expr(expr).span;
            self.type_trace.push((ty, span));
        }
        ty
    }

    /// Internal implementation of expression type inference (does not store ExprInfo).
    fn infer_expr_inner(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        let node = &ast.expr(expr).node;
        match node {
            // ── Literals ──
            Expr::IntLit { raw, suffix } => {
                // Range-check suffixed integer literals at sema time (Bug #72: stage consistency with IR Builder).
                if let Some(suf) = suffix {
                    if let Some(tag) = crate::types::ValueTag::from_name(suf) {
                        if let Some(err) = check_int_literal_range(raw, tag) {
                            self.add_error(&err);
                        }
                    }
                }
                numeric_lit!(self, suffix, expected, int_suffix_to_type, is_int, I32)
            }
            Expr::FloatLit { suffix, .. } => numeric_lit!(self, suffix, expected, float_suffix_to_type, is_float, F64),
            Expr::BoolLit(_) => self.make_builtin(Type::Bool),
            Expr::CharLit(_) => self.make_builtin(Type::Char),
            Expr::StrLit(_) => self.make_builtin(Type::Str),
            Expr::StrInterp(parts) => {
                // Recursively infer the sub-expressions inside the interpolation so their ExprInfo is registered in expr_types.
                // Otherwise the IR compiler's `select_binary_compute_fn` falls back to "i32" when it cannot find the type,
                // and mis-dispatches bool/str (non-integer) types to CF_EQ_I32 (as_i32 on bool is always 0).
                for p in parts {
                    if let InterpolationPart::Expression(e) = p {
                        let _ = self.infer_expr(*e, ast, env, None);
                    }
                }
                self.make_builtin(Type::Str)
            }
            Expr::NullLit => {
                // The null literal has type Nullable<T>, where T is solved via the expected constraint.
                // try_widen_unify handles all expected types (Nullable<T> unifies the inner type;
                // other types try to widen or report an error), so no type-specific special-casing of expected is needed.
                let tv = self.arena.fresh_type_var();
                let ty = self.arena.make_nullable(tv);
                if let Some(exp) = expected {
                    if let Err(e) = self.try_widen_unify(exp, ty) {
                        self.add_error(&format!("null literal incompatible with expected type: {}", e));
                    }
                }
                ty
            }
            Expr::VoidLit => self.make_builtin(Type::Void),

            // ── Identifiers ──
            Expr::Ident(_) => self.infer_ident_expr(expr, ast, env),

            // ── Assignment ──
            Expr::Assign { target, value } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                self.unify_or_constrain(target_ty, val_ty);
                self.make_builtin(Type::Void)
            }
            Expr::CompoundAssign { target, value, .. } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                self.unify_or_constrain(target_ty, val_ty);
                target_ty
            }

            // ── Binary operations ──
            Expr::Binary { .. } => self.infer_binary_expr(expr, ast, env),

            // ── Unary operations ──
            Expr::Unary { operand, .. } => {
                let _ = self.infer_expr(*operand, ast, env, None);
                // ! / ~ / - all return the operand's type.
                self.infer_expr(*operand, ast, env, None)
            }

            // ── Reference / dereference ──
            Expr::RefOf(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_ref(inner_ty, false)
            }
            Expr::Deref(operand) => {
                let operand_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(operand_ty);
                match self.arena.get(resolved) {
                    Type::Ref(_) => self.arena.ref_parts(resolved).0,
                    _ => operand_ty, // Dereferencing a non-reference: return the original type.
                }
            }

            // ── Function calls ──
            Expr::Call { .. } => self.infer_call_expr(expr, ast, env, expected),

            // ── Method calls ──
            Expr::MethodCall { .. }
            | Expr::SafeMethodCall { .. } => self.infer_method_call_expr(expr, ast, env, expected),

            // ── Field access ──
            Expr::FieldAccess { recv, field } => {
                // Qualified-name syntax: Type.Ctor (qualified access of a zero-argument constructor)
                if let Expr::Ident(type_name) = &ast.expr(*recv).node {
                    if let Some((ctor_type_name, field_type_reprs)) =
                        self.check_qualified_ctor(type_name, field)
                    {
                        if field_type_reprs.is_empty() {
                            // Zero-argument constructor: return Adt(type_name)
                            return self.arena.make_adt(ctor_type_name, Box::new([]));
                        }
                        // Constructor with arguments in FieldAccess: report an error
                        let span = ast.expr(expr).span;
                        self.add_error_at(
                            &format!(
                                "constructor '{}' of type '{}' requires arguments; use {}('{}') syntax",
                                field, type_name, field, type_name
                            ),
                            span.line,
                            span.column,
                        );
                        return self.arena.fresh_type_var();
                    }
                }

                let recv_ty = self.infer_expr(*recv, ast, env, None);
                // Detect a ModuleRef receiver: cross-module constant access such as Math.PI.
                // On hit, record recv's expr key → mangled name (module_path.field) into
                // module_const_recv_exprs, so IR compilation skips recv and emits a global_load directly.
                let recv_resolved = self.arena.resolve(recv_ty);
                if let Type::ModuleRef(_) = self.arena.get(recv_resolved) {
                    let (path, module_env) = self.arena.module_ref_parts(recv_resolved);
                    if self.env.lookup_local(module_env, field).is_some() {
                        let mangled = format!("{}.{}", path, field);
                        let recv_key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            recv.0 as u64,
                        );
                        self.sema_result.module_const_recv_exprs.insert(recv_key, mangled);
                    }
                }
                let span = ast.expr(expr).span;
                self.lookup_field_type(recv_ty, field, span.line, span.column)
            }
            Expr::SafeAccess { recv, field } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let resolved = self.arena.resolve(recv_ty);
                // SafeAccess `?.` is only meaningful for Nullable/Ref; for other types it degrades to an ordinary field access.
                let is_nullable = matches!(self.arena.get(resolved), Type::Nullable(_));
                let inner = self.unwrap_ref(recv_ty);
                let span = ast.expr(expr).span;
                let field_ty = self.lookup_field_type(inner, field, span.line, span.column);
                // For a Nullable receiver, the field-access result should also be Nullable (propagating the None semantic).
                if is_nullable {
                    self.arena.make_nullable(field_ty)
                } else {
                    field_ty
                }
            }

            // ── Index / slice ──
            Expr::Index { recv, index } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let _ = self.infer_expr(*index, ast, env, None);
                let resolved = self.arena.resolve(recv_ty);
                match self.arena.get(resolved) {
                    Type::Array(_) => self.arena.array_parts(resolved).0,
                    // Str indexing returns Char (stdlib uses patterns like normalized[0] == '/').
                    Type::Str => self.arena.make(Type::Char),
                    // Unknown/TypeVar/Generic/Adt, etc. do not report errors:
                    // sema v2 does not always infer variable types precisely (e.g. u8[] may be unified as Unknown);
                    // until sema type inference matures, permissively allow these to avoid cascading false positives.
                    _ => self.arena.fresh_type_var(),
                }
            }
            Expr::Slice { recv, start, end, .. } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let _ = self.infer_expr(*start, ast, env, None);
                let _ = self.infer_expr(*end, ast, env, None);
                recv_ty // A slice returns the same type.
            }

            // ── Propagation ──
            Expr::Propagate(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(inner_ty);
                let span = ast.expr(expr).span;
                self.check_propagate(resolved, inner_ty, self.expected_return, span.line, span.column)
            }
            Expr::NonNullAssert(operand) => {
                let operand_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(operand_ty);
                match self.arena.get(resolved) {
                    Type::Nullable(_) => self.arena.nullable_inner(resolved),
                    _ => operand_ty,
                }
            }
            Expr::Elvis { lhs, rhs } => {
                let left_ty = self.infer_expr(*lhs, ast, env, None);
                let right_ty = self.infer_expr(*rhs, ast, env, None);
                let rl = self.arena.resolve(left_ty);
                if let Type::Nullable(_) = self.arena.get(rl) {
                    let inner = self.arena.nullable_inner(rl);
                    if let Err(e) = self.try_widen_unify(inner, right_ty) {
                        self.add_error(&format!("?? default value incompatible with Nullable inner type: {}", e));
                    }
                    inner
                } else if let Type::Throw(_) = self.arena.get(rl) {
                    // Throw<T,E> ?? rhs → returns T, symmetric with Nullable (Bug #28).
                    let value_ty = self.arena.throw_parts(rl).0;
                    if let Err(e) = self.try_widen_unify(value_ty, right_ty) {
                        self.add_error(&format!("?? default value incompatible with Throw value type: {}", e));
                    }
                    value_ty
                } else {
                    left_ty
                }
            }

            // ── Array literals ──
            Expr::ArrayLit { elements, fill } => {
                // Extract the element type from expected so literal elements can be promoted per the annotation.
                // (e.g. in `val data: u8[] = [72, 101]`, 72 should be promoted to u8 rather than the default i32.)
                let expected_elem = expected.and_then(|exp| {
                    let r = self.arena.resolve(exp);
                    match self.arena.get(r) {
                        Type::Array(_) => Some(self.arena.array_parts(r).0),
                        _ => None,
                    }
                });
                // Array fill syntax: [value, ..count] — infer value and count, return runtime-sized array
                if let Some((value, count)) = fill {
                    let value_ty = self.infer_expr(*value, ast, env, expected_elem);
                    // Infer count to register its ExprInfo; length is runtime-determined
                    let _count_ty = self.infer_expr(*count, ast, env, None);
                    return self.arena.make_array(value_ty, None);
                }
                if elements.is_empty() {
                    let elem_ty = expected_elem.unwrap_or_else(|| self.arena.fresh_type_var());
                    return self.arena.make_array(elem_ty, None);
                }
                let first_ty = self.infer_expr(elements[0], ast, env, expected_elem);
                for &e in elements.iter().skip(1) {
                    let elem_ty = self.infer_expr(e, ast, env, expected_elem);
                    if let Err(e_err) = self.try_widen_unify(first_ty, elem_ty) {
                        self.add_error(&format!("array element type mismatch: {}", e_err));
                    }
                }
                self.arena.make_array(first_ty, Some(elements.len() as u64))
            }

            // ── Record literals ──
            Expr::RecordLit(fields) => {
                let field_types: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: Some(f.name.into()),
                        ty: self.infer_expr(f.value, ast, env, None),
                    })
                    .collect();
                self.arena.make_record(field_types.into_boxed_slice(), None)
            }
            Expr::RecordExtend { base, updates } => {
                let base_ty = self.infer_expr(*base, ast, env, None);
                let resolved = self.arena.resolve(base_ty);
                match self.arena.get(resolved) {
                    Type::Record(_) => {
                        let base_fields = self.arena.record_fields(resolved);
                        let name = self.arena.record_name(resolved).map(|s| s.into());
                        let mut all_fields: Vec<FieldType> = base_fields.to_vec();
                        for update in updates.iter() {
                            let update_ty = self.infer_expr(update.value, ast, env, None);
                            let mut found = false;
                            for f in all_fields.iter_mut() {
                                if f.name.as_deref() == Some(update.name) {
                                    f.ty = update_ty;
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                all_fields.push(FieldType {
                                    name: Some(update.name.into()),
                                    ty: update_ty,
                                });
                            }
                        }
                        self.arena.make_record(all_fields.into_boxed_slice(), name)
                    }
                    _ => {
                        let span = ast.expr(expr).span;
                        self.add_error_at("record extend requires record type", span.line, span.column);
                        self.arena.fresh_type_var()
                    }
                }
            }

            // ── Lambda ──
            Expr::Lambda { params, body, is_async, return_type } => {
                let child_env = self.env.child(env);
                let param_types: Vec<TypeHandle> = params
                    .iter()
                    .map(|p| {
                        let param_ty = match p.type_annotation {
                            Some(ta) => self.type_from_ast(ta, ast),
                            None => self.arena.fresh_type_var(),
                        };
                        self.env.define(child_env, p.name, param_ty);
                        param_ty
                    })
                    .collect();
                let body_ty = match body {
                    LambdaBody::Block(b) => self.infer_expr(*b, ast, child_env, None),
                    LambdaBody::Expression(e) => self.infer_expr(*e, ast, child_env, None),
                };
                // ── Unified capture analysis ──
                // Record the capture list for this lambda scope. The IR builder
                // consumes this (replacing its own `collect_free_idents_expr`
                // re-scan); the per-capture mode drives by-val vs by-ref at
                // runtime. Self-reference detection: a named nested function
                // referencing its own name is excluded from captures.
                {
                    let body_expr_id = match body {
                        LambdaBody::Block(b) => *b,
                        LambdaBody::Expression(e) => *e,
                    };
                    let mut param_names: Vec<&str> = params.iter().map(|p| p.name).collect();
                    // No name available at the Lambda expr level (named nested
                    // functions go through `Stmt::LocalDecl`); self-upvalue is
                    // handled there.
                    let _ = &mut param_names;
                    let captures = self.compute_captures(ast, body_expr_id, &param_names, false);
                    if !captures.is_empty() || self.instantiation_ctx.is_none() {
                        // Only record during the HM pass (instantiation mode
                        // reuses the HM-pass capture table).
                        if self.instantiation_ctx.is_none() {
                            let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                            self.sema_result.put_captures(key, &self.current_module_name, captures);
                        }
                    }
                }
                let effective_body_ty = if let Some(rt) = return_type {
                    let annot_ty = self.type_from_ast(*rt, ast);
                    if let Err(e) = self.try_widen_unify(annot_ty, body_ty) {
                        self.add_error(&format!("lambda body type incompatible with declared return type: {}", e));
                    }
                    annot_ty
                } else {
                    body_ty
                };
                let ret_ty = if *is_async {
                    self.arena.make_async(effective_body_ty)
                } else {
                    effective_body_ty
                };
                self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
            }

            // ── if expressions ──
            Expr::If { cond, then_branch, else_branch } => {
                let cond_ty = self.infer_expr(*cond, ast, env, None);
                let bool_ty = self.make_builtin(Type::Bool);
                self.unify_or_constrain(cond_ty, bool_ty);

                // sema v2: extract flow facts (nullable narrowing).
                let (then_facts, else_facts) = analyze_null_check_facts(
                    self.arena,
                    ast,
                    *cond,
                    env,
                    &self.env,
                );

                let then_env = self.env.child(env);
                // Enter the then scope and apply the then facts.
                self.flow_ctx.push_scope();
                for fact in &then_facts {
                    self.flow_ctx.add_fact(fact.clone());
                }
                let then_ty = self.infer_expr(*then_branch, ast, then_env, expected);
                self.flow_ctx.pop_scope();

                if let Some(else_br) = else_branch {
                    let else_env = self.env.child(env);
                    // Enter the else scope and apply the else facts.
                    self.flow_ctx.push_scope();
                    for fact in &else_facts {
                        self.flow_ctx.add_fact(fact.clone());
                    }
                    let else_ty = self.infer_expr(*else_br, ast, else_env, expected);
                    self.flow_ctx.pop_scope();

                    // v2 convergence: use only peer_type to unify branch types (eliminates the try_widen_unify dual-track scheme).
                    // peer_type already inlines Never/Void filtering, numeric widening, and nullable/throw propagation.
                    peer_type(self.arena, &[then_ty, else_ty])
                } else {
                    // No else branch: the implicit else falls through as Void.
                    // peer_type(then, Void) ensures a diverging then (Never) does
                    // not make the whole if diverge — the fall-through path is
                    // reachable. Only an explicit `else { diverge }` yields Never.
                    let void_ty = self.make_builtin(Type::Void);
                    peer_type(self.arena, &[then_ty, void_ty])
                }
            }

            // ── Block expressions ──
            Expr::Block { stmts, trailing } => {
                let child_env = self.env.child(env);
                let mut diverges = false;
                for &stmt in stmts.iter() {
                    if diverges {
                        // Bug #84: code after a diverging statement is unreachable.
                        // Report a warning but continue inferring so the IR builder has
                        // ExprInfo for all expressions (it processes all statements
                        // independently of sema's divergence analysis).
                        let span = ast.stmt(stmt).span;
                        self.add_warning_at("unreachable code after throw/return/break/continue", span.line, span.column);
                    }
                    let stmt_ty = self.infer_stmt(stmt, ast, child_env);
                    // Detect divergence: direct control-flow exits (return/throw/break/
                    // continue) or statements whose inferred type is Never (e.g. an
                    // if/match/block expression where all branches diverge).
                    let is_direct_exit = matches!(
                        &ast.stmt(stmt).node,
                        Stmt::Return { .. } | Stmt::Throw { .. } | Stmt::Break | Stmt::Continue
                    );
                    let is_never = stmt_ty
                        .map(|t| matches!(self.arena.get(self.arena.resolve(t)), Type::Never))
                        .unwrap_or(false);
                    if !diverges && (is_direct_exit || is_never) {
                        diverges = true;
                    }
                }
                if let Some(te) = trailing {
                    if diverges {
                        // Trailing expression after a diverging statement is unreachable.
                        let span = ast.expr(*te).span;
                        self.add_warning_at("unreachable code after throw/return/break/continue", span.line, span.column);
                        // Still infer the trailing expression so the IR builder has
                        // ExprInfo for it (it processes all expressions independently
                        // of sema's divergence analysis). The block's type is Never
                        // regardless of the trailing expression's type.
                        let _ = self.infer_expr(*te, ast, child_env, expected);
                        self.make_builtin(Type::Never)
                    } else {
                        self.infer_expr(*te, ast, child_env, expected)
                    }
                } else if diverges {
                    self.make_builtin(Type::Never)
                } else {
                    self.make_builtin(Type::Void)
                }
            }

            // ── match expressions ──
            Expr::Match { .. } => self.infer_match_expr(expr, ast, env, expected),

            // ── Atomic / Lazy ──
            Expr::Atomic(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_atomic(inner_ty)
            }
            Expr::Lazy(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_lazy(inner_ty)
            }

            // ── select expressions: Go-style channel multiplexing ──
            //
            // Iterate over all arms:
            //   receive arm: create a child env, infer Channel<T> from channel_expr,
            //                extract the element type T for the binding (if any), and infer the body type.
            //   timeout arm: directly infer the body type.
            // Use peer_type to join all body types (consistent with Match; more robust than the Zig side, which only takes the first).
            Expr::Select(arms) => {
                let mut arm_tys: Vec<TypeHandle> = Vec::new();
                for arm in arms.iter() {
                    let child_env = self.env.child(env);
                    self.flow_ctx.push_scope();
                    match arm {
                        crate::ast::Ast::SelectArm::Receive { channel_expr, binding, body } => {
                            // Infer the channel expression's type and extract the element type for the binding.
                            let chan_ty = self.infer_expr(*channel_expr, ast, child_env, None);
                            let resolved = self.arena.resolve(chan_ty);
                            let elem_ty = match self.arena.get(resolved) {
                                // Nullable(Channel<T>) → take Channel's T.
                                Type::Nullable(_) => {
                                    let inner = self.arena.nullable_inner(resolved);
                                    let inner_resolved = self.arena.resolve(inner);
                                    match self.arena.get(inner_resolved) {
                                        Type::Channel(_) => self.arena.channel_elem(inner_resolved),
                                        _ => chan_ty,
                                    }
                                }
                                // Channel<T> → take T.
                                Type::Channel(_) => self.arena.channel_elem(resolved),
                                _ => chan_ty,
                            };
                            if let Some(name) = binding {
                                let _ = self.env.define(child_env, name, elem_ty);
                            }
                            let body_ty = self.infer_expr(*body, ast, child_env, None);
                            arm_tys.push(body_ty);
                        }
                        crate::ast::Ast::SelectArm::Timeout { duration, body } => {
                            // Infer the duration expression too — without this, its ExprInfo
                            // is never written and the IR builder reports "missing ExprInfo".
                            let _ = self.infer_expr(*duration, ast, child_env, None);
                            let body_ty = self.infer_expr(*body, ast, child_env, None);
                            arm_tys.push(body_ty);
                        }
                    }
                    self.flow_ctx.pop_scope();
                }
                if arm_tys.is_empty() {
                    self.make_builtin(Type::Void)
                } else {
                    peer_type(self.arena, &arm_tys)
                }
            }

            // ── inline_trait values: construct a TraitObject type ──
            Expr::InlineTrait(_) => self.infer_inline_trait_expr(expr, ast, env, expected),
        }
    }

    /// Infer an `Expr::Ident` expression (extracted from `infer_expr_inner`).
    fn infer_ident_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Ident(name) => {
                // sema v2: prefer the flow-narrowing result (path-sensitive type refinement).
                if let Some(narrowed_ty) = self.flow_ctx.lookup_narrowed(name) {
                    return narrowed_ty;
                }
                // Resolution order inside a method body (current_this_type non-empty):
                //   1. lookup_local  — local variables and parameters only (no parent traversal)
                //   2. For concrete types: try_implicit_this_field — fields before methods
                //      (prevents same-named methods in the parent env from shadowing fields)
                //   3. env.lookup    — full chain (methods, top-level functions)
                //   4. For trait default methods (TypeVar): try_implicit_this_field — permissive
                //      fallback for fields that can't be verified at trait definition time
                let this_ty_opt = self.current_this_type();
                if let Some(this_ty) = this_ty_opt {
                    // 1. Local variables and parameters only.
                    if let Some(scheme) = self.env.lookup_local(env, name) {
                        return self.freshen_type(scheme);
                    }
                    let is_typevar = matches!(
                        self.arena.get(self.arena.resolve(this_ty)),
                        Type::TypeVar(_)
                    );
                    // 2. Concrete types: fields take precedence over same-named methods.
                    if !is_typevar {
                        if let Some(field_ty) = self.try_implicit_this_field(this_ty, name) {
                            self.pending_implicit_this = Some((
                                expr,
                                crate::sema::Sema::ImplicitThisAccess::Field((*name).to_string().into_boxed_str()),
                            ));
                            return field_ty;
                        }
                    }
                    // 3. Full lookup (methods registered in parent env, top-level functions).
                    if let Some(scheme) = self.env.lookup(env, name) {
                        return self.freshen_type(scheme);
                    }
                    // 4. Trait default methods: permissive field fallback (TypeVar can't
                    //    verify field existence; deferred to monomorphization).
                    if is_typevar {
                        if let Some(field_ty) = self.try_implicit_this_field(this_ty, name) {
                            self.pending_implicit_this = Some((
                                expr,
                                crate::sema::Sema::ImplicitThisAccess::Field((*name).to_string().into_boxed_str()),
                            ));
                            return field_ty;
                        }
                    }
                } else {
                    // Outside methods: full env lookup.
                    if let Some(scheme) = self.env.lookup(env, name) {
                        return self.freshen_type(scheme);
                    }
                }
                // Instantiation mode: the temporary InferContext's env does not contain module-level declarations;
                // query sema_result instead (already resolved in the HM stage).
                if self.instantiation_ctx.is_some() {
                    // Look up from expr_types (the expression's type was already resolved in the HM stage).
                    let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                    if let Some(info) = self.sema_result.get_expr(key) {
                        return info.ty;
                    }
                    // In instantiation mode, do not report an error; return a fresh_type_var.
                    return self.arena.fresh_type_var();
                }
                let span = ast.expr(expr).span;
                self.add_error_at(&format!("undefined variable '{}'", name), span.line, span.column);
                self.arena.fresh_type_var()
            }
            _ => unreachable!("infer_ident_expr called on non-Ident expression"),
        }
    }

    /// Infer an `Expr::Binary` expression (extracted from `infer_expr_inner`).
    fn infer_binary_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Binary { op, lhs, rhs } => {
                let left_ty = self.infer_expr(*lhs, ast, env, None);
                let right_ty = self.infer_expr(*rhs, ast, env, None);
                let left_is_lit = Self::expr_is_literal(ast, *lhs);
                let right_is_lit = Self::expr_is_literal(ast, *rhs);
                let bin_span = ast.expr(expr).span;
                // Lazy<T> subsumption: unwrap Lazy to inner type for binary operations.
                // `lazy(1i32) + 3i32` treats the left operand as i32.
                let left_unwrapped = {
                    let rl = self.arena.resolve(left_ty);
                    if matches!(self.arena.get(rl), Type::Lazy(_)) {
                        self.arena.lazy_value(rl)
                    } else {
                        left_ty
                    }
                };
                let right_unwrapped = {
                    let rr = self.arena.resolve(right_ty);
                    if matches!(self.arena.get(rr), Type::Lazy(_)) {
                        self.arena.lazy_value(rr)
                    } else {
                        right_ty
                    }
                };
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                        let rl = self.arena.resolve(left_unwrapped);
                        let rr = self.arena.resolve(right_unwrapped);
                        if self.arena.get(rl).is_numeric() && self.arena.get(rr).is_numeric() {
                            // Bug #73/#74: strict numeric type checking.
                            // - Bare literals (no suffix) can be promoted freely.
                            // - Explicitly typed operands (suffixed literals or variables)
                            //   require explicit cast for different bit widths or int/float crossing.
                            self.check_numeric_binop_compat(ast, *lhs, *rhs, rl, rr, bin_span);
                            // v2 convergence: peer_type_binary replaces literal_promotion;
                            // literal promotion rules are inlined into peer_type_binary.
                            return peer_type_binary(
                                self.arena,
                                left_unwrapped,
                                right_unwrapped,
                                left_is_lit,
                                right_is_lit,
                            );
                        }
                        self.unify_or_constrain(left_unwrapped, right_unwrapped);
                        left_unwrapped
                    }
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::RefEq | BinaryOp::RefNeq
                    | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
                        let rl = self.arena.resolve(left_unwrapped);
                        let rr = self.arena.resolve(right_unwrapped);
                        if self.arena.get(rl).is_numeric() && self.arena.get(rr).is_numeric() {
                            // Bug #73/#74: same strict checking for comparison ops.
                            self.check_numeric_binop_compat(ast, *lhs, *rhs, rl, rr, bin_span);
                            // v2 convergence: comparison ops use peer_type_binary to unify operand types.
                            let _ = peer_type_binary(
                                self.arena,
                                left_unwrapped,
                                right_unwrapped,
                                left_is_lit,
                                right_is_lit,
                            );
                        } else {
                            self.unify_or_constrain(left_unwrapped, right_unwrapped);
                        }
                        self.make_builtin(Type::Bool)
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        let bool_ty = self.make_builtin(Type::Bool);
                        self.unify_or_constrain(left_unwrapped, bool_ty);
                        self.unify_or_constrain(right_unwrapped, bool_ty);
                        bool_ty
                    }
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                    | BinaryOp::Shl | BinaryOp::Shr => {
                        self.unify_or_constrain(left_unwrapped, right_unwrapped);
                        left_unwrapped
                    }
                    BinaryOp::ConcatList => {
                        // Array concatenation a ++ b: left and right element types must match; the result reuses the left operand's element type.
                        // Avoids creating an orphan fresh_type_var (res_elem would have no constraint to the inputs).
                        let left_elem = self.arena.fresh_type_var();
                        let left_arr = self.arena.make_array(left_elem, None);
                        self.unify_or_constrain(left_unwrapped, left_arr);
                        let right_arr = self.arena.make_array(left_elem, None);
                        self.unify_or_constrain(right_unwrapped, right_arr);
                        self.arena.make_array(left_elem, None)
                    }
                    BinaryOp::Range | BinaryOp::RangeInclusive => {
                        // Range expressions a..b / a..=b return a RangeIterator type
                        // (Range is itself an iterator; For loops statically dispatch through RangeIterator.next).
                        let i64_ty = self.make_builtin(Type::I64);
                        if let Err(e) = self.try_widen_unify(i64_ty, left_unwrapped) {
                            self.add_error(&format!("range operand must be integer: {}", e));
                        }
                        let i64_ty = self.make_builtin(Type::I64);
                        if let Err(e) = self.try_widen_unify(i64_ty, right_unwrapped) {
                            self.add_error(&format!("range operand must be integer: {}", e));
                        }
                        self.arena.make_generic(
                            "RangeIterator".into(),
                            Box::new([]),
                        )
                    }
                    BinaryOp::Elvis => {
                        let rl = self.arena.resolve(left_ty);
                        if let Type::Nullable(_) = self.arena.get(rl) {
                            return self.arena.nullable_inner(rl);
                        }
                        // Throw<T,E> ?? rhs → returns T (the Ok value type), symmetric with Nullable (Bug #28).
                        if let Type::Throw(_) = self.arena.get(rl) {
                            let value_ty = self.arena.throw_parts(rl).0;
                            // Unify rhs with value_ty to ensure the default value's type is compatible.
                            if let Err(e) = self.try_widen_unify(value_ty, right_ty) {
                                self.add_error(&format!("?? default value incompatible with Throw value type: {}", e));
                            }
                            return value_ty;
                        }
                        left_ty
                    }
                }
            }
            _ => unreachable!("infer_binary_expr called on non-Binary expression"),
        }
    }

    /// Infer an `Expr::Match` expression (extracted from `infer_expr_inner`).
    fn infer_match_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Match { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expr(*scrutinee, ast, env, None);
                let resolved_scrutinee = self.arena.resolve(scrutinee_ty);

                // Throw type widening: when a match contains both Ok and Error/error-constructor patterns,
                // the scrutinee should be a Throw<T, E>. If the scrutinee is an Err implementer (e.g. an Error ADT)
                // rather than a Throw, widen it to Throw<fresh, scrutinee_ty> (with scrutinee as error_type),
                // so that Ok(v) matches the value variant and Error(e) matches the error variant and binds the whole error value (not the constructor's fields).
                let resolved_scrutinee = {
                    let is_throw = matches!(
                        self.arena.get(resolved_scrutinee),
                        Type::Throw(_)
                    );
                    let has_ok_arm = arms.iter().any(|arm| {
                        match &ast.pattern(arm.pattern).node {
                            Pattern::Constructor { name, .. } => *name == crate::ir::Compute::CTOR_OK,
                            Pattern::Variable { name } => *name == crate::ir::Compute::CTOR_OK,
                            _ => false,
                        }
                    });
                    if !is_throw && has_ok_arm {
                        // Check whether the scrutinee implements the Err trait (via the witness table).
                        let implements_err = self.arena.type_name(resolved_scrutinee)
                            .and_then(|tn| self.sema_result.type_def_index.get(tn).copied())
                            .map(|idx| self.witness_table.implements("Err", dynamic_type_id(idx)))
                            .unwrap_or(false);
                        let widened = if implements_err {
                            // scrutinee is an error type → Throw<fresh_val, scrutinee>.
                            let fresh_val = self.arena.fresh_type_var();
                            self.arena.make_throw(fresh_val, resolved_scrutinee)
                        } else {
                            // scrutinee is a value type → Throw<scrutinee, fresh_err>.
                            let fresh_err = self.arena.fresh_type_var();
                            self.arena.make_throw(resolved_scrutinee, fresh_err)
                        };
                        self.unify_or_constrain(scrutinee_ty, widened);
                        self.arena.resolve(widened)
                    } else {
                        resolved_scrutinee
                    }
                };

                // sema v2: extract the scrutinee's path (used for ConstructorMatch narrowing).
                let scrutinee_path = expr_path(ast, *scrutinee);

                let mut arm_tys: Vec<TypeHandle> = Vec::new();
                for arm in arms.iter() {
                    let child_env = self.env.child(env);

                    // sema v2: enter the match-arm scope and apply ConstructorMatch narrowing.
                    self.flow_ctx.push_scope();
                    if let Some(ref path) = scrutinee_path {
                        // Check whether this is a constructor pattern; if so, add a ConstructorMatch fact.
                        if let Some((ctor_name, bound_vars)) =
                            extract_constructor_pattern(&ast.pattern(arm.pattern).node, ast)
                        {
                            // Constructor match: the scrutinee is narrowed to this constructor's type.
                            let narrowed_ty = self.arena.make_adt(
                                ctor_name.into(),
                                Box::new([]),
                            );
                            self.flow_ctx.add_fact(FlowFact {
                                path: path.clone().into(),
                                narrowed_ty,
                                kind: NarrowKind::ConstructorMatch {
                                    ctor_name: ctor_name.into(),
                                    bound_vars: bound_vars.into(),
                                },
                            });
                        }
                    }

                    self.infer_pattern(arm.pattern, ast, resolved_scrutinee, child_env);
                    if let Some(guard) = arm.guard {
                        let _ = self.infer_expr(guard, ast, child_env, None);
                    }
                    // Propagate the match's expected type to the arm body,
                    // so expressions that depend on the expected constraint (e.g. NullLit) infer correctly.
                    let body_ty = self.infer_expr(arm.body, ast, child_env, expected);
                    self.flow_ctx.pop_scope();

                    arm_tys.push(body_ty);
                }

                // v2 convergence: use only peer_type to unify all arm types (eliminates the per-arm widen dual-track scheme).
                // peer_type handles single-arm (return directly), multi-arm (join), and all-Never/Void (return Never/Void).
                let result_ty = if arm_tys.is_empty() {
                    self.make_builtin(Type::Void)
                } else {
                    peer_type(self.arena, &arm_tys)
                };
                // Constrain the match result against the outer expected type, so that when the match is the RHS of a let,
                // pending TypeVars in the result type can be solved in reverse.
                if let Some(exp) = expected {
                    self.unify_or_constrain(result_ty, exp);
                }

                // ── Exhaustiveness check ──
                // A match is exhaustive if it has a wildcard `_` or variable-binding arm
                // (without guard). For ADT scrutinees, check that all constructors are covered.
                self.check_match_exhaustive(ast, *scrutinee, resolved_scrutinee, arms);

                result_ty
            }
            _ => unreachable!("infer_match_expr called on non-Match expression"),
        }
    }

    /// Infer an `Expr::InlineTrait` expression (extracted from `infer_expr_inner`).
    ///
    /// Obtain the trait name from the expected type (the val_decl's type annotation),
    /// verify method completeness, and produce TraitObject { trait_name, method_sigs }.
    /// With no expected type, report an error and return a fresh_type_var (an inline_trait without an annotation is not allowed).
    fn infer_inline_trait_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::InlineTrait(methods) => {
                // Obtain the trait name from the expected type.
                let trait_name: Option<Box<str>> = if let Some(exp) = expected {
                    let resolved = self.arena.resolve(exp);
                    match self.arena.get(resolved) {
                        Type::Trait(_) => {
                            let (name, _) = self.arena.trait_parts(resolved);
                            Some(name.into())
                        }
                        Type::TraitObject(_) => {
                            let (trait_name, _) = self.arena.trait_object_parts(resolved);
                            Some(trait_name.into())
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                // Collect the inline_trait's method signatures.
                let method_sigs: Vec<TraitMethodSig> = methods
                    .iter()
                    .map(|m| {
                        let return_type = match m.return_type {
                            Some(rt) => self.type_from_ast(rt, ast),
                            None => self.arena.make(Type::Void),
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

                if let Some(tname) = trait_name {
                    // Verify method completeness: every required method (without a body) in trait_def must appear in the inline_trait.
                    let missing: Vec<String> = if let Some(trait_def) = self.sema_result.get_trait_def(&tname) {
                        trait_def
                            .methods
                            .iter()
                            .filter(|req| !req.has_body)
                            .filter(|req| {
                                !method_sigs
                                    .iter()
                                    .any(|m| m.name == req.name && m.param_count == req.param_count)
                            })
                            .map(|req| {
                                format!(
                                    "inline_trait missing required method {} of trait {} (param count {})",
                                    tname, req.name, req.param_count
                                )
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let span = ast.expr(expr).span;
                    for msg in missing {
                        self.sema_result.errors.push(SemaError::new(&msg, span.line, span.column));
                    }

                    // Type-check each method body: bind types for parameters (use the annotation if present, otherwise a fresh_type_var),
                    // set expected_return, and call infer_expr to populate expr_types for sub-expressions inside the body.
                    // This is the data source for IR-compile-time type queries (e.g. str + str → concat).
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            let method_env = self.env.child(env);
                            for param in m.params.iter() {
                                let param_ty = match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                };
                                self.env.define(method_env, param.name, param_ty);
                            }
                            let prev_return = self.expected_return;
                            self.expected_return =
                                m.return_type.map(|rt| self.type_from_ast(rt, ast));
                            let _ = self.infer_expr(body, ast, method_env, self.expected_return);
                            self.expected_return = prev_return;
                        }
                    }

                    self.arena.make_trait_object(
                        tname,
                        method_sigs.into_boxed_slice(),
                    )
                } else {
                    let span = ast.expr(expr).span;
                    self.sema_result.errors.push(SemaError::new(
                        "inline_trait cannot infer trait name: explicit type annotation required",
                        span.line,
                        span.column,
                    ));
                    self.arena.fresh_type_var()
                }
            }
            _ => unreachable!("infer_inline_trait_expr called on non-InlineTrait expression"),
        }
    }

    /// Infer an `Expr::Call` expression (extracted from `infer_expr_inner`).
    fn infer_call_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Call { callee, args, type_args } => {
                // cast call resolution: __cast_to<T>(x) / __cast_try_to<T>(x).
                // The parser lowers cast(x).to(T) into an ordinary __cast_to<T>(x) Call;
                // sema infers the source type S and returns T (or Throw<T, CastError> for try_to).
                // Lookup goes through the CAST_BUILTINS registry, avoiding name-specific branches.
                if let Expr::Ident(name) = &ast.expr(*callee).node {
                    if let Some(is_try) = CAST_BUILTINS
                        .iter()
                        .find_map(|(n, t)| (*n == *name).then_some(*t))
                    {
                        // Infer the source expression's type.
                        let _ = self.infer_expr(args[0], ast, env, None);
                        // Take the target type T from type_args.
                        let target_ty = match type_args {
                            Some(ta) if !ta.is_empty() => self.type_from_ast(ta[0], ast),
                            _ => self.arena.fresh_type_var(),
                        };
                        if is_try {
                            let err_ty = self.arena.make_adt(
                                "CastError".into(),
                                Box::new([]),
                            );
                            return self.arena.make_throw(target_ty, err_ty);
                        }
                        return target_ty;
                    }
                }

                // ── Constructor multi-mapping disambiguation ──
                // When callee is an Ident that maps to multiple same-named constructors, disambiguate by priority:
                //   1. Type-oriented: when expected_ty is an Adt, select by type_name
                //   2. Arity: when type-oriented disambiguation fails (expected is a TypeVar or not provided),
                //      select the unique constructor matching by arity
                let callee_ty = if let Expr::Ident(name) = &ast.expr(*callee).node {
                    let ctors = self.sema_result.get_ctor_defs(name);
                    if ctors.len() > 1 {
                        let selected: Option<(Box<str>, Box<[TypeRepr]>)> = {
                            let mut found: Option<&CtorDefInfo> = None;
                            // 1. Type-oriented disambiguation
                            if let Some(exp) = expected {
                                let exp_resolved = self.arena.resolve(exp);
                                if let Type::Adt(_) = self.arena.get(exp_resolved) {
                                    let (exp_type_name, _) = self.arena.adt_parts(exp_resolved);
                                    let matches: Vec<_> = ctors.iter()
                                        .filter(|c| c.type_name.as_ref() == exp_type_name)
                                        .collect();
                                    if matches.len() == 1 {
                                        found = Some(matches[0]);
                                    }
                                }
                            }
                            // 2. Arity disambiguation (fallback when type-oriented fails)
                            if found.is_none() {
                                let arity_matches: Vec<_> = ctors.iter()
                                    .filter(|c| c.field_type_reprs.len() == args.len())
                                    .collect();
                                if arity_matches.len() == 1 {
                                    found = Some(arity_matches[0]);
                                }
                            }
                            found.map(|c| (c.type_name.clone(), c.field_type_reprs.clone()))
                        };
                        match selected {
                            Some((type_name, field_type_reprs)) => {
                                let param_types: Vec<TypeHandle> = field_type_reprs
                                    .iter()
                                    .map(|r| self.type_repr_to_handle(r))
                                    .collect();
                                let ret_ty = self.arena.make_adt(type_name, Box::new([]));
                                if param_types.is_empty() {
                                    ret_ty
                                } else {
                                    self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
                                }
                            }
                            None => {
                                let span = ast.expr(expr).span;
                                let type_names: Vec<&str> = ctors.iter()
                                    .map(|c| c.type_name.as_ref())
                                    .collect();
                                self.add_error_at(
                                    &format!(
                                        "ambiguous constructor '{}': defined by types [{}]; use Type.{} to disambiguate or provide a type context",
                                        name,
                                        type_names.join(", "),
                                        name,
                                    ),
                                    span.line,
                                    span.column,
                                );
                                self.arena.fresh_type_var()
                            }
                        }
                    } else if ctors.len() == 1
                        && args.is_empty()
                        && ctors[0].field_type_reprs.is_empty()
                    {
                        // Bug #69: Zero-arg constructor called with `()` syntax.
                        // Zero-arg constructors are registered as values (ADT type), not
                        // function types, so `Unit()` is equivalent to the bare value `Unit`.
                        let ret_ty = self.arena.make_adt(
                            ctors[0].type_name.clone(),
                            Box::new([]),
                        );
                        if let Some(exp) = expected {
                            self.unify_or_constrain(ret_ty, exp);
                        }
                        ret_ty
                    } else {
                        // [Implicit this] Try resolving as this.method(args) before
                        // falling through to infer_expr (which would report undefined).
                        if let Some(this_ty) = self.current_this_type() {
                            if let Some(fn_ty) = self.lookup_method_type(this_ty, name) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Type::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    // Skip params[0] (this), match args with params[1..].
                                    let n = params.len().min(args.len() + 1);
                                    for i in 1..n {
                                        let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                                        self.unify_or_constrain(params[i], arg_ty);
                                    }
                                    // Store callee's ExprInfo so that pending_implicit_this
                                    // (flushed in infer_expr) can attach the implicit_this marker.
                                    // Without this, the marker is lost because we bypass
                                    // infer_expr(callee) on this fast path.
                                    self.store_expr_info(*callee, fn_ty);
                                    self.pending_implicit_this = Some((
                                        *callee,
                                        crate::sema::Sema::ImplicitThisAccess::Method((*name).to_string().into_boxed_str()),
                                    ));
                                    return return_type;
                                }
                            }
                        }
                        self.infer_expr(*callee, ast, env, None)
                    }
                } else {
                    self.infer_expr(*callee, ast, env, None)
                };
                let resolved_callee = self.arena.resolve(callee_ty);

                // Instantiation mode: skip HM unify (types were already checked in the sema HM stage);
                // only infer argument types and return the return type. Monomorphization triggers are orchestrated externally.
                if self.instantiation_ctx.is_some() {
                    // ModuleRef call: look up the function signature from the module env.
                    if let Type::ModuleRef(_) = self.arena.get(resolved_callee) {
                        let (path, module_env) = self.arena.module_ref_parts(resolved_callee);
                        if let Some(func_name) = path.rsplit('.').next() {
                            if let Some(fn_ty) = self.env.lookup_local(module_env, func_name) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Type::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                        let _ = self.infer_expr(arg, ast, env, Some(param_ty));
                                    }
                                    return return_type;
                                }
                            }
                        }
                    }
                    // Ordinary function call: infer argument types and return the return type.
                    let inst_callee = self.instantiate_fn_type(resolved_callee);
                    if let Type::Fn(_) = self.arena.get(inst_callee) {
                        let (params, return_type) = self.arena.fn_parts(inst_callee);
                        let params: Vec<TypeHandle> = params.to_vec();
                        for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                            let _ = self.infer_expr(arg, ast, env, Some(param_ty));
                        }
                        return return_type;
                    }
                    // Non-Fn callee: report an error and return Unknown.
                    let span = ast.expr(expr).span;
                    let callee_name = self
                        .arena
                        .type_name(resolved_callee)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("{:?}", self.arena.get(resolved_callee)));
                    self.add_error_at(
                        &format!("cannot call non-function value of type '{}'", callee_name),
                        span.line,
                        span.column,
                    );
                    for &a in args.iter() {
                        let _ = self.infer_expr(a, ast, env, None);
                    }
                    return self.arena.make(Type::Unknown);
                }

                // ModuleRef call: callee is a module path reference (e.g. "std.reflect.Reflect.format");
                // look up the function signature by its trailing bare name directly in the module env carried by the ModuleRef (no parent-env traversal).
                if let Type::ModuleRef(_) = self.arena.get(resolved_callee) {
                    let (path, module_env) = self.arena.module_ref_parts(resolved_callee);
                    // The trailing segment is the function name (e.g. "std.reflect.Reflect.format" → "format").
                    if let Some(func_name) = path.rsplit('.').next() {
                        if let Some(fn_ty) = self.env.lookup_local(module_env, func_name) {
                            // Instantiate the polymorphic function type to avoid type-constraint clashes across calls.
                            let inst_fn = self.instantiate_fn_type(fn_ty);
                            if let Type::Fn(_) = self.arena.get(inst_fn) {
                                let (params, return_type) = self.arena.fn_parts(inst_fn);
                                let params: Vec<TypeHandle> = params.to_vec();
                                if params.len() == args.len() {
                                    for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                        let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                                        if let Err(e) = self.try_widen_unify(param_ty, arg_ty) {
                                            self.add_error(&format!("argument type incompatible with parameter type: {}", e));
                                        }
                                    }
                                    return return_type;
                                }
                            }
                        }
                    }
                }

                // Instantiate the polymorphic function type (replace rigid vars / unbound TypeVars with fresh non-rigid vars)
                // so each call has its own type variables, avoiding type-constraint clashes across calls.
                let inst_callee = self.instantiate_fn_type(resolved_callee);
                if let Type::Fn(_) = self.arena.get(inst_callee) {
                    let (params, return_type) = self.arena.fn_parts(inst_callee);
                    let params: Vec<TypeHandle> = params.to_vec();
                    if params.len() == args.len() {
                        for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                            let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                            // On unify failure, register a constraint (rather than discarding) so the fixpoint iteration can solve the argument type.
                            self.unify_or_constrain(param_ty, arg_ty);
                        }
                    }
                    // Always return the declared return type, to avoid cascading type loss from argument mismatches.
                    // If there is an expected type, unify the return type with it to solve pending TypeVars in the return type
                    // (e.g. Ok(void) returns Throw<void, '_E>; expected=Throw<void, IOError> solves E=IOError).
                    if let Some(exp) = expected {
                        self.unify_or_constrain(return_type, exp);
                    }
                    return return_type;
                }
                // Fallback: infer all arguments and unify the callee with (args -> ret).
                let ret_ty = self.arena.fresh_type_var();
                let arg_types: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.infer_expr(a, ast, env, None))
                    .collect();
                let expected_fn = self.arena.make_fn(
                    arg_types.into_boxed_slice(),
                    ret_ty,
                );
                self.unify_or_constrain(callee_ty, expected_fn);
                ret_ty
            }
            _ => unreachable!("infer_call_expr called on non-Call expression"),
        }
    }

    /// Infer an `Expr::MethodCall` / `Expr::SafeMethodCall` expression (extracted from `infer_expr_inner`).
    fn infer_method_call_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::MethodCall { recv, method, args, .. }
            | Expr::SafeMethodCall { recv, method, args, .. } => {
                // Qualified-name syntax: Type.Ctor(args) (qualified call of a constructor with arguments)
                if let Expr::Ident(type_name) = &ast.expr(*recv).node {
                    if let Some((ctor_type_name, field_type_reprs)) =
                        self.check_qualified_ctor(type_name, method)
                    {
                        if !field_type_reprs.is_empty() {
                            // Constructor with arguments: build a function type and go through call inference
                            let param_types: Vec<TypeHandle> = field_type_reprs
                                .iter()
                                .map(|r| self.type_repr_to_handle(r))
                                .collect();
                            let ret_ty = self.arena.make_adt(ctor_type_name, Box::new([]));
                            let fn_ty = self.arena.make_fn(
                                param_types.into_boxed_slice(),
                                ret_ty,
                            );
                            let (params, return_type) = self.arena.fn_parts(fn_ty);
                            let params: Vec<TypeHandle> = params.to_vec();
                            if params.len() == args.len() {
                                for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                    let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                                    self.unify_or_constrain(param_ty, arg_ty);
                                }
                            }
                            if let Some(exp) = expected {
                                self.unify_or_constrain(return_type, exp);
                            }
                            // Mark recv as module-func-recv (skip recv during IR compilation)
                            let recv_key = crate::sema::Sema::module_expr_key(
                                &self.current_module_name,
                                recv.0 as u64,
                            );
                            self.sema_result.module_func_recv_exprs.insert(recv_key);
                            return return_type;
                        }
                        // Zero-argument constructor in MethodCall: report an error
                        let span = ast.expr(expr).span;
                        self.add_error_at(
                            &format!(
                                "constructor '{}' of type '{}' takes no arguments; use {}.{} syntax",
                                method, type_name, type_name, method
                            ),
                            span.line,
                            span.column,
                        );
                        return self.arena.fresh_type_var();
                    }
                }

                let recv_ty = self.infer_expr(*recv, ast, env, None);

                // Path 0a: ModuleRef recv → module-path function call.
                // When recv is a ModuleRef (e.g. std.net.UdpSocket), method is a top-level function in that module;
                // look it up by its bare name directly in the module env carried by the ModuleRef (no parent-env traversal).
                let recv_resolved_0a = self.arena.resolve(recv_ty);
                if let Type::ModuleRef(_) = self.arena.get(recv_resolved_0a) {
                    let (mod_path, module_env) = self.arena.module_ref_parts(recv_resolved_0a);
                    let found = self.env.lookup_local(module_env, method);
                    // Directory-module semantics: when lookup_local misses in the current module env,
                    // search sibling modules in the same directory (e.g. Math.sqrt where sqrt lives in Power.kz,
                    // with Math and Power both under the std.math directory).
                    let found = found.or_else(|| {
                        self.lookup_sibling_module_fn(mod_path, module_env, method)
                    });
                    if let Some(fn_ty) = found {
                        let inst_fn = self.instantiate_fn_type(fn_ty);
                        if let Type::Fn(_) = self.arena.get(inst_fn) {
                            let (params, return_type) = self.arena.fn_parts(inst_fn);
                            let params: Vec<TypeHandle> = params.to_vec();
                            let n = params.len().min(args.len());
                            for i in 0..n {
                                let arg_ty = self.infer_expr(args[i], ast, env, Some(params[i]));
                                self.unify_or_constrain(params[i], arg_ty);
                            }
                            // Mark recv as a module-function-call receiver so IR compilation does not pass recv.
                            // (Consistent with path 0b: ModuleRef recv has Module.fun(args) semantics.)
                            let recv_key = module_expr_key(
                                &self.current_module_name,
                                recv.0 as u64,
                            );
                            self.sema_result.module_func_recv_exprs.insert(recv_key);
                            return return_type;
                        }
                    }
                }

                // Path 0b: constructor recv (type name == module name) → module function call (Zig-style @This semantics).
                // When recv is a type constructor (Fn, with return_type Adt) and the type name matches a module name,
                // look up free functions by the method's bare name in that module's env.
                // Typical scenario: after `import std.time.Duration`, Duration.from_millis(100),
                // where Duration is both a type and a module (file with the same name; predefine redefine overwrote the ModuleRef).
                if let Type::Fn(_) = self.arena.get(recv_resolved_0a) {
                    let (_, ret_ty) = self.arena.fn_parts(recv_resolved_0a);
                    let ret_resolved = self.arena.resolve(ret_ty);
                    if let Type::Adt(_) = self.arena.get(ret_resolved) {
                        let (type_name, _) = self.arena.adt_parts(ret_resolved);
                        if let Some(&mod_env) = self.ctor_module_envs.get(type_name) {
                            if let Some(fn_ty) = self.env.lookup_local(mod_env, method) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Type::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    let n = params.len().min(args.len());
                                    for i in 0..n {
                                        let arg_ty = self.infer_expr(args[i], ast, env, Some(params[i]));
                                        self.unify_or_constrain(params[i], arg_ty);
                                    }
                                    // Mark recv as a module-function-call receiver so IR compilation does not pass recv.
                                    let recv_key = module_expr_key(
                                        &self.current_module_name,
                                        recv.0 as u64,
                                    );
                                    self.sema_result.module_func_recv_exprs.insert(recv_key);
                                    return return_type;
                                }
                            }
                        }
                    }
                }

                // Language-level intrinsic tagging: await/recv are recognized uniformly by sema
                // and registered into method_dispatches for IR consumption (eliminates IR-side string guards).
                // await is a general suspend semantic (for all types); recv is tagged only for Channel/Receiver types.
                {
                    let intrinsic = if *method == "await" && args.is_empty() {
                        Some(crate::sema::Sema::IntrinsicKind::Await)
                    } else if *method == "recv" && args.is_empty() {
                        match self.arena.get(recv_resolved_0a) {
                            Type::Channel(_) | Type::Receiver(_) => {
                                Some(crate::sema::Sema::IntrinsicKind::ChannelAwait)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if intrinsic.is_some() {
                        let key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            expr.0 as u64,
                        );
                        self.sema_result.method_dispatches.insert(
                            key,
                            crate::sema::Sema::DispatchInfo {
                                trait_id: 0,
                                method_idx: 0,
                                impl_fn_idx: 0,
                                instance_id: 0,
                                intrinsic,
                            },
                        );
                    }
                }

                // Path 1 (preferred): type-aware method lookup.
                // lookup_method_type looks up the receiver's type against witness_table / func_sigs / builtin methods,
                // ensuring same-named methods (e.g. Instant.add_duration vs DateTime.add_duration) dispatch to the correct signature.
                let method_fn_ty = self.lookup_method_type(recv_ty, method);
                if let Some(fn_ty) = method_fn_ty {
                    let inst_fn = self.instantiate_fn_type(fn_ty);
                    if let Type::Fn(_) = self.arena.get(inst_fn) {
                        let (params, return_type) = self.arena.fn_parts(inst_fn);
                        let params: Vec<TypeHandle> = params.to_vec();
                        // The first parameter is self; skip it.
                        let n = params.len().min(args.len() + 1);
                        for i in 1..n {
                            let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                            self.unify_or_constrain(params[i], arg_ty);
                        }
                        return return_type;
                    }
                }

                // Path 0 (fallback): look up a binding named after the method as an Fn type in env (free function with a self parameter).
                // Use lookup_with_pred to skip same-named non-function bindings (e.g. a local variable shadowing a free function).
                // In Kuzo `recv.method(args)` is sugar for `method(recv, args)`.
                if let Some(fn_ty) = self.env.lookup_with_pred(env, method, |ty| {
                    let r = self.arena.resolve(ty);
                    matches!(self.arena.get(r), Type::Fn(_))
                }) {
                    let inst_fn = self.instantiate_fn_type(fn_ty);
                    if let Type::Fn(_) = self.arena.get(inst_fn) {
                        let (params, return_type) = self.arena.fn_parts(inst_fn);
                        let params: Vec<TypeHandle> = params.to_vec();
                        // The first parameter is self/receiver: unify recv with params[0].
                        // This lets the free function's generic parameters be inferred from the receiver's type (e.g. iter<T> infers T from arr: T[]).
                        if !params.is_empty() {
                            self.unify_or_constrain(params[0], recv_ty);
                        }
                        // The remaining parameters are inferred from args.
                        let n = params.len().min(args.len() + 1);
                        for i in 1..n {
                            let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                            self.unify_or_constrain(params[i], arg_ty);
                        }
                        return return_type;
                    }
                }

                // await is a general suspend semantic: it produces no value; it only suspends the frame waiting for an event.
                // The IR layer uses infer_event_source_kind to decide the event-source kind based on the recv type
                // (AsyncJoin/Channel/Timer); the Sema layer uniformly returns void.
                if *method == "await" && args.is_empty() {
                    return self.make_builtin(Type::Void);
                }

                // Fallback: infer arguments and return a fresh var.
                // For a receiver whose type is already determined (not TypeVar/Unknown/Never), report "method does not exist"
                // to help the user locate the problem; for a TypeVar receiver, silently return a fresh var (inference pending, deferred to the solver).
                let span = ast.expr(expr).span;
                let recv_resolved = self.arena.resolve(recv_ty);
                match self.arena.get(recv_resolved) {
                    Type::TypeVar(_) | Type::Unknown | Type::Never => {
                        // Receiver type pending; silently return a fresh var.
                    }
                    Type::Void => {
                        // void receiver: handled by the IR layer (void method call).
                    }
                    ct => {
                        // Receiver type is determined but method lookup failed: report an error.
                        let recv_name = self.arena.type_name(recv_resolved)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("{:?}", ct));
                        self.add_error_at(
                            &format!("no method '{}' on type '{}'", method, recv_name),
                            span.line,
                            span.column,
                        );
                    }
                }
                for &a in args.iter() {
                    let _ = self.infer_expr(a, ast, env, None);
                }
                self.arena.fresh_type_var()
            }
            _ => unreachable!("infer_method_call_expr called on non-MethodCall expression"),
        }
    }

    /// Integer suffix → corresponding integer TypeHandle (derived from `BUILTIN_TABLE`; returns `None` on miss).
    fn int_suffix_to_type(&mut self, suffix: &str) -> Option<TypeHandle> {
        let tag = crate::types::ValueTag::from_name(suffix)?;
        if tag.is_int() {
            Some(self.arena.from_scalar_name(suffix))
        } else {
            None
        }
    }

    /// Float suffix → corresponding float TypeHandle (derived from `BUILTIN_TABLE`; returns `None` on miss).
    fn float_suffix_to_type(&mut self, suffix: &str) -> Option<TypeHandle> {
        let tag = crate::types::ValueTag::from_name(suffix)?;
        if tag.is_float() {
            Some(self.arena.from_scalar_name(suffix))
        } else {
            None
        }
    }

    /// Builds a Type::Fn type from the owned data of a MethodSigInfo.
    /// Both parameters and the return type are fully resolved from TypeRepr via type_repr_to_handle,
    /// correctly handling nested generics (e.g. Async<Throw<T, E>>), arrays, Nullable, and other compound types,
    /// overcoming the limitation that type_name only stores the top-level name.
    fn build_fn_type_from_sig(
        &mut self,
        param_type_reprs: Vec<TypeRepr>,
        return_type_repr: Option<TypeRepr>,
        _recv_ty: TypeHandle,
    ) -> TypeHandle {
        // ThisType is resolved by type_repr_to_handle via current_this_type();
        // the caller (lookup_method_type) has already pushed recv_ty as self_type.
        let params: Vec<TypeHandle> = param_type_reprs
            .iter()
            .map(|repr| self.type_repr_to_handle(repr))
            .collect();
        let return_type = match return_type_repr {
            Some(repr) => self.type_repr_to_handle(&repr),
            None => self.arena.fresh_type_var(),
        };
        self.arena.make_fn(params.into_boxed_slice(), return_type)
    }

    /// Constructs a TypeHandle from a self-contained TypeRepr (does not depend on AstArena references).
    /// Mirrors the logic of type_from_ast_with_params, but reads from TypeRepr instead of AST TypeNode.
    /// Used to restore cross-module method return types (MethodSigInfo.return_type_repr).
    fn type_repr_to_handle(&mut self, repr: &TypeRepr) -> TypeHandle {
        match repr {
            TypeRepr::Named(name) => {
                let empty_map: FxHashMap<String, TypeHandle> = FxHashMap::default();
                let mut visiting = FxHashSet::default();
                self.resolve_name_to_type(name.as_ref(), &empty_map, &mut visiting)
            }
            TypeRepr::ThisType => match self.current_this_type() {
                Some(ty) => ty,
                None => self.arena.fresh_type_var(),
            },
            TypeRepr::Generic(name, args) => {
                let new_args: Vec<TypeHandle> =
                    args.iter().map(|a| self.type_repr_to_handle(a)).collect();
                let args_box: Box<[TypeHandle]> = new_args.into_boxed_slice();

                // Builtin generic types (Throw/Atomic/Async/Channel, etc.) construct dedicated Type variants.
                if is_builtin_generic_type(name) {
                    return self.make_builtin_generic(name.clone(), args_box);
                }
                // trait definition → Trait type.
                if self.sema_result.get_trait_def(name).is_some() {
                    return self.arena.make_trait(name.clone(), args_box);
                }
                // User-defined generic ADT.
                let has_type_params = self
                    .sema_result
                    .get_type_def(name)
                    .map(|d| !d.type_params.is_empty())
                    .unwrap_or(false);
                if has_type_params {
                    return self.arena.make_adt(name.clone(), args_box);
                }
                // Fallback: construct a Generic (may be undefined or a forward reference; reported on later use).
                self.arena.make_generic(name.clone(), args_box)
            }
            TypeRepr::Nullable(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_nullable(inner_ty)
            }
            TypeRepr::Ref(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_ref(inner_ty, false)
            }
            TypeRepr::RawPtr(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_ref(inner_ty, true)
            }
            TypeRepr::Function(params, return_type) => {
                let p: Vec<TypeHandle> =
                    params.iter().map(|a| self.type_repr_to_handle(a)).collect();
                let r = self.type_repr_to_handle(return_type);
                self.arena.make_fn(p.into_boxed_slice(), r)
            }
            TypeRepr::Array(elem, _) => {
                let elem_ty = self.type_repr_to_handle(elem);
                self.arena.make_array(elem_ty, None)
            }
        }
    }

    /// Looks up the method signature for an object type (returns a function type whose first parameter is self).
    fn lookup_method_type(
        &mut self,
        recv_ty: TypeHandle,
        method: &str,
    ) -> Option<TypeHandle> {
        let resolved = self.arena.resolve(recv_ty);

        // ── Receiver normalization ──
        // Wrapper types (Nullable/Ref) recursively forward method lookup to the inner type,
        // so calls like s?.len() / (&arr).len() auto-unwrap to the correct method table.
        // Nullable's own methods (is_null) are handled via the unified TypeDefInfo path and are not forwarded.
        match self.arena.get(resolved) {
            Type::Nullable(_) => {
                // Nullable's own method (is_null) goes through the TypeDefInfo("nullable") path;
                // other methods are recursively forwarded to the inner type.
                if method != "is_null" {
                    let inner = self.arena.nullable_inner(resolved);
                    return self.lookup_method_type(inner, method);
                }
            }
            Type::Ref(_) => {
                // Ref auto-deref: method lookup on &T forwards to T.
                let inner = self.arena.ref_parts(resolved).0;
                return self.lookup_method_type(inner, method);
            }
            _ => {}
        }

        // Push recv_ty as the Self type so that, inside build_fn_type_from_sig,
        // type_repr_to_handle(ThisType) resolves to the receiver type correctly,
        // without special-casing the first parameter by position.
        self.push_this_type(resolved);

        // Generic type-parameter binding: bind the type definition's type-parameter names (e.g. "T") to the concrete
        // type arguments in the receiver type, so that T in a method signature (e.g. `pub fun next(&self): T?`)
        // is resolved via type_binding_stack to the corresponding type argument in the receiver,
        // rather than producing an orphan fresh_type_var.
        //
        // Handles Adt (user-defined generics) and builtin types (Array/Nullable/Throw/Generic) uniformly,
        // so that generic parameters in builtin-type method signatures also bind correctly.
        let mut pushed_bindings = false;
        let builtin_bindings: Option<(Box<str>, Vec<TypeHandle>)> = match self.arena.get(resolved) {
            Type::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                Some((name.into(), type_args.to_vec()))
            }
            Type::Array(_) => {
                let (element_type, _) = self.arena.array_parts(resolved);
                Some(("array".into(), vec![element_type]))
            }
            Type::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                Some(("nullable".into(), vec![inner]))
            }
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                Some(("Throw".into(), vec![value_type, error_type]))
            }
            // Builtin generic dedicated variants: extract element/value types as type_args bindings.
            Type::Channel(_) => Some(("Channel".into(), vec![self.arena.channel_elem(resolved)])),
            Type::Async(_) => Some(("Async".into(), vec![self.arena.async_value(resolved)])),
            Type::Lazy(_) => Some(("Lazy".into(), vec![self.arena.lazy_value(resolved)])),
            Type::Atomic(_) => Some(("Atomic".into(), vec![self.arena.atomic_elem(resolved)])),
            Type::Sender(_) => Some(("Sender".into(), vec![self.arena.sender_elem(resolved)])),
            Type::Receiver(_) => Some(("Receiver".into(), vec![self.arena.receiver_elem(resolved)])),
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                Some((name.into(), args.to_vec()))
            }
            _ => None,
        };
        if let Some((type_name, actual_args)) = builtin_bindings {
            if let Some(def) = self.sema_result.get_type_def(type_name.as_ref()) {
                if !def.type_params.is_empty() && def.type_params.len() == actual_args.len() {
                    self.type_binding_stack.push();
                    for (pname, &arg) in def.type_params.iter().zip(actual_args.iter()) {
                        self.type_binding_stack.insert_top(pname.as_ref(), arg);
                    }
                    pushed_bindings = true;
                }
            }
        }

        let result = self.lookup_method_type_inner(resolved, method);
        if pushed_bindings {
            self.pop_type_bindings();
        }
        self.pop_this_type();
        result
    }

    fn lookup_method_type_inner(
        &mut self,
        resolved: TypeHandle,
        method: &str,
    ) -> Option<TypeHandle> {
        match self.arena.get(resolved) {
            Type::Trait(_) => {
                let (name, _) = self.arena.trait_parts(resolved);
                // For a trait type (e.g. l: Logger), look up trait_def.methods directly to restore the method signature;
                // parameters use fresh_type_var (a trait method's exact parameter types are determined by the implementing type).
                if let Some(td) = self.sema_result.get_trait_def(name) {
                    if let Some(sig) = td.methods.iter().find(|m| m.name.as_ref() == method) {
                        // params[0] is self, bound to the receiver type (resolved) to avoid producing an orphan TypeVar;
                        // the remaining parameters still use fresh_type_var (exact types are determined by the implementing type).
                        let params: Vec<TypeHandle> = (0..sig.param_count)
                            .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                            .collect();
                        let return_type = sig.return_type;
                        return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                    }
                }
            }
            Type::TypeVar(idx) => {
                // Inside a trait default method, current_this_type() is a rigid TypeVar
                // representing the (unknown) implementing type. Method lookup must consult
                // the current trait's method signatures rather than the receiver's (nonexistent)
                // method table. This enables bare `method()` calls inside trait default bodies.
                if self.arena.type_vars[idx as usize].is_rigid {
                    if let Some(ref trait_name) = self.current_trait_name {
                        if let Some(td) = self.sema_result.get_trait_def(trait_name) {
                            if let Some(sig) = td.methods.iter().find(|m| m.name.as_ref() == method) {
                                let params: Vec<TypeHandle> = (0..sig.param_count)
                                    .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                                    .collect();
                                let return_type = sig.return_type;
                                return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // Builtin type-name mapping: Array/Nullable/Throw are structural variants where arena.type_name returns None
        // or recurses to the inner type. Method lookup needs a unified type name to query type_def_index;
        // map them here to the synthetic TypeDefInfo names used at registration.
        let type_name: Option<String> = match self.arena.get(resolved) {
            Type::Array(_) => Some("array".to_string()),
            Type::Str => Some("str".to_string()),
            Type::Nullable(_) => Some("nullable".to_string()),
            Type::Throw(_) => Some("Throw".to_string()),
            _ => self.arena.type_name(resolved).map(|s| s.to_string()),
        };

        // v2 convergence: path 1 — query witness_table (trait method dispatch, indexed by type_id).
        if let Some(ref name) = type_name {
            let type_id = self
                .sema_result
                .type_def_index
                .get(name.as_str())
                .map(|&idx| dynamic_type_id(idx));
            if let Some(tid) = type_id {
                for entry in self.witness_table.entries() {
                    if entry.type_id != tid {
                        continue;
                    }
                    // Get the signature from TypeDefInfo.methods (looked up by method_name).
                    // Extract owned data to release the sema_result borrow.
                    let sig_data: Option<(Vec<TypeRepr>, Option<TypeRepr>)> =
                        if let Some(&type_idx) = self.sema_result.type_def_index.get(name.as_str()) {
                            self.sema_result.type_defs[&type_idx]
                                .methods
                                .iter()
                                .find(|m| m.name.as_ref() == method)
                                .map(|sig| (sig.param_type_reprs.to_vec(), sig.return_type_repr.clone()))
                        } else {
                            None
                        };
                    if let Some((param_type_reprs, return_type_repr)) = sig_data {
                        return Some(self.build_fn_type_from_sig(param_type_reprs, return_type_repr, resolved));
                    }
                    // TypeDefInfo.methods miss → query trait_def.methods (trait default methods).
                    // When a type implements a trait via `type T: Trait = ...` without overriding a method,
                    // method_slots is empty and the method signature is obtained from trait_def.
                    let trait_sig_data: Option<(u8, TypeHandle)> =
                        self.sema_result
                            .get_trait_def(entry.trait_name.as_ref())
                            .and_then(|td| {
                                td.methods
                                    .iter()
                                    .find(|m| m.name.as_ref() == method)
                                    .map(|m| (m.param_count, m.return_type))
                            });
                    if let Some((param_count, return_type)) = trait_sig_data {
                        // params[0] is self, bound to the receiver type (resolved) to avoid producing an orphan TypeVar;
                        // the remaining parameters use fresh_type_var (exact types are determined by the implementing type).
                        let params: Vec<TypeHandle> = (0..param_count)
                            .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                            .collect();
                        return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                    }
                    // The current trait does not have this method; continue checking other trait implementations.
                }
            }
        }

        // v2: path 1.5 — TraitObject receiver; restore the real signature from method_sigs.
        // First extract the sig data (param_count + return_type) into owned variables,
        // releasing the arena.types borrow before constructing the Fn type.
        let trait_sig_data: Option<(u8, TypeHandle)> =
            if let Type::TraitObject(_) = self.arena.get(resolved) {
                let (_, method_sigs) = self.arena.trait_object_parts(resolved);
                method_sigs
                    .iter()
                    .find(|m| m.name.as_ref() == method)
                    .map(|sig| (sig.param_count, sig.return_type))
            } else {
                None
            };
        if let Some((param_count, return_type)) = trait_sig_data {
            // params[0] is self, bound to the receiver type (resolved) to avoid producing an orphan TypeVar;
            // the remaining parameters use fresh_type_var (exact types are determined by the implementing type).
            let params: Vec<TypeHandle> = (0..param_count)
                .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                .collect();
            return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
        }

        // v2 convergence: path 2 — query TypeDefInfo.methods (the type's own methods, indexed by method_idx).
        if let Some(ref name) = type_name {
            let sig_data: Option<(Vec<TypeRepr>, Option<TypeRepr>)> =
                if let Some(&type_idx) = self.sema_result.type_def_index.get(name.as_str()) {
                    self.sema_result.type_defs[&type_idx]
                        .methods
                        .iter()
                        .find(|m| m.name.as_ref() == method)
                        .map(|sig| (sig.param_type_reprs.to_vec(), sig.return_type_repr.clone()))
                } else {
                    None
                };
            if let Some((param_type_reprs, return_type_repr)) = sig_data {
                return Some(self.build_fn_type_from_sig(param_type_reprs, return_type_repr, resolved));
            }
        }

        None
    }

    /// Try resolving `name` as an instance field of `this_ty` (implicit this).
    /// Returns the field type on success, None on failure (no error reported).
    /// Used by the Ident fallback when lexical lookup fails inside a method body.
    fn try_implicit_this_field(
        &mut self,
        this_ty: TypeHandle,
        name: &str,
    ) -> Option<TypeHandle> {
        let resolved = self.arena.resolve(this_ty);
        // Ref auto-deref: field access on &T forwards to T.
        let inner = match self.arena.get(resolved) {
            Type::Ref(_) => self.arena.ref_parts(resolved).0,
            _ => resolved,
        };
        // Inside a trait default method, this_ty is a rigid TypeVar.
        // We can't verify field existence at trait definition time (the implementing
        // type provides the fields). Be permissive: treat the bare identifier as an
        // implicit this field access and return a fresh_type_var, matching the old
        // behavior where self.field on a TypeVar silently returned a fresh_type_var.
        // Field resolution is deferred to monomorphization specialization.
        if let Type::TypeVar(_) = self.arena.get(inner) {
            return Some(self.arena.fresh_type_var());
        }
        let type_name = self.arena.type_name(inner)?.to_string();
        let field_id = self.sema_result.lookup_field_id(&type_name, name)?;
        // Look up the constructor from the TYPE definition (not ctor_def_index,
        // which can return a wrong constructor when multiple types share the same
        // constructor name, e.g. FileKind.File vs type File = File(...)).
        let def = self.sema_result.get_type_def(&type_name)?;
        let kind = def.kind;
        let repr = {
            let ctor = def.constructors.iter()
                .find(|c| c.field_names.iter().any(|fname| fname.as_deref() == Some(name)))?;
            let idx = match kind {
                TypeDefKind::Record => field_id as usize,
                _ => (field_id as usize).saturating_sub(1),
            };
            ctor.field_type_reprs.get(idx).cloned()?
        };
        Some(self.type_repr_to_handle(&repr))
    }

    /// Looks up the field type for an object type.
    /// line/column are used to locate errors when the field does not exist (passed in by the caller from the AST span).
    fn lookup_field_type(&mut self, recv_ty: TypeHandle, field: &str, line: u32, column: u32) -> TypeHandle {
        let resolved = self.arena.resolve(recv_ty);

        // Ref auto-deref: field access on &T forwards to T.
        // For reference types like &Record / &Adt, strip the Ref first and then take the normal field-lookup path,
        // to avoid the type_name indirection returning None (and silently failing) when the inner is a TypeVar.
        if let Type::Ref(_) = self.arena.get(resolved) {
            let inner = self.arena.ref_parts(resolved).0;
            return self.lookup_field_type(inner, field, line, column);
        }

        // ModuleRef field access: look up the field by bare name in the module env carried by the ModuleRef.
        //
        // Use lookup_local (which does not traverse the parent env chain) to handle uniformly:
        // - submodules: ensure_module_env registers the submodule's short name in the parent env when creating the hierarchical env.
        // - in-module symbols: predeclare_declarations has registered functions/constructors into module_env.
        // On miss, report an error; no string concatenation or prefix check is needed.
        if let Type::ModuleRef(_) = self.arena.get(resolved) {
            let (path, module_env) = self.arena.module_ref_parts(resolved);
            if let Some(sym_ty) = self.env.lookup_local(module_env, field) {
                return sym_ty;
            }
            self.add_error_at(
                &format!("no module or symbol '{}.{}'", path, field),
                line,
                column,
            );
            return self.arena.make(Type::Unknown);
        }

        let type_name = self.arena.type_name(resolved).map(|s| s.to_string());
        if let Some(name) = type_name {
            if let Some(field_id) = self.sema_result.lookup_field_id(&name, field) {
                // Look up the constructor from the TYPE definition (not ctor_def_index,
                // which can return a wrong constructor when multiple types share the same
                // constructor name, e.g. FileKind.File vs type File = File(...)).
                if let Some(def) = self.sema_result.get_type_def(&name) {
                    let kind = def.kind;
                    let idx = match kind {
                        TypeDefKind::Record => field_id as usize,
                        _ => (field_id as usize).saturating_sub(1),
                    };
                    // Find the constructor that actually has this field.
                    if let Some(repr) = def.constructors.iter()
                        .find(|c| c.field_names.iter().any(|fname| fname.as_deref() == Some(field)))
                        .and_then(|ctor| ctor.field_type_reprs.get(idx).cloned())
                    {
                        return self.type_repr_to_handle(&repr);
                    }
                    return self.arena.fresh_type_var();
                }
            }
        }
        // Record structural fields.
        let ct = self.arena.get(resolved);
        if let Type::Record(_) = ct {
            let fields = self.arena.record_fields(resolved);
            for f in fields.iter() {
                if f.name.as_deref() == Some(field) {
                    return f.ty;
                }
            }
        }
        // Channel<T>.sender / .receiver fields: return Sender<T> / Receiver<T>.
        // (Already supported at runtime in Value.rs; the Sema layer fills in the type signature.)
        if let Type::Channel(_) = ct {
            let elem = self.arena.channel_elem(resolved);
            match field {
                "sender" => return self.arena.make_sender(elem),
                "receiver" => return self.arena.make_receiver(elem),
                _ => {}
            }
        }
        // Field not found: for a determined type, report a "no such field" error (consistent with the method-call fallback);
        // for pending types (TypeVar/Unknown/Never/Void), silently return a fresh var, deferring to the solver's global diagnostics.
        match ct {
            Type::Record(_) => {
                self.add_error_at(&format!("no such field '{}' on this type", field), line, column);
                self.arena.fresh_type_var()
            }
            Type::Adt(_) => {
                let (name, _) = self.arena.adt_parts(resolved);
                // For a registered Adt type, report a "no such field" error; for unregistered ones, permissively allow.
                if self.sema_result.get_type_def(name).is_some() {
                    self.add_error_at(
                        &format!("no such field '{}' on type '{}'", field, name),
                        line,
                        column,
                    );
                }
                self.arena.fresh_type_var()
            }
            // Pending types: silently return a fresh var (inference pending, deferred to the solver's global diagnostics).
            Type::TypeVar(_) | Type::Unknown
            | Type::Never | Type::Void => {
                self.arena.fresh_type_var()
            }
            // Determined type but field lookup failed: report an error.
            ct_other => {
                let recv_name = self.arena.type_name(resolved)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{:?}", ct_other));
                self.add_error_at(
                    &format!("no such field '{}' on type '{}'", field, recv_name),
                    line,
                    column,
                );
                self.arena.fresh_type_var()
            }
        }
    }

    // ── infer_stmt ──

    /// Bug #61: render the type annotation string — if the annotation is a named type and is an alias, preserve the alias name.
    fn display_type_annotation(
        &self,
        ta: AstTypeRef,
        ast: &AstArena<'_>,
        annot_ty: TypeHandle,
    ) -> String {
        let type_node = &ast.ty(ta).node;
        if let crate::ast::Ast::TypeNode::Named { name } = type_node {
            if let Some(td) = self.sema_result.get_type_def(name) {
                if td.kind == TypeDefKind::Alias {
                    return (*name).to_string();
                }
            }
        }
        format!("{}", self.arena.display(annot_ty))
    }

    /// Shared logic for ValDecl and VarDecl: type-check the annotation and value,
    /// detect mutability-changing shadowing (Bug #76), and define the binding.
    fn check_local_decl(
        &mut self,
        name: &str,
        type_annotation: Option<crate::ast::Ast::TypeRef>,
        value: crate::ast::Ast::ExprRef,
        is_mutable: bool,
        ast: &AstArena<'_>,
        env: EnvId,
        stmt: StmtId,
    ) -> TypeHandle {
        // kind_check the type annotation.
        if let Some(ta) = type_annotation {
            let mut errors = Vec::new();
            check_type_node(self.sema_result, ast, ta, &[], &mut errors);
            for e in errors {
                self.sema_result.add_error(e);
            }
        }
        let expected_ty = type_annotation.map(|ta| self.type_from_ast(ta, ast));
        let val_ty = self.infer_expr(value, ast, env, expected_ty);
        let bind_ty = if let Some(ta) = type_annotation {
            let annot_ty = self.type_from_ast(ta, ast);
            if self.try_widen_unify(annot_ty, val_ty).is_err() {
                // Bug #61: if the type annotation is a named type and is an alias, preserve the alias name rather than unfolding the underlying type.
                let annot_str = self.display_type_annotation(ta, ast, annot_ty);
                let val_str = format!("{}", self.arena.display(val_ty));
                let span = ast.ty(ta).span;
                self.add_error_at(
                    &format!(
                        "type annotation mismatch: expected '{}', found '{}'",
                        annot_str, val_str
                    ),
                    span.line,
                    span.column,
                );
            }
            annot_ty
        } else {
            val_ty
        };

        // Bug #76: detect mutability-changing shadowing (val→var or var→val).
        // Same-mutability shadowing (val→val, var→var) is allowed.
        let key = (env.0, name.to_string());
        if let Some(&prev_mutable) = self.local_mutability.get(&key) {
            if prev_mutable != is_mutable {
                let prev_kw = if prev_mutable { "var" } else { "val" };
                let new_kw = if is_mutable { "var" } else { "val" };
                let span = ast.stmt(stmt).span;
                self.add_error_at(
                    &format!(
                        "cannot shadow {} '{}' with {} {}: mutability mismatch",
                        prev_kw, name, new_kw, name
                    ),
                    span.line,
                    span.column,
                );
            }
        }
        // Record mutability for this scope.
        self.local_mutability.insert(key, is_mutable);

        // Define the binding. Use redefine to allow same-mutability shadowing
        // (define returns false without updating when the name already exists).
        if self.env.define(env, name, bind_ty) {
            // New binding — already inserted.
        } else {
            // Name already exists — shadowing. Use redefine to update the binding.
            self.env.redefine(env, name, bind_ty);
        }
        // Return the value's inferred type so callers (e.g. Block divergence
        // analysis) can detect `val/var x = <never>` as a diverging statement.
        val_ty
    }

    /// Infers a statement's type. Returns `Some(ty)` when the statement produces a value (expression statements).
    pub fn infer_stmt(
        &mut self,
        stmt: StmtId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> Option<TypeHandle> {
        let node = &ast.stmt(stmt).node;
        match node {
            Stmt::ValDecl { name, type_annotation, value, .. } => {
                let val_ty = self.check_local_decl(name, *type_annotation, *value, false, ast, env, stmt);
                // Propagate Never so Block-level divergence analysis detects
                // `val x = <never>` (e.g. a val bound to an if/match/block that
                // always diverges) as a diverging statement. Bug #84 generality.
                if matches!(self.arena.get(self.arena.resolve(val_ty)), Type::Never) {
                    Some(val_ty)
                } else {
                    None
                }
            }
            Stmt::VarDecl { name, type_annotation, value, .. } => {
                let val_ty = self.check_local_decl(name, *type_annotation, *value, true, ast, env, stmt);
                if matches!(self.arena.get(self.arena.resolve(val_ty)), Type::Never) {
                    Some(val_ty)
                } else {
                    None
                }
            }
            Stmt::Assignment { target, value } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                if self.arena.unify(target_ty, val_ty).is_err() {
                    let target_str = format!("{}", self.arena.display(target_ty));
                    let val_str = format!("{}", self.arena.display(val_ty));
                    let span = ast.stmt(stmt).span;
                    self.add_error_at(
                        &format!(
                            "assignment type mismatch: cannot assign '{}' to '{}'",
                            val_str, target_str
                        ),
                        span.line,
                        span.column,
                    );
                }
                None
            }
            Stmt::FieldAssignment { object, value, .. } => {
                let _ = self.infer_expr(*object, ast, env, None);
                let _ = self.infer_expr(*value, ast, env, None);
                None
            }
            Stmt::CompoundAssignment { target, value, .. } => {
                let _ = self.infer_expr(*target, ast, env, None);
                let _ = self.infer_expr(*value, ast, env, None);
                None
            }
            Stmt::Expression { expr } => {
                let ty = self.infer_expr(*expr, ast, env, None);
                Some(ty)
            }
            Stmt::Return { value } => {
                if let Some(v) = value {
                    // Propagate expected_return to infer_expr so expressions that depend on
                    // the expected constraint (NullLit, match arm, etc.) infer correctly and
                    // avoid creating orphan TypeVars.
                    let expected = self.expected_return;
                    let val_ty = self.infer_expr(*v, ast, env, expected);
                    if let Some(fn_ret) = self.expected_return {
                        if self.unify_return_type(fn_ret, val_ty).is_err() {
                            let ret_str = format!("{}", self.arena.display(fn_ret));
                            let val_str = format!("{}", self.arena.display(val_ty));
                            let span = ast.stmt(stmt).span;
                            self.add_error_at(
                                &format!(
                                    "return type mismatch: expected '{}', found '{}'",
                                    ret_str, val_str
                                ),
                                span.line,
                                span.column,
                            );
                        }
                    }
                    Some(val_ty)
                } else {
                    Some(self.make_builtin(Type::Void))
                }
            }
            Stmt::Defer { expr } => {
                let _ = self.infer_expr(*expr, ast, env, None);
                // Record the defer's capture list. Defer semantics require
                // observing values at function/block exit, so ALL captures are
                // forced to Reference mode (this directly resolves Bug #49:
                // the `val`-snapshot vs defer-latest tension is eliminated
                // because defer never snapshots).
                if self.instantiation_ctx.is_none() {
                    let captures = self.compute_captures(ast, *expr, &[], true);
                    let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                    self.sema_result.put_captures(key, &self.current_module_name, captures);
                }
                None
            }
            Stmt::Throw { expr } => {
                let thrown_ty = self.infer_expr(*expr, ast, env, None);
                let span = ast.stmt(stmt).span;
                self.check_throw_stmt(thrown_ty, span.line, span.column);
                None
            }
            Stmt::Break | Stmt::Continue => None,
            Stmt::For { name, iterable, body } => {
                let span = ast.stmt(stmt).span;
                let iterable_ty = self.infer_expr(*iterable, ast, env, None);
                let child_env = self.env.child(env);
                // Structurally extract the element type from the iterator type and build a
                // constraint rather than using an isolated fresh_type_var.
                // Covers: Array<T>, ArrayIter<T>, RangeIterator, Str→char, Map<K,V>→Entry<K,V>, etc.
                // On extraction failure, falls back to fresh_type_var and lets the fixpoint solver
                // resolve via constraint.
                let item_ty = {
                    let resolved = self.arena.resolve(iterable_ty);
                    let ct = self.arena.get(resolved);
                    // Check whether iterable is a non-iterator type (Array/Str/primitive)
                    let is_non_iterator = match ct {
                        Type::Array(_) => true,
                        ct if ct.is_scalar() => true,
                        _ => false,
                    };
                    if is_non_iterator {
                        let type_name = match ct {
                            Type::Array(_) => "array",
                            _ => ct.name(),
                        };
                        self.add_error_at(
                            &format!(
                                "type '{}' does not implement Iterator; For loops require an iterator type. Use arr.iter() for arrays, str_iter(s) for strings",
                                type_name
                            ),
                            span.line,
                            span.column,
                        );
                    }
                    // Structured element type extraction.
                    self.extract_iterator_element(resolved).unwrap_or_else(|| {
                        // Extraction failed: use fresh_type_var and register a constraint for
                        // the fixpoint solver.
                        let fv = self.arena.fresh_type_var();
                        // Build ArrayIter<fv> as the expected iterator type and unify/constrain
                        // it with the actual iterable_ty.
                        let expected_iter = self.arena.make_generic(
                            "ArrayIter".into(),
                            vec![fv].into_boxed_slice(),
                        );
                        self.unify_or_constrain(iterable_ty, expected_iter);
                        fv
                    })
                };
                self.env.define(child_env, name, item_ty);
                let _ = self.infer_expr(*body, ast, child_env, None);
                None
            }
            Stmt::While { condition, body } => {
                let cond_ty = self.infer_expr(*condition, ast, env, None);
                let bool_ty = self.make_builtin(Type::Bool);
                if self.arena.unify(cond_ty, bool_ty).is_err() {
                    let cond_str = format!("{}", self.arena.display(cond_ty));
                    let span = ast.stmt(stmt).span;
                    self.add_error_at(
                        &format!(
                            "while condition must be bool, found '{}'",
                            cond_str
                        ),
                        span.line,
                        span.column,
                    );
                }
                let _ = self.infer_expr(*body, ast, env, None);
                None
            }
            Stmt::Loop { body } => {
                let _ = self.infer_expr(*body, ast, env, None);
                None
            }
            Stmt::LocalDecl { decl } => {
                // Route through check_decl uniformly: nested function/type/trait declarations
                // share the same processing path.
                // LocalDecl's Box<Decl> has no span; the enclosing Stmt provides it.
                // For nested function declarations, record the capture list (the
                // nested function captures outer-scope variables; self-reference
                // to its own name is excluded).
                if let crate::ast::Ast::Decl::FunDecl { name, params, body, .. } = decl.as_ref() {
                    if self.instantiation_ctx.is_none() {
                        let mut param_names: Vec<&str> = params.iter().map(|p| p.name).collect();
                        param_names.push(name); // self-reference is not a capture
                        let captures = self.compute_captures(ast, *body, &param_names, false);
                        let key = module_expr_key(&self.current_module_name, body.0 as u64);
                        eprintln!("DEBUG sema LocalDecl: name={name} body={} key={key:#x} captures={:?}", body.0, captures);
                        self.sema_result.put_captures(key, &self.current_module_name, captures);
                    }
                }
                self.check_decl(decl.as_ref(), ast.stmt(stmt).span, ast, env);
                None
            }

        }
    }

    // ── infer_pattern ──

    /// Infers the pattern type and introduces bound variables into the environment.
    pub fn infer_pattern(
        &mut self,
        pat: PatternId,
        ast: &AstArena<'_>,
        expected_ty: TypeHandle,
        env: EnvId,
    ) {
        let node = &ast.pattern(pat).node;
        match node {
            Pattern::Wildcard => {}
            Pattern::Literal(lit) => {
                let lit_ty = match lit {
                    PatternLiteral::Int(_) => Some(self.make_builtin(Type::I32)),
                    PatternLiteral::Float(_) => Some(self.make_builtin(Type::F64)),
                    PatternLiteral::Bool(_) => Some(self.make_builtin(Type::Bool)),
                    PatternLiteral::Char(_) => Some(self.make_builtin(Type::Char)),
                    PatternLiteral::String(_) => Some(self.make_builtin(Type::Str)),
                    PatternLiteral::Null => None,
                };
                if let Some(lt) = lit_ty {
                    let resolved = self.arena.resolve(expected_ty);
                    let ct = self.arena.get(resolved).clone();
                    let is_int_expected = ct.is_int();
                    let is_int_lit = matches!(lit, PatternLiteral::Int(_));
                    if !(is_int_lit && is_int_expected) {
                        self.unify_or_constrain(lt, expected_ty);
                    }
                }
            }
            Pattern::Variable { name } => {
                // Upper-case leading char → zero-argument constructor.
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    let sub_pats: Vec<PatternRef> = Vec::new();
                    self.refine_constructor_pattern(name, &sub_pats, expected_ty, ast, env);
                    // Store disambiguation result for the IR builder (same-named constructors).
                    if let Some(ctor) = self.find_ctor_def(name, expected_ty) {
                        self.sema_result.pattern_ctor_types.insert(
                            (self.current_module_name.clone(), pat.0),
                            ctor.type_name.clone(),
                        );
                    }
                } else {
                    self.env.define(env, name, expected_ty);
                }
            }
            Pattern::Constructor { name, patterns } => {
                if !self.refine_constructor_pattern(name, patterns, expected_ty, ast, env) {
                    // Regular constructor fallback: use field_type_reprs (self-contained TypeRepr)
                    // instead of field_type_nodes (AST reference) to avoid cross-module AST arena
                    // mismatches.
                    let field_type_reprs: Box<[TypeRepr]> = self
                        .sema_result
                        .get_ctor_def(name)
                        .map(|c| c.field_type_reprs.clone())
                        .unwrap_or_else(|| Box::new([]));
                    for (i, &sub_pat) in patterns.iter().enumerate() {
                        let sub_ty = if i < field_type_reprs.len() {
                            self.type_repr_to_handle(&field_type_reprs[i])
                        } else {
                            self.arena.fresh_type_var()
                        };
                        self.infer_pattern(sub_pat, ast, sub_ty, env);
                    }
                }
                // Store disambiguation result for the IR builder (same-named constructors).
                if let Some(ctor) = self.find_ctor_def(name, expected_ty) {
                    self.sema_result.pattern_ctor_types.insert(
                        (self.current_module_name.clone(), pat.0),
                        ctor.type_name.clone(),
                    );
                }
            }
            Pattern::Record { fields } => {
                for field in fields.iter() {
                    let field_ty = self.arena.fresh_type_var();
                    self.infer_pattern(field.pattern, ast, field_ty, env);
                }
            }
            Pattern::OrPattern { left, right } => {
                self.infer_pattern(*left, ast, expected_ty, env);
                self.infer_pattern(*right, ast, expected_ty, env);
            }
            Pattern::Guard { pattern, condition } => {
                self.infer_pattern(*pattern, ast, expected_ty, env);
                let cond_ty = self.infer_expr(*condition, ast, env, None);
                let bool_ty = self.make_builtin(Type::Bool);
                self.unify_or_constrain(cond_ty, bool_ty);
            }
        }
    }

    // ── register_builtins ──

    /// Registers builtin functions into the environment.
    pub fn register_builtins(&mut self, env: EnvId) {
        // Panic: (str) -> void
        let str_ty = self.make_builtin(Type::Str);
        let void_ty = self.make_builtin(Type::Void);
        let panic_fn = self.arena.make_fn(
            vec![str_ty].into_boxed_slice(),
            void_ty,
        );
        self.env.define(env, "Panic", panic_fn);

        // type/type_name has been converted to a kuzo wrapper (see Reflect.kz::type_name).
        // Sema no longer registers the `type` builtin.

        // Ok: ∀T,E. (T) -> Throw<T, E>
        // Registered with rigid vars (generic parameters); at call sites instantiate_fn_type
        // instantiates them to fresh non-rigid vars.
        let val_ty = self.arena.fresh_rigid_var();
        let err_ty = self.arena.fresh_rigid_var();
        let throw_ty = self.arena.make_throw(val_ty, err_ty);
        let ok_fn = self.arena.make_fn(
            vec![val_ty].into_boxed_slice(),
            throw_ty,
        );
        self.env.define(env, "Ok", ok_fn);

        // Numeric type constructors: i8/i16/.../f64 etc. as ∀T. (T) -> Self.
        // Registered with rigid vars; instantiated by instantiate_fn_type at call sites.
        for (name, ct) in numeric_builtin_names() {
            let param = self.arena.fresh_rigid_var();
            let ret_ty = self.make_builtin(ct);
            let fn_ty = self.arena.make_fn(
                vec![param].into_boxed_slice(),
                ret_ty,
            );
            self.env.define(env, name, fn_ty);
        }

        // channel<T>(capacity: usize) -> Channel<T>
        // Builtin channel constructor: creates a Channel<T> with the given capacity.
        let usize_ty = self.make_builtin(Type::Usize);
        let t_var3 = self.arena.fresh_rigid_var();
        let chan_ret = self.arena.make_channel(t_var3);
        let chan_fn = self.arena.make_fn(
            vec![usize_ty].into_boxed_slice(),
            chan_ret,
        );
        self.env.define(env, "channel", chan_fn);

        // Value: builtin opaque type (ValueHandle, u32).
        // Reflection primitives receive a Value, internally look up the ValueArena to fetch
        // the HeapObj and match directly.
        // Opaque to Sema (internal structure not exposed); size 4B.
        let value_ty = self.arena.make_generic(
            "Value".into(),
            Box::new([]),
        );
        self.env.define(env, "Value", value_ty);
    }

    // ── check_module ──

    /// Gets or creates the EnvId dedicated to a module path.
    ///
    /// Creates envs level-by-level along path segments, forming a hierarchy:
    ///   "std.io.File" → create env_std (parent=root_env)
    ///                 → env_std_io (parent=env_std)
    ///                 → env_std_io_file (parent=env_std_io)
    ///
    /// Each env level registers the child module's short name → ModuleRef so that
    /// path-based field access can be resolved structurally through the env chain.
    /// Existing path envs are reused (idempotent).
    ///
    /// Returns the EnvId corresponding to the given path.
    fn ensure_module_env(&mut self, full_path: &str, root_env: EnvId) -> EnvId {
        // Cached: return directly.
        if let Some(&eid) = self.module_envs.get(full_path) {
            return eid;
        }
        let segments: Vec<&str> = full_path.split('.').collect();
        let mut current_path = String::new();
        let mut parent_env = root_env;
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 {
                current_path.push('.');
            }
            current_path.push_str(seg);
            // Env for the current path segment: reuse if present, otherwise create.
            let env_id = if let Some(&eid) = self.module_envs.get(&current_path) {
                eid
            } else {
                let eid = self.env.child(parent_env);
                self.module_envs.insert(current_path.clone(), eid);
                eid
            };
            // Register the current segment's short name → ModuleRef in the parent env so that
            // level-by-level field access can resolve it.
            // First segment is registered in root_env; subsequent segments in their parent path env.
            let mod_ref_ty = self.arena.make_module_ref(
                current_path.clone().into_boxed_str(),
                env_id,
            );
            // Do not overwrite existing bindings (user's explicit import / constructor takes priority).
            self.env.define(parent_env, seg, mod_ref_ty);
            parent_env = env_id;
        }
        parent_env
    }

    /// Directory module semantics: look up a function in sibling modules' envs.
    ///
    /// When `sqrt` in `Math.sqrt` is defined in `Power.kz` (rather than `Math.kz`),
    /// derives the directory prefix (e.g. "std.math") from `mod_path` (e.g. "std.math.Math"),
    /// then iterates over the envs of all sibling modules in the same directory
    /// ("std.math.Power", "std.math.Trig", ...) looking up the function by its bare `method` name.
    /// Skips its own env (already checked by the caller via lookup_local).
    fn lookup_sibling_module_fn(
        &self,
        mod_path: &str,
        self_env: EnvId,
        method: &str,
    ) -> Option<TypeHandle> {
        // Derive the directory prefix: the part of mod_path before the last '.'.
        let dot_pos = mod_path.rfind('.')?;
        let dir_prefix = &mod_path[..dot_pos]; // e.g. "std.math"
        let sibling_prefix = format!("{}.", dir_prefix); // "std.math."

        // Iterate over sibling modules in module_envs that start with "std.math.".
        for (path, &env_id) in self.module_envs.iter() {
            if !path.starts_with(&sibling_prefix) {
                continue;
            }
            if path == mod_path {
                continue; // Skip self.
            }
            if env_id == self_env {
                continue; // Skip self env.
            }
            if let Some(ty) = self.env.lookup_local(env_id, method) {
                return Some(ty);
            }
        }
        None
    }

    /// Registers module path aliases into the environment (for same-package module symbol visibility).
    ///
    /// Creates a hierarchical env for each module path and registers the trailing short name
    /// → ModuleRef in root_env, so that same-package modules can be accessed directly via their
    /// short name (e.g. `Calendar` → `ModuleRef("std.time.Calendar", env)`).
    /// Existing bindings are not overwritten (user's explicit import takes priority).
    pub fn register_module_aliases(&mut self, root_env: EnvId, module_paths: &[String]) {
        for path in module_paths {
            if path.is_empty() {
                continue;
            }
            // Ensure the module hierarchy env exists (including intermediate path prefixes).
            let module_env = self.ensure_module_env(path, root_env);
            // Register the trailing short name in root_env (for same-package short-name access).
            if let Some(last_seg) = path.rsplit('.').next() {
                if !last_seg.is_empty() && path.contains('.') {
                    // Do not overwrite existing bindings.
                    if self.env.lookup(root_env, last_seg).is_none() {
                        let mod_ref_ty = self.arena.make_module_ref(
                            path.clone().into_boxed_str(),
                            module_env,
                        );
                        self.env.define(root_env, last_seg, mod_ref_ty);
                    }
                }
            }
        }
    }

    /// Module check entry point: orchestrates populate → predeclare → infer → kind_check → monomorph.
    ///
    /// Returns true when there are no errors. Steps:
    /// 1. populate_module fills the SemaResult definition tables.
    /// 2. Create the root environment and register builtins.
    /// 3. Predeclare functions and type constructors.
    /// 4. Infer expression declarations and function bodies.
    /// 5. Run kind_check.
    /// 6. Collect monomorphization instances.
    pub fn check_module(&mut self, module: &Module<'_>) -> bool {
        // Single-module check: create a new root_env, register builtins, check the module.
        self.reset_state();
        let root_env = self.env.root();
        self.register_builtins(root_env);
        let all_modules = [module];
        self.check_module_with_env(module, root_env, &all_modules)
    }

    /// Multi-module shared-env check entry point.
    ///
    /// Accepts an externally shared `root_env` (already populated with builtins and prior
    /// modules' symbols), and on top of it processes imports, predeclarations, and checks the
    /// current module.
    /// Cross-module symbols are resolved through the shared env chain.
    pub fn check_module_with_env<'m>(
        &mut self,
        module: &'m Module<'m>,
        root_env: EnvId,
        all_modules: &[&'m Module<'m>],
    ) -> bool {
        // 1. Populate the definition tables (if not already populated).
        populate_module(self.arena, self.sema_result, module);

        // 1b. Check for cyclic type aliases (Bug #80).
        self.check_alias_cycles();

        // 1c. Duplicate constructor names across types are now allowed at the
        // definition level (disambiguated by type context or `Type.Ctor`).
        // Ambiguity is reported only at use sites when disambiguation fails
        // (see infer_call). (Bug #81.)

        // 1d. Check for duplicate named fields within a constructor (Bug #82).
        self.check_duplicate_ctor_fields();

        // 2. Reset state (do not reset env; preserve the shared root_env).
        self.reset_state();
        self.reset_per_module_state();
        // Snapshot current type_vars/types length: arena is shared across modules and not reset;
        // diagnostics only count the TypeVars newly added by this module.
        let type_vars_baseline = self.arena.type_vars_len();
        let types_baseline = self.arena.len();
        self.current_module_logical_path = module_logical_path(module.name);
        self.current_module_name = module.name.to_string();

        // 3. Process import declarations: register module reference aliases + import aliases.
        self.process_import_decls(module, root_env);

        // 4. Predeclare functions and type constructors (including extern functions).
        self.predeclare_declarations(module, root_env);

        // 5. Populate the witness table (iterate trait impls).
        self.populate_witness_table(module);

        // 6. Infer declarations.
        // Use module_env as the base environment (rather than root_env) so function bodies can
        // resolve same-module functions via the env chain (predeclare_declarations registers
        // into module_env), while still reaching root_env's global builtins and constructors
        // through the parent chain.
        let check_env = self.current_module_env.unwrap_or(root_env);
        for decl in module.declarations.iter() {
            self.check_decl(&decl.node, decl.span, &module.arena, check_env);
        }

        // 7. kind_check all type annotations.
        self.run_kind_checks(module);

        // 8. Collect monomorphization instances (generic function instances; does not depend on witness_table).
        crate::sema::Monomorph::collect_monomorph_instances(module, all_modules, self.sema_result, self.arena);

        // 9. Solve deferred constraints (with witness table support for trait bound solving).
        // Split borrows of self's different fields: arena as mutable borrow, witness_table as shared borrow.
        let InferContext { arena, solver, witness_table, type_trace, .. } = self;
        solver.solve_with_witness(arena, Some(witness_table));

        // 9.1 Report solver ambiguity errors (Bug #83: generic parameter type
        // unification failures were silently dropped — solver.errors() was never
        // consulted, so e.g. `pair(1i32, 2i64)` silently bound T to i32 and accepted
        // the i64 argument).
        // Only report ambiguity errors (from finalize_solution: a TypeVar was required
        // to bind to multiple distinct concrete types). "type mismatch" errors from
        // the fixpoint loop are NOT reported — they are often false positives because
        // `unify_or_constrain` does strict unify only, while other paths use
        // `try_widen_unify` (widening/nullable/async unfolding) which accepts those
        // same type pairs.
        {
            let solver_errors: Vec<ConstraintError> = solver.errors().to_vec();
            for ce in solver_errors {
                if !ce.reason.as_ref().contains("ambiguous") {
                    continue;
                }
                let (t1, t2) = match &ce.constraint {
                    Constraint::Equality(t1, t2) => (*t1, *t2),
                    _ => continue,
                };
                let r1 = arena.resolve(t1);
                let r2 = arena.resolve(t2);
                let s1 = arena.display(r1);
                let s2 = arena.display(r2);
                self.sema_result.add_error(crate::sema::Sema::SemaError::new(
                    &format!("type mismatch: {} does not unify with {} ({})", s1, s2, ce.reason),
                    ce.line, ce.column,
                ));
            }
        }

        // 9.4 Default unbound non-rigid TypeVars to void (root cause F).
        //
        // Having unbound non-rigid TypeVars after constraint solving is a normal type inference
        // phenomenon:
        // - E in Throw<T, E> is unconstrained (when the function never throws, E has no source of info).
        // - T in ArrayIter<T> is unconstrained (when the iterator is never consumed).
        // - TypeVars produced by generic function instantiation are not constrained by call sites.
        //
        // These TypeVars have no candidate types (not ambiguity); void as the unit type is a safe
        // default: unconstrained means any type satisfies, and void introduces no extra constraint.
        // This is the standard type-inference defaulting technique (ML's defaulting rule), not an
        // error fallback.
        // Real type errors are caught by unify failures and ambiguity detection, independent of
        // this diagnostic.
        let void_ty = arena.make(Type::Void);
        for i in type_vars_baseline..arena.type_vars.len() {
            let tv = &mut arena.type_vars[i];
            if !tv.is_rigid && tv.bound.is_none() {
                tv.bound = Some(void_ty);
            }
        }

        // 9.5 Global residual TypeVar diagnostic: unbound non-rigid TypeVars after solving indicate
        // type inference failure.
        // Only count TypeVars newly added by this module (arena is shared across modules; entries
        // before baseline belong to prior modules).
        let unresolved: Vec<u32> = arena.type_vars.iter().enumerate()
            .skip(type_vars_baseline)
            .filter(|(_, tv)| !tv.is_rigid && tv.bound.is_none())
            .map(|(i, _)| i as u32)
            .collect();

        // Verbose logging (controlled by KUZO_SEMA_TRACE env var): print unresolved TypeVar details
        // for easier diagnosis.
        if !unresolved.is_empty() && std::env::var("KUZO_SEMA_TRACE").is_ok() {
            let unresolved_set: FxHashSet<u32> = unresolved.iter().copied().collect();
            eprintln!(
                "[sema] {} unresolved type variable(s) after constraint solving:",
                unresolved.len()
            );
            for &idx in unresolved.iter().take(50) {
                let tv = &arena.type_vars[idx as usize];
                eprintln!("  TypeVar({}) kind={:?}", idx, tv.kind);
            }
            if unresolved.len() > 50 {
                eprintln!("  ... and {} more", unresolved.len() - 50);
            }
            // Print a sample of type slots containing unresolved TypeVars (up to 30).
            // Only iterate over type slots newly added by this module (entries before baseline
            // belong to prior modules).
            eprintln!("  sample referencing types (baseline={}):", types_baseline);
            let mut shown = 0u32;
            for i in types_baseline..arena.types.len() {
                let h = TypeHandle(i as u32);
                let s = format!("{}", arena.display(h));
                if s.contains("'_") {
                    eprintln!("    types[{}] = {}", i, s);
                    shown += 1;
                    if shown >= 30 { break; }
                }
            }
            // Reverse lookup: iterate type_trace to find expression spans referencing unresolved TypeVars.
            eprintln!("  referencing expression spans:");
            let mut span_shown = 0u32;
            for &(ty, span) in type_trace.iter() {
                if type_contains_any_unresolved(ty, arena, &unresolved_set) {
                    let s = format!("{}", arena.display(ty));
                    eprintln!("    {}:{}  {}", span.line, span.column, s);
                    span_shown += 1;
                    if span_shown >= 50 { break; }
                }
            }
            if span_shown == 0 {
                eprintln!("    (no direct expression references found — TypeVar may be inside fn signature)");
            }
        }

        // 10. Mirror witness_table into sema_result (so the IR layer can access trait method
        // dispatch info).
        // witness_table accumulates across modules; sync the latest state after each check.
        self.sema_result.witness_table = witness_table.clone();

        // 10a. Collect trait default-method monomorphization instances (depends on the mirrored
        // witness_table).
        crate::sema::Monomorph::collect_trait_default_instances(module, self.sema_result);

        // 11. Report global residual TypeVar diagnostics.
        if !unresolved.is_empty() {
            self.add_error_at(
                &format!("{} unresolved type variable(s) after constraint solving", unresolved.len()),
                0, 0,
            );
        }

        !self.sema_result.has_error
    }

    /// Resets the check state (env is not reset; the shared root_env is preserved).
    pub fn reset_state(&mut self) {
        self.expected_return = None;
        self.type_binding_stack = TypeBindingStack::new();
        self.this_binding_stack = ThisBindingStack::new();
        self.solver.reset();
        self.flow_ctx.reset();
        self.type_trace.clear();
        // witness_table is not reset (accumulates across modules, supports multi-module trait impls).
    }

    /// Reset per-module state that reset_state misses.
    /// Called before incremental recheck of a module.
    pub fn reset_per_module_state(&mut self) {
        self.local_mutability.clear();
        self.instantiation_ctx = None;
    }

    /// Processes ImportDecls in the module:
    /// - Full-path import `import std.io.File` → ensure the module hierarchy env exists; the
    ///   first segment is registered as a ModuleRef (field access builds the path level by level:
    ///   std → std.io → std.io.File, resolved via the env chain).
    /// - Selective import `import std.io.File { open }` → look up the symbol in the target
    ///   module's env and register it as an alias.
    fn process_import_decls(&mut self, module: &Module<'_>, env: EnvId) {
        // Register the current module's own path prefix (e.g. std/io/Path.kz → std.io.Path)
        // so in-module self-references (e.g. std.io.Path.last_index_of) can resolve.
        if let Some(logical_path) = module_logical_path(module.name) {
            // ensure_module_env creates the hierarchy env and registers the first-segment
            // ModuleRef in the parent env.
            self.ensure_module_env(&logical_path, env);
        }

        for decl in module.declarations.iter() {
            if let Decl::ImportDecl { module_path, items, .. } = &decl.node {
                if module_path.is_empty() {
                    continue;
                }
                let full_path = module_path.join(".");
                // Ensure the imported module's hierarchy env exists (including intermediate path
                // prefixes and first-segment ModuleRef registration).
                let module_env = self.ensure_module_env(&full_path, env);

                // Selective import: look up symbols in the target module env and register them
                // into the current env.
                if let Some(items) = items {
                    for item in items.iter() {
                        // Look up the symbol by bare name in the module env (does not traverse
                        // the parent env, to avoid importing global symbols).
                        if let Some(sym_ty) = self.env.lookup_local(module_env, item.name) {
                            let local_name = item.alias.unwrap_or(item.name);
                            self.env.define(env, local_name, sym_ty);
                        }
                    }
                }
            }
        }
    }

    /// Populates the witness table: iterates trait impls in the module and registers them.
    ///
    /// For each `impl Trait for Type`, extracts trait_name and type_name, looks up type_def to
    /// obtain the type_id, and registers the methods into the witness table.
    fn populate_witness_table(&mut self, module: &Module<'_>) {
        type TraitImplInfo = (String, String, Vec<(String, u16)>);
        // Collect trait impl info up front to avoid borrowing module while iterating and holding
        // &mut self simultaneously.
        let mut impls: Vec<TraitImplInfo> = Vec::new();

        for decl in module.declarations.iter() {
            if let Decl::TypeDecl { name, implemented_traits, methods, .. } = &decl.node {
                // Look up type_id (using type_def_index + FIRST_DYNAMIC_TYPE_ID offset).
                let type_id = self
                    .sema_result
                    .type_def_index
                    .get(*name)
                    .map(|&idx| dynamic_type_id(idx));

                if let Some(tid) = type_id {
                    // Register a witness entry for each implemented trait.
                    for impl_trait in implemented_traits.iter() {
                        let trait_name = impl_trait.trait_name.to_string();
                        // Collect method slots: method_name → method_idx (position in TypeDefInfo.methods).
                        let method_slots: Vec<(String, u16)> = methods
                            .iter()
                            .enumerate()
                            .map(|(i, m)| (m.name.to_string(), i as u16))
                            .collect();
                        impls.push((trait_name, name.to_string(), method_slots));
                        let _ = tid; // tid is used in the loop below.
                    }
                }
            }
        }

        // Register into the witness table.
        for (trait_name, type_name, method_slots_vec) in impls {
            // Re-query type_id (the previous borrow has been released).
            let type_id = self
                .sema_result
                .type_def_index
                .get(type_name.as_str())
                .map(|&idx| dynamic_type_id(idx));
            if let Some(tid) = type_id {
                let mut slots = FxHashMap::default();
                for (method_name, method_idx) in method_slots_vec {
                    slots.insert(method_name.into_boxed_str(), method_idx);
                }
                self.witness_table
                    .register(&trait_name, tid, &type_name, slots);
                // Record module ownership for incremental purge (witness key).
                let mod_name = self.current_module_name.clone();
                self.sema_result.module_ownership.witness_keys
                    .entry(mod_name)
                    .or_default()
                    .insert((trait_name.clone().into_boxed_str(), tid));
            }
        }
    }

    /// Wraps the return type into `Async<T>` as needed.
    ///
    /// The return type `X` declared by an async function/method actually denotes `Async<X>`
    /// (aligned with the Lambda/Zig implementation).
    /// If the user already wrote `Async<X>` explicitly, do not double-wrap; otherwise wrap as
    /// `Async<ret_ty>`.
    /// Used uniformly by predeclare_declarations, check_decl(FunDecl), type-block methods, and
    /// trait-block methods, to avoid omissions caused by repeated inlining (root cause D:
    /// type/trait-block methods previously missed the wrap).
    fn wrap_async_return(&mut self, ret_ty: TypeHandle, is_async: bool) -> TypeHandle {
        if !is_async {
            return ret_ty;
        }
        let resolved = self.arena.resolve(ret_ty);
        let already_async = matches!(self.arena.get(resolved), Type::Async(_));
        if already_async {
            ret_ty
        } else {
            self.arena.make_async(ret_ty)
        }
    }

    /// Predeclares the module's functions and type constructors into the environment.
    ///
    /// Functions and type constructors are registered into the module-dedicated env
    /// (module_env) rather than root_env.
    /// The module env's parent points to root_env (or the parent path env), so the module can
    /// still access global builtins.
    /// Callers resolve symbols directly by bare name in the module env via the env reference
    /// carried by ModuleRef, without needing mangled names.
    pub fn predeclare_declarations(&mut self, module: &Module<'_>, root_env: EnvId) {
        let module_path = module_logical_path(module.name);
        // Get or create the module-dedicated env (idempotent: ensure_module_env reuses existing envs).
        let module_env = match &module_path {
            Some(mp) => self.ensure_module_env(mp, root_env),
            None => root_env,
        };
        // Record the current module env for use by check_decl (e.g. let-bindings).
        self.current_module_env = Some(module_env);
        for decl in module.declarations.iter() {
            match &decl.node {
                Decl::FunDecl { name, type_params, params, return_type, is_async, .. } => {
                    // Top-level functions disallow a self parameter (detected via the ThisType
                    // type node, not by parameter name).
                    if !params.is_empty() && self.is_this_param(params[0].type_annotation, &module.arena) {
                        self.add_error_at(
                            "this parameter is not allowed in top-level function",
                            decl.span.line,
                            decl.span.column,
                        );
                    }
                    // Generic function: push type_bindings so type_from_ast resolves type
                    // parameters as rigid vars; the predeclared type matches check_decl
                    // (avoids generic params being mis-resolved as Adt).
                    if !type_params.is_empty() {
                        self.push_type_bindings(
                            &type_params.iter().map(|tp| {
                                (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                            }).collect::<Vec<_>>(),
                        );
                    }
                    // All functions are predeclared (including generics): generic functions use
                    // rigid vars as type parameters, solving forward-reference issues (function
                    // bodies can reference later-defined same-module functions).
                    let param_types: Vec<TypeHandle> = params
                        .iter()
                        .map(|p| match p.type_annotation {
                            Some(ta) => self.type_from_ast(ta, &module.arena),
                            None => self.arena.fresh_type_var(),
                        })
                        .collect();
                    // async function: the user-declared return type X actually denotes
                    // Async<X> (aligned with check_decl/Lambda).
                    // - If the user already wrote Async<X> explicitly, do not double-wrap.
                    // - Otherwise wrap as Async<ret_ty_raw>.
                    let ret_ty_raw = match return_type {
                        Some(rt) => self.type_from_ast(*rt, &module.arena),
                        None => self.arena.fresh_type_var(),
                    };
                    let ret_ty = self.wrap_async_return(ret_ty_raw, *is_async);
                    let fn_ty = self.arena.make_fn(
                        param_types.into_boxed_slice(),
                        ret_ty,
                    );
                    // Register into the module-dedicated env (bare name); ModuleRef lookup uses
                    // lookup_local in this env.
                    // Also register into root_env to make it globally visible (cross-module
                    // bare-name reference compatibility):
                    //   define does not overwrite existing bindings; the first registration wins.
                    self.env.define(module_env, name, fn_ty);
                    self.env.define(root_env, name, fn_ty);
                    // Generic function: pop type_bindings (symmetric with check_decl).
                    if !type_params.is_empty() {
                        self.pop_type_bindings();
                    }
                }
                Decl::TypeDecl { name, type_params, def, .. } => {
                    // Predeclare the type constructor.
                    let self_ty = if type_params.is_empty() {
                        self.arena.make_adt((*name).into(), Box::new([]))
                    } else {
                        // Generic type: predeclare with a rigid var.
                        self.arena.fresh_rigid_var()
                    };
                    // Constructors are registered into root_env (not module_env):
                    // constructors are companion symbols of the type, at the same naming level,
                    // and must use redefine to overwrite the ModuleRef alias previously registered
                    // by register_module_aliases, so `DateTime(...)` resolves to the constructor
                    // rather than the ModuleRef.
                    match def {
                        crate::ast::Ast::TypeDef::Adt { constructors } => {
                            for ctor in constructors.iter() {
                                let ctor_fn_ty = self.build_ctor_fn_type(ctor, name, &module.arena);
                                self.env.redefine(root_env, ctor.name, ctor_fn_ty);
                                // Record constructor short name → module env (Zig @This semantics),
                                // so `TypeName.free_func(args)` can fall back to lookup a
                                // in-module free function.
                                self.ctor_module_envs.insert(ctor.name.to_string(), module_env);
                            }
                        }
                        crate::ast::Ast::TypeDef::Newtype { name: ctor_name, inner } => {
                            // newtype constructor: (inner) -> Self.
                            let inner_ty = self.type_from_ast(*inner, &module.arena);
                            let ctor_fn_ty = self.arena.make_fn(
                                vec![inner_ty].into_boxed_slice(),
                                self_ty,
                            );
                            self.env.redefine(root_env, ctor_name, ctor_fn_ty);
                            // Record constructor short name → module env (Zig @This semantics),
                            // so `TypeName.free_func(args)` can fall back to lookup a
                            // in-module free function.
                            self.ctor_module_envs.insert(ctor_name.to_string(), module_env);
                        }
                        _ => {}
                    }
                    let _ = self_ty;
                }
                _ => {}
            }
        }
    }

    /// Builds the function type for a constructor.
    fn build_ctor_fn_type(
        &mut self,
        ctor: &crate::ast::Ast::ConstructorDef<'_>,
        type_name: &str,
        ast: &AstArena<'_>,
    ) -> TypeHandle {
        let param_types: Vec<TypeHandle> = ctor
            .fields
            .iter()
            .map(|f| self.type_from_ast(f.ty, ast))
            .collect();
        let ret_ty = match ctor.return_type {
            Some(rt) => self.type_from_ast(rt, ast),
            None => self.arena.make_adt(type_name.into(), Box::new([])),
        };
        // Zero-argument variants are values, not functions: Leaf's type should be Tree, not
        // () -> Tree.
        if param_types.is_empty() {
            return ret_ty;
        }
        self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
    }

    /// Check for cyclic type aliases (Bug #80).
    ///
    /// Iterates through all type definitions; for each alias, follows the `target_type_name`
    /// chain to detect cycles. The existing `visiting`-based cycle detection in
    /// `resolve_name_to_type` / `resolve_named_type_resolved` is ineffective because the
    /// `target_type` short-circuit (returning the pre-resolved TypeHandle directly) bypasses
    /// the recursive `target_type_name` path where cycle detection lives.
    ///
    /// This function reports a `cyclic type alias` error for each cycle found, e.g.:
    /// `type A = B` + `type B = A` → `cyclic type alias: A -> B -> A`.
    fn check_alias_cycles(&mut self) {
        use std::collections::HashSet;
        // Collect already-reported cycle messages to avoid duplicates across modules
        // (sema_result.type_defs is cumulative across all modules).
        let existing: HashSet<String> = self.sema_result.errors.iter()
            .filter_map(|e| {
                e.message.as_ref().strip_prefix("cyclic type alias: ").map(String::from)
            })
            .collect();
        let mut reported: HashSet<String> = existing;
        let mut cycles_to_report: Vec<String> = Vec::new();
        for td in self.sema_result.type_defs.values() {
            if td.kind != crate::sema::Sema::TypeDefKind::Alias {
                continue;
            }
            let mut chain: Vec<String> = Vec::new();
            let mut visiting: HashSet<String> = HashSet::new();
            let start_name = td.name.to_string();
            chain.push(start_name.clone());
            visiting.insert(start_name.clone());
            let mut current = td.target_type_name.as_deref().map(String::from);
            loop {
                match current {
                    None => break, // No target_name: not a cycle (or target is a non-named type).
                    Some(ref target_name) => {
                        if visiting.contains(target_name) {
                            // Cycle detected: build the cycle path for the error message.
                            chain.push(target_name.clone());
                            let cycle_start = chain.iter().position(|n| n == target_name)
                                .unwrap_or(0);
                            let cycle_path = chain[cycle_start..].join(" -> ");
                            if reported.insert(cycle_path.clone()) {
                                cycles_to_report.push(cycle_path);
                            }
                            break;
                        }
                        // Look up the target's type def to continue the chain.
                        let target_td = self.sema_result.get_type_def(target_name);
                        match target_td {
                            Some(t) if t.kind == crate::sema::Sema::TypeDefKind::Alias => {
                                visiting.insert(target_name.clone());
                                chain.push(target_name.clone());
                                current = t.target_type_name.as_deref().map(String::from);
                            }
                            _ => break, // Target is not an alias: no cycle.
                        }
                    }
                }
            }
        }
        for cycle_path in cycles_to_report {
            self.sema_result.add_error(crate::sema::Sema::SemaError::new(
                &format!("cyclic type alias: {}", cycle_path),
                0, 0,
            ));
        }
    }

    /// Check for duplicate named fields within a single constructor (Bug #82).
    ///
    /// ADT/record constructors may have positional (unnamed) fields, which are
    /// allowed to repeat; only named fields must be unique. Pattern matching and
    /// field access rely on unique names to disambiguate.
    fn check_duplicate_ctor_fields(&mut self) {
        use std::collections::HashSet;
        // Collect (ctor_name, field_name) pairs already reported, to dedupe across
        // modules (type_defs is cumulative).
        let mut reported: HashSet<(String, String)> = self.sema_result.errors.iter()
            .filter_map(|e| {
                e.message.as_ref().strip_prefix("duplicate field '").and_then(|s| {
                    let mut it = s.split("' in constructor ");
                    let field = it.next()?.to_string();
                    let ctor = it.next()?.to_string();
                    Some((field, ctor))
                })
            })
            .collect();
        let mut new_errors: Vec<String> = Vec::new();
        for td in self.sema_result.type_defs.values() {
            for ctor in td.constructors.iter() {
                let mut seen: HashSet<String> = HashSet::new();
                for fname_opt in ctor.field_names.iter() {
                    if let Some(fname) = fname_opt {
                        let fname_s = fname.to_string();
                        let key = (fname_s.clone(), ctor.name.to_string());
                        if reported.contains(&key) {
                            continue;
                        }
                        if !seen.insert(fname_s.clone()) {
                            new_errors.push(format!(
                                "duplicate field '{}' in constructor {}",
                                fname_s, ctor.name
                            ));
                            reported.insert(key);
                        }
                    }
                }
            }
        }
        for msg in new_errors {
            self.sema_result.add_error(crate::sema::Sema::SemaError::new(&msg, 0, 0));
        }
    }

    /// Checks a single declaration (infers function body / expression).
    ///
    /// Takes `&Decl` and `decl_span` as separate parameters: top-level declarations get
    /// span+node from `Spanned<Decl>`, while nested `LocalDecl`'s `Box<Decl>` has no span and
    /// the caller supplies it from the enclosing Stmt.
    fn check_decl(&mut self, decl: &Decl<'_>, decl_span: crate::ast::Ast::Span, ast: &AstArena<'_>, env: EnvId) {
        match decl {
            Decl::FunDecl { name, type_params, params, return_type, body, extern_c_body, is_async, .. } => {
                // Top-level functions disallow a self parameter (detected via the ThisType type
                // node, not by parameter name; self is only allowed inside type/trait block methods).
                if !params.is_empty() && self.is_this_param(params[0].type_annotation, ast) {
                    self.add_error_at(
                        "this parameter is not allowed in top-level function",
                        decl_span.line,
                        decl_span.column,
                    );
                }
                // Create a child environment for the function.
                let fn_env = self.env.child(env);
                // Type parameter bindings.
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                // @extern("C") function: register the signature but skip body type checking
                // (the body is C code, not a Kuzo expression).
                if extern_c_body.is_some() {
                    if !type_params.is_empty() {
                        self.pop_type_bindings();
                    }
                    let _ = name;
                    return;
                }
                // Parameter bindings (also collect parameter types to build the function type).
                let param_types: Vec<TypeHandle> = params.iter().map(|p| {
                    let param_ty = match p.type_annotation {
                        Some(ta) => self.type_from_ast(ta, ast),
                        None => self.arena.fresh_type_var(),
                    };
                    self.env.define(fn_env, p.name, param_ty);
                    param_ty
                }).collect();
                // Return type (use fresh_type_var when unannotated; later unified with the body type).
                // async function: the user-declared return type X actually denotes Async<X>
                // (aligned with the Lambda/Zig implementation).
                // - If the user already wrote Async<X> explicitly, do not double-wrap.
                // - Otherwise wrap as Async<ret_ty_raw>.
                let ret_ty_raw = match return_type {
                    Some(rt) => self.type_from_ast(*rt, ast),
                    None => self.arena.fresh_type_var(),
                };
                let ret_ty = self.wrap_async_return(ret_ty_raw, *is_async);
                // Build the function type and register it into fn_env (supports recursive
                // self-reference) and env (supports subsequent references).
                // Top-level functions were already predeclared by predeclare_declarations; define
                // returns false and does not overwrite.
                let fn_ty = self.arena.make_fn(
                    param_types.into_boxed_slice(),
                    ret_ty,
                );
                self.env.define(fn_env, *name, fn_ty);
                self.env.define(env, *name, fn_ty);
                // Set the expected return type.
                let prev_return = self.expected_return;
                self.expected_return = Some(ret_ty);
                // Infer the function body.
                let body_ty = self.infer_expr(*body, ast, fn_env, self.expected_return);
                // Restore.
                self.expected_return = prev_return;
                // Unify the return type with the body type:
                // - Unannotated return type: ret_ty is a fresh TypeVar; bind via unify_or_constrain.
                // - Annotated return type: unify via unify_return_type, handling async unfolding
                //   (declared Async<Throw<T, E>>, body returns Throw<T', E'> directly; must unfold
                //   the Async layer to unify the inner Throw so TypeVars in E' get solved).
                //   On failure, register an Equality constraint for the solver to retry later.
                if return_type.is_none() {
                    self.unify_or_constrain(ret_ty, body_ty);
                } else if self.unify_return_type(ret_ty, body_ty).is_err() {
                    self.solver.add_equality(ret_ty, body_ty);
                }
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
                let _ = name;
            }
            Decl::ExprDecl { expr, stmt, .. } => {
                if let Some(s) = stmt {
                    let _ = self.infer_stmt(*s, ast, env);
                } else {
                    let _ = self.infer_expr(*expr, ast, env, None);
                }
            }
            Decl::TypeDecl { name, type_params, def, methods, .. } => {
                // Register the nested type definition into sema_result (so constructor calls are
                // recognized during type checking).
                ast_type_decl_to_type_def(self.arena, self.sema_result, *name, type_params, def, ast, decl_span, &self.current_module_name);
                // Register methods into TypeDefInfo.methods (indexed by method_idx).
                // This mirrors what `populate_module` does for top-level TypeDecls. Without this,
                // `lookup_method_idx` cannot find local-type methods and IR dispatch returns void.
                for method in methods.iter() {
                    crate::sema::Sema::ast_method_to_func_sig_pub(self.arena, self.sema_result, *name, method, ast);
                }
                // Type parameter bindings (including kind registration): so references to the
                // generic parameter T inside the type block can be resolved from
                // type_binding_stack.
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                // Build the ADT type handle.
                let self_ty = if type_params.is_empty() {
                    self.arena.make_adt((*name).into(), Box::new([]))
                } else {
                    // Generic type: build Adt { name, type_args: [rigid_T, ...] }.
                    // Use the rigid var from type_binding_stack as type_args, to avoid producing
                    // unresolved TypeVars when fresh_type_var is used as self_ty.
                    let type_args: Vec<TypeHandle> = type_params.iter()
                        .map(|tp| self.lookup_type_binding(tp.name)
                            .unwrap_or_else(|| self.arena.fresh_type_var()))
                        .collect();
                    self.arena.make_adt((*name).into(), type_args.into_boxed_slice())
                };
                // Register constructor function types into the current env (so Call expressions
                // can find the constructors).
                match def {
                    crate::ast::Ast::TypeDef::Record { fields } => {
                        let param_types: Vec<TypeHandle> = fields.iter().map(|f| {
                            self.type_from_ast(f.ty, ast)
                        }).collect();
                        let fn_ty = self.arena.make_fn(
                            param_types.into_boxed_slice(),
                            self_ty,
                        );
                        self.env.define(env, *name, fn_ty);
                    }
                    crate::ast::Ast::TypeDef::Adt { constructors } => {
                        for ctor in constructors {
                            let param_types: Vec<TypeHandle> = ctor.fields.iter().map(|f| {
                                self.type_from_ast(f.ty, ast)
                            }).collect();
                            let fn_ty = if param_types.is_empty() {
                                // Zero-argument variants are values, not functions.
                                self_ty
                            } else {
                                self.arena.make_fn(
                                    param_types.into_boxed_slice(),
                                    self_ty,
                                )
                            };
                            self.env.define(env, ctor.name, fn_ty);
                        }
                    }
                    crate::ast::Ast::TypeDef::Alias { .. } | crate::ast::Ast::TypeDef::Newtype { .. } => {}
                }
                // Type method checking.
                self.push_this_type(self_ty);
                // First register all methods as functions into env (supports bare-name method
                // call syntax `method(recv, args)`), then check method bodies (avoids
                // forward-reference issues).
                for method in methods.iter() {
                    let m_param_types: Vec<TypeHandle> = method.params.iter().map(|p| {
                        if self.is_this_param(p.type_annotation, ast) {
                            self_ty
                        } else {
                            match p.type_annotation {
                                Some(ta) => self.type_from_ast(ta, ast),
                                None => self.arena.fresh_type_var(),
                            }
                        }
                    }).collect();
                    let m_ret_ty_raw = match method.return_type {
                        Some(rt) => self.type_from_ast(rt, ast),
                        None => self.arena.fresh_type_var(),
                    };
                    // async method: wrap the return type as Async<T> (aligned with top-level
                    // FunDecl/predeclare; root cause D).
                    let m_ret_ty = self.wrap_async_return(m_ret_ty_raw, method.is_async);
                    let m_fn_ty = self.arena.make_fn(
                        m_param_types.into_boxed_slice(),
                        m_ret_ty,
                    );
                    self.env.define(env, method.name, m_fn_ty);
                }
                for method in methods.iter() {
                    if let Some(body) = method.body {
                        let method_env = self.env.child(env);
                        for param in method.params.iter() {
                            let param_ty = if self.is_this_param(param.type_annotation, ast) {
                                self.infer_this_param(param.type_annotation, ast)
                            } else {
                                match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                }
                            };
                            self.env.define(method_env, param.name, param_ty);
                        }
                        let prev_return = self.expected_return;
                        let ret_ty_raw = method.return_type.map(|rt| self.type_from_ast(rt, ast));
                        // async method: wrap the return type as Async<T> (aligned with FunDecl;
                        // root cause D); unify_return_type unfolds the Async layer to unify the
                        // inner type with the body type.
                        let ret_ty = ret_ty_raw.map(|t| self.wrap_async_return(t, method.is_async));
                        self.expected_return = ret_ty;
                        let body_ty = self.infer_expr(body, ast, method_env, ret_ty);
                        self.expected_return = prev_return;
                        // Unify the method body type with the declared return type (aligned with
                        // FunDecl):
                        // - Unannotated return type: ret_ty is None → fresh_type_var; bind via
                        //   unify_or_constrain.
                        // - Annotated return type: unify via unify_return_type (handles async
                        //   unfolding); on failure, register an Equality constraint for the
                        //   solver to retry later.
                        // This lets fresh vars produced by expressions that don't depend on
                        // `expected` (e.g. FieldAccess) be solved via ret_ty, avoiding orphan
                        // TypeVars.
                        let ret = ret_ty.unwrap_or_else(|| self.arena.fresh_type_var());
                        if method.return_type.is_none() {
                            self.unify_or_constrain(ret, body_ty);
                        } else if self.unify_return_type(ret, body_ty).is_err() {
                            self.solver.add_equality(ret, body_ty);
                        }
                    }
                }
                self.pop_this_type();
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
            }
            Decl::TraitDecl { name, type_params, methods, .. } => {
                // Register the nested trait definition into sema_result (so trait type
                // annotations are recognized).
                ast_trait_decl_to_trait_def(self.arena, self.sema_result, name, methods, ast);
                // Type parameter bindings (including kind registration): so references to
                // generic parameters inside the trait block can be resolved from
                // type_binding_stack.
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                let self_var = self.push_this_type_var();
                self.current_trait_name = Some((*name).to_string().into_boxed_str());
                for method in methods.iter() {
                    if let Some(body) = method.body {
                        let method_env = self.env.child(env);
                        for param in method.params.iter() {
                            let param_ty = if self.is_this_param(param.type_annotation, ast) {
                                self.infer_this_param(param.type_annotation, ast)
                            } else {
                                match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                }
                            };
                            self.env.define(method_env, param.name, param_ty);
                        }
                        let prev_return = self.expected_return;
                        let ret_ty_raw = method.return_type.map(|rt| self.type_from_ast(rt, ast));
                        // async default method: wrap the return type as Async<T> (aligned with
                        // FunDecl; root cause D); unify_return_type unfolds the Async layer to
                        // unify the inner type with the body type.
                        let ret_ty = ret_ty_raw.map(|t| self.wrap_async_return(t, method.is_async));
                        self.expected_return = ret_ty;
                        let body_ty = self.infer_expr(body, ast, method_env, ret_ty);
                        self.expected_return = prev_return;
                        // Unify the method body type with the declared return type (aligned with
                        // FunDecl):
                        // - Unannotated return type: ret_ty is None → fresh_type_var; bind via
                        //   unify_or_constrain.
                        // - Annotated return type: unify via unify_return_type (handles async
                        //   unfolding); on failure, register an Equality constraint for the
                        //   solver to retry later.
                        // This lets fresh vars produced by expressions that don't depend on
                        // `expected` (e.g. FieldAccess) be solved via ret_ty, avoiding orphan
                        // TypeVars.
                        let ret = ret_ty.unwrap_or_else(|| self.arena.fresh_type_var());
                        if method.return_type.is_none() {
                            self.unify_or_constrain(ret, body_ty);
                        } else if self.unify_return_type(ret, body_ty).is_err() {
                            self.solver.add_equality(ret, body_ty);
                        }
                    }
                }
                self.pop_this_type();
                self.current_trait_name = None;
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
                let _ = (name, self_var);
            }
            _ => {}
        }
    }

    /// Runs kind_check on all type annotations in the module.
    fn run_kind_checks(&mut self, module: &Module<'_>) {
        let mut errors = Vec::new();
        for decl in module.declarations.iter() {
            match &decl.node {
                Decl::FunDecl { params, return_type, .. } => {
                    for p in params.iter() {
                        if let Some(ta) = p.type_annotation {
                            check_type_node(self.sema_result, &module.arena, ta, &[], &mut errors);
                        }
                    }
                    if let Some(rt) = return_type {
                        check_type_node(self.sema_result, &module.arena, *rt, &[], &mut errors);
                    }
                }
                Decl::TypeDecl { def: crate::ast::Ast::TypeDef::Adt { constructors }, .. } => {
                    for ctor in constructors.iter() {
                        for f in ctor.fields.iter() {
                            check_type_node(
                                self.sema_result,
                                &module.arena,
                                f.ty,
                                &[],
                                &mut errors,
                            );
                        }
                        if let Some(rt) = ctor.return_type {
                            check_type_node(
                                self.sema_result,
                                &module.arena,
                                rt,
                                &[],
                                &mut errors,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        for e in errors {
            self.sema_result.add_error(e);
        }
    }
}

/// Builtin scalar name → Type (single derivation point, replacing the three previously duplicated
/// match sites).
///
/// Derived from `Type::BUILTIN_TABLE`: look up ValueTag by name, then dispatch to Type by ValueTag.
/// The name → ValueTag mapping comes from a single source of truth.
///
/// Type names are uniformly lower-case (consistent with .kz source syntax): null/void/bool/char/str
/// and the numeric types.
fn name_to_concrete(name: &str) -> Option<Type> {
    use crate::types::{builtin_info_by_name, ValueTag};
    let info = builtin_info_by_name(name)?;
    let ct = match info.value_tag {
        ValueTag::I8 => Type::I8,
        ValueTag::I16 => Type::I16,
        ValueTag::I32 => Type::I32,
        ValueTag::I64 => Type::I64,
        ValueTag::I128 => Type::I128,
        ValueTag::U8 => Type::U8,
        ValueTag::U16 => Type::U16,
        ValueTag::U32 => Type::U32,
        ValueTag::U64 => Type::U64,
        ValueTag::U128 => Type::U128,
        ValueTag::Isize => Type::Isize,
        ValueTag::Usize => Type::Usize,
        ValueTag::F16 => Type::F16,
        ValueTag::F32 => Type::F32,
        ValueTag::F64 => Type::F64,
        ValueTag::F128 => Type::F128,
        ValueTag::Bool => Type::Bool,
        ValueTag::Char => Type::Char,
        ValueTag::Ref => Type::Str,   // str's value_tag is Ref.
        ValueTag::Null => Type::Null,
        ValueTag::Void => Type::Void,
    };
    Some(ct)
}

/// Returns all numeric builtin type names + Type (derived from BUILTIN_TABLE).
///
/// Replaces the original static `NUMERIC_BUILTIN_NAMES` table; automatically syncs with
/// BUILTIN_TABLE changes.
/// Includes all scalars (including bool/char, consistent with the original table); excludes
/// str/null/void.
fn numeric_builtin_names() -> Vec<(&'static str, Type)> {
    use crate::types::{BUILTIN_TABLE, ValueTag};
    BUILTIN_TABLE.iter()
        .filter(|s| !matches!(s.value_tag, ValueTag::Ref | ValueTag::Null | ValueTag::Void))
        .filter_map(|s| {
            let ct = name_to_concrete(s.name)?;
            Some((s.name, ct))
        })
        .collect()
}

// =========================================================================
// sema v2: Constraint Solver — unified constraint solving engine.
//
// Design philosophy (original, not borrowed from GHC/rustc/Swift):
// - All type relations (equality, subtype, trait bound, narrowing) are unified into Constraint.
// - snapshot/rollback supports speculative inference (match arms, overload selection).
// - Batch solving: solve all at once at function-body end, rather than unifying eagerly.
// - DOD: constraints in Vec, snapshot via length index, subst via FxHashMap.
//
// Relationship with the existing TypeArena::unify:
// The solver calls unify to implement Equality constraints, but adds deferral and rollback
// capability.
// Existing eager unify calls remain compatible; new code may opt into the solver.
// =========================================================================

/// Constraint kinds: unifies all type relations into constraints.
///
/// Design notes:
/// - Equality: most common; directly calls TypeArena::unify.
/// - Subtype: calls is_subtype; on failure records an error but does not abort immediately.
/// - TraitBound: whether `ty` implements a trait (deferred to witness table lookup).
/// - Narrow: path-sensitive narrowing (used by flow narrowing).
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type equality constraint: `t1 = t2`.
    Equality(TypeHandle, TypeHandle),
    /// Subtype constraint: `sub <: sup` (directional, asymmetric).
    Subtype(TypeHandle, TypeHandle),
    /// Trait bound constraint: `ty` implements trait `trait_name<type_args>`.
    TraitBound {
        ty: TypeHandle,
        trait_name: Box<str>,
        type_args: Box<[TypeHandle]>,
    },
    /// Narrowing constraint: on some path `original` is narrowed to `narrowed`.
    /// Used for flow-sensitive narrowing (NonNull/IsCheck/ConstructorMatch).
    Narrow {
        path: Box<str>,
        original: TypeHandle,
        narrowed: TypeHandle,
    },
}

/// Extracts associated span info from a constraint (for error localization).
/// Constraint itself does not carry a span; it is passed in separately by the context that
/// generated the constraint.
/// The line/column fields are retained for ConstraintError compatibility; the solver fills 0
/// to indicate "no span".
impl Constraint {
    /// Human-readable name of the constraint kind.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Constraint::Equality(..) => "Equality",
            Constraint::Subtype(..) => "Subtype",
            Constraint::TraitBound { .. } => "TraitBound",
            Constraint::Narrow { .. } => "Narrow",
        }
    }
}

/// Constraint solving error: records the reason for a solving failure without aborting
/// inference (error recovery).
#[derive(Debug, Clone)]
pub struct ConstraintError {
    pub constraint: Constraint,
    pub reason: Box<str>,
    /// Span info: may be passed in by the constraint generator; solver-internal errors fill 0,0.
    pub line: u32,
    pub column: u32,
}

/// Constraint solver: collects constraints and solves them in batch.
///
/// Design:
/// - `pending`: queue of constraints to solve (FIFO).
/// - `subst`: solved TypeVar → TypeHandle mapping (solving results).
/// - `errors`: records of solving failures (does not abort; error recovery).
pub struct ConstraintSolver {
    pending: Vec<Constraint>,
    subst: FxHashMap<u32, TypeHandle>,
    errors: Vec<ConstraintError>,
    /// All candidate bindings each TypeVar received during fixpoint iteration (multi-value
    /// record).
    ///
    /// key = TypeVar idx, value = list of all target type handles this TypeVar was required to
    /// bind to.
    /// After fixpoint convergence, `finalize_solution` deduplicates and detects ambiguity:
    /// - Unique candidate → write into subst.
    /// - Multiple distinct candidates → flag an ambiguity error (still writes the arena's actual
    ///   solution into subst to avoid cascading false positives).
    candidates: FxHashMap<u32, Vec<TypeHandle>>,
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSolver {
    pub fn new() -> Self {
        ConstraintSolver {
            pending: Vec::new(),
            subst: FxHashMap::default(),
            errors: Vec::new(),
            candidates: FxHashMap::default(),
        }
    }

    /// Adds a constraint to the pending queue.
    #[inline]
    pub fn add(&mut self, c: Constraint) {
        self.pending.push(c);
    }

    /// Convenience method for adding an equality constraint.
    #[inline]
    pub fn add_equality(&mut self, t1: TypeHandle, t2: TypeHandle) {
        self.add(Constraint::Equality(t1, t2));
    }

    /// Convenience method for adding a subtype constraint.
    #[inline]
    pub fn add_subtype(&mut self, sub: TypeHandle, sup: TypeHandle) {
        self.add(Constraint::Subtype(sub, sup));
    }

    /// Convenience method for adding a trait bound constraint.
    #[inline]
    pub fn add_trait_bound(
        &mut self,
        ty: TypeHandle,
        trait_name: &str,
        type_args: &[TypeHandle],
    ) {
        self.add(Constraint::TraitBound {
            ty,
            trait_name: trait_name.into(),
            type_args: type_args.to_vec().into_boxed_slice(),
        });
    }

    /// Solves all pending constraints in batch.
    ///
    /// Solving strategy:
    /// 1. Equality → TypeArena::unify; on success update subst.
    /// 2. Subtype → is_subtype check; on failure record an error.
    /// 3. TraitBound → query via witness table (requires witness_table to be passed in).
    /// 4. Narrow → update the flow fact table (implemented in phase 3).
    ///
    /// After solving, pending is cleared; results go into subst and errors.
    pub fn solve(&mut self, arena: &mut TypeArena) {
        self.solve_with_witness(arena, None)
    }

    /// Solves all pending constraints in batch (with witness table support).
    ///
    /// Fixpoint iteration: repeatedly scan the constraint queue until a round produces no new
    /// bindings.
    /// Constraints have dependencies (constraint A depends on constraint B binding some TypeVar
    /// first); a single FIFO pass may miss solutions due to timing. Fixpoint iteration eliminates
    /// timing dependencies via retries.
    ///
    /// - Equality: when either side still contains a TypeVar, re-enqueue for the next round;
    ///   when both sides are concrete types, record into errors.
    /// - TraitBound: when ty is still a TypeVar, re-enqueue; otherwise query the witness table.
    /// - Subtype/Narrow: single-pass handling (does not propagate TypeVar bindings).
    pub fn solve_with_witness(&mut self, arena: &mut TypeArena, witness: Option<&WitnessTable>) {
        const MAX_ITERATIONS: usize = 1000;
        let mut pending = std::mem::take(&mut self.pending);

        for _iteration in 0..MAX_ITERATIONS {
            if pending.is_empty() {
                break;
            }

            // Take out all current constraints for this round.
            let current = std::mem::take(&mut pending);
            let mut changed = false;

            for c in current {
                match c {
                    Constraint::Equality(t1, t2) => {
                        // Record candidate before resolve/unify (multi-value record).
                        // arena.get returns the raw Type; even if the TypeVar was bound by a
                        // previous unify, get still returns TypeVar(idx), so we can capture
                        // binding requirements from all constraint paths to this TypeVar.
                        self.record_candidate(arena, t1, t2);

                        let r1 = arena.resolve(t1);
                        let r2 = arena.resolve(t2);

                        // Both sides already resolved to the same type; nothing to do.
                        if r1 == r2 {
                            continue;
                        }

                        match arena.unify(r1, r2) {
                            Ok(()) => {
                                changed = true;
                            }
                            Err(_) => {
                                // unify failed: if either side still contains a TypeVar,
                                // re-enqueue for the next round (other constraints may bind
                                // these TypeVars in this round).
                                let r1_has_var = Self::resolve_has_type_var(arena, r1);
                                let r2_has_var = Self::resolve_has_type_var(arena, r2);
                                if r1_has_var || r2_has_var {
                                    pending.push(Constraint::Equality(t1, t2));
                                } else {
                                    // Both sides are concrete types and do not match: real error.
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::Equality(t1, t2),
                                        reason: "type mismatch".into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                        }
                    }
                    Constraint::Subtype(sub, sup) => {
                        if !is_subtype(arena, sub, sup) {
                            self.errors.push(ConstraintError {
                                constraint: Constraint::Subtype(sub, sup),
                                reason: "not a subtype".into(),
                                line: 0,
                                column: 0,
                            });
                        }
                    }
                    Constraint::TraitBound { ty, trait_name, type_args } => {
                        let resolved = arena.resolve(ty);
                        // ty is still a TypeVar: re-enqueue for the next round.
                        if matches!(arena.get(resolved), Type::TypeVar(_)) {
                            pending.push(Constraint::TraitBound {
                                ty,
                                trait_name,
                                type_args,
                            });
                            continue;
                        }

                        // ty is resolved: query the witness table to decide.
                        if let Some(wt) = witness {
                            let ct = arena.get(resolved);
                            let type_id = match ct {
                                Type::Adt(_) | Type::Generic(_) => {
                                    // User type: type_id is registered externally.
                                    // Cannot access sema_result here; skip (handled uniformly by
                                    // check_module).
                                    None
                                }
                                _ => ct.type_id(),
                            };
                            if let Some(tid) = type_id {
                                if !wt.implements(&trait_name, tid) {
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::TraitBound {
                                            ty,
                                            trait_name: trait_name.clone(),
                                            type_args: type_args.clone(),
                                        },
                                        reason: format!(
                                            "type does not implement trait '{}'",
                                            trait_name
                                        )
                                        .into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                            // When type_id is None, defer to check_module.
                        }
                    }
                    Constraint::Narrow { original, narrowed, .. } => {
                        // Narrowing constraint: on a specific path `original` is narrowed to
                        // `narrowed`.
                        // Solving strategy: if `original` is an unbound TypeVar, bind it to
                        // `narrowed`; if `original` is already bound, try unify (the narrowed
                        // type must be compatible with the original).
                        let r_orig = arena.resolve(original);
                        let r_narrow = arena.resolve(narrowed);
                        if let Type::TypeVar(idx) = arena.get(r_orig).clone() {
                            // TypeVar unbound: bind directly to the narrowed type.
                            arena.type_vars[idx as usize].bound = Some(r_narrow);
                            changed = true;
                        } else if r_orig != r_narrow {
                            // Already bound: try unify (narrowed type must be a subtype of the
                            // original).
                            match arena.unify(r_orig, r_narrow) {
                                Ok(()) => { changed = true; }
                                Err(_) => {
                                    // Narrowing conflicts with the original type: record but do
                                    // not abort.
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::Narrow {
                                            path: String::new().into_boxed_str(),
                                            original,
                                            narrowed,
                                        },
                                        reason: "narrowed type conflicts with original".into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Fixpoint: a round with no new bindings and no re-enqueued constraints ends.
            if !changed {
                break;
            }
        }

        // Constraints that did not converge after MAX_ITERATIONS: recorded but not reported
        // (defensive).
        // These are usually TypeVar ↔ TypeVar circular dependencies and do not affect
        // correctness.

        // After fixpoint convergence: build subst from candidates and detect ambiguity.
        self.finalize_solution(arena);
    }

    /// Returns whether the resolved TypeHandle still contains an unbound TypeVar.
    /// Used during fixpoint iteration to decide whether to re-enqueue a constraint.
    fn resolve_has_type_var(arena: &TypeArena, ty: TypeHandle) -> bool {
        let resolved = arena.resolve(ty);
        match arena.get(resolved) {
            Type::TypeVar(_) => true,
            // Every composite type (incl. Channel/Async/Lazy/Atomic/Sender/Receiver and Record)
            // delegates child traversal to `for_each_child`. Short-circuits on the first hit.
            _ => {
                let mut found = false;
                arena.for_each_child(resolved, |c| {
                    if !found && Self::resolve_has_type_var(arena, c) {
                        found = true;
                    }
                });
                found
            }
        }
    }

    /// Records a TypeVar's candidate binding into `candidates` (multi-value record).
    ///
    /// Called **before** unify; uses `arena.get` (raw Type, no resolve) to detect TypeVars.
    /// Even if a TypeVar was already bound to a concrete type by a previous unify, `get` still
    /// returns `TypeVar(idx)`, so we can capture binding requirements from all constraint paths
    /// to this TypeVar for later ambiguity detection.
    ///
    /// - If t1 is a TypeVar and t2 is not → candidates[t1.idx].push(t2).
    /// - If t2 is a TypeVar and t1 is not → candidates[t2.idx].push(t1).
    /// - Both sides TypeVars → do not record (var-var bindings are handled directly by unify).
    pub fn record_candidate(&mut self, arena: &TypeArena, t1: TypeHandle, t2: TypeHandle) {
        match (arena.get(t1), arena.get(t2)) {
            (Type::TypeVar(_), Type::TypeVar(_)) => {
                // Both sides are TypeVars: var-var binding is handled by unify; do not record
                // candidates.
            }
            (Type::TypeVar(idx), _) => {
                self.candidates.entry(idx).or_default().push(t2);
            }
            (_, Type::TypeVar(idx)) => {
                self.candidates.entry(idx).or_default().push(t1);
            }
            _ => {}
        }
    }

    /// After fixpoint convergence, builds the final subst from candidates and detects
    /// ambiguity.
    ///
    /// For each TypeVar's candidate set:
    /// 1. Deduplicate based on structural equality (handles are not interned).
    /// 2. Unique candidate → write into subst.
    /// 3. Multiple distinct candidates → flag an ambiguity error; still write the arena's
    ///    actual solution into subst (to avoid cascading false positives).
    fn finalize_solution(&mut self, arena: &mut TypeArena) {
        let candidates = std::mem::take(&mut self.candidates);
        for (idx, cands) in candidates {
            // Deduplicate based on structural equality (not TypeHandle identity).
            // `make()` does not intern types, so two `Type::Bool` from different call
            // sites have different TypeHandles; comparing by handle would wrongly
            // flag them as distinct candidates and emit a false "ambiguous
            // inference" error (e.g. `identity(true) == true`).
            let mut unique: Vec<TypeHandle> = Vec::new();
            for c in &cands {
                let r = arena.resolve(*c);
                if !unique.iter().any(|&u| types_equal(arena, u, r)) {
                    unique.push(r);
                }
            }

            match unique.len() {
                0 => {} // Impossible (cands is non-empty when iterated).
                1 => {
                    // Unique candidate: write into subst and write back
                    // arena.type_vars[idx].bound.
                    // The write-back is critical: diagnostics check
                    // arena.type_vars[idx].bound; without it, an already-solved TypeVar would
                    // still be flagged as unresolved.
                    let resolved = arena.resolve(unique[0]);
                    self.subst.insert(idx, resolved);
                    arena.type_vars[idx as usize].bound = Some(resolved);
                }
                _ => {
                    // Multiple distinct candidates: ambiguity.
                    // Pick the arena's actual solution (unify picked the first successful one)
                    // and write it into subst to avoid cascading false positives.
                    let resolved = arena.resolve(cands[0]);
                    self.subst.insert(idx, resolved);
                    arena.type_vars[idx as usize].bound = Some(resolved);
                    // Record the ambiguity error.
                    self.errors.push(ConstraintError {
                        constraint: Constraint::Equality(unique[0], unique[1]),
                        reason: format!(
                            "ambiguous inference for TypeVar{}: {} distinct candidates",
                            idx,
                            unique.len()
                        )
                        .into(),
                        line: 0,
                        column: 0,
                    });
                }
            }
        }
    }

    /// Looks up the solving result of a TypeVar.
    #[inline]
    pub fn lookup_subst(&self, var_idx: u32) -> Option<TypeHandle> {
        self.subst.get(&var_idx).copied()
    }

    /// Returns all solving errors.
    #[inline]
    pub fn errors(&self) -> &[ConstraintError] {
        &self.errors
    }

    /// Returns whether there are any solving errors.
    #[inline]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns the number of pending constraints.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clears all state (called on module switch).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.subst.clear();
        self.errors.clear();
        self.candidates.clear();
    }
}


// =========================================================================
// sema v2: Flow-Sensitive Narrowing — general flow fact system.
//
// Design philosophy (original, not borrowed from Kotlin/TS):
// - Generalize the Zig version's nullable narrowing into a general flow fact system.
// - Supports three narrowing kinds: NonNull / IsCheck / ConstructorMatch.
// - DOD: flow facts use arena indices; scopes are stack-managed.
// - Query: lookup_narrowed(path) -> Option<TypeHandle>.
//
// Relationship with the constraint solver:
// Narrowing constraints are managed by FlowContext and do not enter the solver queue
// directly.
// FlowContext is path-sensitive; the solver is path-insensitive.
// =========================================================================

/// Narrowing kinds: covers all flow-sensitive type refinement scenarios in Kuzo.
#[derive(Debug, Clone)]
pub enum NarrowKind {
    /// Non-null narrowing: `if x != null` → x narrows from `Nullable<T>` to `T`.
    NonNull,
    /// Type-test narrowing: `if x is Type` → x narrows to Type.
    /// (Kuzo's `is` expression, similar to Kotlin's smart cast.)
    IsCheck(TypeHandle),
    /// ADT constructor-match narrowing: `match x { Some(v) => ... }` → x narrows to
    /// `Some<T>`.
    /// (GADT type refinement: after constructor matching, type variables gain concrete
    /// information.)
    ConstructorMatch {
        /// Constructor name (e.g. "Some", "None", "Ok", "Err").
        ctor_name: Box<str>,
        /// Bound sub-pattern variable names (used for sub-pattern type refinement).
        bound_vars: Box<[Box<str>]>,
    },
}

/// Flow fact: a type narrowing assertion about a path at some program point.
///
/// `path` is the expression's canonical path (e.g. "x", "obj.field", "a.b.c"), used to
/// reference the same expression from different positions.
#[derive(Debug, Clone)]
pub struct FlowFact {
    /// Expression path (canonicalized as a string).
    pub path: Box<str>,
    /// Narrowed type.
    pub narrowed_ty: TypeHandle,
    /// Narrowing condition.
    pub kind: NarrowKind,
}

/// Flow fact table: stores all flow facts within the current scope.
///
/// DOD: facts in Vec, by_path indexed by FxHashMap.
#[derive(Default)]
pub struct FlowFactTable {
    facts: Vec<FlowFact>,
    /// Indexed by path: path → fact indices.
    by_path: FxHashMap<Box<str>, Vec<u32>>,
}

impl FlowFactTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a flow fact.
    pub fn add(&mut self, fact: FlowFact) {
        let idx = self.facts.len() as u32;
        self.by_path
            .entry(fact.path.clone())
            .or_default()
            .push(idx);
        self.facts.push(fact);
    }

    /// Looks up the latest narrowed type for a path.
    ///
    /// Returns the type of the last narrowing for that path (the same path may be narrowed
    /// multiple times; the latest is taken — facts are in append order, so the last added is
    /// the latest).
    pub fn lookup(&self, path: &str) -> Option<TypeHandle> {
        self.by_path
            .get(path)
            .and_then(|indices| indices.last())
            .and_then(|&idx| self.facts.get(idx as usize))
            .map(|f| f.narrowed_ty)
    }

    /// Looks up the latest flow fact for a path (including kind).
    pub fn lookup_fact(&self, path: &str) -> Option<&FlowFact> {
        self.by_path
            .get(path)
            .and_then(|indices| indices.last())
            .and_then(|&idx| self.facts.get(idx as usize))
    }

    /// Number of facts in the current scope.
    #[inline]
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Returns whether the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// Flow context: stack-managed flow fact scopes.
///
/// Push a new scope when entering an if/match branch; pop when leaving.
/// Queries search from the top of the stack downward (inner scopes shadow outer ones).
pub struct FlowContext {
    scopes: Vec<FlowFactTable>,
}

impl Default for FlowContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowContext {
    pub fn new() -> Self {
        FlowContext {
            scopes: vec![FlowFactTable::new()], // Root scope.
        }
    }

    /// Enters a new scope (if/match branch).
    pub fn push_scope(&mut self) {
        self.scopes.push(FlowFactTable::new());
    }

    /// Leaves the current scope.
    ///
    /// Does not pop the root scope (keeps at least one layer).
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Adds a flow fact to the current (top) scope.
    pub fn add_fact(&mut self, fact: FlowFact) {
        if let Some(top) = self.scopes.last_mut() {
            top.add(fact);
        }
    }

    /// Looks up the narrowed type for a path: searches from the top of the stack downward.
    ///
    /// Inner-scope narrowings shadow outer ones (path-sensitive).
    ///
    /// Bug #35: A ConstructorMatch fact indicates the scrutinee variable was narrowed to an
    /// ADT type. If the fact's bound_vars contains `path`, then `path` has been shadowed by
    /// a pattern variable (e.g. in `match w { W3(w) => w * w }` the pattern variable `w`
    /// shadows the parameter `w`). In that case the scrutinee's narrowed type does not apply
    /// to the pattern variable; this fact should be skipped so infer_expr falls back to env
    /// lookup to get the pattern variable's correct field type.
    pub fn lookup_narrowed(&self, path: &str) -> Option<TypeHandle> {
        for scope in self.scopes.iter().rev() {
            if let Some(fact) = scope.lookup_fact(path) {
                if let NarrowKind::ConstructorMatch { bound_vars, .. } = &fact.kind {
                    if bound_vars.iter().any(|v| v.as_ref() == path) {
                        continue;
                    }
                }
                return Some(fact.narrowed_ty);
            }
        }
        None
    }

    /// Looks up the flow fact for a path (including kind): searches from the top of the
    /// stack downward.
    pub fn lookup_fact(&self, path: &str) -> Option<&FlowFact> {
        for scope in self.scopes.iter().rev() {
            if let Some(fact) = scope.lookup_fact(path) {
                return Some(fact);
            }
        }
        None
    }

    /// Current scope depth.
    #[inline]
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Resets to the root scope (called on function switch).
    pub fn reset(&mut self) {
        self.scopes.truncate(1);
        self.scopes[0] = FlowFactTable::new();
    }
}

/// Extracts flow facts from the condition expression of `if cond { ... }`.
///
/// Returns (then_facts, else_facts):
/// - then_facts: narrowings that hold in the then branch.
/// - else_facts: narrowings that hold in the else branch (condition negated).
///
/// Supports:
/// - `x != null` → then: NonNull(x), else: none.
/// - `x == null` → then: none, else: NonNull(x).
/// - `x is Type` → then: IsCheck(x, Type), else: none.
///
/// (ConstructorMatch is handled by the match expression, not in this function.)
pub fn analyze_null_check_facts(
    arena: &TypeArena,
    ast: &AstArena<'_>,
    cond: ExprId,
    env: EnvId,
    env_arena: &EnvArena,
) -> (Vec<FlowFact>, Vec<FlowFact>) {
    let mut then_facts = Vec::new();
    let mut else_facts = Vec::new();

    // If `path_expr` is a nullable variable path and `null_expr` is a null literal,
    // push a NonNull narrowing fact into `facts`.
    let push_nonnull = |path_expr: ExprId, null_expr: ExprId, facts: &mut Vec<FlowFact>| {
        if let Some(path) = expr_path(ast, path_expr) {
            if matches!(ast.expr(null_expr).node, Expr::NullLit) {
                if let Some(ty) = env_arena.lookup(env, &path) {
                    let resolved = arena.resolve(ty);
                    if let Type::Nullable(_) = arena.get(resolved) {
                        facts.push(FlowFact {
                            path: path.into(),
                            narrowed_ty: arena.nullable_inner(resolved),
                            kind: NarrowKind::NonNull,
                        });
                    }
                }
            }
        }
    };

    let cond_node = &ast.expr(cond).node;
    if let Expr::Binary { op, lhs, rhs } = cond_node {
        match op {
            crate::ast::Ast::BinaryOp::NotEq => {
                // `x != null` / `null != x` → then: NonNull(x).
                push_nonnull(*lhs, *rhs, &mut then_facts);
                push_nonnull(*rhs, *lhs, &mut then_facts);
            }
            crate::ast::Ast::BinaryOp::Eq => {
                // `x == null` / `null == x` → else: NonNull(x).
                push_nonnull(*lhs, *rhs, &mut else_facts);
                push_nonnull(*rhs, *lhs, &mut else_facts);
            }
            _ => {}
        }
    }

    (then_facts, else_facts)
}

/// Extracts the canonical path of an expression (used as the flow narrowing identifier).
///
/// Supports:
/// - `Ident(name)` → `name`.
/// - `FieldAccess(recv, field)` → `{recv_path}.{field}`.
/// - Others → None (not narrowable).
fn expr_path(ast: &AstArena<'_>, expr: ExprId) -> Option<String> {
    match &ast.expr(expr).node {
        Expr::Ident(name) => Some((*name).to_string()),
        Expr::FieldAccess { recv, field } => {
            let recv_path = expr_path(ast, *recv)?;
            Some(format!("{}.{}", recv_path, field))
        }
        _ => None,
    }
}

/// Extracts the constructor name and bound variable names from a pattern node (for
/// ConstructorMatch narrowing).
///
/// Only handles `Constructor { name, patterns }` patterns: extracts the constructor name and
/// the names of all `Variable` bindings in sub-patterns.
///
/// Other patterns (Wildcard/Literal/Variable/Record/Or/Guard) return None.
fn extract_constructor_pattern<'a>(
    pattern: &crate::ast::Ast::Pattern<'a>,
    ast: &'a crate::ast::Ast::AstArena<'a>,
) -> Option<(&'a str, Vec<Box<str>>)> {
    match pattern {
        crate::ast::Ast::Pattern::Constructor { name, patterns } => {
            let mut bound_vars = Vec::new();
            for &pref in patterns {
                collect_pattern_binds(ast, pref, &mut bound_vars);
            }
            Some((*name, bound_vars))
        }
        _ => None,
    }
}

/// Recursively collects the bound variable names in a pattern (the `name` of Variable patterns).
fn collect_pattern_binds<'a>(
    ast: &'a crate::ast::Ast::AstArena<'a>,
    pref: crate::ast::Ast::PatternRef,
    out: &mut Vec<Box<str>>,
) {
    match &ast.pattern(pref).node {
        crate::ast::Ast::Pattern::Variable { name } => {
            out.push((*name).into());
        }
        crate::ast::Ast::Pattern::Constructor { patterns, .. } => {
            for &p in patterns {
                collect_pattern_binds(ast, p, out);
            }
        }
        crate::ast::Ast::Pattern::Record { fields } => {
            for f in fields {
                collect_pattern_binds(ast, f.pattern, out);
            }
        }
        crate::ast::Ast::Pattern::OrPattern { left, right } => {
            collect_pattern_binds(ast, *left, out);
            collect_pattern_binds(ast, *right, out);
        }
        crate::ast::Ast::Pattern::Guard { pattern, .. } => {
            collect_pattern_binds(ast, *pattern, out);
        }
        _ => {}
    }
}

